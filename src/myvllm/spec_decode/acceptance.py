"""Speculative-decoding acceptance (rejection sampling), pure torch math.

Lossless-by-construction: accepting a draft token x with probability
min(1, p(x)/q(x)) (q = the distribution that produced x, p = the target
distribution over the same position) and, on the first rejection at level n,
drawing the replacement from the renormalized (p_n - q_n)_+ keeps the emitted
distribution exactly equal to sampling from p directly.

Contract
--------
* ``target_logits[B, K+1, V]`` -- rows 0..K-1 are the target logits over the
  token *after* each draft position (row i verifies draft token i), row K is
  the bonus distribution (the position after the last draft token).
* ``draft_logits[B, K, V]``  -- row i is the distribution the draft model used
  to sample draft_tokens[:, i].
* ``draft_tokens[B, K]``     -- the proposed draft tokens (int64).

Logits must already be divided by the sampling temperature that produced them
(engine-side, the same scaling the ordinary sampler applies: logits/temperature
before softmax). Acceptance applies a plain softmax and never re-applies a
temperature, so what it compares is exactly the distributions the two samplers
would draw from. Passing greedy logits (tiny temperature already applied) makes
q and p effectively one-hot, which is what the greedy e2e equivalence test uses.

Return
------
* ``num_accepted[B]`` -- number of leading draft tokens accepted (0..K).
* ``extra_token[B]``  -- the token appended *after* the accepted prefix:
  - if num_accepted == K: a bonus token drawn from the target distribution at
    the position after the last draft (so K+1 tokens are committed);
  - if num_accepted == n < K: a replacement drawn from renormalized
    (p_n - q_n)_+ (so n+1 tokens are committed).

Caller assembles the commit as draft_tokens[:, :num_accepted] + extra_token.
"""

from __future__ import annotations

import torch

# Below this, a residual distribution (p - q)_+ has no numerically meaningful
# mass. Rejection at that level is a float artifact (p and q are effectively
# identical); the replacement falls back to p's argmax, which equals the draft
# token in the greedy case and is distributionally correct otherwise.
_RESIDUAL_EPS = 1e-9


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def _sample(probs: torch.Tensor, generator) -> torch.Tensor:
    """Draw one token per row from a distribution in [.., V]."""
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


def rejection_accept(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_logits = target_logits.float()
    draft_logits = draft_logits.float()
    draft_tokens = draft_tokens.long()
    B, K, _ = draft_logits.shape
    assert target_logits.shape[:2] == (B, K + 1), (
        f"target_logits {tuple(target_logits.shape)} vs draft {B}x{K}xV"
    )
    assert draft_tokens.shape == (B, K)

    p = _softmax(target_logits)           # [B, K+1, V]
    q = _softmax(draft_logits)            # [B, K, V]
    p_draft = p[:, :K, :]                 # [B, K, V]

    p_x = p_draft.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)  # [B, K]
    # q is the distribution that actually produced the draft token, so q_x > 0
    # (softmax is never exactly 0); a clamp only protects against float noise.
    q_x = q.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)

    # accept draft token i iff every earlier token was accepted and u_i <= p/q
    alpha = torch.minimum(p_x / q_x.clamp_min(torch.finfo(q_x.dtype).tiny), torch.ones_like(p_x))
    u = torch.rand((B, K), device=target_logits.device, generator=generator)
    rejected = u > alpha
    has_rejection = rejected.any(dim=-1)                  # [B]
    first_rejection = rejected.long().argmax(dim=-1)      # [B], 0 when none
    num_accepted = torch.where(
        has_rejection, first_rejection, torch.full_like(first_rejection, K)
    )

    extra = torch.full((B,), -1, dtype=torch.long, device=target_logits.device)
    bonus_mask = ~has_rejection                            # all drafts accepted
    if bool(bonus_mask.any()):
        bonus_probs = p[:, K, :][bonus_mask]
        extra[bonus_mask] = _sample(bonus_probs, generator)

    rej_mask = has_rejection
    if bool(rej_mask.any()):
        rows = torch.nonzero(rej_mask, as_tuple=False).squeeze(-1)   # rejected seqs
        levels = num_accepted[rows]                                  # first-reject level n
        p_n = p[:, :K, :][rows, levels]                              # [R, V]
        q_n = q[:, :, :][rows, levels]                               # [R, V]
        residual = torch.clamp(p_n - q_n, min=0.0)
        mass = residual.sum(dim=-1)                                  # [R]
        flat = mass <= _RESIDUAL_EPS
        if bool(flat.any()):
            # degenerate (p_n ~ q_n): replacement is p's argmax, which equals the
            # draft token in the greedy case (self-healing spurious rejection)
            extra[rows[flat]] = p_n[flat].argmax(dim=-1)
        if bool((~flat).any()):
            keep = ~flat
            extra[rows[keep]] = _sample(residual[keep] / mass[keep].unsqueeze(-1), generator)

    return num_accepted, extra
