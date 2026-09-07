"""Stage-0 speculative decoding: pure-logic, CPU-only.

Covers the pieces of 实现speculative_decoding.md "阶段 0" that need no GPU or
model wiring:
  * Sequence speculative fields + append_tokens / rollback_tokens + pickle
  * BlockManager.rollback (block accounting reconciliation after truncation)
  * spec_decode/acceptance.py rejection-sampling math
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence, SequenceStage
from myvllm.spec_decode.acceptance import rejection_accept

torch = pytest.importorskip("torch")

# Margin for effectively-one-hot logits. Must be small enough that exp(±BIG)
# stays finite in float32 (BIG=20 gives chosen-token prob that rounds to exactly
# 1.0 in float32, so acceptance is deterministic) but large enough to make
# disagreement essentially a certain rejection.
BIG = 20.0


def greedy_logits(shape, argmax_ids):
    """shape (..., V); logits one-hot at argmax_ids (...,) -> +BIG, rest -BIG."""
    logits = torch.full(shape, -BIG)
    logits.scatter_(-1, argmax_ids.long().unsqueeze(-1), BIG)
    return logits


def commit_tokens(draft_tokens, n, extra):
    """Reconstruct the token list a step commits: accepted drafts + extra."""
    B, K = draft_tokens.shape
    commits = []
    for b in range(B):
        nb = int(n[b])
        commits.append(draft_tokens[b, :nb].tolist() + [int(extra[b])])
    return commits


def greedy_continuation(target_logits):
    """The target's own greedy continuation: argmax of each target logits row."""
    return target_logits.argmax(dim=-1).tolist()


# --------------------------------------------------------------------------
# Sequence: speculative fields, bulk append/rollback, pickle contract
# --------------------------------------------------------------------------


class TestSequenceRollback:
    def test_append_tokens_and_rollback_restore_baseline(self):
        prompt = [1, 2, 3, 4, 5]
        seq = Sequence(prompt, block_size=4)
        baseline_ids = list(seq.token_ids)

        seq.append_tokens([7, 8, 9])
        assert seq.num_tokens == len(prompt) + 3
        assert seq.last_token == 9
        assert seq.token_ids == baseline_ids + [7, 8, 9]

        seq.rollback_tokens(2)
        assert seq.num_tokens == len(prompt) + 1
        assert seq.last_token == 7
        assert seq.token_ids == baseline_ids + [7]

        seq.rollback_tokens(1)
        assert seq.num_tokens == len(prompt)
        assert seq.last_token == prompt[-1]
        assert seq.token_ids == baseline_ids

    def test_rollback_cannot_touch_prompt(self):
        seq = Sequence([1, 2, 3], block_size=4)
        seq.append_token(9)  # one completion token
        with pytest.raises(AssertionError):
            seq.rollback_tokens(2)

    def test_spec_buffer_fields(self):
        seq = Sequence([1, 2, 3], block_size=4)
        assert seq.spec_token_ids == []
        assert seq.num_spec_verified == 0
        seq.set_spec_buffer([10, 11], num_verified=1)
        assert seq.spec_token_ids == [10, 11]
        assert seq.num_spec_verified == 1
        seq.clear_spec_buffer()
        assert seq.spec_token_ids == []
        assert seq.num_spec_verified == 0

    def test_pickle_preserves_spec_state_decode_branch(self):
        seq = Sequence([1, 2, 3], block_size=4)
        seq.append_token(42)
        seq.stage = SequenceStage.DECODE
        seq.set_spec_buffer([100, 101, 102], num_verified=2)

        restored = pickle.loads(pickle.dumps(seq))

        assert restored.stage == SequenceStage.DECODE
        assert restored.num_tokens == 4
        # decode branch still minimizes token_ids to the last token
        assert restored.token_ids == [42]
        assert restored.last_token == 42
        # ... but the pending draft buffer must cross in full
        assert restored.spec_token_ids == [100, 101, 102]
        assert restored.num_spec_verified == 2

    def test_pickle_preserves_spec_state_prefill_branch(self):
        seq = Sequence([1, 2, 3, 4], block_size=4)
        seq.set_spec_buffer([], num_verified=0)
        restored = pickle.loads(pickle.dumps(seq))
        assert restored.token_ids == [1, 2, 3, 4]
        assert restored.spec_token_ids == []


# --------------------------------------------------------------------------
# BlockManager.rollback
# --------------------------------------------------------------------------


def decode_append(bm, seq, token_id):
    """Mirror one scheduler.decode step: reserve the slot, then commit content."""
    # scheduler calls can_append first; with plenty of free blocks it is always True
    bm.append(seq)
    seq.append_token(token_id)


class TestBlockManagerRollback:
    def _fresh(self, num_blocks=8, block_size=4):
        return BlockManager(num_blocks, block_size)

    def _grow(self, bm, seq, tokens):
        for t in tokens:
            decode_append(bm, seq, t)

    def test_rollback_restores_exact_pre_provisional_state(self):
        prompt = [1, 2, 3]
        accepted_extra = [100, 101, 102]      # committed, will stay
        provisional = [200, 201, 202, 203, 204, 205]  # rolled back

        ctrl_bm = self._fresh()
        ctrl = Sequence(prompt, block_size=4)
        ctrl_bm.allocate(ctrl)
        self._grow(ctrl_bm, ctrl, accepted_extra)  # ctrl length = 6 (mid-block)

        subj_bm = self._fresh()
        subj = Sequence(prompt, block_size=4)
        subj_bm.allocate(subj)
        self._grow(subj_bm, subj, accepted_extra)
        self._grow(subj_bm, subj, provisional)    # length = 12, crosses blocks
        subj.rollback_tokens(len(provisional))
        subj_bm.rollback(subj)

        assert subj.num_tokens == ctrl.num_tokens == 6
        assert subj.token_ids == ctrl.token_ids
        # same blocks in the same order, KV locations preserved
        assert subj.block_table == ctrl.block_table
        assert subj.num_cached_tokens == ctrl.num_cached_tokens
        assert subj_bm.used_block_ids == ctrl_bm.used_block_ids
        # per-block hash/content identical for every kept block
        for bid in ctrl.block_table:
            a, b = ctrl_bm.blocks[bid], subj_bm.blocks[bid]
            assert a.hash == b.hash
            assert a.token_ids == b.token_ids
        # freed provisional block is back on the free list
        for bid in ctrl.block_table:
            assert bid not in subj_bm.free_block_ids

    def test_rollback_to_block_boundary_is_consistent(self):
        # landing on a full block boundary: the trailing block must end up
        # finalized with exactly the accepted content (not including rolled-back
        # tokens), so future append()/prefix matching see real content
        bm = self._fresh()
        seq = Sequence([1, 2, 3, 4], block_size=4)  # 4 = full block 0
        bm.allocate(seq)                            # block0 full, finalized
        self._grow(bm, seq, [5, 6, 7, 8])           # -> 8 tokens, block1 full too
        seq.rollback_tokens(4)                       # back to 4
        bm.rollback(seq)

        assert seq.num_tokens == 4
        assert seq.block_table == [0]
        last = bm.blocks[seq.block_table[-1]]
        assert last.token_ids == [1, 2, 3, 4]
        assert last.hash != -1
        assert bm.hash_to_block_id.get(last.hash) == last.block_id

    def test_rollback_idempotent(self):
        bm = self._fresh()
        seq = Sequence([1, 2, 3], block_size=4)
        bm.allocate(seq)
        self._grow(bm, seq, [4, 5, 6, 7, 8, 9])
        seq.rollback_tokens(3)
        bm.rollback(seq)
        first = (list(seq.block_table),
                 [bm.blocks[i].hash for i in seq.block_table],
                 set(bm.used_block_ids), list(bm.free_block_ids))
        bm.rollback(seq)  # already reconciled -> no change
        second = (list(seq.block_table),
                  [bm.blocks[i].hash for i in seq.block_table],
                  set(bm.used_block_ids), list(bm.free_block_ids))
        assert first == second

    def test_rollback_is_noop_on_consistent_sequence(self):
        # seq already consistent (num_tokens == len(token_ids), nothing pending):
        # rollback must not raise or disturb block accounting
        bm = self._fresh()
        seq = Sequence([1, 2, 3], block_size=4)
        bm.allocate(seq)
        self._grow(bm, seq, [4, 5, 6])
        before = (list(seq.block_table), [bm.blocks[i].hash for i in seq.block_table])
        bm.rollback(seq)
        after = (list(seq.block_table), [bm.blocks[i].hash for i in seq.block_table])
        assert seq.num_tokens == 6
        assert after == before

    def test_can_continue_after_rollback(self):
        bm = self._fresh()
        seq = Sequence([1, 2, 3], block_size=4)
        bm.allocate(seq)
        self._grow(bm, seq, [4, 5, 6, 7, 8])   # prompt 3 + 5 = 8 tokens
        seq.rollback_tokens(2)
        bm.rollback(seq)                        # back to 6 tokens
        # appending again from the rolled-back state must not raise
        self._grow(bm, seq, [300, 301, 302, 303])
        assert seq.num_tokens == 3 + 5 - 2 + 4  # = 10
        # 10 tokens at block_size 4 -> 3 blocks
        assert len(seq.block_table) == 3


# --------------------------------------------------------------------------
# Acceptance / rejection sampling
# --------------------------------------------------------------------------


class TestRejectionAccept:
    def test_greedy_all_accepted_plus_bonus(self):
        # target and draft agree on every draft position: under greedy sampling the
        # committed tokens must equal the target's greedy continuation. A spurious
        # float rejection is allowed (it self-heals to the same token), so assert
        # the emitted commit is a non-empty PREFIX of that continuation, not an
        # exact n/bonus split.
        torch.manual_seed(0)
        B, K, V = 3, 2, 6
        draft_tokens = torch.tensor([[1, 3], [0, 4], [5, 2]])
        draft_logits = greedy_logits((B, K, V), draft_tokens)
        bonus_arg = torch.tensor([[0], [1], [3]])
        target_arg = torch.cat([draft_tokens, bonus_arg], dim=-1)
        target_logits = greedy_logits((B, K + 1, V), target_arg)

        n, extra = rejection_accept(target_logits, draft_logits, draft_tokens)

        commits = commit_tokens(draft_tokens, n, extra)
        expected = greedy_continuation(target_logits)   # target greedy next K+1 tokens
        for b in range(B):
            assert len(commits[b]) >= 1
            assert commits[b] == expected[b][: len(commits[b])]

    def test_reject_at_level1_uses_target_replacement(self):
        torch.manual_seed(0)
        B, K, V = 2, 3, 8
        # both sequences: levels 0 and 1 agree with target, level 2 disagrees
        draft_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        target_draft_arg = torch.tensor([[1, 2, 7], [4, 5, 0]])  # level2 differs
        target_arg = torch.cat([target_draft_arg, torch.zeros(B, 1, dtype=torch.long)], dim=-1)
        draft_logits = greedy_logits((B, K, V), draft_tokens)
        target_logits = greedy_logits((B, K + 1, V), target_arg)

        n, extra = rejection_accept(target_logits, draft_logits, draft_tokens)

        assert n.tolist() == [2, 2]               # reject at level 2 -> 2 accepted
        assert extra.tolist() == [7, 0]           # replacement = target argmax at level 2

    def test_reject_at_level0(self):
        torch.manual_seed(0)
        B, K, V = 1, 2, 5
        draft_tokens = torch.tensor([[0, 1]])
        target_arg = torch.tensor([[4, 4, 2]])   # level0 disagrees with draft 0
        draft_logits = greedy_logits((B, K, V), draft_tokens)
        target_logits = greedy_logits((B, K + 1, V), target_arg)

        n, extra = rejection_accept(target_logits, draft_logits, draft_tokens)

        assert n.tolist() == [0]
        assert extra.tolist() == [4]

    def test_degenerate_residual_falls_back_to_target_argmax(self):
        # p == q (both one-hot at the draft token): even if a float artifact
        # triggers a rejection, the residual (p-q)_+ has no mass and the fallback
        # emits p's argmax == the draft token, so the output is unchanged
        # (self-healing). Assert the emitted commit stays a prefix of the target
        # greedy continuation, whatever the RNG does.
        for seed in range(8):
            torch.manual_seed(seed)
            B, K, V = 4, 1, 7
            draft_tokens = torch.randint(0, V, (B, K))
            draft_logits = greedy_logits((B, K, V), draft_tokens)
            bonus_arg = torch.zeros(B, dtype=torch.long) + 6
            target_arg = torch.cat([draft_tokens, bonus_arg.unsqueeze(-1)], dim=-1)
            target_logits = greedy_logits((B, K + 1, V), target_arg)

            n, extra = rejection_accept(target_logits, draft_logits, draft_tokens)

            commits = commit_tokens(draft_tokens, n, extra)
            expected = greedy_continuation(target_logits)
            for b in range(B):
                assert len(commits[b]) >= 1
                assert commits[b] == expected[b][: len(commits[b])]

    def test_emitted_distribution_equals_target_any_draft(self):
        # The defining property of rejection sampling: the emitted first token is
        # distributed as p (the target), no matter how biased q (the draft) is.
        torch.manual_seed(7)
        V = 3
        target_logit = torch.tensor([[2.0, 1.0, -1.0]])   # p0 over first emitted token
        p0 = torch.softmax(target_logit, dim=-1)
        # a very biased draft: almost always proposes token 0 (unlikely under p)
        draft_logit = torch.tensor([[5.0, 0.0, -5.0]])
        q0 = torch.softmax(draft_logit, dim=-1)
        # bonus row is irrelevant for the first-token distribution
        bonus_logit = torch.tensor([[0.0, 0.0, 0.0]])

        N = 120_000
        draft_tokens = torch.multinomial(q0.expand(N, V), 1).squeeze(-1).unsqueeze(-1)
        draft_logits = draft_logit.expand(N, 1, V).contiguous()
        target_logits = torch.cat(
            [target_logit.expand(N, 1, V), bonus_logit.expand(N, 1, V)], dim=1
        ).contiguous()

        n, extra = rejection_accept(target_logits, draft_logits, draft_tokens)

        # assemble the first emitted token per sequence
        emitted = torch.where(
            n.squeeze(-1) == 1,                       # draft accepted -> first token is the draft
            draft_tokens.squeeze(-1),
            extra,
        )
        freq = torch.bincount(emitted, minlength=V).float() / N
        assert torch.allclose(freq, p0.squeeze(0), atol=8e-3), (freq, p0)
