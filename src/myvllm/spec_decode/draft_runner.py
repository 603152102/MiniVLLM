"""Draft model for speculative decoding (self-speculation MVP).

The draft shares the target model's weights AND its KV cache pool. With
identical weights the draft's KV over any accepted prefix is bitwise identical
to the target's, so the draft reads the target's cache for the accepted prefix
and writes only its provisional tail (the spec positions, plus position t-1
whose KV the previous round never computed -- see propose_step) into the
target's slots. The target's verify pass overwrites exactly those slots before
its own attention reads them, so the shared cache stays consistent -- and stays
correct even under draft_skip_layers, where the draft's KV differs from the
target's: the target re-stores every spec slot during verify before its
attention reads them.

This is a thin wrapper by design (审阅 #8): no separate BlockManager, no KV
pool, no weight loading, no dist.init_process_group -- ModelRunner owns the
model and lends it to the draft. spec_method="draft_model" (an independent
draft with its own weights and pool) is reserved for stage 3.
"""

from __future__ import annotations

import torch


class DraftRunner:
    def __init__(self, model, sampler, skip_layers: bool = False):
        self.model = model            # Qwen3ForCausalLM, weights shared with the target
        self.sampler = sampler        # SamplerLayer, same path as ordinary sampling
        self.skip_layers = skip_layers

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Hidden states over the given tokens (varlen 1-D), optionally with
        LayerSkip-style layer skipping (draft runs only every other layer)."""
        m = self.model.model  # Qwen3Model
        x = m.embed_tokens(input_ids)
        layers = m.layers[::2] if self.skip_layers else m.layers
        residual = None
        for layer in layers:
            x, residual = layer(x, residual)
        x, _ = m.norm(x, residual)
        return x
