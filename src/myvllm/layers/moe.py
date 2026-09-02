import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os

from myvllm.layers.linear import ReplicatedLinear

"""
MoE (Mixture-of-Experts) layers.

How to run the tests (same pattern as linear.py):
  1. cd src/myvllm/layers
  2. uv run torchrun --nproc_per_node=1 moe.py   # single GPU
  3. uv run torchrun --nproc_per_node=4 moe.py   # 4 GPUs, or 4 processes
     sharing 1 physical GPU (see the device_count trick in _init_dist) --
     correctness of the EP code path does not depend on physical GPU count.

What this file teaches:
  - MLPExpert: one expert FFN (SwiGLU, same math as Qwen3MLP).
  - MoELayer: single-GPU reference. Router (gate) + top-k selection +
    per-expert forward + weighted scatter-add. The degenerate case
    num_experts=1 / top_k=1 must reduce to a plain FFN (softmax of one
    logit is 1) -- this is the correctness anchor.
  - ExpertParallelMoELayer: experts sharded across ranks, router replicated.
    Per rank the pipeline is:
      gate -> topk -> bucket by expert home rank
        -> dispatch (all-to-all) -> sort by local expert (contiguous GEMM)
        -> expert forward -> combine (all-to-all) -> scatter-add

    Empty buckets are padded with a weight-0 dummy item so every all-to-all
    chunk is non-empty (NCCL dislikes 0-size chunks); a weight-0 item
    contributes nothing to the output. Real systems instead use expert
    capacity (drop overflow tokens) or count-based all-to-all -- see vLLM's
    fused_moe.py / moe_align_block_size.

    Where to plug it into the engine: replace Qwen3MLP in qwen3.py's
    Qwen3DecoderLayer. At that point x is already replicated across ranks
    (the RowParallel all_reduce happened in the attention o_proj), which is
    exactly the precondition EP needs. In EP the sharding unit is the expert,
    so each expert is a complete (non-TP) FFN; TP x EP needs process groups
    and TP-aware experts -- left as an extension.

Reference: Mixtral paper (routing), DeepSeek-V2 (fine-grained experts),
DeepSeek-V3 (EP + all-to-all), vLLM vllm/model_executor/layers/moe.
"""


class MLPExpert(nn.Module):
    """
    One expert FFN: SwiGLU, same math as Qwen3MLP.
    Uses ReplicatedLinear (no TP sharding inside an expert): in pure EP the
    sharding unit is the expert itself, so each expert is a complete FFN.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = True):
        super().__init__()
        self.gate_proj = ReplicatedLinear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = ReplicatedLinear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = ReplicatedLinear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (num_tokens, hidden_size)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    """
    Single-GPU MoE: all experts live on one device. This is the correctness
    reference for the expert-parallel version below.

    Routing: top_k experts picked by the gate logits, weights are the
    softmax over the selected logits (Mixtral style; DeepSeek uses per-expert
    sigmoid instead -- a one-line change in forward).
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        bias: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # router: one logit per expert
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MLPExpert(hidden_size, intermediate_size, bias=bias)
             for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, hidden_size) -- prefill is varlen 2D, decode is 2D, so T = tokens

        # 1. route: pick top_k experts per token
        logits = self.gate(x)                            # (T, E)
        weights, ids = logits.topk(self.top_k, dim=-1)   # (T, k) each
        weights = weights.softmax(dim=-1)                # normalize over selected experts

        # 2. each expert computes on its tokens; weighted outputs scatter-added.
        # This naive loop scans T*k masks per expert (O(T*E*k)) -- fine for a
        # reference. The sorted, contiguous-GEMM version lives in
        # ExpertParallelMoELayer below.
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            for k in range(self.top_k):
                mask = ids[:, k] == e               # tokens whose k-th pick is expert e
                if mask.any():
                    out[mask] += weights[mask, k, None] * self.experts[e](x[mask])
        return out


class ExpertParallelMoELayer(nn.Module):
    """
    Expert parallelism (EP): the experts are sharded across ranks, the router
    is replicated. Rank r owns experts [r * E/W, (r + 1) * E/W).

    A token's top_k experts may live on any rank, so the token's hidden state
    travels to those ranks (dispatch all-to-all), is computed there, and comes
    back (combine all-to-all). Every rank must see the SAME input x.

    The world_size used here IS the EP size (pure EP, no TP). For TP x EP you
    need process groups and TP-aware experts -- leave as an extension.
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        bias: bool = True,
    ):
        super().__init__()
        self.ep_size = dist.get_world_size()
        self.ep_rank = dist.get_rank()
        assert num_experts % self.ep_size == 0, "num_experts must be divisible by EP size"
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts_per_rank = num_experts // self.ep_size

        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MLPExpert(hidden_size, intermediate_size, bias=bias)
             for _ in range(self.experts_per_rank)]
        )

    def _all_to_all(self, out_chunks: list[torch.Tensor], in_chunks: list[torch.Tensor]) -> None:
        """
        Exchange per-rank chunks: rank r sends in_chunks[w] to rank w and
        receives rank w's chunk into out_chunks[w].
        NCCL has a native all_to_all; gloo (used when multiple ranks share
        one GPU for correctness tests) does not, so fall back to gathering
        every rank's chunks and slicing locally. That fallback pickles
        through CPU and is O(W^2) in traffic -- a correctness-only path
        (real deployments use NCCL all_to_all or fused MoE comm libraries
        like DeepEP).
        """
        if dist.get_backend() == "nccl":
            dist.all_to_all(out_chunks, in_chunks)
        else:
            # gathered[w] = rank w's in_chunks list; what rank w sends to ME
            # is gathered[w][self.ep_rank]
            gathered = [None] * self.ep_size
            dist.all_gather_object(gathered, in_chunks)
            for w in range(self.ep_size):
                out_chunks[w].copy_(gathered[w][self.ep_rank])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ep_size == 1:
            return self._forward_no_parallel(x)

        T = x.size(0)
        hidden = x.size(1)

        # ---- 1. route (replicated) ----
        logits = self.gate(x)                            # (T, E)
        weights, ids = logits.topk(self.top_k, dim=-1)   # (T, k)
        weights = weights.softmax(dim=-1)

        # flatten: every token appears top_k times, one item per (token, expert) pair
        flat_ids = ids.reshape(-1)                       # (T*k,)
        flat_weights = weights.reshape(-1)
        token_indices = torch.arange(T, device=x.device).repeat_interleave(self.top_k)
        home_rank = flat_ids // self.experts_per_rank    # where each item must go

        # ---- 2. dispatch: bucket items by expert home rank ----
        send_hidden, send_weights, send_indices, send_ids = [], [], [], []
        send_counts = []
        for w in range(self.ep_size):
            sel = home_rank == w
            if not sel.any():
                # Empty bucket: pad with a weight-0 dummy so every all-to-all
                # chunk is non-empty. A weight-0 item contributes nothing to
                # the final output. The dummy must route to a valid local
                # expert on rank w, so give it global expert w * experts_per_rank
                # (local id 0 on the receiver).
                send_hidden.append(x[:1])
                send_weights.append(flat_weights.new_zeros(1))
                send_indices.append(token_indices.new_zeros(1))
                send_ids.append(flat_ids.new_full((1,), w * self.experts_per_rank))
            else:
                send_hidden.append(x[token_indices[sel]])
                send_weights.append(flat_weights[sel])
                send_indices.append(token_indices[sel])
                send_ids.append(flat_ids[sel])
            send_counts.append(int(sel.sum()))

        # exchange counts: all_gather_object collects each rank's send_counts
        # LIST, so recv_counts_per_rank[w] = rank w's full list; the number of
        # items rank w sends to ME is its entry at index self.ep_rank.
        recv_counts_per_rank = [None] * self.ep_size
        dist.all_gather_object(recv_counts_per_rank, send_counts)
        recv_counts = [recv_counts_per_rank[w][self.ep_rank] for w in range(self.ep_size)]

        # all-to-all: hidden states, gate weights, source token indices, expert ids
        recv_hidden_chunks = [torch.empty(n, hidden, device=x.device, dtype=x.dtype)
                              for n in recv_counts]
        self._all_to_all(recv_hidden_chunks, send_hidden)
        recv_weight_chunks = [torch.empty(n, device=x.device, dtype=x.dtype)
                              for n in recv_counts]
        self._all_to_all(recv_weight_chunks, send_weights)
        recv_index_chunks = [torch.empty(n, device=x.device, dtype=torch.long)
                             for n in recv_counts]
        self._all_to_all(recv_index_chunks, send_indices)
        recv_id_chunks = [torch.empty(n, device=x.device, dtype=torch.long)
                          for n in recv_counts]
        self._all_to_all(recv_id_chunks, send_ids)

        recv_hidden = torch.cat(recv_hidden_chunks)      # (S, hidden), S = sum(recv_counts)
        recv_weights = torch.cat(recv_weight_chunks)
        recv_indices = torch.cat(recv_index_chunks)
        recv_ids = torch.cat(recv_id_chunks)

        # ---- 3. expert forward on local experts ----
        # Sort items by local expert id so each expert's tokens are contiguous:
        # one big GEMM per expert instead of scattered tiny ones. This is THE
        # performance-critical trick of MoE kernels.
        local_ids = recv_ids - self.ep_rank * self.experts_per_rank
        order = torch.argsort(local_ids, stable=True)
        local_ids_sorted = local_ids[order]
        x_sorted = recv_hidden[order]

        out_sorted = torch.zeros_like(x_sorted)
        for e in range(self.experts_per_rank):
            sel = local_ids_sorted == e
            if sel.any():
                out_sorted[sel] = self.experts[e](x_sorted[sel])

        # invert the permutation, then apply the gate weights
        out = torch.zeros_like(x_sorted)
        out[order] = out_sorted
        out = out * recv_weights.unsqueeze(-1)

        # ---- 4. combine: send results back to each token's home rank ----
        # The receive order is rank-major, so split back by recv_counts to get
        # one chunk per source rank; each chunk goes back to where it came from.
        out_chunks = list(torch.split(out, recv_counts))
        index_chunks = list(torch.split(recv_indices, recv_counts))
        # same pattern as dispatch: what rank w sends back to me is rank w's
        # recv_counts entry for self.ep_rank
        combine_recv_counts_per_rank = [None] * self.ep_size
        dist.all_gather_object(combine_recv_counts_per_rank, recv_counts)
        combine_recv_counts = [combine_recv_counts_per_rank[w][self.ep_rank]
                               for w in range(self.ep_size)]

        combine_out_chunks = [torch.empty(n, hidden, device=x.device, dtype=x.dtype)
                              for n in combine_recv_counts]
        self._all_to_all(combine_out_chunks, out_chunks)
        combine_index_chunks = [torch.empty(n, device=x.device, dtype=torch.long)
                                for n in combine_recv_counts]
        self._all_to_all(combine_index_chunks, index_chunks)

        # ---- 5. scatter-add: each token sums its top_k weighted expert outputs ----
        final = torch.zeros_like(x)
        final.index_add_(0, torch.cat(combine_index_chunks), torch.cat(combine_out_chunks))
        return final

    def _forward_no_parallel(self, x: torch.Tensor) -> torch.Tensor:
        # ep_size == 1: same math as MoELayer, no collectives
        logits = self.gate(x)
        weights, ids = logits.topk(self.top_k, dim=-1)
        weights = weights.softmax(dim=-1)
        out = torch.zeros_like(x)
        for e in range(self.experts_per_rank):
            for k in range(self.top_k):
                mask = ids[:, k] == e
                if mask.any():
                    out[mask] += weights[mask, k, None] * self.experts[e](x[mask])
        return out


if __name__ == "__main__":
    # how to run?
    # 1. cd src/myvllm/layers
    # 2. uv run torchrun --nproc_per_node=1 moe.py
    # 3. uv run torchrun --nproc_per_node=4 moe.py

    def _init_dist():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available() and world_size <= torch.cuda.device_count():
            # one GPU per rank: use NCCL (the real multi-GPU path)
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            dist.init_process_group(backend="nccl", init_method="env://",
                                    device_id=local_rank)
        else:
            # ranks share physical GPUs (or no GPU at all): NCCL refuses
            # duplicate devices ("Duplicate GPU detected" -> invalid usage,
            # no env override exists), so use gloo. Correctness is identical;
            # only bandwidth differs, and bandwidth measured on shared GPUs
            # is meaningless anyway.
            if torch.cuda.is_available():
                dev_id = local_rank % torch.cuda.device_count()
                torch.cuda.set_device(dev_id)
                device = torch.device("cuda", dev_id)
            else:
                device = torch.device("cpu")
            dist.init_process_group(backend="gloo", init_method="env://")
        return rank, world_size, local_rank, device

    def _init_weights(layer, seed):
        # ReplicatedLinear allocates torch.empty (framework convention: weights
        # come from checkpoints), and garbage bits can be inf/nan -- tests
        # need finite random weights instead.
        g = torch.Generator(device="cpu").manual_seed(seed)
        for p in layer.parameters():
            # generate on CPU then copy: the generator stays CPU while the
            # parameters may live on CUDA
            p.data.copy_(torch.randn(p.shape, generator=g))

    # MoE with num_experts=1, top_k=1 must equal a single expert FFN
    @torch.no_grad()
    def test_moe_degenerate_to_dense(device):
        hidden, inter, T = 64, 128, 32
        moe = MoELayer(hidden, inter, num_experts=1, top_k=1).to(device)
        dense = MLPExpert(hidden, inter).to(device)

        g = torch.Generator(device="cpu").manual_seed(2026)
        wg = torch.randn(inter, hidden, generator=g)
        wu = torch.randn(inter, hidden, generator=g)
        wd = torch.randn(hidden, inter, generator=g)
        bg = torch.randn(inter, generator=g)
        bu = torch.randn(inter, generator=g)
        bd = torch.randn(hidden, generator=g)

        for expert in (moe.experts[0], dense):
            expert.gate_proj.weight.data.copy_(wg)
            expert.up_proj.weight.data.copy_(wu)
            expert.down_proj.weight.data.copy_(wd)
            expert.gate_proj.bias.data.copy_(bg)
            expert.up_proj.bias.data.copy_(bu)
            expert.down_proj.bias.data.copy_(bd)

        g = torch.Generator(device="cpu").manual_seed(7)
        x = torch.randn(T, hidden, generator=g).to(device)

        y_moe = moe(x)
        y_dense = dense(x)
        max_err = (y_moe - y_dense).abs().max().item()
        ok = torch.allclose(y_moe, y_dense, rtol=1e-4, atol=1e-4)
        if dist.get_rank() == 0:
            print(f"[MoEDegenerate] allclose={ok}, max_abs_err={max_err:.6f}")

    # routing math: weighted top-k scatter-add vs naive per-token reference
    @torch.no_grad()
    def test_moe_single_gpu(device):
        hidden, inter, T = 64, 128, 96
        num_experts, top_k = 4, 2
        moe = MoELayer(hidden, inter, num_experts, top_k).to(device)
        _init_weights(moe, 42)

        g = torch.Generator(device="cpu").manual_seed(2026)
        x = torch.randn(T, hidden, generator=g).to(device)

        y = moe(x)

        # naive reference: recompute routing, per-token per-pick expert forward
        logits = moe.gate(x)
        w, ids = logits.topk(top_k, dim=-1)
        w = w.softmax(dim=-1)
        ref = torch.zeros_like(x)
        for i in range(T):
            for j in range(top_k):
                ref[i] += (w[i, j] * moe.experts[ids[i, j]](x[i:i + 1])).squeeze(0)

        # looser tolerance than linear.py: layer and reference accumulate the
        # top-k contributions in different orders, so float32 rounding differs
        max_err = (y - ref).abs().max().item()
        ok = torch.allclose(y, ref, rtol=1e-3, atol=1e-3)
        if dist.get_rank() == 0:
            print(f"[MoESingleGPU] allclose={ok}, max_abs_err={max_err:.6f}")

    # EP layer vs full single-GPU reference, identical input on all ranks
    @torch.no_grad()
    def test_moe_expert_parallel(device):
        ep_size = dist.get_world_size()
        hidden, inter, T = 64, 128, 96
        num_experts, top_k = 4 * ep_size, 2

        ep_layer = ExpertParallelMoELayer(hidden, inter, num_experts, top_k).to(device)
        ref = MoELayer(hidden, inter, num_experts, top_k).to(device)
        # finite init for the gate only: every expert weight AND bias is
        # overwritten below from `full` (the seeded parameter streams of the
        # two layers cannot align -- ref has num_experts experts, ep_layer
        # only experts_per_rank)
        _init_weights(ref, 42)

        # every rank generates the FULL expert weights and biases from the
        # same seed, then loads the full set into ref and its own shard
        # into ep_layer
        g = torch.Generator(device="cpu").manual_seed(2026)
        full = {e: (
            torch.randn(inter, hidden, generator=g),
            torch.randn(inter, hidden, generator=g),
            torch.randn(hidden, inter, generator=g),
            torch.randn(inter, generator=g),
            torch.randn(inter, generator=g),
            torch.randn(hidden, generator=g),
        ) for e in range(num_experts)}

        ep_rank = dist.get_rank()
        per = num_experts // ep_size
        for e in range(num_experts):
            wg, wu, wd, bg, bu, bd = full[e]
            ref.experts[e].gate_proj.weight.data.copy_(wg)
            ref.experts[e].up_proj.weight.data.copy_(wu)
            ref.experts[e].down_proj.weight.data.copy_(wd)
            ref.experts[e].gate_proj.bias.data.copy_(bg)
            ref.experts[e].up_proj.bias.data.copy_(bu)
            ref.experts[e].down_proj.bias.data.copy_(bd)
            if ep_rank * per <= e < (ep_rank + 1) * per:
                local = e - ep_rank * per
                ep_layer.experts[local].gate_proj.weight.data.copy_(wg)
                ep_layer.experts[local].up_proj.weight.data.copy_(wu)
                ep_layer.experts[local].down_proj.weight.data.copy_(wd)
                ep_layer.experts[local].gate_proj.bias.data.copy_(bg)
                ep_layer.experts[local].up_proj.bias.data.copy_(bu)
                ep_layer.experts[local].down_proj.bias.data.copy_(bd)

        # both layers must route identically, so share the gate weights
        ep_layer.gate.weight.data.copy_(ref.gate.weight.data)

        # identical input on every rank
        g = torch.Generator(device="cpu").manual_seed(7)
        x = torch.randn(T, hidden, generator=g).to(device)

        y_ep = ep_layer(x)
        y_ref = ref(x)
        # same tolerance rationale as MoESingleGPU: scatter-add order differs
        max_err = (y_ep - y_ref).abs().max().item()
        ok = torch.allclose(y_ep, y_ref, rtol=1e-3, atol=1e-3)
        if dist.get_rank() == 0:
            print(f"[MoEExpertParallel] allclose={ok}, max_abs_err={max_err:.6f}")

    rank, world_size, local_rank, device = _init_dist()
    if rank == 0:
        print(f"Running MoE tests with world_size={world_size} on device={device}")

    # The test output 'allclose=True' means passed.
    test_moe_degenerate_to_dense(device)
    test_moe_single_gpu(device)
    test_moe_expert_parallel(device)

    dist.barrier()
    dist.destroy_process_group()
