import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from myvllm.layers.attention import paged_attention_prefill, store_kvcache


def reference_attn(q, k, v, q_start, kv_start, scale):
    """
    手写参考实现（纯 torch，fp32 计算）：带绝对位置因果掩码的 attention。

    q: (q_len, num_heads, head_dim)  chunk 的 query
    k/v: (kv_len, num_kv_heads, head_dim)  该序列全部已计算的 KV
    q_start: chunk 第一个 token 的绝对位置（= 缓存前缀 + 历史 chunk 长度）
    kv_start: 该序列 KV 第一个 token 的绝对位置（无前缀时为 q_start）
    """
    num_heads, head_dim = q.shape[1], q.shape[2]
    num_kv_heads = k.shape[1]
    group = num_heads // num_kv_heads

    # 绝对位置因果掩码：query 位置只能看 <= 自己位置的 KV
    q_pos = torch.arange(q_start, q_start + q.shape[0], device=q.device)
    kv_pos = torch.arange(kv_start, kv_start + k.shape[0], device=q.device)
    causal = (q_pos[:, None] >= kv_pos[None, :]).to(torch.float32)

    out = torch.empty_like(q)
    for h in range(num_heads):
        kk = k[:, h // group].to(torch.float32)
        vv = v[:, h // group].to(torch.float32)
        scores = q[:, h].to(torch.float32) @ kk.T * scale
        scores = scores.masked_fill(causal == 0, float("-inf"))
        p = torch.softmax(scores, dim=-1)
        out[:, h] = (p @ vv).to(q.dtype)
    return out


def build_and_compare(prefix_len, chunk_a, chunk_b, dtype=torch.float16, device="cuda"):
    """
    构造一个混合 batch：
    - 序列 A：prefix_len 个 token 已在缓存（模拟前缀缓存/历史 chunk），本 chunk 有 chunk_a 个 token
    - 序列 B：无前缀（第一个 chunk），本 chunk 有 chunk_b 个 token
    两条序列共用一次 paged prefill 调用，验证 per-seq 边界 + GQA + 因果掩码。
    """
    torch.manual_seed(0)
    block_size = 64
    num_blocks = 16
    num_heads, num_kv_heads, head_dim = 4, 2, 64
    scale = 1.0 / head_dim**0.5

    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)

    # ---- 构造 KV 数据并写入缓存（用真实的 store_kvcache 路径）
    # 序列 A：KV 共 prefix_len + chunk_a 个 token，物理块 0..num_blocks_a-1
    kv_a_len = prefix_len + chunk_a
    k_a = torch.randn(kv_a_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_a = torch.randn(kv_a_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    num_blocks_a = (kv_a_len + block_size - 1) // block_size
    bt_a = list(range(num_blocks_a))
    slots_a = [bt_a[t // block_size] * block_size + t % block_size for t in range(kv_a_len)]
    store_kvcache(k_a, v_a, k_cache, v_cache, torch.tensor(slots_a, device=device), block_size)

    # 序列 B：无前缀，chunk 就是全部 KV，用 A 之后的空闲块（可能占多个块）
    k_b = torch.randn(chunk_b, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_b = torch.randn(chunk_b, num_kv_heads, head_dim, device=device, dtype=dtype)
    num_blocks_b = (chunk_b + block_size - 1) // block_size
    bt_b = list(range(num_blocks_a, num_blocks_a + num_blocks_b))
    slots_b = [bt_b[t // block_size] * block_size + t % block_size for t in range(chunk_b)]
    store_kvcache(k_b, v_b, k_cache, v_cache, torch.tensor(slots_b, device=device), block_size)

    # ---- 本批 chunk 的 query
    q_a = torch.randn(chunk_a, num_heads, head_dim, device=device, dtype=dtype)
    q_b = torch.randn(chunk_b, num_heads, head_dim, device=device, dtype=dtype)
    q_all = torch.cat([q_a, q_b], dim=0)

    # ---- 边界（拼接后的全局坐标）
    cu_q = torch.tensor([0, chunk_a, chunk_a + chunk_b], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, kv_a_len, kv_a_len + chunk_b], dtype=torch.int32, device=device)
    max_num_blocks = max(num_blocks_a, num_blocks_b)
    block_tables = torch.tensor([bt_a + [-1] * (max_num_blocks - len(bt_a)),
                                 bt_b + [-1] * (max_num_blocks - len(bt_b))],
                                dtype=torch.int32, device=device)

    # ---- 内核输出 vs 参考实现
    out = paged_attention_prefill(q_all, k_cache, v_cache, block_tables, cu_q, cu_k,
                                  scale, num_heads, num_kv_heads, head_dim, block_size)

    # A 的 KV 从位置 0 开始（前缀在序列开头），B 的 KV 从其 chunk 起点开始
    ref_a = reference_attn(q_a, k_a, v_a, q_start=prefix_len, kv_start=0, scale=scale)
    ref_b = reference_attn(q_b, k_b, v_b, q_start=kv_a_len, kv_start=kv_a_len, scale=scale)
    ref = torch.cat([ref_a, ref_b], dim=0)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    return out, ref


if __name__ == "__main__":
    cases = [
        # (前缀长度, 序列 A chunk, 序列 B chunk)
        (90, 50, 40),    # 前缀跨块边界（90 = 64 + 26），chunk A 从块中间开始
        (64, 64, 64),    # 前缀恰好整块，两个 chunk 也是整块
        (1, 63, 5),      # 极端短前缀 / 短 chunk
        (0, 128, 37),    # 无前缀 + 跨块 chunk
        (200, 33, 100),  # 长前缀 + 中等 chunk
    ]
    for prefix_len, chunk_a, chunk_b in cases:
        out, ref = build_and_compare(prefix_len, chunk_a, chunk_b)
        err = (out.float() - ref.float()).abs().max().item()
        print(f"prefix={prefix_len:3d} chunk_a={chunk_a:3d} chunk_b={chunk_b:3d} "
              f"max_abs_err={err:.2e}  OK")
    print("ALL CASES PASSED: paged prefill kernel == manual causal attention reference")
