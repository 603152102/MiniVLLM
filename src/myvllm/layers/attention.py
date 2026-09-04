import triton 
import triton.language as tl
from myvllm.utils import get_context
import torch
import torch.nn as nn

@triton.jit
def store_kvcache_kernel(
    key_ptr, # pointer to what we want to store
    value_ptr,
    k_cache_ptr, # pointer to where we want to store
    v_cache_ptr,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr
):
    """
    Store keys and values into paged KV cache.
    Each token is mapped to a slot via slot_mapping.
    Grid layout: (num_tokens, num_kv_heads)
    Cache layout: (num_blocks, block_size, num_kv_heads, head_dim)
    """
    # thread ID, in dimension 0
    token_idx = tl.program_id(0) # each GPU thread processes one token
    # slot ID, where in cache to store this token
    slot_idx = tl.load(slot_mapping_ptr + token_idx)# slot_mapping[token_idx] 的地址
    
    if slot_idx == -1:
        return
    
    # Calculate which block and position within block
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    
    # Process each head
    # program_id(0) = which token
    # program_id(1) = which head
    head_idx = tl.program_id(1)
    
    # it creates a vector [0, 1, ..., head_dim-1]
    # Load key and value for this token and head
    head_offsets = tl.arange(0, head_dim)
    # Input: (num_tokens, num_kv_heads, head_dim)
    # example: input_offset = 5 * (8 * 128) + 3 * 128 + [0, 1, 2, ..., 127]
    #         = 5120 + 384 + [0, 1, 2, ..., 127]
    #         = [5504, 5505, 5506, ..., 5631]
    input_offset = (token_idx * num_kv_heads * head_dim + # skip previous tokens
                    head_idx * head_dim + # skip previous heads
                    head_offsets)

    # Cache: (num_blocks, block_size, num_kv_heads, head_dim)
    cache_offset = (block_idx * block_size * num_kv_heads * head_dim + # skip previous blocks
                   block_offset * num_kv_heads * head_dim + # skip previous positions / token in block
                   head_idx * head_dim + # skip previous kv heads
                   head_offsets) 
    
    # load key and value value floats from the pointers's memory
    #     CUDA                       Triton
    # x = ptr[offset]     ≈      x = tl.load(ptr + offset)
    # ptr[offset] = x     ≈      tl.store(ptr + offset, x)
    key = tl.load(key_ptr + input_offset)
    value = tl.load(value_ptr + input_offset)
    
    # store into cache
    tl.store(k_cache_ptr + cache_offset, key)
    tl.store(v_cache_ptr + cache_offset, value)


def store_kvcache(
    key: torch.Tensor, 
    value: torch.Tensor, 
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor, 
    slot_mapping: torch.Tensor,
    block_size: int
):
    """
    Store key-value pairs into paged cache.
    
    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: (num_tokens,) - maps each token to a cache slot
        block_size: number of tokens per block
    """
    num_tokens, num_kv_heads, head_dim = key.shape
    
    # Make contiguous if needed
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    
    assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"
    assert slot_mapping.numel() == num_tokens, "Slot mapping size must match number of tokens"
    
    grid = (num_tokens, num_kv_heads)
    # launch num_tokens x num_kv_heads threads
    store_kvcache_kernel[grid](
        key, # tensors are automatically converted to pointers by triton
        value,
        k_cache,
        v_cache,
        slot_mapping,
        num_kv_heads=tl.constexpr(num_kv_heads),
        head_dim=tl.constexpr(head_dim),
        block_size=tl.constexpr(block_size)
    )


@triton.jit
def flash_attention_varlen_kernel(
    Q, K, V, O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention kernel for variable-length sequences.
    Each program processes one block of queries for one head in one sequence.
    """
    # Program IDs / Block xyz
    start_m = tl.program_id(0) # block index
    off_h = tl.program_id(1) # head index
    seq_idx = tl.program_id(2) # sequence index

    # Determine which KV head to use (for GQA)
    kv_head_idx = off_h // (num_heads // num_kv_heads)
    
    # Load sequence boundaries
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start
    
    # Early exit if this block is beyond sequence length
    if start_m * BLOCK_M >= seq_len:
        return
    
    # Offset for this block of queries
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) #负责seq中的哪些token
    offs_d = tl.arange(0, head_dim)                    #负责head里的哪些维度
    
    # Query pointers: Q has shape (total_tokens, num_heads, head_dim)
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    
    # Load Q block - shape (BLOCK_M, head_dim)
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Initialize output accumulators
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    
    # Number of blocks to process
    num_blocks = tl.cdiv(seq_len, BLOCK_N)
    
    # Loop over K, V blocks
    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Mask for valid positions
        mask_n = offs_n < seq_len
        
        # K pointers: K has shape (total_tokens, num_kv_heads, head_dim)
        k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]
        
        # Load K block - shape (head_dim, BLOCK_N)
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        # Compute QK^T - shape (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k)
        qk = qk * scale
        
        # Apply causal mask: only attend to positions <= current position
        mask_causal = (offs_m[:, None] + seq_start) >= (offs_n[None, :] + seq_start)
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)
        
        # Online softmax update
        # m_ij     = 当前 block 最大值
        # m_i      = 历史最大值
        # m_i_new  = 加上当前 block 后的总最大值
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)   #新旧坐标转换的缩放系数
        p = tl.exp(qk - m_i_new[:, None])
        
        # Rescale previous accumulator
        acc = acc * alpha[:, None]
        
        # Load V block - shape (BLOCK_N, head_dim)
        v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Accumulate weighted values
        acc = acc + tl.dot(p.to(v.dtype), v)
        
        # Update normalizer
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output: O has shape (total_tokens, num_heads, head_dim)
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Optimized Flash Attention for prefill phase with variable-length sequences.
    
    Args:
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim)
        cu_seqlens: cumulative sequence lengths
        scale: attention scale factor
    
    Returns:
        output: (total_tokens, num_heads, head_dim)
    """
    # Make tensors contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Allocate output
    output = torch.empty_like(q)
    
    # Conservative block sizes to avoid OOM on shared memory
    # Shared memory usage ~ BLOCK_M * BLOCK_N * 4 bytes (for float32 attention scores)
    # + BLOCK_M * head_dim * 4 (for Q)
    # + BLOCK_N * head_dim * 4 (for K, V)
    # Want to keep total < 48KB for most GPUs
    
    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16
    
    # Number of sequences
    num_seqs = cu_seqlens.shape[0] - 1
    
    # Find max sequence length to determine grid size
    cu_seqlens_cpu = cu_seqlens.cpu()
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()
    
    # Calculate grid dimensions - launch all kernels at once
    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)
    
    flash_attention_varlen_kernel[grid](
        q, k, v, output,
        cu_seqlens,
        scale,
        num_heads=tl.constexpr(num_heads),
        num_kv_heads=tl.constexpr(num_kv_heads),
        head_dim=tl.constexpr(head_dim),
        BLOCK_M=tl.constexpr(BLOCK_M),
        BLOCK_N=tl.constexpr(BLOCK_N),
    )
    
    return output


@triton.jit
def paged_attention_decode_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Optimized paged attention kernel for decode phase.
    Processes KV cache in chunks.

    The block table is gathered per token rather than per chunk, so a chunk may
    straddle any number of blocks and no relation between BLOCK_N and block_size
    is assumed. Each lane resolves its own token independently:

        token t  ->  block_tables[batch, t // block_size], slot t % block_size
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Determine which KV head this query head uses (for GQA)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # Load context length
    context_len = tl.load(context_lens_ptr + batch_idx)
    
    # Load query: (batch_size, num_heads, head_dim)
    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)
    
    # Initialize accumulators
    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10
    
    # Calculate total number of chunks to process
    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)
    
    # Process all tokens in chunks
    for chunk_idx in range(max_chunks):
        # Global token index for this chunk
        token_start = chunk_idx * BLOCK_N
        
        # Only process if within valid range
        if token_start < context_len:
            # Determine which tokens in this chunk are valid
            offs_n = token_start + tl.arange(0, BLOCK_N)
            logical_block = offs_n // block_size
            # physical_block * block_size already moves the base pointer to the
            # start of the block, so what is added is the offset *within* the
            # block, not the global token index.
            offs_in_block = offs_n % block_size

            in_range = (offs_n < context_len) & (logical_block < max_num_blocks)

            # One block-table entry per token: a chunk is free to span several
            # blocks, and those blocks need not be adjacent in the cache.
            physical_block = tl.load(
                block_tables_ptr + batch_idx * max_num_blocks + logical_block,
                mask=in_range, other=-1)
            valid = in_range & (physical_block != -1)
            # Masked-out lanes still take part in the address arithmetic, so give
            # them block 0 to keep every computed offset inside the cache.
            physical_block = tl.where(valid, physical_block, 0).to(tl.int64)

            # Cache: (num_blocks, block_size, num_kv_heads, head_dim)
            kv_offset = (physical_block[None, :] * (block_size * num_kv_heads * head_dim)
                         + offs_in_block[None, :] * (num_kv_heads * head_dim)
                         + kv_head_idx * head_dim
                         + offs_d[:, None])

            # Compute attention scores for this chunk
            k = tl.load(k_cache_ptr + kv_offset, mask=valid[None, :], other=0.0)
            k = tl.cast(k, tl.float32)
            score = tl.sum(q[:, None] * k, axis=0) * scale
            qk = tl.where(valid, score, -1e10)

            # Online softmax
            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)

            # Rescale accumulator
            acc = acc * alpha
            l_i = l_i * alpha

            # Accumulate weighted values
            v = tl.load(v_cache_ptr + kv_offset, mask=valid[None, :], other=0.0)
            v = tl.cast(v, tl.float32)
            weight = tl.where(valid, p, 0.0)
            acc = acc + tl.sum(weight[None, :] * v, axis=1)
            l_i = l_i + tl.sum(weight)

            m_i = m_i_new
    
    # Normalize
    output = acc / l_i
    
    # Store output
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output)


def paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int
) -> torch.Tensor:
    """
    Compute attention in decode mode using paged KV cache.
    
    Args:
        query: (batch_size, num_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: (batch_size, max_num_blocks)
        context_lens: (batch_size,)
        scale: attention scale factor
    
    Returns:
        output: (batch_size, num_heads, head_dim)
    """
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    
    # Make contiguous
    query = query.contiguous()
    
    output = torch.empty_like(query)
    
    # Chunk size for processing KV tokens
    BLOCK_N = 64 if head_dim <= 128 else 32
    
    grid = (batch_size, num_heads)
    
    paged_attention_decode_kernel[grid](
        output,
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale=tl.constexpr(scale),
        num_heads=tl.constexpr(num_heads),
        num_kv_heads=tl.constexpr(num_kv_heads),
        head_dim=tl.constexpr(head_dim),
        block_size=tl.constexpr(block_size),
        max_num_blocks=tl.constexpr(max_num_blocks),
        BLOCK_N=tl.constexpr(BLOCK_N),
    )

    return output


@triton.jit
def paged_attention_prefill_kernel(
    output_ptr,          # (total_q, num_heads, head_dim) 结果写回这里
    query_ptr,           # (total_q, num_heads, head_dim) 本 chunk 的 Q
    k_cache_ptr,         # (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache_ptr,         # (num_blocks, block_size, num_kv_heads, head_dim)
    block_tables_ptr,    # (num_seqs, max_num_blocks) 每条序列的块表
    cu_seqlens_q_ptr,    # (num_seqs + 1,) 各序列 chunk 的边界
    cu_seqlens_k_ptr,    # (num_seqs + 1,) 各序列"KV 已有效"的边界（缓存前缀 + 历史 chunk + 本 chunk）
    scale,               # attention 缩放系数 1/sqrt(head_dim)
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_M: tl.constexpr,  # 每个 program 处理的 query 数
    BLOCK_N: tl.constexpr,  # 每轮循环加载的 KV token 数
):
    """
    Chunked prefill 的 paged attention 内核。

    思路 = 现有两个内核的组合：
    - 网格划分沿用 flash_attention_varlen_kernel(program_id(2) = 序列，
      program_id(0) = query block,program_id(1) = head);
    - KV 读取沿用 paged_attention_decode_kernel(逐 token 查块表
      block_tables[seq, t // block_size] 定位物理块);
    - 新增:因果掩码用【绝对位置】——query 在完整序列里的位置是
      k_start + chunk 内偏移，只能 attend 位置 ≤ 自己的 KV。

    为什么可以从缓存读一切:Attention.forward 在调用本内核之前已经把
    本 chunk 的 K/V store 进缓存了，所以"历史 chunk 的 KV"和"本 chunk 的
    KV"在缓存里是同一份数据，一个 paged 读取循环就覆盖全部，无需混合。
    """
    # ---- 网格：program_id(0) = 第几个 query block，program_id(1) = 第几个
    #      head，program_id(2) = 第几条序列（和 flash varlen 内核一致）
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    seq_idx = tl.program_id(2)

    # GQA：多个 query head 共享一个 KV head，例如 16 heads / 8 kv_heads，
    # head 8..15 都读 kv head 7 所在的 cache 区域
    kv_head_idx = off_h // (num_heads // num_kv_heads)

    # ---- 序列边界（都是"拼接后全局坐标"）
    q_start = tl.load(cu_seqlens_q_ptr + seq_idx)   # 本序列 chunk 的起点
    q_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1) # 本序列 chunk 的终点
    k_start = tl.load(cu_seqlens_k_ptr + seq_idx)   # 本序列 KV 有效区间的起点
    k_end = tl.load(cu_seqlens_k_ptr + seq_idx + 1) # 本序列 KV 有效区间的终点

    # ---- 本 program 负责的 query：chunk 内偏移 offs_m
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    chunk_len = q_end - q_start
    mask_m = offs_m < chunk_len   # 超出 chunk 的 query 无效
    # 绝对位置：chunk 的 KV 占据 KV 区间 [k_start, k_end) 的【尾部】
    # chunk_len 个位置，所以 chunk 第一个 token 的绝对位置是
    #   k_end - chunk_len
    # （k_start 前面可能还有前缀缓存的历史 KV；这正是 chunked prefill
    #   位置正确的关键——不能从 0 数，也不能拿 k_start 当起点）
    q_pos = (k_end - chunk_len) + offs_m

    # ---- 加载 Q：(total_q, num_heads, head_dim) 的行主序展开
    offs_d = tl.arange(0, head_dim)
    q = tl.load(
        query_ptr + (q_start + offs_m)[:, None] * num_heads * head_dim + off_h * head_dim + offs_d[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )

    # ---- 在线 softmax 状态（和 flash 内核相同的 m / l 重缩放技巧）
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # ---- 主循环：把本序列 [k_start, k_end) 的 KV 按 BLOCK_N 一段一段读出来
    for block_n in range(0, tl.cdiv(k_end - k_start, BLOCK_N)):
        # 本段覆盖的 KV 的绝对位置（拼接后全局坐标，用于因果掩码）
        offs_n = k_start + block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        in_range = offs_n < k_end

        # 逐 token 查块表。注意块表是按【每条序列自己的 token 位置】索引的
        # （0 开始），而 offs_n 是全局坐标，必须先减掉 k_start 转成序列内
        # 局部位置：逻辑块号 = 局部位置 // block_size
        local_pos = offs_n - k_start
        logical_block = local_pos // block_size
        physical_block = tl.load(
            block_tables_ptr + seq_idx * max_num_blocks + logical_block,
            mask=in_range, other=-1)
        valid = in_range & (physical_block != -1)
        # 被掩掉的 lane 仍参与地址运算，给它们块 0 保证地址不越界（与 decode 内核同款处理）
        physical_block = tl.where(valid, physical_block, 0).to(tl.int64)

        # cache 布局 (num_blocks, block_size, num_kv_heads, head_dim)
        offs_in_block = local_pos % block_size
        kv_offset = (physical_block[None, :] * (block_size * num_kv_heads * head_dim)
                     + offs_in_block[None, :] * (num_kv_heads * head_dim)
                     + kv_head_idx * head_dim
                     + offs_d[:, None])

        # 加载 K：形状 (head_dim, BLOCK_N)，供 tl.dot(q, k) 用
        k = tl.load(k_cache_ptr + kv_offset, mask=valid[None, :], other=0.0)

        # QK^T：(BLOCK_M, BLOCK_N)，再乘 scale
        qk = tl.dot(q, k)
        qk = qk * scale

        # 因果掩码：只能 attend 位置 ≤ 自己的 KV。
        # 注意用的是绝对位置比较：q_pos[:, None] >= offs_n[None, :]
        causal = q_pos[:, None] >= offs_n[None, :]
        qk = tl.where(valid[None, :] & causal, qk, -1e10)

        # ---- 在线 softmax 更新（与 flash 内核逐行对应）
        m_ij = tl.max(qk, axis=1)          # 本段最大值
        m_i_new = tl.maximum(m_i, m_ij)    # 历史最大值 vs 本段最大值
        alpha = tl.exp(m_i - m_i_new)      # 旧结果缩放系数
        p = tl.exp(qk - m_i_new[:, None])  # 本段 softmax 分子
        acc = acc * alpha[:, None]         # 旧累加器重缩放
        l_i = l_i * alpha                  # 旧归一化系数重缩放

        # 加载 V：形状必须是 (BLOCK_N, head_dim)，tl.dot(p, v) 才合法
        # （p 是 (BLOCK_M, BLOCK_N)，左乘要求 v 的第一维是 BLOCK_N）。
        # 注意与 K 的指针写法不同：这里行方向是 KV token（offs_in_block[:, None]），
        # 列方向是 head_dim（offs_d[None, :]）
        v_offset = (physical_block[:, None] * (block_size * num_kv_heads * head_dim)
                    + offs_in_block[:, None] * (num_kv_heads * head_dim)
                    + kv_head_idx * head_dim
                    + offs_d[None, :])
        v = tl.load(v_cache_ptr + v_offset, mask=valid[:, None], other=0.0)
        acc = acc + tl.dot(p.to(v.dtype), v)
        l_i = l_i + tl.sum(p, axis=1)
        m_i = m_i_new

    # ---- 最终归一化并写回
    acc = acc / l_i[:, None]
    o_ptrs = output_ptr + (q_start + offs_m)[:, None] * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(output_ptr.dtype.element_ty), mask=mask_m[:, None])


def paged_attention_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    """
    Chunked prefill 的 paged attention 入口。

    Args:
        q: (total_q, num_heads, head_dim) 本批所有 chunk 的 query
        k_cache/v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: (num_seqs, max_num_blocks)，-1 填充
        cu_seqlens_q: (num_seqs + 1,) chunk 边界（q 长度）
        cu_seqlens_k: (num_seqs + 1,) KV 有效边界（k 长度，≥ q 长度）
    """
    q = q.contiguous()

    output = torch.empty_like(q)

    total_q = q.shape[0]
    num_seqs = cu_seqlens_q.shape[0] - 1
    max_num_blocks = block_tables.shape[1]

    # 块大小沿用 flash 内核的经验值（共享内存受限时的保守选择）
    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16

    grid = (triton.cdiv(total_q, BLOCK_M), num_heads, num_seqs)

    paged_attention_prefill_kernel[grid](
        output,
        q,
        k_cache,
        v_cache,
        block_tables,
        cu_seqlens_q,
        cu_seqlens_k,
        scale,
        num_heads=tl.constexpr(num_heads),
        num_kv_heads=tl.constexpr(num_kv_heads),
        head_dim=tl.constexpr(head_dim),
        block_size=tl.constexpr(block_size),
        max_num_blocks=tl.constexpr(max_num_blocks),
        BLOCK_M=tl.constexpr(BLOCK_M),
        BLOCK_N=tl.constexpr(BLOCK_N),
    )

    return output


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        block_size: int = 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.block_size = block_size
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # Store current k, v into cache if cache is allocated
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            # Ensure k, v are in the right shape: (num_tokens, num_kv_heads, head_dim)
            if k.dim() == 4:
                # Batched: (B, N, num_kv_heads, head_dim) -> reshape to (B*N, num_kv_heads, head_dim)
                B, N, num_kv_heads, head_dim = k.shape
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()
            else:
                # Already in correct shape (num_tokens, num_kv_heads, head_dim)
                k_to_store = k.contiguous()
                v_to_store = v.contiguous()
            
            store_kvcache(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping, self.block_size)

        scale = self.scale / (self.head_dim ** 0.5)

        if context.is_prefill:
            # Prefill: use flash attention
            # Varlen mode: (total_tokens, num_heads, head_dim)
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_q must be provided for varlen attention")
            if context.cu_seqlens_k is not None and cu_seqlens[-1] < context.cu_seqlens_k[-1]:
                # Chunked prefill: part of the KV lives in the cache from
                # earlier chunks, and this chunk's K/V was just stored there
                # as well (see store_kvcache above), so attend over the paged
                # cache -- a single source of truth, no mixing needed.
                # Whole-prompt prefills keep the flash fast path below.
                assert context.block_tables is not None, "block_tables must be provided for chunked prefill"
                o = paged_attention_prefill(
                    q,
                    k_cache,
                    v_cache,
                    context.block_tables,
                    cu_seqlens,
                    context.cu_seqlens_k,
                    scale,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.block_size,
                )
            else:
                o = flash_attention_prefill(q, k, v, cu_seqlens, scale,
                                            self.num_heads, self.num_kv_heads, self.head_dim)
            # Output: (total_tokens, num_heads, head_dim) -> (total_tokens, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        else:
            if context.block_tables is None:
                raise ValueError("block_tables must be provided for paged attention")
            if context.context_lens is None:
                raise ValueError("context_lens must be provided for paged attention")
            o = paged_attention_decode(
                q, 
                k_cache, 
                v_cache,
                context.block_tables,
                context.context_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size
            )
            # o: (batch_size, num_heads, head_dim) -> (batch_size, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)


if __name__ == "__main__":
    # Example usage
    layer = Attention(num_heads=8, head_dim=64).cuda()
    B, N, D = 4, 1024, 512
    q = torch.randn(B, N, D).cuda()
    k = torch.randn(B, N, D).cuda()
    v = torch.randn(B, N, D).cuda()
    layer.k_cache = torch.zeros(B, N, D).cuda()
    layer.v_cache = torch.zeros(B, N, D).cuda()
    slot_mapping = torch.arange(N).cuda()

    for _ in range(10):  # Warm-up iterations
        _ = layer(q, k, v)

    import time
    times = []
    for _ in range(100):  # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(q, k, v)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")