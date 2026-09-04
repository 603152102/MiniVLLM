说清楚这个项目里 chunked prefill 的本质：长 prompt 分块做 prefill，每个 chunk 只算本 chunk 的 K/V 写入缓存，但 attention 要 attend 
  到"缓存里之前所有 token 的 K/V"。中间 chunk 不采样、不出 token，prompt 全部跑完后才开始 decode。所以改动横跨 6 个文件，按依赖顺序：
   
  需要改动的文件
     
  1. engine/sequence.py — 序列状态（基础）
  - 新增 num_prefilled_tokens（已 prefill 的 prompt token 数，含前缀缓存命中的），配套 num_unprefilled_tokens、num_computed_tokens 属性
  - 新增取 chunk 的方法，如 get_prefill_chunk(max_tokens) -> list[int]
  - 状态流转：prompt 没跑完的序列不能标 RUNNING（chunk 阶段不出 token），用 WAITING 或新增状态表示"prefill 进行中"
  - ⚠️ __getstate__/__setstate__ 必须带上新字段——序列要靠 pickle 经共享内存传给 worker

  2. engine/block_manager.py — 块级追加
  - append(seq) 现在只支持 1 个 token（decode），需要泛化成 append_tokens(seq, n)：chunk 一次写多个 token，跨块边界时分配新块、块满时算 hash
    进前缀缓存
  - allocate(seq) 可以保持一次预留全部块（MVP 简化），但"已写 KV 的 token 数"要和 num_cached_tokens 区分开

  3. engine/scheduler.py — 调度策略（核心逻辑）
  - schedule() 现在要求整个 prompt 一次塞进 max_num_batched_tokens（第 44 行），改成：塞不下时取 min(剩余, max_num_batched_tokens - 已调度) 长度的   
    chunk
  - postprocess()：prefill chunk 若 prompt 未跑完 → 不 append token、不检查 EOS/max_tokens，只更新 num_prefilled_tokens
  - 阶段划分：MVP 先让 chunk 预填和 decode 分开跑（同现状的 prefill/decode 交替），进阶再把 chunk 和 decode 混进同一 batch——那才是 vLLM chunked      
    prefill 平滑延迟的精髓
  - 改完要同步更新 tests/test_scheduler.py

  4. engine/model_runner.py — 数据准备（最容易错的地方）
  - prepare_prefill(seqs)：
    - input_ids 只取 chunk 而不是全部剩余 prompt（现在是从 num_cached_tokens 取到末尾）
    - slot_mapping 要支持从块中间开始（第二个 chunk 的开头在某个块内部）——通用写法：逐 token 算 slot = block_table[t // block_size] * block_size + t 
      % block_size
    - cu_seqlens_q ≠ cu_seqlens_k：q 长度 = chunk 长度，k 长度 = 整条序列已计算的长度。这是 chunked prefill 的标志性特征，也是 attention
      分支的判断依据
    - prefill 时也要传 block_tables（现有代码第 303-309 行在 cu_seqlens_q[-1] < cu_seqlens_k[-1] 时已经会传，chunked prefill 下这个条件恒成立）      
  - run() 返回结构：chunk 不完整时不采样，给 step() 一个"这是 chunk、无 token"的信号
  - context 里建议直接带上 positions（见下），而不是让模型自己算

  5. models/qwen3.py — RoPE 位置（最隐蔽的 bug 源）
  - 第 206-213 行位置生成逻辑要改：chunk 的 token 位置 = num_prefilled_tokens + 0..chunk_len-1，不是从 0 重排
  - 推荐方案（vLLM 同款）：model_runner 在 context 里放一个显式 positions 张量，qwen3.py 直接读，不再自己从 cu_seqlens 推导
      分支的判断依据
    - prefill 时也要传 block_tables（现有代码第 303-309 行在 cu_seqlens_q[-1] < cu_seqlens_k[-1] 时已经会传，chunked prefill 下这个条件恒成立）      
  - run() 返回结构：chunk 不完整时不采样，给 step() 一个"这是 chunk、无 token"的信号
  - context 里建议直接带上 positions（见下），而不是让模型自己算

  5. models/qwen3.py — RoPE 位置（最隐蔽的 bug 源）
  - 第 206-213 行位置生成逻辑要改：chunk 的 token 位置 = num_prefilled_tokens + 0..chunk_len-1，不是从 0 重排
  - 推荐方案（vLLM 同款）：model_runner 在 context 里放一个显式 positions 张量，qwen3.py 直接读，不再自己从 cu_seqlens 推导
  - llama.py 重写时同样处理

  6. layers/attention.py + utils/context.py — 新内核（最难的部分）
  - 现在的 flash_attention_varlen_kernel 只 attend 本次 pass 算出的 K/V，不读缓存——这正是 block_manager.py:64-70 注释说的"跨序列前缀复用没启用"的原因
  - 需要的正确做法（vLLM 同款）：两步走——
    a. 已有的 store_kvcache_kernel 先把本 chunk 的 K/V 写入缓存
    b. 新的 paged prefill kernel：从缓存读所有已计算 token 的 K/V（本 chunk 的 K/V 此时已在缓存里），做 flash 式 online softmax 注意力
    - 这样 chunk 内 K/V 和缓存 K/V 是同一个来源，不需要"混合"逻辑，一个 paged 读取 kernel 就搞定
  - Attention.forward prefill 分支加判断：cu_seqlens_q[-1] < cu_seqlens_k[-1]（或 context 加显式 is_chunked_prefill 标志）时走 paged prefill
    路径；prompt 一次跑完的仍走原 flash kernel 快路径
  - utils/context.py：加 is_chunked_prefill / positions 字段

  推荐的实现顺序

  1. sequence + block_manager：状态字段和 append_tokens → 先过单测
  2. scheduler：拆 chunk、更新 postprocess（不混合 decode）→ tests/test_scheduler.py 加 chunked 用例
  3. model_runner + qwen3：chunk 数据准备和位置修正 → 此时 attention 数值还不完全对（只 attend 本 chunk），但可以先用 1-token chunk
     的退化情况验证管线通
  4. 新 paged prefill kernel：写 kernel + 接线 → 数值对齐测试：同一个 prompt，chunked 跑和一次跑完的输出必须逐位一致（allclose）
  5. 进阶：chunk + decode 混合调度

  工作量上第 4 步（paged prefill kernel）占大头，其余是状态管理。如果你要动手，建议从 1、2 步开始，我可以随时帮你看具体实现。



    1. 它被 BlockManager.allocate() 当作 hash 链的起点消费
     
  # block_manager.py allocate()
  h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) ...
  if not no_cache_found:
      seq.num_cached_tokens += self.block_size   # 缓存命中才加

  num_cached_tokens 的增长是块管理器根据 hash 命中情况自己控制的。如果你在 scheduler 里 seq.num_cached_tokens += chunk_len 假装"进度"，下一次
  allocate() 会以为这些 token 全是缓存命中，hash 链直接从中间开始算——前缀缓存记账被污染。
  
  2. deallocate() 会把它硬清零

  # block_manager.py deallocate()
  seq.block_table = []
  seq.num_cached_tokens = 0

  抢占（preemption）时整个字段归零。就算你敢拿它当游标，序列一被抢占，"已经 prefill 了多少"的信息就丢了。

  3. 它的粒度是块（block_size 的整数倍），chunk 却可能停在块中间

  chunk 长度由 max_num_batched_tokens 决定，任意值。第二个 chunk 可能从某个块的中间开始——块粒度字段表达不了"第 1 块已写 130/256 个 token"这种状态。  

  另外还有两个结构性问题：

  4. 状态模型只有"没开始"和"prefill 完"两态。prepare_prefill 永远是整段处理：

  input_ids.extend(token_ids[num_cached_tokens:])   # 一次性取到 prompt 末尾

  序列本身没有"给我下一段 chunk"的 API，scheduler/model_runner 想拆块都没法干净地拆。

  5. 多卡时序列靠 pickle 过共享内存（__getstate__/__setstate__）。没有游标字段，worker 进程收到序列后不知道 chunk 从哪里开始、RoPE
  位置从多少起算——即使 rank 0 改了调度，worker 也算不对。

  该怎么改

  核心就一件事：加一个"已 prefill 的 prompt token 数"游标字段，和 num_cached_tokens 正交。缓存命中数继续归块管理器管，游标归 scheduler 管，互不污染。

  # __init__ 里加一行
  self.num_prefilled_tokens = 0   # 已通过 prefill 计算写入 KV cache 的 prompt token 数

  然后派生三个只读属性（和现有 property 风格一致）：

  @property
  def num_computed_tokens(self):
      """prompt 中 KV 已经有效的 token 数（前缀缓存命中 + chunk 已算）"""
      return self.num_cached_tokens + self.num_prefilled_tokens

  @property
  def num_unprefilled_tokens(self):
      """prompt 中还没 prefill 完的 token 数"""
      return self.num_prompt_tokens - self.num_computed_tokens

  @property
  def is_prefill_complete(self):
      """prompt 是否已全部 prefill 完（可以开始 decode）"""
      return self.num_unprefilled_tokens == 0

  注意 num_computed_tokens = num_cached_tokens + num_prefilled_tokens 这个等式成立的前提是：缓存命中永远是 prompt 的前缀（allocate()
  从开头逐块走、第一个 miss 就停），chunk 永远紧跟在已计算部分之后——这个前提在你的代码里成立。

  再加两个方法，把游标操作封装进类内部（scheduler/model_runner 不直接戳字段，和 append_token 的封装风格一致）：

  def get_next_prefill_tokens(self, max_tokens: int) -> list[int]:
      """取下一个 prefill chunk，最长 max_tokens 个 token"""
      start = self.num_computed_tokens
      return self.token_ids[start:start + max_tokens]

  def advance_prefill(self, n_tokens: int) -> None:
      """标记又有 n_tokens 个 prompt token 完成 prefill"""
      assert 0 <= n_tokens <= self.num_unprefilled_tokens
      self.num_prefilled_tokens += n_tokens

  最后，__getstate__/__setstate__ 必须带上新字段——这是多卡路径的正确性要求：

  def __getstate__(self):
      return (
          self.num_tokens,
          self.num_prompt_tokens,
          self.num_cached_tokens,
          self.num_prefilled_tokens,        # 新增
          self.block_table,
          self.token_ids if self.num_completion_tokens == 0 else self.last_token
      )

  def __setstate__(self, state):
      (
          self.num_tokens,
          self.num_prompt_tokens,
          self.num_cached_tokens,
          self.num_prefilled_tokens,        # 新增
          self.block_table,
          last_token_or_ids
      ) = state
      ...

  两个连带注意点

  1. Scheduler.preempt() 里要重置游标：deallocate() 会把 num_cached_tokens 清零，但 num_prefilled_tokens 也得归零——块被释放了，KV 没了，之前 chunk   
     算的全作废，下次从头 prefill。这个容易漏。
  2. 不需要改的东西：num_tokens、append_token（decode 不受影响）、num_blocks/block(i)（整 prompt 的块划分不变）、last_block_num_tokens（decode       
          self.num_tokens,
          self.num_prompt_tokens,
          self.num_cached_tokens,
          self.num_prefilled_tokens,        # 新增
          self.block_table,
          last_token_or_ids
      ) = state
      ...

  两个连带注意点

  1. Scheduler.preempt() 里要重置游标：deallocate() 会把 num_cached_tokens 清零，但 num_prefilled_tokens 也得归零——块被释放了，KV 没了，之前 chunk   
     算的全作废，下次从头 prefill。这个容易漏。
  2. 不需要改的东西：num_tokens、append_token（decode 不受影响）、num_blocks/block(i)（整 prompt 的块划分不变）、last_block_num_tokens（decode       
     还用得上）。len(seq) 语义也可以保持不变——调度成本从"整个 prompt"改成"chunk 长度"是 scheduler 的改动，不是 Sequence 的责任。

  改完 Sequence 后，下一步就是 Scheduler.schedule() 里的第 44 行：把 len(seq) + current_scheduled_tokens <= max_num_batched_tokens
  的整段判断，换成基于 get_next_prefill_tokens() 的 chunk 判断。要继续的话告诉我。