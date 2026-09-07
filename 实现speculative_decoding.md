# 投机解码（Speculative Decoding）设计方案

## 本质

让一个**草稿模型**（小/快）先猜 k 个 token，**目标模型**用**一次前向**把这 k+1 个 token 全部验证完，
按概率接受其中前 n 个（0 ≤ n ≤ k+1），被拒绝的位置用目标模型自己的分布补一个 token。
目标模型每次前向产出的 token 数从 1 变成 (k+1)×接受率，这就是加速来源。

每个 decode 步 = 三个阶段：

```
propose（草稿解码 k 步，串行，写草稿 KV）
    → verify（目标模型一次前向算 k+1 个位置的 logits，写目标 KV）
        → accept（逐位置拒绝采样 → 接受 n 个 + 补 1 个目标 token）
            → 提交/回滚（接受的部分落进序列；被拒绝部分的 KV 逻辑上作废）
```

关键洞察：**verify 批的形状恰好就是 chunked prefill 的形状**——每序列 q = k+1 个"新"token、
k = 序列全长（cu_seqlens_q ≠ cu_seqlens_k、positions 接续、paged prefill kernel 读缓存 KV）。
你刚做完的 chunked prefill 六步设施（prepare_prefill 的 chunk 切片、slot_mapping 逐 token 解析、
paged_attention_prefill_kernel、context.positions）就是 verify 阶段 80% 的工程，投机解码是它的自然延伸。

## 数据流（单条序列视角）

序列当前有 t 个 token（其中 prompt p 个，completion t-p 个），草稿预算 k（如 4）：

```
轮次开始：seq.token_ids 长度 t（全部已接受），草稿缓冲区为空
1. propose:  草稿模型从位置 t 起自回归猜 k 个 token → spec = [d0..d{k-1}]，
             记录草稿 logits（k 行，行 i = 采 d_i 时用的分布）
2. verify:   目标模型一次前向，输入 = [x] + [d0..d{k-1}]（x = 位置 t-1 的最后一个已接受
             token，k+1 个位置，positions = [t-1..t+k-1]），KV 写入槽位 t-1..t+k-1
             输出 k+1 行 logits：行 0..k-1 = 验证草稿 token 的分布（行 i 在位置 t+i），
             行 k = bonus 分布（位置 t+k 的预测分布）
             为什么带 x：它重算位置 t-1 的 KV。上一轮的 bonus token 被接受后没有目标
             KV（见下），靠本轮的 x 补上。对已正确的槽位这是幂等重算，代价 1 个位置/轮
3. accept:   行 i（i=0..k-1）验证 d_i 的拒绝采样 → 接受 n 个
   - 若 n = k：全部接受，再从行 k 的分布采 1 个 bonus token → 共 k+1 个
   - 若 n < k：位置 n 拒绝，从 max(0, p_target − q_draft) 归一化分布采 1 个 token → 共 n+1 个
4. 提交/回滚:
   - 接受的 n(+1) 个 token append 进 seq.token_ids，seq.num_tokens += n(+1)
   - 被拒绝的槽位：不动 KV（垃圾值无害——attention 只按 seqlens_k = 序列长度读，
     槽位下次被覆盖），只做逻辑回滚记账
   - 关键细节：bonus token 被接受后其槽位没有目标 KV（本轮没前向过它）——下一轮
     verify 的第一个位置 x 恰好重算它。这是"每轮多算 1 个位置"换来的 KV 永远一致
   - 草稿 KV 侧：截断到新长度，下轮从新位置继续 propose
```

## 算法细节（acceptance.py，纯数学、可离线单测）

拒绝采样保证**输出分布与普通温度采样完全一致**（投机解码是正确性无损的加速）：

- 贪心：接受条件 u ~ U(0,1)，u ≤ p(x_i)/q(x_i)（q=草稿分布，p=目标分布，x_i=草稿 token）。
  拒绝后补的 token = argmax p。
- 温度采样（SpecInfer 算法，实现时对照 Chen et al. 2023 "Accelerating LLM Inference with
  Staged Speculative Decoding" 或 vLLM spec_decode_worker.py 的 rejection 实现）：逐位置
  抽 x_i ~ q_i 与 E_i ~ Exp(1)，接受条件 E_i ≥ log p_i(x_i) − log q_i(x_i)（对首个拒绝位置
  i 而言；多位置联合的全局条件见论文 Algorithm 2）。拒绝后从 (p_i − q_i)_+ 归一化采样。
- 需要的输入：目标 logits [k+1, V]（k 个草稿位置 + 1 个 bonus 位置）、草稿 logits [k, V]、
  草稿 token [k]。返回每序列的 accepted 数 + 补的 token（n==k 时为 bonus，n<k 时为 (p_n−q_n)_+
  的替换采样）。logits 由调用方预先按各自的采样温度缩放（与普通 sampler 同一条路径：
  logits/temperature 后 softmax），acceptance 只做纯 softmax、不再带 temperature 参数——见审阅 #1。
- 重要：现有 `sampler.py` 用 Gumbel-Max 技巧直接 argmax，不暴露分布——acceptance 模块自己
  从 logits 算 softmax 概率，不改 sampler 的采样路径；但**随机数消耗顺序**会变化（投机解码
  天然消耗更多 RNG 抽取），这意味着同种子下投机与非投机的 token 序列在温度采样时不同
  ——正确性验证必须用贪心（见"验收"一节）。

## 需要改动的文件

### 1. engine/sequence.py — 序列投机状态（基础）

- 新增字段：`spec_token_ids: list[int]`（草稿缓冲区，含已接受未清空的部分）、
  `num_spec_verified: int`（缓冲区里已有目标 KV 的位置数，回滚后截断用）。
  草稿 logits 体积大，**不放 Sequence**（Sequence 经 pickle 走共享内存传 worker，只带最小状态）。
- 新增方法：`append_tokens(ids)`、`rollback_tokens(n)`（num_tokens -= n、token_ids 截断）。
- ⚠️ `__getstate__/__setstate__` 必须带新字段（同 chunked prefill 的教训）。

### 2. engine/block_manager.py — 块级回滚

- `append(seq)` 逐 token 调用的现有语义保留；verify 阶段对 k+1 个临时槽位逐 token append
  （can_append 在 propose 前一次性预检 k+1 个槽位是否有块可用）。
- 新增 `rollback(seq, n)`：把最后 n 个 token 的记账撤销——
  - `seq.num_tokens -= n`（KV 垃圾值不清理，attention 按 seqlens 读不到）
  - 若最后一块退回 partial：hash 置 -1、从 hash_to_block_id 摘除（撤销 append 时的 finalize）
  - 若整块退回空：释放该块回 free list
- 前缀缓存的 hash 只在 token **被接受**后 finalize：MVP 让 append 照旧 finalize、rollback 撤销，
  避免改动 append 的正常路径。

### 3. engine/model_runner.py — verify 前向（改动量最大）

- 新增 `prepare_verify(seqs)`：与 prepare_prefill 的 chunk 分支同构，但输入 token 来自
  `[seq.last_token] + seq.spec_token_ids` 而非 `seq.token_ids`：
  - input_ids = concat([x] + spec 缓冲区)，positions = range(t-1, t+k)（t = 当前序列长度，
    第一个位置 x 重算上轮 bonus token 的 KV，幂等）
  - seqlens_q = k+1，seqlens_k = t+k（最后一个 spec token attend 到自身为止，因果）
  - slot_mapping = 槽位 t-1..t+k-1（逐 token 解析，跨块安全——chunked prefill 已验证这套写法）
  - 走既有 paged prefill kernel 路径（cu_seqlens_q < cu_seqlens_k 自动路由）
- 新增 `run_verify(seqs) -> target_logits[num_seqs, k+1, V]`：跑前向、按 cu_seqlens_q
  边界把 varlen logits 切回每序列 k+1 行（行 0..k-1 验证草稿 token，行 k = bonus 分布），
  reset_context 前把切片拷回 CPU/engine 侧
- 现有 `run()` 不变；verify 走 eager 路径（不用 CUDA graph——graph 是 decode 形状专用）。

### 4. spec_decode/draft_runner.py — 草稿模型（新文件）

- **不能**复用 ModelRunner：`ModelRunner.__init__` 会二次 `dist.init_process_group` 崩掉。
  做成轻量类：自己加载 Qwen3 模型权重 + 自己的 KV 池（复用 layers/attention 的 paged decode
  kernel 与 prepare_decode 逻辑，拷贝一份简化版）+ 自己的 BlockManager。
- 接口：`propose(seqs, k) -> (draft_tokens[num_seqs, k], draft_logits[num_seqs, k, V])`——
  每序列 k 步自回归，从"已接受前缀"续跑；部分拒绝时草稿 KV 同步回滚。
- 草稿模型选型（按硬件）：
  - **A. 独立草稿模型**（租卡/换卡后，标准方案）：必须同 tokenizer 同 vocab（Qwen3 家族
    最小 0.6B；或用户训练/下载一个同 vocab 小模型）。显存 ≈ 2×权重 + 2×KV 池。
  - **B. 自投机 MVP**（6GB 本地跑通管线用）：草稿 = 同一个 Qwen3-0.6B 权重。不加层跳时
    无加速（propose 与普通 decode 同价），**仅用于验证管线正确性**；之后加 LayerSkip 式
    跳层（draft 前向只跑奇数层）才回到加速。显存只多一套草稿 KV 池。

### 5. spec_decode/acceptance.py — 接受/拒绝采样（新文件，纯 torch 数学）

见"算法细节"。单测覆盖：全接受、全拒绝、部分接受、q≡p 时接受率=1、分布无偏性
（与直接采样对拍的统计检验可后置，贪心等价性放 e2e）。

### 6. engine/llm_engine.py — 编排（新逻辑）

- decode 阶段的新 step 流程（prefill 阶段完全不动，还是 ChunkedScheduler 管）：
  ```
  decode_seqs = 从 running 挑 min(max_num_seqs, 预算/(k+1)) 条 decode 序列
  if 启用投机且全部 can_append(k+1):
      draft_runner.propose(decode_seqs, k)
      target_logits = model_runner.run_verify(decode_seqs)
      accepted, bonus = acceptance(...)
      对每条 seq：append accepted + bonus；stop 条件（EOS/max_tokens/max_model_length）
      在接受的 token 里检查，中途停止时丢弃后面的 token
      rollback 未接受部分（block_manager + 草稿 KV）
  else:
      走现有 prepare_decode 路径（兜底，行为与今天完全一致）
  ```
- **MVP 边界**（明确不做，后续阶段再做）：验证批不与 prefill chunk 混批——有等待队列时
  投机阶段让位给 chunk（或先跑 chunk 再投机），P/D 混合与投机的融合（vLLM 的设计）留到
  阶段 4。这样 preempt、混合批的既有语义完全不受影响。
- 配置键：`enable_speculative`（默认 False）、`num_spec_tokens`（k，默认 4）、
  `spec_method`（"draft_model" / "self"）。

## 与现有设施的关系（复用清单）

| 现有设施 | 投机解码中的角色 |
|---|---|
| prepare_prefill 的 chunk 分支 | verify 批的数据准备模板（input/positions/slot_mapping/cu_seqlens 全同构） |
| paged_attention_prefill_kernel | verify 批的注意力内核，零改动 |
| context.positions / qwen3 读显式 positions | spec token 的 RoPE 位置，零改动 |
| BlockManager + 前缀缓存 hash | 追加/回滚记账；hash 只在接受后 finalize |
| ChunkedScheduler + P/D 混合 | prefill 阶段原样保留；阶段 4 再融合 |
| warmup_decode_and_chunked | 需要补 verify 形状（k+1 chunk）的 Triton 预热，否则第一次 verify 付 JIT |

## 风险点

1. **RNG 顺序变化**：接受采样消耗随机数 → 温度采样下与普通 decode 逐 token 不等价
   （数学分布等价，样本不等价）。e2e 正确性必须用**贪心**（temperature 取 1e-9 绕过
   sampler 的 >1e-10 断言）：贪心下投机输出必须与普通 decode 逐 token 一致，这是最强测试。
2. **pickle 契约**：Sequence 新字段漏写 __getstate__/__setstate__ 会在 world_size>1 时
   静默出错（chunked prefill 踩过的坑）。
3. **回滚的块边界**：verify 的 k+1 个槽位跨块时 rollback 可能释放整块；下一轮 propose 前
   要确保草稿 KV 与目标 KV 长度一致（都以"已接受长度"为准）。
4. **预热**：verify 是新的 batch 形状，warmup 要覆盖，否则第一次 verify 吃 Triton JIT
   （现有 torch.compile 形状 churn 问题的同类变体）。
5. **预检失败回退**：块不够 append k+1 时退化为普通 decode，管线不能断。
6. **正确性优先于速度**：接受率低时投机反而慢（多跑了草稿前向）。k 的默认值与草稿
   质量挂钩，Qwen3-0.6B 自投机预期接受率较高，但独立草稿模型才体现真实收益。

## 推荐的实现顺序（分阶段验收）

- **阶段 0（无 GPU）**：sequence/block_manager 的状态与回滚 + acceptance.py 数学 + 单测。
  验收：rollback 后 seq 状态与块记账复原；acceptance 的贪心等价性、边界 case。
- **阶段 1（自投机跑通，正确性）**：draft_runner（同权重）+ prepare_verify/run_verify +
  llm_engine 编排。验收：**贪心 e2e 与普通 decode 逐 token 一致**（多 prompt、含 EOS 提前停、
  chunked prefill 开着、budget 小），加 warmup 后无冷启动差异。
- **阶段 2（加速验证）**：A/B 基准（复用 tests/bench_pd_mixed.py 的负载模式，贪心跑）
  对比 tokens/s；加 LayerSkip 跳层开关（draft 跑奇数层）看加速。
- **阶段 3（换卡后，独立草稿模型）**：加载真正的草稿模型，调 k 与接受率，实测收益。
- **阶段 4（融合）**：verify 批进入 P/D 混合调度（spec token 作为调度器认识的"chunk"），
  与 prefill 同批——这时才是 vLLM 级的完整形态。

## open question（设计时定稿）

1. ~~bonus token 的分布来源~~ **已定稿**：verify 前向输入 = [最后一个已接受 token x] +
   k 个草稿 token（共 k+1 个位置），输出 k+1 行 logits：行 i（i=0..k-1）验证草稿 token i，
   行 k = bonus 分布。带 x 的目的是重算"上一轮 bonus token 没有目标 KV"的槽位（幂等，
   每轮固定多付 1 个位置 ≈ 1/(k+1) 的开销，换来 KV 永远一致、无需 dummy token、无浪费行）。
2. **草稿 KV 池大小**：独立池按什么比例分？MVP：与目标池同参数、各占显存可用量一半；
   换卡后按草稿模型层数比例调。
3. **verify 批内混入普通 decode**（有序列草稿缓冲为空时）：MVP 不混，整批要么全投机
   要么全普通 decode。

## 审阅意见与待定项（2026-09-06）

核心算法判定：正确、自洽，无原理性错误。下述为动手前要钉死的接口约定 + 边界修正。

### 动手前必须定死

1. **acceptance.py 消费哪份 logits（最高优先）**：sampler 路径是 `logits/temperature → softmax`
   （layers/sampler.py 的 Gumbel-Max：`probs.div_(exponential).argmax`）。acceptance 若要"从 logits
   自己算 softmax"，则必须与 sampler 同一条缩放路径——`run_verify` 返回的要么是已按温度缩放的 logits，
   要么 acceptance 接收 `temperature` 参数并复刻 `logits/temp`。否则贪心 e2e 能靠"误拒后补射 argmax
   仍是同一 token"自纠而通过，但温度采样下分布不严格等价。
   额外测试：q≡p（同权重自投机）时接受率统计 ≈1，且长序列输出分布与普通采样统计不可区分。
2. **投机步内被 preempt**：一条序列处于 propose→verify→commit 之间时若被 P/D 调度器 preempt，
   草稿缓冲、已临时 append 的槽位、草稿 KV 都要原子回滚。最简单约束：**投机步内序列不可抢占，
   preempt 只发生在批边界**。
3. **MVP 期间 `enable_speculative` 与 `enable_prefix_cache` 互斥**：verify 的临时 append 会污染
   prefix cache 的 hash 链；部分接受后命中缓存的槽位需专门处理。文档正文"append 照旧 finalize、
   rollback 撤销 hash"只在 prefix cache 关闭时自洽。写入阶段 4 前的硬约束。

### 叙述/精度修正

4. **位置 off-by-one 措辞**（正文"行 i 在位置 t+i"）：数字对（positions = range(t-1, t+k)，k+1 行），
   但读起来像验证行 i 对应输入位置 t+i。实际是验证行 i 对应对目标 token 位置 t+i 的预测、由输入
   位置 t-1+i 的 logits 产生。实现时按"输入位置 vs 目标 token 位置"列表推导，勿按行叙述。
5. **"绕过 sampler 的 >1e-10 断言"不准确**：sampler 的 1e-10 是 exponential divisor 的
   `clamp_min_`（sampler.py），不是 temperature 断言。小温度取贪心的实践建议不变。
6. **新分配槽位是 k 而非 k+1**：verify 里 x（上轮最后接受 token）的槽位上轮已分配，本轮只是覆写 KV；
   真正新分配的是 k 个草稿槽位。`can_append` 预检按 k 计（x 槽位已存在时），勿多检一格。
7. **Chen et al. 的 Exp(1) 全局条件只是等价路径**：本方案第一个拒绝点就停、bonus 直接采目标分布，
   简单 coin-flip（u ≤ p/q；拒绝后采 (p−q)_+ 归一化）已精确。实现时二选一，别混写。

### 工程建议

8. **draft_runner 勿 fork 全套**：自己再写一份 KV 池 + BlockManager + prepare_decode = 两套记账逻辑
   长期漂移。自投机同权重时草稿 KV 与目标 KV 同构，优先抽复用原语、draft 做瘦封装。
9. **验收测试注释误拒自纠**：贪心下即使浮点使 p(d)/q(d) 略 <1 造成误拒，补射 argmax 仍是同一 token，
   输出不变。日后看到偶发多跑一 forward 别误判回归，在测试里注释该性质。
10. **EOS 截断要撤销 hash**：接受序列中发现 EOS、丢弃尾 token 时，被截断槽位已 finalize 的 hash
    要一并撤销（同 #3），草稿 KV 截到同一长度。
11. **阶段 1 就暴露 `draft_skip_layers` 开关**：LayerSkip 跳层（draft 只跑奇数层）接口先留好，
    阶段 2 不用改 draft_runner。
12. **warmup 免费补 verify 形状**：`warmup_decode_and_chunked` 已含 q<k 假前缀路径，verify 只是
    再加一种每序列 (k+1) 的用例。

### 与现有设施的代码对照（审阅时核对）

- prepare_prefill 已处理 `cu_seqlens_q < cu_seqlens_k`（chunk 跟随已算前缀，src/myvllm/engine/
  model_runner.py 236-237、366-372）→ verify 复用成立。
- layers/sampler.py 是 Gumbel-Max（div exponential → argmax）→ acceptance 自带 softmax 逻辑，
  不改 sampler 采样路径（见 #1）。

## 实现记录（2026-09-07，阶段 0+1 完成）

### 验收结果

- 阶段 0 单测 `tests/test_spec_stage0.py`（15 项）+ 调度器回归（38 项）全绿。
- 阶段 1 e2e `tests/test_spec_e2e.py` 全绿：**贪心下投机 vs 普通 decode 逐 token 完全一致**，
  覆盖多 prompt 批、EOS 提前停、chunked prefill + budget=16、跨 KV 块边界（block_size=64 生成
  120 token）。
- A/B（`tests/bench_spec_ab.py`，贪心，2 prompt × 96 token）：spec off 13.6 tok/s → spec on
  15.4 tok/s（+13%，符合"自投机无跳层时近似持平"的预期——k 次草稿 decode + 1 次 verify chunk
  前向 = k+1 次前向产出 k+1 个 token，收益只来自 verify 的 fused chunk 形状；真实加速在阶段
  2/3 的跳层/独立草稿模型）。

### 与设计方案的差异（实现时定稿）

1. **自投机共享权重与 KV 池**（对照正文 §4 的 B 方案）：草稿直接复用目标模型的权重与 KV 池，
   不加载第二份权重、不建草稿 KV 池——6GB 显存放不下 2×权重。成立的理由：同权重下草稿对已接受
   前缀的 KV 与目标逐位相同，草稿只把 k 个 spec 槽位（+x 槽位，见 3）写进目标缓存，verify 在
   自己的 attention 读之前原样覆盖这些槽位，目标侧永远读不到草稿值（跳层下也安全）。
   `spec_decode/draft_runner.py` 是瘦封装（审阅 #8 的"抽复用原语"），无 BlockManager/加载/init。
2. **"逐 token append" 简化为纯预分配**：spec token 不进 `seq.token_ids`（verify 输入取
   `last_token`(=x) + `spec_token_ids`，与设计正文完全一致），块分配在 propose 前一次性预检
   （审阅 #6 按 k 计）并 `block_manager.preallocate()`；块的 finalize/hash 由 commit 后的
   `rollback()` 统一对账（rollback 改为逐块按内容对账：满块用已接受切片重算 hash，非满块复位，
   顺带修复了"L%bs==1 边界下最后接受 token 的块被误释放/满块未 finalize"的隐患）。
3. **草稿 step 0 必须写 x 槽位**（本方案对设计最重要的修正）：上一轮接受的 bonus/replacement
   的 KV 从未被任何前向计算过，若草稿第一步不写该槽位（原设计的"每轮多算 1 个位置"只给了
   verify），草稿会 attend 垃圾 KV → q_0 系统性错误 → 贪心下**每轮都在位置 0 拒绝**（n=0，
   每轮只产出 1 token，实测 3.5 tok/s）。修复：草稿 step 0 照常 store 位置 t-1（store-then-
   attend 使自注意力读到新鲜 KV），verify 随后重存该槽位（幂等，跳层下目标侧不读草稿值）。
   修复后 q≡p、接受率=1（贪心全接受，n=k+bonus），15.4 tok/s。
4. **verify 需要全部 k+1 行 logits**：`ParallelLMHead` 的 prefill 惯例只保留每序列最后一行
   （正常 prefill 只采样最后一个 token）→ `compute_logits(..., slice_last=False)` 显式关掉，
   verify 才拿得到 k 行验证分布 + 1 行 bonus 分布。
5. `draft_skip_layers` 开关已接入（草稿只跑偶数层）：未训练跳层下接受率降（实测 draft/target
   argmax 一致率 0.5），输出仍无损（拒绝采样保证），真实加速需跳层适配后的草稿——阶段 2 的活。
6. `LLMEngine.exit()` 补 `gc.collect() + torch.cuda.empty_cache()`：torch.compile 产物以引用环
   持有模型张量，同进程跑两个引擎（e2e 的 A/B 模式）时第二个引擎会看不到可用显存。

### 其余说明

- world_size>1 路径已按 shm 协议对称实现（propose_step 按 (round,step) 重播种保证各 rank 草稿
  token 一致），但未在多卡上实测。
- 已知遗留：`rms_forward` 的 torch.compile 形状抖动（2D/3D）仍在（记忆里记录过，未动）。
- 前缀缓存互斥约束成立的前提（跨序列复用未启用）保持至今；启用时需先处理 verify 临时 append
  对 hash 链的污染（审阅 #3）。

