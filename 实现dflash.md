# DFlash（块扩散投机解码）设计方案

> 依据：[DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)（Chen et al., UCSD z-lab, 2026.02）+ 参考实现 [z-lab/dflash](https://github.com/z-lab/dflash)（`dflash/model.py`，本文所有"参考实现"引用均指该文件）。
> 本文档只做设计，不包含实现。

## 本质

投机解码的草稿从"自回归 k 步"换成"**一次前向并行预测一个块**"：草稿模型是一个**依附于 target 的隐藏态适配器**
（5 层、~1e8 参数、无自己的词表），输入 = target 中间若干层的 hidden states 拼接融合后的"上下文特征"，
一次前向同时预测 [上一个已接受 token（锚点）+ γ-1 个 [MASK]] 块里的全部掩码位置。验证、接受、提交与
普通投机解码完全相同——所以这是对刚实现的投机解码管线的一次**草稿侧替换**，verify/accept/commit 全部复用。

```
每轮 = 3 个阶段：
propose:   草稿一次前向（并行 γ-1 个预测位置，q_i = 每位置 softmax）
verify:    target 一次前向验证 γ 个位置（锚点 + γ-1 个草稿 token）——就是我们投机解码的 verify
accept:    标准拒绝采样（acceptance.py 零改动，K = γ-1）→ 接受 n 个 + 补 1 个 bonus token
commit:    提交/回滚与自投机完全相同
```

为什么比自回归草稿快：自回归草稿的 T_draft = γ × t_step（串行），DFlash 的 T_draft = 1 次并行前向——
草稿成本不再随 γ 增长，因此草稿可以做得更深（5 层 vs EAGLE-3 的 1 层），接受长度显著提高。
论文在 Qwen3-8B 上 4.9×（贪心），是 EAGLE-3 的 2.4×。

### 与"0.6B 草稿 + 大模型 target"的关系（重要概念澄清）

DFlash 的草稿**不是**独立的小模型，二者不是"小模型 + 大模型"的关系：

| | 经典投机解码（独立草稿模型） | DFlash |
|---|---|---|
| 草稿是什么 | 同 tokenizer 同 vocab 的独立小 AR 模型（如 0.6B） | 为 target **一对一训练**的隐藏态适配器（~1e8 参数，比 0.6B 小一个数量级） |
| 草稿的词表 | 自己的（必须与 target 同 vocab） | **没有**——直接复用 target 的 embedding 表查输入、target 的 lm_head 算 logits |
| 草稿的上下文 | 自己逐 token 算 KV | target 的隐藏特征（注入每层 K/V），草稿自己**不建模上下文** |
| 与 target 的关系 | 可替换（换草稿不动 target） | **绑定**：特征维度、hidden 大小、词表全部来自 target，换了 target 就必须重新训练 |

所以：
- **"0.6B 草稿 + 更大的 target"** 不是 DFlash 的用法，而是经典投机解码的"独立草稿模型"路线——本项目的管线已预留
  （`spec_method='draft_model'`，投机解码设计文档的 方案 A / 阶段 3）。该路线不需要训练，但草稿推理是串行的，
  加速上限 ~2-3×。
- **DFlash 的用法是反过来的**：target 想多大都行（4B/8B/30B…），草稿永远是为该 target 训练的 ~1e8 小适配器。
  本地 6GB 的 0.6B 若做 target，草稿就是"为 Qwen3-0.6B 训练的适配器"（~400MB，显存可行）；若想给更大的
  target 加速，草稿是在租卡上为那个大 target 训练的（训练时本地不参与）。
- 无"通用草稿"：草稿 checkpoint 与 target 模型 ID 严格对应。

## 数据流（单轮视角）

序列有 t 个已接受 token（位置 0..t-1），特征池存有这些位置的融合特征（每位置 1×H 向量），块大小 γ（默认 16）：

```
轮次开始：seq.num_tokens = t，特征池长度 = t
1. propose: 块 = [token_{t-1}（锚点，干净）] + [mask_token_id] × (γ-1)（掩码位置 t..t+γ-2）
   noise_embedding = target.embed_tokens(块)          ← 复用 target 词表，草稿无 embedding 表
   草稿前向：Q 来自块隐状态；K/V = cat([k_proj(融合特征 0..t-1), k_proj(块隐状态)])
             （每层注入——特征经 fc+norm 融合后喂给每一层的 k_proj/v_proj）
   输出 = 掩码位置 t..t+γ-2 的隐状态 → logits = target.lm_head(隐状态)   ← 复用 target 的 lm_head
   q_i = softmax(logits_i / temperature)，采样 d_0..d_{γ-2}
2. verify: target 一次前向，输入 = [锚点] + [d_0..d_{γ-2}]（γ 个位置 t-1..t+γ-2）
   = 现有 run_verify(seqs, k=γ-1) 的形状，零改动；同时捕获选定 5 个 target 层的 per-position hidden
   logits 行 0..γ-2 验证草稿 token 0..γ-2，行 γ-1 = bonus 分布
3. accept: acceptance.py（K = γ-1）→ 接受 n 个草稿 + 1 个 bonus（n=γ-1 时）或 (p_n-q_n)_+ 替换采样
4. commit: append n 个草稿 + bonus（共 n+1 个）；块记账回滚（与自投机相同）；
   特征池截断到 t+n+1，并把 verify 捕获的 hidden 里"已提交位置"对应的融合特征 append 进特征池
```

与自投机的差异只有两处：(a) propose 从 k 次串行前向变成 1 次并行前向，且草稿不写目标 KV 池
（草稿有自己的特征 KV，见下）；(b) verify 前向多输出一份"选定层 hidden"用于特征提取。其余不变。

## 算法细节

### 草稿模型（`spec_decode/dflash_draft.py`，新）

参考 `dflash/model.py` 的 `Qwen3DFlashDraftModel` / `Qwen3DFlashDecoderLayer` / `Qwen3DFlashAttention`：

- 配置：`num_hidden_layers=5`（Coder 系 8）、hidden/heads/kv_heads/head_dim 与 target 相同（Qwen3 结构
  norms/MLP 同 Qwen3）；全部超参由草稿 checkpoint 的 config.json 携带。
- 参数：每层 {q_proj, k_proj, v_proj, o_proj, input_layernorm, post_attention_layernorm, mlp}，
  顶层 {fc: (num_target_layers×H)→H 无偏置, hidden_norm, norm, rotary_emb}。
  **没有** embed_tokens、**没有** lm_head（两者复用 target 的）。
- 参数量级：0.6B 的 5/28 + fc ≈ 1.1e8 参数 ≈ 430MB fp32——6GB 卡放得下。
- 前向（每层 attention）：
  ```
  q   = q_norm(q_proj(块隐状态))                       # 块内 γ 个位置
  k   = k_norm(cat([k_proj(fused_features), k_proj(块隐状态)]))   # ctx 位置 + 块位置
  v   =       cat([v_proj(fused_features), v_proj(块隐状态)])
  q,k = RoPE(positions)                                # 绝对位置（ctx: 0..t-1，块: t..t+γ-1）
  o   = attention(q, k, v, mask=causal)                # 推理掩码：因果（参考实现为 causal + 可选 sliding window）
  ```
- 草稿 KV：ctx 部分（来自特征投影）跨轮持久；块部分（noise KV）每轮重算。参考实现把两者合并存进草稿
  DynamicCache 并在每轮结束后 crop 到已提交长度；本项目用自己的特征池实现（见"特征池"）。
- logits：`target.lm_head(draft_hidden)`——草稿预测直接用 target 的 lm_head，词表天然一致。

### 特征提取与融合

- 采样层：`build_target_layer_ids(num_target_layers, num_draft_layers)`：start=1, end=num_layers-3 均匀取
  round 值（0-based 层号）。Qwen3-0.6B（28 层）→ **{1, 7, 13, 19, 25}**（跳过第 0 层与最后 3 层）。
  该集合训练时冻结并写进 checkpoint config（`target_layer_ids`），推理必须逐层一致。
- 融合：`fused = hidden_norm(fc(concat(hidden[L_i] for L_i in target_layer_ids)))`，每位置 1×H。
- **取值点语义**：训练时用的是 HF `output_hidden_states` 的语义（hidden_states[i] = 第 i-1 层的
  post-residual 输出，hidden_states[0] = embedding 输出）。MinivLLM 的 residual 流里对应"层输出 x 与
  residual 相加后的值"，实现时必须对齐，否则训练好的 checkpoint 特征分布不匹配、接受率崩塌。
- 提取时机：prefill 前向（prompt 全部位置）+ 每轮 verify 前向（γ 个块位置）。verify 后只保留已提交
  行（acceptance 结果已知的那部分）。

### 特征池（ModelRunner 内，与 KV 池同居）

- MVP：存**融合特征**（每位置 1×H，fp32 4KB/位置；2048 位置/序列 ≈ 8MB），每序列一张连续 GPU 张量，
  ModelRunner 按 seq 管理，轮开始时按 `len(seq)` 截断、轮内按提交数 append。草稿前向每轮对 ctx 现场投影
  k_ctx/v_ctx（5 层 × 2 个 matmul，对长上下文有额外开销）。
- 优化项（阶段 3 再评估）：直接存每层 k_ctx/v_ctx（参考实现做法，"projected features are stored in the
  draft model's KV cache and reused"）——省掉每轮投影，代价是多一个分页特征 KV 池 + 一个双源注意力内核。
- TP（阶段 4）：本代码库 row-parallel 输出经 all_reduce 后 hidden 是全副本，各 rank 提取的特征相同、
  各持一份，propose 在各 rank 本地执行即可，无跨 rank 通信。

### 接受准则

与自投机完全相同（参考实现的 `_rejection_sample` 与 acceptance.py 逐行对应：`rand*q < p` 硬币 ⇔ u ≤ p/q，
拒绝后 (p−q)_+ 归一化替换；贪心 = argmax 逐位比较，误拒自愈性质一致）。**acceptance.py 零改动**，
调用时 K = γ-1。注意：参考实现 temperature>0 且 top_p/top_k < 1 时 q 是截断分布（draft_indices 机制）——
MVP 不做 top_p/top_k（我们的 sampler 也没有），q = 全词表 softmax(logits/temperature)，温度采样等价性
只在此前提下成立（与投机解码的契约相同）。

## 需要改动的文件

### 1. spec_decode/dflash_draft.py — 草稿模型（新文件）

- 用现有原语组装：`layers/layernorm.py`（RMSNorm 语义）、`layers/activation.py`（SiluAndMul）、
  `layers/rotary_embedding.py`（RotaryEmbedding）；草稿注意力 MVP 用 `F.scaled_dot_product_attention`
  + 显式 causal mask（草稿小，不写 Triton 内核；阶段 3 再评估）。
- 权重加载：从 HF safetensors checkpoint 映射参数名（结构是参考实现的 HF 模型，名字基本直映）。
- 边界：只实现 DFlash v1（论文版）；v2 的 GroupedDynamicCausalConv/CandidateSelector 不做。

### 2. models/qwen3.py — 特征提取钩子

- 新增 `forward_with_features(input_ids, layer_ids) -> (hidden_states, features)`：在层循环里捕获选定层的
  post-residual 输出（语义对齐见上）。prefill 与 verify 两条路径共用。

### 3. engine/model_runner.py — run_dflash_round + 特征池

- 新增 `run_dflash_round(seqs, γ) -> (num_accepted, extra)`：propose（一次草稿前向 + 采样 + 记录 q_i）→
  verify（复用 prepare_verify/run_verify 的 k=γ-1 路径 + 特征提取）→ accept（内部调 acceptance.py）→
  返回每序列 n 与 bonus。特征池增删在 runner 内完成（下一轮按 len(seq) 截断，天然与引擎提交对齐）。
- 备选（不推荐）：把 logits 拿回引擎侧做 acceptance（与自投机路径对称）——代价是 γ-1 行草稿 logits +
  γ 行目标 logits + 特征都要过进程边界，且 runner 无法知道 n、特征截断要等下一轮引擎回传。
- prefill 路径：`run()` 的 prefill 分支在 chunk 完成整个 prompt 时提取特征入池（chunked prefill 的中间
  chunk 不提取——特征只在 prompt 全部计算后一次性提取，或每 chunk 提取本 chunk 段，实现时定）。
- warmup：补 γ 宽度的 verify 形状（warmup_decode_and_chunked 已按 k 预热，γ≠4 时补一组）+
  草稿前向形状（SDPA 无 JIT 问题，但 torch.compile 边界要过一遍）。

### 4. engine/llm_engine.py — 编排分发（小改）

- `_run_spec_step` 的 propose 段按 `spec_method` 分发：`'self'` 走现有 propose 循环；`'dflash'` 调
  `run_dflash_round`。预检/预分配/commit/rollback/EOS 截断全部复用（k → γ-1 的预检公式不变：
  `(t+i) % bs == 1`，i=1..γ-1——verify 需要 γ 个位置 t-1..t+γ-2 的槽位，其中 t-1 是 x 槽位已分配）。
- 配置键：`spec_method='dflash'`、`dflash_block_size=16`（γ）、`dflash_draft_path`；
  `target_layer_ids`/`mask_token_id` 从草稿 checkpoint config 读。

### 5. 训练（独立于推理引擎，阶段 2 在租卡上做）

- `train/train_dflash.py` + 数据管线：
  - 语料：目标模型**自己生成**的响应（对齐训练，论文 ~800K 样本 Nemotron PTV2 + CodeAlpaca；研究用途
    可缩到 1-5 万条）。
  - 每序列随机采样 512 个锚点、每锚点后掩码一个块；跨块注意力禁止、块内双向（训练掩码 ≠ 推理的 causal——
    以参考实现的 Flex Attention 掩码为准）；早位加权 loss：w_k = exp(−(k−1)/γ_decay)，γ_decay=7（块 16）。
  - 目标冻结，在线提取特征（或离线预存特征缓存）；AdamW lr 6e-4、clip 1.0、cosine + 0.04 warmup、6 epochs、
    最大序列 3072。
  - 训练图：目标前向（冻结）+ 草稿前向/反向 ≈ 1.5-2GB 权重/激活 + Adam 状态 ~1.5GB——6GB 本地
    batch=1 也极慢，**租卡**（与投机解码阶段 3 的"换卡"同一前提）。

## 与现有设施的关系（复用清单）

| 现有设施 | DFlash 中的角色 |
|---|---|
| prepare_verify / run_verify | verify 前向**零改动**（k = γ-1：输入 [x]+γ-1 草稿 = [锚点+草稿块]，行 0..γ-2 验证、行 γ-1 = bonus） |
| paged_attention_prefill_kernel | verify 的注意力内核，零改动 |
| spec_decode/acceptance.py | **零改动**（K = γ-1；q 来自草稿每位置的 softmax） |
| block_manager preallocate / rollback | 预检、预分配、回滚对账**零改动**（预检公式 k→γ-1） |
| llm_engine _run_spec_step | propose 分支替换为一次调用；commit/EOS/停止条件/回退逻辑复用 |
| 贪心 e2e 等价测试方法 | 同款：投机 vs 普通 decode 逐 token 一致（**与草稿质量无关**——拒绝采样无损，随机权重草稿也通过） |
| target 的 embed_tokens / lm_head | 草稿的输入嵌入与输出头（草稿无自己的词表） |
| RotaryEmbedding / LayerNorm / SiluAndMul | 草稿层组装原语 |

## 风险点

1. **没有 Qwen3-0.6B 的 DFlash checkpoint**：z-lab 只发布了 Qwen3 4B/8B/30B 等大模型的草稿（最小的公开
   checkpoint 是 Llama-3.1-8B / Qwen3-4B 档）。0.6B 的草稿**必须自己训练**——训练是阶段 2 的硬前置，
   且训练质量直接决定加速比（无特征条件的纯扩散草稿只有 ~2.8×，论文表 8）。
2. **特征提取语义必须与训练时逐位对齐**：HF `hidden_states` 是 post-residual 层输出，MinivLLM 的
   residual 流取值点若偏一档，训练好的 checkpoint 会特征失配、接受率崩塌（贪心 e2e 验不出来——它只验
   无损，不验接受率）。
3. **训练/推理掩码不一致是参考实现的既有设定**（训练块内双向、推理因果）——照抄参考实现即可，不要
   "修正"它；训练脚本的稀疏掩码也要逐位复刻。
4. **接受率无法用本机验证**：贪心 e2e 只保证正确性；加速收益依赖阶段 2 的训练质量，本地测不到。
5. 参考实现基于 HF transformers + DynamicCache + SDPA，本项目是自研栈——等价实现有偏差风险，尤其是
   RoPE 应用方式、mask 构造、KV crop 语义（照参考实现语义重写，不照搬代码结构）。
6. 显存测算（fp32, 6GB）：target 2.4GB + 草稿 ~0.43GB + 特征池（2K 位置 ≈ 8MB/序列）+ 目标 KV 池
   现有 ~2GB ≈ 4.9GB——可行但紧；γ=16 时 verify 宽度是自投机的 3 倍，paged prefill 内核需确认
   BLOCK_M 网格与共享内存余量。
7. 投机解码的风险清单（RNG 顺序、pickle 契约、回滚边界、预热、预检回退、EOS 截断）全部继承——
   大部分已在自投机实现中解决，DFlash 不引入新的同类问题（特征池的截断对齐是唯一新增状态）。

## 推荐的实现顺序（分阶段验收）

- **阶段 0（无 GPU）**：dflash_draft 模型结构 + 权重名映射 + 特征提取钩子 + 单测（fc 形状、层集合公式、
  causal mask 构造、特征池截断不变量）。
- **阶段 1（随机权重草稿跑通管线）**：run_dflash_round + 编排分发 + warmup。验收：**贪心 e2e 与普通
  decode 逐 token 一致**（多 prompt、EOS 提前停、chunked prefill 开、小 budget、跨块边界——同自投机的
  测试矩阵）。注意此阶段接受率接近 0、无加速，只验正确性（拒绝采样无损，验收与草稿质量无关）。
- **阶段 2（租卡训练草稿）**：train_dflash.py + 对齐语料管线 → 产出 Qwen3-0.6B 对齐草稿 checkpoint。
  验收：训练集/验证集上的接受长度 τ 与 loss 曲线；有条件的对拍参考实现的 SGLang/HF 后端。
- **阶段 3（本地实测）**：A/B 基准（tests/bench_spec_ab.py 的负载模式）：DFlash vs 普通 decode vs
  自投机；调 γ、接受率；评估特征池优化项（存每层 k_ctx/v_ctx）。
- **阶段 4（融合）**：TP 支持、特征池分页、verify 批进入 P/D 混合调度（同投机解码阶段 4 的路线）。

## open question（设计时定稿）

1. **草稿训练计划（用户决策）**：租卡预算与语料规模；是否先做 1-2 万条小语料验证训练管线再放大。
   训练完成后 `target_layer_ids={1,7,13,19,25}`（28 层时）与 `mask_token_id` 冻结进 checkpoint config。
2. **γ 的默认值**：论文 16（verify 宽度 16）；本地 6GB 上 γ 大 → verify 批更大但轮数更少——按实测调，
   config 键已留。
3. **特征提取时机**（prefill 一次性 vs 每 chunk 增量）：与 chunked prefill 的交互，实现时定（不影响接口）。
4. **是否做 DFlash 2**（GroupedDynamicCausalConv / CandidateSelector）：MVP 明确不做，checkpoint 按 v1 训练。
5. **温度采样下 q 的 top_p/top_k 截断（draft_indices）**：MVP 不做，若日后要支持，acceptance.py 需扩展
   索引化 q（参考实现的 `_rejection_sample` 已示范）。

## 参考实现要点备忘（实现时对照）

- `build_target_layer_ids`：1 层草稿特例 = 中间层；否则 start=1、end=num_layers-3、均匀 round。
- `extract_context_feature`：concat 选定层 hidden（HF 语义下 hidden_states 索引 = 层号 + 1）。
- 推理轮：块 = `[output_ids[start]]`（锚点，干净）+ γ-1 个 mask；`noise_embedding = target 的
  embed_tokens(块) × input_embedding_scale`；草稿输出取后 γ-1 行；verify 后
  `target_hidden` 只保留已提交行；草稿 KV 与目标 KV 均 crop 到已提交长度。
- 贪心接受 = `(草稿 == argmax(verify 行)).cumprod()`；温度接受 = `rand*q < p` 硬币 + (p−q)+ 替换
  （与 acceptance.py 等价）。
- 训练：锚点随机采样（每序列 512 个）、跨块禁连、块内双向、w_k = exp(−(k−1)/7)、目标模型生成的对齐
  语料、在线或离线特征两种模式。
