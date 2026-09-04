import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from myvllm.engine.llm_engine import LLMEngine
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer

CHUNKED = "--chunked" in sys.argv
PROFILE = "--profile" in sys.argv
BUDGET = 16 if CHUNKED else 1024

config = {
    'max_num_sequences': 16,
    'max_num_batched_tokens': BUDGET,
    'max_cached_blocks': 1024,
    'block_size': 256,
    'world_size': 1,
    'model_name_or_path': '/home/MiniVllm/MinivLLM/models/Qwen3-0.6B',
    'enforce_eager': True,
    'vocab_size': 151936,
    'hidden_size': 1024,
    'num_heads': 16,
    'head_dim': 128,
    'num_kv_heads': 8,
    'intermediate_size': 3072,
    'num_layers': 28,
    'tie_word_embeddings': True,
    'base': 1000000,
    'rms_norm_epsilon': 1e-6,
    'qkv_bias': False,
    'scale': 1,
    'max_position': 32768,
    'ffn_bias': False,
    'max_num_batch_tokens': 4096,
    'max_model_length': 128,
    'gpu_memory_utilization': 0.9,
    'eos': 151645,
    'enable_chunked_prefill': CHUNKED,
}

# 固定随机种子：只要两条管线的 logits 数值一致、采样次数与顺序一致，
# 采样结果就应该完全相同（Gumbel 噪声来自 torch RNG）
torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'])
llm = LLMEngine(config=config)
content = "list all prime numbers within 100 and explain how to find them step by step. " * 5
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": content}],
    tokenize=False,
    add_generation_prompt=True,
)
sampling_params = SamplingParams(temperature=0.6, max_tokens=48, max_model_length=256)
if PROFILE:
    # 性能分析模式：只 profile 前 8 步（覆盖慢的 prefill/decode 第一步，同时
    # 避免全程记录把内存吃爆——之前全量 generate() 被 OOM kill 过），导出
    # chrome trace，在浏览器 chrome://tracing 里拖入 trace 文件即可看火焰图
    # （X 轴 = 时间，Y 轴 = 调用层级，条宽 = 耗时）
    from torch.profiler import profile, ProfilerActivity
    llm.add_prompt(prompt, sampling_params)
    # NOTE: 不要开 with_stack=True——每个事件都会存一整条 Python 调用栈，
    # 在这台机器上会把 WSL 内存吃爆被 OOM kill（exit 137）
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(8):
            llm.step()
    trace_path = f"trace_{'chunked' if CHUNKED else 'legacy'}.json"
    prof.export_chrome_trace(trace_path)
    print(f"chrome trace exported to: {trace_path}")
    print("open it at chrome://tracing to see the flame graph")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
else:
    out = llm.generate([prompt], sampling_params)
print(f"mode={'CHUNKED' if CHUNKED else 'LEGACY'} budget={BUDGET}")
print("PROMPT TOKENS:", len(tokenizer.encode(prompt)))
print("PROMPT TEXT:", prompt)
if not PROFILE:
    print("OUTPUT TOKENS:", out['token_ids'][0])
    print("OUTPUT TEXT:", out['text'][0])
