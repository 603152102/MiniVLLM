import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from myvllm.engine.llm_engine import LLMEngine
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer

CHUNKED = "--chunked" in sys.argv
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
out = llm.generate([prompt], SamplingParams(temperature=0.6, max_tokens=48, max_model_length=256))
print(f"mode={'CHUNKED' if CHUNKED else 'LEGACY'} budget={BUDGET}")
print("PROMPT TOKENS:", len(tokenizer.encode(prompt)))
print("PROMPT TEXT:", prompt)
print("OUTPUT TOKENS:", out['token_ids'][0])
print("OUTPUT TEXT:", out['text'][0])
