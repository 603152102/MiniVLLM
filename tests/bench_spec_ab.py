"""Quick A/B: speculative vs ordinary decode, greedy (stage-2 style smoke).

Self-speculation without layer skip does k draft forwards + 1 verify forward
per k+1 committed tokens -- the same forward count as ordinary decode, so the
expectation is rough parity (the verify is one fused chunk forward instead of
k+1 tiny decodes, but propose adds overhead). The design's real speedup comes
with draft_skip_layers / an independent smaller draft model (stage 2/3).
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from myvllm.engine.llm_engine import LLMEngine
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer

MODEL_PATH = "/home/MiniVllm/MinivLLM/models/Qwen3-0.6B"


def make_config(spec):
    return {
        'max_num_sequences': 4,
        'max_num_batched_tokens': 1024,
        'max_cached_blocks': 512,
        'block_size': 256,
        'world_size': 1,
        'model_name_or_path': MODEL_PATH,
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
        'max_model_length': 256,
        'gpu_memory_utilization': 0.9,
        'eos': 151645,
        'enable_chunked_prefill': True,
        'enable_speculative': spec,
        'num_spec_tokens': 4,
        'spec_method': 'self',
    }


def bench(spec):
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
        for t in [
            "Write a short paragraph about why the sky is blue.",
            "Explain the water cycle in three sentences.",
        ]]
    llm = LLMEngine(config=make_config(spec))
    for p in prompts:
        llm.add_prompt(p, SamplingParams(temperature=1e-9, max_tokens=96, max_model_length=256))
    start = time.time()
    decode_tokens = 0
    while not llm.scheduler.is_finished():
        _, n, is_prefill = llm.step()
        if not is_prefill:
            decode_tokens += n
    elapsed = time.time() - start
    llm.exit()
    return decode_tokens, elapsed


if __name__ == "__main__":
    for spec in (False, True):
        tokens, elapsed = bench(spec)
        print(f"spec={'on ' if spec else 'off'} decode tokens={tokens} "
              f"elapsed={elapsed:.1f}s -> {tokens / elapsed:.1f} tok/s")
