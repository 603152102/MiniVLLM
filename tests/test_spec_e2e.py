"""Speculative decoding e2e: greedy equivalence with ordinary decode.

The strongest correctness property from the design (风险 #1): under greedy
sampling (temperature=1e-9, which still passes the sampler's >1e-10 assert)
the speculative pipeline must emit EXACTLY the same tokens as the ordinary
decode path -- rejection sampling is lossless, and a spurious float rejection
self-heals to the target's argmax (审阅 #9: an occasional extra forward is not
a regression, the emitted tokens are unchanged).

Covers: multiple prompts in one engine (spec batches of B > 1), EOS early
stop, chunked prefill + small budget (spec only on pure decode batches), and
generation crossing a KV block boundary (exercises preallocate + rollback on
the boundary). Statistical unbiasedness under temperature sampling (审阅 #1
的额外测试) is deferred to stage 2 per the design's phased acceptance plan.

Needs the GPU; run inside WSL with the project venv.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from myvllm.engine.llm_engine import LLMEngine
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer

MODEL_PATH = "/home/MiniVllm/MinivLLM/models/Qwen3-0.6B"


def make_config(**overrides):
    config = {
        'max_num_sequences': 8,
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
        'enable_chunked_prefill': False,
        'enable_speculative': False,
        'num_spec_tokens': 4,
        'spec_method': 'self',
    }
    config.update(overrides)
    return config


def chat_prompt(tokenizer, text):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


PROMPTS = [
    "The capital of France is",
    "List the first five prime numbers:",
    "Say just the word yes.",
    "What is 2 + 2? Answer in one sentence.",
]


def run_both(prompts, sampling_params, **config_overrides):
    """Run the same prompts twice (spec off / on) and return both token lists."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    results = []
    for spec in (False, True):
        torch.manual_seed(0)
        llm = LLMEngine(config=make_config(enable_speculative=spec, **config_overrides))
        out = llm.generate([chat_prompt(tokenizer, p) for p in prompts], sampling_params)
        llm.exit()  # free the GPU before the next engine loads
        results.append(out['token_ids'])
    return results


def assert_equal(off, on):
    assert len(off) == len(on)
    for i, (a, b) in enumerate(zip(off, on)):
        assert a == b, f"prompt {i}: spec off {a} vs spec on {b}"


def test_greedy_equivalence_multi_prompt():
    sp = SamplingParams(temperature=1e-9, max_tokens=40, max_model_length=256)
    off, on = run_both(PROMPTS, sp)
    assert_equal(off, on)


def test_greedy_equivalence_chunked_small_budget():
    # chunked prefill + tiny batch budget: prompts are prefilled in chunks and
    # decode batches are small; spec runs on pure decode batches, mixed
    # batches fall back to the ordinary path
    sp = SamplingParams(temperature=1e-9, max_tokens=48, max_model_length=256)
    off, on = run_both(
        PROMPTS, sp,
        enable_chunked_prefill=True,
        max_num_batched_tokens=16,
        max_num_sequences=4,
    )
    assert_equal(off, on)


def test_greedy_equivalence_across_block_boundary():
    # generation longer than one KV block: every spec round crosses the
    # preallocate/rollback block-boundary logic (120 tokens > 64-token block)
    sp = SamplingParams(temperature=1e-9, max_tokens=120, max_model_length=192)
    off, on = run_both(
        ["Tell me a short story about a cat and a dog."], sp,
        block_size=64,
    )
    assert_equal(off, on)


if __name__ == "__main__":
    torch.manual_seed(0)
    test_greedy_equivalence_multi_prompt()
    print("multi-prompt: OK")
    test_greedy_equivalence_chunked_small_budget()
    print("chunked small budget: OK")
    test_greedy_equivalence_across_block_boundary()
    print("block boundary: OK")
