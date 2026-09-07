"""A/B benchmark: P/D mixed scheduling ON vs OFF.

ON  = enable_pd_mixed=True (decode-first + chunks fill remaining budget,
      one batch may contain both decode tokens and prefill chunks).
OFF = enable_pd_mixed=False (prefill chunks first; if any chunk is
      scheduled the decode loop is skipped entirely, batches never mix).

Usage (from the repo root):
    .venv/bin/python tests/bench_pd_mixed.py                  # ON
    .venv/bin/python tests/bench_pd_mixed.py --no-pd-mixed    # OFF

Two workloads, both timed with time.perf_counter around an explicit
step() loop (the generate() helper prints per-step and would flood stdout):

1. control: pure-decode workload (short prompts, no prompt arrives while
   other sequences decode). Both modes should behave identically here --
   this guards against measuring a regression that has nothing to do
   with mixing.
2. concurrent: 1 long decode sequence + 8 short prompts all added at
   t=0. OFF stalls the long sequence's decode for every chunk step
   (~56 steps with budget=16); ON keeps decoding 1 token/step while
   chunks ride along in the same batch.

Reported metrics:
- wall: total wall-clock seconds of the step loop
- steps: number of engine steps (forward passes)
- tokens: total tokens processed (sum of num_processed_tokens)
- tok/s: throughput
- long-seq first-decode step: first step where the long sequence got a
  completion token (decode latency under load)
- long-seq stalls: steps where the long sequence was unfinished but
  made no decode progress (0 for a healthy mixed scheduler)

NOTE: sampling with temperature > 0 means ON/OFF may generate different
token ids (the decode sequences are sampled in different orders), so
outputs are compared for plausibility only, not equality.
"""
import argparse
import os
import sys
import time

import torch


def make_config(pd_mixed=True):
    return {
        'max_num_sequences': 16,
        'max_num_batched_tokens': 16,   # small budget -> chunked prefill in action
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
        'enable_chunked_prefill': True,
        'enable_pd_mixed': pd_mixed,
    }


def run_workload(llm, prompt_specs, watch_seq: int | None = None, tokenizer=None):
    """Add all prompts, step until done, return metrics dict.

    prompt_specs: list of (prompt, SamplingParams).
    watch_seq: index (in insertion order) of the sequence whose decode
    progress is tracked for latency/stall metrics.
    """
    for prompt, params in prompt_specs:
        llm.add_prompt(prompt, params)
    watched = list(llm.scheduler.waiting)[watch_seq] if watch_seq is not None else None

    wall_start = time.perf_counter()
    steps = 0
    tokens = 0
    first_decode_step = None
    stalls = 0          # decode-starved steps after the first completion token
    max_stall_run = 0
    stall_run = 0
    watch_done = False
    saw_first_token = False

    while not llm.scheduler.is_finished():
        prev_ct = watched.num_completion_tokens if watched is not None else 0
        _, num_processed_tokens, _ = llm.step()
        steps += 1
        tokens += num_processed_tokens
        if watched is not None and not watch_done:
            if watched.status.name == 'FINISHED':  # str compare: robust across --src imports
                watch_done = True
            else:
                delta = watched.num_completion_tokens - prev_ct
                if delta > 0:
                    if not saw_first_token:
                        saw_first_token = True
                        first_decode_step = steps
                    stall_run = 0
                elif saw_first_token:
                    # the watched sequence is decode-ready but made no
                    # progress this step: decode starvation. (Steps before
                    # the first token are its own prompt being prefilled
                    # in chunks, which is normal.)
                    stalls += 1
                    stall_run += 1
                    max_stall_run = max(max_stall_run, stall_run)
    wall = time.perf_counter() - wall_start

    metrics = {
        'wall': wall,
        'steps': steps,
        'tokens': tokens,
        'tok/s': tokens / wall,
    }
    if watched is not None:
        metrics['first_decode_step'] = first_decode_step
        metrics['stalls'] = stalls
        metrics['max_stall_run'] = max_stall_run
        metrics['watched_tokens'] = len(watched.completion_token_ids)
        metrics['watched_text'] = (
            tokenizer.decode(watched.completion_token_ids) if tokenizer is not None else None
        )
    return metrics


def print_report(name, m):
    print(f"[{name}]")
    print(f"  wall    : {m['wall']:8.2f} s")
    print(f"  steps   : {m['steps']}")
    print(f"  tokens  : {m['tokens']}")
    print(f"  tok/s   : {m['tok/s']:8.1f}")
    if 'first_decode_step' in m:
        print(f"  long-seq first decode at step {m['first_decode_step']}")
        print(f"  long-seq decode-starved steps: {m['stalls']} (max run {m['max_stall_run']})")
        print(f"  long-seq completion tokens: {m['watched_tokens']}")
        if m['watched_text']:
            print(f"  long-seq text: {m['watched_text'][:80]!r}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default=os.path.join(os.path.dirname(__file__), '..', 'src'),
                        help='myvllm source directory to import (default: this repo)')
    parser.add_argument('--no-pd-mixed', action='store_true',
                        help='disable P/D mixed scheduling (pure batches)')
    args = parser.parse_args()
    # insert before importing myvllm so --src selects the code version
    sys.path.insert(0, os.path.abspath(args.src))
    from myvllm.engine.llm_engine import LLMEngine
    from myvllm.sampling_parameters import SamplingParams
    from transformers import AutoTokenizer

    torch.manual_seed(42)
    config = make_config(pd_mixed=not args.no_pd_mixed)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'])

    def make_prompt(content):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
        )

    # tiny untimed run to absorb first-call overheads / compile churn
    llm = LLMEngine(config=config)
    llm.add_prompt(make_prompt("hi"), SamplingParams(temperature=0.6, max_tokens=8, max_model_length=128))
    while not llm.scheduler.is_finished():
        llm.step()
    print(f"mode: src={os.path.abspath(args.src)} pd_mixed={not args.no_pd_mixed}")

    # 1. control: pure-decode workload, no mixing opportunity after the
    #    initial prefills -- ON and OFF must behave identically here
    control_specs = [
        (make_prompt("count from one to twenty."), SamplingParams(temperature=0.6, max_tokens=48, ignore_eos=True, max_model_length=128))
        for _ in range(6)
    ]
    print_report('control (6 short prompts, decode-only)', run_workload(llm, control_specs))

    # 2. concurrent: 1 long decode sequence + 8 short prompts at t=0
    long_content = ("list all prime numbers within 100 and explain how to find them step by step. " * 1)
    short_content = ("the capital of France is " + "paris, a city known for its history. " * 6)
    long_spec = [(make_prompt(long_content),
                  SamplingParams(temperature=0.6, max_tokens=96, ignore_eos=True, max_model_length=256))]
    short_specs = [(make_prompt(short_content),
                    SamplingParams(temperature=0.6, max_tokens=24, ignore_eos=True, max_model_length=128))
                   for _ in range(8)]
    m = run_workload(llm, long_spec + short_specs, watch_seq=0, tokenizer=tokenizer)
    print_report('concurrent (1 long + 8 short)', m)
    # no explicit llm.exit(): it double-fires with the atexit hook and raises
    # AttributeError noise; the interpreter exit path handles cleanup


if __name__ == '__main__':
    main()
