import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.scheduler_chunked import ChunkedScheduler
from myvllm.engine.sequence import Sequence

PROMPT_LEN = 2000   # prompt 比 batch 预算大得多
BUDGET = 1024       # max_num_batched_tokens
BLOCK_SIZE = 256
MAX_CACHED_BLOCKS = 64
MAX_STEPS = 15


def run(scheduler_cls, name):
    scheduler = scheduler_cls(
        max_num_sequences=16,
        max_num_batched_tokens=BUDGET,
        max_cached_blocks=MAX_CACHED_BLOCKS,
        block_size=BLOCK_SIZE,
        eos=0,
    )
    scheduler.add_sequence(Sequence(list(range(PROMPT_LEN)), block_size=BLOCK_SIZE))
    print(f"=== {name} (prompt={PROMPT_LEN} tokens, budget={BUDGET}) ===")
    step = 0
    try:
        while not scheduler.is_finished():
            scheduled, is_prefill = scheduler.schedule()
            if is_prefill:
                tokens = sum(s.num_prefill_chunk_tokens for s in scheduled)
            else:
                tokens = len(scheduled)
            scheduler.postprocess(scheduled, [1] * len(scheduled))
            step += 1
            print(f"  step {step:2d}: {'prefill' if is_prefill else 'decode '}, {tokens:4d} tokens")
            if step >= MAX_STEPS:
                print("  ... (early stop)")
                break
    except RuntimeError as e:
        print(f"  RuntimeError: {str(e).splitlines()[0]}")
    print()


run(Scheduler, "legacy Scheduler (no chunked prefill)")
run(ChunkedScheduler, "ChunkedScheduler (chunked prefill)")
