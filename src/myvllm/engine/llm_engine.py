import atexit
import torch.distributed as dist
import time
import torch.multiprocessing as mp
from typing import Any

from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.scheduler_chunked import ChunkedScheduler
from myvllm.engine.sequence import Sequence, SequenceStage
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer


def worker_process(config, rank, event):
    """Worker process function that initializes ModelRunner and enters loop."""
    # FIRST print before any other code
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()


class LLMEngine:
    def __init__(self, config: dict):
        self.config = config
        world_size = config.get("world_size", 1)
        ctx = mp.get_context("spawn")
        self.processes = []
        self.events = []
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            self.events.append(event)
            self.processes.append(process)
            process.start()
        # start the engine only on the master thread with rank = 0
        self.model_runner = ModelRunner(config, rank=0, event=self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))
        
        # scheduler needs to init after model_runner: when world_size > 1,
        # ModelRunner.__init__ calls dist.init_process_group() which is a
        # collective barrier — rank-0 blocks until all worker ranks have joined.
        # The scheduler should only be created after that rendezvous completes.
        # When world_size == 1 there is no barrier and no real dependency.
        # NOTE: enable_chunked_prefill is only safe to flip once the model
        # runner chunks its prefill input and the paged prefill kernel lands
        # (see 实现chunked_prefill.md steps 3-5); the scheduler alone is not
        # enough for numerically correct output.
        scheduler_cls = ChunkedScheduler if config.get("enable_chunked_prefill", False) else Scheduler
        scheduler_kwargs = dict(
            max_num_sequences=config.get("max_num_sequences", 16),
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            block_size=config.get("block_size", 256),
            eos=config.get("eos", 50256),
        )
        # P/D mixed scheduling only exists in the chunked scheduler; when
        # disabled it falls back to pure batches (chunks first, decode waits)
        if scheduler_cls is ChunkedScheduler:
            scheduler_kwargs["pd_mixed"] = config.get("enable_pd_mixed", True)
        self.scheduler = scheduler_cls(**scheduler_kwargs)

        atexit.register(self.exit)


    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for process in self.processes:
            process.join()

    # call scheduler to schedule the next batch
    # return scheduled sequences and whether it is for prefilling
    # call model_runner.run() to run the model
    # call postprocessor to process the outputs and update sequences and update block manager
    def step(self) -> tuple[list[tuple[int, list[int]]], int, bool]:
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        num_processed_tokens = 0
        if not scheduled_sequences:
            return [], num_processed_tokens, is_prefill
        # run the model
        outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)
        if outputs is None:
            raise RuntimeError("ModelRunner.run() returned no outputs")
        # Move outputs to CPU and convert them to a list
        outputs = outputs.cpu().tolist()
        # count before postprocess: it zeroes num_prefill_chunk_tokens
        if is_prefill:
            # a chunked sequence counts its chunk, a decode sequence in a
            # mixed batch counts one token, an unchunked (legacy) sequence
            # counts the whole remaining prompt
            num_processed_tokens = sum(
                1 if seq.stage == SequenceStage.DECODE else
                seq.num_prefill_chunk_tokens if seq.num_prefill_chunk_tokens > 0 else len(seq) - seq.num_cached_tokens
                for seq in scheduled_sequences
            )
        else:
            num_processed_tokens = len(scheduled_sequences)
        # postprocess the outputs
        self.scheduler.postprocess(scheduled_sequences, outputs)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]

        return outputs, num_processed_tokens, is_prefill


    # add prompt string to the waiting queue by first transforming it to Sequence object
    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        self.scheduler.add_sequence(Sequence(token_ids=self.tokenizer.encode(prompt), block_size=self.config['block_size'],sampling_params=sampling_params))

    # given a list of prompts
    # add_prompt for each prompt
    # call step until all sequences are finished
    # return the generated texts
    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> dict[str, Any]:
        for prompt in prompts:
            self.add_prompt(prompt, sampling_params)
        generated_tokens = {}
        while not self.scheduler.is_finished():
            start_t = time.time()
            outputs, num_processed_tokens, is_prefill = self.step()
            end_t = time.time()
            running_time = end_t - start_t + 1e-10
            if is_prefill:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during prefilling")
            else:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during decoding")
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})

        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        output = {'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens], 'token_ids': generated_tokens}
        return output
