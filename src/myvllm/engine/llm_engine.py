import atexit
import gc
import torch
import torch.distributed as dist
import time
import torch.multiprocessing as mp
from typing import Any

from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.scheduler_chunked import ChunkedScheduler
from myvllm.engine.sequence import Sequence, SequenceStage, SequenceStatus
from myvllm.sampling_parameters import SamplingParams
from myvllm.spec_decode.acceptance import rejection_accept
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

        # Speculative decoding (MVP boundaries, see 实现speculative_decoding.md):
        # self-speculation only -- spec_method='draft_model' (independent draft
        # with its own weights/KV pool) is reserved for stage 3. The spec path
        # only ever runs on pure decode batches (never mixed with prefill
        # chunks), and preemption cannot fire mid-round because schedule() is
        # not re-entered inside a spec step. Cross-sequence prefix-cache reuse
        # is not enabled anywhere in the engine yet, so the design review's
        # mutual-exclusion rule (spec x prefix cache) holds trivially.
        self._spec_enabled = config.get('enable_speculative', False)
        if self._spec_enabled:
            self.num_spec_tokens = config.get('num_spec_tokens', 4)
            assert self.num_spec_tokens >= 1, "num_spec_tokens must be >= 1"
            assert config.get('spec_method', 'self') == 'self', (
                "spec_method='draft_model' is reserved for stage 3 "
                "(independent draft model); only 'self' is implemented"
            )
        # per-engine spec round counter: seeds the draft RNG deterministically
        # across ranks (propose_step is mirrored by TP workers via the shm loop)
        self._spec_round_id = 0

        atexit.register(self.exit)


    def exit(self):
        # idempotent: also safe when the engine was already exited manually
        # (tests run several engines back to back in one process)
        if getattr(self, 'model_runner', None) is not None:
            self.model_runner.call("exit")
            del self.model_runner
        for process in self.processes:
            process.join()
        # Release the model/KV memory so a second engine in the same process
        # sees real free memory: torch.compile artifacts keep the model's
        # tensors alive in reference cycles that only gc can break, and the
        # caching allocator keeps freed segments unless empty_cache returns
        # them to the driver.
        gc.collect()
        torch.cuda.empty_cache()

    # call scheduler to schedule the next batch
    # return scheduled sequences and whether it is for prefilling
    # call model_runner.run() to run the model
    # call postprocessor to process the outputs and update sequences and update block manager
    def step(self) -> tuple[list[tuple[int, list[int]]], int, bool]:
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        if not scheduled_sequences:
            return [], 0, is_prefill
        # a pure decode batch may take the speculative path; any batch with a
        # prefill chunk (or with spec disabled) takes the ordinary path --
        # design MVP rule: verify never mixes with prefill chunks
        if is_prefill or not self._spec_enabled:
            outputs, num_processed_tokens = self._run_normal_step(scheduled_sequences, is_prefill)
        else:
            outputs, num_processed_tokens = self._run_spec_step(scheduled_sequences)
        return outputs, num_processed_tokens, is_prefill

    # the ordinary path: run the batch through the model runner, postprocess
    def _run_normal_step(self, seqs: list[Sequence], is_prefill: bool) -> tuple[list[tuple[int, list[int]]], int]:
        # run the model
        outputs = self.model_runner.call("run", seqs, is_prefill)
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
                for seq in seqs
            )
        else:
            num_processed_tokens = len(seqs)
        # postprocess the outputs
        self.scheduler.postprocess(seqs, outputs)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        return outputs, num_processed_tokens

    # One speculative decode step over a pure decode batch:
    #   propose (k draft steps) -> verify (one target forward) -> accept/commit
    # The scheduler already ran the round-start append (position t-1's block,
    # whose KV the verify pass recomputes idempotently). Here we reserve the k
    # provisional spec slots up front -- the design's "逐 token append" reduced
    # to pure block allocation (spec tokens never enter token_ids, so
    # seq.last_token stays x for the verify input and rollback has nothing to
    # truncate; block finalize hashes are reconciled by rollback after commit).
    # Draft KV writes land in the target cache slots the verify pass
    # overwrites before anything else reads them (see draft_runner.py).
    # NOTE: the scheduler counts 1 token per decode sequence, so a spec batch
    # can exceed max_num_batched_tokens // (k+1); MVP accepts the scheduler's
    # batch as-is rather than trimming (trimming after schedule would double
    # the round-start append on the next step).
    def _run_spec_step(self, seqs: list[Sequence]) -> tuple[list[tuple[int, list[int]]], int]:
        k = self.num_spec_tokens
        block_manager = self.scheduler.block_manager
        block_size = block_manager.block_size

        # precheck (审阅 #6): k new slots per sequence, counted before propose;
        # the x slot (position t-1) already exists from the round-start append
        needed = 0
        for seq in seqs:
            t = len(seq)
            needed += sum(1 for i in range(1, k + 1) if (t + i) % block_size == 1)
        if needed > len(block_manager.free_block_ids):
            # not enough room for the provisional spec slots: degrade to the
            # ordinary decode path for the whole batch (MVP rule: a batch is
            # either fully speculative or fully ordinary)
            return self._run_normal_step(seqs, False)

        # reserve the physical blocks for the k spec slots
        for seq in seqs:
            t = len(seq)
            for i in range(1, k + 1):
                if (t + i) % block_size == 1:
                    block_manager.preallocate(seq)

        # propose: k draft steps, one forward each, sampling with the
        # sequence's own temperature; logits recorded pre-scaled (acceptance's
        # contract). round_id reseeds the draft RNG per step so TP workers
        # (which mirror propose_step through the shm loop) draw the same
        # draft tokens as rank 0 -- their tokens feed the next step's input.
        draft_logits = []
        round_id = self._spec_round_id
        self._spec_round_id += 1
        for i in range(k):
            logits_i, tokens_i = self.model_runner.call("propose_step", seqs, i, k, round_id)
            draft_logits.append(logits_i)
            for seq, tok in zip(seqs, tokens_i.cpu().tolist()):
                seq.spec_token_ids.append(tok)

        # verify: one target forward over [x] + k drafts per sequence
        target_logits = self.model_runner.call("run_verify", seqs, k)
        for seq in seqs:
            seq.num_spec_verified = k

        # accept: rejection sampling, pure math on the returned logits
        draft_tokens = torch.tensor(
            [seq.spec_token_ids for seq in seqs], dtype=torch.long, device=target_logits.device
        )
        num_accepted, extra = rejection_accept(
            target_logits, torch.stack(draft_logits, dim=1), draft_tokens
        )
        num_accepted = num_accepted.cpu().tolist()
        extra = extra.cpu().tolist()

        # commit: accepted drafts + the extra token, checking stop conditions
        # inside the accepted run; on stop the tail is discarded (审阅 #10: the
        # discarded tail's blocks are freed below, undoing any hash state)
        outputs = []
        num_processed_tokens = 0
        for seq, n, extra_token in zip(seqs, num_accepted, extra):
            committed = seq.spec_token_ids[:n] + [extra_token]
            for tok in committed:
                seq.append_token(tok)
                num_processed_tokens += 1
                stop = (
                    (not seq.ignore_eos and tok == self.scheduler.eos)
                    or seq.num_completion_tokens >= seq.max_tokens
                    or (seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length)
                )
                if stop:
                    seq.status = SequenceStatus.FINISHED
                    break
            if seq.is_finished:
                # free everything: deallocate walks the full block table, so
                # the pre-allocated provisional blocks go with it (no rollback
                # needed on the finished path)
                block_manager.deallocate(seq)
                self.scheduler.running.remove(seq)
                outputs.append((seq.seq_id, seq.completion_token_ids))
            else:
                # pop the provisional blocks beyond the committed length and
                # reconcile block contents/hashes against the accepted prefix
                block_manager.rollback(seq)
            seq.clear_spec_buffer()
        return outputs, num_processed_tokens


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
