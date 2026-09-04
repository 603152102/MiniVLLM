from collections import deque
from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence, SequenceStage, SequenceStatus


class ChunkedScheduler:
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int):
        # block manager
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        # sequence queue
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos


    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0


    def add_sequence(self, sequence: Sequence):
        # Reject up front what the block manager could never satisfy, otherwise the
        # sequence sits in `waiting` forever and only surfaces as a stalled engine.
        capacity = len(self.block_manager.blocks)
        if sequence.num_blocks > capacity:
            raise ValueError(
                f"Sequence {sequence.seq_id} needs {sequence.num_blocks} blocks "
                f"({len(sequence)} tokens at block_size={self.block_manager.block_size}) "
                f"but the KV cache only holds {capacity}. "
                f"Raise max_cached_blocks or block_size, or shorten the prompt."
            )
        self.waiting.append(sequence)


    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_sequences = []
        current_scheduled_tokens = 0
        # An empty schedule is only legitimate when this call freed blocks by
        # preempting, so the next call can make progress. See the guard below.
        preempted = False
        # try schedule for chunked prefilling from waiting queue if not exceeding limits
        # a prompt too large for one batch is prefilled in chunks: each chunk
        # computes only its own tokens, and the sequence stays in WAITING (no
        # token is generated) until the whole prompt has been computed
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0]
            # allocate all blocks up front on the first chunk (MVP simplification),
            # later chunks only compute the remaining prompt tokens
            if seq.num_computed_tokens == 0 and not self.block_manager.can_allocate(seq):
                break
            chunk_len = min(seq.num_uncomputed_prompt_tokens, self.max_num_batched_tokens - current_scheduled_tokens)
            if chunk_len <= 0:
                break
            seq = self.waiting.popleft() # remove from waiting
            if seq.num_computed_tokens == 0:
                self.block_manager.allocate(seq)
            # stamp the chunk size for this step: the model runner reads it to
            # slice the chunk, postprocess() reads it to advance the cursor
            seq.num_prefill_chunk_tokens = chunk_len
            # INVARIANT: a scheduled prefill chunk is parked in `running` only
            # for the duration of this step; postprocess() re-homes it before
            # the next schedule() call, so the decode loop below never pops a
            # PREFILL-stage sequence
            self.running.append(seq)
            scheduled_sequences.append(seq)
            current_scheduled_tokens += chunk_len
        if scheduled_sequences:
            return scheduled_sequences, True

        # try schedule for completion from running queue
        while self.running:
            seq = self.running.popleft()
            assert seq.stage == SequenceStage.DECODE, f"seq {seq.seq_id} (stage {seq.stage}) reached the decode loop"
            # use can_append to check whether we can append one more token
            if not self.block_manager.can_append(seq):
                preempted = True
                if self.running:
                    self.running.appendleft(seq)
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)   #没有多余的队列可以释放
                    break
            else:
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    self.running.appendleft(seq)
                    break
                # append one token
                self.block_manager.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += 1 # only one token for completion

        # re-add to running queue in the same order
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))
        elif not preempted and (self.waiting or self.running):
            # Nothing was scheduled and nothing was preempted, so no engine state
            # changed: every later schedule() would take the same decisions and
            # LLMEngine.generate() would spin forever. Fail loudly instead.
            raise RuntimeError(
                "Scheduler made no progress: "
                f"{len(self.waiting)} waiting and {len(self.running)} running sequences, "
                f"{len(self.block_manager.free_block_ids)} of "
                f"{len(self.block_manager.blocks)} blocks free. "
                "This means either a sequence that cannot fit in the KV cache, or "
                "blocks leaked because their ref_count never returned to 0."
            )

        return scheduled_sequences, False


    def preempt(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        # deallocate() freed the sequence's blocks, so every KV computed so far
        # is gone: the prompt must be recomputed from scratch in chunks
        seq.stage = SequenceStage.PREFILL
        seq.num_computed_tokens = 0
        seq.num_prefill_chunk_tokens = 0
        self.waiting.appendleft(seq)


    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        partial_seqs = []
        for seq, token_id in zip(seqs, token_ids):
            if seq.stage == SequenceStage.PREFILL:
                # this seq was scheduled for a prefill chunk in this step
                seq.num_computed_tokens += seq.num_prefill_chunk_tokens
                seq.num_prefill_chunk_tokens = 0
                if not seq.is_prefill_done:
                    # intermediate chunk: no token is sampled and no stopping
                    # condition is checked, the sequence only advances its
                    # prefill cursor and goes back to waiting for the next chunk
                    self.running.remove(seq)
                    partial_seqs.append(seq)
                    continue
                # final chunk: the prompt is fully computed, the sampled token
                # is the first completion token and decode starts now
                seq.stage = SequenceStage.DECODE
                seq.status = SequenceStatus.RUNNING
                # fall through to the token handling below
            seq.append_token(token_id)
            # Check stopping conditions:
            # EOS token
            # Reached max_tokens limit (number of completion tokens)
            # Reached max_model_length limit (total sequence length including prompt)
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        # re-add partial sequences to the front of the waiting queue in their
        # original order, so each prompt is fully prefilled FCFS
        if partial_seqs:
            self.waiting.extendleft(reversed(partial_seqs))
