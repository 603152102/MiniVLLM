import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from myvllm.engine.scheduler_chunked import ChunkedScheduler
from myvllm.engine.sequence import Sequence, SequenceStage, SequenceStatus
from unittest.mock import MagicMock


def make_scheduler(
    max_num_batched_tokens=100,
    max_num_sequences=10,
    max_cached_blocks=100,
    block_size=4,
    pd_mixed=True,
):
    return ChunkedScheduler(
        max_num_sequences=max_num_sequences,
        max_num_batched_tokens=max_num_batched_tokens,
        max_cached_blocks=max_cached_blocks,
        block_size=block_size,
        eos=0,
        pd_mixed=pd_mixed,
    )


def inject_running(scheduler: ChunkedScheduler, *seqs: Sequence):
    """Put sequences directly into the running queue, ready for decoding."""
    for seq in seqs:
        seq.status = SequenceStatus.RUNNING
        seq.stage = SequenceStage.DECODE
        seq.num_computed_tokens = seq.num_prompt_tokens
        scheduler.running.append(seq)


def all_tracked(scheduler: ChunkedScheduler, scheduled: list[Sequence]) -> set:
    """Return the set of all sequences the scheduler currently knows about."""
    return set(scheduler.running) | set(scheduler.waiting) | set(scheduled)


class TestChunkedPrefillSplitting:
    """
    A prompt larger than max_num_batched_tokens is prefilled in chunks, each
    bounded by the token budget. The final chunk completes the prompt and
    starts decode.
    """

    def test_chunks_split_across_steps(self):
        scheduler = make_scheduler(max_num_batched_tokens=4)
        seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], block_size=4)
        scheduler.add_sequence(seq)

        stamps = []
        # two partial chunks of 4 each
        for _ in range(2):
            scheduled, is_prefill = scheduler.schedule()
            assert is_prefill
            assert scheduled == [seq]
            stamps.append(seq.num_prefill_chunk_tokens)
            scheduler.postprocess(scheduled, [99])
            assert seq.stage == SequenceStage.PREFILL
            assert seq.num_completion_tokens == 0
        # the third chunk completes the prompt
        scheduled, is_prefill = scheduler.schedule()
        assert is_prefill
        stamps.append(seq.num_prefill_chunk_tokens)
        scheduler.postprocess(scheduled, [42])

        assert stamps == [4, 4, 2]
        assert seq.num_computed_tokens == 10
        assert seq.stage == SequenceStage.DECODE
        assert seq.status == SequenceStatus.RUNNING
        assert seq.num_completion_tokens == 1
        assert seq in scheduler.running
        assert seq not in scheduler.waiting

    def test_chunk_len_capped_by_budget(self):
        scheduler = make_scheduler(max_num_batched_tokens=7)
        seq_a = Sequence([1] * 6, block_size=4)
        seq_b = Sequence([2] * 6, block_size=4)
        scheduler.add_sequence(seq_a)
        scheduler.add_sequence(seq_b)

        scheduled, is_prefill = scheduler.schedule()
        assert is_prefill
        assert seq_a.num_prefill_chunk_tokens == 6
        assert seq_b.num_prefill_chunk_tokens == 1
        assert sum(s.num_prefill_chunk_tokens for s in scheduled) <= 7

        tracked = all_tracked(scheduler, scheduled)
        assert seq_a in tracked
        assert seq_b in tracked

        scheduler.postprocess(scheduled, [99, 99])
        assert seq_a.stage == SequenceStage.DECODE
        assert seq_a in scheduler.running
        assert seq_b.stage == SequenceStage.PREFILL
        assert seq_b in scheduler.waiting
        assert seq_b.num_computed_tokens == 1

    def test_max_num_sequences_caps_chunk_batch(self):
        scheduler = make_scheduler(max_num_sequences=1)
        seq_a = Sequence([1, 2, 3], block_size=4)
        seq_b = Sequence([4, 5, 6], block_size=4)
        scheduler.add_sequence(seq_a)
        scheduler.add_sequence(seq_b)

        scheduled, is_prefill = scheduler.schedule()
        assert is_prefill
        assert scheduled == [seq_a]
        assert seq_b in scheduler.waiting


class TestNoTokenDuringPartialChunk:
    """
    An intermediate prefill chunk must not produce a token: postprocess only
    advances the prefill cursor and sends the sequence back to waiting.
    """

    def test_partial_chunk_does_not_append_token(self):
        scheduler = make_scheduler(max_num_batched_tokens=2)
        seq = Sequence([1, 2, 3, 4, 5], block_size=4)
        scheduler.add_sequence(seq)

        scheduled, is_prefill = scheduler.schedule()
        assert is_prefill
        scheduler.postprocess(scheduled, [7])

        assert seq.num_tokens == 5
        assert seq.num_completion_tokens == 0
        assert seq.status == SequenceStatus.WAITING
        assert seq.stage == SequenceStage.PREFILL
        assert not scheduler.is_finished()

    def test_partial_chunk_returns_to_waiting_and_advances_cursor(self):
        scheduler = make_scheduler(max_num_batched_tokens=3)
        seq = Sequence([1, 2, 3, 4, 5, 6], block_size=4)
        scheduler.add_sequence(seq)

        scheduled, _ = scheduler.schedule()
        assert seq.num_prefill_chunk_tokens == 3
        scheduler.postprocess(scheduled, [99])

        assert seq.num_computed_tokens == 3
        assert seq in scheduler.waiting
        assert seq not in scheduler.running

    def test_partial_chunk_order_preserved(self):
        scheduler = make_scheduler(max_num_batched_tokens=4)
        seq_a = Sequence([1] * 6, block_size=4)
        seq_b = Sequence([2] * 6, block_size=4)
        scheduler.add_sequence(seq_a)
        scheduler.add_sequence(seq_b)

        scheduled, _ = scheduler.schedule()
        scheduler.postprocess(scheduled, [99])

        # seq_a goes back to the front of waiting, FCFS order preserved
        assert list(scheduler.waiting) == [seq_a, seq_b]

    def test_stamp_zeroed_after_postprocess(self):
        scheduler = make_scheduler(max_num_batched_tokens=2)
        seq = Sequence([1, 2, 3, 4, 5], block_size=4)
        scheduler.add_sequence(seq)

        # partial chunk
        scheduled, _ = scheduler.schedule()
        scheduler.postprocess(scheduled, [99])
        assert seq.num_prefill_chunk_tokens == 0

        # second partial chunk
        scheduled, _ = scheduler.schedule()
        scheduler.postprocess(scheduled, [99])
        assert seq.num_prefill_chunk_tokens == 0

        # final chunk
        scheduled, _ = scheduler.schedule()
        scheduler.postprocess(scheduled, [99])
        assert seq.stage == SequenceStage.DECODE
        assert seq.num_prefill_chunk_tokens == 0

    def test_scheduled_seq_status_waits_not_running(self):
        scheduler = make_scheduler(max_num_batched_tokens=2)
        seq = Sequence([1, 2, 3, 4, 5], block_size=4)
        scheduler.add_sequence(seq)

        scheduled, _ = scheduler.schedule()
        # parked in running only for the duration of the step, but the status
        # must not become RUNNING while the prompt is still being prefilled
        assert seq in scheduler.running
        assert seq.status == SequenceStatus.WAITING
        assert seq.stage == SequenceStage.PREFILL


class TestFinalChunkStartsDecode:
    """
    The chunk that completes the prompt samples the first completion token
    and transitions the sequence to decode.
    """

    def test_final_chunk_appends_first_decode_token(self):
        scheduler = make_scheduler(max_num_batched_tokens=100)
        seq = Sequence([1, 2, 3, 4, 5], block_size=4)
        scheduler.add_sequence(seq)

        scheduled, is_prefill = scheduler.schedule()
        assert is_prefill
        scheduler.postprocess(scheduled, [42])

        assert seq.stage == SequenceStage.DECODE
        assert seq.status == SequenceStatus.RUNNING
        assert seq.num_completion_tokens == 1
        assert seq.last_token == 42
        assert seq in scheduler.running
        assert seq not in scheduler.waiting

    def test_final_chunk_eos_finishes(self):
        scheduler = make_scheduler()  # eos=0
        seq = Sequence([1, 2, 3], block_size=4)
        scheduler.add_sequence(seq)

        scheduled, _ = scheduler.schedule()
        scheduler.postprocess(scheduled, [0])  # sampled token is EOS

        assert seq.status == SequenceStatus.FINISHED
        assert seq not in scheduler.running
        assert seq not in scheduler.waiting
        assert seq.block_table == []  # blocks deallocated

    def test_next_schedule_is_decode(self):
        scheduler = make_scheduler(max_num_batched_tokens=100)
        seq = Sequence([1, 2, 3], block_size=4)
        scheduler.add_sequence(seq)

        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [42])

        # no waiting sequences left: the next step decodes
        scheduled, is_prefill = scheduler.schedule()
        assert not is_prefill
        assert scheduled == [seq]


class TestPreemptResetsChunkState:
    """
    Preemption frees the sequence's KV cache blocks, so the prefill cursor
    and stage must reset: the prompt is recomputed from scratch in chunks.
    """

    def test_preempt_resets_cursor_stage_stamp(self):
        scheduler = make_scheduler()
        seq = Sequence([1, 2, 3], block_size=4)
        inject_running(scheduler, seq)

        mock_bm = MagicMock()
        mock_bm.can_append.return_value = False
        mock_bm.deallocate.return_value = None
        scheduler.block_manager = mock_bm

        scheduled, is_prefill = scheduler.schedule()
        # preemption resets the chunk state, and the freed sequence is
        # rescheduled as a prefill chunk in the same call (no wasted step)
        assert is_prefill
        assert scheduled == [seq]
        assert seq.status == SequenceStatus.WAITING
        assert seq.stage == SequenceStage.PREFILL
        assert seq.num_computed_tokens == 0
        assert seq.num_prefill_chunk_tokens == 3
        mock_bm.deallocate.assert_called_once_with(seq)

    def test_preempted_seq_reprefills_from_scratch(self):
        scheduler = make_scheduler()
        seq = Sequence([1, 2, 3], block_size=4)
        inject_running(scheduler, seq)

        mock_bm = MagicMock()
        mock_bm.can_append.return_value = False
        mock_bm.deallocate.return_value = None
        mock_bm.can_allocate.return_value = True
        mock_bm.allocate.return_value = None
        scheduler.block_manager = mock_bm

        scheduled, is_prefill = scheduler.schedule()
        # the preempted sequence's prompt is rescheduled for prefill in the
        # same call, allocating blocks again from scratch
        assert is_prefill
        assert scheduled == [seq]
        mock_bm.deallocate.assert_called_once_with(seq)
        mock_bm.can_allocate.assert_called_once_with(seq)
        assert seq.num_prefill_chunk_tokens == 3
        assert seq.stage == SequenceStage.PREFILL
        assert seq.num_computed_tokens == 0


class TestBug2ChunkBudget:
    """
    When the token budget is exhausted mid-queue, the head sequence that did
    not fit must stay in waiting, exactly like the original Bug2 regression.
    """

    def test_head_seq_not_lost_when_budget_exhausted(self):
        scheduler = make_scheduler(max_num_batched_tokens=2)
        seq_a = Sequence([1] * 6, block_size=4)
        seq_b = Sequence([2] * 6, block_size=4)
        scheduler.add_sequence(seq_a)
        scheduler.add_sequence(seq_b)

        scheduled, is_prefill = scheduler.schedule()
        assert is_prefill
        assert scheduled == [seq_a]
        assert seq_a.num_prefill_chunk_tokens == 2

        assert seq_b in scheduler.waiting, (
            "Bug 2 (chunk variant): seq_b was dropped when the token budget "
            "was exhausted instead of staying in waiting"
        )
        tracked = all_tracked(scheduler, scheduled)
        assert seq_a in tracked
        assert seq_b in tracked


class TestBug1CanAppendFailure:
    """
    Setup: 2 sequences in running, can_append returns False for the first.
    Expected: seq_a keeps its place and decodes; seq_b (the preempted one)
    is immediately rescheduled as a prefill chunk in the same step, with
    its prefill state reset. Neither may disappear.
    """

    def test_seqs_not_lost(self):
        scheduler = make_scheduler()
        seq_a = Sequence([1, 2, 3], block_size=4)
        seq_b = Sequence([4, 5, 6], block_size=4)
        inject_running(scheduler, seq_a, seq_b)

        mock_bm = MagicMock()
        # First call (for seq_a): cannot append; subsequent calls: True
        mock_bm.can_append.side_effect = [False, True, True, True]
        mock_bm.append.return_value = None
        mock_bm.deallocate.return_value = None
        scheduler.block_manager = mock_bm

        scheduled, is_prefill = scheduler.schedule()
        # the batch contains the rescheduled prefill chunk of seq_b
        assert is_prefill

        tracked = all_tracked(scheduler, scheduled)
        assert seq_a in tracked, "Bug 1: seq_a disappeared"
        assert seq_b in tracked, "Bug 1: seq_b disappeared"

        # the preempted seq_b is rescheduled as a prefill chunk in the same
        # step: parked in running for this step, not left in waiting
        assert seq_b not in scheduler.waiting
        assert seq_b.status == SequenceStatus.WAITING
        assert seq_b.stage == SequenceStage.PREFILL
        assert seq_b.num_computed_tokens == 0
        assert seq_b.num_prefill_chunk_tokens == 3  # whole 3-token prompt re-prefilled


class TestDecodeOrderPreservation:
    def test_running_order_preserved(self):
        scheduler = make_scheduler(max_num_batched_tokens=10)
        seq_a = Sequence([1], block_size=4)
        seq_b = Sequence([2], block_size=4)
        inject_running(scheduler, seq_a, seq_b)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_append.return_value = True
        scheduler.block_manager.append.return_value = None

        scheduled, is_prefill = scheduler.schedule()
        assert not is_prefill
        assert len(scheduled) == 2
        # both should be back in running after schedule()
        assert seq_a in scheduler.running
        assert seq_b in scheduler.running


class TestNoProgressGuard:
    """
    When the waiting head can never allocate and nothing else can run, the
    scheduler must fail loudly instead of returning an empty schedule.
    """

    def test_raises_when_head_cannot_allocate(self):
        scheduler = make_scheduler()
        mock_bm = MagicMock()
        mock_bm.can_allocate.return_value = False
        mock_bm.blocks = [None] * 100  # add_sequence capacity check
        mock_bm.free_block_ids = []  # guard error message
        scheduler.block_manager = mock_bm
        scheduler.add_sequence(Sequence([1, 2], block_size=4))

        with pytest.raises(RuntimeError, match="Scheduler made no progress"):
            scheduler.schedule()


class TestOversizeRejection:
    def test_add_sequence_rejects_prompt_larger_than_kv_cache(self):
        scheduler = make_scheduler(max_cached_blocks=2)  # 2 blocks * 4 tokens
        seq = Sequence([1] * 9, block_size=4)  # needs 3 blocks

        with pytest.raises(ValueError):
            scheduler.add_sequence(seq)


class TestMixedPrefillDecode:
    """
    P/D mixed scheduling: decode sequences and prefill chunks share one
    batch. Decode is scheduled first so its latency is never starved, then
    the remaining token budget is filled with chunks from waiting.
    """

    def _mixed_setup(self, budget, prompt_len=6):
        scheduler = make_scheduler(max_num_batched_tokens=budget)
        # decode sequence in running
        seq_d = Sequence([1, 2, 3], block_size=4)
        inject_running(scheduler, seq_d)
        # waiting sequence with a chunkable prompt
        seq_p = Sequence(list(range(10, 10 + prompt_len)), block_size=4)
        scheduler.add_sequence(seq_p)

        mock_bm = MagicMock()
        mock_bm.can_append.return_value = True
        mock_bm.append.return_value = None
        mock_bm.can_allocate.return_value = True
        mock_bm.allocate.return_value = None
        scheduler.block_manager = mock_bm
        return scheduler, seq_d, seq_p

    def test_decode_and_chunk_share_one_batch(self):
        scheduler, seq_d, seq_p = self._mixed_setup(budget=8)
        scheduled, is_prefill = scheduler.schedule()

        assert is_prefill
        # decode comes first, the chunk fills the remaining budget
        assert scheduled == [seq_d, seq_p]
        assert seq_p.num_prefill_chunk_tokens == 6  # min(prompt 6, budget left 7)

    def test_decode_scheduled_before_chunks(self):
        scheduler, seq_d, seq_p = self._mixed_setup(budget=3)
        scheduled, is_prefill = scheduler.schedule()

        assert scheduled == [seq_d, seq_p]
        assert seq_p.num_prefill_chunk_tokens == 2  # 3 budget - 1 decode token

    def test_pure_decode_when_budget_exhausted(self):
        scheduler, seq_d, seq_p = self._mixed_setup(budget=1)
        scheduled, is_prefill = scheduler.schedule()

        assert scheduled == [seq_d]
        assert not is_prefill  # no chunk in the batch -> pure decode
        assert seq_p in scheduler.waiting
        assert seq_p.num_prefill_chunk_tokens == 0

    def test_mixed_postprocess(self):
        # batch = decode seq + final-chunk seq + partial-chunk seq
        scheduler = make_scheduler(max_num_batched_tokens=10)
        seq_d = Sequence([1, 2, 3], block_size=4)
        inject_running(scheduler, seq_d)
        seq_final = Sequence([1, 2, 3], block_size=4)   # prompt 3 -> one final chunk
        seq_partial = Sequence([1] * 8, block_size=4)   # prompt 8 -> 6-token chunk, partial
        scheduler.add_sequence(seq_final)
        scheduler.add_sequence(seq_partial)

        mock_bm = MagicMock()
        mock_bm.can_append.return_value = True
        mock_bm.append.return_value = None
        mock_bm.can_allocate.return_value = True
        mock_bm.allocate.return_value = None
        scheduler.block_manager = mock_bm

        scheduled, is_prefill = scheduler.schedule()
        assert scheduled == [seq_d, seq_final, seq_partial]
        assert seq_final.num_prefill_chunk_tokens == 3
        assert seq_partial.num_prefill_chunk_tokens == 6  # 10 - 1 - 3

        scheduler.postprocess(scheduled, [7, 8, 99])
        # decode seq appended its token and stays running
        assert seq_d.num_completion_tokens == 1
        assert seq_d in scheduler.running
        # final chunk transitions to decode and appends the sampled token
        assert seq_final.stage == SequenceStage.DECODE
        assert seq_final.num_completion_tokens == 1
        assert seq_final in scheduler.running
        # partial chunk: no token, cursor advanced, back to waiting
        assert seq_partial.stage == SequenceStage.PREFILL
        assert seq_partial.num_completion_tokens == 0
        assert seq_partial.num_computed_tokens == 6
        assert seq_partial in scheduler.waiting
        assert seq_partial not in scheduler.running

    def test_preempt_then_chunk_in_same_batch(self):
        scheduler = make_scheduler(max_num_batched_tokens=8)
        seq_d = Sequence([1, 2, 3], block_size=4)
        inject_running(scheduler, seq_d)
        seq_p = Sequence([1] * 6, block_size=4)
        scheduler.add_sequence(seq_p)

        mock_bm = MagicMock()
        mock_bm.can_append.side_effect = [False, True, True]
        mock_bm.append.return_value = None
        mock_bm.deallocate.return_value = None
        mock_bm.can_allocate.return_value = True
        mock_bm.allocate.return_value = None
        scheduler.block_manager = mock_bm

        scheduled, is_prefill = scheduler.schedule()
        # the decode seq was preempted and its prompt rescheduled as a chunk
        # in the same call, sharing the budget with the waiting prompt
        assert is_prefill
        assert scheduled == [seq_d, seq_p]
        assert seq_d.stage == SequenceStage.PREFILL
        assert seq_d.num_computed_tokens == 0
        assert seq_d.num_prefill_chunk_tokens == 3   # its own prompt recompute
        assert seq_p.num_prefill_chunk_tokens == 5   # 8 budget - 3

    def test_decode_order_preserved_after_mixed(self):
        scheduler = make_scheduler(max_num_batched_tokens=8)
        seq_a = Sequence([1], block_size=4)
        seq_b = Sequence([2], block_size=4)
        inject_running(scheduler, seq_a, seq_b)
        seq_p = Sequence([3] * 4, block_size=4)
        scheduler.add_sequence(seq_p)

        mock_bm = MagicMock()
        mock_bm.can_append.return_value = True
        mock_bm.append.return_value = None
        mock_bm.can_allocate.return_value = True
        mock_bm.allocate.return_value = None
        scheduler.block_manager = mock_bm

        scheduled, is_prefill = scheduler.schedule()
        assert scheduled[:2] == [seq_a, seq_b]
        assert is_prefill
        # decode seqs are re-added to running in their original order, the
        # parked chunk sits behind them
        assert list(scheduler.running)[:2] == [seq_a, seq_b]

    def test_no_sequence_lost(self):
        scheduler, seq_d, seq_p = self._mixed_setup(budget=3)
        scheduled, _ = scheduler.schedule()

        tracked = all_tracked(scheduler, scheduled)
        assert seq_d in tracked
        assert seq_p in tracked


class TestPickleRoundtrip:
    """
    Sequences cross process boundaries via pickle when world_size > 1: the
    chunked prefill state must survive the roundtrip.
    """

    def test_mid_prefill_state_survives_pickle(self):
        seq = Sequence([1, 2, 3, 4, 5, 6], block_size=4)
        seq.stage = SequenceStage.PREFILL
        seq.num_computed_tokens = 3
        seq.num_prefill_chunk_tokens = 2

        restored = pickle.loads(pickle.dumps(seq))

        assert restored.num_computed_tokens == 3
        assert restored.stage == SequenceStage.PREFILL
        assert restored.num_prefill_chunk_tokens == 2
        # mid-prefill: no completion tokens, the full token_ids are shipped
        assert restored.token_ids == [1, 2, 3, 4, 5, 6]
        assert restored.last_token == 6

    def test_decode_state_survives_pickle(self):
        seq = Sequence([1, 2, 3], block_size=4)
        seq.append_token(42)
        seq.stage = SequenceStage.DECODE
        seq.num_computed_tokens = 3

        restored = pickle.loads(pickle.dumps(seq))

        assert restored.stage == SequenceStage.DECODE
        assert restored.num_computed_tokens == 3
        assert restored.num_prefill_chunk_tokens == 0
        # decode: only the last token is shipped and reconstructed
        assert restored.token_ids == [42]
        assert restored.last_token == 42


class TestPureModeNoMixing:
    """
    enable_pd_mixed=False: batches are pure. Prefill chunks are scheduled
    first, and whenever any chunk is scheduled the decode queue waits for
    the next step — the legacy pre-P/D-mixed behavior.
    """

    def _pure_setup(self, budget, prompt_len=6):
        scheduler = make_scheduler(max_num_batched_tokens=budget, pd_mixed=False)
        # decode sequence in running
        seq_d = Sequence([1, 2, 3], block_size=4)
        inject_running(scheduler, seq_d)
        # waiting sequence with a chunkable prompt
        seq_p = Sequence(list(range(10, 10 + prompt_len)), block_size=4)

        mock_bm = MagicMock()
        mock_bm.can_append.return_value = True
        mock_bm.append.return_value = None
        mock_bm.can_allocate.return_value = True
        mock_bm.allocate.return_value = None
        mock_bm.blocks = [None] * 100  # add_sequence capacity check
        scheduler.block_manager = mock_bm
        scheduler.add_sequence(seq_p)
        return scheduler, seq_d, seq_p

    def test_chunk_scheduled_alone_decode_waits(self):
        scheduler, seq_d, seq_p = self._pure_setup(budget=8)
        scheduled, is_prefill = scheduler.schedule()

        # the chunk is scheduled alone; the decode sequence is never mixed in
        assert scheduled == [seq_p]
        assert is_prefill
        assert seq_p.num_prefill_chunk_tokens == 6  # whole 6-token prompt
        assert seq_d not in scheduled
        assert seq_d in scheduler.running  # untouched, still queued for decode

    def test_decode_runs_when_no_chunk(self):
        scheduler, seq_d, seq_p = self._pure_setup(budget=8)
        # finish the only waiting prompt first
        scheduled, _ = scheduler.schedule()
        scheduler.postprocess(scheduled, [99])

        scheduled, is_prefill = scheduler.schedule()
        assert scheduled == [seq_d, seq_p]
        assert not is_prefill
        assert seq_d in scheduler.running
        assert seq_p in scheduler.running  # both re-added in order

    def test_preempted_seq_waits_for_next_step(self):
        scheduler = make_scheduler(pd_mixed=False)
        seq_a = Sequence([1, 2, 3], block_size=4)
        seq_b = Sequence([4, 5, 6], block_size=4)
        inject_running(scheduler, seq_a, seq_b)

        mock_bm = MagicMock()
        mock_bm.can_append.side_effect = [False, True]
        mock_bm.append.return_value = None
        mock_bm.deallocate.return_value = None
        scheduler.block_manager = mock_bm

        scheduled, is_prefill = scheduler.schedule()
        # unlike pd_mixed mode, the preempted seq is NOT re-chunked in the
        # same step: it stays in waiting until the next schedule() call
        assert scheduled == [seq_a]
        assert not is_prefill
        assert seq_b in scheduler.waiting
        assert seq_b.stage == SequenceStage.PREFILL
        assert seq_b.num_computed_tokens == 0
        assert seq_b.num_prefill_chunk_tokens == 0
