from enum import Enum, auto
import math
from itertools import count 
from myvllm.sampling_parameters import SamplingParams
from copy import copy


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class SequenceStage(Enum):
    PREFILL = auto()
    DECODE = auto()


class Sequence:
    counter = count()

    def __init__(self, token_ids: list[int], block_size: int, sampling_params = SamplingParams()):
        self.block_size = block_size # number of tokens per block
        # record sequence id
        self.seq_id = next(Sequence.counter)
        # status
        self.status = SequenceStatus.WAITING
        # token ids, need copy so that it is a new list, won't be affected by outside changes
        self.token_ids = copy(token_ids)
        # last token
        self.last_token = self.token_ids[-1] if self.token_ids else None
        # num_tokens, num_prompt_tokens
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        # num_cached_tokens = 0
        self.num_cached_tokens = 0
        # block_table
        self.block_table = []
        # sampling_params' related things
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.max_model_length = sampling_params.max_model_length
        self.num_computed_tokens = 0  #chunked_prefill计算过的token数
        # computation phase: PREFILL while the prompt is still being computed
        # in chunks (no tokens generated), DECODE once generation starts
        self.stage = SequenceStage.PREFILL
        # number of prompt tokens scheduled for prefill in the current step,
        # valid only between schedule() and postprocess() of that step
        self.num_prefill_chunk_tokens = 0
        # speculative decoding state (only meaningful on the engine side and on
        # worker runs that consume them; empty when spec is off):
        # spec_token_ids: draft tokens proposed by the draft model, not yet
        #   committed to token_ids (or committed-but-not-yet-cleared tail)
        # num_spec_verified: how many of spec_token_ids already have target KV
        #   written (accepted across verify; used to truncate after rollback)
        self.spec_token_ids: list[int] = []
        self.num_spec_verified: int = 0

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, idx):
        return self.token_ids[idx]
    
    @property
    def is_prefill(self):
        return self.num_computed_tokens < self.num_prompt_tokens

    @property
    def is_prefill_done(self):
        return self.num_computed_tokens >= self.num_prompt_tokens

    @property
    def num_uncomputed_prompt_tokens(self):
        return max(
            self.num_prompt_tokens - self.num_computed_tokens,
            0,
        )

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return int(math.ceil(self.num_cached_tokens / self.block_size))

    @property
    def num_blocks(self):
        return int(math.ceil(self.num_tokens / self.block_size))

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - max(self.num_blocks - 1, 0) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks, f"Block index {i} out of range [0, {self.num_blocks})"
        if i == self.num_blocks - 1:
            return self.token_ids[-self.last_block_num_tokens:]
        else:
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            return self.token_ids[start_idx : end_idx]

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # Commit a run of accepted tokens in one call (speculative verify commits
    # n accepted drafts + a bonus/replacement token together). Mirrors
    # append_token() for a list; block accounting is done separately by the
    # BlockManager / scheduler after commit, as with single-token appends.
    def append_tokens(self, token_ids: list[int]):
        self.token_ids.extend(token_ids)
        if token_ids:
            self.last_token = token_ids[-1]
        self.num_tokens += len(token_ids)

    # Undo the last n committed completion tokens (discarding a bonus/replacement
    # tail on EOS, or a failed commit). Prompt tokens are never rolled back.
    def rollback_tokens(self, n: int):
        assert 0 <= n <= self.num_completion_tokens, (
            f"rollback_tokens({n}) exceeds completion tokens "
            f"({self.num_completion_tokens})"
        )
        if n == 0:
            return
        self.num_tokens -= n
        del self.token_ids[self.num_tokens:]
        self.last_token = self.token_ids[-1] if self.token_ids else None

    # (Re)store the pending draft buffer. Called after propose() fills it and
    # after commit/rollback truncates it. keep_verified = number of leading
    # spec tokens whose target KV is already valid (survives an accept).
    def set_spec_buffer(self, token_ids: list[int], num_verified: int = 0):
        self.spec_token_ids = token_ids
        self.num_spec_verified = num_verified

    def clear_spec_buffer(self):
        self.spec_token_ids = []
        self.num_spec_verified = 0

    def __getstate__(self):
        # spec_token_ids must cross the shared-memory boundary in full even on
        # the decode branch (where token_ids is minimized to last_token): the
        # worker needs the pending draft tokens to build the verify input.
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_computed_tokens,
            self.stage,
            self.num_prefill_chunk_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
            self.spec_token_ids,
            self.num_spec_verified,
        )

    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_computed_tokens,
            self.stage,
            self.num_prefill_chunk_tokens,
            self.block_table,
            last_token_or_ids,
            self.spec_token_ids,
            self.num_spec_verified,
        ) = state
        # Check if this is prefill (num_completion_tokens == 0) or decode phase
        num_completion_tokens = self.num_tokens - self.num_prompt_tokens
        if num_completion_tokens == 0:
            # Prefill: last_token_or_ids is the full token_ids list
            self.token_ids = last_token_or_ids
        else:
            # Decode: last_token_or_ids is just the last token
            self.token_ids = [last_token_or_ids]
        # Restore last_token attribute
        self.last_token = self.token_ids[-1] if self.token_ids else None
