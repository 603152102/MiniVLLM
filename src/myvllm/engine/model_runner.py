import math
import torch
import pickle
import torch.distributed as dist
from pathlib import Path
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from myvllm.models.qwen3 import Qwen3ForCausalLM
#from myvllm.models.llama import LlamaForCausalLM
from myvllm.layers.sampler import SamplerLayer
from myvllm.engine.sequence import Sequence, SequenceStage
from myvllm.utils import *

class ModelRunner:
    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
        self.config = config
        self.event = event

        # set distributed config
        self.block_size = config['block_size']
        self.world_size = config['world_size']
        self.enforce_eager = config.get('enforce_eager', False)

        self.rank = rank
        dist.init_process_group('nccl', "tcp://localhost:12345", world_size=config['world_size'], rank=rank)
        torch.cuda.set_device(rank)

        # set model
        path_str = self.config['model_name_or_path']
        model_name = Path(path_str).name
        match model_name:
            case 'Qwen3-0.6B':
                self.model = Qwen3ForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    num_heads=config['num_heads'],
                    head_dim=config['head_dim'],
                    scale=config['scale'],
                    num_kv_heads=config['num_kv_heads'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    qkv_bias=config['qkv_bias'],
                    base=config['base'],
                    max_position=config['max_position'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    tie_word_embeddings=config['tie_word_embeddings'],
                    block_size=self.block_size,
                )
            # case 'Llama-3.2-1B-Instruct':
            #     self.model = LlamaForCausalLM(
            #         vocab_size=config['vocab_size'],
            #         hidden_size=config['hidden_size'],
            #         head_dim=config['head_dim'],
            #         num_qo_heads=config['num_qo_heads'],
            #         num_kv_heads=config['num_kv_heads'],
            #         has_attn_bias=config['has_attn_bias'],
            #         rms_norm_epsilon=config['rms_norm_epsilon'],
            #         rope_base=config['rope_base'],
            #         max_position_embeddings=config['max_position_embeddings'],
            #         intermediate_size=config['intermediate_size'],
            #         ffn_bias=config['ffn_bias'],
            #         num_layers=config['num_layers'],
            #         block_size=self.block_size,
            #         tie_word_embeddings=config['tie_word_embeddings'],
            #     )
            case _:
                raise Exception(f"Unsupported model: {config['model_name_or_path']}")

        # Load weights in GPU (model moved to GPU before loading weights)
        self.model = self.model.cuda(rank)

        # Load pretrained weights if model_name_or_path is provided
        if config.get('model_name_or_path'):
            from myvllm.utils.loader import load_weights_from_checkpoint
            load_weights_from_checkpoint(self.model, config['model_name_or_path'])

        # Load weights in CPU (move the model to GPU after loading weights)
        # self.model = self.model.cuda(rank)

        self.sampler = SamplerLayer()

        # speculative decoding: the draft runner shares this model's weights
        # and KV pool (see spec_decode/draft_runner.py); None when spec is off
        self.draft_runner = None
        if config.get('enable_speculative', False):
            from myvllm.spec_decode.draft_runner import DraftRunner
            self.draft_runner = DraftRunner(
                self.model,
                self.sampler,
                skip_layers=config.get('draft_skip_layers', False),
            )

        # Store default dtype before it's needed in allocate_kv_cache
        self.default_dtype = torch.get_default_dtype()

        # Debug flag for first decode step
        self._first_decode = False

        # warm up model so that we know peak memory usage
        self.warmup_model()
        # allocate kv cache
        self.allocate_kv_cache()
        # warm up the paths warmup_model cannot reach: decode (paged decode
        # kernel) and chunked prefill (paged prefill kernel) both need the KV
        # cache, so their JIT compilation would otherwise hit the first real
        # step of the first request
        self.warmup_decode_and_chunked()
        # capture cuda graph for decoding
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device(f'cuda:{rank}')
        torch.set_default_dtype(self.default_dtype)

        # IMPORTANT: Set up shared memory and barrier AFTER all model initialization
        # This ensures both ranks complete warmup/allocation before rank 1 enters its event loop
        if self.world_size > 1:
            # Synchronize before setting up shared memory
            dist.barrier()
            if self.rank == 0:
                # Try to clean up existing shared memory first
                try:
                    old_shm = SharedMemory(name='myvllm')
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass  # Doesn't exist, which is fine
                self.shm = SharedMemory(name='myvllm', create=True, size=2**20)
                # Barrier to ensure rank 1 waits until shared memory is created
                dist.barrier()
            else:
                # Wait for rank 0 to create shared memory
                dist.barrier()
                self.shm = SharedMemory(name='myvllm')
                # Don't call self.loop() here - let the spawning code handle it
                # Otherwise we'll be stuck in an infinite loop during __init__

    # only use read when rank != 0
    def read_shm(self):
        assert self.world_size > 1 and self.rank != 0, "read_shm can only be called when world_size > 1 and rank != 0"
        self.event.wait()
        n = int.from_bytes(self.shm.buf[:4], 'little') # read length
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    # only use write when rank == 0
    def write_shm(self, method_name: str, args: tuple):
        assert self.world_size > 1 and self.rank == 0, "write_shm can only be called when world_size > 1 and rank == 0"
        # encode the length first
        # Flatten: (method_name, args) where args is a tuple -> (method_name, *args)
        data = pickle.dumps((method_name, *args))
        n = len(data)
        self.shm.buf[:4] = n.to_bytes(4, 'little')
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # close shared memory, destroy process group, delete graphs
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs
            del self.graph_vars
        torch.cuda.synchronize()
        # Check if process group exists before destroying
        if dist.is_initialized():
            dist.destroy_process_group()
    
    # wait to read method and args from shared memory
    # execute the method with args
    # write results back to shared memory
    def loop(self):
        assert self.world_size > 1 and self.rank != 0, "loop can only be called when world_size > 1 and rank != 0"
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args) # Unpack args when calling
            if method_name == 'exit':
                self.exit()
                break

    # will be called by both rank == 0 and rank != 0
    # given method name and args from shared memory
    # execute the method and return results
    def call(self, method_name: str, *args: object):
        if self.world_size > 1 and self.rank == 0: # will be called in main engine
            self.write_shm(method_name, args)
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")

    # cleanup memory
    # compute max number of sequence based on max token and max model length
    # run empty sequence to warm up the model
    # clear memory
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_tokens = self.config['max_num_batch_tokens']
        max_model_length = self.config['max_model_length']
        batch_size = max_tokens // max_model_length
        seqs = [Sequence(token_ids=[0]*max_model_length, block_size=self.config['block_size']) for _ in range(batch_size)]
        # 等价于
        # seqs = []
        #     for _ in range(batch_size):
        #     seqs.append(
        #         Sequence(
        #             token_ids=[0] * max_model_length,
        #             block_size=self.config['block_size'],
        #         )
        #     )
        self.run(seqs, is_prefill=True)
        torch.cuda.empty_cache()

    # warm up the decode path (paged decode kernel + sampler in decode shape)
    # and the chunked prefill path (paged prefill kernel) with fake sequences
    # whose block tables point into the just-allocated KV cache, so that all
    # Triton JIT compilation and the sampler's torch.compile happen at init
    # time instead of during the first real request.
    # NOTE: max_num_blocks is a constexpr in both paged kernels, so each
    # distinct value compiles a separate kernel instance. Warm up both the
    # common case (1 block) and the longest sequence the config allows, or a
    # real request with a different block count would still pay a fresh
    # compile on its first step.
    def warmup_decode_and_chunked(self):
        # run eagerly even when cuda graphs are enabled: the graphs are not
        # captured yet at this point, and the goal is only to compile the
        # kernels, which graph capture later reuses
        enforce_eager = self.enforce_eager
        self.enforce_eager = True
        try:
            block_size = self.block_size
            max_blocks = math.ceil(self.config['max_model_length'] / block_size)
            for num_blocks in sorted({1, max_blocks}):
                seq_len = num_blocks * block_size
                # decode path: one token per sequence, paged decode kernel
                # reads the block table, sampler compiles for its decode shape
                seq = Sequence(token_ids=[0] * seq_len, block_size=block_size)
                seq.block_table = list(range(num_blocks))
                self.run([seq], is_prefill=False)
                # chunked prefill path: a chunk that follows an already
                # computed prefix, so cu_seqlens_q < cu_seqlens_k and the
                # paged prefill kernel runs
                half = block_size // 2
                seq = Sequence(token_ids=[0] * seq_len, block_size=block_size)
                seq.block_table = list(range(num_blocks))
                seq.num_computed_tokens = seq_len - half
                seq.num_prefill_chunk_tokens = half
                self.run([seq], is_prefill=True)
                # speculative verify path: per-sequence (k+1)-token chunk
                # following a computed prefix (same paged prefill kernel, same
                # cu_seqlens_q < cu_seqlens_k signature -- 审阅 #12: free
                # warmup, the kernel is already compiled for this block count)
                if self.draft_runner is not None:
                    k = self.config.get('num_spec_tokens', 4)
                    spec_seq_len = max(seq_len - k, k + 1)
                    vseq = Sequence(token_ids=[0] * spec_seq_len, block_size=block_size)
                    vseq.block_table = list(range(num_blocks))
                    vseq.spec_token_ids = [0] * k
                    self.run_verify([vseq], k)
        finally:
            self.enforce_eager = enforce_eager
        torch.cuda.empty_cache()

    # allocate kv cache memory blocks for model
    def allocate_kv_cache(self):
        # find all available memory
        free_mem, total_mem = torch.cuda.mem_get_info()
        total_free_mem = free_mem * self.config['gpu_memory_utilization']
        peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak']
        current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current']
        # reserve some room for peak memory usage during model execution
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)
        
        # find parameters to compute kv cache size
        num_layers = self.config['num_layers']
        num_kv_heads = self.config['num_kv_heads'] // self.world_size
        head_dim = self.config['head_dim'] if 'head_dim' in self.config else self.config['hidden_size'] // self.config['num_heads']

        # check whether the current free memory can hold at least one block
        # compute the actual byte required of each block
        block_bytes = self.block_size * 2 * num_layers * num_kv_heads * head_dim * self.default_dtype.itemsize
        num_available_kv_blocks = int(available_mem // block_bytes)
        assert num_available_kv_blocks >= 1, f'Not enough memory to hold at least one block of KV cache on rank {self.rank}'
        
        # Synchronize max_cached_blocks across all ranks.
        # Each rank independently computed num_available_kv_blocks from its own
        # free GPU memory. Ranks may differ slightly: rank-0 carries extra overhead
        # (NCCL buffers, process-group state) so it often has less free memory than
        # workers. Without sync, the scheduler (which runs only on rank-0) would use
        # rank-0's local value and could allocate more blocks than some rank can hold,
        # causing an OOM on that rank during KV cache writes.
        if self.world_size > 1:
            print(f"[Rank {self.rank}] Local max_cached_blocks: {num_available_kv_blocks}")
            per_rank_max_blocks_tensor = torch.tensor(
                num_available_kv_blocks,
                dtype=torch.long,
                device=f'cuda:{self.rank}'
            )
            # all_reduce with MIN: every rank learns the most conservative limit,
            # i.e. the block count that even the most memory-constrained rank can serve.
            # This single agreed-upon value is then stored in config so the Scheduler
            # (initialized afterwards on rank-0) never allocates more blocks than any
            # rank can physically hold.
            dist.all_reduce(per_rank_max_blocks_tensor, op=dist.ReduceOp.MIN)
            self.config['max_cached_blocks'] = per_rank_max_blocks_tensor.item()
        else:
            # Single GPU: no cross-rank sync needed; use the local value directly.
            self.config['max_cached_blocks'] = num_available_kv_blocks
        if self.rank == 0:
            print(f"[Rank 0] Global max_cached_blocks (min): {self.config['max_cached_blocks']}")

        # allocate max possible kv cache for the model, instead for each sequence
        # this is the key for paged attention: one giant KV cache pool, divided into blocks
        # IMPORTANT: Use zeros() instead of empty() to avoid garbage values
        allocated_kv_cache = torch.zeros(2, self.config['num_layers'], self.config['max_cached_blocks'], self.block_size, num_kv_heads, head_dim, device=f'cuda:{self.rank}')
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = allocated_kv_cache[0, layer_id]
                module.v_cache = allocated_kv_cache[1, layer_id]
                layer_id += 1

    # given seqs
    # prepare the data needed for a prefill forward pass
    # taking prefix cache, chunked prefill and P/D mixed batches into
    # consideration: input_ids, positions, cu_seqlens_q/k, slot_mapping
    # (where to write new KV values), block_tables (where to read KV values)
    # a PREFILL-stage sequence computes only the chunk stamped in
    # num_prefill_chunk_tokens (a stamp of 0 means no chunking: legacy
    # scheduler / warmup, the rest of the prompt is computed in one pass);
    # a DECODE-stage sequence contributes one token, like prepare_decode
    # cu_seqlens_q = [0, 3, 5, 9]
    #               │  │  │  │
    #               │  │  │  └─ end of seq3 (position 9)
    #               │  │  └──── end of seq2 (position 5)
    #               │  └─────── end of seq1 (position 3)
    #               └────────── start (position 0)
    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        # length: sum of all chunk lengths after prefix cache
        input_ids = []
        # absolute RoPE position of every new token: a chunk continues the
        # prompt, so its positions do not restart at 0
        positions = []
        # length: sum of all chunk lengths after prefix cache
        slot_mappings = []
        # length: num_seqs
        seqlens_q = []
        # length: num_seqs
        seqlens_k = []
        # length: num_seqs + 1
        cu_seqlens_q = [0]
        # length: num_seqs + 1
        cu_seqlens_k = [0]
        # block_tables: num_seqs x num_blocks (padded)
        block_tables = []
        for seq in seqs:
            if seq.stage == SequenceStage.DECODE:
                # decode sequence in a mixed batch: one new token per
                # sequence. Semantics identical to prepare_decode: context
                # length = len(seq) (attention excludes the new token itself)
                # and position = len(seq) - 1
                input_ids.append(seq.last_token)
                positions.append(len(seq) - 1)
                seqlens_q.append(1)
                seqlens_k.append(len(seq))
                cu_seqlens_q.append(cu_seqlens_q[-1] + 1)
                cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])
                if seq.block_table:
                    t = len(seq)
                    slot_mappings.append(seq.block_table[t // self.block_size] * self.block_size + t % self.block_size)
                continue
            token_ids = seq.token_ids
            start = seq.num_cached_tokens + seq.num_computed_tokens
            if seq.num_prefill_chunk_tokens > 0:
                end = start + seq.num_prefill_chunk_tokens
            else:
                # no chunk stamped: prefill everything that remains
                end = len(token_ids)
            input_ids.extend(token_ids[start:end])
            positions.extend(range(start, end))
            # q length = chunk length, k length = everything whose KV is valid
            # after this pass (cached prefix + previous chunks + this chunk);
            # under chunked prefill they differ, which is the signature of a
            # chunked pass
            seqlens_q.append(end - start)
            seqlens_k.append(end)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])
            if seq.block_table:
                # one slot per new token; a chunk may start and end mid-block,
                # so resolve each token's slot independently
                for t in range(start, end):
                    slot_mappings.append(seq.block_table[t // self.block_size] * self.block_size + t % self.block_size)
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            # pad block_tables
            all_block_tables = [seq.block_table for seq in seqs]
            max_num_blocks = max(len(bt) for bt in all_block_tables)
            for i, seq in enumerate(seqs):
                block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
                block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)

        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=slot_mapping_tensor,
            context_lens=None,
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
            positions=torch.tensor(positions, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
        )
        return input_ids


    # prepare input data for decoding
    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids = []
        context_lens = []   
        slot_mappings = []  
        block_tables = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            context_lens.append(len(seq))
            slot_mappings.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        for i, seq in enumerate(seqs):
            block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
            block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            context_lens=context_lens_tensor,
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
            # decode position of each sequence: its latest token
            positions=context_lens_tensor - 1,
        )
        return input_ids

    # prepare the temperature
    def prepare_sample(self, seqs: list[Sequence]) -> None:
        return torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    # when prefilling, directly compute model forward + logits
    # when decoding, use cuda graph execution to speed up
    # allocate input_ids, positions, slot_mapping, context_lens, block_tables, outputs
    # into graph_variable, and then replay the graph
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill or self.enforce_eager:
            # For varlen prefill, keep input_ids as 1D (concatenated tokens)
            # Do NOT unsqueeze - flash_attn_varlen_func expects 1D input with cu_seqlens
            hidden_states = self.model(input_ids)
            logits = self.model.compute_logits(hidden_states)
        else:
            bs = input_ids.size(0)
            context = get_context()

            # finds smallest captured graph that fits the batch size
            graph = self.graphs[next(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)]
            vars = self.graph_vars
            # copy input data into graph variables
            vars['input_ids'][:bs].copy_(input_ids)
            vars['slot_mapping'][:bs].fill_(-1)
            vars['slot_mapping'][:bs].copy_(context.slot_mapping)
            vars["context_lens"].zero_()
            vars['context_lens'][:bs].copy_(context.context_lens)
            vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # replay the graph
            graph.replay()
            logits = self.model.compute_logits(vars['outputs'][:bs])

        return logits


    # prepare prefill
    # prepare sample
    # run model
    # sample logits
    # reset context
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)
        logits = self.run_model(input_ids, is_prefill)
        # only sample when rank == 0
        token_ids = None
        if self.rank == 0:
            if is_prefill:
                # only sequences whose chunk completes the prompt are sampled:
                # their last chunk token is the last prompt token, so the
                # sampled token is the first completion token. Intermediate
                # chunks emit no token and get a -1 sentinel instead.
                final_indices = [
                    i for i, seq in enumerate(seqs)
                    if seq.num_prefill_chunk_tokens == 0
                    or seq.num_computed_tokens + seq.num_prefill_chunk_tokens >= seq.num_prompt_tokens
                ]
                token_ids = torch.full((len(seqs),), -1, dtype=torch.long, device=f'cuda:{self.rank}')
                if final_indices:
                    sampled = self.sampler(
                        logits[final_indices],
                        self.prepare_sample([seqs[i] for i in final_indices]),
                    )
                    token_ids[final_indices] = sampled
            else:
                token_ids = self.sampler(logits, self.prepare_sample(seqs))
        reset_context()
        return token_ids

    # ------------------------------------------------------------------
    # speculative decoding: draft propose steps + verify forward
    # ------------------------------------------------------------------

    # One draft decoding step (step i of k). Input = the token the draft
    # produced last (step 0: the sequence's last accepted token x), one
    # paged-decode forward whose KV for the input position t+i-1 lands in the
    # shared target cache, then a temperature-scaled sample for position t+i.
    # Step 0 DOES store position t-1: the last accepted token (the previous
    # round's bonus/replacement) has no KV yet -- nothing ever computed it --
    # and without this store the draft's first step attends a garbage slot,
    # making q_0 systematically wrong and the round always reject at level 0.
    # The verify pass re-stores slot t-1 with the target's own KV before its
    # attention reads it, so the target never sees the draft's value (correct
    # even under draft_skip_layers, where the two differ).
    # The RNG is reseeded per (round, step) so every rank draws the same draft
    # tokens (TP workers mirror this method through the shm loop and their
    # tokens feed the next step's input, so they must agree with rank 0).
    # Returns (scaled_logits [B, V], tokens [B]) where scaled_logits are the
    # logits divided by each sequence's temperature -- the exact distribution
    # acceptance.py compares against the target's (审阅 #1: one scaling path).
    def propose_step(self, seqs: list[Sequence], i: int, k: int, round_id: int):
        assert self.draft_runner is not None, "speculative decoding is not enabled"
        torch.manual_seed(round_id * 10007 + i)  # deterministic across ranks
        block_size = self.block_size
        input_ids = []
        context_lens = []
        slot_mappings = []
        for seq in seqs:
            t = len(seq)
            if i == 0:
                assert not seq.spec_token_ids, "draft buffer must be empty at step 0"
                input_ids.append(seq.last_token)
            else:
                input_ids.append(seq.spec_token_ids[i - 1])
            # KV of the input position t+i-1 lands in its canonical slot
            p = t + i - 1
            slot_mappings.append(seq.block_table[p // block_size] * block_size + p % block_size)
            context_lens.append(t + i)
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        block_tables = [bt + [-1] * (max_num_blocks - len(bt)) for bt in all_block_tables]
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            context_lens=context_lens_tensor,
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            positions=context_lens_tensor - 1,
        )
        with torch.inference_mode():
            hidden_states = self.draft_runner.forward(input_ids)
            logits = self.model.compute_logits(hidden_states)
        temperatures = self.prepare_sample(seqs)
        scaled_logits = logits / temperatures.unsqueeze(-1)
        tokens = self.sampler(logits, temperatures)
        reset_context()
        return scaled_logits, tokens

    # Prepare the verify forward: per sequence q = k+1 tokens -- the last
    # accepted token x (position t-1, whose KV is recomputed idempotently) and
    # the k draft tokens (positions t..t+k-1). The batch is chunk-shaped
    # (cu_seqlens_q < cu_seqlens_k), so the existing paged prefill kernel runs:
    # the chunk's KV occupies the tail of the KV range, which is exactly the
    # verify layout. Slots are resolved per token, cross-block safe.
    def prepare_verify(self, seqs: list[Sequence], k: int) -> torch.Tensor:
        block_size = self.block_size
        input_ids = []
        positions = []
        slot_mappings = []
        seqlens_q = []
        seqlens_k = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        for seq in seqs:
            t = len(seq)
            assert len(seq.spec_token_ids) == k, (
                f"expected {k} draft tokens, got {len(seq.spec_token_ids)}"
            )
            input_ids.extend([seq.last_token] + list(seq.spec_token_ids))
            positions.extend(range(t - 1, t + k))
            seqlens_q.append(k + 1)
            seqlens_k.append(t + k)
            cu_seqlens_q.append(cu_seqlens_q[-1] + k + 1)
            cu_seqlens_k.append(cu_seqlens_k[-1] + t + k)
            for p in range(t - 1, t + k):
                slot_mappings.append(seq.block_table[p // block_size] * block_size + p % block_size)
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        block_tables = [bt + [-1] * (max_num_blocks - len(bt)) for bt in all_block_tables]
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q=k + 1,
            max_seqlen_k=max(seqlens_k),
            slot_mapping=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            context_lens=None,
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            positions=torch.tensor(positions, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
        )
        return input_ids

    # One verify forward: the target model over [x] + k drafts per sequence.
    # Returns target_logits [B, k+1, V] scaled by each sequence's temperature:
    # rows 0..k-1 are the distributions that verify draft tokens 0..k-1, row k
    # is the bonus distribution (the position after the last draft token).
    def run_verify(self, seqs: list[Sequence], k: int) -> torch.Tensor:
        input_ids = self.prepare_verify(seqs, k)
        with torch.inference_mode():
            hidden_states = self.model(input_ids)
            # slice_last=False: keep every row (the prefill convention would
            # keep only the bonus row, but acceptance needs all k+1)
            logits = self.model.compute_logits(hidden_states, slice_last=False)
        B = len(seqs)
        logits = logits.view(B, k + 1, -1)
        temperatures = self.prepare_sample(seqs)
        reset_context()
        return logits / temperatures.view(B, 1, 1)

    # capture the CUDA graph:
    # pre-allocation at maximum sizes: allocated onece and reuse for all graphs
    # capture for different common batch sizes: [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    # with torch.cuda.graph(graph, self.graph_pool):
    #        run model() and exact sequence of CUDA kernels for running self.model() will be captured
    # (later use graph.replay() to run the captured graph)
    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        max_bs = self.config['max_num_seqs']
        max_len = self.config['max_model_length']
        max_num_blocks = math.ceil(max_len / self.block_size)
        # for decoding, input is always of shape (batch_size, 1)
        input_ids = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # for paged attention
        # where to write new KV values in the cache
        slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # how many tokens each sequence has processed
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # where to read KV values in the cache
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=f'cuda:{self.rank}')
        # output logits
        outputs = torch.zeros(max_bs, self.config['vocab_size'], device=f'cuda:{self.rank}')

        # graphs to be captured for different batch sizes
        batch_sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        graph_pool = None

        for batch_size in reversed(batch_sizes):
            graph = torch.cuda.CUDAGraph()
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping[:batch_size],
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
            )
            outputs[:batch_size] = self.model(input_ids[:batch_size])

            with torch.cuda.graph(graph, graph_pool):
                outputs[:batch_size] = self.model(input_ids[:batch_size])
                if graph_pool is None:
                    graph_pool = graph.pool()
            # store the captured graph
            self.graphs[batch_size] = graph

            # make sure that the capture is done before resetting and next capture
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )