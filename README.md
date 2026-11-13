# Build Your Own micro-vLLM (GPT-2 edition)

A 15-step, test-driven path to writing an LLM inference engine from scratch in
Python, following the architecture of
[nopnoping/micro-vllm @ f9bb574](https://github.com/nopnoping/micro-vllm/tree/f9bb574d4cd9080887633fea75769c6d0bf8197c)
(itself a teaching-sized vLLM v1). We swap Llama/Qwen for **GPT-2** so the model
is boring and the *engine* is the star: the scheduler, the paged KV cache, the
model runner, the worker and the executor.

Every step has a pytest file. You are done with a step when its test file is green.

```
byo-micro-vllm/
├── minivllm/          <- YOUR code. Every function body is `raise NotImplementedError("Step N ...")`
├── reference/minivllm <- a complete solution (peek only when stuck)
├── tests/test_01 … test_15
└── examples/generate.py
```

```bash
pip install -r requirements.txt            # torch, transformers, pytest
pytest tests/test_01_request.py            # your code
MINIVLLM_IMPL=reference pytest             # the reference: 117 passed
```

Tests run on CPU with a tiny random GPT-2 that is saved to a temp dir like a
real checkpoint (no network). On a CUDA box the same tests run on the GPU, and
the GPU-only tests (fp16 paged attention, memory profiling, real `gpt2` text vs
HF) switch on automatically.

---

## The architecture you are building

```
 LLM.generate(prompts)                                     (Step 14)
   │  tokenize, add_request ×N, loop engine.step()
   ▼
 Engine.step()                                             (Step 6)
   │   plan   = scheduler.schedule()        ─┐  CPU, pure Python
   │   result = executor.execute(plan)       │  ── only immutable SchedulerOutput
   │   scheduler.update(plan, result)       ─┘     and ModelRunnerOutput cross here
   ▼
 Scheduler  ──────────── KVCacheManager ──── BlockPool       (Steps 1-5, 15)
   Request state machine    block tables       ref-counted page ids
   ─────────────────────────────────────────────────────────────────────
 Executor  (Single / MultiProcess)                          (Step 13)
   ▼
 Worker   (one per GPU)                                      (Step 13)
   ▼
 ModelRunner  owns: model weights + KV cache tensor          (Step 12)
   │  SchedulerOutput → flat input_ids/positions + AttentionMetadata
   ▼
 GPT2LMHeadModel  (flat [T] tokens, no padding)              (Steps 7, 10)
   ▼
 paged_attention(q,k,v, layer_cache, metadata)               (Steps 8, 9)
   ▼
 Sampler  (greedy / temperature / top-k / top-p / seed)      (Step 11)
```

The one idea that ties it together: **the scheduler and the GPU never share
mutable state.** The scheduler owns *which* page holds which token (integers);
the runner owns *what* is in the pages (tensors). The `SchedulerOutput` plan is
the only bridge, which is exactly why the executor can move the GPU side into
another process (Step 13) without the scheduler noticing.

### File map vs micro-vllm

| minivllm                         | micro-vllm @ f9bb574                         | what we simplified |
|----------------------------------|----------------------------------------------|--------------------|
| `core/request.py`                | `core/request.py`                            | — |
| `core/block.py`                  | `core/block.py`                              | no `BlockTable` class, plain lists |
| `core/kv_cache_manager.py`       | `core/kv_cache_manager.py`, `cache_group_manager.py`, `prefix_cache.py` | one cache group (GPT-2 has no sliding window) |
| `core/scheduler.py`              | `core/scheduler.py`                          | same policy; preempted requests go to the *front* of the queue |
| `core/scheduler_output.py`       | `core/scheduler_output.py`                   | one block table per request |
| `engine/engine.py`               | `engine/engine.py`, `engine/output.py`       | — |
| `executor/*`                     | `executor/interface.py`, `single_process.py` | **added** `multi_process.py` |
| `worker/worker.py`               | `worker/worker.py`                           | — |
| `worker/model_runner.py`         | `worker/model_runner.py`                     | no CUDA graphs |
| `attention/*`                    | `attention/attention.py`, `operators/reference.py` | reference operator only (no Triton) |
| `model/gpt2.py`                  | `models/llama.py` + `model_loader/*`         | GPT-2, loaded through `transformers` |
| `sampling/*`                     | `sampling/*`                                 | — |
| `llm.py`                         | `llm.py`                                     | — |

---

## Step 1 — Request state machine and sampling parameters

**Files:** `sampling/params.py`, `core/request.py`  **Test:** `pytest tests/test_01_request.py`

A request is a little state machine: `WAITING → RUNNING → (WAITING on preemption) → FINISHED`.
The counters matter more than the states:

```
prompt_token_ids  [p0 p1 p2 p3]           output_token_ids [o0 o1]
num_known_tokens   = 6
num_computed_tokens= 5   ← K/V for tokens 0..4 is in the KV cache
num_pending_tokens = 1   ← o1 was sampled last step; its K/V gets computed next step
```

The whole engine is "advance `num_computed_tokens` toward `num_known_tokens`,
then sample one more token". A decode step is just a request with exactly one
pending token. Implement the validation rules in the docstrings and two
invariants: you can only append a sampled token when nothing is pending, and
`preempt()` wipes computed progress but **keeps** generated tokens (they will be
recomputed as if they were prompt).

## Step 2 — BlockPool: the physical page allocator

**File:** `core/block.py` (`__init__`, properties, `allocate`, `release`, `ref_count`)  **Test:** `tests/test_02_block_pool.py`

The KV cache is a big tensor of `num_blocks` pages, each holding `block_size`
tokens. The pool only manages integer page ids. Rules:

- The **last id is the null block**, never allocated. Padded block-table entries
  point at it so every index is valid.
- Free ids live in a FIFO `deque`. `allocate(n)` is **all-or-nothing**: return
  `None` and change nothing if `n` pages are not free.
- Pages are **ref-counted** (needed for prefix sharing in Step 15). `release`
  appends to the *tail*, so the most recently freed pages are reused last.

Skip the `[Step 15]` methods for now.

## Step 3 — KVCacheManager: block tables

**File:** `core/kv_cache_manager.py` (Step-3 methods)  **Test:** `tests/test_03_kv_cache_manager.py`

Each request owns a *block table*: logical block `i` (tokens `[i*bs, (i+1)*bs)`)
→ physical page id. `allocate_slots(req, num_computed, num_new)` grows the table
just enough to cover `num_computed + num_new` tokens, atomically. Notice in the
test that two interleaved requests get **non-contiguous** pages — that is the
entire point of paging: no fragmentation, no reserving `max_model_len` per request.

`free_blocks` (preemption: keep the registration) vs `free` (request finished).
Ignore `prefix_hit` until Step 15.

## Step 4 — Scheduler: continuous batching and chunked prefill

**Files:** `core/scheduler_output.py` (given), `core/scheduler.py` (Step-4 methods)  **Test:** `tests/test_04_scheduler.py`

`schedule()` builds one immutable plan per engine step under two budgets:
`max_num_batched_tokens` and `max_num_seqs`.

1. **Running requests first.** Each asks for `min(pending, budget, max_model_len - computed)`
   tokens: 1 for decode, or the next chunk of a long prompt.
2. **Then waiting requests, FCFS.** Stop at the first one that does not fit
   (no queue jumping — see `test_fcfs_head_of_line_blocking`).

There is no prefill phase and no decode phase. A long prompt simply takes
several steps (chunked prefill) and shares each step with other requests'
decodes. `requires_sampling` is true only for the chunk that reaches the end of
the known tokens.

`update(plan, result)` commits: advance computed tokens, append the sampled
token, check finish (EOS, unless `ignore_eos` → `max_tokens` → `max_model_len`),
and free finished requests' blocks. Enforce **one plan in flight**: a second
`schedule()` before `update()` raises. For now, when allocation fails just stop
scheduling (preemption is next step). Do **not** call the prefix-cache methods yet.

## Step 5 — Preemption by recomputation, and abort

**File:** `core/scheduler.py` (`_preempt_last`, `abort_request`, and the preemption loop in `schedule`)  **Test:** `tests/test_05_preemption.py`

When a *running* request needs a new page and none is free, steal pages from the
**newest** running request: `preempt()` it, free its blocks, push it to the
**front** of the waiting queue, retry. When it comes back it recomputes prompt +
generated tokens in one prefill (see `test_preempted_request_recomputes_prompt_and_outputs`).
vLLM v1 does exactly this (recompute instead of swapping to CPU): a prefill is
compute-bound and fast, so recomputing is usually cheaper than PCIe copies.

## Step 6 — Engine loop (with a fake executor)

**Files:** `executor/interface.py` (given), `engine/engine.py`  **Test:** `tests/test_06_engine.py`

```python
plan = scheduler.schedule(); result = executor.execute(plan); scheduler.update(plan, result)
```

The test's `FakeExecutor` behaves like a model *with a KV cache*: it stores what
it has computed per request and its "next token" depends on the whole sequence.
If you schedule wrong positions, double-feed a chunk, or forget that preemption
resets progress, outputs change. The parametrized test asserts outputs are
identical under roomy, chunked, queued and heavily preempting configurations.
**This is the most important invariant of the whole engine: scheduling decisions
must never change results.** You now have a complete engine with no GPU code.

## Step 7 — GPT-2 as a flat-token model

**File:** `model/gpt2.py` (Step-7 methods)  **Test:** `tests/test_07_gpt2_model.py`

Serving engines don't use `[batch, seq]` padded tensors. The model takes a flat
`input_ids: [T]` and `positions: [T]` and returns `[T, E]`. For now
`_dense()` handles one sequence with causal SDPA.

GPT-2 specifics: learned position embeddings (`wpe`), pre-LayerNorm blocks,
`gelu` with tanh approximation, and `lm_head` **tied** to `wte`. HF stores
`c_attn`, `c_proj`, `c_fc` as `Conv1D` with weight shape `[in, out]`, so
transpose them into `nn.Linear`. Skip `attn.bias` buffers (they are causal masks,
not weights). The test compares logits to `transformers` at `1e-4`.

## Step 8 — Paged KV cache addressing

**File:** `attention/metadata.py`  **Test:** `tests/test_08_paged_kv_cache.py`

One layer's cache is `[2, num_blocks, block_size, H, D]`. View dims 1-2 as a flat
array of *slots*; token `p` of a request lives at

```
slot = block_table[p // block_size] * block_size + p % block_size
```

Implement `map_cache_slots`, `write_kv_cache` (an `index_copy_` scatter) and
`gather_kv_cache` (an `index_select`). Read `AttentionMetadata`: `slot_mapping`
says where to write the new tokens, `query_start_loc` (cu_seqlens) says which
rows of the flat batch belong to which request, `context_lens` says how much
history each request sees after this step.

## Step 9 — Paged attention

**File:** `attention/paged_attention.py`  **Test:** `tests/test_09_paged_attention.py`

1. Write this step's K/V into the cache *first* (a token attends to itself).
2. For each request, gather its history `[0, context_len)`.
3. Its queries sit at absolute positions `[context_len - q_len, context_len)`;
   mask keys with position > query position.
4. Softmax in fp32.

One call serves a **mixed batch**: a fresh prefill, a decode, and a middle prefill
chunk at the same time (`test_mixed_batch_equals_per_request_dense`). The Python
loop is the readable version; FlashAttention-varlen / vLLM's paged kernels /
micro-vllm's `triton_paged.py` do the same math in one fused kernel. Writing that
kernel in Triton is a great follow-up exercise.

## Step 10 — GPT-2 on the paged cache

**File:** `model/gpt2.py` (`GPT2Attention._paged`, and make `forward` dispatch)  **Test:** `tests/test_10_gpt2_paged.py`

Wire attention to `paged_attention` when metadata is present. Every layer gets
its own cache tensor; all layers share one `AttentionMetadata` (same tokens, same
pages). The test runs chunked prefill + token-by-token decode and checks every
logit against HF's full-sequence forward, and runs three requests at different
stages in one forward. `test_cache_is_actually_used` corrupts the cache to prove
decode really reads it.

## Step 11 — Sampler

**File:** `sampling/sampler.py`  **Test:** `tests/test_11_sampler.py`

Greedy = argmax. Otherwise temperature → top-k → top-p → softmax → multinomial.
The subtle part is the RNG: seed a fresh generator with `seed + sample_index`
(the index of the token within its request). Then the k-th token of a request is
the same whether it was sampled alone, in a batch of 64, or after a preemption.
Global RNG would make outputs depend on batch composition.

## Step 12 — ModelRunner

**File:** `worker/model_runner.py`, plus `kv_cache_bytes_per_block` in `model/gpt2.py`  **Test:** `tests/test_12_model_runner.py`

The runner is the GPU side of the plan:

- **`initialize_kv_cache`** sizes the cache. With `num_blocks=None` on CUDA it
  profiles: run a dummy `max_num_batched_tokens` forward, measure peak activation
  memory, then `budget = total × gpu_memory_utilization − used − peak`, and
  `num_blocks = budget // bytes_per_block` where
  `bytes_per_block = 2 × layers × block_size × hidden × dtype_size`
  (≈ 576 KiB per 16-token block for gpt2-small in fp16). Then allocate **one**
  tensor `[L, 2, num_blocks, block_size, H, D]`. This is the "use all leftover
  VRAM for KV" trick that gives vLLM its throughput.
- **`build_model_input`** flattens the plan: concatenated ids, absolute positions,
  slot mapping, `query_start_loc`, `context_lens`, padded block tables (pad = null
  block), and the row of the **last** token of each request that samples.
- **`execute`** = forward → logits for sample rows only → Sampler. Computing
  logits for every token would cost `T × vocab`, mostly thrown away.

## Step 13 — Worker and executors

**Files:** `worker/worker.py`, `executor/single_process.py`, `executor/multi_process.py`  **Test:** `tests/test_13_worker_executor.py`

The **Worker** is the per-device object: build model → measure memory → allocate
KV cache, in that order, and report `num_blocks` upward so the scheduler can size
its BlockPool to match the real tensor.

The **Executor** decides *where* workers live. `SingleProcessExecutor` calls the
worker directly. `MultiProcessExecutor` starts a `spawn` child (CUDA can't be
forked), and ships pickled `SchedulerOutput`s over a `Pipe`. The engine can't
tell them apart — that is the payoff of keeping plans immutable and
plain-Python. vLLM's tensor-parallel executor is this plus N workers and NCCL
all-reduces inside the model.

## Step 14 — LLM facade, end to end

**File:** `llm.py`  **Test:** `tests/test_14_llm_end_to_end.py`

Wire it up in the right order (Executor → `num_blocks` → KVCacheManager →
Scheduler → Engine), accept text or token ids, default `eos_token_id` from the
tokenizer, return results in **prompt order**. The test compares greedy outputs
with `transformers.generate()` under four configurations (plain batching,
chunked prefill, queueing, preemption) and proves preemption actually happened.
With network access, `test_real_gpt2_text` checks real `gpt2` text against HF.

Then run the real thing:

```bash
python examples/generate.py
```

## Step 15 — Automatic prefix caching

**Files:** the `[Step 15]` methods in `core/block.py` and `core/kv_cache_manager.py`, plus two `if config.enable_prefix_cache:` hooks in `scheduler.py`  **Test:** `tests/test_15_prefix_cache.py`

Identify each **full** block by a chained hash: `h_i = H(h_{i-1}, tokens_i)`, so
a block's identity includes its entire prefix. After each step, index newly
completed blocks (`cache_full_blocks`). On admission, walk the prompt's hashes
(`find_prefix_hit`) and reuse matching pages by bumping their ref count;
start the request at `num_computed_tokens = hits × block_size`.

Details that the tests pin down:

- Never cover the last token: the model must compute ≥ 1 token to get logits.
- A freed page keeps its hash and stays in the free queue — it is *reusable* until
  the allocator hands it out as a fresh page, which evicts the hash (LRU via the
  FIFO free queue from Step 2).
- Reusing a cached page with `ref_count == 0` must pull it out of the free queue
  and count against the free budget, atomically with the rest of the allocation.

Shared system prompts now skip most of their prefill
(`test_cache_hit_skips_prefill_compute`), and outputs still match HF exactly.

---

## Where to go next

- Replace the Python loop in `paged_attention` with a Triton kernel (compare with micro-vllm's `attention/operators/triton_paged.py`).
- CUDA graphs for decode-only batches (micro-vllm's `_capture_cuda_graphs`).
- Port the model to Llama: RoPE (positions enter attention, not embeddings) and GQA (`repeat_interleave` K/V heads).
- Tensor parallelism: `MultiProcessExecutor` with N workers, column/row-parallel linears, `torch.distributed` all-reduce.
- An async front end (OpenAI-compatible server) that calls `engine.step()` in a background loop.
