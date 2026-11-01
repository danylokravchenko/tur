# Serving technique gaps

Created: 2026-11-01. Status: research backlog; nothing here has been built or measured in Tur.

This page compares Tur with vLLM, llama.cpp, SGLang, oMLX, BaseRT and recent papers, and lists techniques Tur lacks. Tur's existing work (chunked prefill, prefix snapshots, device sampling, exact speculation, the expert cache) is left out. Every number comes from its source, on that source's hardware and models, and doesn't predict Tur's results. The order is a judgment of value against effort.

## Build first

These keep output exact and cost little to try.

### Structured output over HTTP

The engine supports llguidance grammars, but the server always passes `grammar: None` (`crates/tur/src/server/routes/chat.rs`). It accepts only `tool_choice=auto`, and the code has no `json_schema` or `logprobs`. Exposing `response_format`, required and named tool choice, and `logprobs` closes the largest gap with other OpenAI-compatible servers.

Two techniques make this better than a plain mask. llama.cpp's [lazy grammars](https://docs.rs/llama-cpp-4/0.7.0/llama_cpp_4/chat/) stay dormant until a tool-call marker appears, so reasoning runs unconstrained. [XGrammar](https://arxiv.org/pdf/2411.15100) and [llguidance](https://github.com/guidance-ai/llguidance) also skip sampling when the grammar leaves one possible token. Such a token has probability 1 under the mask, so the shortcut stays exact and a JSON key can run as one multi-token pass. Tur currently rejects grammars that ask for this (`crates/tur-engine/src/backend/guidance.rs`), and it would need a rule for advancing the random stream so that same-seed results stay identical.

### Request ordering beyond arrival order

Prompts run in arrival order, with no priorities or per-client fairness. The tenant only namespaces the prefix cache, so one client's 8K-token prompts can hold back another's short question.

Remaining prompt length is known exactly, so the scheduler could run the shortest remaining prompt first and keep `max_queue_wait` as the bound on starvation. This is a proposal for Tur rather than a published result. The nearest evidence is [Learning to Rank](https://arxiv.org/abs/2408.15792v1), which reports 2.8× lower chatbot latency by approximating shortest-job-first, using a small helper model to rank output lengths.

If several clients share one Mac, three designs apply. [Niyama](https://arxiv.org/abs/2503.22562v1) blends earliest-deadline-first with shortest-job-first across service classes, which would separate interactive from batch work. [VTC](https://arxiv.org/pdf/2401.00588) gives per-client fairness counted in tokens, with a 2× bound on the service gap between backlogged clients. [Autellix](https://arxiv.org/abs/2502.13965v1) favors agent programs that have received the least total service, and reports 4 to 15× higher program throughput at the same latency.

### A drafter that needs no model

No learned drafter exists for Qwen, and the MTP head turns prefix reuse off. A table-based drafter avoids both problems.

[Token Recycling](https://arxiv.org/abs/2408.08696v1) keeps the 8 most likely next tokens for each vocabulary token, in under 2 MB, and refreshes them from the logits of every verified position. llama.cpp's [`ngram-mod`](https://huggingface.co/OpenTransformer/llama.cpp-prismml/blob/main/docs/speculative.md) shares a hash pool of n-grams across slots. Tur should split any shared table by `cache_salt`, so that one tenant's text can't influence another's drafts or timing.

Token Recycling's report of about 2× comes from greedy decoding with a draft tree. Tur's recurrent state favors a single chain, and device sampling returns only the chosen token, so collecting candidates has its own cost. Replaying recorded token traces offline can bound the acceptance rate before any kernel work.

### Reacting to memory pressure while running

The spare-memory check runs only at load, and serving four concurrent requests on a 5.9 GiB plan swapped out 251 MiB. The server already samples system memory every 10 seconds, so it could use that reading to evict snapshots and lower `max_running` before macOS swaps.

### Keeping state across tool calls

[InferCept](https://arxiv.org/abs/2402.01869v2) reports that recomputing context after external calls takes 37 to 40% of forward time, and [Tokencake](https://arxiv.org/abs/2510.18586v1) targets the same idle state. Tur already keeps a zero-copy snapshot at the end of each request, which removes most of that cost until eviction takes it. A response ending in `finish_reason=tool_calls` is a strong sign that the conversation will return, so eviction could protect that snapshot for a bounded time.

## Measure before committing

### Layered prefill for MoE models

Chunked prefill sends every chunk through every layer, so each chunk reads the experts again. On gpt-oss-20b, chunking at 1,024 instead of 256 tokens cut a 2,037-token prompt from 32 s to 18 s (see [Expert cache](../architecture/expert-cache.md)). [Layered prefill](https://arxiv.org/abs/2510.08055) splits the work by layers instead: each pass advances the whole prompt through one group of layers while decodes run through all of them, so each layer sees the prompt once. On H100s it reports up to 39% less expert-load traffic, with mean first-token time falling from 2.80 s to 1.23 s and mean token gaps from 32.9 ms to 21.5 ms.

The paper doesn't size the buffers that hold activations between groups, and those would enter the memory plan. Tur would also need recurrent state at every snapshot boundary, since prefix reuse depends on it. Gains are largest on long prompts, and dense models have no expert reads to save.

### A shorter path from SSD to GPU

Expert reads go into host memory, then a new staging buffer, then a blit into a private arena (`crates/tur-metal/src/experts.rs`). Uploads and copies took 31 s of an earlier 83 s run, before uploads moved into the read threads. [`MTLIOCommandQueue`](https://developer.apple.com/documentation/metal/resource-loading.md) loads files directly into GPU resources, and reading into a shared arena would drop the blit. Either may need bindings that conflict with the ban on `unsafe`, so this needs a decision on dependencies first. Re-measure the current split before starting.

### M5 Neural Accelerators

[BaseRT](https://arxiv.org/abs/2607.19438) routes prefill matrix work through Metal 4 tensor kernels and reports up to 6.4× prompt speed over llama.cpp and 3.9× over MLX on an M5 Pro, while decode stays on existing kernels. [Apple's MLX write-up](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) reports up to 4× faster first tokens against an M4. BaseRT is a self-reported preprint, and only its abstract was read for this page. Both support retrying TensorOps and chunked DeltaNet prefill (see [Rejected experiments](../design/rejected.md#worth-retrying-on-other-hardware)), which needs an M5 to test.

### Idle sleep

[vLLM's sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode.html) releases weights and K/V without stopping the server. Tur holds its whole memory plan while idle, which matters on a Mac someone is also using. Dropping everything and reloading on the next request is the simple version, and its value depends on reload time, which needs measuring per checkpoint and format.

### Wired memory

MLX exposes a [wired limit](https://github.com/ml-explore/mlx-lm/issues/883) that keeps weights resident on macOS 15 and later. Reports of its benefit are anecdotal, and users describe unrecoverable systems when too much is pinned. Try it only if first-token time after idle proves to be a problem.

## Not recommended

- Running predicted experts without checking them, as [Speculating Experts](https://arxiv.org/abs/2603.19289) does, breaks Tur's exactness rule. Its better predictor adds under a point on gpt-oss (0.857 against 0.849) over the next-layer router signal Tur already reads ahead with.
- A general SSD tier for K/V, as in [oMLX](https://jimmysong.io/ai/omlx) and vLLM, adds steady SSD writes and conflicts with the expert cache's rule of making none. Saving pinned prefixes on request is the most this could justify.
- [Purgeable buffers](https://developer.apple.com/documentation/metal/mtlpurgeablestate.md) suit resources unused for some time, [according to Apple](https://developer.apple.com/forums/thread/698161), and the expert arena is one large buffer in constant use.
- Prefill/decode disaggregation and Mooncake-style connectors move state between machines, and one Mac is Tur's unit of deployment.

## Limits of this survey

Most papers were read as abstracts or excerpts, and the sources' claims weren't reproduced. [CommitMoE](https://ojs.aaai.org/index.php/AAAI/article/view/39454), which uses router certainty to guide offloading, blocked the fetch, so only a search summary was seen, and it may be worth a full read. [ExpertFlow](https://arxiv.org/pdf/2410.17954v2) and [FIRM-MoE](https://ojs.aaai.org/index.php/AAAI/article/view/39106) were also seen only as search summaries. The llama.cpp details come from mirrors and wrapper documentation, so check them against upstream. Thermal-aware scheduling and K/V compression weren't covered, and the search on swap-versus-recompute preemption found nothing useful.
