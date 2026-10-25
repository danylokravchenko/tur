# Tuning

The defaults favor one resident model and interactive latency. This page explains what
each knob trades, gives recipes for common situations, and shows how to measure the
result. The [Architecture](../architecture/overview.md) chapters explain the mechanisms.

## Persist tuning settings

Store reusable settings in [`tur.toml`](configuration.md) instead of repeating flags.
For example:

```toml
[engine]
prefill_chunk_size = 512

[server]
max_batch_size = 2
request_timeout_secs = 120
```

`tur serve --max-batch-size 4` overrides just that setting for one start. Advanced
scheduler timing, waiting-token and streaming limits are configurable in the file
without adding more command-line flags.

## Read the memory plan first

Every start logs the plan:

```text
Memory plan resident_bytes=3017015772 weights=1309441280 state_pool=181469184 scratch_peak=251789312
```

```text
resident = weights + state pool (K/V pages, recurrent slots, snapshots, drafts)
         + scratch peak + in-flight outputs + overhead (a tenth of the working set)
```

Tur leases this plan before reading any weight. It refuses a plan beyond the device's
recommended working set rather than letting macOS swap. If a start fails with a memory
error, shrink one of the terms using the table below.

| Term | Knob | Effect |
| --- | --- | --- |
| Weights | `--weight-format q4k`, or a [prepacked](quantization.md) artifact | Q4K: about 70% of Q8_0's bytes at load time, about a third as a tied-model artifact |
| K/V pages | `--kv-format q8` | Half of F16's K/V bytes, within 1e-4 nats of F32; slower prefill |
| Recurrent slots | `--recurrent-state f16` | Half the recurrent bytes, 6–7% faster batched decode; error grows with decode length, and speculation turns off |
| State pool | `--state-memory-mb` | Default: an 8K-token context per concurrent request |
| Snapshots | `--prefix-snapshots N` | Default: one per running request, capped at 1/8 of the pool |
| Concurrency | `--max-batch-size` | Each running request holds two recurrent slots |

Measured state pool on Qwen3.5-0.8B, for one request and 8K tokens: 181 MB with the
defaults, 134 MB with `--kv-format q8`, 144 MB with `--recurrent-state f16`.

## Knobs and their trade-offs

| Knob | Raise it to | Lower it to | Default |
| --- | --- | --- | --- |
| `--max-batch-size` | Serve more users at once; more aggregate tokens/s | Cut per-request latency and memory | 8 |
| `--prefill-chunk-size` | Finish long prompts sooner (better TTFT for them) | Interleave prompts more finely with decodes (smoother streaming) | 256 |
| `--max-batch-tokens` | Allow bigger prefill passes | Bound pass duration | 2048+ |
| `--max-waiting` | Absorb bursts | Shed load early with 429s | 256 |
| `--state-memory-mb` | Longer contexts, more snapshots | Free memory for weights or other processes | derived |
| `--prefix-snapshots` | Keep more shared prefixes warm | Free pool memory | per running request |
| `--speculative-draft` | More tokens per pass on repetitive output | Steadier token timing | 0 |

The scheduler already adapts the prefill budget every pass, targeting about 60 ms or
twice the decode-only cost. These knobs set the bounds it works within (see
[Scheduling](../architecture/scheduling.md)).

## Recipes

### A local assistant: one user, lowest latency

```sh
tur serve --max-batch-size 2 --prefill-chunk-size 512
```

- A small batch keeps every pass short.
- Larger prefill chunks get a long first prompt done in fewer passes.
- Add `--mtp` for 1.12–1.38× faster generation on novel text at one to four sequences.
  Output is identical, but tokens arrive in bursts, and prefix reuse turns off.

### A chat service: many concurrent users

```sh
tur serve --max-batch-size 16 --max-waiting 512 --request-timeout-secs 120 --kv-format q8
```

- A larger batch gives more aggregate throughput. Batched decode is bound by
  recurrent-state bandwidth, so per-user speed drops as the batch grows.
- `--kv-format q8` halves K/V memory, leaving room for the extra contexts.
- The timeout fails stuck requests with HTTP 504; the queue bound turns overload into
  fast 429s rather than unbounded waits.
- Keep prefix reuse on: every user's system prompt is shared.

### Agents and shared system prompts

```sh
tur serve --max-batch-size 8 --prefix-snapshots 16
```

- Agent loops resend a growing conversation, and reuse caches most of it. On the 0.8B, an
  agent loop cached 83% of its prompt tokens.
- Users who reach a cold shared prompt at the same time wait for the first one to
  compute it, so they don't compute it eight times.
- Leave `--mtp` off: it disables prefix reuse.
- When most requests share a warm prompt and their first token matters more than
  throughput, let restorable requests pass older cold ones:

  ```toml
  [engine]
  prefix_restorable_first = true
  ```

  On the 0.8B, warm requests queued behind cold ones got their first token 5.4× sooner
  (4.43 → 0.82 s), while cold requests waited 15% longer and throughput fell 10%. The
  queue-age limit (`max_queue_wait_ms`, 2 s) still bounds how long a cold request is
  passed over.
- Separate tenants with the `cache_salt` request field.
- Check `usage.prompt_tokens_details.cached_tokens` in responses to see what was reused.

### Code editing, extraction, structured output

```sh
tur serve --speculative-draft 4
```

Prompt lookup drafts tokens copied from the request's own history: quoted code,
repeated keys, echoed text. On repetitive output a single request ran 3.25× faster.
Drafting adapts itself and switches off when it stops paying, for example at high
concurrency.

### Long contexts on a 16 GB Mac

```sh
tur serve --max-batch-size 2 --kv-format q8 --state-memory-mb 3072
```

- K/V grows with context, and recurrent state does not, so long contexts are mostly a
  K/V cost.
- Q8 K/V roughly halves it.
- An explicit pool size gives the few running requests more room than the 8K-per-request
  default.
- Requests that can never fit fail immediately with an out-of-memory error.

### The smallest footprint

```sh
tur quantize --model ~/models/m --revision m1 --output ~/models/m-q4k --calibration calib.txt
tur serve --model ~/models/m-q4k --revision m1-q4k --kv-format q8 --recurrent-state f16 --max-batch-size 2
```

Every term of the plan shrinks. F16 recurrent state costs some accuracy on long
generations and disables speculation; check quality on your own text first (see
[Choosing a quantization](choosing-quantization.md#check-quality-on-your-own-text)).

### Reproducible output

Batched passes can change logits in their last bits, because kernels reduce in different
orders at different batch shapes. Same seed, different neighbours can therefore mean a
different sampled token. Use the library's `PipelineConfig::reproducible`, which runs one
request at a time in fixed chunks, without reuse or speculation. Concurrent bit-exact
batching was measured and rejected as 0.52–0.87× slower.

## Measure the result

- **Per request:** `GenerationStats` and the HTTP `usage` block report prompt, cached and
  generated tokens. The CLI logs elapsed time and finish reason.
- **Under load:** the serving benchmark reports throughput, TTFT and inter-token latency
  percentiles for your configuration:

  ```sh
  TUR_QWEN35_0_8B=$HOME/models/qwen3.5-0.8b TUR_BENCH_WORKLOADS=decode,mixed \
  TUR_BENCH_CONCURRENCY=1,4,8,16 just bench serving > serving.json
  ```

- **Quality:** run `kv_quality` before adopting any K/V, state or weight format (see
  [Benchmarks](../development/benchmarks.md)).

Change one knob at a time, repeat each run at least three times, and compare on the same
Mac in the same power mode.
