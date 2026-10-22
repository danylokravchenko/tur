# Tiled Metal attention

The paged attention path now has an F32 matrix-unit kernel for prefill and chunked
prefill. It supports head dimensions 64, 128 and 256, GQA, F32/F16/Q8 K/V storage,
and the existing sigmoid output gate. It reads the persistent page pools directly
and never materializes the attention matrix.

Each 128-thread threadgroup handles up to 32 consecutive queries of one sequence
and one head. Its four simdgroups own eight query rows each. Keys are swept in
32-row blocks; K and V are staged separately in 64-dimension chunks, dequantized
once per threadgroup. Online softmax and output accumulators remain F32. Staging
and physical slot indices use 8,320 bytes of threadgroup memory.

Every loop indexing a matrix-fragment array requests full unrolling, including
the outer dimension-chunk loops. Q fragments are loaded as needed, keeping their
lifetime short instead of retaining both full Q and output fragment arrays.
The timing improvement is consistent with the suspected register spills; no
hardware spill counters were collected.

Partial query tiles clamp each lane's query row before loading its fragment.
Clamping the starting row of an eight-row matrix load is insufficient: subsequent
rows can still extend beyond the query allocation. Masked key rows are staged as
zero, and causal limits are applied before softmax.

Default dispatch requires at least 16 threadgroups (query tiles times heads) and
an average of at least 16 queries per tile. This preserves split-K parallelism
for decode and avoids wasting matrix work on sparse query tiles. The M1 sweep
places the crossover at 64 queries with eight heads. Tuning value 0 forces tiled
execution for supported layouts; `usize::MAX` selects the per-query path. Runs
with non-consecutive positions sharing a page table use the per-query path.

## Reproduction and scope

```sh
cargo nextest run --offline --release -p tur-metal --test-threads 1
MTL_SHADER_VALIDATION=1 cargo test --offline --release -p tur-metal --test kernels tiled_attention
cargo nextest run --offline --release -p tur-engine --test qwen3_5_fused --test-threads 1
TUR_BENCH_ONLY=tiled cargo bench --offline -p tur-metal --bench kernels > tiled.json
```

`TUR_BENCH_SHAPE=<substring>` restricts the tiled benchmark to matching shapes.
Benchmark stdout is JSON; timing summaries go to stderr. Each result includes
the kernel selected by default tuning as well as forced tiled and per-query
timings. Timing uses completed GPU work, three warmups and 15 samples with
adaptive repetitions, as in the other kernel benchmarks.

The artifacts are [before](baselines/2026-10-22-m1-tiled-attention/before.json)
and [after](baselines/2026-10-22-m1-tiled-attention/after.json). The baseline is
the unfinished rewrite present at the start of this change, including its
existing unroll hints. These are synthetic kernel measurements on the 16 GB
Apple M1, with eight query heads, two K/V heads and head dimension 256. They do
not measure model or serving throughput, and other devices remain unmeasured.

Differential coverage includes every supported head dimension and Q8 group size,
whole and partial tiles, unaligned key counts, shuffled pages whose unused rows
contain NaN, multiple sequences, different GQA ratios, page sizes 1/7/16/32/64,
and a continuation after 2,047 keys. The bound is maximum absolute error divided
by the reference's maximum magnitude, below 1e-5 against scalar F64 attention
over the values actually stored in each K/V format. Dispatch tests cover decode,
the measured crossover and unsupported dimensions. A 513-token fused-model
layer/logit comparison exercises default tiled dispatch with a partial final tile.

Final validation: [47 tests passed](baselines/2026-10-22-m1-tiled-attention/nextest.log)
(36 `tur-metal` tests and 11 fused-model regressions); the three existing real-checkpoint
tests remained ignored. Both tiled tests also passed with
[Metal shader validation](baselines/2026-10-22-m1-tiled-attention/shader-validation.log).
Clippy with warnings denied, formatting and whitespace checks passed.

## M1 timing results

Times are median milliseconds per attention call. The speedup column compares the
finished tiled kernel with the per-query kernel measured in the same sweep.
Default dispatch retains per-query execution for decode and the smaller shapes.

| Shape | K/V | Rewrite before | Tiled after | Per-query | Speedup | Default |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| prefill 16 | f32 | 0.128 | 0.040 | 0.014 | 0.35× | per_query |
| prefill 16 | f16 | 0.129 | 0.041 | 0.014 | 0.35× | per_query |
| prefill 16 | q8_32 | 0.134 | 0.043 | 0.019 | 0.45× | per_query |
| prefill 32 | f32 | 0.106 | 0.048 | 0.034 | 0.70× | per_query |
| prefill 32 | f16 | 0.108 | 0.050 | 0.054 | 1.06× | per_query |
| prefill 32 | q8_32 | 0.111 | 0.054 | 0.044 | 0.81× | per_query |
| prefill 64 | f32 | 0.270 | 0.085 | 0.099 | 1.15× | tiled |
| prefill 64 | f16 | 0.268 | 0.090 | 0.095 | 1.05× | tiled |
| prefill 64 | q8_32 | 0.275 | 0.099 | 0.144 | 1.46× | tiled |
| prefill 128 | f32 | 0.979 | 0.205 | 0.341 | 1.67× | tiled |
| prefill 128 | f16 | 0.966 | 0.210 | 0.328 | 1.56× | tiled |
| prefill 128 | q8_32 | 0.971 | 0.224 | 0.518 | 2.31× | tiled |
| prefill 512 | f32 | 17.826 | 1.914 | 4.868 | 2.54× | tiled |
| prefill 512 | f16 | 17.478 | 1.953 | 4.731 | 2.42× | tiled |
| prefill 512 | q8_32 | 17.605 | 2.110 | 7.541 | 3.57× | tiled |
| prefill 2048 | f32 | 249.672 | 26.744 | 78.674 | 2.94× | tiled |
| prefill 2048 | f16 | 240.358 | 26.209 | 74.504 | 2.84× | tiled |
| prefill 2048 | q8_32 | 239.120 | 29.499 | 118.877 | 4.03× | tiled |
| prefill 8192 | f32 | 3969.261 | 422.856 | 1235.221 | 2.92× | tiled |
| prefill 8192 | f16 | 3780.729 | 428.877 | 1187.428 | 2.77× | tiled |
| prefill 8192 | q8_32 | 3751.572 | 483.444 | 1872.617 | 3.87× | tiled |
| chunk 16 after 2048 | f32 | 5.281 | 2.015 | 1.369 | 0.68× | per_query |
| chunk 16 after 2048 | f16 | 5.352 | 2.037 | 1.264 | 0.62× | per_query |
| chunk 16 after 2048 | q8_32 | 5.501 | 2.221 | 1.919 | 0.86× | per_query |
| chunk 64 after 2048 | f32 | 8.871 | 2.240 | 5.372 | 2.40× | tiled |
| chunk 64 after 2048 | f16 | 8.853 | 2.219 | 4.937 | 2.22× | tiled |
| chunk 64 after 2048 | q8_32 | 9.054 | 2.516 | 7.711 | 3.07× | tiled |
| chunk 256 after 8192 | f32 | 221.244 | 28.573 | 83.320 | 2.92× | tiled |
| chunk 256 after 8192 | f16 | 215.519 | 28.746 | 76.995 | 2.68× | tiled |
| chunk 256 after 8192 | q8_32 | 214.617 | 31.765 | 120.710 | 3.80× | tiled |
| chunk 32 x4 after 1024 | f32 | 12.400 | 2.306 | 5.038 | 2.19× | tiled |
| chunk 32 x4 after 1024 | f16 | 11.160 | 2.327 | 4.759 | 2.05× | tiled |
| chunk 32 x4 after 1024 | q8_32 | 11.114 | 2.642 | 7.619 | 2.88× | tiled |
| decode ctx 2048 x8 | f32 | 58.384 | 6.529 | 1.301 | 0.20× | per_query |
| decode ctx 2048 x8 | f16 | 55.586 | 6.418 | 0.707 | 0.11× | per_query |
| decode ctx 2048 x8 | q8_32 | 55.348 | 7.264 | 1.005 | 0.14× | per_query |
