# Audio

Tur takes speech two ways. Gemma 4 E2B and E4B carry their own audio encoder, so a chat request can include a clip and the model hears it directly. Whisper large-v3-turbo runs beside the chat model and turns uploaded speech into text at `/v1/audio/transcriptions`. Qwen3.5 checkpoints have no audio encoder and reject audio input.

Both paths start the same way. `tur_models::audio::Audio` decodes a WAV file (8 to 32-bit integer PCM or 32 and 64-bit float, any channel count), mixes it to mono and resamples it to 16 kHz with a windowed-sinc filter. The server never fetches media from URLs; clips arrive as base64 or as an uploaded file.

## Gemma 4 hears the clip

Gemma's chat template renders an audio part as one `<|audio|>` placeholder. Tur expands it to `<|audio>`, one position per 40 ms of audio, and `<audio|>`, as Transformers' processor does. A 30-second clip becomes 750 positions, which is the longest clip the encoder accepts.

The encoder is a 12-layer conformer, about 300 million parameters, from Google's Universal Speech Model family:

- 128 log-mel features every 10 ms pass through two stride-2 convolutions, which cut 100 frames per second to 25.
- Each layer runs a feed-forward block, local self-attention, a gated depthwise convolution and a second feed-forward block. Attention looks at the current frame and the 11 before it, with a relative position term for each distance.
- A projection maps the encoder's 1536-wide output into the text model's embedding space.

Each soft token then replaces the input embedding at its position. The text model sees its pad token there for everything else it reads per token, which is what Transformers feeds Gemma's per-layer embeddings. Everything after the embedding (the decoder layers, paged K/V, scheduling, sampling and speculation) runs as for text, so a clip costs one prefill of its soft tokens and nothing per generated token.

The encoder runs when a request is submitted, on the worker thread, and its output stays with the request until it finishes. A request that is preempted and recomputed reuses the rows instead of encoding the clip again.

### Prefix reuse with audio

Prefix reuse compares tokens, and every position of a clip carries the same placeholder, so two different clips of the same length would look identical. Tur gives each audio position an identity token above the vocabulary, hashed from the clip's samples and the position's index. Prefix keys and restore checks then tell clips apart: a repeated request with the same clip restores its whole prompt, and a different clip shares only the text before it. Packing maps identity tokens back to the pad token and attaches the encoder rows as `EmbeddedInput`s of the packed pass.

## Whisper transcribes the clip

Whisper is an encoder-decoder model with cross-attention, a shape the paged hybrid engine doesn't run, so it gets its own runtime on Candle tensor operations. A `TranscriptionWorker` owns it on a dedicated thread and transcribes one clip at a time from a bounded queue.

- The encoder reads one 30-second window of 128 mel features and returns 1500 states.
- The decoder starts from `<|startoftranscript|>`, the language, the task and `<|notimestamps|>`, and decodes greedily with the checkpoint's suppressed tokens, keeping a key-value cache of its own tokens and the encoder states' cross-attention keys and values.
- Without a `language`, the first window picks the most likely language token, as Transformers' `detect_language` does.
- Longer clips are cut into consecutive 30-second windows. A word that straddles a boundary can be split, since Whisper's timestamp-guided seeking isn't implemented.

The worker leases its weights and one encoder pass from the device budget before the chat model sizes its state pool, so both fit the same memory plan.

## Decisions

- Spectrograms run on the host with an F64 FFT over frames windowed in F32, as Transformers computes them. Gemma's `ln(mel + 0.001)` turns tiny energy errors in near-silent bins into large feature errors: an F32 transform missed the reference by 1.1e-4 of the feature range, and this order matches it to 5e-8. Whisper's reference uses torch's F32 STFT, and Tur's features land within 3e-5 of it.
- The Gemma encoder computes Transformers' blocked attention as dense attention over the band of 12 keys before each query. With the shipped settings the two are the same function, and the dense form needs no block bookkeeping.
- Frames that Transformers pads and masks are dropped before the encoder. Every layer either looks back in time or masks them, so the valid outputs are unchanged.
- Conformance comes from pinned references. On small Transformers 5.20.2 fixtures, Gemma's soft tokens match within 1.3e-6 relative error and every text layer with audio within 4e-6, on CPU and Metal. Whisper's encoder states and logits pass the 1e-4 gate, and its greedy decode through the cache reproduces the reference tokens.
- Encoder projections are `Q8_0` beside a packed text model and F32 beside an F32 one. On the real E2B checkpoint, `Q8_0` soft tokens agree with the F32 Transformers tower at a mean cosine similarity of 0.99994 (minimum 0.99965) on a 5.3-second clip, which took 267 ms to encode.
- Whisper large-v3-turbo in `Q8_0` and in F32 produces the same transcript as Transformers on that clip. `Q8_0` is the default: it holds 0.88 GB of weights instead of 3.2 GB.
- WAV is the only container. MP3, FLAC and Opus decoders would be new dependencies.
- Prepacked artifacts hold the text tower only, so a model loaded from one has no audio encoder and rejects audio.
