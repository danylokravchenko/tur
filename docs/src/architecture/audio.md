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

The server encodes clips on a dedicated thread with its own Metal queue, so text passes keep running while a clip encodes, then hands the request to the worker with the rows attached. A stream sharing the GPU with a 30-second clip keeps its median pass time. The rows stay with the request until it finishes, so one that is preempted and recomputed never encodes its clip again.

### Prefix reuse with audio

Prefix reuse compares tokens, and every position of a clip carries the same placeholder, so two different clips of the same length would look identical. Tur gives each audio position an identity token above the vocabulary, hashed from the clip's samples and the position's index. Prefix keys and restore checks then tell clips apart: a repeated request with the same clip restores its whole prompt, and a different clip shares only the text before it. Packing maps identity tokens back to the pad token and attaches the encoder rows as `EmbeddedInput`s of the packed pass.

## Whisper transcribes the clip

Whisper is an encoder-decoder model with cross-attention, a shape the paged hybrid engine doesn't run, so it gets its own runtime. A `TranscriptionWorker` owns it on a dedicated thread and transcribes one clip at a time from a bounded queue.

- The encoder reads one 30-second window of 128 mel features and returns 1500 states. Its self-attention runs on the tiled kernel that long-prompt prefill uses, reading the projected keys and values in place as a single page of a K/V pool; projections, norms and convolutions run on Candle.
- Each decoder layer writes its own keys and values into a preallocated pool, so a step costs nearly the same at token 270 as at token 30.
- The decoder starts from `<|startoftranscript|>`, the language and the task, and decodes greedily with the checkpoint's suppressed tokens, keeping a key-value cache of its own tokens and the encoder states' cross-attention keys and values.
- Without a `language`, the first window picks the most likely language token, as Transformers' `detect_language` does.

### Long audio

A clip of up to 30 seconds is one window, and a plain transcript adds `<|notimestamps|>` to the prompt. A longer clip, or a request for timed output, decodes with timestamp tokens instead. They arrive in pairs that close one segment and open the next, under the rules of Transformers' `WhisperTimeStampLogitsProcessor`.

Timestamps drive the seek. A window usually ends partway through a sentence, so Tur keeps the segments that closed, drops the text after the last pair, and starts the next window at that timestamp. A word at a window's edge is therefore decoded again, whole, in the next window. A window ending in a single timestamp has no speech after it, and the seek moves past the whole window.

### Prompts, retries and word timings

A `prompt` is read by the decoder before every window, which helps it spell names and match a style. With `condition_on_previous_text` it also reads the tokens decoded so far, up to half its positions: this follows context across windows, and also carries a mistake forward.

A window can go wrong two ways: the decoder loops on the same phrase, or it guesses through unclear audio. A loop compresses unusually well, and a guess has a low average log-probability, so the server and CLI retry such a window at rising temperatures, as OpenAI's Whisper does. A window that retries at 0.5 or above is not read by the next one.

Word timings come from the decoder's cross-attention, which tracks the audio as it writes. A few heads named by the checkpoint are recorded per token, smoothed, and aligned to the audio by dynamic time warping, and tokens are then grouped into words with their punctuation. The fused attention keeps no weights, so recording computes these heads separately, and only when word timings are requested.

### Silence

Whisper can confidently invent text over silence: on a clip with a 70-second gap, it wrote "Thank you. Thank you." there, and so does Transformers. Two checks guard against this. A window whose loudest 20 ms stays below about -50 dBFS is not encoded or decoded at all, which also saves its 2 seconds of encoder time. Otherwise, a window with a high `<|nospeech|>` probability and a low average log-probability yields no text, the rule OpenAI's Whisper and Transformers apply.

Features for a long clip are computed once over the whole clip and normalized by its loudest frame, so a quiet stretch keeps its level relative to the rest. The last window is padded with zeros in feature space.

The worker leases its weights and one encoder pass from the device budget before the chat model sizes its state pool, so both fit the same memory plan.

## Decisions

- Spectrograms run on the host with an F64 FFT over frames windowed in F32, as Transformers computes them. Gemma's `ln(mel + 0.001)` turns tiny energy errors in near-silent bins into large feature errors: an F32 transform missed the reference by 1.1e-4 of the feature range, and this order matches it to 5e-8. Whisper's reference uses torch's F32 STFT, and Tur's features land within 3e-5 of it.
- The Gemma encoder computes Transformers' blocked attention as dense attention over the band of 12 keys before each query. With the shipped settings the two are the same function, and the dense form needs no block bookkeeping.
- Frames that Transformers pads and masks are dropped before the encoder. Every layer either looks back in time or masks them, so the valid outputs are unchanged.
- Conformance comes from pinned references. On small Transformers 5.20.2 fixtures, Gemma's soft tokens match within 1.3e-6 relative error and every text layer with audio within 4e-6, on CPU and Metal. Whisper's encoder states and logits pass the 1e-4 gate, and its greedy decode through the cache reproduces the reference tokens.
- Encoder projections are `Q8_0` beside a packed text model and F32 beside an F32 one. On the real E2B checkpoint, `Q8_0` soft tokens agree with the F32 Transformers tower at a mean cosine similarity of 0.99994 (minimum 0.99965) on a 5.3-second clip, which took 267 ms to encode.
- Whisper large-v3-turbo in `Q8_0` and in F32 produces the same transcript as Transformers on that clip. `Q8_0` is the default: it holds about 0.85 GB of weights instead of 3.2 GB.
- Long-form decoding follows Transformers' `generate`: F32 reproduces its tokens and segment times exactly on an 80-second clip. `Q8_0` weights shift close decisions, giving 24 segments instead of 15, with one word of 239 differing in punctuation.
- Whisper's attention uses Tur's kernels, because Candle's matmuls and softmax take 124 ms per layer on the encoder's shape and the tiled kernel 13 ms. One window's encoder went from 2.8 s to 2.0 s on the M1, and transcribing a short clip from 3.5 s to 2.3 s. Projections keep Candle's GEMM, the fastest measured at these shapes.
- Gemma's audio encoder and Whisper each own a Metal device, because one queue runs first-in, first-out: on a shared queue a 30-second clip's 1.1 s of GPU work would hold every later text pass behind it. On separate queues the slowest text pass is 110 to 165 ms during that encode, against 110 ms alone, which itself takes 2.7 s instead of 1.45 s while sharing the GPU. During an 80-second transcription, a stream's longest pause was 0.45 s. Direct `InferencePipeline` callers without a server still encode at submission, and wait for it.
- WAV is the only container. MP3, FLAC and Opus decoders would be new dependencies.
- Artifacts carry the audio encoder, its projections packed in `Q8_0` like the text tower's (F32 beside an F32 plan), which adds about 0.3 GB. Artifacts written before this hold the text tower only and still load, rejecting audio.
- Prompts, conditioning and token times follow Transformers: on the 80-second clip in F32, prompted and conditioned decodes equal its tokens exactly, and at least 95% of word starts land within one encoder state (20 ms) of its token times.
- Beside `Q8_0` weights, the decoder's output head is Q4K and its cross-attention keys and values F16, because those are its largest memory reads. With paged self-attention, a step went from 6.2 to 4.6 ms at 30 tokens and from 9.2 to about 5.8 ms at 270, and an 80-second clip from 9.06 to 8.5 s; encoder passes dominate the rest. The Q4K head adds about 0.5 s to loading. F32 weights keep F32 everywhere, as the reference comparison needs.
- Silence detection follows OpenAI's thresholds (no-speech probability 0.6, average log-probability -1) and matches Transformers token for token on a clip with a 70-second gap, invented text included. The energy gate is Tur's addition and removes that text; `Transcriber` leaves both off unless a `Fallback` asks for them, and `tur serve` and `tur transcribe` turn them on.
- Fallback uses OpenAI's defaults: temperatures 0 to 1 in steps of 0.2, compression ratio 2.4 and average log-probability -1. Transformers leaves it off by default, so `Transcriber` does too and the reference tests compare single greedy attempts; `tur serve` and `tur transcribe` turn it on. A window's retries use a seed derived from its position, so a run repeats exactly.
