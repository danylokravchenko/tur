# Audio references

`tone.wav` is 1.234 s of 16 kHz PCM16: a chirp, a 3 kHz partial and a deterministic hiss. `gemma4/` is a small Gemma 4 model with a two-layer audio encoder, and `whisper/` a small Whisper model, both with deterministic weights. `scripts/audio_reference.py --tiny` generates all of them with Transformers 5.20.2 and torch 2.14.1 on the CPU in F32, together with each model's features, encoder outputs, logits (and, for Gemma, per-layer states with the clip in the prompt) in `golden.safetensors`. They are numerical architecture oracles, not real model checkpoints.

The Gemma encoder's projections use seeded uniform weights of unit gain, and its clip bounds bind on a few percent of activations. Sinusoidal weights made rows nearly cancel, and the encoder's per-row norms then amplified rounding beyond any useful gate.

Real-checkpoint goldens are generated separately from a WAV file (for example one written by `say -o speech.wav --data-format=LEI16@16000 "..."`), with `--whisper` for `openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9` and `--gemma` for `google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7`. The ignored tests read them from `TUR_WHISPER_GOLDEN` and `TUR_GEMMA4_AUDIO_GOLDEN`, the checkpoints from `TUR_WHISPER` and `TUR_GEMMA4_E2B`, and the clip from `TUR_AUDIO_SPEECH`.
