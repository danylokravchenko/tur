"""Pinned audio references: Transformers 5.20.2, torch 2.14.1, CPU F32.

Tiny fixtures are offline numerical oracles, not evidence that a real model works.
  python -I scripts/audio_reference.py --tiny crates/tur/tests/fixtures/audio
Real checkpoints (immutable revisions), on a WAV file such as one written by `say`:
  python -I scripts/audio_reference.py --whisper <dir> --wav speech.wav --golden /tmp/whisper.safetensors
  python -I scripts/audio_reference.py --gemma <dir> --wav speech.wav --golden /tmp/gemma4-audio.safetensors
"""
import argparse
import json
import wave
from pathlib import Path
import numpy as np
import torch
import transformers
from safetensors.torch import save_file
from transformers import (
    Gemma4AudioFeatureExtractor, Gemma4Config, Gemma4ForConditionalGeneration,
    WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration,
)

GEMMA_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
WHISPER_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
RATE = 16000


def tone():
    """1.234 s of a chirp, a high partial and a deterministic hiss, as PCM16 samples."""
    t = np.arange(int(1.234 * RATE)) / RATE
    x = 0.5 * np.sin(2 * np.pi * (220 + 400 * t) * t) + 0.1 * np.sin(2 * np.pi * 3000 * t)
    x += 0.05 * np.sin(np.arange(t.size) * 12.9898) * np.sin(np.arange(t.size) * 0.013)
    return np.round(np.clip(x, -1, 1) * 32767).astype(np.int16)


def write_wav(path, samples):
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(RATE)
        file.writeframes(samples.tobytes())


def read_wav(path):
    with wave.open(str(path), "rb") as file:
        if file.getsampwidth() != 2 or file.getframerate() != RATE:
            raise ValueError("reference WAV must be 16 kHz PCM16")
        data = np.frombuffer(file.readframes(file.getnframes()), dtype=np.int16)
        return data.reshape(-1, file.getnchannels()).astype(np.float32).mean(axis=1) / 32768.0


def float32(config):
    """The config with every dtype F32. Built in BF16, Transformers rounds non-persistent
    buffers such as the audio encoder's sinusoid timescales, and `.float()` keeps them."""
    return json.loads(json.dumps(config).replace('"bfloat16"', '"float32"').replace('"float16"', '"float32"'))


def deterministic(model, clips=None, fan_in=()):
    """Sinusoidal weights; matrices under a `fan_in` prefix are scaled by 1/sqrt(inputs),
    so deep stacks keep O(1) activations and F32 rounding stays far below the gate."""
    with torch.no_grad():
        for index, (name, value) in enumerate(model.state_dict().items()):
            base = torch.arange(value.numel(), dtype=torch.float64).reshape(value.shape)
            scale = 0.04 if "norm" not in name else 0.08
            data = torch.sin(base * 0.37 + index * 1.7) * scale
            if value.dim() >= 2 and any(name.startswith(p) for p in fan_in):
                # Seeded uniform weights of unit gain: sinusoidal rows nearly cancel, and
                # the encoder's per-row norms would amplify rounding on those rows.
                noise = torch.rand(value.shape, generator=torch.Generator().manual_seed(index), dtype=torch.float64)
                data = (noise - 0.5) * 2 * (3.0 / value[0].numel()) ** 0.5
            if "norm" in name or "scale" in name or "scalar" in name:
                data += 1.0
            for suffix, bound in (clips or {}).items():
                if name.endswith(suffix):
                    data = torch.full_like(data, bound)
            value.copy_(data.to(torch.bfloat16).float())


def tiny_gemma(root, samples):
    text = dict(
        model_type="gemma4_text", hidden_size=64, intermediate_size=128, vocab_size=96,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=64, global_head_dim=512, num_global_key_value_heads=1,
        max_position_embeddings=2048, rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh", attention_bias=False,
        attention_dropout=0.0, attention_k_eq_v=False, tie_word_embeddings=True,
        layer_types=["sliding_attention", "full_attention"] * 2,
        rope_parameters={
            "sliding_attention": dict(rope_type="default", rope_theta=10000.0),
            "full_attention": dict(rope_type="proportional", rope_theta=1000000.0, partial_rotary_factor=0.25),
        }, sliding_window=4, num_kv_shared_layers=2,
        hidden_size_per_layer_input=32, vocab_size_per_layer_input=96,
        use_double_wide_mlp=True, final_logit_softcapping=30.0,
        use_bidirectional_attention=None, enable_moe_block=False,
        num_experts=None, top_k_experts=None, moe_intermediate_size=None,
        eos_token_id=1, bos_token_id=2, pad_token_id=0, dtype="bfloat16",
    )
    audio = dict(
        model_type="gemma4_audio", hidden_size=64, num_attention_heads=2, num_hidden_layers=2,
        subsampling_conv_channels=[128, 8], output_proj_dims=48, conv_kernel_size=5,
        attention_chunk_size=12, attention_context_left=13, attention_context_right=0,
        attention_logit_cap=50.0, attention_invalid_logits_value=-1e9, residual_weight=0.5,
        gradient_clipping=1e10, hidden_act="silu", rms_norm_eps=1e-6, use_clipped_linears=True,
        dtype="bfloat16",
    )
    tokens = dict(audio_token_id=90, boa_token_id=91, eoa_token_id=92, eoa_token_index=92,
                  image_token_id=93, boi_token_id=94, eoi_token_id=95, video_token_id=89)
    config = dict(architectures=["Gemma4ForConditionalGeneration"], model_type="gemma4",
                  text_config=text, audio_config=audio, vision_config=None,
                  tie_word_embeddings=True, eos_token_id=1, **tokens)
    model = Gemma4ForConditionalGeneration(Gemma4Config(**float32(config))).float().eval()
    model.config.text_config._attn_implementation = "eager"
    # Clip bounds that bind on some activations, so clipping is exercised.
    deterministic(model, {"input_min": -2.0, "input_max": 2.0, "output_min": -1.5, "output_max": 1.5},
                  fan_in=("model.audio_tower.", "model.embed_audio."))
    directory = root / "gemma4"
    directory.mkdir(parents=True, exist_ok=True)
    weights = {name: value.to(torch.bfloat16).contiguous() for name, value in model.state_dict().items() if name != "lm_head.weight"}
    save_file(weights, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    waveform = samples.astype(np.float32) / 32768.0
    features = Gemma4AudioFeatureExtractor()([waveform], return_tensors="pt")
    with torch.inference_mode():
        audio_out = model.model.get_audio_features(features["input_features"], features["input_features_mask"], return_dict=True)
    mask = audio_out.attention_mask[0]
    soft = audio_out.pooler_output[0][mask]
    count = int(mask.sum())
    ids = torch.tensor([[2, 7, 5, 91] + [90] * count + [92, 9, 3]])
    data = {
        "features": features["input_features"][0].float().contiguous(),
        "features_mask": features["input_features_mask"][0].to(torch.uint8).contiguous(),
        "tower": audio_out.last_hidden_state[0][mask].float().contiguous(),
        "soft_tokens": soft.float().contiguous(),
        "tokens": ids.to(torch.int64),
    }
    handles = []
    for index, layer in enumerate(model.model.language_model.layers):
        def record(_module, _inputs, result, index=index):
            data[f"layer.{index}"] = result.detach().float().squeeze(0).contiguous()
        handles.append(layer.register_forward_hook(record))
    with torch.inference_mode():
        result = model(input_ids=ids, input_features=features["input_features"], input_features_mask=features["input_features_mask"], use_cache=False)
    for handle in handles:
        handle.remove()
    data["logits"] = result.logits.squeeze(0).float().contiguous()
    save_file(data, directory / "golden.safetensors", metadata={"transformers": transformers.__version__, "torch": torch.__version__, "revision": "synthetic-gemma4-audio-v1"})


def tiny_whisper(root, samples):
    config = dict(
        architectures=["WhisperForConditionalGeneration"], model_type="whisper",
        vocab_size=64, num_mel_bins=128, d_model=64, encoder_layers=2, encoder_attention_heads=4,
        decoder_layers=2, decoder_attention_heads=4, encoder_ffn_dim=128, decoder_ffn_dim=128,
        max_source_positions=1500, max_target_positions=32, activation_function="gelu",
        scale_embedding=False, tie_word_embeddings=True, decoder_start_token_id=3,
        pad_token_id=1, bos_token_id=2, eos_token_id=1, dtype="float16",
    )
    model = WhisperForConditionalGeneration(WhisperConfig(**float32(config))).float().eval()
    model.config._attn_implementation = "eager"
    deterministic(model)
    directory = root / "whisper"
    directory.mkdir(parents=True, exist_ok=True)
    weights = {name: value.to(torch.float16).contiguous() for name, value in model.state_dict().items() if name != "proj_out.weight"}
    save_file(weights, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    waveform = samples.astype(np.float32) / 32768.0
    extractor = WhisperFeatureExtractor(feature_size=128)
    features = torch.from_numpy(extractor(waveform, sampling_rate=RATE, return_tensors="np")["input_features"])
    ids = torch.tensor([[3, 5, 7, 9, 11]])
    with torch.inference_mode():
        encoded = model.model.encoder(features).last_hidden_state
        logits = model(input_features=features, decoder_input_ids=ids).logits
        greedy = [3, 5]
        for _ in range(6):
            step = model(encoder_outputs=(encoded,), decoder_input_ids=torch.tensor([greedy])).logits
            greedy.append(int(step[0, -1].argmax()))
    data = {
        "features": features[0].float().contiguous(),
        "encoded": encoded[0].float().contiguous(),
        "tokens": ids.to(torch.int64),
        "logits": logits[0].float().contiguous(),
        "greedy": torch.tensor(greedy, dtype=torch.int64),
    }
    save_file(data, directory / "golden.safetensors", metadata={"transformers": transformers.__version__, "torch": torch.__version__, "revision": "synthetic-whisper-v1"})


def tiny(root):
    root.mkdir(parents=True, exist_ok=True)
    samples = tone()
    write_wav(root / "tone.wav", samples)
    tiny_gemma(root, samples)
    tiny_whisper(root, samples)


def real_whisper(directory, wav, output):
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained(directory, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(directory, local_files_only=True, dtype=torch.float32).eval()
    waveform = read_wav(wav)
    features = processor(waveform, sampling_rate=RATE, return_tensors="pt").input_features
    with torch.inference_mode():
        generated = model.generate(features, language="en", task="transcribe", do_sample=False, num_beams=1)
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    save_file({"features": features[0].contiguous(), "greedy": generated[0].contiguous()}, output,
              metadata={"transformers": transformers.__version__, "torch": torch.__version__, "revision": WHISPER_REVISION, "text": text})
    print(f"wrote {output}: {text!r}", flush=True)


def real_gemma(directory, wav, output):
    """The real audio tower and projector in F32; the 5B text tower does not fit beside them on a 16 GB Mac."""
    from safetensors import safe_open
    from transformers import AutoProcessor, Gemma4AudioConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel, Gemma4MultimodalEmbedder
    raw = float32(json.loads((directory / "config.json").read_text()))
    audio_config = Gemma4AudioConfig(**raw["audio_config"])
    tower = Gemma4AudioModel(audio_config).float().eval()
    embedder = Gemma4MultimodalEmbedder(audio_config, Gemma4Config(**raw).text_config).float().eval()
    with safe_open(directory / "model.safetensors", framework="pt", device="cpu") as file:
        for module, prefix in ((tower, "model.audio_tower."), (embedder, "model.embed_audio.")):
            state = {name: file.get_tensor(prefix + name).float() for name in module.state_dict()}
            module.load_state_dict(state, strict=True)
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    messages = [{"role": "user", "content": [{"type": "audio", "audio": read_wav(wav)}, {"type": "text", "text": "Transcribe this audio."}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=False)
    with torch.inference_mode():
        audio = tower(inputs["input_features"], inputs["input_features_mask"], return_dict=True)
        soft = embedder(audio.last_hidden_state)[0][audio.attention_mask[0]]
    save_file({"tokens": inputs["input_ids"][0].contiguous(), "soft_tokens": soft.float().contiguous()}, output,
              metadata={"transformers": transformers.__version__, "torch": torch.__version__, "revision": GEMMA_REVISION})
    print(f"wrote {output}: {soft.shape[0]} soft tokens, {inputs['input_ids'].shape[-1]} prompt tokens", flush=True)


def main():
    if transformers.__version__ != "5.20.2" or torch.__version__.split("+")[0] != "2.14.1":
        raise RuntimeError("use pinned transformers 5.20.2 and torch 2.14.1")
    torch.set_num_threads(4)
    args = argparse.ArgumentParser()
    args.add_argument("--tiny", type=Path)
    args.add_argument("--whisper", type=Path)
    args.add_argument("--gemma", type=Path)
    args.add_argument("--wav", type=Path)
    args.add_argument("--golden", type=Path)
    options = args.parse_args()
    if options.tiny:
        tiny(options.tiny)
    elif options.whisper and options.wav and options.golden:
        real_whisper(options.whisper, options.wav, options.golden)
    elif options.gemma and options.wav and options.golden:
        real_gemma(options.gemma, options.wav, options.golden)
    else:
        args.error("provide --tiny, or --whisper/--gemma with --wav and --golden")


if __name__ == "__main__":
    main()
