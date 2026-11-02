"""Pinned image references: Transformers 5.20.2, torch 2.14.1, torchvision 0.29.2, CPU F32.

Tiny fixtures are offline numerical oracles, not evidence that a real model works.
  python -I scripts/vision_reference.py --tiny crates/tur/tests/fixtures/vision
Real checkpoints (immutable revisions), on any image file:
  python -I scripts/vision_reference.py --qwen <dir> --image photo.png --golden /tmp/qwen-image.safetensors
  python -I scripts/vision_reference.py --gemma <dir> --image photo.png --golden /tmp/gemma-image.safetensors
Add --video clip.mp4 (H.264, decoded with PyAV) to either for a video answer or tower outputs instead.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import transformers
from PIL import Image
from safetensors.torch import save_file
from transformers import (
    Gemma4Config, Gemma4ForConditionalGeneration, Gemma4ImageProcessor,
    Qwen2VLImageProcessor, Qwen3_5Config, Qwen3_5ForConditionalGeneration,
)

QWEN_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
GEMMA_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
PROMPT = "Describe this image in one sentence."


def pattern(width, height):
    """Colour ramps with a diagonal stripe and a deterministic hiss, as RGB bytes."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    image = np.stack([
        255 * x / max(width - 1, 1),
        255 * y / max(height - 1, 1),
        127.5 + 127.5 * np.sin((x + 2 * y) * 0.37),
    ], axis=-1)
    image += 18 * np.sin(np.arange(width * height).reshape(height, width, 1) * 12.9898)
    return Image.fromarray(np.round(np.clip(image, 0, 255)).astype(np.uint8), "RGB")


def moving(frames, width, height):
    """A pattern that slides, as `[frames, height, width, 3]` RGB bytes."""
    base = np.array(pattern(width + frames * 3, height))
    return np.stack([base[:, 3 * t:3 * t + width] for t in range(frames)])


def write_clip(path, frames, fps):
    """An H.264 MP4 with B-frames, and what PyAV decodes from it."""
    import av
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return read_clip(path)[0]


def read_clip(path):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(stream)])
        return frames, float(stream.average_rate)


def float32(config):
    return json.loads(json.dumps(config).replace('"bfloat16"', '"float32"'))


def deterministic(model, fan_in=()):
    """Sinusoidal weights, with unit-gain seeded matrices under a `fan_in` prefix, so stacks
    keep O(1) activations and F32 rounding stays far below the gate."""
    with torch.no_grad():
        for index, (name, value) in enumerate(model.state_dict().items()):
            base = torch.arange(value.numel(), dtype=torch.float64).reshape(value.shape)
            data = torch.sin(base * 0.37 + index * 1.7) * (0.04 if "norm" not in name else 0.08)
            if value.dim() >= 2 and any(name.startswith(p) for p in fan_in):
                noise = torch.rand(value.shape, generator=torch.Generator().manual_seed(index), dtype=torch.float64)
                data = (noise - 0.5) * 2 * (3.0 / value[0].numel()) ** 0.5
            if "norm" in name:
                data += 1.0
            value.copy_(data.to(torch.bfloat16).float())


def record_layers(layers, data):
    handles = []
    for index, layer in enumerate(layers):
        def record(_module, _inputs, result, index=index):
            state = result[0] if isinstance(result, tuple) else result
            data[f"layer.{index}"] = state.detach().float().squeeze(0).contiguous()
        handles.append(layer.register_forward_hook(record))
    return handles


def write(directory, model, config, processor, data):
    directory.mkdir(parents=True, exist_ok=True)
    weights = {n: v.to(torch.bfloat16).contiguous() for n, v in model.state_dict().items() if n != "lm_head.weight"}
    save_file(weights, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (directory / "preprocessor_config.json").write_text(json.dumps(processor, indent=2) + "\n")
    save_file(data, directory / "golden.safetensors", metadata={"transformers": transformers.__version__, "torch": torch.__version__})


def tiny_qwen(root, small, large):
    half = dict(rope_type="default", rope_theta=10000.0, partial_rotary_factor=0.25, mrope_section=[3, 3, 2], mrope_interleaved=True)
    text = dict(
        model_type="qwen3_5_text", hidden_size=64, intermediate_size=128, vocab_size=96, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64, hidden_act="silu", max_position_embeddings=2048,
        rms_norm_eps=1e-6, attention_bias=False, attention_dropout=0.0, attn_output_gate=True, tie_word_embeddings=True,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4, linear_key_head_dim=128, linear_num_key_heads=2, linear_value_head_dim=128,
        linear_num_value_heads=2, mtp_num_hidden_layers=0, rope_parameters=half, eos_token_id=1, pad_token_id=0, dtype="bfloat16",
    )
    vision = dict(
        model_type="qwen3_5", depth=2, hidden_size=128, hidden_act="gelu_pytorch_tanh", intermediate_size=256,
        num_heads=2, in_channels=3, patch_size=16, spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
        num_position_embeddings=64, deepstack_visual_indexes=[], dtype="bfloat16",
    )
    tokens = dict(image_token_id=90, video_token_id=95, vision_start_token_id=92, vision_end_token_id=93)
    config = dict(architectures=["Qwen3_5ForConditionalGeneration"], model_type="qwen3_5", text_config=text, vision_config=vision, tie_word_embeddings=True, **tokens)
    settings = dict(size=dict(shortest_edge=4096, longest_edge=65536), patch_size=16, temporal_patch_size=2, merge_size=2,
                    image_mean=[0.5] * 3, image_std=[0.5] * 3, image_processor_type="Qwen2VLImageProcessorFast")
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**float32(config))).float().eval()
    deterministic(model, fan_in=("model.visual.", "model.language_model.layers"))
    data = {}
    processor = Qwen2VLImageProcessor(**{k: v for k, v in settings.items() if k != "image_processor_type"})
    for name, image in (("small", small), ("large", large)):
        out = processor(images=[image], return_tensors="pt")
        data[f"{name}.pixels"] = out["pixel_values"].float().contiguous()
        data[f"{name}.grid"] = out["image_grid_thw"][0].to(torch.int64)
    grid = out_small = processor(images=[small], return_tensors="pt")
    with torch.inference_mode():
        vision_out = model.model.visual(grid["pixel_values"], grid_thw=grid["image_grid_thw"])
    data["tower"] = vision_out.last_hidden_state.float().contiguous()
    soft = vision_out.pooler_output.float().contiguous()
    data["soft_tokens"] = soft
    # Text before, the clip of soft tokens, text after: rotary positions restart from the grid.
    ids = torch.tensor([[2, 7, 92] + [90] * soft.shape[0] + [93, 9, 3, 5, 6]])
    types = (ids == 90).long()
    with torch.inference_mode():
        positions, _ = model.model.get_rope_index(ids, mm_token_type_ids=types, image_grid_thw=grid["image_grid_thw"])
    data["tokens"] = ids[0].to(torch.int64)
    data["rotary"] = positions[:, 0].t().to(torch.int64).contiguous()
    handles = record_layers(model.model.language_model.layers, data)
    with torch.inference_mode():
        result = model(input_ids=ids, mm_token_type_ids=types, use_cache=False, **out_small)
    for handle in handles:
        handle.remove()
    data["logits"] = result.logits.squeeze(0).float().contiguous()
    tiny_qwen_video(model, data)
    write(root / "qwen3_5", model, config, settings, data)
    video = dict(settings, size=dict(shortest_edge=4096, longest_edge=262144), video_processor_type="Qwen3VLVideoProcessor")
    video.pop("image_processor_type")
    (root / "qwen3_5" / "video_preprocessor_config.json").write_text(json.dumps(video, indent=2) + "\n")


def tiny_qwen_video(model, data):
    from transformers import Qwen3VLVideoProcessor
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.video_utils import VideoMetadata
    frames = moving(10, 61, 93)
    processor = Qwen3VLVideoProcessor(size=dict(shortest_edge=4096, longest_edge=262144), patch_size=16, temporal_patch_size=2,
                                      merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3, cap_pixels_per_frame=False)
    metadata = VideoMetadata(total_num_frames=10, fps=5.0, width=61, height=93, frames_indices=list(range(10)), duration=2.0)
    out = processor(videos=[frames], video_metadata=[metadata], return_metadata=True, return_tensors="pt")
    t, h, w = out["video_grid_thw"][0].tolist()
    indices = out["video_metadata"][0].frames_indices
    stamps = Qwen3VLProcessor._calculate_timestamps(None, list(indices), 5.0, 2)
    with torch.inference_mode():
        vision = model.model.visual(out["pixel_values_videos"], grid_thw=out["video_grid_thw"])
    soft = vision.pooler_output.float().contiguous()
    per_frame = h * w // 4
    # Label positions stand in for a tokenized timestamp.
    tokens = [2, 92]
    for _ in range(t):
        tokens += [5, 6, 92] + [95] * per_frame + [93]
    tokens += [93, 9]
    ids = torch.tensor([tokens])
    types = (ids == 95).long() * 2
    with torch.inference_mode():
        positions, _ = model.model.get_rope_index(ids, mm_token_type_ids=types, video_grid_thw=out["video_grid_thw"])
        result = model(input_ids=ids, mm_token_type_ids=types, pixel_values_videos=out["pixel_values_videos"], video_grid_thw=out["video_grid_thw"], use_cache=False)
    data.update({
        "video.frames": torch.from_numpy(frames).contiguous(),
        "video.pixels": out["pixel_values_videos"].float().contiguous(),
        "video.grid": out["video_grid_thw"][0].to(torch.int64),
        "video.indices": torch.tensor(list(indices), dtype=torch.int64),
        "video.stamps": torch.tensor(stamps, dtype=torch.float32),
        "video.tower": vision.last_hidden_state.float().contiguous(),
        "video.soft_tokens": soft,
        "video.tokens": ids[0].to(torch.int64),
        "video.rotary": positions[:, 0].t().to(torch.int64).contiguous(),
        "video.logits": result.logits.squeeze(0).float().contiguous(),
    })


def tiny_gemma(root, small, large):
    text = dict(
        model_type="gemma4_text", hidden_size=64, intermediate_size=128, vocab_size=96, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64, global_head_dim=512, num_global_key_value_heads=1,
        max_position_embeddings=2048, rms_norm_eps=1e-6, hidden_activation="gelu_pytorch_tanh", attention_bias=False,
        attention_dropout=0.0, attention_k_eq_v=False, tie_word_embeddings=True,
        layer_types=["sliding_attention", "full_attention"] * 2,
        rope_parameters={
            "sliding_attention": dict(rope_type="default", rope_theta=10000.0),
            "full_attention": dict(rope_type="proportional", rope_theta=1000000.0, partial_rotary_factor=0.25),
        }, sliding_window=4, num_kv_shared_layers=2, hidden_size_per_layer_input=32, vocab_size_per_layer_input=96,
        use_double_wide_mlp=True, final_logit_softcapping=30.0, use_bidirectional_attention=None, enable_moe_block=False,
        num_experts=None, top_k_experts=None, moe_intermediate_size=None, eos_token_id=1, bos_token_id=2, pad_token_id=0, dtype="bfloat16",
    )
    vision = dict(
        model_type="gemma4_vision", hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, head_dim=64, global_head_dim=64, patch_size=16, pooling_kernel_size=3, position_embedding_size=64,
        hidden_activation="gelu_pytorch_tanh", rms_norm_eps=1e-6, standardize=False, use_clipped_linears=True,
        attention_bias=False, attention_dropout=0.0, rope_parameters=dict(rope_type="axial", rope_theta=100.0), dtype="bfloat16",
    )
    tokens = dict(audio_token_id=89, boa_token_id=88, eoa_token_id=87, eoa_token_index=87, image_token_id=90, boi_token_id=91, eoi_token_id=92, video_token_id=93)
    config = dict(architectures=["Gemma4ForConditionalGeneration"], model_type="gemma4", text_config=text, vision_config=vision, audio_config=None, tie_word_embeddings=True, eos_token_id=1, **tokens)
    settings = dict(image_processor=dict(image_processor_type="Gemma4ImageProcessor", max_soft_tokens=70, patch_size=16, pooling_kernel_size=3, image_seq_length=70))
    model = Gemma4ForConditionalGeneration(Gemma4Config(**float32(config))).float().eval()
    model.config.text_config._attn_implementation = "eager"
    model.config.vision_config._attn_implementation = "eager"
    deterministic(model, fan_in=("model.vision_tower.", "model.embed_vision."))
    processor = Gemma4ImageProcessor(max_soft_tokens=70, patch_size=16, pooling_kernel_size=3)
    data = {}
    for name, image in (("small", small), ("large", large)):
        out = processor(images=[image], return_tensors="pt")
        count = int(out["num_soft_tokens_per_image"][0]) * 9
        data[f"{name}.pixels"] = out["pixel_values"][0][:count].float().contiguous()
        data[f"{name}.positions"] = out["image_position_ids"][0][:count].to(torch.int64).contiguous()
    out = processor(images=[small], return_tensors="pt")
    with torch.inference_mode():
        features = model.model.get_image_features(out["pixel_values"], out["image_position_ids"], return_dict=True)
    data["tower"] = features.last_hidden_state.float().contiguous()
    soft = features.pooler_output[0].float().contiguous()
    data["soft_tokens"] = soft
    ids = torch.tensor([[2, 7, 91] + [90] * soft.shape[0] + [92, 9, 3]])
    data["tokens"] = ids[0].to(torch.int64)
    handles = record_layers(model.model.language_model.layers, data)
    with torch.inference_mode():
        result = model(input_ids=ids, pixel_values=out["pixel_values"], image_position_ids=out["image_position_ids"], use_cache=False)
    for handle in handles:
        handle.remove()
    data["logits"] = result.logits.squeeze(0).float().contiguous()
    tiny_gemma_video(model, data)
    (root / "gemma4").mkdir(parents=True, exist_ok=True)
    directory = root / "gemma4"
    write(directory, model, config, settings["image_processor"], data)
    (directory / "preprocessor_config.json").unlink()
    settings["video_processor"] = dict(video_processor_type="Gemma4VideoProcessor", max_soft_tokens=70, num_frames=4, patch_size=16, pooling_kernel_size=3)
    (directory / "processor_config.json").write_text(json.dumps(settings, indent=2) + "\n")


def tiny_gemma_video(model, data):
    from transformers import Gemma4VideoProcessor
    from transformers.video_utils import VideoMetadata
    frames = moving(10, 61, 93)
    processor = Gemma4VideoProcessor(max_soft_tokens=70, num_frames=4, patch_size=16, pooling_kernel_size=3)
    metadata = VideoMetadata(total_num_frames=10, fps=5.0, width=61, height=93, frames_indices=list(range(10)), duration=2.0)
    out = processor(videos=[frames], video_metadata=[metadata], return_metadata=True, return_tensors="pt")
    tokens_per_frame = int(out["num_soft_tokens_per_video"][0])
    count = tokens_per_frame * 9
    with torch.inference_mode():
        features = model.model.get_video_features(out["pixel_values_videos"], out["video_position_ids"], out["num_frames_per_video"], return_dict=True)
    soft = features.pooler_output[0].float().contiguous()
    frames_used = int(out["num_frames_per_video"][0])
    tokens = [2]
    for _ in range(frames_used):
        tokens += [5, 91] + [93] * tokens_per_frame + [92]
    ids = torch.tensor([tokens])
    with torch.inference_mode():
        # Transformers 5.20.2's forward drops `num_frames_per_video`, so the features go in encoded.
        result = model(input_ids=ids, mm_encoder_outputs={"video": features}, use_cache=False)
    data.update({
        "video.frames": torch.from_numpy(frames).contiguous(),
        "video.pixels": out["pixel_values_videos"][:, :count].float().contiguous(),
        "video.indices": torch.tensor(list(out["video_metadata"][0].frames_indices), dtype=torch.int64),
        "video.soft_tokens": soft,
        "video.tokens": ids[0].to(torch.int64),
        "video.logits": result.logits.squeeze(0).float().contiguous(),
    })


def tiny(root):
    root.mkdir(parents=True, exist_ok=True)
    frames = write_clip(root / "clip.mp4", moving(20, 64, 48), 10)
    save_file({"frames": torch.from_numpy(frames).contiguous()}, root / "clip.safetensors")
    small, large = pattern(61, 93), pattern(800, 533)
    small.save(root / "small.png")
    large.save(root / "large.png")
    tiny_qwen(root, small, large)
    tiny_gemma(root, small, large)


def real_video(directory, video_path, output, qwen):
    """A greedy answer on 0.8B, or E2B's prompt: PyAV frames through each model's own processor."""
    from transformers import AutoProcessor
    from transformers.video_utils import VideoMetadata
    frames, fps = read_clip(video_path)
    metadata = VideoMetadata(total_num_frames=len(frames), fps=fps, width=frames.shape[2], height=frames.shape[1], frames_indices=list(range(len(frames))), duration=len(frames) / fps)
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Describe this video in one sentence."}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[text], videos=[frames], video_metadata=[metadata], return_tensors="pt", add_special_tokens=False)
    data = {"tokens": inputs["input_ids"][0].to(torch.int64).contiguous()}
    if qwen:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(directory, dtype=torch.float32, local_files_only=True).eval()
        with torch.inference_mode():
            data["soft_tokens"] = model.model.visual(inputs["pixel_values_videos"], grid_thw=inputs["video_grid_thw"]).pooler_output.float().contiguous()
            generated = model.generate(**inputs, max_new_tokens=48, do_sample=False)[0, inputs["input_ids"].shape[1]:]
        data["greedy"] = generated.to(torch.int64).contiguous()
        print(repr(processor.decode(generated)), flush=True)
    save_file(data, output, metadata={"transformers": transformers.__version__, "torch": torch.__version__})
    print(f"wrote {output}: {data['tokens'].numel()} prompt tokens", flush=True)


def real_qwen(directory, image_path, output):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(directory, dtype=torch.float32, local_files_only=True).eval()
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=False)
    data = {"tokens": inputs["input_ids"][0].to(torch.int64), "pixels": inputs["pixel_values"].float().contiguous(), "grid": inputs["image_grid_thw"][0].to(torch.int64)}
    with torch.inference_mode():
        data["soft_tokens"] = model.model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]).pooler_output.float().contiguous()
        logits = model(**inputs, use_cache=False).logits[0]
        data["last_logits"] = logits[-1].float().contiguous()
        generated = model.generate(**inputs, max_new_tokens=48, do_sample=False)[0, inputs["input_ids"].shape[1]:]
    data["greedy"] = generated.to(torch.int64).contiguous()
    save_file(data, output, metadata={"transformers": transformers.__version__, "torch": torch.__version__, "revision": QWEN_REVISION})
    print(f"wrote {output}: grid {inputs['image_grid_thw'][0].tolist()}, {data['tokens'].numel()} prompt tokens\n{processor.decode(generated)!r}", flush=True)


def real_gemma(directory, image_path, output):
    """The vision tower and projector in F32, with the prompt's tokens."""
    from safetensors import safe_open
    from transformers import AutoProcessor, Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder, Gemma4VisionModel
    raw = float32(json.loads((directory / "config.json").read_text()))
    vision_config = Gemma4VisionConfig(**raw["vision_config"])
    tower = Gemma4VisionModel(vision_config).float().eval()
    embedder = Gemma4MultimodalEmbedder(vision_config, Gemma4Config(**raw).text_config).float().eval()
    with safe_open(directory / "model.safetensors", framework="pt", device="cpu") as file:
        for module, prefix in ((tower, "model.vision_tower."), (embedder, "model.embed_vision.")):
            module.load_state_dict({n: file.get_tensor(prefix + n).float() for n in module.state_dict()}, strict=True)
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=False)
    with torch.inference_mode():
        hidden = tower(pixel_values=inputs["pixel_values"], pixel_position_ids=inputs["image_position_ids"]).last_hidden_state
        soft = embedder(hidden.unsqueeze(0))[0]
    save_file({"tokens": inputs["input_ids"][0].contiguous(), "soft_tokens": soft.float().contiguous()}, output,
              metadata={"transformers": transformers.__version__, "torch": torch.__version__, "revision": GEMMA_REVISION})
    print(f"wrote {output}: {soft.shape[0]} soft tokens, {inputs['input_ids'].shape[-1]} prompt tokens", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", type=Path)
    parser.add_argument("--qwen", type=Path)
    parser.add_argument("--gemma", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--golden", type=Path)
    args = parser.parse_args()
    torch.manual_seed(0)
    if args.tiny:
        tiny(args.tiny)
    if args.video:
        real_video(args.qwen or args.gemma, args.video, args.golden, args.qwen is not None)
    elif args.qwen:
        real_qwen(args.qwen, args.image, args.golden)
    elif args.gemma:
        real_gemma(args.gemma, args.image, args.golden)


if __name__ == "__main__":
    main()
