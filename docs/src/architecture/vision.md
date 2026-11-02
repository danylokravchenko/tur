# Images and video

Qwen3.5-family checkpoints and Gemma 4 carry a vision encoder, so a chat request can include pictures and videos and the model looks at them directly. gpt-oss has no vision encoder and rejects both.

Every image starts the same way. `tur_models::image::Image` decodes a PNG, JPEG, WebP or GIF (its first frame) to RGB, applies EXIF orientation and drops any alpha channel. Decoders check the claimed size before allocating, so a small file can't request a huge bitmap. The server never fetches URLs; images arrive as `data:` URLs or, from `tur run`, as files.

## From pixels to soft tokens

Each model resizes the image as its Transformers processor does, cuts it into 16-pixel patches, and runs them through a vision transformer whose layers see every patch of the image. The tower's output is a set of soft tokens, each covering a block of patches.

Resampling follows `torchvision`'s path for 8-bit tensors: antialiased bicubic filters in fixed point, along rows and then columns, with each pass rounded to 8 bits. Tur's output equals it byte for byte, which a float filter does not manage.

Qwen3.5 sets each side to a multiple of 32 pixels and keeps the area between the checkpoint's minimum and Tur's limit of 1,048,576 pixels. Each 2×2 block of patches merges into one soft token, so a soft token covers 32×32 pixels and a 1024×1024 image makes 1024 of them. Learned positions are resampled bilinearly from the checkpoint's square table to each image's grid, and attention rotates keys and queries by a patch's row and column.

Gemma 4 resizes to the largest multiple of 48 pixels that fits 2520 patches, then averages each 3×3 block into a soft token, up to the 280 its checkpoints declare. Learned positions are looked up per axis, and keys and queries rotate in separate runs for x and y.

The towers attend through the tiled kernel of long-prompt prefill, reading projected keys and values in place as one page of a K/V pool, as Whisper's encoder does. Head widths that kernel doesn't have (72 on the larger checkpoints) run on Candle's matmuls instead. Rotation runs on `axial_rope`, whose every frequency reads its own coordinate of a position. Projections are `Q8_0` beside a packed text model and F32 beside an F32 one.

## In the prompt

The template renders an image part as one placeholder. Tur expands it to one position per soft token, adding the frame Gemma's template leaves out, and each soft token replaces the input embedding at its position. Everything after the embedding (decoder layers, paged K/V, scheduling, sampling and speculation) runs as for text, so an image costs one prefill of its soft tokens and nothing per generated token.

Qwen's text rotary needs one more thing. Its frequencies are interleaved between three coordinates (time, row and column), which are all equal for text and differ across an image. A soft token rotates at the image's start plus its row or column. The text after an image continues from the start plus the grid's longer side, so a rotary position can differ from the slot its keys occupy. Packed passes therefore carry `axial` positions beside their slot positions. A pass holding an image's request sends all its rows through `axial_rope`, which computes the same rotation as `qk_norm_rope` when a token's three coordinates are equal. Passes without one keep the plain kernel.

The server encodes images on the thread that encodes audio clips, with the tower on its own Metal queue, so text passes keep running while an image encodes. The rows stay with the request until it finishes, so one that is preempted and recomputed never encodes its image again.

Prefix reuse tells images apart the way it does clips: each soft-token position holds an identity token above the vocabulary, hashed from the image's size and pixels and the position's index. A repeated request with the same image restores its prompt, and a different image of the same size shares only the text before it.

## Video

A video is H.264 in an MP4 file, arriving as a `data:` URL or, from `tur run`, a file. Tur reads its header when the request arrives and keeps the file rather than its frames. Which frames to decode depends on the model, so a long clip never sits in memory as bitmaps. The encoder decodes up to the last frame the model samples and drops the rest as they pass, with B-frames returned in display order.

Qwen3.5 samples two frames per second, at least 4 and at most 128, and resizes them all alike so their pixels together fit a budget. A pair of frames forms one temporal step, and the tower's 3D patches read both. Attention stays within a step, so cost grows with the number of steps rather than their square. In the prompt, each step becomes `<1.4 seconds>` followed by its own frame and grid, and text rotary reads each step as one image.

Gemma 4 samples 32 evenly spaced frames, or every frame of a shorter clip. Each is a separate pass through the tower at 70 soft tokens, and each is written as `00:01` before its own image frame.

## Reusing encoded rows

A model keeps the rows of its recent inputs, up to 64 MiB, and a request for the same content takes them instead of encoding again. Prefix reuse restores a repeated prompt's state, but its rows are needed anyway in case the request is preempted, so without this every repeat still paid for the tower. Rows are found by a SHA-256 of the content, so the same picture sent again as a new file hits, and a client can't craft a collision to read another's rows. Weights never change under a loaded model, so only capacity evicts an entry, least recently used first. The 64 MiB are part of the model's memory plan whether or not they fill.

## Decisions

- Conformance comes from pinned references. On small Transformers 5.20.2 fixtures, preprocessed pixels are exact, and both towers' states and soft tokens match within 8e-7 relative error on CPU and Metal. Qwen's text layers and logits with an image match within 3.2e-6. Gemma's match within 6.2e-4, on a fixture whose small activations magnify F32 rounding differently on each device.
- On the real 0.8B checkpoint, F32 soft tokens match Transformers within 2.3e-4 relative error, and greedy answers equal Transformers' tokens on a 640×480 picture in F32. `Q8_0` weights, a prepacked `Q8_0` artifact, lookup drafts and MTP drafts all give that same answer.
- The `Q8_0` towers agree with F32 Transformers at a mean cosine similarity of 0.9994 (minimum 0.970) on Qwen3.5-0.8B and 0.9999 (minimum 0.9994) on Gemma 4 E2B. They hold 108 MiB and 215 MiB of weights.
- Images are limited to 1,048,576 pixels. The 0.8B checkpoint allows 16.7 million, 16 times as many patches and about 256 times the attention work, which from the 1.56 s measured at the limit would take minutes per image on an M1. A larger image is scaled down as `smart_resize` scales one to its `max_pixels`. The fixtures sit under the limit, so the reference comparisons don't depend on it.
- `axial_rope` is one kernel for all three layouts: Qwen's text coordinates, Qwen's patches with one rotated run, and Gemma's patches with two. It costs the plain kernel's time within 4% at 1024 tokens, and 18% more at 64, only in passes holding an image's request.
- Video follows Transformers on small fixtures: preprocessed pixels are exact, soft tokens match within 1e-4 relative error, and logits within the same gates as images. On the real 0.8B checkpoint a 5-second clip gives the Transformers answer word for word in F32 and in `Q8_0`, with its 1561 prompt tokens, and E2B matches its 2319 prompt tokens. Those runs decode frames differently: Transformers' through PyAV, Tur's through OpenH264, whose frames differ from PyAV's by less than 4 of 255 on average.
- H.264 decoding uses OpenH264, built from source through the `openh264` crate, so Tur needs no system library or external tool. Decoding runs on the encoder's thread, and a parser or decoder panic becomes that request's error. HEVC, which iPhones record by default, and AV1 are rejected, and so are fragmented files. MP4 rotation metadata is ignored, as PyAV ignores it.
- Tur's limits on video are tighter than the checkpoints': 4.2 million pixels across a Qwen clip's frames against 25 million, which holds a clip to 2048 soft tokens, and 128 sampled frames against 768. A clip longer than 64 seconds is therefore sampled more sparsely than the reference. Gemma's 32 frames are its checkpoint's own, and where Transformers rejects a clip shorter than that, Tur uses every frame.
- Repeating a request at 324 soft tokens over HTTP went from 0.37 s to 0.06 s with rows reused, and a 1560-token video repeat takes 0.14 s against 3 s for the first request. Row reuse costs its 64 MiB; a larger budget has no measured workload to justify it.
- Inputs are identified by SHA-256 rather than FNV-1a, so prefix reuse can't be steered by a crafted collision either.
- Each tower owns a Metal device, because one queue runs first-in, first-out and an image's GPU work would hold every later text pass behind it. While the largest image encodes on the 0.8B checkpoint, a stream's median pass goes from 17.6 to 18.4 ms and its slowest stays near 85 ms.
- Native MTP drafts read the head's own plain positions rather than axial ones. Transformers defines no reference for that head with images, and every draft is verified, so only the acceptance rate could differ.
- EXIF orientation is applied, which Transformers leaves to the caller: a phone photo reaches the model upright. Other reference steps follow Transformers, including dropping alpha rather than compositing it.
- Artifacts carry the tower, its projections packed like the text tower's, which adds about 108 MiB at 0.8B. Artifacts written before this hold the text tower only and still load, rejecting images. GGUF text files hold no tower either.
