#![forbid(unsafe_code)]

use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use clap::Parser;
use parking_lot::RwLock;
use serde_json::{Value, json};
use std::{error::Error, io, sync::Arc, time::Instant};
use tokenizers::Tokenizer;
use tur::{
    ModelFactory, ModelSource,
    backend::{
        InferenceEngine,
        chat_template::{ChatTemplate, Message},
    },
    models::{
        ModelImpl, Qwen35ModelForCausalLM,
        kv_cache::{BlockAllocator, KvCacheImpl, PagedKvCache},
    },
};
use uuid::Uuid;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const SEED: u64 = 299_792_458;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 3)]
    iterations: usize,
    #[arg(long, default_value_t = 16)]
    decode_tokens: usize,
    #[arg(long, value_delimiter = ',', default_value = "128,2048")]
    prompt_lengths: Vec<usize>,
    #[arg(long)]
    prefix_cache: bool,
}

fn memory(device: &Device) -> usize {
    match device {
        #[cfg(feature = "metal")]
        Device::Metal(native) => native.current_allocated_size(),
        _ => 0,
    }
}

fn prompt(
    tokenizer: &Tokenizer,
    template: &ChatTemplate,
    target: usize,
) -> Result<(String, Vec<u32>)> {
    let mut words = target;
    for _ in 0..8 {
        let text = template.format(&[Message::user("Rust ".repeat(words))], None, true, false)?;
        let ids = tokenizer
            .encode(text.clone(), false)
            .map_err(|error| io::Error::other(error.to_string()))?
            .get_ids()
            .to_vec();
        if ids.len() == target {
            return Ok((text, ids));
        }
        words = words
            .checked_add(target)
            .and_then(|n| n.checked_sub(ids.len()))
            .ok_or_else(|| {
                io::Error::other("prompt length is smaller than chat template overhead")
            })?;
    }
    Err(io::Error::other("could not construct requested exact prompt length").into())
}

fn paged_snapshot(caches: &[Vec<PagedKvCache>], allocator: &BlockAllocator) -> Result<Value> {
    let mut full_bytes = 0_usize;
    let mut page_bytes = 0_usize;
    let mut pages = 0_usize;
    for request in caches {
        for layer in request {
            if let Some((k, v)) = layer.get_state() {
                full_bytes += (k.elem_count() + v.elem_count()) * k.dtype().size_in_bytes();
            }
            for id in layer.block_table() {
                let block = allocator.get_block(*id)?;
                for tensor in [&block.k, &block.v].into_iter().flatten() {
                    page_bytes += tensor.elem_count() * tensor.dtype().size_in_bytes();
                }
                pages += 1;
            }
        }
    }
    Ok(
        json!({"full_cache_tensor_bytes": full_bytes, "physical_page_tensor_bytes": page_bytes, "pages": pages}),
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    let hf_home = std::path::PathBuf::from(std::env::var("HF_HOME")?);
    let checkpoints: Value =
        serde_json::from_str(&std::fs::read_to_string(hf_home.join("checkpoints.json"))?)?;
    let gguf_path = hf_home.join("hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q4_K_M.gguf");
    let content =
        candle_core::quantized::gguf_file::Content::read(&mut std::fs::File::open(gguf_path)?)?;
    let mut formats = std::collections::BTreeMap::<String, (usize, usize)>::new();
    for info in content.tensor_infos.values() {
        let bytes = info
            .shape
            .elem_count()
            .checked_div(info.ggml_dtype.block_size())
            .and_then(|blocks| blocks.checked_mul(info.ggml_dtype.type_size()))
            .ok_or_else(|| io::Error::other("invalid packed tensor byte count"))?;
        let entry = formats.entry(format!("{:?}", info.ggml_dtype)).or_default();
        entry.0 += 1;
        entry.1 = entry
            .1
            .checked_add(bytes)
            .ok_or_else(|| io::Error::other("packed weight byte total overflow"))?;
    }
    if std::env::var("HF_ENDPOINT")? != "http://127.0.0.1:9" {
        return Err(io::Error::other(
            "baseline requires the isolated pinned cache and disabled remote endpoint",
        )
        .into());
    }
    if args.iterations == 0 || args.decode_tokens == 0 || args.prompt_lengths.is_empty() {
        return Err(
            io::Error::other("iterations, token counts and prompt list must be nonempty").into(),
        );
    }
    #[cfg(feature = "metal")]
    if metal::Device::all().is_empty() {
        return Err(io::Error::other("no accessible Metal device").into());
    }
    let device = Device::new_metal(0)?;
    let factory = ModelFactory::<Qwen35ModelForCausalLM>::new(
        ModelSource::HuggingFace("Qwen/Qwen3-0.6B".to_string()),
        Some("Q4_K_M".to_string()),
        device.clone(),
        DType::F32,
    );
    let load_start = Instant::now();
    let mut builder = InferenceEngine::builder(&factory, device.clone())
        .seed(SEED)
        .temperature(0.0)
        .repeat_penalty(1.0);
    if args.prefix_cache {
        builder = builder.with_prefix_cache(16, 4096);
    }
    let (mut engine, tokenizer, template) = builder.build()?;
    device.synchronize()?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let loaded_bytes = memory(&device);
    let template =
        template.ok_or_else(|| io::Error::other("pinned checkpoint template is missing"))?;
    let mut cases = Vec::new();
    for length in args.prompt_lengths {
        let (text, tokens) = prompt(&tokenizer, &template, length)?;
        let mut runs = Vec::new();
        for iteration in 0..=args.iterations {
            device.synchronize()?;
            let start = Instant::now();
            let (first, _, hit, cached) = engine.prefill(&tokens)?;
            device.synchronize()?;
            let ttft = start.elapsed().as_secs_f64();
            let mut generated = vec![first];
            let mut decode_seconds = Vec::new();
            for index in 1..args.decode_tokens {
                let offset = length
                    .checked_add(index - 1)
                    .ok_or_else(|| io::Error::other("position overflow"))?;
                device.synchronize()?;
                let start = Instant::now();
                let next = engine.decode_step(&generated, offset)?;
                device.synchronize()?;
                decode_seconds.push(start.elapsed().as_secs_f64());
                generated.push(next);
            }
            if iteration > 0 {
                runs.push(json!({"ttft_seconds": ttft, "decode_seconds": decode_seconds, "generated_ids": generated,
                    "cache_hit": hit, "reported_cached_tokens": cached, "metal_allocated_bytes": memory(&device)}));
            }
        }
        cases.push(json!({"prompt_text": text, "prompt_ids": tokens, "prompt_tokens": length, "runs": runs}));
    }
    engine.model_mut().clear_kv_cache();
    // Inspect actual per-request page payload and the duplicate full cache on the current engine path.
    let allocator = Arc::new(RwLock::new(BlockAllocator::new(4096, 16)));
    let (_, tokens) = prompt(&tokenizer, &template, 128)?;
    let mut caches: Vec<Vec<PagedKvCache>> = (0..2)
        .map(|_| {
            (0..engine.model().num_layers())
                .map(|_| PagedKvCache::new(Arc::clone(&allocator), 2))
                .collect()
        })
        .collect();
    let ids = [Uuid::new_v4(), Uuid::new_v4()];
    let batch: Vec<_> = ids.iter().map(|id| (*id, tokens.clone(), 0)).collect();
    let start = Instant::now();
    let first = engine.prefill_batch(&batch, Some(&mut caches))?;
    device.synchronize()?;
    let prefill_seconds = start.elapsed().as_secs_f64();
    let before = paged_snapshot(&caches, &allocator.read())?;
    let decode: Vec<_> = first
        .into_iter()
        .map(|(id, token)| (id, vec![token], 128))
        .collect();
    let start = Instant::now();
    engine.decode_batch(&decode, Some(&mut caches))?;
    device.synchronize()?;
    let decode_seconds = start.elapsed().as_secs_f64();
    let after = paged_snapshot(&caches, &allocator.read())?;
    drop(caches);
    let reclaimed_pages = allocator.read().num_allocated();
    if reclaimed_pages != 0 {
        return Err(io::Error::other("paged baseline leaked allocated blocks").into());
    }
    let logits = Tensor::arange(0_u32, 151_936_u32, &device)?
        .to_dtype(DType::F32)?
        .affine(1e-5, 0.0)?;
    let mut sampling = Vec::new();
    for temperature in [0.0, 0.7] {
        let mut sampler = LogitsProcessor::new(SEED, Some(temperature), Some(0.95));
        let mut samples = Vec::new();
        let mut ids = Vec::new();
        for _ in 0..20 {
            device.synchronize()?;
            let start = Instant::now();
            ids.push(sampler.sample(&logits)?);
            device.synchronize()?;
            samples.push(start.elapsed().as_secs_f64());
        }
        sampling.push(json!({"temperature": temperature, "samples_seconds": samples, "sampled_ids": ids,
            "readback_payload_bytes_from_backend_path": if temperature == 0.0 { 4 } else { 151_936 * 4 }}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1, "model": "Qwen/Qwen3-0.6B", "quantization": "Q4_K_M", "activation_dtype": "f32",
        "checkpoints": checkpoints,
        "gguf_tensor_formats_count_and_payload_bytes": formats,
        "gguf_quantization_version": format!("{:?}", content.metadata.get("general.quantization_version")),
            "seed": SEED, "temperature": 0.0, "repeat_penalty": 1.0, "prefix_cache": args.prefix_cache,
            "eos_policy": "fixed output length for baseline; EOS is recorded but does not stop the loop",
            "first_process_model_load_seconds": load_seconds, "filesystem_cache": "uncontrolled; may already be warm",
            "loaded_metal_allocated_bytes": loaded_bytes, "final_metal_allocated_bytes": memory(&device),
            "cases": cases, "sampling": sampling,
            "paged_probe": {"requests": 2, "prompt_tokens_each": 128, "prefill_seconds": prefill_seconds,
                "one_decode_seconds": decode_seconds, "before_decode": before, "after_decode": after,
                "allocated_pages_after_request_drop": reclaimed_pages},
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_snapshot_counts_duplicate_cache_and_partial_page_padding() -> Result<()> {
        let allocator = Arc::new(RwLock::new(BlockAllocator::new(2, 16)));
        let mut cache = PagedKvCache::new(Arc::clone(&allocator), 2);
        let k = Tensor::zeros((1, 2, 17, 4), DType::F32, &Device::Cpu)?;
        let v = Tensor::ones((1, 2, 17, 4), DType::F32, &Device::Cpu)?;
        let _ = cache.append(&k, &v)?;
        let caches = vec![vec![cache]];
        let snapshot = paged_snapshot(&caches, &allocator.read())?;
        assert_eq!(snapshot["full_cache_tensor_bytes"], 1088);
        assert_eq!(snapshot["physical_page_tensor_bytes"], 2048);
        assert_eq!(snapshot["pages"], 2);
        drop(caches);
        assert_eq!(allocator.read().num_allocated(), 0);
        Ok(())
    }
}
