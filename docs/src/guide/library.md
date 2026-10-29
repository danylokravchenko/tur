# Rust library

The `tur` crate exposes the engine for use inside your own process. The workspace isn't published to crates.io, so depend on it by git or path:

```toml
[dependencies]
tur = { git = "https://github.com/danylokravchenko/tur" }
```

## One request

```rust,ignore
use std::path::PathBuf;
use tur::{
    InferencePipeline, ModelFactory, ModelSource, Result,
    backend::pipeline::{GenerationRequest, PipelineConfig},
    hybrid::SamplingOptions,
    models::{DEFAULT_KV_LAYOUT, DEFAULT_RECURRENT_STATE, WeightFormat},
};

fn main() -> Result<()> {
    let factory = ModelFactory {
        source: ModelSource::LocalPath(PathBuf::from("/models/qwen3.5-0.8b")),
        revision: "my-checkpoint".into(),
        format: Some(WeightFormat::Q8_0), // None for a prepacked artifact
        device: tur::shared::inference_device(false)?,
        kv: DEFAULT_KV_LAYOUT,
        recurrent: DEFAULT_RECURRENT_STATE,
    };
    let mut pipeline = InferencePipeline::new(&factory, PipelineConfig::new(1, 256))?;

    let mut request = GenerationRequest::new("Write one sentence about the Moon.".into(), 64);
    request.sampling = SamplingOptions {
        temperature: Some(0.0),
        ..SamplingOptions::default()
    };
    let stats = pipeline.run(&request, |text| print!("{text}"))?;
    println!("\n{} tokens, finished by {}", stats.generated_tokens, stats.finish_reason);
    Ok(())
}
```

`InferencePipeline::new` validates the checkpoint, plans memory against the device and loads the weights. `run` needs an idle pipeline and streams decoded text to the callback.

## Many requests

The pipeline has a single owner: you submit requests, then step it. Each step runs one batched pass over every scheduled request and returns their new text and any finished results. One request failing never fails the others.

```rust,ignore
use std::collections::HashMap;
use tur::{InferencePipeline, Result, backend::pipeline::GenerationRequest};

fn serve_batch(pipeline: &mut InferencePipeline, prompts: Vec<String>) -> Result<()> {
    let mut texts = HashMap::new();
    for prompt in prompts {
        let id = pipeline.submit_request(&GenerationRequest::new(prompt, 128))?;
        texts.insert(id, String::new());
    }
    while pipeline.active_request_count() > 0 {
        let step = pipeline.step_streaming()?;
        for (id, fragment) in step.new_tokens {
            if let Some(text) = texts.get_mut(&id) {
                text.push_str(&fragment);
            }
        }
        for (id, outcome) in step.completed {
            match outcome {
                Ok(stats) => println!("{id}: {} ({})", stats.generated_text, stats.finish_reason),
                Err(error) => eprintln!("{id} failed: {error}"),
            }
        }
    }
    Ok(())
}
```

`step_streaming` is `begin_step` followed by `finish_step`. Calling the two separately lets you submit or cancel requests while the GPU runs the pass, which is how the HTTP server overlaps new requests with execution. At most one pass is in flight.

## Requests

`GenerationRequest::new(prompt, max_tokens)` makes a single user turn. Its fields are public:

| Field | Meaning |
| --- | --- |
| `messages` | The full conversation (`ChatMessage` with `Role::{System, User, Assistant, Tool}`) |
| `tools` | Function definitions rendered by the template |
| `render` | `enable_thinking`, `reasoning_effort`, `preserve_thinking`, `add_generation_prompt` |
| `sampling` | `SamplingOptions`; `None` fields use the checkpoint's defaults |
| `seed` | Seed of this request's random stream |
| `timeout` | Deadline measured from submission |
| `cache_salt` | Prefix-reuse namespace |
| `grammar` | A `TopLevelGrammar` (llguidance) the output must match; the request ends when the grammar does |

A finished request returns `GenerationStats`, with the generated text and any parsed `tool_calls`, prompt, generated and cached token counts, time to first token and total time, how often it was preempted, how many tokens were drafted and accepted, and the `finish_reason`.

## Configuration

`PipelineConfig::new(max_batch_size, prefill_chunk_size)` gives the defaults the CLI uses. Its public fields are:

- `scheduler` (`SchedulerConfig`): running and waiting limits, the pass token cap, the pass time target and the queue-age limit.
- `prefix` (`PrefixConfig`): snapshot spacing, count and memory share. `PrefixConfig::DISABLED` turns reuse off.
- `speculation`: `SpeculationConfig::new(drafts)`, or `with_mtp(drafts)` to add the MTP head.
- `state_bytes`: the state pool size, or `None` to derive it.
- `device_sampling`: sample on the GPU (the default) or the CPU.

`PipelineConfig::reproducible(chunk)` runs requests one at a time for bit-reproducible output.

To run several models on one GPU, create one `DeviceBudget` and pass it to `InferencePipeline::with_budget` for each, so their memory plans share the device.

The lower-level pieces (`HybridEngine`, `HybridModel`, state pools and kernels) are public too; see [Execution engine](../architecture/execution.md).
