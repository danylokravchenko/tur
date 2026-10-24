# Rust library

The `tur` crate re-exports the engine for use inside your own process. The workspace is
not published, so depend on it by path or git:

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
    models::qwen3_5::{DEFAULT_KV_LAYOUT, DEFAULT_RECURRENT_STATE, WeightFormat},
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

`InferencePipeline::new` validates the checkpoint, plans memory against the device
budget and loads the weights. `run` requires an idle pipeline and streams decoded text
to the callback.

## Many requests

The pipeline is a single-owner state machine: submit requests, then step it. Each step
runs one packed pass over every scheduled request and returns their new text and any
completed outcomes. One request's failure never fails the others.

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

`step_streaming` is `begin_step` followed by `finish_step`. Calling them separately
lets you submit or cancel requests while the GPU runs the pass; this is how the HTTP
worker overlaps intake with execution. At most one pass is in flight.

## Requests

`GenerationRequest::new(prompt, max_tokens)` makes one user turn. The fields are public:

| Field | Meaning |
| --- | --- |
| `messages` | Full conversation (`ChatMessage` with `Role::{System, User, Assistant, Tool}`) |
| `tools` | Function definitions rendered by the template |
| `render` | `enable_thinking`, `reasoning_effort`, `preserve_thinking`, `add_generation_prompt` |
| `sampling` | `SamplingOptions`; `None` fields inherit the checkpoint's defaults |
| `seed` | Random stream of this request |
| `timeout` | Deadline measured from submission |
| `cache_salt` | Prefix-reuse namespace |
| `grammar` | A `TopLevelGrammar` (llguidance) the output must match; the request completes when the grammar does |

Completed requests return `GenerationStats`, which holds:

- the generated text and any parsed `tool_calls`;
- prompt, generated and cached token counts;
- time to the first token and total elapsed time;
- preemption count, and drafted and accepted token counts;
- `finish_reason`.

## Configuration

`PipelineConfig::new(max_batch_size, prefill_chunk_size)` gives the defaults the CLI
uses. Its public fields:

- `scheduler` (`SchedulerConfig`): running and waiting limits, the pass token cap, the
  prefill target and the queue-age guard.
- `prefix` (`PrefixConfig`): snapshot spacing, count and state share;
  `PrefixConfig::DISABLED` turns reuse off.
- `speculation` (`SpeculationConfig::new(drafts)`, or `with_mtp(drafts)` to add the
  head).
- `state_bytes`: the state pool size; `None` derives it.
- `device_sampling`: sample on the GPU (default) or the host.

`PipelineConfig::reproducible(chunk)` serializes requests for bit-reproducible output.

To run several models on one device, create one `DeviceBudget` and pass it to
`InferencePipeline::with_budget` for each, so their plans share the device's memory.

Lower-level contracts (`HybridEngine`, `HybridModel`, state pools and kernels) are
public too; see [Execution engine](../architecture/execution.md).
