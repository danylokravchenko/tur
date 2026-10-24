#![allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
#![forbid(unsafe_code)]
//! Prepacked artifacts through the model factory and the inference pipeline.
mod common;
use common::Checkpoint;
use tur::{
    InferencePipeline, ModelSource, Result,
    backend::{
        pipeline::{GenerationRequest, PipelineConfig},
        prefix::PrefixConfig,
        speculation::SpeculationConfig,
    },
    models::qwen3_5::{
        Conversion, Packer, WeightFormat, WeightPlan, artifact_config, write_artifact,
    },
};

#[cfg(feature = "metal")]
#[test]
fn test_factory_serves_artifacts_exactly_like_load_time_conversion() -> Result<()> {
    let fixture = Checkpoint::fused_mtp()?;
    let output = Checkpoint(fixture.0.with_extension("prepacked"));
    let config = artifact_config(&fixture.0)?;
    write_artifact(
        &Conversion {
            source: &fixture.0,
            identity: "tiny-adapter-fixture-v1",
            output: &output.0,
            calibration: "",
        },
        &WeightPlan::uniform(&config, WeightFormat::Q4K)?,
        &Packer::RTN,
    )?;
    let mut converted = fixture.factory();
    converted.device = candle_core::Device::new_metal(0)?;
    converted.format = Some(WeightFormat::Q4K);
    let prepacked = tur::ModelFactory {
        source: ModelSource::LocalPath(output.0.clone()),
        format: None,
        ..converted.clone()
    };
    let request = GenerationRequest::new(
        (2..40)
            .map(|t| format!("t{t}"))
            .collect::<Vec<_>>()
            .join(" "),
        24,
    );
    let mtp = PipelineConfig {
        prefix: PrefixConfig::DISABLED,
        speculation: SpeculationConfig::with_mtp(1),
        ..PipelineConfig::new(1, 256)
    };
    for config in [PipelineConfig::new(1, 256), mtp] {
        let expected = InferencePipeline::new(&converted, config)?.run(&request, |_| {})?;
        let actual = InferencePipeline::new(&prepacked, config)?.run(&request, |_| {})?;
        assert_eq!(actual.generated_text, expected.generated_text);
        assert_eq!(actual.generated_tokens, 24);
    }
    // A conversion format cannot reinterpret an artifact.
    let mut wrong = prepacked;
    wrong.format = Some(WeightFormat::Q4K);
    assert!(InferencePipeline::new(&wrong, PipelineConfig::new(1, 256)).is_err());
    Ok(())
}
