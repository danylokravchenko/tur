//! File defaults for the CLI. The engine continues to use its existing configuration types.
use std::{fmt::Display, io::ErrorKind, path::Path, str::FromStr};

use serde::{Deserialize, Deserializer, de::Error as _};
use tur::{Result, TurError, models::qwen3_5::parse_recurrent_state};

use crate::{
    args::{ModelArgs, PipelineArgs},
    run, serve,
};

// Only absent CLI/environment values inherit file defaults. In particular, false and
// zero must remain explicit overrides rather than being confused with omission.
macro_rules! inherit {
    ($($field:ident),+ $(,)?) => {
        pub(crate) fn inherit(&mut self, defaults: Self) {
            $(self.$field = self.$field.take().or(defaults.$field);)+
        }
    };
}
pub(crate) use inherit;

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct FileSettings {
    pub(crate) model: ModelArgs,
    pub(crate) engine: PipelineArgs,
    pub(crate) server: serve::Args,
    pub(crate) generation: run::Args,
}

impl FileSettings {
    pub(crate) fn load(explicit: Option<&Path>, disabled: bool) -> Result<Self> {
        if disabled {
            return Ok(Self::default());
        }
        Self::read(
            explicit.unwrap_or_else(|| Path::new("tur.toml")),
            explicit.is_some(),
        )
    }

    fn read(path: &Path, required: bool) -> Result<Self> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) if !required && error.kind() == ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(error) => {
                return Err(TurError::Other(format!(
                    "cannot read configuration {}: {error}",
                    path.display()
                )));
            }
        };
        let mut settings: Self = toml::from_str(&text).map_err(|error| {
            TurError::Other(format!("invalid configuration {}: {error}", path.display()))
        })?;
        settings
            .model
            .relative_to(path.parent().unwrap_or_else(|| Path::new(".")));
        Ok(settings)
    }
}

pub(crate) fn optional_from_str<'de, D, T>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: FromStr,
    T::Err: Display,
{
    Option::<String>::deserialize(deserializer)?
        .map(|value| value.parse().map_err(D::Error::custom))
        .transpose()
}

pub(crate) fn optional_recurrent<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<candle_core::DType>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer)?
        .map(|value| match value.as_str() {
            "f32" | "f16" => parse_recurrent_state(&value).map_err(D::Error::custom),
            _ => Err(D::Error::custom("recurrent state must be f32 or f16")),
        })
        .transpose()
}

pub(crate) fn reasoning_effort(value: &str) -> std::result::Result<String, String> {
    match value {
        "low" | "medium" | "xhigh" => Ok(value.into()),
        _ => Err("reasoning effort must be low, medium or xhigh".into()),
    }
}

pub(crate) fn optional_reasoning_effort<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer)?
        .map(|value| reasoning_effort(&value).map_err(D::Error::custom))
        .transpose()
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert fallible TOML configuration"
)]
mod tests {
    use super::*;

    #[test]
    fn test_example_file_loads() -> Result<()> {
        let settings = FileSettings::load(
            Some(Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../tur.example.toml"
            ))),
            false,
        )?;
        let pipeline = settings.engine.config(8)?;
        pipeline.scheduler.validate()?;
        pipeline
            .prefix
            .validate(settings.model.kv_layout().page_tokens)?;
        Ok(())
    }

    #[test]
    fn test_unknown_keys_and_invalid_types_are_rejected() -> Result<()> {
        let empty: FileSettings =
            toml::from_str("").map_err(|error| TurError::Other(error.to_string()))?;
        empty.engine.config(1)?.scheduler.validate()?;
        for text in [
            "unknown = 1",
            "[unknown]",
            "[model]\nsoruce = 'org/model'",
            "[model]\nsource = 10",
            "[model]\nkv_format = 'q4'",
            "[model]\nweight_format = 'q2'",
            "[model]\nrecurrent_state = 'bf16'",
            "[engine]\nstate_memory_mb = -1",
            "[engine]\nmtp = 'true'",
            "[server]\nlisten = 'localhost:8080'",
            "[server]\nmax_batch_size = 1.5",
            "[generation]\nreasoning_effort = 'high'",
            "[generation]\ntop_k = '10'",
            "[model]\ncpu_reference = true",
            "[generation]\nprompt = 'hello'",
            "[generation]\ntools = 'tools.json'",
            "[server]\nmodel = 'wrong-section'",
        ] {
            assert!(
                toml::from_str::<FileSettings>(text).is_err(),
                "accepted {text}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_explicit_missing_file_errors_and_no_config_bypasses_it() -> Result<()> {
        let missing = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/does-not-exist/tur.toml"
        ));
        let error = FileSettings::load(Some(missing), false)
            .err()
            .ok_or_else(|| TurError::Other("missing configuration was accepted".into()))?;
        assert!(error.to_string().contains("cannot read configuration"));
        FileSettings::read(missing, false)?;
        FileSettings::load(Some(missing), true)?;
        Ok(())
    }
}
