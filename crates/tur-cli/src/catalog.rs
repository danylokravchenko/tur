//! Short names for the models Tur has generated from, each with the commit and limits it ran at.
use std::io::Write as _;

use tur::{Result, backend::factory::is_downloaded};

use crate::args;

/// One supported model behind a short name.
pub(crate) struct Model {
    pub(crate) name: &'static str,
    /// The family's default, the name people reach for first.
    pub(crate) alias: &'static str,
    pub(crate) repository: &'static str,
    pub(crate) revision: &'static str,
    /// Files to download, as measured in the Hugging Face cache.
    download: &'static str,
    about: &'static str,
    pub(crate) expert_cache_mb: Option<usize>,
    pub(crate) max_batch_size: Option<usize>,
}

pub(crate) const MODELS: &[Model] = &[
    Model {
        name: "qwen3.5:0.8b",
        alias: "qwen",
        repository: "Qwen/Qwen3.5-0.8B",
        revision: "2fc06364715b967f1860aea9cf38778875588b17",
        download: "1.6 GB",
        about: "Small and fast; runs on any Apple Silicon Mac",
        expert_cache_mb: None,
        max_batch_size: None,
    },
    Model {
        name: "gemma4:e2b",
        alias: "gemma",
        repository: "google/gemma-4-E2B-it",
        revision: "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        download: "9.6 GB",
        about: "Text, images, video and WAV audio; about 9 GB of memory",
        expert_cache_mb: None,
        max_batch_size: None,
    },
    // 2 GiB and one request are the measured setup: 12.8 GiB of weights exceed a 16 GB Mac's
    // working set, larger caches have swapped, and four requests swapped 251 MiB.
    Model {
        name: "gpt-oss:20b",
        alias: "gpt-oss",
        repository: "openai/gpt-oss-20b",
        revision: "6cee5e81ee83917806bbde320786a8fb61efebee",
        download: "13 GB",
        about: "Larger than memory; experts read from the SSD into 2 GiB",
        expert_cache_mb: Some(2048),
        max_batch_size: Some(1),
    },
];

/// The model called `name`, by its full name or its family's alias.
pub(crate) fn find(name: &str) -> Option<&'static Model> {
    MODELS
        .iter()
        .find(|model| model.name == name || model.alias == name)
}

/// Download a model ahead of its first run.
#[derive(Debug, clap::Args)]
pub(crate) struct PullArgs {
    /// A name from `tur list`, or an organization/name Hugging Face repository.
    model: String,
    /// Commit to download. Default: the name's pinned commit, or a repository's latest.
    #[arg(long)]
    revision: Option<String>,
}

pub(crate) fn pull(pull: &PullArgs) -> Result<()> {
    let entry = find(&pull.model);
    let model = entry.map_or(pull.model.as_str(), |entry| entry.repository);
    let revision = pull
        .revision
        .as_deref()
        .or(entry.map(|entry| entry.revision));
    let (source, revision) = args::identity(model, revision)?;
    let directory = tur::backend::factory::resolve_checkpoint(&source, &revision)?;
    writeln!(std::io::stdout(), "{}", directory.display())?;
    Ok(())
}

/// Print every name, what it is, and whether its weights are already here.
pub(crate) fn list() -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    writeln!(
        stdout,
        "{:<9} {:<13} {:<7} {:<10} ABOUT",
        "NAME", "FULL NAME", "SIZE", "STATE"
    )?;
    for model in MODELS {
        let state = if is_downloaded(model.repository, model.revision) {
            "downloaded"
        } else {
            "-"
        };
        writeln!(
            stdout,
            "{:<9} {:<13} {:<7} {state:<10} {}",
            model.alias, model.name, model.download, model.about
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert fallible catalog lookups"
)]
mod tests {
    use super::*;

    #[test]
    fn test_full_names_and_aliases_find_one_pinned_model() {
        for model in MODELS {
            assert_eq!(model.revision.len(), 40, "{}", model.name);
            assert!(model.repository.contains('/'), "{}", model.name);
            for name in [model.name, model.alias] {
                assert_eq!(find(name).map(|found| found.name), Some(model.name));
            }
        }
        assert!(find("llama").is_none());
    }
}
