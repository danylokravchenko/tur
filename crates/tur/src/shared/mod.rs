pub use errors::*;
use tracing_subscriber::{
    EnvFilter, Layer as _, layer::SubscriberExt as _, util::SubscriberInitExt as _,
};
pub use tur_core::errors;

/// Initialize stderr logging, returning invalid setup or duplicate-subscriber errors.
pub fn init_tracing() -> Result<()> {
    let mut env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    for directive in ["ureq=error", "tokenizers=error", "rustls=error"] {
        env_filter = env_filter.add_directive(
            directive
                .parse()
                .map_err(|error| TurError::Other(format!("Invalid logging directive: {error}")))?,
        );
    }
    let console_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_writer(std::io::stderr)
        .with_file(false)
        .with_line_number(false)
        .with_thread_names(true)
        .with_thread_ids(true)
        .with_target(true)
        .with_filter(env_filter);
    tracing_subscriber::registry()
        .with(console_layer)
        .try_init()
        .map_err(|error| TurError::Other(format!("Cannot initialize logging: {error}")))
}

/// Select Metal for inference or the CPU numerical reference for development.
/// Check enumeration before calling Candle, whose device constructor indexes the list.
pub fn inference_device(cpu_reference: bool) -> Result<candle_core::Device> {
    if cpu_reference {
        return Ok(candle_core::Device::Cpu);
    }
    #[cfg(feature = "metal")]
    {
        if metal::Device::all().is_empty() {
            return Err(TurError::Other(
                "no accessible Metal device; GPU access is required".into(),
            ));
        }
        Ok(candle_core::Device::new_metal(0)?)
    }
    #[cfg(not(feature = "metal"))]
    Err(TurError::Other(
        "Metal support is disabled; use --cpu-reference for development".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_reference_selection() -> Result<()> {
        assert!(inference_device(true)?.is_cpu());
        Ok(())
    }
}
