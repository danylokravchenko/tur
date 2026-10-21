pub mod errors;

pub use errors::*;
use tracing_subscriber::{
    EnvFilter, Layer as _, layer::SubscriberExt as _, util::SubscriberInitExt as _,
};

pub fn init_tracing() {
    let registry = tracing_subscriber::registry();

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(tracing::level_filters::LevelFilter::TRACE.to_string()))
        .add_directive("ureq=error".parse().unwrap())
        .add_directive("tokenizers=error".parse().unwrap())
        .add_directive("rustls=error".parse().unwrap());

    let console_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_file(false)
        .with_line_number(false)
        .with_thread_names(true)
        .with_thread_ids(true)
        .with_target(true)
        .with_filter(env_filter.clone());

    let subscriber = registry.with(console_layer);

    subscriber.try_init().unwrap();
}
