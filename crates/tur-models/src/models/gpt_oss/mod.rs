//! gpt-oss text decoder: alternating sliding and full attention with sinks, and routed
//! experts stored in MXFP4.
mod config;
mod model;
mod prune;
pub use config::Config;
pub use model::GptOss;
pub use prune::{
    ExpertProbe, LayerPlan, LayerProbe, Pruning, divergence, plan_pruning, prune, refine,
};
