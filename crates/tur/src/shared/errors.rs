pub type Result<T> = std::result::Result<T, TurError>;

#[derive(thiserror::Error, Debug)]
pub enum TurError {
    #[error("Candle Error: {0}")]
    CandleError(#[from] candle_core::Error),
    #[error("Tokenizer Error: {0}")]
    Tokenizer(String),
    #[error("HF Hub Error: {0}")]
    HfHub(String),
    #[error("Guidance Error: {0}")]
    Guidance(String),
    #[error("IO failure")]
    Io(#[from] std::io::Error),
    #[error("Json failure")]
    Json(#[from] serde_json::Error),
    #[error("Unhandled error: {0}")]
    Unhandled(#[from] Box<dyn std::error::Error + Send + Sync>),
    #[error("Unknown error: {0}")]
    Other(String),
    #[error("Batch Manager Error: {0}")]
    BatchManager(String),
    #[error("Request not found: {0}")]
    RequestNotFound(String),
    #[error("Invalid request phase: {0}")]
    InvalidPhase(String),
    #[error("Memory allocation error: {0}")]
    MemoryAllocation(String),
    #[error("Block not found: {0}")]
    BlockNotFound(usize),
}
