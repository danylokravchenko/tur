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
    #[error("Request not found: {0}")]
    RequestNotFound(String),
    #[error("Invalid execution or state transition: {0}")]
    InvalidExecution(String),
    #[error("Memory allocation error: {0}")]
    MemoryAllocation(String),
    /// Admission rejected a request because a bounded queue is full.
    #[error("Request queue is full: {0}")]
    QueueFull(String),
    /// A request did not finish before its submission deadline.
    #[error("Request deadline exceeded")]
    DeadlineExceeded,
    /// The execution owner is no longer accepting requests.
    #[error("Inference worker unavailable")]
    WorkerUnavailable,
}
