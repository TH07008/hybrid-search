use thiserror::Error;

/// Errors produced by the hybrid index.
#[derive(Debug, Error)]
pub enum HybridError {
    #[error("vector dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("duplicate document id: {0}")]
    DuplicateId(String),

    #[error("empty query: neither text nor vector provided")]
    EmptyQuery,

    #[error("BM25/tantivy error: {0}")]
    Tantivy(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("index is empty")]
    EmptyIndex,
}

impl From<tantivy::TantivyError> for HybridError {
    fn from(e: tantivy::TantivyError) -> Self {
        HybridError::Tantivy(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, HybridError>;