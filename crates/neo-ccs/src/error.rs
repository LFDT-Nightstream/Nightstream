use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimMismatch {
    pub expected: (usize, usize),
    pub got: (usize, usize),
}

#[derive(Debug, Error)]
pub enum CcsError {
    #[error("dimension mismatch in {context}: expected {expected:?}, got {got:?}")]
    Dim {
        context: &'static str,
        expected: (usize, usize),
        got: (usize, usize),
    },

    #[error("length mismatch in {context}: expected {expected}, got {got}")]
    Len {
        context: &'static str,
        expected: usize,
        got: usize,
    },

    #[error("n must be a power-of-two; got n={n}")]
    NNotPowerOfTwo { n: usize },

    #[error("relation failed at row {row}: residual is nonzero")]
    RowFail { row: usize },

    #[error("relation error: {0}")]
    Relation(#[from] RelationError),
}

#[derive(Debug, Error)]
pub enum RelationError {
    #[error("invalid structure: matrices are empty or have inconsistent shapes")]
    InvalidStructure,

    #[error("polynomial arity mismatch: poly arity {poly_arity} vs t={t}")]
    PolyArity { poly_arity: usize, t: usize },

    #[error("{0}")]
    Message(String),
}

impl From<&str> for RelationError {
    fn from(msg: &str) -> Self {
        RelationError::Message(msg.to_string())
    }
}

impl From<String> for RelationError {
    fn from(msg: String) -> Self {
        RelationError::Message(msg)
    }
}
