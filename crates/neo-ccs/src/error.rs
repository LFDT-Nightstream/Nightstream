use thiserror::Error;

/// Dimension mismatch details.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimMismatch {
    /// expected (rows, cols)
    pub expected: (usize, usize),
    /// got (rows, cols)
    pub got: (usize, usize),
}

/// Errors thrown by CCS checks and consistency checks.
#[derive(Debug, Error)]
pub enum CcsError {
    /// A dimension mismatch occurred.
    #[error("dimension mismatch in {context}: expected {expected:?}, got {got:?}")]
    Dim {
        /// Where the error occurred.
        context: &'static str,
        /// Expected dims.
        expected: (usize, usize),
        /// Got dims.
        got: (usize, usize),
    },

    /// An invalid length (usually vector length) occurred.
    #[error("length mismatch in {context}: expected {expected}, got {got}")]
    Len {
        /// Where.
        context: &'static str,
        /// Expected.
        expected: usize,
        /// Got.
        got: usize,
    },

    /// n must be a power-of-two (so that n = 2^ell matches len(r)).
    #[error("n must be a power-of-two; got n={n}")]
    NNotPowerOfTwo {
        /// The bad n.
        n: usize,
    },

    /// A relation failed on a specific row (index) with a nonzero residual.
    #[error("relation failed at row {row}: residual is nonzero")]
    RowFail {
        /// Row index.
        row: usize,
    },

    /// General relation error with message.
    #[error("relation error: {0}")]
    Relation(#[from] RelationError),
}

/// Errors thrown when building or validating a relation instance.
#[derive(Debug, Error)]
pub enum RelationError {
    /// The structure contains an empty list of matrices or inconsistent shapes.
    #[error("invalid structure: matrices are empty or have inconsistent shapes")]
    InvalidStructure,

    /// Polynomial arity does not match t.
    #[error("polynomial arity mismatch: poly arity {poly_arity} vs t={t}")]
    PolyArity {
        /// Polynomial arity
        poly_arity: usize,
        /// Expected number of matrices
        t: usize,
    },

    /// General string error message
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
