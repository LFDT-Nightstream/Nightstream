use thiserror::Error;

mod assignment_transport;
pub(crate) use assignment_transport::encode_field;
pub(crate) mod json;
use json::canonical_field;
mod matrix_program;
pub(crate) use matrix_program::{external_outputs, permutation_rows, Entries};
mod v1_2;
pub(crate) use v1_2::{
    BLOCK_LOCALS, BLOCK_OUTPUTS, COMMITMENT_WORDS, DECOMPOSITION_BASE, DECOMPOSITION_K_RHO, DECOMPOSITION_NORM_BOUND,
    EVALUATIONS, PI_CCS_V1_2_MATRIX_COUNT, PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT, RING_DEGREE, RING_NODES, SBOX_ROWS,
};
pub use v1_2::{PI_CCS_V1_2_ROUND_COUNT, PUBLIC_WORDS};
mod pi_ccs_v1_2_transcript;
pub use pi_ccs_v1_2_transcript::{
    derive_light_transcript, derive_refresh_transcript, PiCcsV1_2Transcript, LIGHT_DOMAIN, REFRESH_DOMAIN,
};
#[derive(Debug, Error)]
pub enum PackageError {
    #[error("failed to read circuit package: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid circuit-package JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("circuit-package bytes are not the canonical Lean encoding")]
    NonCanonicalBytes,
    #[error("malformed circuit package: {0}")]
    Malformed(&'static str),
    #[error("circuit package exceeds a size limit: {0}")]
    TooLarge(&'static str),
    /// A caller argument is invalid; the package is not at fault.
    #[error("invalid input: {0}")]
    Input(&'static str),
    #[error("{what}: expected {expected}, got {actual}")]
    InputLength {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("unsatisfied assertion row {row}")]
    UnsatisfiedAssertionRow { row: usize },
    #[error("noncanonical field word in {location}: {value}")]
    NonCanonicalField { location: &'static str, value: u64 },
    #[error("{kind} does not match: expected {expected:?}, computed {computed:?}")]
    IdentityMismatch {
        kind: IdentityKind,
        expected: [u64; 4],
        computed: [u64; 4],
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentityKind {
    Relation,
    Package,
    VerificationKeyBinding,
    VerifierContext,
}

impl std::fmt::Display for IdentityKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Relation => "relation identifier",
            Self::Package => "final package identity",
            Self::VerificationKeyBinding => "verification-key binding",
            Self::VerifierContext => "verifier-context package identity",
        })
    }
}
