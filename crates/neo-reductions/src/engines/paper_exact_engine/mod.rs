pub mod paper_joint;
mod paper_matrix;
mod paper_ring;
mod paper_rows;
pub mod prove;
mod rlc_dec;
mod transcript;
pub mod verify;

pub use rlc_dec::{
    dec_reduction_paper_exact_with_commit_check, dec_reduction_paper_exact_with_rows,
    rlc_claim_paper_exact_with_commit_mix, rlc_reduction_paper_exact_with_commit_mix,
};

pub use paper_rows::PaperMatrixRows;
pub use prove::{paper_exact_prove, paper_exact_prove_with_rows};
pub use transcript::encode_proof;
pub use verify::{paper_exact_verify, paper_exact_verify_with_trace};
