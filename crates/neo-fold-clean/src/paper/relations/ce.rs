//! Definition 13 — CE claims.
//!
//! Type alias over `neo_ccs::CeClaim`. The full struct fields (commitment,
//! public input matrix X, evaluation point r, ring-digit y_j, aux_openings,
//! …) live in `neo-ccs`; we keep one paper-named alias here so the audit
//! glossary maps exactly.

use neo_ajtai::Commitment;
use neo_ccs::CeClaim as NeoCeClaim;
use neo_math::{F, K};

/// CE claim — Definition 13.
pub type CeClaim = NeoCeClaim<Commitment, F, K>;
