//! Fold-digest consistency for the in-circuit Π_RLC verifier.
//!
//! All input digests are pinned to the checked parent digest. Equality of
//! prover-supplied digests is not treated as proof of the folded relation.
//!
//! Owns: equality of every input fold digest with the combined fold digest.
//!
//! Does not own: digest computation, ring folding, or transcript sampling.
//!
//! Emits constraints: one equality for each digest field and input.
//!
//! | Input | Constraint |
//! | --- | --- |
//! | input digest field | equals the combined digest field |

use super::Error;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};

/// Enforce the shared fold digest without mixing it by rho.
pub fn enforce_rlc_fold_digest_consistency(
    builder: &mut R1csBuilder,
    input_fold_digests: &[&[Var]],
    combined_fold_digest: &[Var],
) -> Result<(), Error> {
    if input_fold_digests.is_empty() {
        return Err(Error::Empty);
    }
    let len = combined_fold_digest.len();
    for (idx, fold_digest) in input_fold_digests.iter().enumerate() {
        if fold_digest.len() != len {
            return Err(Error::ShapeMismatch {
                what: "fold digest length",
                expected: format!("{len}"),
                got: format!("{} at idx {idx}", fold_digest.len()),
            });
        }
        for (&input, &combined) in fold_digest.iter().zip(combined_fold_digest) {
            builder.enforce_eq(&Lc::from_var(input), &Lc::from_var(combined));
        }
    }
    Ok(())
}
