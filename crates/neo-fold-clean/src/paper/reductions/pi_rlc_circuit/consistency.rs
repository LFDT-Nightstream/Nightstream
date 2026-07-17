//! Implementation sidecar-consistency branch adjacent to Π_RLC.V.
//!
//! Owns: repeated `s_col` and fold-digest equalities.
//!
//! Does not own: paper CE arithmetic, claim validity, transcript challenges,
//! or a proof that either equality family is necessary.
//!
//! Emits constraints: yes.
//!
//! Authority boundary: all inputs are pinned to the checked parent value;
//! equality of prover-supplied digests alone is not treated as validity.
//!
//! | Stage path | Function | Equation | Multiplicity | Emitted rows/formula | Lowered gate | Lean theorem |
//! |---|---|---|---:|---|---|---|
//! | `consistency.s_col` | `enforce_rlc_s_col_consistency` | `input_i.s_col[j]=parent.s_col[j]` | inputs × coordinates × 2 limbs | one equality each | linear | `repeatedAuthorityBindings_iff_parentBinding` |
//! | `consistency.fold_digest` | `enforce_rlc_fold_digest_consistency` | `input_i.digest[j]=parent.digest[j]` | inputs × digest lanes | one equality each | linear | `repeatedAuthorityBindings_iff_parentBinding` |

use super::Error;
use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};

/// Enforce the shared NC column-domain point without mixing it by rho.
pub fn enforce_rlc_s_col_consistency(
    builder: &mut R1csBuilder,
    input_s_cols: &[Vec<KVar>],
    combined_s_col: &[KVar],
) -> Result<(), Error> {
    if input_s_cols.is_empty() {
        return Err(Error::Empty);
    }
    let len = combined_s_col.len();
    for (idx, s_col) in input_s_cols.iter().enumerate() {
        if s_col.len() != len {
            return Err(Error::ShapeMismatch {
                what: "s_col length",
                expected: format!("{len}"),
                got: format!("{} at idx {idx}", s_col.len()),
            });
        }
        for (input, combined) in s_col.iter().zip(combined_s_col.iter()) {
            builder.enforce_eq(&Lc::from_var(input.c0), &Lc::from_var(combined.c0));
            builder.enforce_eq(&Lc::from_var(input.c1), &Lc::from_var(combined.c1));
        }
    }
    Ok(())
}

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
