//! Owns the frozen Phase 0 payload encode/unpack/reconstruction semantics.
//!
//! It owns:
//! - schema full-width metadata
//! - u64-word to field-eval encoding for the six frozen v1 families
//! - packed-column packing/unpacking over `PackedColumnEval`
//! - verifier-computable word reconstruction from unpacked field evaluations
//!
//! It does not own:
//! - stage claim emission
//! - real Ajtai commitment-vector storage
//! - Phase 1 or Phase 2 convergence logic

use neo_math::{from_complex, KExtensions, D, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::opening_eval_claims::{EvalClaimError, FamilyEvalSchemaId, PackedColumnEval};

pub const PHASE0_WORD_LIMB_BITS: usize = 32;
pub const PHASE0_WORD_LIMB_COUNT: usize = 64 / PHASE0_WORD_LIMB_BITS;
const PHASE0_WORD_LIMB_MASK: u64 = (1u64 << PHASE0_WORD_LIMB_BITS) - 1;

pub fn phase0_full_width_for_schema(schema: FamilyEvalSchemaId) -> usize {
    1 + phase0_word_count_for_schema(schema) * PHASE0_WORD_LIMB_COUNT
}

pub fn phase0_word_count_for_schema(schema: FamilyEvalSchemaId) -> usize {
    match schema {
        FamilyEvalSchemaId::Stage1Rows => 23,
        FamilyEvalSchemaId::Stage2RegisterReads | FamilyEvalSchemaId::Stage2RegisterWrites => 5,
        FamilyEvalSchemaId::Stage2RamEvents
        | FamilyEvalSchemaId::Stage2TwistLinks
        | FamilyEvalSchemaId::Stage3Continuity => 6,
    }
}

pub(crate) fn encode_words_to_field_evals_f(
    schema: FamilyEvalSchemaId,
    words: &[u64],
) -> Result<Vec<F>, EvalClaimError> {
    let expected = phase0_word_count_for_schema(schema);
    let actual = words.len();
    if expected != actual {
        return Err(EvalClaimError::WordCountMismatch {
            schema,
            expected,
            actual,
        });
    }

    let mut out = Vec::with_capacity(phase0_full_width_for_schema(schema));
    out.push(F::ONE);
    for &word in words {
        for shift in (0..64).step_by(PHASE0_WORD_LIMB_BITS) {
            let limb = (word >> shift) & PHASE0_WORD_LIMB_MASK;
            out.push(F::from_u64(limb));
        }
    }
    Ok(out)
}

pub fn encode_words_to_field_evals_k(schema: FamilyEvalSchemaId, words: &[u64]) -> Result<Vec<K>, EvalClaimError> {
    Ok(encode_words_to_field_evals_f(schema, words)?
        .into_iter()
        .map(|value| from_complex(value, F::ZERO))
        .collect())
}

pub fn encode_packed_column_evals_k(
    schema: FamilyEvalSchemaId,
    field_evals: &[K],
) -> Result<Vec<PackedColumnEval>, EvalClaimError> {
    let expected = phase0_full_width_for_schema(schema);
    let actual = field_evals.len();
    if expected != actual {
        return Err(EvalClaimError::FieldEvalWidthMismatch {
            schema,
            expected,
            actual,
        });
    }

    let mut out = (0..schema.packed_column_count())
        .map(|_| PackedColumnEval {
            coeffs: std::array::from_fn(|_| K::ZERO),
        })
        .collect::<Vec<_>>();
    for (index, &value) in field_evals.iter().enumerate() {
        let column_index = index / D;
        let coeff_index = index % D;
        out[column_index].coeffs[coeff_index] = value;
    }
    Ok(out)
}

pub fn unpack_column_evals_k(
    schema: FamilyEvalSchemaId,
    column_evals: &[PackedColumnEval],
) -> Result<Vec<K>, EvalClaimError> {
    let expected = schema.packed_column_count();
    let actual = column_evals.len();
    if expected != actual {
        return Err(EvalClaimError::PackedColumnCountMismatch {
            schema,
            expected,
            actual,
        });
    }

    let full_width = phase0_full_width_for_schema(schema);
    let mut out = Vec::with_capacity(full_width);
    for index in 0..full_width {
        let column_index = index / D;
        let coeff_index = index % D;
        out.push(column_evals[column_index].coeffs[coeff_index]);
    }
    Ok(out)
}

pub fn reconstruct_words_from_field_evals(
    schema: FamilyEvalSchemaId,
    field_evals: &[K],
) -> Result<Vec<u64>, EvalClaimError> {
    let expected = phase0_full_width_for_schema(schema);
    let actual = field_evals.len();
    if expected != actual {
        return Err(EvalClaimError::FieldEvalWidthMismatch {
            schema,
            expected,
            actual,
        });
    }
    if field_evals.first().copied() != Some(K::ONE) {
        return Err(EvalClaimError::FieldEvalLeadingOneMismatch { schema });
    }

    let logical_values = &field_evals[1..];
    let mut words = Vec::with_capacity(phase0_word_count_for_schema(schema));
    for (word_index, word_limbs) in logical_values
        .chunks_exact(PHASE0_WORD_LIMB_COUNT)
        .enumerate()
    {
        let mut word = 0u64;
        for (limb_index, limb) in word_limbs.iter().enumerate() {
            let [real, imag] = limb.as_coeffs();
            if imag != F::ZERO {
                return Err(EvalClaimError::NonBaseFieldLimb {
                    schema,
                    word_index,
                    limb_index,
                });
            }
            let value = real.as_canonical_u64();
            if value > PHASE0_WORD_LIMB_MASK {
                return Err(EvalClaimError::LimbOutOfRange {
                    schema,
                    word_index,
                    limb_index,
                    value,
                });
            }
            word |= value << (limb_index * PHASE0_WORD_LIMB_BITS);
        }
        words.push(word);
    }
    Ok(words)
}
