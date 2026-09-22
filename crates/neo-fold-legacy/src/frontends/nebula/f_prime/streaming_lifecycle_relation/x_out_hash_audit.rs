//! Exact normalized XOut preimage and public-hash audit metadata.

use neo_math::F;

use crate::engine::r1cs_circuit::builder::{CanonicalU64Audit, Poseidon2HashAudit};
use crate::frontends::r1cs_f_prime::SparseR1cs;

use super::{NebulaFPrimeRelationError, NebulaFPrimeStreamingPublicLayout, X_OUT_PREIMAGE_FIELDS};

/// Exact source binding from one XOut digest field to its private canonical
/// decomposition and verifier-owned public bits.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingXOutPublicWordAudit {
    canonical: CanonicalU64Audit,
    canonical_rows: std::ops::Range<usize>,
    public_bit_cols: [usize; 64],
    equality_rows: [usize; 64],
}

impl NebulaFPrimeStreamingXOutPublicWordAudit {
    pub const fn field_col(&self) -> usize {
        self.canonical.field_col
    }

    pub const fn canonical_bit_cols(&self) -> &[usize; 64] {
        &self.canonical.bit_cols
    }

    pub fn canonical_rows(&self) -> std::ops::Range<usize> {
        self.canonical_rows.clone()
    }

    pub const fn high_is_max_col(&self) -> usize {
        self.canonical.high_is_max_col
    }

    pub const fn inverse_col(&self) -> usize {
        self.canonical.inverse_col
    }

    pub const fn public_bit_cols(&self) -> &[usize; 64] {
        &self.public_bit_cols
    }

    pub const fn equality_rows(&self) -> &[usize; 64] {
        &self.equality_rows
    }
}

/// Exact normalized source-row metadata for one after-state XOut hash and
/// its four canonical public words. The source matrices remain authoritative.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingXOutHashAudit {
    hash: Poseidon2HashAudit,
    public_words: [NebulaFPrimeStreamingXOutPublicWordAudit; 4],
}

impl NebulaFPrimeStreamingXOutHashAudit {
    pub fn hash(&self) -> &Poseidon2HashAudit {
        &self.hash
    }

    pub const fn public_words(&self) -> &[NebulaFPrimeStreamingXOutPublicWordAudit; 4] {
        &self.public_words
    }
}

/// Exact normalized source columns consumed by the before-state and
/// after-state XOut Poseidon2 rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingXOutPreimageColumns {
    pub(super) before: [usize; X_OUT_PREIMAGE_FIELDS],
    pub(super) after: [usize; X_OUT_PREIMAGE_FIELDS],
}

impl NebulaFPrimeStreamingXOutPreimageColumns {
    pub fn before(&self) -> &[usize; X_OUT_PREIMAGE_FIELDS] {
        &self.before
    }

    pub fn after(&self) -> &[usize; X_OUT_PREIMAGE_FIELDS] {
        &self.after
    }
}

/// Exact semantic values consumed by the before-state and after-state XOut
/// Poseidon2 rows, captured before source-column normalization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingXOutPreimageValues {
    pub(super) before: [F; X_OUT_PREIMAGE_FIELDS],
    pub(super) after: [F; X_OUT_PREIMAGE_FIELDS],
}

impl NebulaFPrimeStreamingXOutPreimageValues {
    pub fn before(&self) -> &[F; X_OUT_PREIMAGE_FIELDS] {
        &self.before
    }

    pub fn after(&self) -> &[F; X_OUT_PREIMAGE_FIELDS] {
        &self.after
    }

    /// Check the complete typed lifecycle target against one normalized
    /// source assignment. This compares values, not digests.
    pub fn is_satisfied_by(&self, columns: &NebulaFPrimeStreamingXOutPreimageColumns, assignment: &[F]) -> bool {
        self.before
            .iter()
            .zip(columns.before())
            .chain(self.after.iter().zip(columns.after()))
            .all(|(expected, column)| assignment.get(*column) == Some(expected))
    }
}

pub(super) fn exact_after_x_out_hash_audit(
    source: &SparseR1cs,
    x_out: &NebulaFPrimeStreamingXOutPreimageColumns,
) -> Result<NebulaFPrimeStreamingXOutHashAudit, NebulaFPrimeRelationError> {
    let matches = source
        .poseidon2_hash_audits()
        .iter()
        .filter(|audit| audit.input_cols.as_slice() == x_out.after())
        .cloned()
        .collect::<Vec<_>>();
    let [hash] = matches.as_slice() else {
        return Err(NebulaFPrimeRelationError::Geometry(format!(
            "streaming lifecycle after-state XOut hash match count {} != 1",
            matches.len()
        )));
    };
    let public_bits = NebulaFPrimeStreamingPublicLayout::production().after_state_digest_bits();
    if public_bits.len() != 4 * 64 || public_bits.end > source.m_in {
        return Err(NebulaFPrimeRelationError::Geometry(format!(
            "streaming lifecycle after-state XOut public range {}..{} is outside public width {}",
            public_bits.start, public_bits.end, source.m_in
        )));
    }
    let public_words = hash
        .output_cols
        .iter()
        .enumerate()
        .map(|(lane, &field_col)| {
            let matches = source
                .canonical_u64_decompositions()
                .iter()
                .filter(|decomposition| decomposition.field_col == field_col)
                .collect::<Vec<_>>();
            let [decomposition] = matches.as_slice() else {
                return Err(NebulaFPrimeRelationError::Geometry(format!(
                    "streaming lifecycle after-state XOut lane {lane} canonical match count {} != 1",
                    matches.len()
                )));
            };
            let public_bit_cols = std::array::from_fn(|bit| public_bits.start + lane * 64 + bit);
            let canonical_rows = source.canonical_u64_source_rows(field_col).ok_or_else(|| {
                NebulaFPrimeRelationError::Geometry(format!(
                    "streaming lifecycle after-state XOut lane {lane} has no unique canonical-u64 source range"
                ))
            })?;
            if canonical_rows.len() != 69 || canonical_rows.end > source.n {
                return Err(NebulaFPrimeRelationError::Geometry(format!(
                    "streaming lifecycle after-state XOut lane {lane} canonical-u64 row range {:?} is invalid for {} rows",
                    canonical_rows, source.n
                )));
            }
            let equality_rows = decomposition
                .bit_cols
                .iter()
                .zip(public_bit_cols)
                .enumerate()
                .map(|(bit, (&canonical, public))| {
                    let matches = source
                        .equality_pairs()
                        .iter()
                        .filter(|&&(_, lhs, rhs)| {
                            (lhs == canonical && rhs == public) || (lhs == public && rhs == canonical)
                        })
                        .map(|&(row, _, _)| row)
                        .collect::<Vec<_>>();
                    let [row] = matches.as_slice() else {
                        return Err(NebulaFPrimeRelationError::Geometry(format!(
                            "streaming lifecycle after-state XOut lane {lane} bit {bit} equality match count {} != 1",
                            matches.len()
                        )));
                    };
                    Ok(*row)
                })
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .map_err(|rows: Vec<usize>| {
                    NebulaFPrimeRelationError::Geometry(format!(
                        "streaming lifecycle after-state XOut lane {lane} equality row count {} != 64",
                        rows.len()
                    ))
                })?;
            Ok(NebulaFPrimeStreamingXOutPublicWordAudit {
                canonical: CanonicalU64Audit {
                    field_col: decomposition.field_col,
                    bit_cols: decomposition.bit_cols,
                    high_is_max_col: decomposition.high_is_max_col,
                    inverse_col: decomposition.inverse_col,
                },
                canonical_rows,
                public_bit_cols,
                equality_rows,
            })
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|words: Vec<NebulaFPrimeStreamingXOutPublicWordAudit>| {
            NebulaFPrimeRelationError::Geometry(format!(
                "streaming lifecycle after-state XOut public word count {} != 4",
                words.len()
            ))
        })?;
    Ok(NebulaFPrimeStreamingXOutHashAudit {
        hash: hash.clone(),
        public_words,
    })
}
