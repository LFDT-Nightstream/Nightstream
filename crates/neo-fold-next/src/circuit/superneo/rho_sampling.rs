//! Owns fixed-round Π_RLC rho sampling gadgets for the RV32IM main relation circuit.
//!
//! This mirrors the repo's circuit-friendly `sample_rot_rhos_n` contract:
//! transcript-bound digest bits are mapped into the Goldilocks strong-set
//! alphabet with a fixed number of transcript squeezes.

use crate::spartan_backend::SpartanF;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_math::{D, F};

use super::transcript::Poseidon2TranscriptCircuit;

mod candidates;
mod matrix;
mod words;

#[allow(unused_imports)]
pub use matrix::alloc_rot_rho_matrices_from_native;
pub use matrix::materialize_goldilocks_rot_matrices;

use candidates::{
    compact_first_accepted_goldilocks_coeffs, map_u16_bits_to_goldilocks_candidate, GoldilocksCoeffCandidateVar,
};
use words::digest_u16_words;

pub(super) const ALPHABET_LEN: usize = 5;
pub(super) const U16S_PER_DIGEST32: usize = 16;
pub(super) const DIGESTS_PER_RHO: usize = D.div_ceil(U16S_PER_DIGEST32);
pub(super) const CANDIDATE_WORDS_PER_RHO: usize = DIGESTS_PER_RHO * U16S_PER_DIGEST32;
pub(super) const RHO_REJECTION_SLACK: usize = CANDIDATE_WORDS_PER_RHO - D;
pub(super) const U16_MOD5_WEIGHTS: [u64; 16] = [1, 2, 4, 3, 1, 2, 4, 3, 1, 2, 4, 3, 1, 2, 4, 3];

#[derive(Clone)]
pub struct RotRhoVar {
    pub coeffs: Vec<AllocatedNum<SpartanF>>,
    pub coeff_values: Vec<F>,
}

impl RotRhoVar {
    pub(super) fn from_coeffs(coeffs: Vec<AllocatedNum<SpartanF>>, coeff_values: Vec<F>) -> Self {
        Self { coeffs, coeff_values }
    }
}

#[derive(Clone)]
pub struct RotRhoMatrixVar {
    rows: usize,
    cols: usize,
    entries: Vec<AllocatedNum<SpartanF>>,
    entry_values: Vec<F>,
}

impl RotRhoMatrixVar {
    pub(super) fn from_entries(
        rows: usize,
        cols: usize,
        entries: Vec<AllocatedNum<SpartanF>>,
        entry_values: Vec<F>,
    ) -> Self {
        Self {
            rows,
            cols,
            entries,
            entry_values,
        }
    }

    pub fn entry(&self, row: usize, col: usize) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
        if row >= self.rows || col >= self.cols {
            return Err(SynthesisError::Unsatisfiable);
        }
        Ok(self.entries[row * self.cols + col].clone())
    }

    pub fn entry_value(&self, row: usize, col: usize) -> Result<F, SynthesisError> {
        if row >= self.rows || col >= self.cols {
            return Err(SynthesisError::Unsatisfiable);
        }
        Ok(self.entry_values[row * self.cols + col])
    }
}

pub fn sample_goldilocks_rot_rhos<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    tr: &mut Poseidon2TranscriptCircuit,
    count: usize,
    label: &str,
) -> Result<Vec<RotRhoVar>, SynthesisError> {
    let mut out = Vec::with_capacity(count);
    for rho_idx in 0..count {
        tr.append_const_fields_raw(
            cs.namespace(|| format!("{label}_rho_index_{rho_idx}")),
            &[
                SpartanF::from_canonical_u64(0),
                SpartanF::from_canonical_u64(rho_idx as u64),
            ],
        )?;
        let mut candidates = Vec::with_capacity(CANDIDATE_WORDS_PER_RHO);
        for digest_idx in 0..DIGESTS_PER_RHO {
            tr.append_const_fields_raw(
                cs.namespace(|| format!("{label}_rho_chunk_msg_{rho_idx}_{digest_idx}")),
                &[
                    SpartanF::from_canonical_u64(1),
                    SpartanF::from_canonical_u64(rho_idx as u64 + digest_idx as u64),
                ],
            )?;
            let digest = tr.digest32(cs.namespace(|| format!("{label}_rho_digest_{rho_idx}_{digest_idx}")))?;
            let digest_values = core::array::from_fn(|idx| tr.state_values()[idx]);
            let words = digest_u16_words(
                cs.namespace(|| format!("{label}_rho_words_{rho_idx}_{digest_idx}")),
                &digest,
                &digest_values,
                &format!("{label}_rho_words_{rho_idx}_{digest_idx}"),
            )?;
            for (word_idx, (word_bits, word_value)) in words.into_iter().enumerate() {
                let (coeff, coeff_value, reject_bit, reject_value) = map_u16_bits_to_goldilocks_candidate(
                    cs.namespace(|| format!("{label}_rho_coeff_{rho_idx}_{digest_idx}_{word_idx}")),
                    &word_bits,
                    word_value,
                    &format!("{label}_rho_coeff_{rho_idx}_{digest_idx}_{word_idx}"),
                )?;
                candidates.push(GoldilocksCoeffCandidateVar {
                    coeff,
                    coeff_value,
                    reject_bit,
                    reject_value,
                });
            }
        }
        let (coeffs, coeff_values) = compact_first_accepted_goldilocks_coeffs(
            cs.namespace(|| format!("{label}_rho_accept_{rho_idx}")),
            &candidates,
            &format!("{label}_rho_accept_{rho_idx}"),
        )?;
        if coeffs.len() != D || coeff_values.len() != D {
            return Err(SynthesisError::Unsatisfiable);
        }
        out.push(RotRhoVar::from_coeffs(coeffs, coeff_values));
    }
    Ok(out)
}
