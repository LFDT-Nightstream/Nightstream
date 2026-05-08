//! Exports the verifier-shaped direct `F'` body, excluding terminal checks.
//!
//! This module owns the R1CS surface that should eventually be lowered and
//! folded as Construction-2 `enc(F')`. It deliberately uses `prove_final_ce =
//! false`: final semantic CE consistency belongs only at terminal compression.

use super::ivc::{DirectCcsFPrimeSnarkError, DirectCcsIvcState};
use super::r1cs_export::DirectSparseR1csExport;
use super::terminal_measure::measure_direct_ccs_f_prime_constraints;

pub const DIRECT_CCS_F_PRIME_VERIFIER_BODY_DEFAULT_MEASURE_ROW_LIMIT: usize = 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectCcsFPrimeVerifierBodyShape {
    pub public_inputs: usize,
    pub constraints: usize,
    pub nifs_chunk_constraints_by_chunk: Vec<usize>,
    pub nifs_chunk_meta_constraints: usize,
    pub nifs_pi_ccs_constraints: usize,
    pub nifs_pi_rlc_constraints: usize,
    pub nifs_pi_dec_constraints: usize,
    pub public_link_constraints: usize,
    pub construction2_fold_constraints: usize,
    pub chunk_done_constraints: usize,
    pub final_ce_relation_constraints: usize,
}

impl DirectCcsFPrimeVerifierBodyShape {
    pub fn nifs_constraints(&self) -> usize {
        self.nifs_chunk_constraints_by_chunk.iter().sum()
    }
}

pub fn measure_latest_direct_ccs_f_prime_verifier_body_if_small(
    state: &DirectCcsIvcState,
) -> Result<Option<DirectCcsFPrimeVerifierBodyShape>, DirectCcsFPrimeSnarkError> {
    if state.structure().n > DIRECT_CCS_F_PRIME_VERIFIER_BODY_DEFAULT_MEASURE_ROW_LIMIT {
        return Ok(None);
    }
    measure_latest_direct_ccs_f_prime_verifier_body(state).map(Some)
}

pub fn measure_latest_direct_ccs_f_prime_verifier_body(
    state: &DirectCcsIvcState,
) -> Result<DirectCcsFPrimeVerifierBodyShape, DirectCcsFPrimeSnarkError> {
    let circuit = state.latest_circuit()?.terminal_circuit(false);
    let measured = measure_direct_ccs_f_prime_constraints(&circuit)?;
    let nifs_chunk_meta_constraints = measured
        .chunk_stage_breakdowns
        .iter()
        .map(|breakdown| breakdown.stages.chunk_meta)
        .sum();
    let nifs_pi_ccs_constraints = measured
        .chunk_stage_breakdowns
        .iter()
        .map(|breakdown| breakdown.stages.pi_ccs)
        .sum();
    let nifs_pi_rlc_constraints = measured
        .chunk_stage_breakdowns
        .iter()
        .map(|breakdown| breakdown.stages.pi_rlc)
        .sum();
    let nifs_pi_dec_constraints = measured
        .chunk_stage_breakdowns
        .iter()
        .map(|breakdown| breakdown.stages.pi_dec)
        .sum();
    Ok(DirectCcsFPrimeVerifierBodyShape {
        public_inputs: measured.public_inputs,
        constraints: measured.terminal_f_prime_constraints,
        nifs_chunk_constraints_by_chunk: measured.chunk_constraints_by_chunk,
        nifs_chunk_meta_constraints,
        nifs_pi_ccs_constraints,
        nifs_pi_rlc_constraints,
        nifs_pi_dec_constraints,
        public_link_constraints: measured.public_link_constraints,
        construction2_fold_constraints: measured.construction2_fold_constraints,
        chunk_done_constraints: measured.chunk_done_constraints,
        final_ce_relation_constraints: 0,
    })
}

pub fn export_latest_direct_ccs_f_prime_verifier_body_r1cs(
    state: &DirectCcsIvcState,
) -> Result<DirectSparseR1csExport, DirectCcsFPrimeSnarkError> {
    let circuit = state.latest_circuit()?.terminal_circuit(false);
    super::r1cs_export::direct_sparse_r1cs_export_from_spartan_circuit(&circuit)
}
