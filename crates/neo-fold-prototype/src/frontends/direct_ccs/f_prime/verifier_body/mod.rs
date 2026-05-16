//! Exports the verifier-shaped direct `F'` body, excluding terminal checks.
//!
//! This module owns the R1CS surface that should eventually be lowered and
//! folded as Construction-2 `enc(F')`. It deliberately uses `prove_final_ce =
//! false`: final semantic CE consistency belongs only at terminal compression.

use super::super::adapter::{direct_sparse_r1cs_export_from_spartan_circuit, DirectSparseR1csExport};
use super::super::state::{DirectCcsFPrimeSnarkError, DirectCcsIvcState};
use super::super::terminal::measure::{measure_direct_ccs_f_prime_constraints, DirectCcsFPrimeConstraintBreakdown};
use crate::superneo_nifs_circuit::SuperNeoNifsChunkFullBreakdown;

pub const DIRECT_CCS_F_PRIME_VERIFIER_BODY_DEFAULT_MEASURE_ROW_LIMIT: usize = 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectCcsFPrimeVerifierBodyShape {
    pub public_inputs: usize,
    pub constraints: usize,
    pub nifs: DirectCcsFPrimeVerifierBodyNifsShape,
    pub public_link_constraints: usize,
    pub construction2_fold_constraints: usize,
    pub chunk_done_constraints: usize,
    pub final_ce_relation_constraints: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectCcsFPrimeVerifierBodyNifsShape {
    pub chunk_constraints_by_chunk: Vec<usize>,
    pub chunk_meta_constraints: usize,
    pub pi_ccs_constraints: usize,
    pub pi_rlc_constraints: usize,
    pub pi_dec_constraints: usize,
}

impl DirectCcsFPrimeVerifierBodyShape {
    pub fn nifs_constraints(&self) -> usize {
        self.nifs.constraints()
    }

    fn from_terminal_measurement(measured: DirectCcsFPrimeConstraintBreakdown) -> Self {
        Self {
            public_inputs: measured.public_inputs,
            constraints: measured.terminal_f_prime_constraints,
            nifs: DirectCcsFPrimeVerifierBodyNifsShape::from_terminal_measurement(&measured),
            public_link_constraints: measured.public_link_constraints,
            construction2_fold_constraints: measured.construction2_fold_constraints,
            chunk_done_constraints: measured.chunk_done_constraints,
            final_ce_relation_constraints: 0,
        }
    }
}

impl DirectCcsFPrimeVerifierBodyNifsShape {
    pub fn constraints(&self) -> usize {
        self.chunk_constraints_by_chunk.iter().sum()
    }

    fn from_terminal_measurement(measured: &DirectCcsFPrimeConstraintBreakdown) -> Self {
        Self {
            chunk_constraints_by_chunk: measured.chunk_constraints_by_chunk.clone(),
            chunk_meta_constraints: sum_nifs_stage(&measured.chunk_stage_breakdowns, |breakdown| {
                breakdown.stages.chunk_meta
            }),
            pi_ccs_constraints: sum_nifs_stage(&measured.chunk_stage_breakdowns, |breakdown| breakdown.stages.pi_ccs),
            pi_rlc_constraints: sum_nifs_stage(&measured.chunk_stage_breakdowns, |breakdown| breakdown.stages.pi_rlc),
            pi_dec_constraints: sum_nifs_stage(&measured.chunk_stage_breakdowns, |breakdown| breakdown.stages.pi_dec),
        }
    }
}

fn sum_nifs_stage(
    chunks: &[SuperNeoNifsChunkFullBreakdown],
    stage_rows: impl Fn(&SuperNeoNifsChunkFullBreakdown) -> usize,
) -> usize {
    chunks.iter().map(stage_rows).sum()
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
    Ok(DirectCcsFPrimeVerifierBodyShape::from_terminal_measurement(measured))
}

pub fn export_latest_direct_ccs_f_prime_verifier_body_r1cs(
    state: &DirectCcsIvcState,
) -> Result<DirectSparseR1csExport, DirectCcsFPrimeSnarkError> {
    let circuit = state.latest_circuit()?.terminal_circuit(false);
    direct_sparse_r1cs_export_from_spartan_circuit(&circuit)
}
