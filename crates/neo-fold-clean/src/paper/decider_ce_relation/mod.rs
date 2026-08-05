//! Terminal CE relation for the selected one-joint accumulator.
//!
//! This module closes final claims against their opened witnesses. It checks
//! the Ajtai opening, public-input projection, complete low-norm carrier,
//! identity-first ring evaluations at the joint point, and their constant
//! terms. Transcript replay and compact terminal proof backends are owned by
//! the decider orchestration.

mod commitment;
mod evaluation;
mod witness;

pub(crate) use commitment::{enforce_ajtai_opening, enforce_ajtai_slice_opening, enforce_x_projection};
pub(crate) use evaluation::{enforce_ct_from_y_ring, enforce_y_ring_from_z_at_r};
pub(crate) use witness::alloc_final_witness;

use thiserror::Error;

use crate::engine::r1cs_circuit::builder::TerminalCeClaimAudit;
use crate::engine::r1cs_circuit::R1csBuilder;
use crate::lifecycle::Preprocessing;
use crate::paper::reductions::pi_dec_circuit::CeClaimWires;
use crate::paper::relations::WitnessMat;

#[derive(Debug, Error)]
pub(crate) enum CeRelationError {
    #[error("decider_ce_relation: claim/witness count mismatch (claims={claims}, witnesses={witnesses})")]
    CountMismatch { claims: usize, witnesses: usize },
    #[error("decider_ce_relation: claim {index} {what} shape mismatch (expected {expected}, got {got})")]
    ShapeMismatch {
        index: usize,
        what: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("decider_ce_relation: Ajtai global setup unavailable for d={d}, cols={cols}")]
    AjtaiSetupMissing { d: usize, cols: usize },
    #[error("decider_ce_relation: balanced-alphabet gadget is undefined for b={b}")]
    InvalidNormBound { b: u32 },
    #[error("decider_ce_relation: claim {index} Nebula adv presence does not match preprocessing")]
    NebulaAdvPresence { index: usize },
}

/// Close final CE claims against the authoritative witness openings.
pub(crate) fn enforce_final_ce_relations(
    builder: &mut R1csBuilder,
    prep: &Preprocessing,
    final_claims_wires: &[CeClaimWires],
    final_witnesses: &[WitnessMat],
) -> Result<(), CeRelationError> {
    enforce_claim_family(builder, prep, final_claims_wires, final_witnesses)
}

/// Close the strict PiDEC child claims against their witness openings.
pub(crate) fn enforce_final_dec_children_relations(
    builder: &mut R1csBuilder,
    prep: &Preprocessing,
    final_claims_wires: &[CeClaimWires],
    final_witnesses: &[WitnessMat],
) -> Result<(), CeRelationError> {
    enforce_claim_family(builder, prep, final_claims_wires, final_witnesses)
}

fn enforce_claim_family(
    builder: &mut R1csBuilder,
    prep: &Preprocessing,
    claims: &[CeClaimWires],
    witnesses: &[WitnessMat],
) -> Result<(), CeRelationError> {
    validate_count(claims, witnesses)?;
    let expected_m = prep.structure().m;
    for (index, (claim, witness)) in claims.iter().zip(witnesses).enumerate() {
        let witness_wires =
            alloc_final_witness(builder, witness, expected_m).map_err(|error| witness_shape_err(index, error))?;
        enforce_one_claim(builder, prep, index, claim, &witness_wires)?;
    }
    Ok(())
}

fn enforce_one_claim(
    builder: &mut R1csBuilder,
    prep: &Preprocessing,
    index: usize,
    claim: &CeClaimWires,
    witness_wires: &witness::FinalWitnessWires,
) -> Result<(), CeRelationError> {
    let structure = prep.structure();
    let b = prep.params.b();
    let claim_start = builder.rows();
    let claim_first_column = builder.cols();
    validate_claim_shape(prep, index, claim)?;

    let phase_start = builder.rows();
    enforce_ajtai_opening(
        builder,
        &prep.log,
        witness_wires,
        &claim.c_data,
        claim.c_d,
        claim.c_kappa,
    )
    .map_err(|error| ajtai_setup_err(index, error))?;

    match (prep.nebula(), claim.adv.as_ref()) {
        (None, None) => {}
        (Some(nebula), Some(adv)) => {
            let ops_pp = nebula.scheme.ops_module().verification_pp().map_err(|_| {
                let (d, cols) = nebula.scheme.ops_module().dims();
                CeRelationError::AjtaiSetupMissing { d, cols }
            })?;
            let mem_pp = nebula.scheme.mem_module().verification_pp().map_err(|_| {
                let (d, cols) = nebula.scheme.mem_module().dims();
                CeRelationError::AjtaiSetupMissing { d, cols }
            })?;
            let ranges = nebula.scheme.ranges();
            for (commitment, columns, pp) in [
                (&adv.ops, ranges.ops.clone(), ops_pp.as_ref()),
                (&adv.is, ranges.is.clone(), mem_pp.as_ref()),
                (&adv.fs, ranges.fs.clone(), mem_pp.as_ref()),
            ] {
                enforce_ajtai_slice_opening(
                    builder,
                    witness_wires,
                    &commitment.data,
                    commitment.d,
                    commitment.kappa,
                    columns,
                    pp,
                )
                .map_err(|error| ajtai_setup_err(index, error))?;
            }
        }
        _ => return Err(CeRelationError::NebulaAdvPresence { index }),
    }
    builder.record_row_family("terminal_ce.claim.commitment", phase_start);

    let phase_start = builder.rows();
    enforce_x_projection(builder, witness_wires, claim).map_err(|error| projection_err(index, error))?;
    builder.record_row_family("terminal_ce.claim.public_input", phase_start);

    let phase_start = builder.rows();
    let norm_first_allocated_column = builder.cols();
    witness::enforce_balanced_alphabet(builder, witness_wires, b)
        .map_err(|error| CeRelationError::InvalidNormBound { b: error.b })?;
    builder.record_row_family("terminal_ce.claim.norm", phase_start);

    let phase_start = builder.rows();
    enforce_y_ring_from_z_at_r(builder, prep, witness_wires, claim).map_err(|error| y_ring_err(index, error))?;
    builder.record_row_family("terminal_ce.claim.evaluations", phase_start);

    let phase_start = builder.rows();
    enforce_ct_from_y_ring(builder, claim).map_err(|error| y_ring_err(index, error))?;
    builder.record_row_family("terminal_ce.claim.constant_term", phase_start);

    builder.record_terminal_ce_claim(TerminalCeClaimAudit {
        row_start: claim_start,
        row_end: builder.rows(),
        first_allocated_column: claim_first_column,
        norm_bound: b,
        expected_public_width: prep.public_input_len,
        structure_rows: structure.n,
        structure_columns: structure.m,
        witness_rows: witness_wires.rows,
        witness_columns: witness_wires.cols,
        witness_cols: witness_wires.values.iter().map(|wire| wire.col()).collect(),
        norm_first_allocated_column,
        commitment_cols: claim.c_data.iter().map(|wire| wire.col()).collect(),
        commitment_d: claim.c_d,
        commitment_kappa: claim.c_kappa,
        public_cols: claim.x.iter().map(|wire| wire.col()).collect(),
        public_rows: claim.x_rows,
        public_width: claim.x_cols,
        public_input_len: claim.m_in,
        point_cols: claim
            .r
            .iter()
            .map(|value| [value.c0.col(), value.c1.col()])
            .collect(),
        evaluation_cols: claim
            .y_ring
            .iter()
            .map(|row| row.iter().map(|wire| wire.col()).collect())
            .collect(),
        constant_term_cols: claim
            .ct
            .iter()
            .map(|value| [value.c0.col(), value.c1.col()])
            .collect(),
    });
    builder.record_program_range("terminal_ce.claim", claim_start, claim_first_column);
    Ok(())
}

fn validate_count(claims: &[CeClaimWires], witnesses: &[WitnessMat]) -> Result<(), CeRelationError> {
    if claims.len() != witnesses.len() {
        return Err(CeRelationError::CountMismatch {
            claims: claims.len(),
            witnesses: witnesses.len(),
        });
    }
    Ok(())
}

fn validate_claim_shape(prep: &Preprocessing, index: usize, claim: &CeClaimWires) -> Result<(), CeRelationError> {
    let expected_m = prep.structure().m;
    if claim.m_in > expected_m {
        return Err(CeRelationError::ShapeMismatch {
            index,
            what: "m_in vs structure.m",
            expected: expected_m,
            got: claim.m_in,
        });
    }
    if let Some(expected) = prep.public_input_len {
        if claim.m_in != expected {
            return Err(CeRelationError::ShapeMismatch {
                index,
                what: "m_in vs prep.public_input_len",
                expected,
                got: claim.m_in,
            });
        }
    }
    Ok(())
}

fn witness_shape_err(index: usize, error: witness::AllocError) -> CeRelationError {
    CeRelationError::ShapeMismatch {
        index,
        what: error.what(),
        expected: error.expected(),
        got: error.got(),
    }
}

fn ajtai_setup_err(index: usize, error: commitment::AjtaiOpeningError) -> CeRelationError {
    match error {
        commitment::AjtaiOpeningError::SetupMissing { d, cols } => CeRelationError::AjtaiSetupMissing { d, cols },
        commitment::AjtaiOpeningError::Shape { what, expected, got } => CeRelationError::ShapeMismatch {
            index,
            what,
            expected,
            got,
        },
    }
}

fn projection_err(index: usize, error: commitment::XProjectionError) -> CeRelationError {
    CeRelationError::ShapeMismatch {
        index,
        what: error.what(),
        expected: error.expected(),
        got: error.got(),
    }
}

fn y_ring_err(index: usize, error: evaluation::YRingError) -> CeRelationError {
    CeRelationError::ShapeMismatch {
        index,
        what: error.what(),
        expected: error.expected(),
        got: error.got(),
    }
}
