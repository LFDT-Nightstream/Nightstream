//! Physical profile recovery for the steady-recursive PiCCS output digest.
//!
//! Owns: recovering the source count, matrix count, and serialized field
//! count from retained stage markers and seeded Phi81 block geometry.
//!
//! Does not own: output validity, source authority, SIS/Poseidon2 refinement,
//! transcript binding, constraint costs, or permission to remove rows.
//!
//! Emits constraints: no.
//!
//! Authority boundary: stage labels and compact blocks are compiler evidence,
//! not semantic authority. The recovered profile is useful only after it is
//! compared with an independent protocol specification.
//!
//! | Physical owner | Recovered fact | Check |
//! |---|---|---|
//! | repeated `output_message_hashes.digest.preimage.source_headers` stages | source count | at least one marker |
//! | primary seeded Phi81 block between SIS and claim stages | serialized field count | exactly two ordered blocks in the interval |
//! | canonical output-message layout | matrix count | exactly one profile produces that field count |
//! | typed `y_zcol` source columns and PiRLC input evaluators | cross-phase consumer binding | all 1,620 columns equal coordinate-for-coordinate |
//! | retained Poseidon2 trace | final envelope schedule | 64 inputs, 16 absorbs, one pad, 17 matched permutations |

mod envelope;
mod hash;
mod projection;
mod sis;

pub use envelope::PiCcsOutputEnvelopePrefixAudit;
pub use projection::{
    PiCcsOutputYZcolProducerEntryAudit, PiCcsOutputYZcolProjectionAudit, PiCcsOutputYZcolProjectionInputAudit,
    PiRlcYZcolKMulAudit, PiRlcYZcolLinearCombinationAudit, PiRlcYZcolPolynomialEvaluationAudit,
    PiRlcYZcolProductFactorAudit, PiRlcYZcolProductIdentityAudit, PiRlcYZcolProjectionIdentityAudit,
    PiRlcYZcolProjectionLeafRowMappingAudit, PiRlcYZcolProjectionLimbAudit, PiRlcYZcolProjectionLoweredFragmentAudit,
    PiRlcYZcolProjectionLoweringDisposition, PiRlcYZcolProjectionProfileAudit, PiRlcYZcolProjectionRowAudit,
    PiRlcYZcolProjectionRowMappingAudit, PiRlcYZcolProjectionSharedAudit,
};
pub use sis::{CanonicalOpeningAudit, CanonicalOpeningPlacement, PiCcsOutputSisPhysicalAudit, SeededPhi81BlockAudit};

use crate::engine::r1cs_circuit::builder::{Poseidon2HashAudit, BALANCED_TERNARY_DIGITS};
use crate::engine::r1cs_circuit::PhysicalStageRange;
use crate::frontends::r1cs_f_prime::{SelectiveRowMappingAudit, SparseR1cs};
use crate::paper::reductions::pi_ccs_output_message::Profile;
use crate::paper::reductions::pi_ccs_split_nc_circuit::stage;

use super::R1csIvcError;

/// Exact PiCCS output-message dimensions recovered from one physical arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PiCcsOutputDigestProfileAudit {
    source_count: usize,
    matrix_count: usize,
    output_field_count: usize,
}

impl PiCcsOutputDigestProfileAudit {
    pub fn source_count(self) -> usize {
        self.source_count
    }

    pub fn matrix_count(self) -> usize {
        self.matrix_count
    }

    pub fn output_field_count(self) -> usize {
        self.output_field_count
    }
}

/// Compact physical audit of the stabilized output-evaluation digest path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiCcsOutputDigestAudit {
    profile: PiCcsOutputDigestProfileAudit,
    sis: PiCcsOutputSisPhysicalAudit,
    y_zcol_projection: PiCcsOutputYZcolProjectionAudit,
    envelope_prefix: PiCcsOutputEnvelopePrefixAudit,
    hash: Poseidon2HashAudit,
}

impl PiCcsOutputDigestAudit {
    pub fn profile(&self) -> PiCcsOutputDigestProfileAudit {
        self.profile
    }

    pub fn sis(&self) -> &PiCcsOutputSisPhysicalAudit {
        &self.sis
    }

    pub fn y_zcol_projection(&self) -> &PiCcsOutputYZcolProjectionAudit {
        &self.y_zcol_projection
    }

    pub fn envelope_prefix(&self) -> &PiCcsOutputEnvelopePrefixAudit {
        &self.envelope_prefix
    }

    pub fn hash(&self) -> &Poseidon2HashAudit {
        &self.hash
    }
}

pub(super) fn recover(
    arm: &SparseR1cs,
    rows: &SelectiveRowMappingAudit,
    arm_index: usize,
) -> Result<PiCcsOutputDigestAudit, R1csIvcError> {
    let stages = arm.physical_stage_ranges();
    let source_count = stages
        .iter()
        .filter(|range| range.path() == stage::OUTPUT_MESSAGE_PREIMAGE_SOURCE_HEADERS)
        .count();
    if source_count == 0 {
        return Err(invalid("physical arm contains no PiCCS output source-header stage"));
    }

    let sis_start = unique_stage_start(stages, stage::OUTPUT_MESSAGE_SIS)?;
    let claim_start = unique_stage_start(stages, stage::OUTPUT_MESSAGE_CLAIM)?;
    if sis_start >= claim_start {
        return Err(invalid(format!(
            "PiCCS output SIS starts at row {sis_start}, but its claim starts at row {claim_start}"
        )));
    }

    let mut blocks = arm
        .a
        .seeded_phi81_blocks()
        .iter()
        .filter(|block| sis_start <= block.row_start() && block.row_end() <= claim_start)
        .collect::<Vec<_>>();
    blocks.sort_by_key(|block| block.row_start());
    let [primary, compression] = blocks.as_slice() else {
        return Err(invalid(format!(
            "expected exactly two seeded Phi81 blocks between PiCCS output SIS row {sis_start} and claim row {claim_start}, found {}",
            blocks.len()
        )));
    };
    if primary.row_end() > compression.row_start() {
        return Err(invalid(format!(
            "PiCCS output seeded blocks overlap at rows {}..{} and {}..{}",
            primary.row_start(),
            primary.row_end(),
            compression.row_start(),
            compression.row_end()
        )));
    }
    if primary.word_width() != BALANCED_TERNARY_DIGITS || compression.word_width() != BALANCED_TERNARY_DIGITS {
        return Err(invalid(format!(
            "PiCCS output seeded blocks use word widths {} and {}, expected {BALANCED_TERNARY_DIGITS}",
            primary.word_width(),
            compression.word_width()
        )));
    }

    let output_field_count = primary.word_starts().len();
    if output_field_count == 0 || output_field_count <= compression.word_starts().len() {
        return Err(invalid(format!(
            "primary PiCCS output block has {output_field_count} words; digest compression has {}",
            compression.word_starts().len()
        )));
    }

    let mut matrix_count = None;
    for candidate in 0..=output_field_count {
        if Profile::new(source_count, candidate).field_count() != output_field_count {
            continue;
        }
        if matrix_count.replace(candidate).is_some() {
            return Err(invalid(format!(
                "PiCCS output stream with {source_count} sources and {output_field_count} fields has multiple compatible matrix counts"
            )));
        }
    }
    let matrix_count = matrix_count.ok_or_else(|| {
        invalid(format!(
            "PiCCS output stream with {source_count} sources and {output_field_count} fields has no compatible matrix count"
        ))
    })?;

    let profile = PiCcsOutputDigestProfileAudit {
        source_count,
        matrix_count,
        output_field_count,
    };
    let sis = sis::recover(
        arm,
        sis_start,
        claim_start,
        Profile::new(source_count, matrix_count),
        primary,
        compression,
    )?;
    let y_zcol_projection = projection::recover(arm, Profile::new(source_count, matrix_count), &sis, rows, arm_index)?;
    let hash = hash::recover(arm, sis_start, claim_start)?;
    let envelope_prefix = envelope::recover(
        arm,
        Profile::new(source_count, matrix_count),
        compression.row_end(),
        sis.compression().output_columns(),
        &hash,
    )?;
    Ok(PiCcsOutputDigestAudit {
        profile,
        sis,
        y_zcol_projection,
        envelope_prefix,
        hash,
    })
}

fn unique_stage_start(stages: &[PhysicalStageRange], path: &'static str) -> Result<usize, R1csIvcError> {
    let mut matching = stages.iter().filter(|range| range.path() == path);
    let Some(range) = matching.next() else {
        return Err(invalid(format!(
            "physical arm omits required PiCCS output stage `{path}`"
        )));
    };
    let start = range.row_start();
    let duplicates = matching.count();
    if duplicates != 0 {
        return Err(invalid(format!(
            "physical arm contains {} copies of unique PiCCS output stage `{path}`",
            duplicates + 1
        )));
    }
    Ok(start)
}

pub(super) fn invalid(detail: impl Into<String>) -> R1csIvcError {
    R1csIvcError::InvalidPiCcsOutputDigestAudit { detail: detail.into() }
}
