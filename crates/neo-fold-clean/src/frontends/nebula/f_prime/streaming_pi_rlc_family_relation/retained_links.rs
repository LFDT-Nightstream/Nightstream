//! Exact normalized slot audit for the production PiRLC body-overlay links.
//!
//! Owns exhaustive comparison of the two parity link maps with the source
//! field contracts and the final low-norm slots used by the link compiler.
//! The family schedule and the separate overlay receipt prove that these two
//! maps cover all 110 families. This module does not own selector authority,
//! shifted-ternary canonicality, row satisfaction, recursive orchestration,
//! or commitment hardness.

use super::retained_overlay::{
    FINAL_ACTIVE_DIGIT_START, FINAL_OUTPUT_START, FINAL_ZERO_DIGIT_START, INPUT_WIDTH, OUTPUT_RADIX, OUTPUT_WIDTH,
};
use super::{
    build_production_pi_rlc_family_body_low_norm_r1cs, production_pi_rlc_family_overlay_link_runs,
    production_pi_rlc_family_overlay_links_for_family, NebulaFPrimePiRlcFamilyBodySynthesis,
    NebulaFPrimePiRlcFamilyRelationError, COMMITMENT_OUTPUT_FIELDS, DIGIT_COUNT, FAMILY_INPUT_FIELDS,
    PI_RLC_FAMILY_BODY_PUBLIC_OUTPUTS, PI_RLC_FAMILY_COUNT, PI_RLC_FAMILY_LINK_FIELDS,
};
use crate::frontends::nebula::f_prime::NebulaFPrimePiRlcFamilyReplayArmKind;

const SCHEMA_VERSION: u64 = 1;
const BODY_FINAL_COLUMNS: usize = 2_521_314;
const OVERLAY_FINAL_COLUMNS: usize = 35_856;
const PARITY_COUNT: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiRlcFamilyNormalizedLinkRunAudit {
    body_source_start: usize,
    overlay_source_start: usize,
    outer_count: usize,
    body_source_stride: usize,
    overlay_source_stride: usize,
    field_count: usize,
    body_final_start: usize,
    overlay_final_start: usize,
    final_outer_stride: usize,
    final_field_stride: usize,
    width: usize,
    radix: u64,
}

impl NebulaFPrimePiRlcFamilyNormalizedLinkRunAudit {
    pub const fn body_source_start(self) -> usize {
        self.body_source_start
    }

    pub const fn overlay_source_start(self) -> usize {
        self.overlay_source_start
    }

    pub const fn outer_count(self) -> usize {
        self.outer_count
    }

    pub const fn body_source_stride(self) -> usize {
        self.body_source_stride
    }

    pub const fn overlay_source_stride(self) -> usize {
        self.overlay_source_stride
    }

    pub const fn field_count(self) -> usize {
        self.field_count
    }

    pub const fn body_final_start(self) -> usize {
        self.body_final_start
    }

    pub const fn overlay_final_start(self) -> usize {
        self.overlay_final_start
    }

    pub const fn final_outer_stride(self) -> usize {
        self.final_outer_stride
    }

    pub const fn final_field_stride(self) -> usize {
        self.final_field_stride
    }

    pub const fn width(self) -> usize {
        self.width
    }

    pub const fn radix(self) -> u64 {
        self.radix
    }

    pub const fn link_count(self) -> usize {
        self.outer_count * self.field_count
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiRlcFamilyNormalizedLinkAudit {
    schema_version: u64,
    family_count: usize,
    parity_count: usize,
    public_output_count: usize,
    body_final_columns: usize,
    overlay_final_columns: usize,
    link_count_per_family: usize,
    total_link_count: usize,
    phase_kinds: [usize; PARITY_COUNT],
    runs: [NebulaFPrimePiRlcFamilyNormalizedLinkRunAudit; 3],
}

impl NebulaFPrimePiRlcFamilyNormalizedLinkAudit {
    pub const fn schema_version(&self) -> u64 {
        self.schema_version
    }

    pub const fn family_count(&self) -> usize {
        self.family_count
    }

    pub const fn parity_count(&self) -> usize {
        self.parity_count
    }

    pub const fn public_output_count(&self) -> usize {
        self.public_output_count
    }

    pub const fn body_final_columns(&self) -> usize {
        self.body_final_columns
    }

    pub const fn overlay_final_columns(&self) -> usize {
        self.overlay_final_columns
    }

    pub const fn link_count_per_family(&self) -> usize {
        self.link_count_per_family
    }

    pub const fn total_link_count(&self) -> usize {
        self.total_link_count
    }

    pub const fn phase_kinds(&self) -> [usize; PARITY_COUNT] {
        self.phase_kinds
    }

    pub const fn runs(&self) -> [NebulaFPrimePiRlcFamilyNormalizedLinkRunAudit; 3] {
        self.runs
    }
}

fn link_error(reason: impl Into<String>) -> NebulaFPrimePiRlcFamilyRelationError {
    NebulaFPrimePiRlcFamilyRelationError::NormalizedLinks(reason.into())
}

const fn expected_runs() -> [NebulaFPrimePiRlcFamilyNormalizedLinkRunAudit; 3] {
    [
        NebulaFPrimePiRlcFamilyNormalizedLinkRunAudit {
            body_source_start: 46_055,
            overlay_source_start: 1,
            outer_count: 1,
            body_source_stride: 41,
            overlay_source_stride: 41,
            field_count: 41,
            body_final_start: 1_059_804,
            overlay_final_start: FINAL_ZERO_DIGIT_START,
            final_outer_stride: 41,
            final_field_stride: 1,
            width: INPUT_WIDTH,
            radix: 2,
        },
        NebulaFPrimePiRlcFamilyNormalizedLinkRunAudit {
            body_source_start: 46_096,
            overlay_source_start: 42,
            outer_count: FAMILY_INPUT_FIELDS,
            body_source_stride: 122,
            overlay_source_stride: DIGIT_COUNT,
            field_count: DIGIT_COUNT,
            body_final_start: 19_332,
            overlay_final_start: FINAL_ACTIVE_DIGIT_START,
            final_outer_stride: DIGIT_COUNT,
            final_field_stride: 1,
            width: INPUT_WIDTH,
            radix: 2,
        },
        NebulaFPrimePiRlcFamilyNormalizedLinkRunAudit {
            body_source_start: 144_918,
            overlay_source_start: 33_252,
            outer_count: 1,
            body_source_stride: COMMITMENT_OUTPUT_FIELDS,
            overlay_source_stride: COMMITMENT_OUTPUT_FIELDS,
            field_count: COMMITMENT_OUTPUT_FIELDS,
            body_final_start: 1_076_091,
            overlay_final_start: FINAL_OUTPUT_START,
            final_outer_stride: COMMITMENT_OUTPUT_FIELDS * OUTPUT_WIDTH,
            final_field_stride: OUTPUT_WIDTH,
            width: OUTPUT_WIDTH,
            radix: OUTPUT_RADIX,
        },
    ]
}

/// Audit the exact normalized source fields and final low-norm slots read by
/// every production PiRLC body-overlay equality row.
pub fn production_pi_rlc_family_normalized_link_audit(
) -> Result<NebulaFPrimePiRlcFamilyNormalizedLinkAudit, NebulaFPrimePiRlcFamilyRelationError> {
    let expected = expected_runs();
    let source_runs = production_pi_rlc_family_overlay_link_runs();
    for (index, (source, run)) in source_runs.into_iter().zip(expected).enumerate() {
        if source.phase_field_start() != run.body_source_start
            || source.overlay_field_start() != run.overlay_source_start
            || source.outer_count() != run.outer_count
            || source.phase_stride() != run.body_source_stride
            || source.overlay_stride() != run.overlay_source_stride
            || source.field_count() != run.field_count
        {
            return Err(link_error(format!(
                "source link run {index} differs from the exact recipe"
            )));
        }
    }

    for (parity, kind) in [
        NebulaFPrimePiRlcFamilyReplayArmKind::Even,
        NebulaFPrimePiRlcFamilyReplayArmKind::Odd,
    ]
    .into_iter()
    .enumerate()
    {
        let body = NebulaFPrimePiRlcFamilyBodySynthesis::production(kind);
        if body.public_columns() != PI_RLC_FAMILY_BODY_PUBLIC_OUTPUTS + 1 {
            return Err(link_error(format!(
                "parity {parity} has the wrong normalized public prefix"
            )));
        }
        let first_public_source = (0..PI_RLC_FAMILY_BODY_PUBLIC_OUTPUTS)
            .filter_map(|index| body.public_output_column(index))
            .min()
            .ok_or_else(|| link_error(format!("parity {parity} has no public output columns")))?;
        let last_linked_source = expected
            .into_iter()
            .map(|run| {
                run.body_source_start - PI_RLC_FAMILY_BODY_PUBLIC_OUTPUTS
                    + (run.outer_count - 1) * run.body_source_stride
                    + run.field_count
                    - 1
            })
            .max()
            .expect("three link runs");
        if first_public_source <= last_linked_source {
            return Err(link_error(format!(
                "parity {parity} moves a public output through the linked private prefix"
            )));
        }
        for run in expected {
            for (outer, field) in [(0, 0), (run.outer_count - 1, run.field_count - 1)] {
                let physical =
                    run.body_source_start - PI_RLC_FAMILY_BODY_PUBLIC_OUTPUTS + outer * run.body_source_stride + field;
                let normalized = run.body_source_start + outer * run.body_source_stride + field;
                if body.normalized_field_column_for_artifact(physical) != Some(normalized) {
                    return Err(link_error(format!(
                        "parity {parity} source field {physical} does not normalize to {normalized}"
                    )));
                }
            }
        }
    }

    let body_relation = build_production_pi_rlc_family_body_low_norm_r1cs()?;
    if body_relation.structure().m != BODY_FINAL_COLUMNS {
        return Err(link_error("normalized body has the wrong final column count"));
    }

    let source_runs = production_pi_rlc_family_overlay_link_runs();
    for parity in 0..PARITY_COUNT {
        let family = parity;
        let contract = production_pi_rlc_family_overlay_links_for_family(0, family, source_runs);
        if contract.overlay_kind != family
            || contract.phase_kind != 10 + parity
            || contract.fields.len() != PI_RLC_FAMILY_LINK_FIELDS
        {
            return Err(link_error(format!(
                "family {family} link header differs from the exact recipe"
            )));
        }
        let mut cursor = 0;
        for run in expected {
            for outer in 0..run.outer_count {
                for field in 0..run.field_count {
                    let link = &contract.fields[cursor];
                    cursor += 1;
                    let body_source = run.body_source_start + outer * run.body_source_stride + field;
                    let overlay_source = run.overlay_source_start + outer * run.overlay_source_stride + field;
                    let body_final =
                        run.body_final_start + outer * run.final_outer_stride + field * run.final_field_stride;
                    let overlay_final =
                        run.overlay_final_start + outer * run.final_outer_stride + field * run.final_field_stride;
                    if link.phase_field != body_source
                        || link.overlay_field != overlay_source
                        || body_relation.field_slot(parity, body_source) != Some((body_final, run.width))
                        || overlay_final + run.width > OVERLAY_FINAL_COLUMNS
                    {
                        return Err(link_error(format!(
                            "family {family} link {} differs from its normalized source or final slot",
                            cursor - 1
                        )));
                    }
                }
            }
        }
        if cursor != PI_RLC_FAMILY_LINK_FIELDS {
            return Err(link_error(format!("family {family} link census drifted")));
        }
    }

    Ok(NebulaFPrimePiRlcFamilyNormalizedLinkAudit {
        schema_version: SCHEMA_VERSION,
        family_count: PI_RLC_FAMILY_COUNT,
        parity_count: PARITY_COUNT,
        public_output_count: PI_RLC_FAMILY_BODY_PUBLIC_OUTPUTS,
        body_final_columns: BODY_FINAL_COLUMNS,
        overlay_final_columns: OVERLAY_FINAL_COLUMNS,
        link_count_per_family: PI_RLC_FAMILY_LINK_FIELDS,
        total_link_count: PI_RLC_FAMILY_COUNT * PI_RLC_FAMILY_LINK_FIELDS,
        phase_kinds: [10, 11],
        runs: expected,
    })
}

const _: () = assert!(PI_RLC_FAMILY_BODY_PUBLIC_OUTPUTS == 640);
const _: () = assert!(PI_RLC_FAMILY_LINK_FIELDS == 33_359);
const _: () = assert!(PI_RLC_FAMILY_COUNT * PI_RLC_FAMILY_LINK_FIELDS == 3_669_490);
