//! Compact lifecycle boundary shared by the phased Nebula F-prime relation.
//!
//! This module keeps only the state-XOut, cursor, and semantic-envelope rows
//! that every scheduled work item needs. The monolithic base/recursive F-prime
//! circuits remain the reference source for correspondence audits; they are not
//! embedded as the common component of every phased work item.

use std::ops::Range;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::frontends::r1cs_f_prime::{
    lower_field_r1cs, normalized_field_column, FieldR1csLoweringError, SparseR1cs,
};
use crate::paper::construction2::TRIVIAL_PC;
use crate::paper::digest::StateXOutDigestMode;
use crate::paper::f_prime::digest_circuit::StateXOutDigestInputs;

use super::super::streaming_lifecycle_relation::NebulaFPrimeStreamingLifecycleArm;
use super::super::streaming_phase_envelope::{
    enforce_streaming_lifecycle_semantic_link, streaming_phase_semantic_digest,
    StreamingLifecycleSemanticLinkWires, STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS,
};
use super::super::streaming_public::NebulaFPrimeStreamingPublicLayout;
use super::super::streaming_state_envelope::enforce_streaming_state_x_out;

const KERNEL_BEFORE_LOCAL_STATE_FAMILY: &str =
    "fprime.streaming.lifecycle.kernel.before.local_state_digest";
const KERNEL_BEFORE_DELAYED_PAYLOAD_FAMILY: &str =
    "fprime.streaming.lifecycle.kernel.before.delayed_payload.raw_bits";
const KERNEL_AFTER_LOCAL_STATE_FAMILY: &str =
    "fprime.streaming.lifecycle.kernel.after.local_state_digest";
const KERNEL_AFTER_DELAYED_PAYLOAD_FAMILY: &str =
    "fprime.streaming.lifecycle.kernel.after.delayed_payload.raw_bits";
const KERNEL_BASE_INITIAL_ROWS_FAMILY: &str =
    "fprime.streaming.lifecycle.kernel.base.initial_rows";
const KERNEL_CURSOR_ROWS_FAMILY: &str = "fprime.streaming.lifecycle.kernel.cursor_rows";

#[derive(Debug, Error)]
pub(super) enum NebulaFPrimeStreamingLifecycleKernelError {
    #[error(transparent)]
    Lowering(#[from] FieldR1csLoweringError),
    #[error("compact streaming lifecycle kernel: {0}")]
    Geometry(String),
}

#[derive(Clone, Debug)]
pub(super) struct NebulaFPrimeStreamingLifecycleKernelEnvelopeFields {
    before_local_state_digest: Range<usize>,
    before_delayed_payload: Range<usize>,
    after_local_state_digest: Range<usize>,
    after_delayed_payload: Range<usize>,
}

impl NebulaFPrimeStreamingLifecycleKernelEnvelopeFields {
    pub(super) fn before_local_state_digest(&self) -> Range<usize> {
        self.before_local_state_digest.clone()
    }

    pub(super) fn before_delayed_payload(&self) -> Range<usize> {
        self.before_delayed_payload.clone()
    }

    pub(super) fn after_local_state_digest(&self) -> Range<usize> {
        self.after_local_state_digest.clone()
    }

    pub(super) fn after_delayed_payload(&self) -> Range<usize> {
        self.after_delayed_payload.clone()
    }
}

/// Exact compact base and recursive source arms plus satisfying Rust
/// assignments. These source assignments are retained for correspondence and
/// same-assignment tests; production composition consumes the two shapes.
pub(super) struct NebulaFPrimeStreamingLifecycleKernelSourceArms {
    arms: [SparseR1cs; 2],
    assignments: [Vec<F>; 2],
    phase_envelope_fields: [NebulaFPrimeStreamingLifecycleKernelEnvelopeFields; 2],
    recursive_prior_state_digest_columns: [usize; 4],
}

impl NebulaFPrimeStreamingLifecycleKernelSourceArms {
    pub(super) fn arms(&self) -> &[SparseR1cs; 2] {
        &self.arms
    }

    pub(super) fn arm(&self, arm: NebulaFPrimeStreamingLifecycleArm) -> &SparseR1cs {
        &self.arms[arm.index()]
    }

    pub(super) fn assignment(&self, arm: NebulaFPrimeStreamingLifecycleArm) -> &[F] {
        &self.assignments[arm.index()]
    }

    pub(super) fn phase_envelope_fields(
        &self,
        arm: NebulaFPrimeStreamingLifecycleArm,
    ) -> &NebulaFPrimeStreamingLifecycleKernelEnvelopeFields {
        &self.phase_envelope_fields[arm.index()]
    }

    pub(super) const fn recursive_prior_state_digest_columns(&self) -> [usize; 4] {
        self.recursive_prior_state_digest_columns
    }
}

struct SynthesizedKernelArm {
    source: SparseR1cs,
    assignment: Vec<F>,
    phase_envelope_fields: NebulaFPrimeStreamingLifecycleKernelEnvelopeFields,
    before_x_out_digest_columns: [usize; 4],
}

pub(super) fn production_streaming_lifecycle_kernel_source_arms(
) -> Result<NebulaFPrimeStreamingLifecycleKernelSourceArms, NebulaFPrimeStreamingLifecycleKernelError> {
    let base = synthesize_kernel_arm(NebulaFPrimeStreamingLifecycleArm::Base)?;
    let recursive = synthesize_kernel_arm(NebulaFPrimeStreamingLifecycleArm::Recursive)?;
    Ok(NebulaFPrimeStreamingLifecycleKernelSourceArms {
        arms: [base.source, recursive.source],
        assignments: [base.assignment, recursive.assignment],
        phase_envelope_fields: [base.phase_envelope_fields, recursive.phase_envelope_fields],
        recursive_prior_state_digest_columns: recursive.before_x_out_digest_columns,
    })
}

fn synthesize_kernel_arm(
    arm: NebulaFPrimeStreamingLifecycleArm,
) -> Result<SynthesizedKernelArm, NebulaFPrimeStreamingLifecycleKernelError> {
    let public = NebulaFPrimeStreamingPublicLayout::production();
    let before_cursor_value = match arm {
        NebulaFPrimeStreamingLifecycleArm::Base => 0,
        NebulaFPrimeStreamingLifecycleArm::Recursive => 17,
    };
    let after_cursor_value = before_cursor_value + 1;
    let before_local_values = digest_values(100 + 20 * arm.index());
    let after_local_values = digest_values(110 + 20 * arm.index());
    let before_payload_values = match arm {
        NebulaFPrimeStreamingLifecycleArm::Base => {
            vec![F::ZERO; STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS]
        }
        NebulaFPrimeStreamingLifecycleArm::Recursive => {
            (0..STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS)
                .map(|index| F::from_bool(index % 3 == 0))
                .collect()
        }
    };
    let after_payload_values = (0..STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS)
        .map(|index| F::from_bool(index % 5 == 0))
        .collect::<Vec<_>>();
    let before_semantic_values =
        streaming_phase_semantic_digest(before_local_values, &before_payload_values);
    let after_semantic_values =
        streaming_phase_semantic_digest(after_local_values, &after_payload_values);

    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage(match arm {
        NebulaFPrimeStreamingLifecycleArm::Base => "nebula.streaming.lifecycle.kernel.base",
        NebulaFPrimeStreamingLifecycleArm::Recursive => {
            "nebula.streaming.lifecycle.kernel.recursive"
        }
    });

    let before_semantic = alloc_digest(&mut builder, before_semantic_values);
    let after_semantic = alloc_digest(&mut builder, after_semantic_values);

    let before_local_start = builder.cols();
    let before_local = alloc_digest(&mut builder, before_local_values);
    builder.record_column_family(KERNEL_BEFORE_LOCAL_STATE_FAMILY, before_local_start);

    let before_payload_start = builder.cols();
    let before_payload = builder.alloc_vec(&before_payload_values);
    builder.record_column_family(
        KERNEL_BEFORE_DELAYED_PAYLOAD_FAMILY,
        before_payload_start,
    );

    let after_local_start = builder.cols();
    let after_local = alloc_digest(&mut builder, after_local_values);
    builder.record_column_family(KERNEL_AFTER_LOCAL_STATE_FAMILY, after_local_start);

    let after_payload_start = builder.cols();
    let after_payload = builder.alloc_vec(&after_payload_values);
    builder.record_column_family(KERNEL_AFTER_DELAYED_PAYLOAD_FAMILY, after_payload_start);

    enforce_streaming_lifecycle_semantic_link(
        &mut builder,
        StreamingLifecycleSemanticLinkWires {
            before_semantic_digest: before_semantic,
            after_semantic_digest: after_semantic,
            before_local_state_digest: before_local,
            after_local_state_digest: after_local,
            before_delayed_payload: &before_payload,
            after_delayed_payload: &after_payload,
        },
    );

    if arm == NebulaFPrimeStreamingLifecycleArm::Base {
        let row_start = builder.rows();
        for &bit in &before_payload {
            builder.enforce_zero(&Lc::from_var(bit));
        }
        builder.record_row_family(KERNEL_BASE_INITIAL_ROWS_FAMILY, row_start);
    }

    let cursor_row_start = builder.rows();
    let before_cursor = builder.alloc(F::from_usize(before_cursor_value));
    let after_cursor = builder.alloc(F::from_usize(after_cursor_value));
    let successor =
        Lc::from_var(before_cursor).add_scaled(&Lc::from_const(F::ONE), F::ONE);
    builder.enforce_eq(&Lc::from_var(after_cursor), &successor);
    if arm == NebulaFPrimeStreamingLifecycleArm::Base {
        builder.enforce_zero(&Lc::from_var(before_cursor));
    }
    builder.record_row_family(KERNEL_CURSOR_ROWS_FAMILY, cursor_row_start);

    let verifier_digest = alloc_fixture_digest(&mut builder, 1_000);
    let header_digest = alloc_fixture_digest(&mut builder, 1_100);
    let initial_boundary = alloc_fixture_digest(&mut builder, 1_200);
    let before_boundary = alloc_fixture_digest(&mut builder, 1_300 + 20 * arm.index());
    let after_boundary = alloc_fixture_digest(&mut builder, 1_310 + 20 * arm.index());
    let before_accumulator =
        alloc_fixture_digest(&mut builder, 1_400 + 20 * arm.index());
    let after_accumulator =
        alloc_fixture_digest(&mut builder, 1_410 + 20 * arm.index());
    let before_lane_digest =
        alloc_fixture_digest(&mut builder, 1_500 + 20 * arm.index());
    let after_lane_digest =
        alloc_fixture_digest(&mut builder, 1_510 + 20 * arm.index());
    let pc = alloc_bound_constant(&mut builder, TRIVIAL_PC as usize);

    let before_x_out = enforce_streaming_state_x_out(
        &mut builder,
        &StateXOutDigestInputs {
            mode: StateXOutDigestMode::Stateful,
            vk_fs_digest: verifier_digest,
            pi_ccs_header_bundle: header_digest,
            structure_digest: header_digest,
            chunk_count: before_cursor,
            step_count: before_cursor,
            initial_boundary,
            current_boundary: before_boundary,
            pc,
            semantic_acc: before_semantic,
            construction2_acc: before_accumulator,
            public_trace: before_boundary,
        },
        before_lane_digest,
    );
    let after_x_out = enforce_streaming_state_x_out(
        &mut builder,
        &StateXOutDigestInputs {
            mode: StateXOutDigestMode::Stateful,
            vk_fs_digest: verifier_digest,
            pi_ccs_header_bundle: header_digest,
            structure_digest: header_digest,
            chunk_count: after_cursor,
            step_count: after_cursor,
            initial_boundary,
            current_boundary: after_boundary,
            pc,
            semantic_acc: after_semantic,
            construction2_acc: after_accumulator,
            public_trace: after_boundary,
        },
        after_lane_digest,
    );

    let before_cursor_bits = decompose_var_to_u64_bits(&mut builder, before_cursor);
    let after_cursor_bits = decompose_var_to_u64_bits(&mut builder, after_cursor);
    let mut public_outputs = Vec::with_capacity(public.logical_columns() - 1);
    public_outputs.extend(after_x_out.public_bits);
    public_outputs.extend(before_x_out.public_bits);
    public_outputs.extend(before_cursor_bits);
    public_outputs.extend(after_cursor_bits);
    if public_outputs.len() + 1 != public.logical_columns() {
        return Err(NebulaFPrimeStreamingLifecycleKernelError::Geometry(format!(
            "logical public width {} != {}",
            public_outputs.len() + 1,
            public.logical_columns()
        )));
    }

    let source_columns = builder.cols();
    let before_x_out_digest_columns = before_x_out
        .digest
        .into_iter()
        .map(|wire| {
            normalized_field_column(source_columns, &public_outputs, wire.col()).ok_or_else(|| {
                NebulaFPrimeStreamingLifecycleKernelError::Geometry(format!(
                    "before XOut digest column {} is outside source width {source_columns}",
                    wire.col()
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|columns: Vec<usize>| {
            NebulaFPrimeStreamingLifecycleKernelError::Geometry(format!(
                "before XOut digest width {} != 4",
                columns.len()
            ))
        })?;

    builder.begin_encoding_stage("complete");
    if let Some(row) = builder.first_unsatisfied_row() {
        return Err(NebulaFPrimeStreamingLifecycleKernelError::Geometry(format!(
            "synthesized {arm:?} source is unsatisfied at row {row}"
        )));
    }
    let (source, assignment) = lower_field_r1cs(builder, &public_outputs)?.into_parts();
    if source.m_in != public.logical_columns() {
        return Err(NebulaFPrimeStreamingLifecycleKernelError::Geometry(format!(
            "normalized public width {} != {}",
            source.m_in,
            public.logical_columns()
        )));
    }
    source.is_satisfied_by(&assignment).map_err(|error| {
        NebulaFPrimeStreamingLifecycleKernelError::Geometry(error.to_string())
    })?;
    let phase_envelope_fields = exact_phase_envelope_fields(&source)?;

    Ok(SynthesizedKernelArm {
        source,
        assignment,
        phase_envelope_fields,
        before_x_out_digest_columns,
    })
}

fn exact_phase_envelope_fields(
    source: &SparseR1cs,
) -> Result<
    NebulaFPrimeStreamingLifecycleKernelEnvelopeFields,
    NebulaFPrimeStreamingLifecycleKernelError,
> {
    Ok(NebulaFPrimeStreamingLifecycleKernelEnvelopeFields {
        before_local_state_digest: exact_private_family(
            source,
            KERNEL_BEFORE_LOCAL_STATE_FAMILY,
            4,
        )?,
        before_delayed_payload: exact_private_family(
            source,
            KERNEL_BEFORE_DELAYED_PAYLOAD_FAMILY,
            STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS,
        )?,
        after_local_state_digest: exact_private_family(
            source,
            KERNEL_AFTER_LOCAL_STATE_FAMILY,
            4,
        )?,
        after_delayed_payload: exact_private_family(
            source,
            KERNEL_AFTER_DELAYED_PAYLOAD_FAMILY,
            STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS,
        )?,
    })
}

fn exact_private_family(
    source: &SparseR1cs,
    name: &'static str,
    expected_len: usize,
) -> Result<Range<usize>, NebulaFPrimeStreamingLifecycleKernelError> {
    let mut matches = source
        .column_family_ranges()
        .iter()
        .filter(|family| family.name == name);
    let family = matches.next().ok_or_else(|| {
        NebulaFPrimeStreamingLifecycleKernelError::Geometry(format!(
            "source is missing {name}"
        ))
    })?;
    if matches.next().is_some() {
        return Err(NebulaFPrimeStreamingLifecycleKernelError::Geometry(format!(
            "source contains duplicate {name} ranges"
        )));
    }
    let range = family.column_start..family.column_end;
    if range.len() != expected_len || range.start < source.m_in || range.end > source.m {
        return Err(NebulaFPrimeStreamingLifecycleKernelError::Geometry(format!(
            "{name} range [{}, {}) is not an exact private width-{expected_len} range inside {}..{}",
            range.start, range.end, source.m_in, source.m
        )));
    }
    Ok(range)
}

fn digest_values(start: usize) -> [F; 4] {
    std::array::from_fn(|lane| F::from_usize(start + lane))
}

fn alloc_digest(builder: &mut R1csBuilder, values: [F; 4]) -> [Var; 4] {
    values.map(|value| builder.alloc(value))
}

fn alloc_fixture_digest(builder: &mut R1csBuilder, start: usize) -> [Var; 4] {
    alloc_digest(builder, digest_values(start))
}

fn alloc_bound_constant(builder: &mut R1csBuilder, value: usize) -> Var {
    let value = F::from_usize(value);
    let wire = builder.alloc(value);
    builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(value));
    wire
}

#[cfg(test)]
mod tests {
    use neo_math::{D, F};
    use p3_field::PrimeCharacteristicRing;

    use crate::frontends::r1cs_f_prime::build_multi_branch_selective_low_norm_r1cs_with_alignment;

    use super::*;

    #[test]
    fn exact_rust_assignments_satisfy_both_kernel_arms() {
        let lifecycle = production_streaming_lifecycle_kernel_source_arms().unwrap();
        for arm in [
            NebulaFPrimeStreamingLifecycleArm::Base,
            NebulaFPrimeStreamingLifecycleArm::Recursive,
        ] {
            lifecycle
                .arm(arm)
                .is_satisfied_by(lifecycle.assignment(arm))
                .unwrap();
        }
    }

    #[test]
    fn base_before_payload_is_zero_and_recursive_payload_is_binary() {
        let lifecycle = production_streaming_lifecycle_kernel_source_arms().unwrap();

        let base_fields =
            lifecycle.phase_envelope_fields(NebulaFPrimeStreamingLifecycleArm::Base);
        let mut base = lifecycle
            .assignment(NebulaFPrimeStreamingLifecycleArm::Base)
            .to_vec();
        base[base_fields.before_delayed_payload().start] = F::ONE;
        assert!(lifecycle
            .arm(NebulaFPrimeStreamingLifecycleArm::Base)
            .is_satisfied_by(&base)
            .is_err());

        let recursive_fields =
            lifecycle.phase_envelope_fields(NebulaFPrimeStreamingLifecycleArm::Recursive);
        let mut recursive = lifecycle
            .assignment(NebulaFPrimeStreamingLifecycleArm::Recursive)
            .to_vec();
        recursive[recursive_fields.before_delayed_payload().start] = F::from_u64(2);
        assert!(lifecycle
            .arm(NebulaFPrimeStreamingLifecycleArm::Recursive)
            .is_satisfied_by(&recursive)
            .is_err());
    }

    #[test]
    fn compiled_kernel_common_relation_fits_the_joint_domain() {
        let lifecycle = production_streaming_lifecycle_kernel_source_arms().unwrap();
        let common = build_multi_branch_selective_low_norm_r1cs_with_alignment(
            lifecycle.arms(),
            0,
            D,
            0,
        )
        .unwrap();
        assert!(common.structure().n <= 1 << 24);
        assert!(common.structure().m <= 16_777_206);
    }
}
