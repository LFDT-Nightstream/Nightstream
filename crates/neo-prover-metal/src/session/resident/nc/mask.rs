//! Mask-native NC rounds before the dense ring-lane crossover.
//!
//! The original signed masks are immutable. Challenges fold a small basis
//! table until one dense materialization is cheaper than further mask lookup.

use std::mem::size_of;

use neo_math::D;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::{MetalNcSumcheckPlan, NC_THREADS};
use crate::session::MetalSession;
use crate::{KWords, MetalError};

impl MetalSession {
    pub(super) fn encode_nc_mask_round(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        plan: &mut MetalNcSumcheckPlan,
        trace_round: Option<usize>,
    ) -> Result<(), MetalError> {
        let source = plan
            .mask_source
            .as_ref()
            .ok_or(MetalError::Shape("resident NC mask source is unavailable"))?;
        if source.round_encoded
            || source.folded
            || plan.dense
            || plan.rows < 2
            || plan.width > 32
            || !plan.width.is_power_of_two()
        {
            return Err(MetalError::Shape("resident NC mask round is out of sequence"));
        }
        let (round_shape, round_shape_offset) = trace_round.map_or((&plan.shape, 0), |round| {
            (&plan.round_shapes, round * 5 * size_of::<u64>())
        });
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_ccs.nc.mask_round")));
        encoder.setComputePipelineState(&self.nc_round_mask_partials);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.eq_tables[plan.current_slot]), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&source.masks), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(round_shape), round_shape_offset, 2);
            encoder.setBuffer_offset_atIndex(Some(&source.shape), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&plan.weights), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&source.basis[source.basis_slot]), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&plan.partials), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&source.active_witnesses), 0, 7);
        }
        let groups = (plan.rows / 2).div_ceil(64).max(1);
        self.dispatch_threadgroups(&encoder, &self.nc_round_mask_partials, groups, NC_THREADS);
        encoder.endEncoding();
        plan.mask_source
            .as_mut()
            .expect("NC mask source exists above")
            .round_encoded = true;
        Ok(())
    }

    pub(super) fn encode_nc_mask_fold(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        plan: &mut MetalNcSumcheckPlan,
        challenge: KWords,
    ) -> Result<(), MetalError> {
        self.write_shared(&plan.challenge, &[challenge.c0, challenge.c1])?;
        self.write_shared(
            &plan.fold_shape,
            &[
                plan.active_witness_count as u64,
                plan.rows as u64,
                plan.width as u64,
                u64::from(plan.dense),
            ],
        )?;
        self.encode_nc_mask_fold_from_buffer(command, plan, None)
    }

    pub(super) fn encode_nc_mask_trace_fold(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        plan: &mut MetalNcSumcheckPlan,
        round: usize,
        challenge_offset: usize,
    ) -> Result<(), MetalError> {
        self.encode_nc_mask_fold_from_buffer(command, plan, Some((round, challenge_offset)))
    }

    /// Advances the mask-native state after one encoded round, folding only the
    /// shared basis until the fixed crossover requires one dense materialization.
    fn encode_nc_mask_fold_from_buffer(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        plan: &mut MetalNcSumcheckPlan,
        trace: Option<(usize, usize)>,
    ) -> Result<(), MetalError> {
        let source = plan
            .mask_source
            .as_ref()
            .ok_or(MetalError::Shape("resident NC mask source is unavailable"))?;
        if !source.round_encoded
            || source.folded
            || plan.dense
            || plan.rows < 2
            || plan.width > 32
            || !plan.width.is_power_of_two()
            || (!source.direct_compact && plan.width != 1)
        {
            return Err(MetalError::Shape("resident NC mask fold is out of sequence"));
        }
        let direct_compact = source.direct_compact;
        let basis_slot = source.basis_slot;
        let next_basis_slot = basis_slot ^ 1;
        let next_slot = plan.current_slot ^ 1;
        let next_rows = plan.rows / 2;
        let (challenge, challenge_offset, fold_shape, fold_shape_offset) =
            trace.map_or((&plan.challenge, 0, &plan.fold_shape, 0), |(round, offset)| {
                (
                    &plan.challenge_log,
                    offset,
                    &plan.round_fold_shapes,
                    round * 4 * size_of::<u64>(),
                )
            });

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.fold_k_table);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.eq_tables[plan.current_slot]), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(challenge), challenge_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(&plan.eq_tables[next_slot]), 0, 2);
        }
        self.dispatch(&encoder, &self.fold_k_table, next_rows);
        encoder.endEncoding();

        if !direct_compact {
            // Small inputs pay for one direct mask-to-table fold and then use
            // the ordinary compact table kernels.
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setLabel(Some(&NSString::from_str("nightstream.pi_ccs.nc.mask_fold")));
            encoder.setComputePipelineState(&self.nc_fold_signed_masks);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&source.masks), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(challenge), challenge_offset, 1);
                encoder.setBuffer_offset_atIndex(Some(&source.shape), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&plan.digit_values[next_slot]), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&source.active_witnesses), 0, 4);
            }
            self.dispatch(
                &encoder,
                &self.nc_fold_signed_masks,
                plan.active_witness_count * next_rows,
            );
            encoder.endEncoding();
        } else {
            // Large inputs fold only the shared basis. At width 32 the next
            // window would exceed D, so materialize the dense state once.
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setLabel(Some(&NSString::from_str("nightstream.pi_ccs.nc.mask_basis")));
            encoder.setComputePipelineState(&self.nc_expand_mask_basis);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&source.basis[basis_slot]), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(challenge), challenge_offset, 1);
                encoder.setBuffer_offset_atIndex(Some(fold_shape), fold_shape_offset, 2);
                encoder.setBuffer_offset_atIndex(Some(&source.basis[next_basis_slot]), 0, 3);
            }
            self.dispatch(&encoder, &self.nc_expand_mask_basis, 2 * plan.width);
            encoder.endEncoding();

            if plan.width == 32 {
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setLabel(Some(&NSString::from_str("nightstream.pi_ccs.nc.mask_dense")));
                encoder.setComputePipelineState(&self.nc_materialize_mask_dense);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&source.masks), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&source.basis[next_basis_slot]), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&source.shape), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(fold_shape), fold_shape_offset, 3);
                    encoder.setBuffer_offset_atIndex(Some(&plan.digit_values[next_slot]), 0, 4);
                    encoder.setBuffer_offset_atIndex(Some(&source.active_witnesses), 0, 5);
                }
                self.dispatch(
                    &encoder,
                    &self.nc_materialize_mask_dense,
                    plan.active_witness_count * next_rows * D,
                );
                encoder.endEncoding();
            }
        }

        let source = plan
            .mask_source
            .as_mut()
            .expect("NC mask source exists above");
        source.round_encoded = false;
        source.basis_slot = next_basis_slot;
        source.folded = !direct_compact || plan.width == 32;
        plan.current_slot = next_slot;
        plan.rows = next_rows;
        if source.folded {
            plan.width = if direct_compact { D } else { 2 };
            plan.dense = direct_compact;
        } else {
            plan.width *= 2;
        }
        Ok(())
    }
}
