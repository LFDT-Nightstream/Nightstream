//! Cached Metal execution of the protocol's fixed-seed SIS digest maps.

use std::mem::size_of;

use neo_fold_clean::paper::reductions::accumulator_sis_circuit::{
    accelerator_balanced_ternary_message, accumulator_digest_envelope_prefix, SisAccumulatorConfig,
    SIS_DIGEST_COMPRESSION_CONFIG,
};
use neo_math::{D, F};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{Buffer, MetalAjtaiLowNormPlan, MetalSession};
use crate::MetalError;

const BALANCED_TERNARY_DIGITS: usize = 41;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SisMapKey {
    seed: [u8; 32],
    kappa: usize,
    field_count: usize,
}

pub(super) struct MetalSisMap {
    key: SisMapKey,
    plan: MetalAjtaiLowNormPlan,
    message: Buffer,
    masks: Buffer,
    shape: Buffer,
}

impl MetalSession {
    /// Compute the canonical fixed-seed SIS/Poseidon2 accumulator digest.
    ///
    pub fn sis_accumulator_digest(&self, config: SisAccumulatorConfig, fields: &[F]) -> Result<[F; 4], MetalError> {
        if fields.is_empty() || config.kappa == 0 {
            return Err(MetalError::Shape("SIS digest requires fields and nonzero kappa"));
        }
        let binding = self.sis_commit_fields(config, fields)?;
        let compression = self.sis_commit_fields(SIS_DIGEST_COMPRESSION_CONFIG, &binding)?;
        let mut envelope = accumulator_digest_envelope_prefix(config, fields.len());
        envelope.extend_from_slice(&compression);
        Ok(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&envelope))
    }

    /// Large-message SIS digest with every intermediate kept on device.
    pub(crate) fn sis_accumulator_digest_resident(
        &self,
        config: SisAccumulatorConfig,
        fields: &[F],
    ) -> Result<[F; 4], MetalError> {
        if fields.is_empty() || config.kappa == 0 {
            return Err(MetalError::Shape("SIS digest requires fields and nonzero kappa"));
        }
        let binding_key = SisMapKey {
            seed: config.seed,
            kappa: config.kappa,
            field_count: fields.len(),
        };
        let compression_key = SisMapKey {
            seed: SIS_DIGEST_COMPRESSION_CONFIG.seed,
            kappa: SIS_DIGEST_COMPRESSION_CONFIG.kappa,
            field_count: config.kappa * D,
        };
        let binding_index = self.ensure_sis_map(binding_key)?;
        let compression_index = self.ensure_sis_map(compression_key)?;

        let field_words = fields
            .iter()
            .map(PrimeField64::as_canonical_u64)
            .collect::<Vec<_>>();
        let field_words = self.buffer_from_slice(&field_words)?;
        let prefix = accumulator_digest_envelope_prefix(config, fields.len());
        let prefix_words = prefix
            .iter()
            .map(PrimeField64::as_canonical_u64)
            .collect::<Vec<_>>();
        let compression_words = SIS_DIGEST_COMPRESSION_CONFIG.kappa * D;
        if !compression_words.is_multiple_of(2) {
            return Err(MetalError::Shape(
                "SIS compression width must contain whole extension words",
            ));
        }
        let mut envelope_words = prefix_words.clone();
        envelope_words.resize(prefix_words.len() + compression_words, 0);
        let envelope = self.buffer_from_slice(&envelope_words)?;
        let envelope_shape = self.buffer_from_slice(&[envelope_words.len() as u64])?;
        let digest = self.buffer(4 * size_of::<u64>())?;
        let command = self.independent_command_buffer("nightstream.sis.digest")?;

        {
            let maps = self.sis_maps.borrow();
            let binding = &maps[binding_index];
            let compression = &maps[compression_index];
            self.encode_sis_message(&command, binding, &field_words)?;
            let binding_words = self.encode_ajtai_low_norm_masks(&command, &binding.plan, &binding.masks)?;
            self.encode_sis_message(&command, compression, binding_words)?;
            let compression_words_buffer =
                self.encode_ajtai_low_norm_masks(&command, &compression.plan, &compression.masks)?;

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.copy_k_words);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(compression_words_buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&envelope), prefix_words.len() * size_of::<u64>(), 1);
            }
            self.dispatch(&encoder, &self.copy_k_words, compression_words / 2);
            encoder.endEncoding();

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.poseidon2_hash_uniform);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&envelope), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&digest), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&self.poseidon2_constants), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&envelope_shape), 0, 3);
            }
            self.dispatch(&encoder, &self.poseidon2_hash_uniform, 1);
            encoder.endEncoding();
        }

        self.finish(&command)?;
        let words: [u64; 4] = self
            .read_buffer::<u64>(&digest, 4)
            .try_into()
            .expect("Metal SIS digest has fixed width");
        Ok(words.map(F::from_u64))
    }

    fn sis_commit_fields(&self, config: SisAccumulatorConfig, fields: &[F]) -> Result<Vec<F>, MetalError> {
        let message = accelerator_balanced_ternary_message(fields);
        if message.is_empty() || !message.len().is_multiple_of(D) {
            return Err(MetalError::Shape("SIS message has invalid ring-column dimensions"));
        }
        let key = SisMapKey {
            seed: config.seed,
            kappa: config.kappa,
            field_count: fields.len(),
        };
        let map_index = self.ensure_sis_map(key)?;
        let maps = self.sis_maps.borrow();
        let words = self.ajtai_low_norm_with_plan_independent(&maps[map_index].plan, &message)?;
        if words.len() != config.kappa * D {
            return Err(MetalError::Shape("Metal SIS commitment has invalid dimensions"));
        }
        Ok(words.into_iter().map(F::from_u64).collect())
    }

    fn encode_sis_message(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        map: &MetalSisMap,
        fields: &ProtocolObject<dyn MTLBuffer>,
    ) -> Result<(), MetalError> {
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.sis_balanced_ternary_message);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(fields), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&map.message), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&map.shape), 0, 2);
        }
        self.dispatch(&encoder, &self.sis_balanced_ternary_message, map.key.field_count);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.sis_pack_signed_masks);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&map.message), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&map.masks), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&map.shape), 0, 2);
        }
        self.dispatch(&encoder, &self.sis_pack_signed_masks, map.plan.cols);
        encoder.endEncoding();
        Ok(())
    }

    fn ensure_sis_map(&self, key: SisMapKey) -> Result<usize, MetalError> {
        if let Some(index) = self.sis_maps.borrow().iter().position(|map| map.key == key) {
            return Ok(index);
        }
        let message_digits = key
            .field_count
            .checked_mul(BALANCED_TERNARY_DIGITS)
            .ok_or(MetalError::Shape("SIS message dimensions overflow"))?;
        let message_cols = message_digits.div_ceil(D);
        let plan = self.prepare_ajtai_low_norm_seeded(key.seed, key.kappa, message_cols)?;
        let message = self.buffer(message_cols * D * size_of::<i8>())?;
        let masks = self.buffer(2 * message_cols * size_of::<u64>())?;
        let shape = self.buffer_from_slice(&[key.field_count as u64, message_cols as u64])?;
        let mut maps = self.sis_maps.borrow_mut();
        maps.push(MetalSisMap {
            key,
            plan,
            message,
            masks,
            shape,
        });
        Ok(maps.len() - 1)
    }
}
