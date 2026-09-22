//! Device witness prefixes. Early folds store indices into shared field values;
//! later folds own only the current dense prefix. Missing coordinates are zero.

use neo_math::{F, K};
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};
use p3_field::PrimeCharacteristicRing;

use super::{support::k_words, Buffer, MetalError, MetalSession, MetalWitnessMasks};

enum Prefix {
    Masks(Buffer),
    Codes {
        data: Buffer,
        values: Buffer,
        alphabet: usize,
        zero: usize,
    },
    Dense(Buffer),
}

pub(super) struct AssignmentTables {
    prefix: Prefix,
    pub(super) sources: Buffer,
    len: usize,
    count: usize,
    magnitudes: usize,
}

impl AssignmentTables {
    pub(super) fn new(session: &MetalSession, masks: &MetalWitnessMasks, width: usize) -> Result<Self, MetalError> {
        let magnitudes = masks.magnitudes();
        let values = (0..2 * magnitudes + 1)
            .map(|value| K::from(F::from_usize(value) - F::from_usize(magnitudes)))
            .collect::<Vec<_>>();
        let sources = masks
            .active_witnesses()
            .iter()
            .map(|&source| u64::from(source))
            .collect::<Vec<_>>();
        Ok(Self {
            prefix: Prefix::Masks(session.buffer_from_slice(&k_words(&values))?),
            sources: session.buffer_from_slice(if sources.is_empty() { &[0] } else { &sources })?,
            len: width,
            count: sources.len(),
            magnitudes,
        })
    }

    // Shader ABI: 0 = original masks, 1/2 = index bytes, 16 = dense K.
    pub(super) fn encoding(&self) -> usize {
        match &self.prefix {
            Prefix::Masks(_) => 0,
            Prefix::Codes { alphabet, .. } if *alphabet <= usize::from(u8::MAX) + 1 => 1,
            Prefix::Codes { .. } => 2,
            Prefix::Dense(_) => size_of::<K>(),
        }
    }

    pub(super) fn data<'a>(&'a self, masks: &'a MetalWitnessMasks) -> &'a Buffer {
        match &self.prefix {
            Prefix::Masks(_) => masks.words(),
            Prefix::Codes { data, .. } | Prefix::Dense(data) => data,
        }
    }

    pub(super) fn values(&self) -> &Buffer {
        match &self.prefix {
            Prefix::Masks(values) | Prefix::Codes { values, .. } | Prefix::Dense(values) => values,
        }
    }

    pub(super) fn fold_allocation_bytes(&self) -> Result<usize, MetalError> {
        if self.count == 0 {
            return Ok(0);
        }
        let alphabet = match self.prefix {
            Prefix::Masks(_) => 2 * self.magnitudes + 1,
            Prefix::Codes { alphabet, .. } => alphabet,
            Prefix::Dense(_) => 0,
        };
        let next = alphabet
            .checked_mul(alphabet)
            .filter(|&n| n > 0 && n <= usize::from(u16::MAX) + 1);
        let entry_bytes = match next {
            Some(n) if n <= usize::from(u8::MAX) + 1 => 1,
            Some(_) => 2,
            None => size_of::<K>(),
        };
        self.count
            .checked_mul(self.len.div_ceil(2))
            .and_then(|n| n.checked_mul(entry_bytes))
            .and_then(|n| n.checked_add(next.unwrap_or(0) * size_of::<K>() + 10 * size_of::<u64>()))
            .ok_or(MetalError::Shape("assignment fold allocation size overflow"))
    }

    pub(super) fn fold(
        &mut self,
        session: &MetalSession,
        masks: &MetalWitnessMasks,
        challenge: K,
    ) -> Result<(), MetalError> {
        let next_len = self.len.div_ceil(2);
        if self.count == 0 {
            self.len = next_len;
            return Ok(());
        }
        let (alphabet, zero) = match self.prefix {
            Prefix::Masks(_) => (2 * self.magnitudes + 1, self.magnitudes),
            Prefix::Codes { alphabet, zero, .. } => (alphabet, zero),
            Prefix::Dense(_) => (0, 0),
        };
        let next_alphabet = alphabet
            .checked_mul(alphabet)
            .filter(|&size| size > 0 && size <= usize::from(u16::MAX) + 1);
        let bytes = match next_alphabet {
            Some(size) if size <= usize::from(u8::MAX) + 1 => 1,
            Some(_) => 2,
            None => size_of::<K>(),
        };
        let entries = self
            .count
            .checked_mul(next_len)
            .ok_or(MetalError::Shape("assignment prefix length overflow"))?;
        let output_bytes = entries
            .checked_mul(bytes)
            .ok_or(MetalError::Shape("assignment prefix size overflow"))?;
        let output = session.buffer(output_bytes)?;
        let challenge = session.buffer_from_slice(&k_words(&[challenge]))?;
        let shape = session.buffer_from_slice(&[
            self.len as u64,
            self.count as u64,
            masks.blocks() as u64,
            self.magnitudes as u64,
            self.encoding() as u64,
            alphabet as u64,
            zero as u64,
            bytes as u64,
        ])?;
        let next_values = next_alphabet
            .map(|size| session.buffer(size * size_of::<K>()))
            .transpose()?;
        let command = session.command_buffer("nightstream.pi_ccs.joint.assignment.fold")?;
        if let Some(values) = &next_values {
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&session.joint_fold_assignment_values);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(self.values()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&challenge), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&shape), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(values), 0, 3);
            }
            session.dispatch(&encoder, &session.joint_fold_assignment_values, next_alphabet.unwrap());
            encoder.endEncoding();
        }
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&session.joint_fold_assignments);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(self.data(masks)), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&challenge), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&output), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&self.sources), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(self.values()), 0, 5);
        }
        session.dispatch(&encoder, &session.joint_fold_assignments, entries);
        encoder.endEncoding();
        session.finish(&command)?;
        self.prefix = match next_values {
            Some(values) => Prefix::Codes {
                data: output,
                values,
                alphabet: next_alphabet.unwrap(),
                zero: zero + alphabet * zero,
            },
            None => Prefix::Dense(output),
        };
        self.len = next_len;
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/joint_assignments.rs"]
mod tests;
