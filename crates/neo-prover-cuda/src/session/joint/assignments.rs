//! Device witness prefixes. Early folds store indices into shared field values;
//! later folds own only the current dense prefix. Missing coordinates are zero.

use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

use super::{Buffer, CudaError, CudaSession, CudaWitnessMasks, Grid};

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

#[repr(u64)]
#[derive(Clone, Copy)]
pub(super) enum AssignmentEncoding {
    Masks = 0,
    U8Codes = 1,
    U16Codes = 2,
    Dense = 16,
}

impl AssignmentEncoding {
    fn entry_bytes(self) -> usize {
        match self {
            Self::Masks => 0,
            Self::U8Codes => size_of::<u8>(),
            Self::U16Codes => size_of::<u16>(),
            Self::Dense => size_of::<K>(),
        }
    }

    fn for_codes(alphabet: Option<usize>) -> Self {
        match alphabet {
            Some(size) if size <= usize::from(u8::MAX) + 1 => Self::U8Codes,
            Some(_) => Self::U16Codes,
            None => Self::Dense,
        }
    }
}

pub(super) struct AssignmentTables {
    prefix: Prefix,
    pub(super) sources: Buffer,
    len: usize,
    count: usize,
    magnitudes: usize,
}

impl AssignmentTables {
    pub(super) fn new(session: &CudaSession, masks: &CudaWitnessMasks, width: usize) -> Result<Self, CudaError> {
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
            prefix: Prefix::Masks(session.buffer_from_slice(&super::support::k_words(&values))?),
            sources: session.buffer_from_slice(if sources.is_empty() { &[0] } else { &sources })?,
            len: width,
            count: sources.len(),
            magnitudes,
        })
    }

    pub(super) fn encoding(&self) -> AssignmentEncoding {
        match &self.prefix {
            Prefix::Masks(_) => AssignmentEncoding::Masks,
            Prefix::Codes { alphabet, .. } => AssignmentEncoding::for_codes(Some(*alphabet)),
            Prefix::Dense(_) => AssignmentEncoding::Dense,
        }
    }

    pub(super) fn data<'a>(&'a self, masks: &'a CudaWitnessMasks) -> &'a Buffer {
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

    pub(super) fn fold_allocation_bytes(&self) -> Result<usize, CudaError> {
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
        let entry_bytes = AssignmentEncoding::for_codes(next).entry_bytes();
        self.count
            .checked_mul(self.len.div_ceil(2))
            .and_then(|n| n.checked_mul(entry_bytes))
            .and_then(|n| n.checked_add(next.unwrap_or(0) * size_of::<K>() + 10 * size_of::<u64>()))
            .ok_or(CudaError::Shape("assignment fold allocation size overflow"))
    }

    pub(super) fn fold(
        &mut self,
        session: &CudaSession,
        masks: &CudaWitnessMasks,
        challenge: K,
    ) -> Result<(), CudaError> {
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
        let next_encoding = AssignmentEncoding::for_codes(next_alphabet);
        let bytes = next_encoding.entry_bytes();
        let entries = self
            .count
            .checked_mul(next_len)
            .ok_or(CudaError::Shape("assignment prefix length overflow"))?;
        let output_bytes = entries
            .checked_mul(bytes)
            .ok_or(CudaError::Shape("assignment prefix size overflow"))?;
        let output = session.buffer(output_bytes)?;
        let (real, imaginary) = neo_math::KExtensions::to_limbs_u64(&challenge);
        let challenge = [real, imaginary];
        let shape = [
            self.len as u64,
            self.count as u64,
            masks.blocks() as u64,
            self.magnitudes as u64,
            self.encoding() as u64,
            alphabet as u64,
            zero as u64,
            next_encoding as u64,
        ];
        let next_values = next_alphabet
            .map(|size| session.buffer(size * size_of::<K>()))
            .transpose()?;
        let kernels = &session.kernels.assignments;
        if let Some(values) = &next_values {
            session.launch(Grid::Threads(next_alphabet.unwrap()), |stream, config| {
                kernels.joint_fold_assignment_values(
                    stream,
                    config,
                    self.values().ptr(),
                    challenge,
                    shape,
                    values.ptr(),
                )
            })?;
        }
        session.launch(Grid::Threads(entries), |stream, config| {
            kernels.joint_fold_assignments(
                stream,
                config,
                self.data(masks).ptr(),
                challenge,
                shape,
                output.ptr(),
                self.sources.ptr(),
                self.values().ptr(),
            )
        })?;
        session.finish()?;
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
