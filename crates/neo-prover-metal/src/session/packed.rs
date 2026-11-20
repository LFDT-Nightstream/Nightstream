use super::*;
use neo_math::{KExtensions, F, K};
use neo_reductions::{
    optimized_engine::{PackedGateNorm, PaperJointOracleInput},
    PiCcsError,
};
use objc2_metal::MTLCommandEncoder;
use p3_field::PrimeCharacteristicRing;
use std::{ffi::c_void, ptr::NonNull};

const LAST_ROUND: usize = 7;
const WIDTH: usize = 128;
const TERNARY: [u16; 256] = {
    let mut table = [0; 256];
    let mut i = 1usize;
    while i < 256 {
        table[i] = table[i & (i - 1)] + 3u16.pow(i.trailing_zeros());
        i += 1;
    }
    table
};

struct Packed {
    device: Device,
    queue: Queue,
    pipelines: [Pipeline; 4],
    checked: Option<Buffer>,
    positive: Buffer,
    negative: Buffer,
    gate: Buffer,
    norm: Option<Buffer>,
    codebook: Vec<K>,
    code_buffer: Option<Buffer>,
    rows: usize,
    width: usize,
    last: Option<(usize, ([K; 8], [K; 4]))>,
}

impl MetalSession {
    pub(crate) fn packed_rounds(
        &self,
        input: &PaperJointOracleInput<'_>,
    ) -> Result<Option<Box<dyn PackedGateNorm>>, MetalError> {
        if !input.packed_gate_norm_shape() {
            return Ok(None);
        }
        let rows = input.rows.shape().rows;
        let width = input.dims.assignment_width;
        let gate = rows.div_ceil(2) * 4 * 16;
        let masks = width.div_ceil(64) * 16;
        let tables = (rows * 32).max(width.div_ceil(16) * 16);
        let reductions = (rows.div_ceil(256) * 8 + width.div_ceil(256) * 4) * 32;
        let factor_bits = input.dims.variables - 1;
        let factors = ((1usize << (factor_bits / 2)) + (1usize << factor_bits.div_ceil(2))) * 16;
        let handoff = (4 * rows.div_ceil(1 << LAST_ROUND) + width.div_ceil(1 << LAST_ROUND)) * 16;
        let peak = gate + masks + tables + reductions + 2 * factors + 2 * 6561 * 16 + handoff;
        if peak > input.workspace_bytes {
            return Ok(None);
        }
        let Some((positive, negative)) = input.fresh_witnesses[0].Z.packed_signed_unit_column_masks() else {
            return Ok(None);
        };
        if positive.len() != width.div_ceil(64) || negative.len() != positive.len() {
            return Ok(None);
        }
        let Some(checked) = input.rows.take_checked_fresh_rows() else {
            return Ok(None);
        };
        if checked.len() != rows {
            return Ok(None);
        }
        Ok(Some(Box::new(Packed::new(self, checked, positive, negative, width)?)))
    }
}
impl Packed {
    fn new(
        session: &MetalSession,
        checked: Vec<[F; 4]>,
        positive: &[u64],
        negative: &[u64],
        width: usize,
    ) -> Result<Self, MetalError> {
        let rows = checked.len();
        let base = session.buffer_from_slice(&checked)?;
        drop(checked);
        Ok(Self {
            device: session.device.clone(),
            queue: session.queue.clone(),
            pipelines: session.packed.clone(),
            checked: Some(base),
            positive: session.buffer_from_slice(positive)?,
            negative: session.buffer_from_slice(negative)?,
            gate: session.buffer(rows.div_ceil(2) * 4 * 16)?,
            norm: None,
            codebook: vec![-K::ONE, K::ZERO, K::ONE],
            code_buffer: None,
            rows,
            width,
            last: None,
        })
    }
}

fn words_at(buffer: &Buffer, index: usize) -> K {
    let words = host_view::<u64>(buffer, buffer.length() / 8);
    K::from_coeffs([F::from_u64(words[2 * index]), F::from_u64(words[2 * index + 1])])
}

impl Packed {
    fn buffer(&self, bytes: usize) -> Result<Buffer, MetalError> {
        let _guard = reserve_on(&self.device, bytes)?;
        self.device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::Buffer { bytes })
    }
    fn upload(&self, data: &[K]) -> Result<Buffer, MetalError> {
        let bytes = size_of_val(data);
        let _guard = reserve_on(&self.device, bytes)?;
        // SAFETY: the immutable slice covers bytes, and Metal copies it synchronously.
        unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr().cast_mut().cast::<c_void>()).unwrap(),
                bytes,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::Buffer { bytes })
    }
    fn evaluate_round(&mut self, point: &[K], weights: (&[K], &[K])) -> Result<([K; 8], [K; 4]), MetalError> {
        let r = point.len();
        let challenge = point.last().copied().unwrap_or(K::ZERO);
        if r <= 3 {
            if r > 0 {
                let k = self.codebook.len();
                self.codebook = (0..k * k)
                    .map(|i| {
                        let a = self.codebook[i % k];
                        a + (self.codebook[i / k] - a) * challenge
                    })
                    .collect();
            }
            self.code_buffer = Some(self.upload(&self.codebook)?);
        }
        if r == 2 {
            self.checked = None;
        }
        if r == 4 {
            self.norm = Some(self.buffer(self.width.div_ceil(16) * 16)?);
        }
        let low = self.upload(weights.0)?;
        let high = self.upload(weights.1)?;
        let command = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandBuffer)?;
        command.setLabel(Some(&NSString::from_str("nightstream.packed_piccs")));
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        let mut reductions = Vec::with_capacity(2);
        for (kernel, (length, terms)) in [(self.rows, 8usize), (self.width, 4usize)]
            .into_iter()
            .enumerate()
        {
            let count = length.div_ceil(1 << (r + 1));
            let groups = count.div_ceil(WIDTH);
            let partial = self.buffer(groups * terms * 16)?;
            let last_groups = groups.div_ceil(WIDTH);
            let sums = self.buffer(last_groups * terms * 16)?;
            let (x, y) = challenge.to_limbs_u64();
            let shape = [
                (count as u64) | ((self.width as u64) << 32),
                (self.rows as u64) | ((r as u64) << 32),
                weights.0.len() as u64,
                0,
                x,
                y,
            ];
            encoder.setComputePipelineState(&self.pipelines[kernel]);
            let buffers = if kernel == 0 {
                [
                    self.checked.as_ref().unwrap_or(&self.gate),
                    &self.gate,
                    &low,
                    &high,
                    &partial,
                ]
            } else {
                [
                    &self.positive,
                    self.norm.as_ref().unwrap_or(&self.positive),
                    &low,
                    &high,
                    &partial,
                ]
            };
            // SAFETY: The accepted dimensions bound every access in packed.metal; each scalar argument is copied.
            unsafe {
                for (slot, b) in buffers.iter().enumerate() {
                    encoder.setBuffer_offset_atIndex(Some(b), 0, slot);
                }
                bytes(&encoder, 5, &shape);
                if kernel == 1 {
                    encoder.setBuffer_offset_atIndex(self.code_buffer.as_deref(), 0, 6);
                    bytes(&encoder, 7, &TERNARY);
                    encoder.setBuffer_offset_atIndex(Some(&self.negative), 0, 8);
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(line(groups), line(WIDTH));
            }
            reductions.push((partial, sums, groups, last_groups, terms));
        }
        encoder.endEncoding();
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        for (i, (partial, sums, groups, last_groups, _)) in reductions.iter().enumerate() {
            encoder.setComputePipelineState(&self.pipelines[2 + i]);
            // SAFETY: each block reduces at most WIDTH partials into exactly one allocated output.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(partial), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(sums), 0, 1);
                bytes(&encoder, 2, &(*groups as u32));
                encoder.dispatchThreadgroups_threadsPerThreadgroup(line(*last_groups), line(WIDTH));
            }
        }
        encoder.endEncoding();
        finish(&command)?;
        let sum = |i: usize, j: usize| {
            let (_, b, _, groups, terms) = &reductions[i];
            (0..*groups).map(|g| words_at(b, g * terms + j)).sum()
        };
        Ok((std::array::from_fn(|j| sum(0, j)), std::array::from_fn(|j| sum(1, j))))
    }
}
fn line(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}
unsafe fn bytes<T>(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, slot: usize, value: &T) {
    // SAFETY: callers pass initialized scalar arrays and the Metal encoder copies the bytes.
    unsafe {
        encoder.setBytes_length_atIndex(NonNull::from(value).cast(), size_of::<T>(), slot);
    }
}
impl PackedGateNorm for Packed {
    fn evaluate(&mut self, point: &[K], weights: (&[K], &[K])) -> Result<([K; 8], [K; 4]), PiCcsError> {
        let round = point.len();
        if let Some((r, terms)) = self.last {
            if r == round {
                return Ok(terms);
            }
        }
        let terms = objc2::rc::autoreleasepool(|_| self.evaluate_round(point, weights)).map_err(crate::oracle_error)?;
        self.last = Some((round, terms));
        Ok(terms)
    }
    fn take_tables(&mut self, round: usize) -> Option<(Vec<K>, Vec<K>)> {
        if round != LAST_ROUND {
            return None;
        }
        let count = self.rows.div_ceil(1 << round);
        let gates = (0..4 * count)
            .map(|i| words_at(&self.gate, 4 * ((i % count) << (round - 1)) + i / count))
            .collect();
        let norm = self.norm.as_ref()?;
        let values = (0..self.width.div_ceil(1 << round))
            .map(|i| words_at(norm, i << (round - 4)))
            .collect();
        Some((gates, values))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/packed_rounds.rs"]
mod tests;
