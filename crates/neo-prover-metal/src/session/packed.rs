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

type GateNormCoefficients = ([K; 8], [K; 4]);

enum RoundState {
    Ready,
    Submitted(usize, RoundReadback),
    Complete(usize, GateNormCoefficients),
}

struct RoundReadback {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    gate: (Buffer, usize),
    norm: (Buffer, usize),
}

struct Packed {
    round: RoundState,
    device: Device,
    queue: Queue,
    activity: Arc<ActivityCounters>,
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
        optional_packed(Packed::new(self, checked, positive, negative, width))
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
            activity: Arc::clone(&session.activity),
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
            round: RoundState::Ready,
        })
    }
}

fn optional_packed(result: Result<Packed, MetalError>) -> Result<Option<Box<dyn PackedGateNorm>>, MetalError> {
    match result {
        Ok(packed) => Ok(Some(Box::new(packed))),
        Err(MetalError::Buffer { .. } | MetalError::MemoryLimit { .. }) => Ok(None),
        Err(error) => Err(error),
    }
}

fn extension(words: [u64; 2]) -> K {
    K::from_coeffs(words.map(F::from_u64))
}

fn sum_groups<const N: usize>(buffer: &Buffer, groups: usize) -> [K; N] {
    let partials = host_view::<[[u64; 2]; N]>(buffer, groups);
    std::array::from_fn(|i| partials.iter().map(|terms| extension(terms[i])).sum())
}

impl Packed {
    fn buffer(&self, bytes: usize) -> Result<Buffer, MetalError> {
        let _guard = reserve_on(&self.device, bytes)?;
        let buffer = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::Buffer { bytes })?;
        self.activity
            .allocated_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(buffer)
    }
    fn upload(&self, data: &[K]) -> Result<Buffer, MetalError> {
        let bytes = size_of_val(data);
        let _guard = reserve_on(&self.device, bytes)?;
        // SAFETY: the immutable slice covers bytes, and Metal copies it synchronously.
        let buffer = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr().cast_mut().cast::<c_void>()).unwrap(),
                bytes,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::Buffer { bytes })?;
        self.activity
            .allocated_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        self.activity
            .uploaded_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(buffer)
    }
    fn encode(&mut self, r: usize, challenge: K, weights: (&[K], &[K])) -> Result<RoundReadback, MetalError> {
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
        self.activity
            .command_buffers
            .fetch_add(1, Ordering::Relaxed);
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
                self.bytes(&encoder, 5, &shape);
                if kernel == 1 {
                    encoder.setBuffer_offset_atIndex(self.code_buffer.as_deref(), 0, 6);
                    self.bytes(&encoder, 7, &TERNARY);
                    encoder.setBuffer_offset_atIndex(Some(&self.negative), 0, 8);
                }
                self.activity.dispatches.fetch_add(1, Ordering::Relaxed);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(line(groups), line(WIDTH));
            }
            reductions.push((partial, sums, groups, last_groups));
        }
        encoder.endEncoding();
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        for (i, (partial, sums, groups, last_groups)) in reductions.iter().enumerate() {
            encoder.setComputePipelineState(&self.pipelines[2 + i]);
            // SAFETY: each block reduces at most WIDTH partials into exactly one allocated output.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(partial), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(sums), 0, 1);
                self.bytes(&encoder, 2, &(*groups as u32));
                self.activity.dispatches.fetch_add(1, Ordering::Relaxed);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(line(*last_groups), line(WIDTH));
            }
        }
        encoder.endEncoding();
        command.commit();
        let read_bytes: usize = reductions.iter().map(|(_, sums, ..)| sums.length()).sum();
        self.activity
            .downloaded_bytes
            .fetch_add(read_bytes as u64, Ordering::Relaxed);
        let Ok([(_, gate, _, gate_groups), (_, norm, _, norm_groups)]) = <[_; 2]>::try_from(reductions) else {
            unreachable!("one gate and one norm reduction")
        };
        Ok(RoundReadback {
            command,
            gate: (gate, gate_groups),
            norm: (norm, norm_groups),
        })
    }
}
fn line(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}
impl Packed {
    unsafe fn bytes<T>(&self, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, slot: usize, value: &T) {
        // SAFETY: callers pass initialized scalar arrays and the Metal encoder copies the bytes.
        unsafe {
            encoder.setBytes_length_atIndex(NonNull::from(value).cast(), size_of::<T>(), slot);
        }
        self.activity
            .uploaded_bytes
            .fetch_add(size_of::<T>() as u64, Ordering::Relaxed);
    }
}
impl PackedGateNorm for Packed {
    fn submit(&mut self, round: usize, challenge: K, weights: (&[K], &[K])) -> Result<(), PiCcsError> {
        let readback =
            objc2::rc::autoreleasepool(|_| self.encode(round, challenge, weights)).map_err(crate::oracle_error)?;
        self.round = RoundState::Submitted(round, readback);
        Ok(())
    }

    fn coefficients(&mut self, round: usize) -> Result<GateNormCoefficients, PiCcsError> {
        let readback = match &self.round {
            RoundState::Complete(index, terms) if *index == round => return Ok(*terms),
            RoundState::Submitted(index, readback) if *index == round => readback,
            _ => {
                return Err(PiCcsError::InvalidInput(
                    "packed round has no matching submission".into(),
                ))
            }
        };
        self.activity.host_waits.fetch_add(1, Ordering::Relaxed);
        wait(&readback.command).map_err(crate::oracle_error)?;
        let terms = (
            sum_groups(&readback.gate.0, readback.gate.1),
            sum_groups(&readback.norm.0, readback.norm.1),
        );
        self.round = RoundState::Complete(round, terms);
        Ok(terms)
    }
    fn take_tables(&mut self, round: usize) -> Option<(Vec<K>, Vec<K>)> {
        if round != LAST_ROUND {
            return None;
        }
        let norm = host_view::<[u64; 2]>(self.norm.as_ref()?, self.width.div_ceil(16));
        let gate = host_view::<[[u64; 2]; 4]>(&self.gate, self.rows.div_ceil(2));
        let mut gates = Vec::with_capacity(4 * self.rows.div_ceil(1 << round));
        for column in 0..4 {
            gates.extend(
                gate.iter()
                    .step_by(1 << (round - 1))
                    .map(|row| extension(row[column])),
            );
        }
        let values: Vec<_> = norm
            .iter()
            .step_by(1 << (round - 4))
            .copied()
            .map(extension)
            .collect();
        let bytes = (gates.len() + values.len()) * size_of::<K>();
        self.activity
            .downloaded_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Some((gates, values))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/packed_rounds.rs"]
mod tests;
