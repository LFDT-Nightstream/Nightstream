//! Device-resident Poseidon2 transcript mirror.
//!
//! Owns: the device sponge state buffer, the round-constant upload, and the
//! op-stream launcher mirroring `neo_transcript::Poseidon2Transcript` raw
//! absorb/challenge semantics bit-exactly (`parity transcript` gate).
//! Does not own: transcript authority — the host paper-layer transcript
//! stays canonical until fold challenges are sourced from this mirror.

use cuda_core::{DeviceBuffer, DriverError};
use neo_math::F;
use p3_field::PrimeField64;

use crate::device::{copy_host_to_device, uninit_u64_device_buffer, upload_u64_device_buffer, Device};
use crate::field::f_from_device_word;
use crate::graph::GraphAllocations;
use crate::kernels::poseidon2::{
    launch_transcript_absorb_device_challenge, launch_transcript_bind_device_fields_sample_rlc_rhos,
    launch_transcript_io_ops, launch_transcript_ops, launch_transcript_sample_rlc_rhos, DeviceFieldBindPrefix,
    Poseidon2KernelModule, EXTERNAL_HALF_ROUNDS, INTERNAL_ROUNDS, MAX_DEVICE_FIELD_BIND_PREFIX_WORDS, OP_ABSORB,
    OP_ABSORB_DEVICE, OP_CHALLENGE, OP_CHALLENGE_DEVICE, RC_DIAG, RC_INITIAL, RC_INTERNAL, RC_TERMINAL, RC_WORDS,
    ST_WORDS, WIDTH,
};

/// One logical transcript step against the device sponge.
pub enum TranscriptOp {
    /// `absorb_elem` for each field in order. Callers include any length
    /// prefix the mirrored host op absorbs.
    AbsorbFields(Vec<F>),
    /// `challenge_fields_raw(n)`: absorb ONE, permute, squeeze up to
    /// DIGEST_LEN lanes per iteration until `n` values are produced.
    Challenge(usize),
}

/// Transcript op stream with explicit host/device payload ownership.
pub enum TranscriptIoOp {
    /// Absorb raw fields uploaded by the host wrapper.
    AbsorbHost(Vec<F>),
    /// Absorb `len` raw field words starting at `offset` in a device buffer.
    AbsorbDevice { offset: usize, len: usize },
    /// Write `challenge_fields_raw(len)` into the returned host vector.
    ChallengeHost(usize),
    /// Write `challenge_fields_raw(len)` into a device buffer at `offset`.
    ChallengeDevice { offset: usize, len: usize },
}

pub(crate) struct EncodedTranscriptIo {
    pub(crate) op_words: Vec<u64>,
    pub(crate) host_payload: Vec<u64>,
    pub(crate) host_out_len: usize,
}

pub(crate) fn encode_transcript_io_ops(ops: &[TranscriptIoOp]) -> EncodedTranscriptIo {
    let mut op_words = Vec::with_capacity(3 * ops.len());
    let mut host_payload = Vec::new();
    let mut host_out_len = 0usize;
    for op in ops {
        match op {
            TranscriptIoOp::AbsorbHost(fs) => {
                let offset = host_payload.len();
                op_words.extend([OP_ABSORB, fs.len() as u64, offset as u64]);
                host_payload.extend(fs.iter().map(|f| f.as_canonical_u64()));
            }
            TranscriptIoOp::AbsorbDevice { offset, len } => {
                op_words.extend([OP_ABSORB_DEVICE, *len as u64, *offset as u64]);
            }
            TranscriptIoOp::ChallengeHost(len) => {
                op_words.extend([OP_CHALLENGE, *len as u64, host_out_len as u64]);
                host_out_len += len;
            }
            TranscriptIoOp::ChallengeDevice { offset, len } => {
                op_words.extend([OP_CHALLENGE_DEVICE, *len as u64, *offset as u64]);
            }
        }
    }
    if host_payload.is_empty() {
        host_payload.push(0);
    }
    EncodedTranscriptIo {
        op_words,
        host_payload,
        host_out_len,
    }
}

/// Device buffers and offsets for one absorb-from-device / challenge-to-device
/// transcript step. Naming these slots keeps protocol call sites readable:
/// coeffs are the payload, challenges are the output.
pub struct DeviceIoSlots<'a> {
    pub payload: &'a DeviceBuffer<u64>,
    pub payload_offset: usize,
    pub payload_len: usize,
    pub out: &'a mut DeviceBuffer<u64>,
    pub out_offset: usize,
}

/// Upload the canonical round constants in the kernel `RC_*` layout.
pub fn upload_round_constants(device: &Device) -> Result<DeviceBuffer<u64>, DriverError> {
    let rc = neo_ccs::crypto::poseidon2_goldilocks::round_constants();
    assert_eq!(rc.initial.len(), EXTERNAL_HALF_ROUNDS, "initial external rounds");
    assert_eq!(rc.terminal.len(), EXTERNAL_HALF_ROUNDS, "terminal external rounds");
    assert_eq!(rc.internal.len(), INTERNAL_ROUNDS, "internal rounds");
    let mut words = vec![0u64; RC_WORDS];
    for (r, row) in rc.initial.iter().enumerate() {
        words[RC_INITIAL + WIDTH * r..][..WIDTH].copy_from_slice(row);
    }
    words[RC_INTERNAL..RC_INTERNAL + INTERNAL_ROUNDS].copy_from_slice(&rc.internal);
    for (r, row) in rc.terminal.iter().enumerate() {
        words[RC_TERMINAL + WIDTH * r..][..WIDTH].copy_from_slice(row);
    }
    words[RC_DIAG..RC_DIAG + WIDTH].copy_from_slice(&rc.diag);
    upload_u64_device_buffer(device.stream(), &words)
}

/// The device sponge: WIDTH state lanes plus the absorb cursor, advanced by
/// op streams without any host readback in between.
pub struct DeviceTranscript {
    st: DeviceBuffer<u64>,
}

impl DeviceTranscript {
    /// Seed from a host sponge position (`Poseidon2Transcript::state()` /
    /// `absorbed()` — the backend-handoff snapshot shape).
    pub fn from_state_and_absorbed(device: &Device, state: [F; WIDTH], absorbed: usize) -> Result<Self, DriverError> {
        Ok(Self {
            st: upload_u64_device_buffer(device.stream(), &state_words(state, absorbed))?,
        })
    }

    /// Reset an existing device sponge without changing its allocation. Whole
    /// phase graph capture depends on this buffer address staying stable.
    pub(crate) fn reset_state_and_absorbed(
        &mut self,
        device: &Device,
        state: [F; WIDTH],
        absorbed: usize,
    ) -> Result<(), DriverError> {
        copy_host_to_device(device.stream(), &self.st, &state_words(state, absorbed))
    }

    /// Execute `ops` in order on device; returns every challenge output,
    /// concatenated.
    pub fn run(
        &mut self,
        device: &Device,
        module: &Poseidon2KernelModule,
        rc: &DeviceBuffer<u64>,
        ops: &[TranscriptOp],
    ) -> Result<Vec<F>, DriverError> {
        if ops.is_empty() {
            return Ok(Vec::new());
        }
        let mut op_words = Vec::with_capacity(2 * ops.len());
        let mut payload = Vec::new();
        let mut out_len = 0usize;
        for op in ops {
            match op {
                TranscriptOp::AbsorbFields(fs) => {
                    op_words.extend([OP_ABSORB, fs.len() as u64]);
                    payload.extend(fs.iter().map(|f| f.as_canonical_u64()));
                }
                TranscriptOp::Challenge(n) => {
                    op_words.extend([OP_CHALLENGE, *n as u64]);
                    out_len += n;
                }
            }
        }
        if payload.is_empty() {
            payload.push(0);
        }
        let stream = device.stream();
        let ops_dev = upload_u64_device_buffer(stream, &op_words)?;
        let pay_dev = upload_u64_device_buffer(stream, &payload)?;
        let mut out_dev = uninit_u64_device_buffer(stream, out_len.max(1))?;
        launch_transcript_ops(module, stream, &mut self.st, &ops_dev, &pay_dev, &mut out_dev, rc)?;
        let out = out_dev.to_host_vec(stream)?;
        device.sync()?;
        Ok(out[..out_len]
            .iter()
            .map(|&w| f_from_device_word(w))
            .collect())
    }

    /// Execute an IO op stream. Device absorbs read from `device_payload`;
    /// device challenges are written to `device_out` without a host readback.
    /// Returned values are only those requested by `ChallengeHost` ops.
    pub fn run_io(
        &mut self,
        device: &Device,
        module: &Poseidon2KernelModule,
        rc: &DeviceBuffer<u64>,
        ops: &[TranscriptIoOp],
        device_payload: &DeviceBuffer<u64>,
        device_out: &mut DeviceBuffer<u64>,
    ) -> Result<Vec<F>, DriverError> {
        if ops.is_empty() {
            return Ok(Vec::new());
        }
        let encoded = encode_transcript_io_ops(ops);
        let stream = device.stream();
        let ops_dev = upload_u64_device_buffer(stream, &encoded.op_words)?;
        let host_payload_dev = upload_u64_device_buffer(stream, &encoded.host_payload)?;
        let mut host_out_dev = uninit_u64_device_buffer(stream, encoded.host_out_len.max(1))?;
        self.enqueue_io(
            device,
            module,
            rc,
            &ops_dev,
            &host_payload_dev,
            device_payload,
            &mut host_out_dev,
            device_out,
        )?;
        let out = host_out_dev.to_host_vec(stream)?;
        device.sync()?;
        Ok(out[..encoded.host_out_len]
            .iter()
            .map(|&w| f_from_device_word(w))
            .collect())
    }

    /// Enqueue a prebuilt IO op stream without downloading or synchronizing.
    /// Callers own the op/payload/output buffers and must keep them live
    /// until later stream synchronization.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_io(
        &mut self,
        device: &Device,
        module: &Poseidon2KernelModule,
        rc: &DeviceBuffer<u64>,
        ops: &DeviceBuffer<u64>,
        host_payload: &DeviceBuffer<u64>,
        device_payload: &DeviceBuffer<u64>,
        host_out: &mut DeviceBuffer<u64>,
        device_out: &mut DeviceBuffer<u64>,
    ) -> Result<(), DriverError> {
        launch_transcript_io_ops(
            module,
            device.stream(),
            &mut self.st,
            ops,
            host_payload,
            device_payload,
            host_out,
            device_out,
            rc,
        )
    }

    pub fn enqueue_absorb_device_challenge(
        &mut self,
        device: &Device,
        module: &Poseidon2KernelModule,
        rc: &DeviceBuffer<u64>,
        len_prefix: u64,
        slots: DeviceIoSlots<'_>,
    ) -> Result<(), DriverError> {
        launch_transcript_absorb_device_challenge(
            module,
            device.stream(),
            &mut self.st,
            len_prefix,
            slots.payload,
            slots.payload_offset,
            slots.payload_len,
            slots.out,
            slots.out_offset,
            rc,
        )
    }

    pub fn enqueue_sample_rlc_rhos(
        &mut self,
        device: &Device,
        module: &Poseidon2KernelModule,
        rc: &DeviceBuffer<u64>,
        count: usize,
        coeffs_out: &mut DeviceBuffer<u64>,
    ) -> Result<(), DriverError> {
        launch_transcript_sample_rlc_rhos(module, device.stream(), &mut self.st, count, coeffs_out, rc)
    }

    pub fn enqueue_bind_device_fields_sample_rlc_rhos(
        &mut self,
        device: &Device,
        module: &Poseidon2KernelModule,
        rc: &DeviceBuffer<u64>,
        label: &'static [u8],
        device_fields: &DeviceBuffer<u64>,
        count: usize,
        coeffs_out: &mut DeviceBuffer<u64>,
    ) -> Result<(), DriverError> {
        launch_transcript_bind_device_fields_sample_rlc_rhos(
            module,
            device.stream(),
            &mut self.st,
            device_field_bind_prefix(label, device_fields.len()),
            device_fields,
            count,
            coeffs_out,
            rc,
        )
    }

    /// Download `(state, absorbed)` — the host snapshot shape.
    pub fn state_and_absorbed(&self, device: &Device) -> Result<([F; WIDTH], usize), DriverError> {
        let words = self.st.to_host_vec(device.stream())?;
        device.sync()?;
        Ok(Self::decode_state_words(&words))
    }

    /// Queue a download of the raw transcript state words. The caller may
    /// batch this with other D2H copies and synchronize once.
    pub fn state_words_to_host(&self, device: &Device) -> Result<Vec<u64>, DriverError> {
        self.st.to_host_vec(device.stream())
    }

    pub(crate) fn state_words_mut(&mut self) -> &mut DeviceBuffer<u64> {
        &mut self.st
    }

    pub fn decode_state_words(words: &[u64]) -> ([F; WIDTH], usize) {
        assert!(
            words.len() >= ST_WORDS,
            "device transcript state download returned too few words"
        );
        let state = core::array::from_fn(|i| f_from_device_word(words[i]));
        (state, words[WIDTH] as usize)
    }

    pub(crate) fn record_graph_allocations(&self, allocations: &mut GraphAllocations) {
        allocations.push(&self.st);
    }
}

fn state_words(state: [F; WIDTH], absorbed: usize) -> [u64; ST_WORDS] {
    let mut words = [0u64; ST_WORDS];
    for (w, f) in words.iter_mut().zip(state.iter()) {
        *w = f.as_canonical_u64();
    }
    words[WIDTH] = absorbed as u64;
    words
}

fn device_field_bind_prefix(label: &[u8], field_count: usize) -> DeviceFieldBindPrefix {
    const BYTES_PER_LIMB: usize = 7;
    let mut words = [0u64; MAX_DEVICE_FIELD_BIND_PREFIX_WORDS];
    words[0] = label.len() as u64;
    let mut word_count = 1usize;
    for chunk in label.chunks(BYTES_PER_LIMB) {
        assert!(
            word_count < MAX_DEVICE_FIELD_BIND_PREFIX_WORDS,
            "device transcript label exceeds packed prefix capacity"
        );
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        words[word_count] = u64::from_le_bytes(limb);
        word_count += 1;
    }
    DeviceFieldBindPrefix {
        words,
        word_count: word_count as u64,
        field_count: field_count as u64,
    }
}
