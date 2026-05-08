//! Owns an in-circuit Poseidon2 transcript matching `neo_transcript::Poseidon2Transcript`.
//!
//! This module only owns transcript state evolution and challenge squeezing.
//! It does not own RV32IM relation semantics.

mod lane;
mod packing;
mod permutation;

use lane::TranscriptLane;
use packing::{pack_bytes, pack_u64s};
use permutation::permute_state;

use crate::spartan_backend::SpartanF;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError, Variable};
use ff::Field;

const APP_DOMAIN: &[u8] = b"neo/transcript/v1|poseidon2-goldilocks-w8-r4";
const WIDTH: usize = neo_params::poseidon2_goldilocks::WIDTH;
const RATE: usize = neo_params::poseidon2_goldilocks::RATE;
const DIGEST_LEN: usize = neo_params::poseidon2_goldilocks::DIGEST_LEN;
const GOLDILOCKS_S_BOX_DEGREE: u64 = 7;

#[derive(Clone)]
pub struct Poseidon2TranscriptCircuit {
    state: [TranscriptLane; WIDTH],
    absorbed: usize,
}

impl Poseidon2TranscriptCircuit {
    pub fn new_raw_fields<CS: ConstraintSystem<SpartanF>>(
        mut cs: CS,
        fields: &[SpartanF],
    ) -> Result<Self, SynthesisError> {
        let state = core::array::from_fn(|_| TranscriptLane::from_constant(SpartanF::ZERO));
        let mut transcript = Self { state, absorbed: 0 };
        transcript.append_const_fields_raw(cs.namespace(|| "raw_domain"), fields)?;
        Ok(transcript)
    }

    pub fn new<CS: ConstraintSystem<SpartanF>>(mut cs: CS, app_label: &'static [u8]) -> Result<Self, SynthesisError> {
        let state = core::array::from_fn(|_| TranscriptLane::from_constant(SpartanF::ZERO));
        let mut transcript = Self { state, absorbed: 0 };
        transcript.append_message(cs.namespace(|| "app_domain"), APP_DOMAIN, app_label)?;
        Ok(transcript)
    }

    pub fn from_state(
        state: [AllocatedNum<SpartanF>; WIDTH],
        state_values: [SpartanF; WIDTH],
        absorbed: usize,
    ) -> Result<Self, SynthesisError> {
        if absorbed > RATE {
            return Err(SynthesisError::Unsatisfiable);
        }
        Ok(Self {
            state: core::array::from_fn(|i| TranscriptLane::from_allocated(state[i].clone(), state_values[i])),
            absorbed,
        })
    }

    pub fn from_constant_state(state: [SpartanF; WIDTH], absorbed: usize) -> Result<Self, SynthesisError> {
        if absorbed > RATE {
            return Err(SynthesisError::Unsatisfiable);
        }
        Ok(Self {
            state: state.map(TranscriptLane::from_constant),
            absorbed,
        })
    }

    pub fn append_message<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        label: &'static [u8],
        msg: &[u8],
    ) -> Result<(), SynthesisError> {
        self.absorb_packed_bytes_with_len(cs.namespace(|| "label"), label)?;
        self.absorb_packed_bytes_with_len(cs.namespace(|| "msg"), msg)?;
        Ok(())
    }

    pub fn append_fields<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        label: &'static [u8],
        fields: &[AllocatedNum<SpartanF>],
        field_values: &[SpartanF],
    ) -> Result<(), SynthesisError> {
        if fields.len() != field_values.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        self.absorb_packed_bytes_with_len(cs.namespace(|| "label"), label)?;
        self.absorb_constant(
            cs.namespace(|| "field_len"),
            SpartanF::from_canonical_u64(fields.len() as u64),
        )?;
        self.absorb_variable_slice(
            cs.namespace(|| "fields"),
            &fields
                .iter()
                .map(|field| field.get_variable())
                .collect::<Vec<_>>(),
            field_values,
        )?;
        Ok(())
    }

    pub fn append_field_vars<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        label: &'static [u8],
        field_vars: &[Variable],
        field_values: &[SpartanF],
    ) -> Result<(), SynthesisError> {
        if field_vars.len() != field_values.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        self.absorb_packed_bytes_with_len(cs.namespace(|| "label"), label)?;
        self.absorb_constant(
            cs.namespace(|| "field_len"),
            SpartanF::from_canonical_u64(field_vars.len() as u64),
        )?;
        self.absorb_variable_slice(cs.namespace(|| "fields"), field_vars, field_values)?;
        Ok(())
    }

    pub fn append_field_linear_combinations<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        label: &'static [u8],
        field_terms: &[Vec<(Variable, SpartanF)>],
        field_constants: &[SpartanF],
        field_values: &[SpartanF],
    ) -> Result<(), SynthesisError> {
        if field_terms.len() != field_constants.len() || field_terms.len() != field_values.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        self.absorb_packed_bytes_with_len(cs.namespace(|| "label"), label)?;
        self.absorb_constant(
            cs.namespace(|| "field_len"),
            SpartanF::from_canonical_u64(field_terms.len() as u64),
        )?;
        let lanes = field_terms
            .iter()
            .zip(field_constants.iter())
            .zip(field_values.iter())
            .map(|((terms, constant), value)| TranscriptLane::from_terms(terms.clone(), *constant, *value))
            .collect::<Vec<_>>();
        self.absorb_lane_slice(cs.namespace(|| "fields"), &lanes)?;
        Ok(())
    }

    pub fn append_field_linear_combinations_raw<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        field_terms: &[Vec<(Variable, SpartanF)>],
        field_constants: &[SpartanF],
        field_values: &[SpartanF],
    ) -> Result<(), SynthesisError> {
        if field_terms.len() != field_constants.len() || field_terms.len() != field_values.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        self.absorb_constant(
            cs.namespace(|| "field_len"),
            SpartanF::from_canonical_u64(field_terms.len() as u64),
        )?;
        let lanes = field_terms
            .iter()
            .zip(field_constants.iter())
            .zip(field_values.iter())
            .map(|((terms, constant), value)| TranscriptLane::from_terms(terms.clone(), *constant, *value))
            .collect::<Vec<_>>();
        self.absorb_lane_slice(cs.namespace(|| "fields"), &lanes)?;
        Ok(())
    }

    pub fn append_const_fields<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        label: &'static [u8],
        fields: &[SpartanF],
    ) -> Result<(), SynthesisError> {
        self.absorb_packed_bytes_with_len(cs.namespace(|| "label"), label)?;
        self.absorb_constant(
            cs.namespace(|| "field_len"),
            SpartanF::from_canonical_u64(fields.len() as u64),
        )?;
        self.absorb_constant_slice(cs.namespace(|| "fields"), fields)?;
        Ok(())
    }

    pub fn append_const_fields_raw<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        fields: &[SpartanF],
    ) -> Result<(), SynthesisError> {
        self.absorb_constant(
            cs.namespace(|| "field_len"),
            SpartanF::from_canonical_u64(fields.len() as u64),
        )?;
        self.absorb_constant_slice(cs.namespace(|| "fields"), fields)?;
        Ok(())
    }

    pub fn append_u64s<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        label: &'static [u8],
        values: &[u64],
    ) -> Result<(), SynthesisError> {
        self.absorb_packed_bytes_with_len(cs.namespace(|| "label"), label)?;
        self.absorb_constant(
            cs.namespace(|| "u64_len"),
            SpartanF::from_canonical_u64(values.len() as u64),
        )?;
        let packed = pack_u64s(values);
        self.absorb_constant_slice(cs.namespace(|| "u64_words"), &packed)?;
        Ok(())
    }

    pub fn append_u64_halves<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        label: &'static [u8],
        halves: &[AllocatedNum<SpartanF>],
        half_values: &[SpartanF],
        word_count: usize,
    ) -> Result<(), SynthesisError> {
        if halves.len() != half_values.len() || halves.len() != word_count.saturating_mul(2) {
            return Err(SynthesisError::Unsatisfiable);
        }
        self.absorb_packed_bytes_with_len(cs.namespace(|| "label"), label)?;
        self.absorb_constant(
            cs.namespace(|| "u64_len"),
            SpartanF::from_canonical_u64(word_count as u64),
        )?;
        self.absorb_variable_slice(
            cs.namespace(|| "u64_words"),
            &halves
                .iter()
                .map(|half| half.get_variable())
                .collect::<Vec<_>>(),
            half_values,
        )?;
        Ok(())
    }

    pub fn append_packed_bytes<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        label: &'static [u8],
        packed_bytes: &[AllocatedNum<SpartanF>],
        packed_values: &[SpartanF],
        byte_len: usize,
    ) -> Result<(), SynthesisError> {
        if packed_bytes.len() != packed_values.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        self.absorb_packed_bytes_with_len(cs.namespace(|| "label"), label)?;
        self.absorb_constant(
            cs.namespace(|| "byte_len"),
            SpartanF::from_canonical_u64(byte_len as u64),
        )?;
        self.absorb_variable_slice(
            cs.namespace(|| "packed_bytes"),
            &packed_bytes
                .iter()
                .map(|value| value.get_variable())
                .collect::<Vec<_>>(),
            packed_values,
        )?;
        Ok(())
    }

    pub fn append_field_vars_raw<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        field_vars: &[Variable],
        field_values: &[SpartanF],
    ) -> Result<(), SynthesisError> {
        if field_vars.len() != field_values.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        self.absorb_constant(
            cs.namespace(|| "field_len"),
            SpartanF::from_canonical_u64(field_vars.len() as u64),
        )?;
        self.absorb_variable_slice(cs.namespace(|| "fields"), field_vars, field_values)?;
        Ok(())
    }

    pub fn challenge_fields<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        label: &'static [u8],
        n: usize,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        self.append_message(cs.namespace(|| "challenge_label"), b"chal/label", label)?;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            self.absorb_constant(cs.namespace(|| format!("challenge_gate_{}", out.len())), SpartanF::ONE)?;
            self.permute(cs.namespace(|| format!("challenge_permute_{}", out.len())))?;
            for i in 0..DIGEST_LEN.min(n - out.len()) {
                out.push(
                    self.state[i].allocate_canonical(cs.namespace(|| format!("chal_allocate_{}_{i}", out.len())))?,
                );
            }
        }
        Ok(out)
    }

    pub fn challenge_fields_raw<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        n: usize,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            self.absorb_constant(cs.namespace(|| format!("challenge_gate_{}", out.len())), SpartanF::ONE)?;
            self.permute(cs.namespace(|| format!("challenge_permute_{}", out.len())))?;
            for i in 0..DIGEST_LEN.min(n - out.len()) {
                out.push(
                    self.state[i]
                        .allocate_canonical(cs.namespace(|| format!("chal_allocate_raw_{}_{i}", out.len())))?,
                );
            }
        }
        Ok(out)
    }

    pub fn digest32<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        cs: CS,
    ) -> Result<[AllocatedNum<SpartanF>; DIGEST_LEN], SynthesisError> {
        Ok(self.digest32_with_values(cs)?.0)
    }

    pub fn digest32_with_values<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
    ) -> Result<([AllocatedNum<SpartanF>; DIGEST_LEN], [SpartanF; DIGEST_LEN]), SynthesisError> {
        self.absorb_constant(cs.namespace(|| "digest_padding"), SpartanF::ONE)?;
        self.permute(cs.namespace(|| "digest_permute"))?;
        let values = core::array::from_fn(|i| self.state[i].value);
        let mut out = Vec::with_capacity(DIGEST_LEN);
        for i in 0..DIGEST_LEN {
            out.push(self.state[i].allocate_canonical(cs.namespace(|| format!("digest_allocate_{i}")))?);
        }
        Ok((out.try_into().map_err(|_| SynthesisError::Unsatisfiable)?, values))
    }

    pub fn state_values(&self) -> [SpartanF; WIDTH] {
        core::array::from_fn(|i| self.state[i].value)
    }

    pub fn state_fields<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
    ) -> Result<[AllocatedNum<SpartanF>; WIDTH], SynthesisError> {
        let mut out = Vec::with_capacity(WIDTH);
        for i in 0..WIDTH {
            out.push(self.state[i].allocate_canonical(cs.namespace(|| format!("state_allocate_{i}")))?);
        }
        out.try_into().map_err(|_| SynthesisError::Unsatisfiable)
    }

    pub fn enforce_state_values<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        expected: &[SpartanF; WIDTH],
        label: &str,
    ) -> Result<(), SynthesisError> {
        for (idx, (lane, expected_value)) in self.state.iter().zip(expected.iter()).enumerate() {
            cs.enforce(
                || format!("{label}_{idx}"),
                |_| lane.lc::<CS>(),
                |lc| lc + CS::one(),
                |lc| lc + (*expected_value, CS::one()),
            );
        }
        Ok(())
    }

    pub fn absorbed(&self) -> usize {
        self.absorbed
    }

    pub fn constant_snapshot(&self) -> Option<([SpartanF; WIDTH], usize)> {
        self.state_is_constant()
            .then(|| (self.state_values(), self.absorbed))
    }

    pub fn restore_constant_state(&mut self, state: [SpartanF; WIDTH], absorbed: usize) -> Result<(), SynthesisError> {
        if absorbed > RATE {
            return Err(SynthesisError::Unsatisfiable);
        }
        self.state = state.map(TranscriptLane::from_constant);
        self.absorbed = absorbed;
        Ok(())
    }

    fn absorb_constant<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        value: SpartanF,
    ) -> Result<(), SynthesisError> {
        if self.absorbed >= RATE {
            self.permute(cs.namespace(|| "permute"))?;
        }
        self.state[self.absorbed] = TranscriptLane::from_constant(value);
        self.absorbed += 1;
        Ok(())
    }

    fn absorb_constant_slice<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        values: &[SpartanF],
    ) -> Result<(), SynthesisError> {
        let mut idx = 0usize;
        while self.absorbed < RATE && idx < values.len() {
            self.state[self.absorbed] = TranscriptLane::from_constant(values[idx]);
            self.absorbed += 1;
            idx += 1;
        }
        if self.absorbed == RATE {
            self.permute(cs.namespace(|| "const_fill_permute"))?;
        }
        while values.len() - idx >= RATE {
            for lane in 0..RATE {
                self.state[lane] = TranscriptLane::from_constant(values[idx + lane]);
            }
            self.absorbed = RATE;
            self.permute(cs.namespace(|| format!("const_chunk_permute_{idx}")))?;
            idx += RATE;
        }
        while idx < values.len() {
            self.state[self.absorbed] = TranscriptLane::from_constant(values[idx]);
            self.absorbed += 1;
            idx += 1;
        }
        Ok(())
    }

    fn absorb_variable_slice<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        values: &[Variable],
        value_natives: &[SpartanF],
    ) -> Result<(), SynthesisError> {
        if values.len() != value_natives.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        let mut idx = 0usize;
        while self.absorbed < RATE && idx < values.len() {
            self.state[self.absorbed] = TranscriptLane::from_variable(values[idx], value_natives[idx]);
            self.absorbed += 1;
            idx += 1;
        }
        if self.absorbed == RATE {
            self.permute(cs.namespace(|| "value_fill_permute"))?;
        }
        while values.len() - idx >= RATE {
            for lane in 0..RATE {
                self.state[lane] = TranscriptLane::from_variable(values[idx + lane], value_natives[idx + lane]);
            }
            self.absorbed = RATE;
            self.permute(cs.namespace(|| format!("value_chunk_permute_{idx}")))?;
            idx += RATE;
        }
        while idx < values.len() {
            self.state[self.absorbed] = TranscriptLane::from_variable(values[idx], value_natives[idx]);
            self.absorbed += 1;
            idx += 1;
        }
        Ok(())
    }

    fn absorb_lane_slice<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        lanes: &[TranscriptLane],
    ) -> Result<(), SynthesisError> {
        let mut idx = 0usize;
        while self.absorbed < RATE && idx < lanes.len() {
            self.state[self.absorbed] = lanes[idx].clone();
            self.absorbed += 1;
            idx += 1;
        }
        if self.absorbed == RATE {
            self.permute(cs.namespace(|| "lane_fill_permute"))?;
        }
        while lanes.len() - idx >= RATE {
            for lane in 0..RATE {
                self.state[lane] = lanes[idx + lane].clone();
            }
            self.absorbed = RATE;
            self.permute(cs.namespace(|| format!("lane_chunk_permute_{idx}")))?;
            idx += RATE;
        }
        while idx < lanes.len() {
            self.state[self.absorbed] = lanes[idx].clone();
            self.absorbed += 1;
            idx += 1;
        }
        Ok(())
    }

    fn absorb_packed_bytes_with_len<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        bytes: &[u8],
    ) -> Result<(), SynthesisError> {
        self.absorb_constant(
            cs.namespace(|| "byte_len"),
            SpartanF::from_canonical_u64(bytes.len() as u64),
        )?;
        let packed = pack_bytes(bytes);
        self.absorb_constant_slice(cs.namespace(|| "packed_bytes"), &packed)?;
        Ok(())
    }

    fn permute<CS: ConstraintSystem<SpartanF>>(&mut self, cs: CS) -> Result<(), SynthesisError> {
        self.state = permute_state(cs, &self.state)?;
        self.absorbed = 0;
        Ok(())
    }

    fn state_is_constant(&self) -> bool {
        self.state.iter().all(TranscriptLane::is_constant)
    }
}

pub(crate) fn hash_field_linear_combinations_raw<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    field_terms: &[Vec<(Variable, SpartanF)>],
    field_constants: &[SpartanF],
    field_values: &[SpartanF],
) -> Result<[AllocatedNum<SpartanF>; DIGEST_LEN], SynthesisError> {
    if field_terms.len() != field_constants.len() || field_terms.len() != field_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let lanes = field_terms
        .iter()
        .zip(field_constants.iter())
        .zip(field_values.iter())
        .map(|((terms, constant), value)| TranscriptLane::from_terms(terms.clone(), *constant, *value))
        .collect::<Vec<_>>();
    hash_lane_slice_raw(cs.namespace(|| "hash_field_linear_combinations"), &lanes)
}

fn hash_lane_slice_raw<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    lanes: &[TranscriptLane],
) -> Result<[AllocatedNum<SpartanF>; DIGEST_LEN], SynthesisError> {
    let mut state = core::array::from_fn(|_| TranscriptLane::from_constant(SpartanF::ZERO));

    for (chunk_idx, chunk) in lanes.chunks(RATE).enumerate() {
        for (lane_idx, lane) in chunk.iter().enumerate() {
            state[lane_idx] = state[lane_idx].add(lane);
        }
        state = permute_state(cs.namespace(|| format!("permute_after_chunk_{chunk_idx}")), &state)?;
    }

    state[0] = state[0].add(&TranscriptLane::from_constant(SpartanF::ONE));
    state = permute_state(cs.namespace(|| "permute_after_padding"), &state)?;

    let mut out = Vec::with_capacity(DIGEST_LEN);
    for digest_idx in 0..DIGEST_LEN {
        out.push(state[digest_idx].allocate_canonical(cs.namespace(|| format!("digest_{digest_idx}")))?);
    }
    out.try_into().map_err(|_| SynthesisError::Unsatisfiable)
}
