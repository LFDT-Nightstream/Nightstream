//! In-circuit Poseidon2 transcript matching `neo_transcript::Poseidon2Transcript`.
//!
//! This module owns transcript state evolution and challenge squeezing. It does
//! not own RV32IM or direct-CCS relation semantics.

mod absorb;
mod hash;
mod lane;
mod packing;
mod permutation;
mod snapshot;
mod squeeze;

use lane::TranscriptLane;
use packing::{pack_bytes, pack_u64s};

use crate::spartan_backend::SpartanF;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError, Variable};
use ff::Field;

pub(crate) use hash::hash_field_linear_combinations_raw;

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
}
