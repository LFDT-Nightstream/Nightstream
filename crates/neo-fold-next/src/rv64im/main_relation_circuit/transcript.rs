//! Owns an in-circuit Poseidon2 transcript matching `neo_transcript::Poseidon2Transcript`.
//!
//! This module only owns transcript state evolution and challenge squeezing.
//! It does not own RV64IM relation semantics.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, Index, SynthesisError, Variable};
use core::cmp::Ordering;
use ff::Field;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, MATRIX_DIAG_8_GOLDILOCKS};
use p3_poseidon2::{poseidon2_round_numbers_128, ExternalLayerConstants};
use rand_chacha_p3::rand_core::{Rng, SeedableRng};
use rand_chacha_p3::ChaCha8Rng;
use spartan2::provider::goldi::F as SpartanF;
use std::sync::LazyLock;

const APP_DOMAIN: &[u8] = b"neo/transcript/v1|poseidon2-goldilocks-w8-r4";
const WIDTH: usize = neo_params::poseidon2_goldilocks::WIDTH;
const RATE: usize = neo_params::poseidon2_goldilocks::RATE;
const DIGEST_LEN: usize = neo_params::poseidon2_goldilocks::DIGEST_LEN;
const GOLDILOCKS_S_BOX_DEGREE: u64 = 7;

struct Poseidon2RoundConstants {
    initial: Vec<[SpartanF; WIDTH]>,
    terminal: Vec<[SpartanF; WIDTH]>,
    internal: Vec<SpartanF>,
    internal_diag_m_1: [SpartanF; WIDTH],
}

static POSEIDON2_CONSTANTS: LazyLock<Poseidon2RoundConstants> = LazyLock::new(build_poseidon2_constants);

#[derive(Clone)]
pub struct Poseidon2TranscriptCircuit {
    state: [TranscriptLane; WIDTH],
    absorbed: usize,
}

#[derive(Clone)]
struct TranscriptLane {
    terms: Vec<(Variable, SpartanF)>,
    constant: SpartanF,
    value: SpartanF,
    allocated: Option<AllocatedNum<SpartanF>>,
}

impl TranscriptLane {
    fn from_allocated(value: AllocatedNum<SpartanF>, native: SpartanF) -> Self {
        Self {
            terms: vec![(value.get_variable(), SpartanF::ONE)],
            constant: SpartanF::ZERO,
            value: native,
            allocated: Some(value),
        }
    }

    fn from_variable(variable: Variable, native: SpartanF) -> Self {
        Self {
            terms: vec![(variable, SpartanF::ONE)],
            constant: SpartanF::ZERO,
            value: native,
            allocated: None,
        }
    }

    fn from_terms(terms: Vec<(Variable, SpartanF)>, constant: SpartanF, native: SpartanF) -> Self {
        Self {
            terms: compact_terms(terms),
            constant,
            value: native,
            allocated: None,
        }
    }

    fn from_constant(native: SpartanF) -> Self {
        Self {
            terms: vec![],
            constant: native,
            value: native,
            allocated: None,
        }
    }

    fn is_constant(&self) -> bool {
        self.terms.is_empty()
    }

    fn add(&self, other: &Self) -> Self {
        let terms = if self.terms.is_empty() {
            other.terms.clone()
        } else if other.terms.is_empty() {
            self.terms.clone()
        } else {
            merge_compact_terms(&self.terms, &other.terms)
        };
        Self {
            terms,
            constant: self.constant + other.constant,
            value: self.value + other.value,
            allocated: None,
        }
    }

    fn lc<CS: ConstraintSystem<SpartanF>>(&self) -> bellpepper_core::LinearCombination<SpartanF> {
        let mut res = bellpepper_core::LinearCombination::zero();
        res = res + (self.constant, CS::one());
        for (v, c) in &self.terms {
            res = res + (*c, *v);
        }
        res
    }

    fn ensure_allocated<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
    ) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
        if let Some(alloc) = &self.allocated {
            return Ok(alloc.clone());
        }
        let out = AllocatedNum::alloc(cs.namespace(|| "alloc"), || Ok(self.value))?;
        cs.enforce(
            || "enforce_alloc",
            |lc| lc + out.get_variable(),
            |lc| lc + CS::one(),
            |_| self.lc::<CS>(),
        );
        self.allocated = Some(out.clone());
        self.terms = vec![(out.get_variable(), SpartanF::ONE)];
        self.constant = SpartanF::ZERO;
        Ok(out)
    }
}

fn compact_terms(mut terms: Vec<(Variable, SpartanF)>) -> Vec<(Variable, SpartanF)> {
    if terms.len() <= 1 {
        return terms
            .into_iter()
            .filter(|(_, coeff)| *coeff != SpartanF::ZERO)
            .collect();
    }

    terms.sort_unstable_by(|(left, _), (right, _)| compare_variables(*left, *right));

    let mut compacted = Vec::with_capacity(terms.len());
    for (variable, coeff) in terms {
        if coeff == SpartanF::ZERO {
            continue;
        }
        if let Some((last_variable, last_coeff)) = compacted.last_mut() {
            if *last_variable == variable {
                *last_coeff += coeff;
                if *last_coeff == SpartanF::ZERO {
                    compacted.pop();
                }
                continue;
            }
        }
        compacted.push((variable, coeff));
    }
    compacted
}

fn compare_variables(left: Variable, right: Variable) -> Ordering {
    match (left.get_unchecked(), right.get_unchecked()) {
        (Index::Input(left_idx), Index::Input(right_idx)) => left_idx.cmp(&right_idx),
        (Index::Aux(left_idx), Index::Aux(right_idx)) => left_idx.cmp(&right_idx),
        (Index::Input(_), Index::Aux(_)) => Ordering::Less,
        (Index::Aux(_), Index::Input(_)) => Ordering::Greater,
    }
}

fn merge_compact_terms(left: &[(Variable, SpartanF)], right: &[(Variable, SpartanF)]) -> Vec<(Variable, SpartanF)> {
    let mut merged = Vec::with_capacity(left.len() + right.len());
    let mut left_idx = 0usize;
    let mut right_idx = 0usize;

    while left_idx < left.len() && right_idx < right.len() {
        let (left_var, left_coeff) = left[left_idx];
        let (right_var, right_coeff) = right[right_idx];
        match compare_variables(left_var, right_var) {
            Ordering::Less => {
                merged.push((left_var, left_coeff));
                left_idx += 1;
            }
            Ordering::Greater => {
                merged.push((right_var, right_coeff));
                right_idx += 1;
            }
            Ordering::Equal => {
                let coeff = left_coeff + right_coeff;
                if coeff != SpartanF::ZERO {
                    merged.push((left_var, coeff));
                }
                left_idx += 1;
                right_idx += 1;
            }
        }
    }

    merged.extend_from_slice(&left[left_idx..]);
    merged.extend_from_slice(&right[right_idx..]);
    merged
}

fn combine_scaled_lanes(lanes: &[(&TranscriptLane, SpartanF)]) -> TranscriptLane {
    let mut terms_len = 0usize;
    let mut constant = SpartanF::ZERO;
    let mut value = SpartanF::ZERO;
    for (lane, scalar) in lanes {
        if *scalar == SpartanF::ZERO {
            continue;
        }
        terms_len += lane.terms.len();
        constant += lane.constant * *scalar;
        value += lane.value * *scalar;
    }
    if terms_len == 0 {
        return TranscriptLane::from_constant(value);
    }

    let mut terms = Vec::with_capacity(terms_len);
    for (lane, scalar) in lanes {
        if *scalar == SpartanF::ZERO || lane.terms.is_empty() {
            continue;
        }
        if *scalar == SpartanF::ONE {
            terms.extend(lane.terms.iter().copied());
        } else {
            terms.extend(
                lane.terms
                    .iter()
                    .map(|(variable, coeff)| (*variable, *coeff * *scalar)),
            );
        }
    }

    TranscriptLane {
        terms: compact_terms(terms),
        constant,
        value,
        allocated: None,
    }
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
                out.push(self.state[i].ensure_allocated(cs.namespace(|| format!("chal_allocate_{}_{i}", out.len())))?);
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
                    self.state[i].ensure_allocated(cs.namespace(|| format!("chal_allocate_raw_{}_{i}", out.len())))?,
                );
            }
        }
        Ok(out)
    }

    pub fn digest32<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
    ) -> Result<[AllocatedNum<SpartanF>; DIGEST_LEN], SynthesisError> {
        self.absorb_constant(cs.namespace(|| "digest_padding"), SpartanF::ONE)?;
        self.permute(cs.namespace(|| "digest_permute"))?;
        Ok(core::array::from_fn(|i| {
            self.state[i]
                .ensure_allocated(cs.namespace(|| format!("digest_allocate_{i}")))
                .expect("digest lanes must be allocated")
        }))
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
            out.push(self.state[i].ensure_allocated(cs.namespace(|| format!("state_allocate_{i}")))?);
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
        out.push(state[digest_idx].ensure_allocated(cs.namespace(|| format!("digest_{digest_idx}")))?);
    }
    out.try_into().map_err(|_| SynthesisError::Unsatisfiable)
}

fn pack_bytes(bytes: &[u8]) -> Vec<SpartanF> {
    const BYTES_PER_LIMB: usize = 7;
    let mut packed = Vec::with_capacity(bytes.len().div_ceil(BYTES_PER_LIMB));
    let mut i = 0usize;
    while i < bytes.len() {
        let end = (i + BYTES_PER_LIMB).min(bytes.len());
        let mut limb = [0u8; 8];
        limb[..(end - i)].copy_from_slice(&bytes[i..end]);
        packed.push(SpartanF::from_canonical_u64(u64::from_le_bytes(limb)));
        i = end;
    }
    packed
}

fn pack_u64s(values: &[u64]) -> Vec<SpartanF> {
    let mut packed = Vec::with_capacity(values.len() * 2);
    for value in values {
        packed.push(SpartanF::from_canonical_u64(value & 0xFFFF_FFFF));
        packed.push(SpartanF::from_canonical_u64(value >> 32));
    }
    packed
}

fn build_poseidon2_constants() -> Poseidon2RoundConstants {
    let mut rng = ChaCha8Rng::from_seed(neo_params::poseidon2_goldilocks::SEED);
    let (rounds_f, rounds_p) =
        poseidon2_round_numbers_128::<Goldilocks>(WIDTH, GOLDILOCKS_S_BOX_DEGREE).expect("Poseidon2 width 8 rounds");
    let external = ExternalLayerConstants::<Goldilocks, WIDTH>::new_from_rng(rounds_f, &mut rng);
    let internal = (0..rounds_p)
        .map(|_| convert_goldilocks(Goldilocks::from_u64(rng.next_u64())))
        .collect::<Vec<_>>();

    Poseidon2RoundConstants {
        initial: external
            .get_initial_constants()
            .iter()
            .copied()
            .map(convert_goldilocks_array)
            .collect(),
        terminal: external
            .get_terminal_constants()
            .iter()
            .copied()
            .map(convert_goldilocks_array)
            .collect(),
        internal,
        internal_diag_m_1: convert_goldilocks_array(MATRIX_DIAG_8_GOLDILOCKS),
    }
}

fn permute_state<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    state: &[TranscriptLane; WIDTH],
) -> Result<[TranscriptLane; WIDTH], SynthesisError> {
    let constants = &*POSEIDON2_CONSTANTS;

    let mut state = external_linear_layer(cs.namespace(|| "initial_external_layer"), state)?;

    for (round_idx, round_constants) in constants.initial.iter().enumerate() {
        let mut next = state.clone();
        for i in 0..WIDTH {
            next[i] = sbox_with_round_constant(
                cs.namespace(|| format!("initial_round_{round_idx}_{i}")),
                &state[i],
                round_constants[i],
            )?;
        }
        state = external_linear_layer(cs.namespace(|| format!("initial_round_{round_idx}_linear")), &next)?;
    }

    for (round_idx, round_constant) in constants.internal.iter().copied().enumerate() {
        let mut next = state.clone();
        next[0] = sbox_with_round_constant(
            cs.namespace(|| format!("internal_round_{round_idx}_0")),
            &state[0],
            round_constant,
        )?;
        state = internal_linear_layer(
            cs.namespace(|| format!("internal_round_{round_idx}_linear")),
            &next,
            constants.internal_diag_m_1,
        )?;
    }

    for (round_idx, round_constants) in constants.terminal.iter().enumerate() {
        let mut next = state.clone();
        for i in 0..WIDTH {
            next[i] = sbox_with_round_constant(
                cs.namespace(|| format!("terminal_round_{round_idx}_{i}")),
                &state[i],
                round_constants[i],
            )?;
        }
        state = external_linear_layer(cs.namespace(|| format!("terminal_round_{round_idx}_linear")), &next)?;
    }

    Ok(state)
}

fn sbox_with_round_constant<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    input: &TranscriptLane,
    round_constant: SpartanF,
) -> Result<TranscriptLane, SynthesisError> {
    let shifted_value = input.value + round_constant;

    let shifted_sq_value = shifted_value.square();
    let shifted_sq = AllocatedNum::alloc(cs.namespace(|| "shift_sq"), || Ok(shifted_sq_value))?;
    cs.enforce(
        || "shift_sq_enforce",
        |_| input.lc::<CS>() + (round_constant, CS::one()),
        |_| input.lc::<CS>() + (round_constant, CS::one()),
        |lc| lc + shifted_sq.get_variable(),
    );
    let shifted_sq_lane = TranscriptLane::from_allocated(shifted_sq, shifted_sq_value);

    let shifted_4 = square_lane(cs.namespace(|| "shift_4"), &shifted_sq_lane)?;
    let shifted_6 = mul_lanes(cs.namespace(|| "shift_6"), &shifted_4, &shifted_sq_lane)?;

    let out_value = shifted_6.value * shifted_value;
    let out = AllocatedNum::alloc(cs.namespace(|| "out"), || Ok(out_value))?;
    cs.enforce(
        || "out_enforce",
        |_| shifted_6.lc::<CS>(),
        |_| input.lc::<CS>() + (round_constant, CS::one()),
        |lc| lc + out.get_variable(),
    );

    Ok(TranscriptLane::from_allocated(out, out_value))
}

fn external_linear_layer<CS: ConstraintSystem<SpartanF>>(
    _cs: CS,
    state: &[TranscriptLane; WIDTH],
) -> Result<[TranscriptLane; WIDTH], SynthesisError> {
    let left = apply_mat4(&state[0..4]);
    let right = apply_mat4(&state[4..8]);

    let two = SpartanF::from_canonical_u64(2);
    let mut out = core::array::from_fn(|i| left[i % 4].clone());
    for i in 0..4 {
        out[i] = combine_scaled_lanes(&[(&left[i], two), (&right[i], SpartanF::ONE)]);
        out[i + 4] = combine_scaled_lanes(&[(&left[i], SpartanF::ONE), (&right[i], two)]);
    }
    Ok(out)
}

fn apply_mat4(state: &[TranscriptLane]) -> [TranscriptLane; 4] {
    let two = SpartanF::from_canonical_u64(2);
    let three = SpartanF::from_canonical_u64(3);

    let row_0 = combine_scaled_lanes(&[
        (&state[0], two),
        (&state[1], three),
        (&state[2], SpartanF::ONE),
        (&state[3], SpartanF::ONE),
    ]);
    let row_1 = combine_scaled_lanes(&[
        (&state[0], SpartanF::ONE),
        (&state[1], two),
        (&state[2], three),
        (&state[3], SpartanF::ONE),
    ]);
    let row_2 = combine_scaled_lanes(&[
        (&state[0], SpartanF::ONE),
        (&state[1], SpartanF::ONE),
        (&state[2], two),
        (&state[3], three),
    ]);
    let row_3 = combine_scaled_lanes(&[
        (&state[0], three),
        (&state[1], SpartanF::ONE),
        (&state[2], SpartanF::ONE),
        (&state[3], two),
    ]);

    [row_0, row_1, row_2, row_3]
}

fn internal_linear_layer<CS: ConstraintSystem<SpartanF>>(
    _cs: CS,
    state: &[TranscriptLane; WIDTH],
    diag_m_1: [SpartanF; WIDTH],
) -> Result<[TranscriptLane; WIDTH], SynthesisError> {
    let sum_inputs = state
        .iter()
        .map(|lane| (lane, SpartanF::ONE))
        .collect::<Vec<_>>();
    let sum = combine_scaled_lanes(&sum_inputs);

    let out = core::array::from_fn(|i| combine_scaled_lanes(&[(&sum, SpartanF::ONE), (&state[i], diag_m_1[i])]));
    Ok(out)
}

fn square_lane<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    input: &TranscriptLane,
) -> Result<TranscriptLane, SynthesisError> {
    let out_value = input.value.square();
    let out = AllocatedNum::alloc(cs.namespace(|| "value"), || Ok(out_value))?;
    cs.enforce(
        || "square",
        |_| input.lc::<CS>(),
        |_| input.lc::<CS>(),
        |lc| lc + out.get_variable(),
    );
    Ok(TranscriptLane::from_allocated(out, out_value))
}

fn mul_lanes<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    left: &TranscriptLane,
    right: &TranscriptLane,
) -> Result<TranscriptLane, SynthesisError> {
    let out_value = left.value * right.value;
    let out = AllocatedNum::alloc(cs.namespace(|| "value"), || Ok(out_value))?;
    cs.enforce(
        || "mul",
        |_| left.lc::<CS>(),
        |_| right.lc::<CS>(),
        |lc| lc + out.get_variable(),
    );
    Ok(TranscriptLane::from_allocated(out, out_value))
}

fn convert_goldilocks(value: Goldilocks) -> SpartanF {
    SpartanF::from_canonical_u64(value.as_canonical_u64())
}

fn convert_goldilocks_array<const N: usize>(values: [Goldilocks; N]) -> [SpartanF; N] {
    values.map(convert_goldilocks)
}
