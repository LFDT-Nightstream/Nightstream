//! Absorb and permutation internals for the Poseidon2 transcript circuit.

use super::permutation::permute_state;
use super::*;

impl Poseidon2TranscriptCircuit {
    pub(super) fn absorb_constant<CS: ConstraintSystem<SpartanF>>(
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

    pub(super) fn absorb_constant_slice<CS: ConstraintSystem<SpartanF>>(
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

    pub(super) fn absorb_variable_slice<CS: ConstraintSystem<SpartanF>>(
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

    pub(super) fn absorb_lane_slice<CS: ConstraintSystem<SpartanF>>(
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

    pub(super) fn absorb_packed_bytes_with_len<CS: ConstraintSystem<SpartanF>>(
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

    pub(super) fn permute<CS: ConstraintSystem<SpartanF>>(&mut self, cs: CS) -> Result<(), SynthesisError> {
        self.state = permute_state(cs, &self.state)?;
        self.absorbed = 0;
        Ok(())
    }
}
