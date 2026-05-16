//! Snapshot and state-access helpers for the Poseidon2 transcript circuit.

use super::*;

impl Poseidon2TranscriptCircuit {
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

    fn state_is_constant(&self) -> bool {
        self.state.iter().all(TranscriptLane::is_constant)
    }
}
