use bellpepper_core::num::AllocatedNum;
use bellpepper_core::{ConstraintSystem, SynthesisError};

use crate::gadgets::poseidon2::{permute_w8, RATE, WIDTH};
use crate::CircuitF;

#[derive(Clone)]
pub struct Poseidon2Sponge {
    pub state: [AllocatedNum<CircuitF>; WIDTH],
    absorbed: usize,
    permute_count: usize,
    one: AllocatedNum<CircuitF>,
    scope: String,
}

impl Poseidon2Sponge {
    pub fn set_scope(&mut self, scope: String) {
        self.scope = scope;
    }

    pub fn new<CS: ConstraintSystem<CircuitF>>(cs: &mut CS, label: &str) -> Result<Self, SynthesisError> {
        let mut state: Vec<AllocatedNum<CircuitF>> = Vec::with_capacity(WIDTH);
        for i in 0..WIDTH {
            let z = AllocatedNum::alloc(cs.namespace(|| format!("{label}_st_{i}")), || Ok(CircuitF::from(0u64)))?;
            cs.enforce(
                || format!("{label}_st_{i}_is_zero"),
                |lc| lc + z.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc,
            );
            state.push(z);
        }
        let state: [AllocatedNum<CircuitF>; WIDTH] = state
            .try_into()
            .map_err(|_| SynthesisError::Unsatisfiable)?;

        let one = AllocatedNum::alloc(cs.namespace(|| format!("{label}_one")), || Ok(CircuitF::from(1u64)))?;
        cs.enforce(
            || format!("{label}_one_is_one"),
            |lc| lc + one.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + (CircuitF::from(1u64), CS::one()),
        );
        Ok(Self {
            state,
            absorbed: 0,
            permute_count: 0,
            one,
            scope: label.to_owned(),
        })
    }

    pub fn absorb<CS: ConstraintSystem<CircuitF>>(
        &mut self,
        cs: &mut CS,
        x: AllocatedNum<CircuitF>,
    ) -> Result<(), SynthesisError> {
        // `absorb_elem` semantics (matches native `Poseidon2Transcript::absorb_elem`):
        // permute *before* writing if buffer is full.
        if self.absorbed >= RATE {
            self.permute(cs, &format!("{}_permute_{}", self.scope, self.permute_count))?;
        }
        self.state[self.absorbed] = x;
        self.absorbed += 1;
        Ok(())
    }

    /// Absorb a field element with `absorb_slice` semantics (matches native `Poseidon2Transcript::absorb_slice`):
    /// permute when the buffer becomes full *after* writing.
    pub fn absorb_slice_elem<CS: ConstraintSystem<CircuitF>>(
        &mut self,
        cs: &mut CS,
        x: AllocatedNum<CircuitF>,
    ) -> Result<(), SynthesisError> {
        // Handle the case where a previous `absorb_elem` filled the buffer exactly.
        if self.absorbed >= RATE {
            self.permute(cs, &format!("{}_permute_{}", self.scope, self.permute_count))?;
        }
        self.state[self.absorbed] = x;
        self.absorbed += 1;
        if self.absorbed == RATE {
            self.permute(cs, &format!("{}_permute_{}", self.scope, self.permute_count))?;
        }
        Ok(())
    }

    fn permute<CS: ConstraintSystem<CircuitF>>(&mut self, cs: &mut CS, label: &str) -> Result<(), SynthesisError> {
        let mut cs_ns = cs.namespace(|| label.to_string());
        self.permute_count += 1;
        permute_w8(&mut cs_ns, &mut self.state)?;
        self.absorbed = 0;
        Ok(())
    }

    pub fn digest32<CS: ConstraintSystem<CircuitF>>(
        &mut self,
        cs: &mut CS,
        label: &str,
    ) -> Result<[AllocatedNum<CircuitF>; 4], SynthesisError> {
        // Domain gate before squeezing (must be constrained to 1).
        self.absorb(cs, self.one.clone())?;
        self.permute(cs, &format!("{}_{}_permute_{}", self.scope, label, self.permute_count))?;
        Ok([
            self.state[0].clone(),
            self.state[1].clone(),
            self.state[2].clone(),
            self.state[3].clone(),
        ])
    }
}
