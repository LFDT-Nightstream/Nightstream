//! Owns the RV32IM F' carried-accumulator slot surface.

use crate::proof::Carry;
use crate::rv32im::SimpleKernelError;

pub const RV32IM_MAIN_RECURSION_ACCUMULATOR_SLOTS: usize = 1;

#[derive(Clone, Debug)]
pub(crate) struct Rv32imMainRecursionAccumulatorBundle {
    main: Carry,
}

#[derive(Clone, Debug)]
pub(crate) struct Rv32imMainRecursionAccumulatorArray<const SLOTS: usize> {
    slots: [Rv32imMainRecursionAccumulatorBundle; SLOTS],
}

pub(crate) type Rv32imMainRecursionAccumulatorSurface =
    Rv32imMainRecursionAccumulatorArray<RV32IM_MAIN_RECURSION_ACCUMULATOR_SLOTS>;

impl<const SLOTS: usize> Rv32imMainRecursionAccumulatorArray<SLOTS> {
    pub(crate) fn try_from_carry(main: &Carry, label: &str) -> Result<Self, SimpleKernelError> {
        if main.claims.len() != main.witnesses.len() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM main recursion {label} requires one witness per carried CE claim"
            )));
        }
        if SLOTS != 1 {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM main recursion {label} only supports the current single-PC specialization"
            )));
        }
        Ok(Self {
            slots: core::array::from_fn(|_| Rv32imMainRecursionAccumulatorBundle { main: main.clone() }),
        })
    }

    pub(crate) fn slot(&self, slot: usize) -> Result<&Rv32imMainRecursionAccumulatorBundle, SimpleKernelError> {
        self.slots.get(slot).ok_or_else(|| {
            SimpleKernelError::Bridge(format!(
                "RV32IM main recursion accumulator slot {slot} is out of bounds for {SLOTS} slots"
            ))
        })
    }
}

impl Rv32imMainRecursionAccumulatorBundle {
    pub(crate) fn carry(&self) -> &Carry {
        &self.main
    }
}
