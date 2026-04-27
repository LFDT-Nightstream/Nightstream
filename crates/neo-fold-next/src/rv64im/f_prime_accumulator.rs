//! Owns the RV64IM F' carried-accumulator slot surface.

use crate::proof::Carry;
use crate::rv64im::SimpleKernelError;

pub const RV64IM_MAIN_RECURSION_ACCUMULATOR_SLOTS: usize = 1;

#[derive(Clone, Debug)]
pub(crate) struct Rv64imMainRecursionAccumulatorBundle {
    main: Carry,
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imMainRecursionAccumulatorArray<const SLOTS: usize> {
    slots: [Rv64imMainRecursionAccumulatorBundle; SLOTS],
}

pub(crate) type Rv64imMainRecursionAccumulatorSurface =
    Rv64imMainRecursionAccumulatorArray<RV64IM_MAIN_RECURSION_ACCUMULATOR_SLOTS>;

impl<const SLOTS: usize> Rv64imMainRecursionAccumulatorArray<SLOTS> {
    pub(crate) fn try_from_carry(main: &Carry, label: &str) -> Result<Self, SimpleKernelError> {
        if main.claims.len() != main.witnesses.len() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM main recursion {label} requires one witness per carried CE claim"
            )));
        }
        if SLOTS != 1 {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM main recursion {label} only supports the current single-PC specialization"
            )));
        }
        Ok(Self {
            slots: core::array::from_fn(|_| Rv64imMainRecursionAccumulatorBundle { main: main.clone() }),
        })
    }

    pub(crate) fn slot(&self, slot: usize) -> Result<&Rv64imMainRecursionAccumulatorBundle, SimpleKernelError> {
        self.slots.get(slot).ok_or_else(|| {
            SimpleKernelError::Bridge(format!(
                "RV64IM main recursion accumulator slot {slot} is out of bounds for {SLOTS} slots"
            ))
        })
    }
}

impl Rv64imMainRecursionAccumulatorBundle {
    pub(crate) fn carry(&self) -> &Carry {
        &self.main
    }
}
