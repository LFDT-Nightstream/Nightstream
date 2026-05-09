//! Recursive direct-CCS state machine.
//!
//! This module owns the live carrier state. The child modules keep the public
//! flow separated by protocol operation: append, summarize, then compress.

use super::*;

mod append;
mod compress;
mod report;

#[derive(Clone)]
pub struct DirectCcsRecursiveIvcState {
    direct: DirectCcsIvcState,
    f_prime_chain: DirectCcsFPrimeChain,
}

impl DirectCcsRecursiveIvcState {
    pub fn new_with_canonical_zero_carry(program: DirectCcsProgram) -> Result<Self, DirectCcsFPrimeSnarkError> {
        Ok(Self {
            direct: DirectCcsIvcState::new_with_canonical_zero_carry(program)?,
            f_prime_chain: DirectCcsFPrimeChain::new(),
        })
    }

    pub fn direct_state(&self) -> &DirectCcsIvcState {
        &self.direct
    }
}
