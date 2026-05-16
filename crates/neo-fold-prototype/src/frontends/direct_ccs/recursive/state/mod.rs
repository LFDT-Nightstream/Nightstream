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
    pub fn start(program: DirectCcsProgram) -> Result<Self, DirectCcsFPrimeSnarkError> {
        Ok(Self {
            direct: DirectCcsIvcState::start(program)?,
            f_prime_chain: DirectCcsFPrimeChain::new(),
        })
    }

    pub fn direct_state(&self) -> &DirectCcsIvcState {
        &self.direct
    }
}

pub fn start_direct_ccs_proof_state(
    program: DirectCcsProgram,
) -> Result<DirectCcsRecursiveIvcState, DirectCcsFPrimeSnarkError> {
    DirectCcsRecursiveIvcState::start(program)
}
