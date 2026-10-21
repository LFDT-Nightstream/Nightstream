//! Π_CCS, Π_RLC, and Π_DEC engines. Owns the optimized and paper-exact
//! backends, the cross-checking wrapper, and the shared protocol and receipt
//! modules.

#![allow(non_snake_case)]

pub mod utils;

#[derive(Clone, Debug, PartialEq)]
pub struct DecOutcome<Ff = neo_math::F> {
    pub children: Vec<neo_ccs::CeClaim<neo_ajtai::Commitment, Ff, neo_math::K>>,
    pub ok_y: bool,
    pub ok_x: bool,
    pub ok_c: bool,
}

impl<Ff> DecOutcome<Ff> {
    pub fn rejected() -> Self {
        Self {
            children: Vec::new(),
            ok_y: false,
            ok_x: false,
            ok_c: false,
        }
    }
}

pub type RlcMixer<Ff = neo_math::F> = fn(&[neo_ccs::Mat<Ff>], &[neo_ajtai::Commitment]) -> neo_ajtai::Commitment;

pub mod optimized_engine;
pub mod pi_ccs;
pub mod pi_ccs_joint;
pub mod pi_ccs_joint_protocol;
pub mod pi_ccs_protocol;
pub mod pi_rlc_dec;

#[cfg(feature = "paper-exact")]
pub mod paper_exact_engine;

#[cfg(feature = "paper-exact")]
pub mod crosscheck_engine;

pub use pi_ccs::{OptimizedEngine, PiCcsEngine, PiCcsProof};

#[cfg(feature = "paper-exact")]
pub use pi_ccs::PaperExactEngine;

#[cfg(feature = "paper-exact")]
pub use crosscheck_engine::CrossCheckEngine;
