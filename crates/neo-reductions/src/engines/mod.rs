//! Π_CCS, Π_RLC, and Π_DEC engines. Owns the optimized and paper-exact
//! backends, the cross-checking wrapper, and the shared protocol and receipt
//! modules.

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

    pub fn into_children(self) -> Result<Vec<neo_ccs::CeClaim<neo_ajtai::Commitment, Ff, neo_math::K>>, DecRejected> {
        if self.ok_y && self.ok_x && self.ok_c {
            Ok(self.children)
        } else {
            Err(DecRejected {
                ok_y: self.ok_y,
                ok_x: self.ok_x,
                ok_c: self.ok_c,
            })
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecRejected {
    pub ok_y: bool,
    pub ok_x: bool,
    pub ok_c: bool,
}

pub type RlcMixer<Ff = neo_math::F> = fn(&[neo_ccs::Mat<Ff>], &[neo_ajtai::Commitment]) -> neo_ajtai::Commitment;

pub mod optimized_engine;
pub mod pi_ccs_joint;
pub mod pi_ccs_joint_protocol;
pub mod pi_ccs_protocol;
pub mod pi_rlc_dec;

#[cfg(feature = "paper-exact")]
pub mod paper_exact_engine;

#[cfg(feature = "paper-exact")]
pub mod crosscheck_engine;

pub use optimized_engine::PiCcsProof;
