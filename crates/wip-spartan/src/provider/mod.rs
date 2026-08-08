//! Goldilocks, Poseidon2, and WHIR backend used by Nightstream.

pub mod goldi;
pub mod pcs;
pub mod poseidon2;

use serde::{Deserialize, Serialize};

use crate::traits::Engine;

use self::pcs::whir_pc::WhirPcsP3;
use self::poseidon2::Poseidon2Transcript;

/// WIP Spartan over Goldilocks with a Poseidon2 transcript and WHIR PCS.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GoldilocksWhirEngine;

impl Engine for GoldilocksWhirEngine {
  type Base = goldi::F;
  type Scalar = goldi::F;
  type GE = goldi::UnitPoint;
  type TE = Poseidon2Transcript<Self>;
  type PCS = WhirPcsP3<Self>;
}
