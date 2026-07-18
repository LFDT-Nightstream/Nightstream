//! This module implements Spartan's traits using the following several different combinations

// public modules to be used as an commitment engine with Spartan
pub mod goldi;
pub mod keccak;
#[cfg(not(target_arch = "wasm32"))]
pub mod pasta;
pub mod pcs;
pub mod poseidon2;
#[cfg(not(target_arch = "wasm32"))]
pub mod pt256;
#[cfg(not(target_arch = "wasm32"))]
pub mod traits;

#[cfg(not(target_arch = "wasm32"))]
mod msm;

use crate::{
  provider::{keccak::Keccak256Transcript, pcs::merkle_mle_pc::HashMlePCS},
  traits::Engine,
};

#[cfg(not(target_arch = "wasm32"))]
use crate::provider::{
  pasta::{pallas, vesta},
  pcs::hyrax_pc::HyraxPCS,
  pt256::{p256, t256},
};

#[cfg(feature = "p3_backend")]
use crate::provider::pcs::merkle_mle_pc_p3::HashMlePcsP3;
#[cfg(feature = "p3_backend")]
use crate::provider::pcs::whir_pc::WhirPcsP3;
#[cfg(feature = "p3_backend")]
use crate::provider::poseidon2::Poseidon2Transcript;
use core::fmt::Debug;
use serde::{Deserialize, Serialize};

/// An implementation of the Spartan Engine trait with Pallas curve and Hyrax commitment scheme
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PallasHyraxEngine;

/// An implementation of the Spartan Engine trait with Vesta curve and Hyrax commitment scheme
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VestaHyraxEngine;

/// An implementation of the Spartan Engine trait with P256 curve and Hyrax commitment scheme
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct P256HyraxEngine;

/// An implementation of the Spartan Engine trait with T256 curve and Hyrax commitment scheme
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct T256HyraxEngine;

#[cfg(not(target_arch = "wasm32"))]
impl Engine for PallasHyraxEngine {
  type Base = pallas::Base;
  type Scalar = pallas::Scalar;
  type GE = pallas::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = HyraxPCS<Self>;
}

#[cfg(not(target_arch = "wasm32"))]
impl Engine for VestaHyraxEngine {
  type Base = vesta::Base;
  type Scalar = vesta::Scalar;
  type GE = vesta::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = HyraxPCS<Self>;
}

#[cfg(not(target_arch = "wasm32"))]
impl Engine for P256HyraxEngine {
  type Base = p256::Base;
  type Scalar = p256::Scalar;
  type GE = p256::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = HyraxPCS<Self>;
}

#[cfg(not(target_arch = "wasm32"))]
impl Engine for T256HyraxEngine {
  type Base = t256::Base;
  type Scalar = t256::Scalar;
  type GE = t256::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = HyraxPCS<Self>;
}

/// An implementation of the Spartan Engine trait with Pallas curve and Hash-MLE PCS (Keccak Merkle)
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PallasMerkleMleEngine;

/// An implementation of the Spartan Engine trait with Vesta curve and Hash-MLE PCS
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VestaMerkleMleEngine;

/// An implementation of the Spartan Engine trait with P256 curve and Hash-MLE PCS
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct P256MerkleMleEngine;

/// An implementation of the Spartan Engine trait with T256 curve and Hash-MLE PCS
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct T256MerkleMleEngine;

#[cfg(not(target_arch = "wasm32"))]
impl Engine for PallasMerkleMleEngine {
  type Base = pallas::Base;
  type Scalar = pallas::Scalar;
  type GE = pallas::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = HashMlePCS<Self>;
}

#[cfg(not(target_arch = "wasm32"))]
impl Engine for VestaMerkleMleEngine {
  type Base = vesta::Base;
  type Scalar = vesta::Scalar;
  type GE = vesta::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = HashMlePCS<Self>;
}

#[cfg(not(target_arch = "wasm32"))]
impl Engine for P256MerkleMleEngine {
  type Base = p256::Base;
  type Scalar = p256::Scalar;
  type GE = p256::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = HashMlePCS<Self>;
}

#[cfg(not(target_arch = "wasm32"))]
impl Engine for T256MerkleMleEngine {
  type Base = t256::Base;
  type Scalar = t256::Scalar;
  type GE = t256::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = HashMlePCS<Self>;
}

/// An implementation of the Spartan Engine trait with Goldilocks field and Hash-MLE PCS (Keccak)
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GoldilocksMerkleMleEngine;

impl Engine for GoldilocksMerkleMleEngine {
  type Base = crate::provider::goldi::F;
  type Scalar = crate::provider::goldi::F;
  type GE = crate::provider::goldi::UnitPoint;
  type TE = Keccak256Transcript<Self>;
  type PCS = HashMlePCS<Self>;
}

/// An implementation of the Spartan Engine trait with Goldilocks field and Hash-MLE PCS (p3/Poseidon2)
#[cfg(feature = "p3_backend")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GoldilocksP3MerkleMleEngine;

#[cfg(feature = "p3_backend")]
impl Engine for GoldilocksP3MerkleMleEngine {
  type Base = crate::provider::goldi::F;
  type Scalar = crate::provider::goldi::F;
  type GE = crate::provider::goldi::UnitPoint;
  type TE = Poseidon2Transcript<Self>;
  type PCS = HashMlePcsP3<Self>;
}

/// Toy Spartan over Goldilocks with a Poseidon2-backed WHIR polynomial commitment.
///
/// This engine is standalone: selecting it does not connect Toy Spartan to the
/// SuperNeo folding lifecycle.
#[cfg(feature = "p3_backend")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GoldilocksWhirEngine;

#[cfg(feature = "p3_backend")]
impl Engine for GoldilocksWhirEngine {
  type Base = crate::provider::goldi::F;
  type Scalar = crate::provider::goldi::F;
  type GE = crate::provider::goldi::UnitPoint;
  type TE = Poseidon2Transcript<Self>;
  type PCS = WhirPcsP3<Self>;
}

/*
/// An implementation of the Spartan Engine trait with Pallas curve and IPA PCS
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PallasIPAEngine;

/// An implementation of the Spartan Engine trait with Vesta curve and IPA PCS
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VestaIPAEngine;

/// An implementation of the Spartan Engine trait with P256 curve and IPA PCS
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct P256IPAEngine;

/// An implementation of the Spartan Engine trait with T256 curve and IPA PCS
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct T256IPAEngine;

impl Engine for PallasIPAEngine {
  type Base = pallas::Base;
  type Scalar = pallas::Scalar;
  type GE = pallas::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = IPAPCS<Self>;
}

impl Engine for VestaIPAEngine {
  type Base = vesta::Base;
  type Scalar = vesta::Scalar;
  type GE = vesta::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = IPAPCS<Self>;
}

impl Engine for P256IPAEngine {
  type Base = p256::Base;
  type Scalar = p256::Scalar;
  type GE = p256::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = IPAPCS<Self>;
}

impl Engine for T256IPAEngine {
  type Base = t256::Base;
  type Scalar = t256::Scalar;
  type GE = t256::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = IPAPCS<Self>;
}
*/
