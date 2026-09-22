//! Definition 11 (Structure) and Definition 12 (CCS).
//!
//! Type aliases over `neo-ccs` so the paper layer and the engine speak the
//! same wire format. No marshaling at this seam.

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim as NeoCcsClaim, CcsStructure, CcsWitness as NeoCcsWitness};
use neo_math::F;

/// Structure  s = ({M_j}_{j∈[t]}, f) — Definition 11.
pub type Structure = CcsStructure<F>;

/// CCS instance — Definition 12 public part: (commitment c, public input x, m_in).
pub type CcsClaim = NeoCcsClaim<Commitment, F>;

/// CCS witness — Definition 12 private part: (witness w, decomposition matrix Z).
pub type CcsWitness = NeoCcsWitness<F>;
