//! The shape constants of the profile Nightstream P64, k_rho = 16, as the
//! Lean-emitted packages lay them out. Each value that the profile fixes comes
//! from `neo-params`; the other files of this crate read these, so a value has
//! one definition.

use neo_params::nightstream_p64_k16 as profile;

pub(crate) const DECOMPOSITION_BASE: u64 = profile::B_BASE as u64;
pub(crate) const DECOMPOSITION_K_RHO: u32 = profile::K_RHO;
pub(crate) const DECOMPOSITION_NORM_BOUND: u64 = profile::B;

/// Lean `ringDegree`.
pub(crate) const RING_DEGREE: usize = profile::D;
/// Lean `QuotientProduct.node`: a ring product is checked at this many points.
pub(crate) const RING_NODES: usize = 2 * RING_DEGREE;
/// Lean `ProductionAssignment.publicWidth`: `ringDegree * publicRingColumns`.
pub const PUBLIC_WORDS: usize = 5 * RING_DEGREE;
/// The words of one Ajtai commitment: κ ring elements.
pub(crate) const COMMITMENT_WORDS: usize = profile::KAPPA as usize * RING_DEGREE;

pub(crate) const PI_CCS_V1_2_MATRIX_COUNT: usize = 4;
/// The K values of one evaluation family: `Eval_K`, then `Eval_A` for each matrix.
pub(crate) const EVALUATIONS: usize = (PI_CCS_V1_2_MATRIX_COUNT + 1) * RING_DEGREE;
pub const PI_CCS_V1_2_ROUND_COUNT: usize = 28;
pub(crate) const PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT: usize = 9;

/// A packed Poseidon2 block (Lean `Gadgets/Poseidon2/Permutation`): its
/// locals, the offset of its 16 output locals, and its S-box rows.
pub(crate) const BLOCK_LOCALS: usize = 1096;
pub(crate) const BLOCK_OUTPUTS: usize = 1080;
pub(crate) const SBOX_ROWS: usize = 150;
