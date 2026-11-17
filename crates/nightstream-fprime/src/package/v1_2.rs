use neo_params::nightstream_p64_k14 as profile;

pub(crate) const DECOMPOSITION_BASE: u64 = profile::B_BASE as u64;
pub(crate) const DECOMPOSITION_K_RHO: u32 = profile::K_RHO;
pub(crate) const DECOMPOSITION_NORM_BOUND: u64 = profile::B;

/// Lean `ringDegree`.
pub(crate) const RING_DEGREE: usize = profile::D;
/// Lean `QuotientProduct.node`.
pub(crate) const RING_NODES: usize = 2 * RING_DEGREE;
const PUBLIC_RING_COLUMNS: usize = 5;
/// Lean `ProductionAssignment.publicWidth`.
pub const PUBLIC_WORDS: usize = PUBLIC_RING_COLUMNS * RING_DEGREE;
pub const COMMITMENT_WORDS: usize = profile::KAPPA as usize * RING_DEGREE;

pub const PI_CCS_V1_2_MATRIX_COUNT: usize = 4;
pub const EVALUATIONS: usize = (PI_CCS_V1_2_MATRIX_COUNT + 1) * RING_DEGREE;
pub const PI_CCS_V1_2_ROUND_COUNT: usize = 28;
pub const PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT: usize = 9;

// Lean `Gadgets/Poseidon2/Permutation`.
pub const BLOCK_LOCALS: usize = 1096;
pub(crate) const BLOCK_OUTPUTS: usize = 1080;
pub(crate) const SBOX_ROWS: usize = 150;
