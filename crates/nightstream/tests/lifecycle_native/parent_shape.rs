use neo_math::D;
use nightstream_fprime::{
    PI_CCS_V1_2_MATRIX_COUNT, PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_2_ROUND_COUNT, PI_CCS_V1_2_SOURCE_COUNT,
};

use crate::folding::Params;

/// `CompressionKey::relation` passes `Relation::new` the sizes of the Lean
/// `parentShape_of` (`formal/compression-lean/CompressionLean/Final/RustSetup.lean`):
/// five public blocks, the bound `compression_bound` = 17 · T · (b - 1) = 3672
/// below `B = 2^16`, and 28 point variables, Lean's `cubeVariables`, which
/// bound the run rows. F′ has four matrices.
#[test]
fn parent_relation_arguments_are_the_lean_constants() {
    let params = Params::production();
    assert_eq!(PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS, 5 * D);
    assert_eq!(PI_CCS_V1_2_ROUND_COUNT, 28);
    assert_eq!(PI_CCS_V1_2_MATRIX_COUNT, 4);
    let bound = PI_CCS_V1_2_SOURCE_COUNT as u32 * params.inner().T() * (params.b() - 1);
    assert_eq!(bound, 3672);
    assert_eq!(params.big_b(), 1 << 16);
}
