//! SuperNeo §7.1 + Ajtai §3.2: a fresh instance's witness image must satisfy
//! `||w||_∞ < b`; with b = 2 (goldilocks_127) this collapses to the binary
//! constraint `w_j ∈ {0, 1}`. Likewise `x` is the bit-image of a Poseidon2
//! digest and must be binary.
//!
//! This test locks the low-norm invariant at the public image surface:
//! every entry of the low-norm witness image must be `F::ZERO` or `F::ONE`,
//! and every entry of `x_i.field_image()` must be `F::ZERO` or `F::ONE`.

use neo_fold_next::rv64im::{
    build_rv64im_main_recursion_construction2_default_fresh_instance,
    build_rv64im_main_recursion_construction2_f_prime_low_norm_witness_image,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::support::{default_full_width_from_advice, single_step_advices};

#[test]
fn f_prime_low_norm_witness_and_x_are_binary() {
    let advices = single_step_advices();
    let u_perp = build_rv64im_main_recursion_construction2_default_fresh_instance(
        advices[0].verifier_key_fs(),
        default_full_width_from_advice(&advices[0]),
    )
    .expect("build canonical u_perp");

    for (step, advice) in advices.iter().enumerate() {
        assert!(
            advice.x_i().is_binary_low_norm(),
            "step {step}: x_i must be binary under b = 2"
        );
        for (field_index, value) in advice.x_i().field_image().into_iter().enumerate() {
            assert!(
                value == F::ZERO || value == F::ONE,
                "step {step}: x_i.field_image()[{field_index}] must be 0 or 1 (is {:?})",
                value
            );
        }

        let low_norm = build_rv64im_main_recursion_construction2_f_prime_low_norm_witness_image(advice, &u_perp)
            .expect("build low-norm witness image");
        for (field_index, value) in low_norm.binary_values().iter().enumerate() {
            assert!(
                *value == F::ZERO || *value == F::ONE,
                "step {step}: low-norm witness bit {field_index} must be 0 or 1 (is {:?})",
                value
            );
        }
    }
}
