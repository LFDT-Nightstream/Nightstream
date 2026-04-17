//! HyperNova Construction 2 §6.2 / SuperNeo Def 12: the default fresh input
//! surface must expose an explicit default witness under the same full-width
//! commitment scheme as the inductive path.

use neo_fold_next::rv64im::{
    build_rv64im_main_recursion_construction2_default_fresh_instance,
    build_rv64im_main_recursion_construction2_default_full_width_from_ccs_shape,
    build_rv64im_main_recursion_construction2_default_low_norm_witness_image,
    build_rv64im_main_recursion_construction2_default_pair,
    build_rv64im_main_recursion_construction2_default_pair_for_full_width,
    build_rv64im_main_recursion_construction2_f_prime_ccs_shape,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::support::{default_full_width_from_advice, default_full_width_from_relations, single_step_advices};

#[test]
fn f_prime_u_perp_exposes_explicit_default_witness() {
    let advices = single_step_advices();
    let base_advice = &advices[0];
    let full_width = default_full_width_from_advice(base_advice);
    let relation_cover_width = default_full_width_from_relations();
    let ccs_shape = build_rv64im_main_recursion_construction2_f_prime_ccs_shape(core::slice::from_ref(base_advice))
        .expect("derive explicit native F' shape");
    let shape_width = build_rv64im_main_recursion_construction2_default_full_width_from_ccs_shape(&ccs_shape)
        .expect("derive explicit default width from native shape");
    let explicit_default_pair = build_rv64im_main_recursion_construction2_default_pair_for_full_width(
        base_advice.verifier_key_fs(),
        full_width,
    )
    .expect("build explicit default pair from vk_fs and width");
    let default_pair =
        build_rv64im_main_recursion_construction2_default_pair(base_advice.verifier_key_fs(), full_width)
            .expect("build explicit default pair");
    let default_fresh_pair = default_pair.u_perp().clone();
    let default_witness_pair = default_pair.w_perp().clone();
    let default_witness = build_rv64im_main_recursion_construction2_default_low_norm_witness_image(
        base_advice.verifier_key_fs(),
        full_width,
    )
    .expect("build explicit default witness image");

    assert!(
        !default_witness.binary_values().is_empty(),
        "the default witness image must be explicit and width-bound, not omitted"
    );
    assert!(
        default_witness
            .binary_values()
            .iter()
            .all(|value| *value == F::ZERO),
        "the canonical default witness W_perp must be the zero low-norm witness under the base-case width"
    );

    let default_fresh =
        build_rv64im_main_recursion_construction2_default_fresh_instance(base_advice.verifier_key_fs(), full_width)
            .expect("build default u_perp");
    assert_eq!(
        default_witness_pair, default_witness,
        "the public default-pair builder must expose the same canonical w_perp as the standalone witness builder"
    );
    assert_eq!(
        default_fresh_pair, default_fresh,
        "the public default-pair builder must expose the same canonical u_perp as the standalone fresh-instance builder"
    );
    assert_eq!(
        relation_cover_width, full_width,
        "the canonical default width must be derived from the relation-owned fixed native shape cover, not from a probe advice bootstrap"
    );
    assert_eq!(
        shape_width, full_width,
        "the canonical default width must be owned by the fixed native F' shape, not by ad hoc advice cargo"
    );
    assert_eq!(
        explicit_default_pair.u_perp(),
        &default_fresh_pair,
        "the explicit (vk_fs, full_width) default-pair constructor must agree with the advice-layout wrapper"
    );
    assert_eq!(
        explicit_default_pair.w_perp(),
        &default_witness_pair,
        "the explicit (vk_fs, full_width) default-pair constructor must agree with the advice-layout wrapper"
    );
    assert_eq!(
        default_fresh.x_i().bytes(),
        [0; 32],
        "the canonical default fresh input must expose x_perp = 0"
    );
    assert!(
        default_fresh
            .commitment()
            .commitment()
            .data
            .iter()
            .all(|value| *value == F::ZERO),
        "by Ajtai linearity the commitment to the explicit zero witness must be the canonical zero commitment"
    );
    assert_eq!(
        base_advice.construction2_input_fresh_instance(),
        Some(&default_fresh),
        "the authoritative first native F' advice must thread the witness-backed default u_perp, not a placeholder shell"
    );
}
