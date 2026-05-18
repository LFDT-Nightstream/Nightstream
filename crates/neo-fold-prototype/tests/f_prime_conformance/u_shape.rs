//! Native RV32IM F' threads only the Construction-2 `x_i` image.
//!
//! The authoritative `u_i.C` commitment is derived by the terminal committed
//! R2 proof from the packed SuperNeo witness. Native recursive steps therefore
//! carry an x-only placeholder, and tampering witness cargo outside `x_i` must
//! not make that placeholder look authoritative.

use neo_fold_prototype::rv32im::audit::{
    audit_build_rv32im_main_recursion_construction2_fresh_instance_with_explicit_x_i,
    rv32im_main_recursion_advice_tamper_ccs_replay_first_round_coeff,
};
use neo_fold_prototype::rv32im::{
    build_rv32im_main_recursion_construction2_default_fresh_instance,
    build_rv32im_main_recursion_construction2_fresh_instance_with_input,
};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use super::support::{default_full_width_from_advice, single_step_advices};

#[test]
fn f_prime_native_fresh_instance_keeps_u_c_placeholder() {
    let advices = single_step_advices();
    let u_perp = build_rv32im_main_recursion_construction2_default_fresh_instance(
        advices[0].verifier_key_fs(),
        default_full_width_from_advice(&advices[0]),
    )
    .expect("build canonical u_perp");

    let baseline = build_rv32im_main_recursion_construction2_fresh_instance_with_input(&advices[0], &u_perp)
        .expect("build baseline fresh instance");
    let mut tampered_advice = advices[0].clone();
    rv32im_main_recursion_advice_tamper_ccs_replay_first_round_coeff(&mut tampered_advice);
    let tampered = audit_build_rv32im_main_recursion_construction2_fresh_instance_with_explicit_x_i(
        &tampered_advice,
        &u_perp,
        baseline.x_i().clone(),
    )
    .expect("build tampered fresh instance with fixed x_i");

    assert_eq!(
        baseline.x_i(),
        tampered.x_i(),
        "tampering carried Π_CCS replay cargo in π_fold must not flip x; the u-shape binding failure we \
         are probing lives in the w half of [x || w]"
    );
    assert_eq!(
        baseline.commitment(),
        tampered.commitment(),
        "native RV32IM F' must keep u_i.C as a non-authoritative x-only placeholder; terminal R2 owns the \
         committed witness binding"
    );
    let commitment = baseline.commitment().commitment();
    assert_eq!(commitment.d, D);
    assert_eq!(commitment.kappa, 1);
    assert!(commitment.data.iter().all(|value| *value == F::ZERO));
}
