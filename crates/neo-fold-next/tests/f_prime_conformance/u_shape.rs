//! SuperNeo §7.1 u-shape: a fresh Π_SuperNeo instance is `u = (c, x)` with
//! `c = L([x, w])` — the Ajtai commitment binds the full padded vector
//! `[x || w]`, not just the witness. Consequently any change in either the
//! x-image or the low-norm witness must change `c`.
//!
//! This test locks that binding at the public Construction-2 fresh-instance
//! API: tampering the Pi_DEC-child cargo carried inside π_fold flips a
//! position in the low-norm witness image, which must propagate to `c`.

use neo_fold_next::rv64im::audit::rv64im_main_recursion_advice_tamper_dec_child_commitment_first_word;
use neo_fold_next::rv64im::{
    build_rv64im_main_recursion_construction2_default_fresh_instance,
    build_rv64im_main_recursion_construction2_fresh_instance_with_input,
};

use super::support::{default_full_width_from_advice, single_step_advices};

#[test]
fn f_prime_fresh_instance_commitment_binds_low_norm_witness() {
    let advices = single_step_advices();
    let u_perp = build_rv64im_main_recursion_construction2_default_fresh_instance(
        advices[0].verifier_key_fs(),
        default_full_width_from_advice(&advices[0]),
    )
    .expect("build canonical u_perp");

    let baseline = build_rv64im_main_recursion_construction2_fresh_instance_with_input(&advices[0], &u_perp)
        .expect("build baseline fresh instance");
    let mut tampered_advice = advices[0].clone();
    rv64im_main_recursion_advice_tamper_dec_child_commitment_first_word(&mut tampered_advice, 0);
    let tampered = build_rv64im_main_recursion_construction2_fresh_instance_with_input(&tampered_advice, &u_perp)
        .expect("build tampered fresh instance");

    assert_eq!(
        baseline.x_i(),
        tampered.x_i(),
        "tampering Pi_DEC-child cargo in π_fold must not flip x; the u-shape binding failure we \
         are probing lives in the w half of [x || w]"
    );
    assert_ne!(
        baseline.commitment(),
        tampered.commitment(),
        "SuperNeo §7.1 u-shape c = L([x, w]) must bind the full Pi_DEC-child cargo carried in the \
         low-norm witness; flipping that cargo must change c"
    );
}
