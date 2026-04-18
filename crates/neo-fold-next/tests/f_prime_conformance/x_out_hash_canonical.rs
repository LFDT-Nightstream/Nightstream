//! HyperNova Construction 2 §6.3 step 5:
//!
//!   x_{i+1} = hash( vk_fs, i+1, z_0, z_{i+1}, U_{i+1}, pc_{i+1} )
//!
//! and nothing else. This test locks the canonical 6-input hash shape —
//! rebuilding the state image from those inputs must reproduce `x_out` bit-
//! exactly. Any future change that threads an additional digest into `x_out`
//! (chain digests, terminal-handle digests, etc.) will break this test.

use neo_fold_next::rv64im::audit::evaluate_rv64im_main_recursion_f_prime_advice;
use neo_fold_next::rv64im::Rv64imMainRecursionConstruction2StateImage;

use super::support::single_step_advices;

#[test]
fn f_prime_x_out_depends_only_on_hn_paper_canonical_inputs() {
    let advices = single_step_advices();
    for (step, advice) in advices.iter().enumerate() {
        let step_image = evaluate_rv64im_main_recursion_f_prime_advice(advice)
            .unwrap_or_else(|err| panic!("step {step}: evaluate advice: {err}"));

        let rebuilt = Rv64imMainRecursionConstruction2StateImage::from_parts(
            advice.verifier_key_fs().clone(),
            step_image.chunk_count(),
            *advice.z_0(),
            *step_image.z_next(),
            step_image.pc_next(),
            step_image.folded_accumulator_digest(),
        )
        .encoded_public_input();

        assert_eq!(
            rebuilt,
            *step_image.x_out(),
            "step {step}: HN Construction-2 §6.3 step 5 requires \
             x_{{i+1}} = hash(vk_fs, i+1, z_0, z_{{i+1}}, U_{{i+1}}, pc_{{i+1}}); \
             a mismatch here means x_out absorbs more than the canonical 6 inputs"
        );
    }
}
