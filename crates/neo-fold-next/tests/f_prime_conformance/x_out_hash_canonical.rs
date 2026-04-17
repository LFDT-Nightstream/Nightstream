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
use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};

use super::support::single_step_advices;

fn digest32_as_fields(digest: [u8; 32]) -> [F; 4] {
    core::array::from_fn(|limb| {
        let start = limb * 8;
        F::new(u64::from_le_bytes(
            digest[start..start + 8].try_into().expect("digest limb"),
        ))
    })
}

fn legacy_v3_x_out(
    advice: &neo_fold_next::rv64im::Rv64imMainRecursionFPrimeAdvice,
    step_image: &neo_fold_next::rv64im::Rv64imMainRecursionFPrimeStepImage,
) -> neo_fold_next::rv64im::Rv64imEncodedPublicInput {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_f_prime_x_out");
    tr.append_message(b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/version", b"v3");
    tr.append_message(
        b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/vk_fs",
        &advice.verifier_key_fs().expected_digest(),
    );
    tr.append_u64s(
        b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/meta",
        &[step_image.chunk_count(), step_image.pc_next()],
    );
    tr.append_fields_iter(
        b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/z_0",
        4,
        digest32_as_fields(*advice.z_0()),
    );
    tr.append_fields_iter(
        b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/z_i",
        4,
        digest32_as_fields(*step_image.z_next()),
    );
    tr.append_fields_iter(
        b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/folded_accumulator_digest",
        4,
        digest32_as_fields(step_image.folded_accumulator_digest()),
    );
    tr.append_fields_iter(
        b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/step_statement_chain_digest",
        4,
        digest32_as_fields(step_image.step_statement_chain_digest()),
    );
    tr.append_fields_iter(
        b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/bridge_handoff_chain_digest",
        4,
        digest32_as_fields(step_image.bridge_handoff_chain_digest()),
    );
    tr.append_fields_iter(
        b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/terminal_handle_digest",
        4,
        digest32_as_fields(step_image.terminal_handle_digest()),
    );
    neo_fold_next::rv64im::Rv64imEncodedPublicInput::from_digest_bytes(tr.digest32())
}

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

#[test]
fn f_prime_x_out_rejects_legacy_v3_digest_shell_preimage() {
    let advices = single_step_advices();
    let mut observed_nonzero_legacy_shell = false;

    for (step, advice) in advices.iter().enumerate() {
        let step_image = evaluate_rv64im_main_recursion_f_prime_advice(advice)
            .unwrap_or_else(|err| panic!("step {step}: evaluate advice: {err}"));
        if step_image.step_statement_chain_digest() != [0; 32]
            || step_image.bridge_handoff_chain_digest() != [0; 32]
            || step_image.terminal_handle_digest() != [0; 32]
        {
            observed_nonzero_legacy_shell = true;
        }

        assert_ne!(
            legacy_v3_x_out(advice, &step_image),
            *step_image.x_out(),
            "step {step}: the legacy v3 preimage with step/bridge/terminal digest shells must not reproduce the current v4 canonical x_out"
        );
    }

    assert!(
        observed_nonzero_legacy_shell,
        "expected at least one conformance step to carry non-zero legacy shell digests so the v3->v4 rejection regression is meaningful"
    );
}
