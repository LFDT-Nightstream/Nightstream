#![allow(non_snake_case)]
#![allow(deprecated)]

#[path = "common/fixtures.rs"]
mod fixtures;

use fixtures::{
    build_twist_shout_2step_fixture, build_twist_shout_2step_fixture_bad_lookup, prove, verify, verify_and_finalize,
};
use neo_ajtai::Commitment as Cmt;
use neo_fold::finalize::{FinalizeReport, ObligationFinalizer};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::ShardObligations;
use neo_fold::PiCcsError;
use neo_math::{F, K};
use neo_memory::MemInit;
use neo_transcript::Transcript;
use p3_field::PrimeCharacteristicRing;

struct RequireValLane;

impl ObligationFinalizer<Cmt, F, K> for RequireValLane {
    type Error = PiCcsError;

    fn finalize(&mut self, obligations: &ShardObligations<Cmt, F, K>) -> Result<FinalizeReport, Self::Error> {
        if obligations.val.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "expected non-empty val-lane obligations for Twist".into(),
            ));
        }
        Ok(FinalizeReport {
            did_finalize_main: !obligations.main.is_empty(),
            did_finalize_val: !obligations.val.is_empty(),
        })
    }
}

#[test]
fn twist_shout_end_to_end_and_finalize() {
    let fx = build_twist_shout_2step_fixture(1);

    for mode in [FoldingMode::Optimized, FoldingMode::PaperExact] {
        let proof = prove(mode.clone(), &fx);
        let _ = verify(mode.clone(), &fx, &proof).expect("verify should succeed");

        let mut fin = RequireValLane;
        verify_and_finalize(mode, &fx, &proof, &mut fin).expect("verify_and_finalize should succeed");
    }
}

#[test]
fn mini_zkvm_lookup_table_is_semantically_required() {
    // Valid lookup trace.
    let fx_ok = build_twist_shout_2step_fixture(2);
    let proof_ok = prove(FoldingMode::Optimized, &fx_ok);
    let _ = verify(FoldingMode::Optimized, &fx_ok, &proof_ok).expect("valid fixture must verify");

    // Invalid lookup trace (step 1 uses wrong table value).
    let fx_bad = match std::panic::catch_unwind(|| build_twist_shout_2step_fixture_bad_lookup(2)) {
        // In debug builds, encoding performs a semantic check and is expected to reject
        // inconsistent lookup traces early (panic).
        Err(_) => return,
        Ok(fx_bad) => fx_bad,
    };

    let mut tr_bad = neo_transcript::Poseidon2Transcript::new(b"twist-shout/fixture");
    let res = neo_fold::shard::fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_bad,
        &fx_bad.params,
        &fx_bad.ccs,
        &fx_bad.steps_witness,
        &fx_bad.acc_init,
        &fx_bad.acc_wit_init,
        &fx_bad.l,
        fx_bad.mixers,
    );
    if let Ok(proof_bad) = res {
        assert!(
            verify(FoldingMode::Optimized, &fx_bad, &proof_bad).is_err(),
            "invalid lookup must not verify"
        );
    }
}

#[test]
fn redteam_splice_steps_from_different_proofs_must_fail() {
    let fx_a = build_twist_shout_2step_fixture(10);
    let fx_b = build_twist_shout_2step_fixture(11);

    let proof_a = prove(FoldingMode::Optimized, &fx_a);
    let proof_b = prove(FoldingMode::Optimized, &fx_b);

    let _ = verify(FoldingMode::Optimized, &fx_a, &proof_a).expect("fx_a should verify");
    let _ = verify(FoldingMode::Optimized, &fx_b, &proof_b).expect("fx_b should verify");

    let mut bad = proof_a.clone();
    bad.steps[1] = proof_b.steps[1].clone();

    assert!(
        verify(FoldingMode::Optimized, &fx_a, &bad).is_err(),
        "splicing step proofs across transcripts must fail"
    );
}

#[test]
fn redteam_drop_val_fold_must_fail() {
    let fx = build_twist_shout_2step_fixture(3);
    let mut proof = prove(FoldingMode::Optimized, &fx);

    proof.steps[0].val_fold = None;

    assert!(
        verify(FoldingMode::Optimized, &fx, &proof).is_err(),
        "missing val_fold must fail verification"
    );
}

#[test]
fn redteam_tamper_twist_init_in_public_input_must_fail() {
    let fx = build_twist_shout_2step_fixture(4);
    let proof = prove(FoldingMode::Optimized, &fx);
    let _ = verify(FoldingMode::Optimized, &fx, &proof).expect("baseline should verify");

    let mut fx_bad = fx.clone();
    let inst = &mut fx_bad.steps_instance[0].mem_insts[0];
    inst.init = MemInit::Sparse(vec![(0, F::ONE)]);

    assert!(
        verify(FoldingMode::Optimized, &fx_bad, &proof).is_err(),
        "tampering Twist init in public input must fail verification"
    );
}
