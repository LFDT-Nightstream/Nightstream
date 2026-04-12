use bellpepper_core::{test_cs::TestConstraintSystem, ConstraintSystem};
use neo_fold_next::rv64im::main_relation_circuit::k_field::{alloc_constant_k, enforce_k_eq, KNum};
use neo_fold_next::rv64im::main_relation_circuit::sumcheck_replay::verify_sumcheck_rounds;
use neo_fold_next::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use neo_math::F as GoldilocksF;
use neo_math::{from_complex, K as NeoK};
use neo_reductions::sumcheck::{
    poly_eval_k, round_coeff_fields, verify_sumcheck_rounds_poseidon_v3 as verify_native_sumcheck_rounds,
    SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG,
};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use spartan2::provider::goldi::F as SpartanF;

fn k(re: u64, im: u64) -> NeoK {
    from_complex(GoldilocksF::from_u64(re), GoldilocksF::from_u64(im))
}

#[test]
fn rv64im_main_relation_sumcheck_replay_matches_native_transcript_path() {
    let delta = SpartanF::from_canonical_u64(7);
    let prefix = b"neo.fold.next/test/sumcheck_replay";

    let round0 = vec![k(3, 0), k(5, 1), k(7, 0)];
    let initial_sum = poly_eval_k(&round0, NeoK::ZERO) + poly_eval_k(&round0, NeoK::ONE);

    let mut round0_transcript = Poseidon2Transcript::new(prefix);
    round0_transcript.append_message(b"prefix", b"ok");
    round0_transcript.append_fields_raw(&[GoldilocksF::from_u64(SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);
    round0_transcript.append_fields_raw(&round_coeff_fields(&round0));
    let c = round0_transcript.challenge_fields_raw(2);
    let first_challenge = from_complex(c[0], c[1]);
    let running_after_round0 = poly_eval_k(&round0, first_challenge);
    let round1 = vec![NeoK::ZERO, running_after_round0];
    let rounds = vec![round0.clone(), round1.clone()];

    let mut native = Poseidon2Transcript::new(prefix);
    native.append_message(b"prefix", b"ok");
    native.append_fields_raw(&[GoldilocksF::from_u64(SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);
    let (native_challenges, native_final, ok) = verify_native_sumcheck_rounds(&mut native, 2, initial_sum, &rounds);
    assert!(ok, "native sumcheck replay must accept");

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let mut transcript =
        Poseidon2TranscriptCircuit::new(cs.namespace(|| "transcript"), prefix).expect("circuit transcript");
    transcript
        .append_message(cs.namespace(|| "prefix"), b"prefix", b"ok")
        .expect("prefix");
    let initial_sum_var = alloc_constant_k(&mut cs, KNum::from_neo_k(initial_sum), "initial_sum").expect("initial sum");
    let round_vars = rounds
        .iter()
        .enumerate()
        .map(|(round_idx, round)| {
            round
                .iter()
                .enumerate()
                .map(|(coeff_idx, coeff)| {
                    alloc_constant_k(
                        &mut cs,
                        KNum::from_neo_k(*coeff),
                        &format!("round_{round_idx}_coeff_{coeff_idx}"),
                    )
                    .expect("coeff")
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (challenge_vars, final_sum_var) = verify_sumcheck_rounds(
        &mut cs,
        &mut transcript,
        2,
        &initial_sum_var,
        &round_vars,
        &rounds,
        &native_challenges,
        delta,
        "sumcheck",
    )
    .expect("circuit sumcheck replay");

    for (idx, value) in native_challenges.iter().enumerate() {
        let expected = alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("expected_challenge_{idx}"))
            .expect("expected challenge");
        enforce_k_eq(&mut cs, &challenge_vars[idx], &expected, &format!("challenge_eq_{idx}"));
    }
    let expected_final =
        alloc_constant_k(&mut cs, KNum::from_neo_k(native_final), "expected_final").expect("expected final");
    enforce_k_eq(&mut cs, &final_sum_var, &expected_final, "final_eq");

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}
