//! Normal selected PiCCS proving from an executed witness, without images or openings supplied to the prover.

use neo_ajtai::{nightstream_fprime_setup::commit_production_signed_units, Commitment};
use neo_ccs::{CcsClaim, CcsWitness, CeClaim, Mat};
use neo_fold_clean::{paper::params::Params, Poseidon2HashChainV1Package};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::optimized_engine::optimized_verify_with_trace;
use neo_transcript::Poseidon2Transcript;
use nightstream_fprime::{
    load_poseidon2_hash_chain_v1_package, PI_CCS_V1_1_MATRIX_COUNT, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS,
    PI_CCS_V1_1_ROUND_COUNT, PI_DEC_V1_1_CHILD_COUNT,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, time::Instant};

#[derive(Deserialize)]
struct Fixture(u64, [u64; 4], Vec<u64>, Vec<u64>, Value);

fn artifact(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts")
        .join(name)
}

fn words(values: &[K]) -> Vec<[u64; 2]> {
    values
        .iter()
        .map(|value| value.as_coeffs().map(|limb| limb.as_canonical_u64()))
        .collect()
}

#[test]
#[ignore = "Full selected base PiCCS call; witness, commitment and both cache passes are inside the 300-second test cap."]
fn selected_pi_ccs_prover_from_actual_base_sources() {
    let started = Instant::now();
    let expected: Value = serde_json::from_str(include_str!("fixtures/stage1_base_pi_ccs.json")).unwrap();
    let bytes = fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).unwrap();
    let producer = load_poseidon2_hash_chain_v1_package(&bytes).unwrap();
    let fixture_bytes = fs::read(artifact("nightstream-fprime-stage1-base-step-fixture-v1.json")).unwrap();
    // Test-data provenance only; this SHA does not enter a protocol binding.
    assert_eq!(
        json!(format!("{:x}", Sha256::digest(&fixture_bytes))),
        expected["base_fixture_sha256"]
    );
    let Fixture(schema, context, private, public, fixture_result) = serde_json::from_slice(&fixture_bytes).unwrap();
    drop(fixture_result);
    assert_eq!(schema, 1);
    assert_eq!(
        context,
        producer
            .production_verifier_binding()
            .unwrap()
            .verifier_context()
            .digest()
    );
    let physical = producer.execute_witness(&private, &public).unwrap();
    let logical = producer.execute_logical_assignment(&physical).unwrap();
    let logical_width = logical.len();
    let blocks = logical_width.div_ceil(D);
    let mut carrier = logical.balanced_values().to_vec();
    carrier.resize(blocks * D, 0);
    let commitment = commit_production_signed_units(&carrier).unwrap();
    let mut positive = vec![0u64; blocks];
    let mut negative = vec![0u64; blocks];
    for (index, &value) in carrier.iter().enumerate() {
        let mask = 1u64 << (index % D);
        match value {
            0 => {}
            1 => positive[index / D] |= mask,
            -1 => negative[index / D] |= mask,
            _ => panic!("non-unit actual source"),
        }
    }
    let public_width = PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS;
    let x = (0..public_width)
        .map(|index| F::from_u64(logical.value(index).unwrap()))
        .collect::<Vec<_>>();
    let mut prior_digest = [0u8; 32];
    for lane in 0..4 {
        let word = (0..64).fold(0u64, |word, bit| {
            let digit = x[1 + lane * 64 + bit].as_canonical_u64();
            assert!(digit <= 1);
            word | (digit << bit)
        });
        prior_digest[lane * 8..lane * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    let fresh = CcsClaim {
        c: commitment,
        x,
        m_in: public_width,
        adv: None,
    };
    let witness = CcsWitness {
        w: Vec::new(),
        Z: Mat::<F>::compact_signed_unit_from_column_masks(D, blocks, &positive, &negative).unwrap(),
    };
    drop((
        producer,
        physical,
        logical,
        carrier,
        positive,
        negative,
        private,
        public,
        fixture_bytes,
    ));
    println!("actual_sources_and_commitment_elapsed={:?}", started.elapsed());

    let package = Poseidon2HashChainV1Package::load(&bytes).unwrap();
    drop(bytes);
    assert_eq!(
        json!(package.structural_identifier()),
        expected["structural_identifier"]
    );
    assert_eq!(json!(package.package_identity()), expected["package_identity"]);
    assert_eq!(package.structure().m, logical_width);
    let params = Params::for_ccs_shape(
        package.structure().n,
        package.structure().m,
        package.structure().t(),
        package.structure().max_degree(),
    )
    .unwrap();
    let initial = CeClaim {
        c: Commitment::zeros(D, params.inner().kappa as usize),
        X: Mat::zero(D, public_width / D, F::ZERO),
        r: vec![K::ZERO; PI_CCS_V1_1_ROUND_COUNT],
        eval_k: vec![K::ZERO; D.next_power_of_two()],
        eval_a: vec![vec![K::ZERO; D.next_power_of_two()]; PI_CCS_V1_1_MATRIX_COUNT],
        m_in: public_width,
        fold_digest: prior_digest,
        adv: None,
    };
    let running = vec![initial; PI_DEC_V1_1_CHILD_COUNT];
    let running_witnesses = vec![Mat::virtual_constant(D, blocks, F::ZERO); PI_DEC_V1_1_CHILD_COUNT];
    let proof = package
        .prove_pi_ccs(
            std::slice::from_ref(&fresh),
            std::slice::from_ref(&witness),
            &running,
            &running_witnesses,
        )
        .unwrap();
    println!("actual_selected_prover_elapsed={:?}", started.elapsed());
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed([F::ZERO; 8], 0);
    let (valid, trace) = optimized_verify_with_trace(
        &mut transcript,
        params.inner(),
        package.structure(),
        std::slice::from_ref(&fresh),
        &running,
        &proof.outputs,
        &proof.sumcheck,
    )
    .unwrap();
    assert!(valid);
    assert_eq!(
        json!(proof
            .sumcheck
            .sumcheck_rounds
            .iter()
            .map(|round| words(round))
            .collect::<Vec<_>>()),
        expected["rounds"]
    );
    assert!(proof
        .outputs
        .iter()
        .all(|output| output.r == trace.round_challenges));
    let terminal = trace.terminal_components;
    let observed = json!([
        u64::from(valid),
        words(&trace.alpha),
        words(&[trace.gamma])[0],
        trace
            .pre_sumcheck_state
            .map(|value| value.as_canonical_u64()),
        words(&trace.round_challenges),
        trace
            .round_states
            .iter()
            .map(|state| state.map(|value| value.as_canonical_u64()))
            .collect::<Vec<_>>(),
        words(&proof.outputs[0].r),
        words(&[trace.initial_claim])[0],
        words(&trace.round_claims),
        words(&[
            terminal.eval_k,
            terminal.eval_a,
            terminal.ccs,
            terminal.norm,
            terminal.terminal,
            trace.terminal_claim
        ]),
        proof
            .outputs
            .iter()
            .map(|output| output
                .c
                .data
                .iter()
                .map(|value| value.as_canonical_u64())
                .collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        proof
            .outputs
            .iter()
            .map(|output| (0..public_width)
                .map(|column| output.X[(column % D, column / D)].as_canonical_u64())
                .collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        proof
            .outputs
            .iter()
            .map(|output| words(&output.eval_k[..D]))
            .collect::<Vec<_>>(),
        proof
            .outputs
            .iter()
            .map(|output| output
                .eval_a
                .iter()
                .map(|matrix| words(&matrix[..D]))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        trace.outgoing_state.map(|value| value.as_canonical_u64())
    ]);
    assert_eq!(observed, expected["phase"]);
    assert!(proof
        .outputs
        .iter()
        .all(|output| output.eval_a[13].iter().all(|value| *value == K::ZERO)));
    println!("full_selected_pi_ccs_elapsed={:?}", started.elapsed());
}
