//! Run the full optimized prover on the selected honest fixture sources.
//! Preparation owns the exact matrix images and selected-key commitments;
//! ring/child evaluators own the complete openings. This action reuses those
//! checked files, computes every round, and checks the complete Lean result.

use std::{fs, path::Path, time::Instant};

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsWitness, CeClaim, Mat};
use neo_fold_clean::paper::params::Params;
use neo_math::{from_complex, KExtensions, D, F, K};
use neo_reductions::{
    engines::pi_ccs_joint_protocol::V1_1OutputOpening,
    optimized_engine::{optimized_prove_with_complete_oracle, optimized_verify_with_trace},
};
use neo_transcript::Poseidon2Transcript;
use nightstream_fprime::load_per_application_package;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde_json::{json, Value};

use super::{
    oracle::{MATRICES, ROUNDS},
    rounds,
};

const MODULUS: u64 = 0xffff_ffff_0000_0001;
const PUBLIC: usize = 270;
const RUNNING: usize = 16;
type Claim = CeClaim<Commitment, F, K>;

fn canonical(value: &Value) {
    match value {
        Value::Array(values) => values.iter().for_each(canonical),
        Value::Number(number) => assert!(
            number.as_u64().is_some_and(|word| word < MODULUS),
            "canonical field word"
        ),
        _ => panic!("numeric fixture arrays"),
    }
}

fn field(word: u64) -> F {
    assert!(word < MODULUS);
    F::from_u64(word)
}

fn extension(words: [u64; 2]) -> K {
    from_complex(field(words[0]), field(words[1]))
}

fn extensions(value: &Value) -> Vec<K> {
    serde_json::from_value::<Vec<[u64; 2]>>(value.clone())
        .expect("extension array")
        .into_iter()
        .map(extension)
        .collect()
}

fn words(values: &[K]) -> Vec<[u64; 2]> {
    values
        .iter()
        .map(|value| <[u64; 2]>::from(value.to_limbs_u64()))
        .collect()
}

fn fields(values: &[F]) -> Vec<u64> {
    values
        .iter()
        .map(|value| value.as_canonical_u64())
        .collect()
}

fn digest_bytes(words: [u64; 4]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for (lane, word) in words.into_iter().enumerate() {
        bytes[lane * 8..lane * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn commitment(value: &Value) -> Commitment {
    let words: Vec<u64> = serde_json::from_value(value.clone()).expect("commitment words");
    assert_eq!(words.len(), 22 * D);
    Commitment {
        d: D,
        kappa: 22,
        data: words.into_iter().map(field).collect(),
    }
}

fn public_matrix(value: &Value) -> Mat<F> {
    let words: Vec<u64> = serde_json::from_value(value.clone()).expect("public input words");
    assert_eq!(words.len(), PUBLIC);
    let mut result = Mat::zero(D, PUBLIC / D, F::ZERO);
    for (column, word) in words.into_iter().enumerate() {
        result[(column % D, column / D)] = field(word);
    }
    result
}

fn public_words(value: &Mat<F>) -> Vec<u64> {
    (0..PUBLIC)
        .map(|column| value[(column % D, column / D)].as_canonical_u64())
        .collect()
}

fn padded(value: &Value) -> Vec<K> {
    let mut values = extensions(value);
    assert_eq!(values.len(), D);
    values.resize(D.next_power_of_two(), K::ZERO);
    values
}

fn running_claims(value: &Value, prior_digest: [u64; 4]) -> Vec<Claim> {
    assert_eq!(value.as_array().expect("running statement").len(), 5);
    for field in 1..5 {
        assert_eq!(
            value[field]
                .as_array()
                .expect("ordered running family")
                .len(),
            RUNNING
        );
    }
    let point = extensions(&value[0]);
    assert_eq!(point.len(), ROUNDS);
    (0..RUNNING)
        .map(|source| {
            assert_eq!(value[4][source].as_array().expect("running matrices").len(), MATRICES);
            Claim {
                c: commitment(&value[1][source]),
                X: public_matrix(&value[2][source]),
                r: point.clone(),
                eval_k: padded(&value[3][source]),
                eval_a: (0..MATRICES)
                    .map(|matrix| padded(&value[4][source][matrix]))
                    .collect(),
                m_in: PUBLIC,
                fold_digest: digest_bytes(prior_digest),
                adv: None,
            }
        })
        .collect()
}

fn openings(input: &Value) -> Vec<V1_1OutputOpening> {
    assert_eq!(input[4].as_array().expect("output K families").len(), RUNNING + 1);
    assert_eq!(input[5].as_array().expect("output A families").len(), RUNNING + 1);
    (0..=RUNNING)
        .map(|source| {
            let eval_k = extensions(&input[4][source]);
            assert_eq!(eval_k.len(), D);
            assert_eq!(input[5][source].as_array().expect("output matrices").len(), MATRICES);
            let eval_a = (0..MATRICES)
                .map(|matrix| {
                    let family = extensions(&input[5][source][matrix]);
                    assert_eq!(family.len(), D);
                    family
                })
                .collect();
            V1_1OutputOpening { eval_k, eval_a }
        })
        .collect()
}

pub fn generate(
    candidate: &Path,
    expected_identity: [u64; 4],
    cache: &Path,
    lean_result: &Path,
    running_prefix: Option<&Path>,
    folded_children: Option<&Path>,
    output: &Path,
) {
    let started = Instant::now();
    assert!(!output.exists(), "use a fresh native prover output");
    assert_eq!(
        running_prefix.is_some(),
        folded_children.is_some(),
        "nonzero running sources are actual ordered children"
    );
    let phase: Value = serde_json::from_slice(&fs::read(lean_result).expect("checked honest Lean result"))
        .expect("numeric Lean result");
    canonical(&phase);
    assert_eq!(phase[0], 1);
    assert_eq!(phase[1].as_array().expect("full PiCCS input").len(), 7);
    assert_eq!(phase[1][0], 2);
    assert_eq!(phase[2], phase[1][6], "one exact running statement");
    assert_eq!(
        phase[5]
            .as_array()
            .expect("complete Lean phase result")
            .len(),
        15
    );
    assert_eq!(phase[5][0], 1, "accepted honest reference");

    // The expected structural identity is an independent verifier-owned input.
    // It is never taken from the candidate or the fixture's own metadata.
    let package = load_per_application_package(&fs::read(candidate).expect("selected package"), expected_identity)
        .expect("independently selected structural identity");
    let context = package
        .production_verifier_binding()
        .expect("selected public-seed setup")
        .verifier_context()
        .digest();
    let structure = package
        .ccs_structure_header()
        .expect("selected logical relation header");
    let prepared = rounds::prepare(cache, lean_result, running_prefix, folded_children);
    assert_eq!(prepared.identity, expected_identity);
    assert_eq!(prepared.context, context);
    assert_eq!(prepared.logical_width, package.logical_column_count());
    assert_eq!(prepared.row_count, package.row_count());
    assert_eq!(prepared.carrier_width, prepared.logical_width.div_ceil(D) * D);
    assert_eq!(prepared.polynomial.arity(), structure.f.arity());
    assert_eq!(prepared.polynomial.terms().len(), structure.f.terms().len());
    for (cached, selected) in prepared.polynomial.terms().iter().zip(structure.f.terms()) {
        assert_eq!(cached.coeff, selected.coeff);
        assert_eq!(cached.exps, selected.exps);
    }
    assert_eq!(prepared.public[0], 1);
    assert!(prepared.public[1..257].iter().all(|&word| word <= 1));
    assert!(prepared.public[257..].iter().all(|&word| word == 0));
    let prior_digest =
        std::array::from_fn(|lane| (0..64).fold(0, |word, bit| word | (prepared.public[1 + lane * 64 + bit] << bit)));
    assert!(prior_digest.iter().all(|&word| word < MODULUS));
    let fresh = CcsClaim {
        c: commitment(&phase[1][1]),
        x: prepared.public.iter().copied().map(field).collect(),
        m_in: PUBLIC,
        adv: None,
    };
    let running = running_claims(&phase[2], prior_digest);
    let (native_fresh, native_running) = prepared.oracle.native_witnesses(prepared.logical_width);
    assert_eq!(
        public_words(&native_fresh),
        prepared.public,
        "fresh native source projection"
    );
    for (source, (witness, claim)) in native_running.iter().zip(&running).enumerate() {
        assert_eq!(
            public_words(witness),
            public_words(&claim.X),
            "running source {source} projection"
        );
    }
    let witness = CcsWitness {
        w: Vec::new(),
        Z: native_fresh,
    };
    assert_eq!(witness.private_len(PUBLIC, structure.m), Some(structure.m - PUBLIC));
    let mut oracle = prepared
        .oracle
        .with_output_openings(extensions(&phase[5][4]), openings(&phase[1]));
    let params =
        Params::for_ccs_shape(structure.n, structure.m, MATRICES, structure.max_degree()).expect("selected parameters");
    assert_eq!(
        (params.inner().b, params.inner().k_rho, params.inner().kappa),
        (2, 16, 22)
    );
    println!("selected native sources ready: {:?}", started.elapsed());
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed([F::ZERO; 8], 0);
    let (outputs, proof, perf, trace) = optimized_prove_with_complete_oracle(
        &mut transcript,
        params.inner(),
        &structure,
        std::slice::from_ref(&fresh),
        std::slice::from_ref(&witness),
        &running,
        &native_running,
        &prepared.challenges,
        &mut oracle,
    )
    .expect("complete selected honest optimized prover");
    assert_eq!(trace.alpha, prepared.challenges.alpha);
    assert_eq!(trace.gamma, prepared.challenges.gamma);
    assert_eq!(trace.pre_sumcheck_state, prepared.state);
    assert_eq!(trace.initial_claim, prepared.initial);
    assert_eq!(
        json!(proof
            .sumcheck_rounds
            .iter()
            .map(|round| words(round))
            .collect::<Vec<_>>()),
        phase[1][3],
        "all computed round messages"
    );
    assert_eq!(json!(words(&trace.round_challenges)), phase[5][4]);
    assert_eq!(
        json!(trace
            .round_states
            .iter()
            .map(|state| fields(state))
            .collect::<Vec<_>>()),
        phase[5][5]
    );
    assert_eq!(json!(words(&trace.round_claims)), phase[5][8]);
    assert_eq!(
        json!(outputs
            .iter()
            .map(|claim| fields(&claim.c.data))
            .collect::<Vec<_>>()),
        phase[5][10]
    );
    assert_eq!(
        json!(outputs
            .iter()
            .map(|claim| public_words(&claim.X))
            .collect::<Vec<_>>()),
        phase[5][11]
    );
    assert_eq!(
        json!(outputs
            .iter()
            .map(|claim| words(&claim.eval_k[..D]))
            .collect::<Vec<_>>()),
        phase[5][12]
    );
    assert_eq!(
        json!(outputs
            .iter()
            .map(|claim| claim
                .eval_a
                .iter()
                .map(|family| words(&family[..D]))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>()),
        phase[5][13]
    );
    assert_eq!(json!(fields(&trace.outgoing_state)), phase[5][14]);
    let mut verifier_transcript = Poseidon2Transcript::from_state_and_absorbed([F::ZERO; 8], 0);
    let (accepted, verified) = optimized_verify_with_trace(
        &mut verifier_transcript,
        params.inner(),
        &structure,
        std::slice::from_ref(&fresh),
        &running,
        &outputs,
        &proof,
    )
    .expect("verify the actual native prover output");
    assert!(accepted);
    assert_eq!(trace.events, verified.events, "same full prover/verifier transcript");
    assert_eq!(transcript.state(), verifier_transcript.state());
    let terminal = verified.terminal_components;
    assert_eq!(
        json!(words(&[
            terminal.eval_k,
            terminal.eval_a,
            terminal.ccs,
            terminal.norm,
            terminal.terminal,
            verified.terminal_claim
        ])),
        phase[5][9]
    );

    let generated = json!([
        2,
        prepared.commitment,
        prepared.public,
        proof
            .sumcheck_rounds
            .iter()
            .map(|round| words(round))
            .collect::<Vec<_>>(),
        outputs
            .iter()
            .map(|claim| words(&claim.eval_k[..D]))
            .collect::<Vec<_>>(),
        outputs
            .iter()
            .map(|claim| claim
                .eval_a
                .iter()
                .map(|family| words(&family[..D]))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        phase[2]
    ]);
    assert_eq!(
        generated, phase[1],
        "complete native input equals the checked honest input"
    );
    let mut encoded = serde_json::to_vec(&generated).expect("native PiCCSInputCheck schema 2");
    encoded.push(b'\n');
    fs::write(output, encoded).expect("native prover result");
    println!(
        "complete_native_prover_sources={} rounds={} matrices={} prover_ms={:.3} elapsed={:?}",
        outputs.len(),
        proof.sumcheck_rounds.len(),
        MATRICES,
        perf.total_ms,
        started.elapsed()
    );
}
