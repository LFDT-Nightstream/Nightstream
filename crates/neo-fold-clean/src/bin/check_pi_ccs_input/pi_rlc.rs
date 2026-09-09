//! Compare the actual PiCCS handoff and all indexed PiRLC results.

use neo_fold_clean::engine::optimized;
use neo_fold_clean::engine::transcript::{Poseidon2TranscriptSnapshot, Transcript};
use neo_fold_clean::paper::{pi_rlc, relations::ajtai_rlc_mixer};
use neo_reductions::api;

use super::*;

fn evaluation_words(values: &[K]) -> Vec<[u64; 2]> {
    assert_eq!(values.len(), D.next_power_of_two());
    assert!(values[D..].iter().all(|value| *value == K::ZERO));
    extensions(&values[..D])
}

fn partial_value(claim: &RunningClaim) -> Value {
    assert_eq!((claim.c.d, claim.c.kappa, claim.m_in), (D, 22, PUBLIC));
    assert_eq!(claim.c.data.len(), COMMITMENT);
    assert_eq!(claim.r.len(), ROUNDS);
    assert_eq!(claim.eval_a.len(), MATRICES);
    json!([
        field_words(&claim.c.data),
        public_words(&claim.X),
        evaluation_words(&claim.eval_k),
        claim
            .eval_a
            .iter()
            .map(|family| evaluation_words(family))
            .collect::<Vec<_>>()
    ])
}

fn nonzero(claim: &RunningClaim) -> [bool; 3] {
    [
        claim.c.data.iter().any(|word| *word != F::ZERO),
        public_words(&claim.X).iter().any(|word| *word != 0),
        claim
            .eval_k
            .iter()
            .chain(claim.eval_a.iter().flatten())
            .any(|word| *word != K::ZERO),
    ]
}

fn claimed_parent(result: &[Value], digest: [u8; 32]) -> RunningClaim {
    let commitment: Vec<u64> = serde_json::from_value(result[3].clone()).expect("parent commitment");
    let public: Vec<u64> = serde_json::from_value(result[4].clone()).expect("parent public input");
    let point: Vec<[u64; 2]> = serde_json::from_value(result[5].clone()).expect("parent point");
    let eval_k: Vec<[u64; 2]> = serde_json::from_value(result[6].clone()).expect("parent pad evaluation");
    let eval_a: Vec<Vec<[u64; 2]>> = serde_json::from_value(result[7].clone()).expect("parent matrix evaluations");
    assert_eq!(commitment.len(), COMMITMENT);
    assert_eq!(point.len(), ROUNDS);
    assert_eq!(eval_a.len(), MATRICES);
    RunningClaim {
        c: Commitment {
            d: D,
            kappa: 22,
            data: commitment.into_iter().map(F::from_u64).collect(),
        },
        X: public_matrix(&public),
        r: point.into_iter().map(extension).collect(),
        eval_k: padded_family(&eval_k),
        eval_a: eval_a.iter().map(|family| padded_family(family)).collect(),
        m_in: PUBLIC,
        fold_digest: digest,
        adv: None,
    }
}

pub(super) fn check(
    params: &Params,
    structure: &CcsStructure<F>,
    inputs: &[RunningClaim],
    initial: &Poseidon2Transcript,
    expected_identity: [u64; 4],
    encoded_input: &Value,
    encoded_result: &Value,
) -> RunningClaim {
    assert_eq!(inputs.len(), SOURCES);
    assert_eq!(initial.absorbed(), 0);
    let input_value = json!([
        field_words(&initial.state()),
        extensions(&inputs[0].r),
        inputs
            .iter()
            .map(|claim| field_words(&claim.c.data))
            .collect::<Vec<_>>(),
        inputs
            .iter()
            .map(|claim| public_words(&claim.X))
            .collect::<Vec<_>>(),
        inputs
            .iter()
            .map(|claim| evaluation_words(&claim.eval_k))
            .collect::<Vec<_>>(),
        inputs
            .iter()
            .map(|claim| claim
                .eval_a
                .iter()
                .map(|family| evaluation_words(family))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        expected_identity
    ]);
    assert!(input_value == *encoded_input, "exact PiCCS-to-PiRLC input handoff");
    let result = encoded_result.as_array().expect("complete PiRLC result");
    assert_eq!(result.len(), 11);
    assert_eq!(result[0], json!(1));
    let mut transcript = initial.clone();
    let rhos = optimized::sample_rho_n(&mut transcript, params, SOURCES).expect("optimized PiRLC sampler");
    let challenges = rhos
        .iter()
        .map(|rho| {
            (0..D)
                .map(|row| rho.as_mat()[(row, 0)].as_canonical_u64())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let membership = challenges
        .iter()
        .map(|rho| {
            u64::from(
                rho.len() == D
                    && rho
                        .iter()
                        .all(|word| matches!(*word, 0 | 1 | 2) || *word == MODULUS - 1 || *word == MODULUS - 2),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(membership, vec![1; SOURCES]);
    let partials = (1..=SOURCES)
        .map(|count| {
            api::rlc_public(
                structure,
                params.inner(),
                &rhos[..count],
                &inputs[..count],
                ajtai_rlc_mixer,
                D.next_power_of_two().trailing_zeros() as usize,
            )
            .expect("optimized indexed PiRLC combination")
        })
        .collect::<Vec<_>>();
    let expected = claimed_parent(result, inputs[0].fold_digest);
    let mut verifier = Transcript::session();
    verifier.restore_snapshot(Poseidon2TranscriptSnapshot::from_state_and_absorbed(
        initial.state(),
        initial.absorbed(),
    ));
    let parent = pi_rlc::verify(
        &mut verifier,
        params,
        structure,
        ajtai_rlc_mixer,
        inputs,
        &pi_rlc::Proof { combined: expected },
    )
    .expect("native PiRLC wrapper accepts the supplied parent");
    assert_eq!(
        Some(&parent),
        partials.last(),
        "native parent equals final indexed combination"
    );
    assert_eq!(
        verifier.snapshot().state(),
        transcript.state(),
        "full PiRLC transcript endpoint"
    );
    assert_eq!(verifier.snapshot().absorbed(), transcript.absorbed());
    assert_eq!(transcript.absorbed(), 0);
    let assurance = nonzero(&parent);
    let computed = json!([
        1,
        challenges,
        membership,
        field_words(&parent.c.data),
        public_words(&parent.X),
        extensions(&parent.r),
        evaluation_words(&parent.eval_k),
        parent
            .eval_a
            .iter()
            .map(|family| evaluation_words(family))
            .collect::<Vec<_>>(),
        partials.iter().map(partial_value).collect::<Vec<_>>(),
        field_words(&transcript.state()),
        [
            u64::from(
                inputs
                    .iter()
                    .all(|claim| nonzero(claim).into_iter().all(|value| value))
            ),
            u64::from(assurance[0]),
            u64::from(assurance[1]),
            u64::from(assurance[2])
        ]
    ]);
    for (index, (actual, expected)) in computed.as_array().unwrap().iter().zip(result).enumerate() {
        assert!(actual == expected, "complete Lean/optimized PiRLC result field {index}");
    }
    println!("pi_ccs_to_pi_rlc=passed complete_fields=11 indexed_sources=17 native_wrapper=true");
    parent
}
