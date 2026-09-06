//! Compare all PiCCSInputCheck schema-2 values with both Rust verifiers.
//! Inputs contain the complete running statement and caller-supplied proof.
//! Opening validity and assignment satisfaction are separate gates.

use std::{env, fs, path::Path, time::Instant};

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim, Mat, SparsePoly, Term};
use neo_fold_clean::paper::params::Params;
use neo_math::{from_complex, KExtensions, D, F, K};
use neo_reductions::{
    engines::{
        paper_exact_engine::paper_exact_verify_with_trace, pi_ccs_joint::ProtocolTrace,
        pi_ccs_joint_protocol::assemble_proof,
    },
    optimized_engine::optimized_verify_with_trace,
    PiCcsError,
};
use neo_transcript::Poseidon2Transcript;
use nightstream_fprime::{derive_pi_ccs_v1_1_transcript, load_per_application_package, PackageCcsRelation};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::Deserialize;
use serde_json::{json, Value};

#[path = "../../tests/nifs/pi_ccs_positive_mutations.rs"]
mod mutation_checks;

const MODULUS: u64 = 0xffff_ffff_0000_0001;
const ROUNDS: usize = 28;
const COEFFICIENTS: usize = 10;
const MATRICES: usize = 14;
const RUNNING: usize = 16;
const SOURCES: usize = 17;
const PUBLIC: usize = 270;
const COMMITMENT: usize = 1_188;

type RunningClaim = CeClaim<Commitment, F, K>;

#[derive(Deserialize)]
struct Input(
    u64,
    Vec<u64>,
    Vec<u64>,
    Vec<Vec<[u64; 2]>>,
    Vec<Vec<[u64; 2]>>,
    Vec<Vec<Vec<[u64; 2]>>>,
    RunningInput,
);

#[derive(Deserialize)]
struct RunningInput(
    Vec<[u64; 2]>,
    Vec<Vec<u64>>,
    Vec<Vec<u64>>,
    Vec<Vec<[u64; 2]>>,
    Vec<Vec<Vec<[u64; 2]>>>,
);

fn canonical_words(value: &Value) {
    match value {
        Value::Array(values) => values.iter().for_each(canonical_words),
        Value::Number(number) => {
            assert!(
                number.as_u64().is_some_and(|word| word < MODULUS),
                "canonical field word"
            );
        }
        _ => panic!("expected numeric arrays"),
    }
}

fn read_value(path: &Path) -> Value {
    let bytes = fs::read(path).expect("serialized phase file");
    let value: Value = serde_json::from_slice(&bytes).expect("numeric JSON");
    canonical_words(&value);
    let canonical = serde_json::to_vec(&value).expect("canonical numeric JSON");
    assert!(
        bytes == canonical || bytes.strip_suffix(b"\n") == Some(canonical.as_slice()),
        "canonical bytes with optional final newline"
    );
    value
}

fn extension(words: [u64; 2]) -> K {
    from_complex(F::from_u64(words[0]), F::from_u64(words[1]))
}

fn extension_words(value: K) -> [u64; 2] {
    value.to_limbs_u64().into()
}

fn field_words(values: &[F]) -> Vec<u64> {
    values
        .iter()
        .map(|value| value.as_canonical_u64())
        .collect()
}

fn extensions(values: &[K]) -> Vec<[u64; 2]> {
    values.iter().copied().map(extension_words).collect()
}

fn digest_bytes(words: [u64; 4]) -> [u8; 32] {
    let mut bytes = [0; 32];
    for (lane, word) in words.into_iter().enumerate() {
        bytes[lane * 8..lane * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn public_matrix(words: &[u64]) -> Mat<F> {
    assert_eq!(words.len(), PUBLIC);
    let mut matrix = Mat::zero(D, PUBLIC / D, F::ZERO);
    for (column, &word) in words.iter().enumerate() {
        matrix[(column % D, column / D)] = F::from_u64(word);
    }
    matrix
}

fn public_words(matrix: &Mat<F>) -> Vec<u64> {
    assert_eq!((matrix.rows(), matrix.cols()), (D, PUBLIC / D));
    (0..PUBLIC)
        .map(|column| matrix[(column % D, column / D)].as_canonical_u64())
        .collect()
}

fn padded_family(values: &[[u64; 2]]) -> Vec<K> {
    assert_eq!(values.len(), D);
    let mut values = values.iter().copied().map(extension).collect::<Vec<_>>();
    values.resize(D.next_power_of_two(), K::ZERO);
    values
}

fn relation(header: &PackageCcsRelation) -> CcsStructure<F> {
    assert_eq!(header.cube_variables(), ROUNDS);
    assert_eq!(header.matrix_sources().len(), MATRICES);
    assert_eq!(header.degree_bound(), COEFFICIENTS - 1);
    let polynomial = SparsePoly::new(
        MATRICES,
        header
            .terms()
            .iter()
            .map(|term| Term {
                coeff: F::from_u64(term.coefficient()),
                exps: term
                    .exponents()
                    .iter()
                    .map(|&exponent| u32::try_from(exponent).expect("polynomial exponent"))
                    .collect(),
            })
            .collect(),
    );
    CcsStructure::new_verifier_artifact_header(1 << ROUNDS, header.column_count(), MATRICES, polynomial)
        .expect("selected Lean verifier header")
}

fn result(accepted: bool, trace: &ProtocolTrace, outputs: &[RunningClaim]) -> Value {
    assert_eq!(trace.alpha.len(), ROUNDS);
    assert_eq!(trace.round_challenges.len(), ROUNDS);
    assert_eq!(trace.round_states.len(), ROUNDS);
    assert_eq!(trace.round_claims.len(), ROUNDS);
    assert!(outputs
        .iter()
        .all(|output| output.r == trace.round_challenges));
    let terminal = trace.terminal_components;
    json!([
        u64::from(accepted),
        extensions(&trace.alpha),
        extension_words(trace.gamma),
        field_words(&trace.pre_sumcheck_state),
        extensions(&trace.round_challenges),
        trace
            .round_states
            .iter()
            .map(|state| field_words(state))
            .collect::<Vec<_>>(),
        extensions(&outputs[0].r),
        extension_words(trace.initial_claim),
        extensions(&trace.round_claims),
        extensions(&[
            terminal.eval_k,
            terminal.eval_a,
            terminal.ccs,
            terminal.norm,
            terminal.terminal,
            trace.terminal_claim
        ]),
        outputs
            .iter()
            .map(|output| field_words(&output.c.data))
            .collect::<Vec<_>>(),
        outputs
            .iter()
            .map(|output| public_words(&output.X))
            .collect::<Vec<_>>(),
        outputs
            .iter()
            .map(|output| extensions(&output.eval_k[..D]))
            .collect::<Vec<_>>(),
        outputs
            .iter()
            .map(|output| output
                .eval_a
                .iter()
                .map(|matrix| extensions(&matrix[..D]))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        field_words(&trace.outgoing_state)
    ])
}

fn main() {
    let started = Instant::now();
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    assert_eq!(
        arguments.len(),
        8,
        "usage: check_pi_ccs_input <candidate> <id0> <id1> <id2> <id3> <input> <Lean-result> <accept|reject|proof-mutations|statement-mutations|output-mutations|point-mutations>"
    );
    let expected_identity = std::array::from_fn(|lane| arguments[lane + 1].parse().expect("identity word"));
    let expected_acceptance = match arguments[7].as_str() {
        "accept" | "proof-mutations" | "statement-mutations" | "output-mutations" | "point-mutations" => true,
        "reject" => false,
        _ => panic!("expected outcome must be accept, reject, or a named mutation group"),
    };
    let bytes = fs::read(&arguments[0]).expect("Lean candidate package");
    let package = load_per_application_package(&bytes, expected_identity).expect("selected Lean structural identity");
    let structure = relation(package.ccs_relation());
    drop(package);
    drop(bytes);
    let params = Params::for_ccs_shape(structure.n, structure.m, MATRICES, structure.max_degree())
        .expect("selected Nightstream parameters");
    assert_eq!(
        (params.inner().b, params.inner().k_rho, params.inner().kappa),
        (2, 16, 22)
    );

    let raw_input = read_value(Path::new(&arguments[5]));
    let Input(schema, commitment, public, rounds, eval_k, eval_a, running_input) =
        serde_json::from_value(raw_input.clone()).expect("PiCCSInputCheck schema");
    assert_eq!(schema, 2);
    let RunningInput(prior_point, running_commitments, running_public, running_k, running_a) = running_input;
    assert_eq!(prior_point.len(), ROUNDS);
    assert_eq!(running_commitments.len(), RUNNING);
    assert!(running_commitments
        .iter()
        .all(|words| words.len() == COMMITMENT));
    assert_eq!(running_public.len(), RUNNING);
    assert!(running_public.iter().all(|words| words.len() == PUBLIC));
    assert_eq!(running_k.len(), RUNNING);
    assert!(running_k.iter().all(|family| family.len() == D));
    assert_eq!(running_a.len(), RUNNING);
    assert!(running_a
        .iter()
        .all(|family| family.len() == MATRICES && family.iter().all(|matrix| matrix.len() == D)));
    assert_eq!((commitment.len(), public.len()), (COMMITMENT, PUBLIC));
    assert_eq!(rounds.len(), ROUNDS);
    assert!(rounds.iter().all(|round| round.len() == COEFFICIENTS));
    assert_eq!((eval_k.len(), eval_a.len()), (SOURCES, SOURCES));
    assert!(eval_k.iter().all(|family| family.len() == D));
    assert!(eval_a
        .iter()
        .all(|family| family.len() == MATRICES && family.iter().all(|matrix| matrix.len() == D)));
    assert_eq!(public[0], 1, "public encHash marker");
    assert!(public[1..257].iter().all(|&word| word <= 1), "public digest bits");
    assert!(public[257..].iter().all(|&word| word == 0), "public encHash tail");
    let prior_digest =
        std::array::from_fn(|lane| (0..64).fold(0, |word, bit| word | (public[1 + lane * 64 + bit] << bit)));
    assert!(
        prior_digest.iter().all(|&word| word < MODULUS),
        "canonical decoded digest"
    );
    let public_blocks = vec![prior_digest.to_vec(), commitment.clone(), public.clone()];
    let mut claimed = Vec::with_capacity(RUNNING * (MATRICES + 1) * D * 2);
    for coefficient in 0..D {
        for source in 0..RUNNING {
            claimed.extend(running_k[source][coefficient]);
        }
    }
    for coefficient in 0..D {
        for matrix in 0..MATRICES {
            for source in 0..RUNNING {
                claimed.extend(running_a[source][matrix][coefficient]);
            }
        }
    }
    let verifier_blocks = vec![prior_point.iter().flatten().copied().collect(), claimed];
    let mut output_words = Vec::new();
    for source in 0..SOURCES {
        output_words.extend(eval_k[source].iter().flatten().copied());
        output_words.extend(eval_a[source].iter().flatten().flatten().copied());
    }
    let replay = derive_pi_ccs_v1_1_transcript(&public_blocks, &verifier_blocks, &rounds, &output_words)
        .expect("independent complete input transcript");
    let point = replay
        .round_point()
        .iter()
        .copied()
        .map(extension)
        .collect::<Vec<_>>();
    let outgoing = replay.outgoing_state();
    let fold_digest = digest_bytes(outgoing[..4].try_into().expect("four output digest words"));
    let fresh = CcsClaim {
        c: Commitment {
            d: D,
            kappa: 22,
            data: commitment.iter().map(|&word| F::from_u64(word)).collect(),
        },
        x: public.iter().map(|&word| F::from_u64(word)).collect(),
        m_in: PUBLIC,
        adv: None,
    };
    let running = (0..RUNNING)
        .map(|source| RunningClaim {
            c: Commitment {
                d: D,
                kappa: 22,
                data: running_commitments[source]
                    .iter()
                    .map(|&word| F::from_u64(word))
                    .collect(),
            },
            X: public_matrix(&running_public[source]),
            r: prior_point.iter().copied().map(extension).collect(),
            eval_k: padded_family(&running_k[source]),
            eval_a: running_a[source]
                .iter()
                .map(|family| padded_family(family))
                .collect(),
            m_in: PUBLIC,
            fold_digest: digest_bytes(prior_digest),
            adv: None,
        })
        .collect::<Vec<_>>();
    let outputs = (0..SOURCES)
        .map(|source| RunningClaim {
            c: if source == 0 {
                fresh.c.clone()
            } else {
                running[source - 1].c.clone()
            },
            X: if source == 0 {
                public_matrix(&public)
            } else {
                running[source - 1].X.clone()
            },
            r: point.clone(),
            eval_k: padded_family(&eval_k[source]),
            eval_a: eval_a[source]
                .iter()
                .map(|family| padded_family(family))
                .collect(),
            m_in: PUBLIC,
            fold_digest,
            adv: None,
        })
        .collect::<Vec<_>>();
    let proof = assemble_proof(
        rounds
            .iter()
            .map(|round| round.iter().copied().map(extension).collect())
            .collect(),
    );
    let lean = read_value(Path::new(&arguments[6]));
    let lean = lean.as_array().expect("Lean checked-result envelope");
    assert_eq!(lean.len(), 6);
    assert_eq!(lean[0], json!(1));
    assert_eq!(lean[1], raw_input, "identical serialized input and proof");
    assert_eq!(
        lean[2],
        json!([
            extensions(&running[0].r),
            running
                .iter()
                .map(|claim| field_words(&claim.c.data))
                .collect::<Vec<_>>(),
            running
                .iter()
                .map(|claim| public_words(&claim.X))
                .collect::<Vec<_>>(),
            running
                .iter()
                .map(|claim| extensions(&claim.eval_k[..D]))
                .collect::<Vec<_>>(),
            running
                .iter()
                .map(|claim| claim
                    .eval_a
                    .iter()
                    .map(|family| extensions(&family[..D]))
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>()
        ]),
        "identical complete decoded running statement"
    );
    assert_eq!(lean[3], json!(public_blocks));
    assert_eq!(lean[4], json!(verifier_blocks));
    assert_eq!(lean[5].as_array().expect("complete Lean result").len(), 15);
    assert_eq!(
        lean[5][0],
        json!(u64::from(expected_acceptance)),
        "required Lean outcome"
    );

    let mut paper_transcript = Poseidon2Transcript::new_v1_1();
    let paper = paper_exact_verify_with_trace(
        &mut paper_transcript,
        params.inner(),
        &structure,
        std::slice::from_ref(&fresh),
        &running,
        &outputs,
        &proof,
    );
    let mut optimized_transcript = Poseidon2Transcript::new_v1_1();
    let optimized = optimized_verify_with_trace(
        &mut optimized_transcript,
        params.inner(),
        &structure,
        std::slice::from_ref(&fresh),
        &running,
        &outputs,
        &proof,
    );
    if !expected_acceptance {
        if let (Err(PiCcsError::SumcheckError(paper)), Err(PiCcsError::SumcheckError(optimized))) = (&paper, &optimized)
        {
            println!("pi_ccs_early_rejection=passed Lean=false paper_exact={paper:?} optimized={optimized:?}");
            println!("complete_rust_trace=unavailable_after_early_rejection");
            return;
        }
    }
    let (paper_accepted, paper_trace) = paper.expect("PaperExact complete verifier result");
    let (optimized_accepted, optimized_trace) = optimized.expect("optimized complete verifier result");
    assert_eq!(paper_accepted, expected_acceptance);
    assert_eq!(optimized_accepted, expected_acceptance);
    assert_eq!(
        result(paper_accepted, &paper_trace, &outputs),
        lean[5],
        "complete Lean/PaperExact phase values"
    );
    assert_eq!(
        result(optimized_accepted, &optimized_trace, &outputs),
        lean[5],
        "complete Lean/optimized phase values"
    );
    println!(
        "pi_ccs_complete_phase_values=passed accepted={expected_acceptance} elapsed={:?}",
        started.elapsed()
    );
    println!("opening_validity_and_assignment_satisfaction=separate_required_gates");
    if arguments[7] == "proof-mutations" {
        mutation_checks::check_proof_mutations(params.inner(), &structure, &fresh, &running, &outputs, &proof);
    }
    if matches!(
        arguments[7].as_str(),
        "statement-mutations" | "output-mutations" | "point-mutations"
    ) {
        mutation_checks::check_claim_mutations(
            params.inner(),
            &structure,
            &fresh,
            &running,
            &outputs,
            &proof,
            &arguments[7],
        );
    }
}
