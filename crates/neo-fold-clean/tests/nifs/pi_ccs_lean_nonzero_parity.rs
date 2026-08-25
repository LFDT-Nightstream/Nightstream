//! Complete nonzero PiCCS parity against the Lean-emitted Stage 1 fixture.

use std::{fs, path::PathBuf};

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim, Mat, SparsePoly, Term};
use neo_fold_clean::paper::params::Params;
use neo_math::{from_complex, KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::paper_exact_engine::paper_exact_verify_with_trace;
use neo_reductions::engines::pi_ccs_joint::ProtocolTrace;
use neo_reductions::optimized_engine::optimized_verify_with_trace;
use neo_reductions::PiCcsProof;
use neo_transcript::Poseidon2Transcript;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{de::IgnoredAny, Deserialize, Serialize};

const MODULUS: u64 = 0xffff_ffff_0000_0001;
const SOURCE_COUNT: usize = 17;
const RUNNING_COUNT: usize = 16;
const MATRIX_COUNT: usize = 14;
const COEFFICIENT_COUNT: usize = 54;
const ROUND_COUNT: usize = 25;
const ROUND_COEFFICIENT_COUNT: usize = 10;
const STATE_PREIMAGE_WORDS: usize = 42_475;
const STATE_DOMAIN_TAG: [u64; 23] = [
    72, 121, 112, 101, 114, 78, 111, 118, 97, 47, 78, 73, 86, 67, 47, 115, 116, 97, 116, 101, 47, 118, 49,
];

type FreshClaim = CcsClaim<Commitment, F>;
type RunningClaim = CeClaim<Commitment, F, K>;

#[derive(Deserialize)]
struct RawParity(u64, RawInput, RawResult);

#[derive(Deserialize)]
struct RawInput(
    Vec<u64>,
    Vec<u64>,
    Vec<u64>,
    [u64; 4],
    [u64; 4],
    Vec<u64>,
    Vec<Vec<[u64; 2]>>,
    Vec<Vec<[u64; 2]>>,
    Vec<Vec<Vec<[u64; 2]>>>,
    Vec<Vec<u64>>,
    Vec<Vec<u64>>,
);

#[derive(Deserialize)]
struct RawResult(
    u64,
    Vec<[u64; 2]>,
    [u64; 2],
    [u64; 8],
    Vec<[u64; 2]>,
    Vec<[u64; 8]>,
    Vec<[u64; 2]>,
    [u64; 2],
    Vec<[u64; 2]>,
    Vec<[u64; 2]>,
    Vec<Vec<u64>>,
    Vec<Vec<u64>>,
    Vec<Vec<[u64; 2]>>,
    Vec<Vec<Vec<[u64; 2]>>>,
    [u64; 8],
    Vec<u64>,
);

#[derive(Deserialize)]
struct RawPackage(
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    RawRelation,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
);

#[derive(Deserialize)]
struct RawRelation(u64, u64, u64, Vec<u64>, u64, Vec<RawTerm>);

#[derive(Deserialize)]
struct RawTerm(u64, Vec<u64>);

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct PhaseResult {
    accepted: bool,
    alpha: Vec<[u64; 2]>,
    gamma: [u64; 2],
    pre_sumcheck_state: [u64; 8],
    round_challenges: Vec<[u64; 2]>,
    round_states: Vec<[u64; 8]>,
    r_prime: Vec<[u64; 2]>,
    initial_claim: [u64; 2],
    round_claims: Vec<[u64; 2]>,
    terminal_components: Vec<[u64; 2]>,
    output_commitments: Vec<Vec<u64>>,
    output_public_inputs: Vec<Vec<u64>>,
    output_eval_k: Vec<Vec<[u64; 2]>>,
    output_eval_a: Vec<Vec<Vec<[u64; 2]>>>,
    outgoing_state: [u64; 8],
}

impl PhaseResult {
    fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("canonical PiCCS phase-result JSON")
    }
}

fn package_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-v1.json")
}

fn parity_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-piccs-parity-v1.json")
}

fn field(value: u64) -> F {
    assert!(value < MODULUS, "canonical Goldilocks word");
    F::from_u64(value)
}

fn extension(value: [u64; 2]) -> K {
    from_complex(field(value[0]), field(value[1]))
}

fn extension_words(value: K) -> [u64; 2] {
    value.to_limbs_u64().into()
}

fn state_words(state: [F; 8]) -> [u64; 8] {
    state.map(|value| value.as_canonical_u64())
}

fn fields(words: &[u64]) -> Vec<F> {
    words.iter().map(|word| field(*word)).collect()
}

fn commitment(words: &[u64], kappa: usize) -> Commitment {
    assert_eq!(words.len(), D * kappa, "commitment width");
    Commitment {
        d: D,
        kappa,
        data: fields(words),
    }
}

fn public_input(words: &[u64]) -> Mat<F> {
    assert_eq!(words.len(), D, "one-ring public input");
    let mut output = Mat::zero(D, 1, F::ZERO);
    for (row, word) in words.iter().enumerate() {
        output[(row, 0)] = field(*word);
    }
    output
}

fn padded_family(values: &[[u64; 2]]) -> Vec<K> {
    assert_eq!(values.len(), COEFFICIENT_COUNT, "evaluation coefficient count");
    let mut output = values.iter().copied().map(extension).collect::<Vec<_>>();
    output.resize(D.next_power_of_two(), K::ZERO);
    output
}

fn evaluation_block(words: &[u64]) -> (Vec<K>, Vec<Vec<K>>) {
    let expected = (MATRIX_COUNT + 1) * COEFFICIENT_COUNT * 2;
    assert_eq!(words.len(), expected, "running evaluation block");
    let values = words
        .chunks_exact(2)
        .map(|value| extension([value[0], value[1]]))
        .collect::<Vec<_>>();
    let mut eval_k = values[..COEFFICIENT_COUNT].to_vec();
    eval_k.resize(D.next_power_of_two(), K::ZERO);
    let eval_a = values[COEFFICIENT_COUNT..]
        .chunks_exact(COEFFICIENT_COUNT)
        .map(|matrix| {
            let mut matrix = matrix.to_vec();
            matrix.resize(D.next_power_of_two(), K::ZERO);
            matrix
        })
        .collect();
    (eval_k, eval_a)
}

fn framed_payload<'a>(words: &'a [u64], cursor: &mut usize, expected_len: usize, label: &str) -> &'a [u64] {
    assert_eq!(
        words.get(*cursor).copied(),
        Some(expected_len as u64),
        "{label} length prefix"
    );
    let start = *cursor + 1;
    let end = start + expected_len;
    let payload = words
        .get(start..end)
        .unwrap_or_else(|| panic!("{label} payload"));
    *cursor = end;
    payload
}

fn output_digest(state: [u64; 8]) -> [u8; 32] {
    let mut digest = [0u8; 32];
    for (lane, word) in state[..4].iter().enumerate() {
        digest[lane * 8..(lane + 1) * 8].copy_from_slice(&word.to_le_bytes());
    }
    digest
}

fn relation(raw: &RawRelation) -> CcsStructure<F> {
    let rows = usize::try_from(raw.0).expect("relation rows fit usize");
    let columns = usize::try_from(raw.1).expect("relation columns fit usize");
    assert_eq!(raw.2, ROUND_COUNT as u64, "relation cube variables");
    assert_eq!(raw.3, (0..MATRIX_COUNT as u64).collect::<Vec<_>>());
    assert_eq!(raw.4, (ROUND_COEFFICIENT_COUNT - 1) as u64);
    let terms = raw
        .5
        .iter()
        .map(|term| {
            assert_eq!(term.1.len(), MATRIX_COUNT, "relation term arity");
            Term {
                coeff: field(term.0),
                exps: term
                    .1
                    .iter()
                    .map(|exponent| u32::try_from(*exponent).expect("relation exponent fits u32"))
                    .collect(),
            }
        })
        .collect();
    let polynomial = SparsePoly::new(MATRIX_COUNT, terms);
    assert_eq!(polynomial.max_degree() as usize + 1, ROUND_COEFFICIENT_COUNT - 1);
    CcsStructure::new_verifier_artifact_header(rows, columns, MATRIX_COUNT, polynomial).expect("Lean relation header")
}

fn statement_claims(input: &RawInput, params: &NeoParams) -> (FreshClaim, Vec<RunningClaim>) {
    let preimage = &input.0;
    assert_eq!(preimage.len(), STATE_PREIMAGE_WORDS, "canonical state preimage width");
    assert_eq!(&preimage[..STATE_DOMAIN_TAG.len()], STATE_DOMAIN_TAG);

    let mut cursor = STATE_DOMAIN_TAG.len();
    let context = framed_payload(preimage, &mut cursor, 4, "verifier context");
    assert_eq!(context, input.4, "prior verifier context");
    assert_eq!(&input.1[..STATE_DOMAIN_TAG.len()], STATE_DOMAIN_TAG);
    let mut output_cursor = STATE_DOMAIN_TAG.len();
    let output_context = framed_payload(&input.1, &mut output_cursor, 4, "output verifier context");
    assert_eq!(output_context, input.4, "output verifier context");
    let _iteration = *preimage.get(cursor).expect("iteration word");
    cursor += 1;
    let _z0 = framed_payload(preimage, &mut cursor, 4, "initial application state");
    let _current = framed_payload(preimage, &mut cursor, 4, "current application state");
    let point_words = framed_payload(preimage, &mut cursor, ROUND_COUNT * 2, "shared running point");
    let point = point_words
        .chunks_exact(2)
        .map(|value| extension([value[0], value[1]]))
        .collect::<Vec<_>>();
    assert_eq!(point.len(), ROUND_COUNT);

    let mut running = Vec::with_capacity(RUNNING_COUNT);
    for source in 0..RUNNING_COUNT {
        let commitment_words = framed_payload(preimage, &mut cursor, D * params.kappa as usize, "running commitment");
        let public_input_words = framed_payload(preimage, &mut cursor, D, "running public input");
        let evaluation_words = framed_payload(
            preimage,
            &mut cursor,
            (MATRIX_COUNT + 1) * COEFFICIENT_COUNT * 2,
            "running evaluations",
        );
        let (eval_k, eval_a) = evaluation_block(evaluation_words);
        running.push(RunningClaim {
            c: commitment(commitment_words, params.kappa as usize),
            X: public_input(public_input_words),
            r: point.clone(),
            eval_k,
            eval_a,
            m_in: D,
            fold_digest: [0; 32],
            adv: None,
        });
        assert_eq!(running.len(), source + 1);
    }

    assert_eq!(preimage.get(cursor).copied(), Some(1), "one-based program counter");
    cursor += 1;
    assert_eq!(cursor, preimage.len(), "complete state preimage consumption");

    let blocks = &input.9;
    assert_eq!(blocks.len(), 3, "digest-only public block count");
    assert_eq!(blocks[0], input.3, "prior-state digest block");
    assert_eq!(blocks[1], input.5, "fresh commitment block");
    assert_eq!(blocks[2], input.2, "fresh public-input block");
    let fresh = FreshClaim {
        c: commitment(&blocks[1], params.kappa as usize),
        x: fields(&blocks[2]),
        m_in: D,
        adv: None,
    };
    (fresh, running)
}

fn verify_input_blocks(input: &RawInput, running: &[RunningClaim]) {
    assert_eq!(input.10.len(), 2, "verifier-input block count");
    let prior_point = running[0]
        .r
        .iter()
        .copied()
        .flat_map(extension_words)
        .collect::<Vec<_>>();
    assert_eq!(input.10[0], prior_point, "verifier prior point");
    let mut evaluations = Vec::with_capacity(RUNNING_COUNT * (MATRIX_COUNT + 1) * D * 2);
    for coefficient in 0..D {
        for claim in running {
            evaluations.extend(extension_words(claim.eval_k[coefficient]));
        }
    }
    for coefficient in 0..D {
        for matrix in 0..MATRIX_COUNT {
            for claim in running {
                evaluations.extend(extension_words(claim.eval_a[matrix][coefficient]));
            }
        }
    }
    assert_eq!(input.10[1], evaluations, "verifier prior evaluations");
}

fn outputs(result: &RawResult, params: &NeoParams) -> Vec<RunningClaim> {
    assert_eq!(result.10.len(), SOURCE_COUNT);
    assert_eq!(result.11.len(), SOURCE_COUNT);
    assert_eq!(result.12.len(), SOURCE_COUNT);
    assert_eq!(result.13.len(), SOURCE_COUNT);
    let digest = output_digest(result.14);
    (0..SOURCE_COUNT)
        .map(|source| RunningClaim {
            c: commitment(&result.10[source], params.kappa as usize),
            X: public_input(&result.11[source]),
            r: result.6.iter().copied().map(extension).collect(),
            eval_k: padded_family(&result.12[source]),
            eval_a: result.13[source]
                .iter()
                .map(|matrix| padded_family(matrix))
                .collect(),
            m_in: D,
            fold_digest: digest,
            adv: None,
        })
        .collect()
}

fn proof(input: &RawInput) -> PiCcsProof {
    assert_eq!(input.6.len(), ROUND_COUNT);
    assert!(input
        .6
        .iter()
        .all(|round| round.len() == ROUND_COEFFICIENT_COUNT));
    PiCcsProof::new(
        input
            .6
            .iter()
            .map(|round| round.iter().copied().map(extension).collect())
            .collect(),
    )
}

fn lean_result(raw: &RawResult) -> PhaseResult {
    assert_eq!(raw.15, vec![1; 4], "Lean nonzero assurance flags");
    PhaseResult {
        accepted: raw.0 == 1,
        alpha: raw.1.clone(),
        gamma: raw.2,
        pre_sumcheck_state: raw.3,
        round_challenges: raw.4.clone(),
        round_states: raw.5.clone(),
        r_prime: raw.6.clone(),
        initial_claim: raw.7,
        round_claims: raw.8.clone(),
        terminal_components: raw.9.clone(),
        output_commitments: raw.10.clone(),
        output_public_inputs: raw.11.clone(),
        output_eval_k: raw.12.clone(),
        output_eval_a: raw.13.clone(),
        outgoing_state: raw.14,
    }
}

fn engine_result(accepted: bool, trace: &ProtocolTrace, outputs: &[RunningClaim]) -> PhaseResult {
    assert!(outputs
        .iter()
        .all(|output| output.r == trace.round_challenges));
    let terminal = trace.terminal_components;
    PhaseResult {
        accepted,
        alpha: trace.alpha.iter().copied().map(extension_words).collect(),
        gamma: extension_words(trace.gamma),
        pre_sumcheck_state: state_words(trace.pre_sumcheck_state),
        round_challenges: trace
            .round_challenges
            .iter()
            .copied()
            .map(extension_words)
            .collect(),
        round_states: trace
            .round_states
            .iter()
            .copied()
            .map(state_words)
            .collect(),
        r_prime: outputs[0].r.iter().copied().map(extension_words).collect(),
        initial_claim: extension_words(trace.initial_claim),
        round_claims: trace
            .round_claims
            .iter()
            .copied()
            .map(extension_words)
            .collect(),
        terminal_components: vec![
            extension_words(terminal.eval_k),
            extension_words(terminal.eval_a),
            extension_words(terminal.ccs),
            extension_words(terminal.norm),
            extension_words(terminal.terminal),
            extension_words(trace.terminal_claim),
        ],
        output_commitments: outputs
            .iter()
            .map(|output| {
                output
                    .c
                    .data
                    .iter()
                    .map(|value| value.as_canonical_u64())
                    .collect()
            })
            .collect(),
        output_public_inputs: outputs
            .iter()
            .map(|output| {
                (0..output.m_in)
                    .map(|row| output.X[(row, 0)].as_canonical_u64())
                    .collect()
            })
            .collect(),
        output_eval_k: outputs
            .iter()
            .map(|output| {
                output.eval_k[..D]
                    .iter()
                    .copied()
                    .map(extension_words)
                    .collect()
            })
            .collect(),
        output_eval_a: outputs
            .iter()
            .map(|output| {
                output
                    .eval_a
                    .iter()
                    .map(|matrix| matrix[..D].iter().copied().map(extension_words).collect())
                    .collect()
            })
            .collect(),
        outgoing_state: state_words(trace.outgoing_state),
    }
}

struct EngineFixture {
    structure: CcsStructure<F>,
    params: Params,
    fresh: FreshClaim,
    running: Vec<RunningClaim>,
    outputs: Vec<RunningClaim>,
    proof: PiCcsProof,
    expected: PhaseResult,
}

fn engine_fixture() -> EngineFixture {
    let parity: RawParity = serde_json::from_slice(&fs::read(parity_path()).expect("Lean PiCCS parity bytes"))
        .expect("Lean PiCCS parity JSON");
    assert_eq!(parity.0, 6, "PiCCS parity schema");
    let package: RawPackage =
        serde_json::from_slice(&fs::read(package_path()).expect("Lean package bytes")).expect("Lean package JSON");
    let structure = relation(&package.4);
    let params = Params::for_ccs_shape(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-bound Nightstream parameters");
    let (fresh, running) = statement_claims(&parity.1, params.inner());
    verify_input_blocks(&parity.1, &running);
    let outputs = outputs(&parity.2, params.inner());
    let proof = proof(&parity.1);
    EngineFixture {
        structure,
        params,
        fresh,
        running,
        outputs,
        proof,
        expected: lean_result(&parity.2),
    }
}

fn optimized_rejects(
    fixture: &EngineFixture,
    fresh: &FreshClaim,
    running: &[RunningClaim],
    outputs: &[RunningClaim],
    proof: &PiCcsProof,
) -> bool {
    let mut transcript = Poseidon2Transcript::new_v1_1();
    match optimized_verify_with_trace(
        &mut transcript,
        fixture.params.inner(),
        &fixture.structure,
        std::slice::from_ref(fresh),
        running,
        outputs,
        proof,
    ) {
        Ok((accepted, _)) => !accepted,
        Err(_) => true,
    }
}

fn changed_word(word: u64) -> u64 {
    if word + 1 == MODULUS {
        0
    } else {
        word + 1
    }
}

fn changed_extension(value: K) -> K {
    let mut words = extension_words(value);
    words[0] = changed_word(words[0]);
    extension(words)
}

#[test]
fn lean_paper_exact_and_optimized_match_complete_nonzero_pi_ccs_result() {
    let parity: RawParity = serde_json::from_slice(&fs::read(parity_path()).expect("Lean PiCCS parity bytes"))
        .expect("Lean PiCCS parity JSON");
    assert_eq!(parity.0, 6, "PiCCS parity schema");
    assert_eq!(parity.1 .0, parity.1 .1, "pilot preimage pair");
    assert_eq!(parity.1 .2, parity.1 .9[2]);
    assert_eq!(parity.1 .3.len(), 4);
    assert_eq!(parity.1 .3.as_slice(), parity.1 .9[0].as_slice());
    assert_eq!(parity.1 .5, parity.1 .9[1]);
    assert_eq!(parity.1 .7, parity.2 .12, "output Eval_K input/result");
    assert_eq!(parity.1 .8, parity.2 .13, "output Eval_A input/result");

    let package: RawPackage =
        serde_json::from_slice(&fs::read(package_path()).expect("Lean package bytes")).expect("Lean package JSON");
    let structure = relation(&package.4);
    let params = Params::for_ccs_shape(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-bound Nightstream parameters");
    let security = params
        .inner()
        .padded_row_security_check_for_shape(
            structure.n,
            structure.m,
            structure.t(),
            structure.max_degree(),
            neo_params::goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
        )
        .expect("shape-bound PiCCS security census");
    assert_eq!(params.lambda(), security.security_bits);
    let (fresh, running) = statement_claims(&parity.1, params.inner());
    verify_input_blocks(&parity.1, &running);
    let outputs = outputs(&parity.2, params.inner());
    assert_eq!(outputs[0].c, fresh.c);
    assert_eq!(parity.2 .11[0], parity.1 .2);
    for source in 0..RUNNING_COUNT {
        assert_eq!(outputs[source + 1].c, running[source].c);
        assert_eq!(outputs[source + 1].X, running[source].X);
    }
    let proof = proof(&parity.1);

    let mut paper_transcript = Poseidon2Transcript::new_v1_1();
    let (paper_accepted, paper_trace) = paper_exact_verify_with_trace(
        &mut paper_transcript,
        params.inner(),
        &structure,
        std::slice::from_ref(&fresh),
        &running,
        &outputs,
        &proof,
    )
    .expect("PaperExact verifies Lean PiCCS proof");

    let mut optimized_transcript = Poseidon2Transcript::new_v1_1();
    let (optimized_accepted, optimized_trace) = optimized_verify_with_trace(
        &mut optimized_transcript,
        params.inner(),
        &structure,
        std::slice::from_ref(&fresh),
        &running,
        &outputs,
        &proof,
    )
    .expect("optimized verifies Lean PiCCS proof");

    let lean = lean_result(&parity.2);
    let paper = engine_result(paper_accepted, &paper_trace, &outputs);
    let optimized = engine_result(optimized_accepted, &optimized_trace, &outputs);
    assert!(lean.accepted, "Lean fixture acceptance");
    assert_eq!(paper, lean, "PaperExact complete result equals Lean");
    assert_eq!(optimized, lean, "optimized complete result equals Lean");
    assert_eq!(paper.canonical_bytes(), lean.canonical_bytes());
    assert_eq!(optimized.canonical_bytes(), lean.canonical_bytes());
}

#[test]
fn optimized_rejects_every_nonzero_pi_ccs_statement_family_mutation() {
    let fixture = engine_fixture();

    for coordinate in 0..ROUND_COUNT {
        let mut running = fixture.running.clone();
        for claim in &mut running {
            claim.r[coordinate] = changed_extension(claim.r[coordinate]);
        }
        assert!(
            optimized_rejects(&fixture, &fixture.fresh, &running, &fixture.outputs, &fixture.proof,),
            "shared prior-point coordinate {coordinate}",
        );
    }

    for source in 0..RUNNING_COUNT {
        let mut running = fixture.running.clone();
        running[source].c.data[0] += F::ONE;
        assert!(optimized_rejects(
            &fixture,
            &fixture.fresh,
            &running,
            &fixture.outputs,
            &fixture.proof,
        ));

        let mut running = fixture.running.clone();
        running[source].X[(0, 0)] += F::ONE;
        assert!(optimized_rejects(
            &fixture,
            &fixture.fresh,
            &running,
            &fixture.outputs,
            &fixture.proof,
        ));

        let mut running = fixture.running.clone();
        running[source].eval_k[0] = changed_extension(running[source].eval_k[0]);
        assert!(optimized_rejects(
            &fixture,
            &fixture.fresh,
            &running,
            &fixture.outputs,
            &fixture.proof,
        ));

        for matrix in 0..MATRIX_COUNT {
            let mut running = fixture.running.clone();
            running[source].eval_a[matrix][0] = changed_extension(running[source].eval_a[matrix][0]);
            assert!(
                optimized_rejects(&fixture, &fixture.fresh, &running, &fixture.outputs, &fixture.proof,),
                "running source {source} Eval_A matrix {matrix}",
            );
        }
    }

    let mut fresh = fixture.fresh.clone();
    fresh.c.data[0] += F::ONE;
    assert!(optimized_rejects(
        &fixture,
        &fresh,
        &fixture.running,
        &fixture.outputs,
        &fixture.proof,
    ));

    let mut fresh = fixture.fresh.clone();
    fresh.x[0] += F::ONE;
    assert!(optimized_rejects(
        &fixture,
        &fresh,
        &fixture.running,
        &fixture.outputs,
        &fixture.proof,
    ));
}

#[test]
fn optimized_rejects_every_nonzero_pi_ccs_proof_and_output_family_mutation() {
    let fixture = engine_fixture();

    for round in 0..ROUND_COUNT {
        let mut proof = fixture.proof.clone();
        proof.sumcheck_rounds[round][0] = changed_extension(proof.sumcheck_rounds[round][0]);
        assert!(
            optimized_rejects(&fixture, &fixture.fresh, &fixture.running, &fixture.outputs, &proof,),
            "SumCheck round {round}",
        );
    }

    for source in 0..SOURCE_COUNT {
        let mut outputs = fixture.outputs.clone();
        outputs[source].c.data[0] += F::ONE;
        assert!(optimized_rejects(
            &fixture,
            &fixture.fresh,
            &fixture.running,
            &outputs,
            &fixture.proof,
        ));

        let mut outputs = fixture.outputs.clone();
        outputs[source].X[(0, 0)] += F::ONE;
        assert!(optimized_rejects(
            &fixture,
            &fixture.fresh,
            &fixture.running,
            &outputs,
            &fixture.proof,
        ));

        let mut outputs = fixture.outputs.clone();
        outputs[source].eval_k[0] = changed_extension(outputs[source].eval_k[0]);
        assert!(
            optimized_rejects(&fixture, &fixture.fresh, &fixture.running, &outputs, &fixture.proof,),
            "output source {source} Eval_K",
        );

        for matrix in 0..MATRIX_COUNT {
            let mut outputs = fixture.outputs.clone();
            outputs[source].eval_a[matrix][0] = changed_extension(outputs[source].eval_a[matrix][0]);
            assert!(
                optimized_rejects(&fixture, &fixture.fresh, &fixture.running, &outputs, &fixture.proof,),
                "output source {source} Eval_A matrix {matrix}",
            );
        }
    }
}

#[test]
fn complete_result_comparator_detects_every_pi_ccs_result_family_mutation() {
    let fixture = engine_fixture();
    let expected = fixture.expected;
    let baseline = expected.canonical_bytes();
    let differs = |mutated: &PhaseResult| {
        assert_ne!(mutated.canonical_bytes(), baseline);
    };

    let mut mutated = expected.clone();
    mutated.accepted = !mutated.accepted;
    differs(&mutated);

    for coordinate in 0..expected.alpha.len() {
        let mut mutated = expected.clone();
        mutated.alpha[coordinate][0] = changed_word(mutated.alpha[coordinate][0]);
        differs(&mutated);
    }
    let mut mutated = expected.clone();
    mutated.gamma[0] = changed_word(mutated.gamma[0]);
    differs(&mutated);
    for lane in 0..8 {
        let mut mutated = expected.clone();
        mutated.pre_sumcheck_state[lane] = changed_word(mutated.pre_sumcheck_state[lane]);
        differs(&mutated);
    }
    for round in 0..expected.round_challenges.len() {
        let mut mutated = expected.clone();
        mutated.round_challenges[round][0] = changed_word(mutated.round_challenges[round][0]);
        differs(&mutated);
    }
    for round in 0..expected.round_states.len() {
        for lane in 0..8 {
            let mut mutated = expected.clone();
            mutated.round_states[round][lane] = changed_word(mutated.round_states[round][lane]);
            differs(&mutated);
        }
    }
    for coordinate in 0..expected.r_prime.len() {
        let mut mutated = expected.clone();
        mutated.r_prime[coordinate][0] = changed_word(mutated.r_prime[coordinate][0]);
        differs(&mutated);
    }
    let mut mutated = expected.clone();
    mutated.initial_claim[0] = changed_word(mutated.initial_claim[0]);
    differs(&mutated);
    for round in 0..expected.round_claims.len() {
        let mut mutated = expected.clone();
        mutated.round_claims[round][0] = changed_word(mutated.round_claims[round][0]);
        differs(&mutated);
    }
    for component in 0..expected.terminal_components.len() {
        let mut mutated = expected.clone();
        mutated.terminal_components[component][0] = changed_word(mutated.terminal_components[component][0]);
        differs(&mutated);
    }
    for source in 0..SOURCE_COUNT {
        let mut mutated = expected.clone();
        mutated.output_commitments[source][0] = changed_word(mutated.output_commitments[source][0]);
        differs(&mutated);

        let mut mutated = expected.clone();
        mutated.output_public_inputs[source][0] = changed_word(mutated.output_public_inputs[source][0]);
        differs(&mutated);

        let mut mutated = expected.clone();
        mutated.output_eval_k[source][0][0] = changed_word(mutated.output_eval_k[source][0][0]);
        differs(&mutated);

        for matrix in 0..MATRIX_COUNT {
            let mut mutated = expected.clone();
            mutated.output_eval_a[source][matrix][0][0] = changed_word(mutated.output_eval_a[source][matrix][0][0]);
            differs(&mutated);
        }
    }
    for lane in 0..8 {
        let mut mutated = expected.clone();
        mutated.outgoing_state[lane] = changed_word(mutated.outgoing_state[lane]);
        differs(&mutated);
    }
}
