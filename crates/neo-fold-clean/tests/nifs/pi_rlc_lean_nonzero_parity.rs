//! Complete nonzero PiRLC value parity against the Lean-emitted Stage 1 fixture.

use std::{fs, path::PathBuf};

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim, Mat, SparsePoly, Term};
use neo_fold_clean::engine::{optimized, paper_exact};
use neo_fold_clean::paper::{params::Params, relations::ajtai_rlc_mixer};
use neo_math::{from_complex, KExtensions, D, F, K};
use neo_reductions::{api, common::RotRho, engines::paper_exact_engine};
use neo_transcript::Poseidon2Transcript;
use nightstream_fprime::PI_CCS_V1_1_ROUND_COUNT as ROUND_COUNT;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};

const MODULUS: u64 = 0xffff_ffff_0000_0001;
const SOURCE_COUNT: usize = 17;
const MATRIX_COUNT: usize = 14;
const COEFFICIENT_COUNT: usize = 54;
const PUBLIC_INPUT_WORDS: usize = 270;

type Claim = CeClaim<Commitment, F, K>;

#[derive(Deserialize)]
struct Artifact(u64, RawInput, RawResult);

#[derive(Clone, Deserialize)]
struct RawInput(
    [u64; 8],
    Vec<[u64; 2]>,
    Vec<Vec<u64>>,
    Vec<Vec<u64>>,
    Vec<Vec<[u64; 2]>>,
    Vec<Vec<Vec<[u64; 2]>>>,
);

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
struct RawPartial(Vec<u64>, Vec<u64>, Vec<[u64; 2]>, Vec<Vec<[u64; 2]>>);

#[derive(Clone, Deserialize)]
struct RawResult(
    u64,
    Vec<Vec<u64>>,
    Vec<u64>,
    Vec<u64>,
    Vec<u64>,
    Vec<[u64; 2]>,
    Vec<[u64; 2]>,
    Vec<Vec<[u64; 2]>>,
    Vec<RawPartial>,
    [u64; 8],
    Vec<u64>,
);

#[derive(Deserialize)]
struct RawRelation(u64, u64, u64, Vec<u64>, u64, Vec<RawTerm>);

#[derive(Deserialize)]
struct RawTerm(u64, Vec<u64>);

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct PartialResult {
    commitment: Vec<u64>,
    public_input: Vec<u64>,
    eval_k: Vec<[u64; 2]>,
    eval_a: Vec<Vec<[u64; 2]>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct PhaseResult {
    accepted: bool,
    challenges: Vec<Vec<u64>>,
    membership: Vec<bool>,
    commitment: Vec<u64>,
    public_input: Vec<u64>,
    point: Vec<[u64; 2]>,
    eval_k: Vec<[u64; 2]>,
    eval_a: Vec<Vec<[u64; 2]>>,
    partials: Vec<PartialResult>,
    outgoing_state: [u64; 8],
}

impl PhaseResult {
    fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("canonical PiRLC phase-result JSON")
    }
}

fn artifact_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-pirlc-parity-v1.json")
}

fn pi_ccs_artifact_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-piccs-parity-v1.json")
}

fn package_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-v1.json")
}

fn field(word: u64) -> F {
    assert!(word < MODULUS, "canonical Goldilocks word");
    F::from_u64(word)
}

fn extension(words: [u64; 2]) -> K {
    from_complex(field(words[0]), field(words[1]))
}

fn extension_words(value: K) -> [u64; 2] {
    value.to_limbs_u64().into()
}

fn state_words(transcript: &Poseidon2Transcript) -> [u64; 8] {
    transcript.state().map(|value| value.as_canonical_u64())
}

fn commitment(words: &[u64]) -> Commitment {
    assert_eq!(words.len() % D, 0, "whole commitment rows");
    Commitment {
        d: D,
        kappa: words.len() / D,
        data: words.iter().copied().map(field).collect(),
    }
}

fn public_input(words: &[u64]) -> Mat<F> {
    assert_eq!(words.len(), PUBLIC_INPUT_WORDS, "five-ring public input");
    let mut output = Mat::zero(D, PUBLIC_INPUT_WORDS / D, F::ZERO);
    for (index, word) in words.iter().copied().enumerate() {
        output[(index % D, index / D)] = field(word);
    }
    output
}

fn public_input_words(value: &Mat<F>, word_count: usize) -> Vec<u64> {
    assert_eq!(value.rows(), D);
    assert_eq!(value.cols(), word_count.div_ceil(D));
    (0..word_count)
        .map(|index| value[(index % D, index / D)].as_canonical_u64())
        .collect()
}

fn padded_family(values: &[[u64; 2]]) -> Vec<K> {
    assert_eq!(values.len(), COEFFICIENT_COUNT);
    let mut output = values.iter().copied().map(extension).collect::<Vec<_>>();
    output.resize(D.next_power_of_two(), K::ZERO);
    output
}

fn fold_digest(state: [u64; 8]) -> [u8; 32] {
    let mut digest = [0u8; 32];
    for (lane, word) in state[..4].iter().enumerate() {
        digest[lane * 8..(lane + 1) * 8].copy_from_slice(&word.to_le_bytes());
    }
    digest
}

fn claims(input: &RawInput) -> Vec<Claim> {
    assert_eq!(input.1.len(), ROUND_COUNT);
    assert_eq!(input.2.len(), SOURCE_COUNT);
    assert_eq!(input.3.len(), SOURCE_COUNT);
    assert_eq!(input.4.len(), SOURCE_COUNT);
    assert_eq!(input.5.len(), SOURCE_COUNT);
    let point = input.1.iter().copied().map(extension).collect::<Vec<_>>();
    let digest = fold_digest(input.0);
    (0..SOURCE_COUNT)
        .map(|source| {
            assert_eq!(input.5[source].len(), MATRIX_COUNT);
            Claim {
                c: commitment(&input.2[source]),
                X: public_input(&input.3[source]),
                r: point.clone(),
                eval_k: padded_family(&input.4[source]),
                eval_a: input.5[source]
                    .iter()
                    .map(|family| padded_family(family))
                    .collect(),
                m_in: PUBLIC_INPUT_WORDS,
                fold_digest: digest,
                adv: None,
            }
        })
        .collect()
}

fn expected_claim(input: &RawInput, result: &RawResult) -> Claim {
    Claim {
        c: commitment(&result.3),
        X: public_input(&result.4),
        r: result.5.iter().copied().map(extension).collect(),
        eval_k: padded_family(&result.6),
        eval_a: result
            .7
            .iter()
            .map(|family| padded_family(family))
            .collect(),
        m_in: PUBLIC_INPUT_WORDS,
        fold_digest: fold_digest(input.0),
        adv: None,
    }
}

fn expected_partial_claim(input: &RawInput, partial: &RawPartial) -> Claim {
    Claim {
        c: commitment(&partial.0),
        X: public_input(&partial.1),
        r: input.1.iter().copied().map(extension).collect(),
        eval_k: padded_family(&partial.2),
        eval_a: partial
            .3
            .iter()
            .map(|family| padded_family(family))
            .collect(),
        m_in: PUBLIC_INPUT_WORDS,
        fold_digest: fold_digest(input.0),
        adv: None,
    }
}

fn relation() -> CcsStructure<F> {
    let package: serde_json::Value =
        serde_json::from_slice(&fs::read(package_path()).expect("Lean package bytes")).expect("Lean package JSON");
    assert_eq!(package[0].as_u64(), Some(8), "Lean package-plan schema");
    assert_eq!(package[1][0].as_u64(), Some(7), "Lean static-package schema");
    let raw: RawRelation = serde_json::from_value(package[1][4].clone()).expect("Lean relation tuple");
    assert_eq!(raw.2, ROUND_COUNT as u64);
    let active_rows = usize::try_from(raw.0).expect("relation rows");
    let padded_rows = 1usize
        .checked_shl(u32::try_from(raw.2).expect("relation cube variables fit u32"))
        .expect("relation padded row domain fits usize");
    assert!(active_rows <= padded_rows, "active rows fit the padded relation domain");
    assert_eq!(raw.3, (0..MATRIX_COUNT as u64).collect::<Vec<_>>());
    let terms = raw
        .5
        .iter()
        .map(|term| Term {
            coeff: field(term.0),
            exps: term
                .1
                .iter()
                .map(|value| u32::try_from(*value).expect("term exponent"))
                .collect(),
        })
        .collect();
    let polynomial = SparsePoly::new(MATRIX_COUNT, terms);
    assert_eq!(
        u32::try_from(raw.4).expect("relation degree bound"),
        polynomial.max_degree() + 1
    );
    CcsStructure::new_verifier_artifact_header(
        padded_rows,
        usize::try_from(raw.1).expect("relation columns"),
        MATRIX_COUNT,
        polynomial,
    )
    .expect("Lean relation header")
}

fn artifact() -> Artifact {
    serde_json::from_slice(&fs::read(artifact_path()).expect("Lean PiRLC parity bytes"))
        .expect("Lean PiRLC parity JSON")
}

fn challenge_words(rhos: &[RotRho]) -> Vec<Vec<u64>> {
    rhos.iter()
        .map(|rho| {
            (0..D)
                .map(|row| rho.as_mat()[(row, 0)].as_canonical_u64())
                .collect()
        })
        .collect()
}

fn strong_member(words: &[u64]) -> bool {
    words
        .iter()
        .all(|word| matches!(*word, 0 | 1 | 2) || *word == MODULUS - 1 || *word == MODULUS - 2)
}

fn claim_result(claim: &Claim) -> (Vec<u64>, Vec<u64>, Vec<[u64; 2]>, Vec<[u64; 2]>, Vec<Vec<[u64; 2]>>) {
    (
        claim
            .c
            .data
            .iter()
            .map(|value| value.as_canonical_u64())
            .collect(),
        public_input_words(&claim.X, claim.m_in),
        claim.r.iter().copied().map(extension_words).collect(),
        claim.eval_k[..D]
            .iter()
            .copied()
            .map(extension_words)
            .collect(),
        claim
            .eval_a
            .iter()
            .map(|family| family[..D].iter().copied().map(extension_words).collect())
            .collect(),
    )
}

fn claim_partial_result(claim: &Claim) -> PartialResult {
    let (commitment, public_input, _, eval_k, eval_a) = claim_result(claim);
    PartialResult {
        commitment,
        public_input,
        eval_k,
        eval_a,
    }
}

fn lean_partial_result(partial: &RawPartial) -> PartialResult {
    assert_eq!(partial.3.len(), MATRIX_COUNT, "Lean partial Eval_A families");
    PartialResult {
        commitment: partial.0.clone(),
        public_input: partial.1.clone(),
        eval_k: partial.2.clone(),
        eval_a: partial.3.clone(),
    }
}

fn engine_result(
    accepted: bool,
    rhos: &[RotRho],
    claim: &Claim,
    partials: &[Claim],
    transcript: &Poseidon2Transcript,
) -> PhaseResult {
    let challenges = challenge_words(rhos);
    let (commitment, public_input, point, eval_k, eval_a) = claim_result(claim);
    PhaseResult {
        accepted,
        membership: challenges
            .iter()
            .map(|words| strong_member(words))
            .collect(),
        challenges,
        commitment,
        public_input,
        point,
        eval_k,
        eval_a,
        partials: partials.iter().map(claim_partial_result).collect(),
        outgoing_state: state_words(transcript),
    }
}

fn lean_result(result: &RawResult) -> PhaseResult {
    assert_eq!(result.10, vec![1; 4], "Lean nonzero assurance flags");
    PhaseResult {
        accepted: result.0 == 1,
        challenges: result.1.clone(),
        membership: result.2.iter().map(|value| *value == 1).collect(),
        commitment: result.3.clone(),
        public_input: result.4.clone(),
        point: result.5.clone(),
        eval_k: result.6.clone(),
        eval_a: result.7.clone(),
        partials: result.8.iter().map(lean_partial_result).collect(),
        outgoing_state: result.9,
    }
}

fn paper_claim(structure: &CcsStructure<F>, params: &Params, rhos: &[RotRho], inputs: &[Claim]) -> Claim {
    let matrices = rhos
        .iter()
        .map(|rho| rho.as_mat().clone())
        .collect::<Vec<_>>();
    paper_exact_engine::rlc_claim_paper_exact_with_commit_mix(
        structure,
        params.inner(),
        &matrices,
        inputs,
        D.next_power_of_two().trailing_zeros() as usize,
        ajtai_rlc_mixer,
    )
}

fn optimized_claim(structure: &CcsStructure<F>, params: &Params, rhos: &[RotRho], inputs: &[Claim]) -> Claim {
    api::rlc_public(
        structure,
        params.inner(),
        rhos,
        inputs,
        ajtai_rlc_mixer,
        D.next_power_of_two().trailing_zeros() as usize,
    )
    .expect("optimized public PiRLC relation")
}

fn paper_prefix_claims(structure: &CcsStructure<F>, params: &Params, rhos: &[RotRho], inputs: &[Claim]) -> Vec<Claim> {
    (1..=SOURCE_COUNT)
        .map(|count| paper_claim(structure, params, &rhos[..count], &inputs[..count]))
        .collect()
}

fn optimized_prefix_claims(
    structure: &CcsStructure<F>,
    params: &Params,
    rhos: &[RotRho],
    inputs: &[Claim],
) -> Vec<Claim> {
    (1..=SOURCE_COUNT)
        .map(|count| optimized_claim(structure, params, &rhos[..count], &inputs[..count]))
        .collect()
}

fn assert_handoff_matches_pi_ccs() {
    let pi_rlc: serde_json::Value =
        serde_json::from_slice(&fs::read(artifact_path()).expect("PiRLC bytes")).expect("PiRLC JSON");
    let pi_ccs: serde_json::Value =
        serde_json::from_slice(&fs::read(pi_ccs_artifact_path()).expect("PiCCS bytes")).expect("PiCCS JSON");
    assert_eq!(pi_rlc[1][0], pi_ccs[2][14], "outgoing transcript handoff");
    assert_eq!(pi_rlc[1][1], pi_ccs[2][6], "shared point handoff");
    assert_eq!(pi_rlc[1][2], pi_ccs[2][10], "commitment handoff");
    assert_eq!(pi_rlc[1][3], pi_ccs[2][11], "public-input handoff");
    assert_eq!(pi_rlc[1][4], pi_ccs[2][12], "Eval_K handoff");
    assert_eq!(pi_rlc[1][5], pi_ccs[2][13], "Eval_A handoff");
}

#[test]
fn lean_paper_exact_and_optimized_match_complete_nonzero_pi_rlc_result() {
    assert_handoff_matches_pi_ccs();
    let Artifact(schema, input, result) = artifact();
    assert_eq!(schema, 2);
    assert_eq!(result.0, 1, "Lean PiRLC acceptance");
    assert_eq!(result.1.len(), SOURCE_COUNT);
    assert!(result.1.iter().all(|rho| rho.len() == D));
    assert_eq!(result.2, vec![1; SOURCE_COUNT]);
    assert_eq!(result.8.len(), SOURCE_COUNT, "Lean indexed partial count");

    let structure = relation();
    let params = Params::for_ccs_shape(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-bound Nightstream parameters");
    let inputs = claims(&input);
    let expected = expected_claim(&input, &result);

    let initial = input.0.map(field);
    let mut paper_transcript = Poseidon2Transcript::from_state_and_absorbed(initial, 0);
    let paper_rhos =
        paper_exact::sample_rho_n(&mut paper_transcript, &params, SOURCE_COUNT).expect("PaperExact PiRLC sampler");
    let mut optimized_transcript = Poseidon2Transcript::from_state_and_absorbed(initial, 0);
    let optimized_rhos =
        optimized::sample_rho_n(&mut optimized_transcript, &params, SOURCE_COUNT).expect("optimized PiRLC sampler");
    assert_eq!(paper_transcript.absorbed(), 0);
    assert_eq!(optimized_transcript.absorbed(), 0);
    assert_eq!(challenge_words(&paper_rhos), result.1);
    assert_eq!(challenge_words(&optimized_rhos), result.1);
    assert!(result.1.iter().all(|rho| strong_member(rho)));

    let paper_claim = paper_claim(&structure, &params, &paper_rhos, &inputs);
    let optimized_claim = optimized_claim(&structure, &params, &optimized_rhos, &inputs);
    let paper_partials = paper_prefix_claims(&structure, &params, &paper_rhos, &inputs);
    let optimized_partials = optimized_prefix_claims(&structure, &params, &optimized_rhos, &inputs);
    let lean_partials = result
        .8
        .iter()
        .map(|partial| expected_partial_claim(&input, partial))
        .collect::<Vec<_>>();
    assert_eq!(paper_partials, lean_partials, "PaperExact indexed claims equal Lean");
    assert_eq!(optimized_partials, lean_partials, "optimized indexed claims equal Lean");
    assert_eq!(paper_partials, optimized_partials, "independent indexed engines agree");
    assert_eq!(paper_partials.last(), Some(&paper_claim), "PaperExact final prefix");
    assert_eq!(
        optimized_partials.last(),
        Some(&optimized_claim),
        "optimized final prefix"
    );
    assert_eq!(paper_claim, expected, "PaperExact complete claim equals Lean");
    assert_eq!(optimized_claim, expected, "optimized complete claim equals Lean");
    assert!(paper_exact::verify_pi_rlc(
        &params,
        &structure,
        &paper_rhos,
        &inputs,
        &expected,
        ajtai_rlc_mixer,
    ));
    assert!(optimized::verify_pi_rlc(
        &params,
        &structure,
        &optimized_rhos,
        &inputs,
        &expected,
        ajtai_rlc_mixer,
    )
    .expect("optimized PiRLC verifier"));

    let lean = lean_result(&result);
    let paper = engine_result(true, &paper_rhos, &paper_claim, &paper_partials, &paper_transcript);
    let optimized = engine_result(
        true,
        &optimized_rhos,
        &optimized_claim,
        &optimized_partials,
        &optimized_transcript,
    );
    assert_eq!(paper, lean);
    assert_eq!(optimized, lean);
    assert_eq!(paper.canonical_bytes(), lean.canonical_bytes());
    assert_eq!(optimized.canonical_bytes(), lean.canonical_bytes());
}

fn assert_both_reject(
    structure: &CcsStructure<F>,
    params: &Params,
    rhos: &[RotRho],
    inputs: &[Claim],
    changed: &Claim,
) {
    assert!(!paper_exact::verify_pi_rlc(
        params,
        structure,
        rhos,
        inputs,
        changed,
        ajtai_rlc_mixer,
    ));
    assert!(!optimized::verify_pi_rlc(params, structure, rhos, inputs, changed, ajtai_rlc_mixer,).unwrap_or(false));
}

fn assert_mutated_source_detected(
    structure: &CcsStructure<F>,
    params: &Params,
    rhos: &[RotRho],
    changed_inputs: &[Claim],
    source: usize,
    lean_partial: &RawPartial,
) {
    let count = source + 1;
    let paper = paper_claim(structure, params, &rhos[..count], &changed_inputs[..count]);
    let optimized = optimized_claim(structure, params, &rhos[..count], &changed_inputs[..count]);
    assert_eq!(paper, optimized, "engines agree on mutated prefix {source}");
    let lean = lean_partial_result(lean_partial);
    assert_ne!(
        claim_partial_result(&paper),
        lean,
        "mutated source family changes prefix {source}"
    );
}

fn assert_mutated_partial_detected(changed: &RawPartial, paper: &Claim, optimized: &Claim, prefix: usize) {
    let changed = lean_partial_result(changed);
    assert_ne!(
        changed,
        claim_partial_result(paper),
        "PaperExact detects mutated partial {prefix}"
    );
    assert_ne!(
        changed,
        claim_partial_result(optimized),
        "optimized detects mutated partial {prefix}"
    );
}

fn bump_word(word: &mut u64) {
    *word = if *word + 1 == MODULUS { 0 } else { *word + 1 };
}

#[test]
fn both_engines_detect_every_indexed_pi_rlc_family_mutation() {
    let Artifact(schema, input, result) = artifact();
    assert_eq!(schema, 2);
    assert_eq!(result.8.len(), SOURCE_COUNT);
    let structure = relation();
    let params = Params::for_ccs_shape(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-bound Nightstream parameters");
    let inputs = claims(&input);
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(input.0.map(field), 0);
    let rhos = optimized::sample_rho_n(&mut transcript, &params, SOURCE_COUNT).expect("PiRLC sampler");
    let paper_partials = paper_prefix_claims(&structure, &params, &rhos, &inputs);
    let optimized_partials = optimized_prefix_claims(&structure, &params, &rhos, &inputs);

    for source in 0..SOURCE_COUNT {
        let mut changed = inputs.clone();
        changed[source].c.data[0] += F::ONE;
        assert_mutated_source_detected(&structure, &params, &rhos, &changed, source, &result.8[source]);

        let mut changed = inputs.clone();
        changed[source].X[(0, 0)] += F::ONE;
        assert_mutated_source_detected(&structure, &params, &rhos, &changed, source, &result.8[source]);

        let mut changed = inputs.clone();
        changed[source].eval_k[0] += K::ONE;
        assert_mutated_source_detected(&structure, &params, &rhos, &changed, source, &result.8[source]);

        for matrix in 0..MATRIX_COUNT {
            let mut changed = inputs.clone();
            changed[source].eval_a[matrix][0] += K::ONE;
            assert_mutated_source_detected(&structure, &params, &rhos, &changed, source, &result.8[source]);
        }
    }

    for prefix in 0..SOURCE_COUNT {
        let mut changed = result.8[prefix].clone();
        bump_word(&mut changed.0[0]);
        assert_mutated_partial_detected(&changed, &paper_partials[prefix], &optimized_partials[prefix], prefix);

        let mut changed = result.8[prefix].clone();
        bump_word(&mut changed.1[0]);
        assert_mutated_partial_detected(&changed, &paper_partials[prefix], &optimized_partials[prefix], prefix);

        let mut changed = result.8[prefix].clone();
        bump_word(&mut changed.2[0][0]);
        assert_mutated_partial_detected(&changed, &paper_partials[prefix], &optimized_partials[prefix], prefix);

        for matrix in 0..MATRIX_COUNT {
            let mut changed = result.8[prefix].clone();
            bump_word(&mut changed.3[matrix][0][0]);
            assert_mutated_partial_detected(&changed, &paper_partials[prefix], &optimized_partials[prefix], prefix);
        }
    }
}

#[test]
fn both_engines_reject_pi_rlc_value_and_transcript_mutations() {
    let Artifact(_, input, result) = artifact();
    let structure = relation();
    let params = Params::for_ccs_shape(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-bound Nightstream parameters");
    let inputs = claims(&input);
    let expected = expected_claim(&input, &result);
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(input.0.map(field), 0);
    let rhos = optimized::sample_rho_n(&mut transcript, &params, SOURCE_COUNT).expect("PiRLC sampler");

    let mut changed = expected.clone();
    changed.c.data[0] += F::ONE;
    assert_both_reject(&structure, &params, &rhos, &inputs, &changed);

    let mut changed = expected.clone();
    changed.X[(0, 0)] += F::ONE;
    assert_both_reject(&structure, &params, &rhos, &inputs, &changed);

    let mut changed = expected.clone();
    changed.r[0] += K::ONE;
    assert_both_reject(&structure, &params, &rhos, &inputs, &changed);

    let mut changed = expected.clone();
    changed.eval_k[0] += K::ONE;
    assert_both_reject(&structure, &params, &rhos, &inputs, &changed);

    for matrix in 0..MATRIX_COUNT {
        let mut changed = expected.clone();
        changed.eval_a[matrix][0] += K::ONE;
        assert_both_reject(&structure, &params, &rhos, &inputs, &changed);
    }

    let mut changed_inputs = inputs.clone();
    changed_inputs[0].eval_k[0] += K::ONE;
    assert_both_reject(&structure, &params, &rhos, &changed_inputs, &expected);

    let mut changed_state = input.0;
    changed_state[0] = if changed_state[0] + 1 == MODULUS {
        0
    } else {
        changed_state[0] + 1
    };
    let mut changed_transcript = Poseidon2Transcript::from_state_and_absorbed(changed_state.map(field), 0);
    let changed_rhos = optimized::sample_rho_n(&mut changed_transcript, &params, SOURCE_COUNT)
        .expect("mutated transcript still samples");
    assert_ne!(challenge_words(&changed_rhos), result.1);
    assert_both_reject(&structure, &params, &changed_rhos, &inputs, &expected);
    assert_ne!(state_words(&changed_transcript), result.9);
}
