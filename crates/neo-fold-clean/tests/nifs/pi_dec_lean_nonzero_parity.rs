//! Complete nonzero PiDEC parity against the Lean-emitted Stage 1 fixture.

use std::{fs, path::PathBuf};

use neo_ajtai::{scale_commitment_add_inplace, Commitment};
use neo_ccs::{CcsStructure, CeClaim, Mat, SparsePoly, Term};
use neo_fold_clean::{
    engine::{optimized, paper_exact},
    frontends::r1cs_f_prime::production::pi_ccs_v1_1_state_hash,
    paper::{params::Params, relations::ajtai_dec_mixer},
};
use neo_math::{from_complex, KExtensions, D, F, K};
use neo_reductions::split_b_matrix_k;
use nightstream_fprime::{
    PI_CCS_V1_1_ROUND_COUNT, PI_CCS_V1_1_STATE_PREIMAGE_WORDS, POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const MODULUS: u64 = 0xffff_ffff_0000_0001;
const CHILD_COUNT: usize = 16;
const MATRIX_COUNT: usize = 14;
const COEFFICIENT_COUNT: usize = 54;
const PUBLIC_INPUT_WORDS: usize = 270;
const COMMITMENT_WORDS: usize = 1_188;
const COMBINED_BOUND: u64 = 1 << 16;
type Claim = CeClaim<Commitment, F, K>;

#[derive(Deserialize)]
struct Artifact(u64, RawInput, RawResult);

#[derive(Clone, Deserialize)]
struct RawClaim(
    Vec<u64>,
    Vec<u64>,
    Vec<[u64; 2]>,
    Vec<[u64; 2]>,
    Vec<Vec<[u64; 2]>>,
    u64,
);

#[derive(Clone, Deserialize)]
struct RawInput(
    RawClaim,
    Vec<Vec<u64>>,
    Vec<Vec<[u64; 2]>>,
    Vec<Vec<Vec<[u64; 2]>>>,
    Vec<Vec<u64>>,
    [u64; 8],
    [u64; 4],
);

#[derive(Clone, Deserialize)]
struct RawResult(
    u64,
    u64,
    Vec<Vec<u64>>,
    Vec<u64>,
    Vec<Vec<u64>>,
    Vec<u64>,
    u64,
    Vec<u64>,
    u64,
    Vec<[u64; 2]>,
    u64,
    Vec<Vec<[u64; 2]>>,
    u64,
    Vec<RawClaim>,
    [u64; 8],
    u64,
    Vec<u64>,
    Vec<u64>,
    [u64; 4],
);

#[derive(Deserialize)]
struct RawRelation(u64, u64, u64, Vec<u64>, u64, Vec<RawTerm>);

#[derive(Deserialize)]
struct RawTerm(u64, Vec<u64>);

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct ClaimResult {
    commitment: Vec<u64>,
    public_input: Vec<u64>,
    point: Vec<[u64; 2]>,
    eval_k: Vec<[u64; 2]>,
    eval_a: Vec<Vec<[u64; 2]>>,
    stage: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct PhaseResult {
    accepted: bool,
    parent_bounded: bool,
    digits: Vec<Vec<u64>>,
    parent_bound_results: Vec<bool>,
    digit_range_results: Vec<Vec<bool>>,
    recomposed_commitment: Vec<u64>,
    commitment_equation: bool,
    recomposed_public_input: Vec<u64>,
    public_input_equation: bool,
    recomposed_eval_k: Vec<[u64; 2]>,
    eval_k_equation: bool,
    recomposed_eval_a: Vec<Vec<[u64; 2]>>,
    eval_a_equation: bool,
    children: Vec<ClaimResult>,
    outgoing_state: [u64; 8],
    unbounded_rejected: bool,
}

impl PhaseResult {
    fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("canonical PiDEC phase-result JSON")
    }
}

fn artifact_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-pidec-parity-v1.json")
}

fn pi_ccs_artifact_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-piccs-parity-v1.json")
}

fn pi_rlc_artifact_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-pirlc-parity-v1.json")
}

fn package_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")
}

fn artifact() -> Artifact {
    serde_json::from_slice(&fs::read(artifact_path()).expect("Lean PiDEC parity bytes"))
        .expect("Lean PiDEC parity JSON")
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

fn fields(words: &[u64]) -> Vec<F> {
    words.iter().copied().map(field).collect()
}

fn commitment(words: &[u64]) -> Commitment {
    assert_eq!(words.len(), COMMITMENT_WORDS, "PiDEC commitment width");
    Commitment {
        d: D,
        kappa: words.len() / D,
        data: fields(words),
    }
}

fn commitment_words(value: &Commitment) -> Vec<u64> {
    value
        .data
        .iter()
        .map(|word| word.as_canonical_u64())
        .collect()
}

fn public_input(words: &[u64]) -> Mat<F> {
    assert_eq!(words.len(), PUBLIC_INPUT_WORDS, "PiDEC public-input width");
    let mut output = Mat::zero(D, PUBLIC_INPUT_WORDS / D, F::ZERO);
    for (index, word) in words.iter().copied().enumerate() {
        output[(index % D, index / D)] = field(word);
    }
    output
}

fn public_input_words(value: &Mat<F>) -> Vec<u64> {
    assert_eq!(value.rows(), D);
    assert_eq!(value.cols(), PUBLIC_INPUT_WORDS / D);
    (0..PUBLIC_INPUT_WORDS)
        .map(|index| value[(index % D, index / D)].as_canonical_u64())
        .collect()
}

fn padded_family(values: &[[u64; 2]]) -> Vec<K> {
    assert_eq!(values.len(), COEFFICIENT_COUNT, "PiDEC evaluation width");
    let mut output = values.iter().copied().map(extension).collect::<Vec<_>>();
    output.resize(D.next_power_of_two(), K::ZERO);
    output
}

fn fold_digest(state: [u64; 8]) -> [u8; 32] {
    let mut output = [0; 32];
    for (lane, word) in state[..4].iter().enumerate() {
        output[lane * 8..(lane + 1) * 8].copy_from_slice(&word.to_le_bytes());
    }
    output
}

fn claim(
    commitment_words: &[u64],
    public_words: &[u64],
    point: &[[u64; 2]],
    eval_k: &[[u64; 2]],
    eval_a: &[Vec<[u64; 2]>],
    state: [u64; 8],
) -> Claim {
    assert_eq!(eval_a.len(), MATRIX_COUNT, "PiDEC Eval_A matrix count");
    Claim {
        c: commitment(commitment_words),
        X: public_input(public_words),
        r: point.iter().copied().map(extension).collect(),
        eval_k: padded_family(eval_k),
        eval_a: eval_a.iter().map(|family| padded_family(family)).collect(),
        m_in: PUBLIC_INPUT_WORDS,
        fold_digest: fold_digest(state),
        adv: None,
    }
}

fn claim_from_raw(raw: &RawClaim, state: [u64; 8]) -> Claim {
    claim(&raw.0, &raw.1, &raw.2, &raw.3, &raw.4, state)
}

fn claims(input: &RawInput) -> (Claim, Vec<Claim>) {
    assert_eq!(input.1.len(), CHILD_COUNT);
    assert_eq!(input.2.len(), CHILD_COUNT);
    assert_eq!(input.3.len(), CHILD_COUNT);
    assert_eq!(input.4.len(), CHILD_COUNT);
    let parent = claim_from_raw(&input.0, input.5);
    let children = (0..CHILD_COUNT)
        .map(|child| {
            claim(
                &input.1[child],
                &input.4[child],
                &input.0 .2,
                &input.2[child],
                &input.3[child],
                input.5,
            )
        })
        .collect();
    (parent, children)
}

fn relation() -> CcsStructure<F> {
    let bytes = fs::read(package_path()).expect("Lean package bytes");
    let loaded =
        nightstream_fprime::load_poseidon2_hash_chain_v1_package(&bytes).expect("verifier-owned production package");
    assert_eq!(
        loaded
            .production_verifier_binding()
            .expect("fixed production binding")
            .package_identity(),
        POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY
    );
    let package: serde_json::Value = serde_json::from_slice(&bytes).expect("Lean package JSON");
    assert_eq!(package[1][0].as_u64(), Some(8), "Lean inner-package schema");
    let raw: RawRelation = serde_json::from_value(package[1][4].clone()).expect("Lean relation tuple");
    assert_eq!(raw.2, PI_CCS_V1_1_ROUND_COUNT as u64);
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

fn params(structure: &CcsStructure<F>) -> Params {
    let params = Params::for_ccs_shape(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-bound Nightstream parameters");
    assert_eq!(params.b(), 2);
    assert_eq!(params.k_rho(), CHILD_COUNT as u32);
    params
}

fn centered(word: u64) -> i64 {
    if word <= (MODULUS - 1) / 2 {
        i64::try_from(word).expect("positive centered word")
    } else {
        -i64::try_from(MODULUS - word).expect("negative centered word")
    }
}

fn centered_magnitude(word: u64) -> u64 {
    centered(word).unsigned_abs()
}

fn signed_word(value: i64) -> u64 {
    if value >= 0 {
        value as u64
    } else {
        MODULUS - value.unsigned_abs()
    }
}

fn paper_digits(parent: &[u64]) -> Option<Vec<Vec<u64>>> {
    let mut output = vec![vec![0; parent.len()]; CHILD_COUNT];
    for (coordinate, word) in parent.iter().copied().enumerate() {
        let mut remaining = centered(word);
        for child in &mut output {
            let digit = remaining % 2;
            child[coordinate] = signed_word(digit);
            remaining = (remaining - digit) / 2;
        }
        if remaining != 0 {
            return None;
        }
    }
    Some(output)
}

fn optimized_digits(parent: &Mat<F>) -> Option<Vec<Vec<u64>>> {
    split_b_matrix_k(parent, CHILD_COUNT, 2)
        .ok()
        .map(|children| children.iter().map(public_input_words).collect())
}

fn powers_f() -> Vec<F> {
    let mut power = F::ONE;
    (0..CHILD_COUNT)
        .map(|_| {
            let current = power;
            power *= F::from_u64(2);
            current
        })
        .collect()
}

fn powers_k() -> Vec<K> {
    let mut power = K::ONE;
    (0..CHILD_COUNT)
        .map(|_| {
            let current = power;
            power *= K::from(F::from_u64(2));
            current
        })
        .collect()
}

fn paper_commitment(children: &[Claim]) -> Commitment {
    let mut output = Commitment::zeros(children[0].c.d, children[0].c.kappa);
    let mut power = F::ONE;
    for child in children {
        scale_commitment_add_inplace(&mut output, power, &child.c);
        power *= F::from_u64(2);
    }
    output
}

fn optimized_commitment(children: &[Claim]) -> Commitment {
    ajtai_dec_mixer(
        &children
            .iter()
            .map(|child| child.c.clone())
            .collect::<Vec<_>>(),
        2,
    )
}

fn paper_public_input(children: &[Claim]) -> Vec<u64> {
    let mut output = vec![F::ZERO; PUBLIC_INPUT_WORDS];
    let mut power = F::ONE;
    for child in children {
        let words = public_input_words(&child.X);
        for (target, word) in output.iter_mut().zip(words) {
            *target += power * field(word);
        }
        power *= F::from_u64(2);
    }
    output
        .into_iter()
        .map(|word| word.as_canonical_u64())
        .collect()
}

fn optimized_public_input(children: &[Claim]) -> Vec<u64> {
    let powers = powers_f();
    (0..PUBLIC_INPUT_WORDS)
        .into_par_iter()
        .map(|coordinate| {
            children
                .iter()
                .zip(&powers)
                .fold(F::ZERO, |sum, (child, power)| {
                    sum + *power * child.X[(coordinate % D, coordinate / D)]
                })
                .as_canonical_u64()
        })
        .collect()
}

fn paper_eval_k(children: &[Claim]) -> Vec<K> {
    let mut output = vec![K::ZERO; D.next_power_of_two()];
    let mut power = K::ONE;
    for child in children {
        for (target, value) in output.iter_mut().zip(&child.eval_k) {
            *target += power * *value;
        }
        power *= K::from(F::from_u64(2));
    }
    output
}

fn optimized_eval_k(children: &[Claim]) -> Vec<K> {
    let powers = powers_k();
    (0..D.next_power_of_two())
        .into_par_iter()
        .map(|coefficient| {
            children
                .iter()
                .zip(&powers)
                .fold(K::ZERO, |sum, (child, power)| sum + *power * child.eval_k[coefficient])
        })
        .collect()
}

fn paper_eval_a(children: &[Claim]) -> Vec<Vec<K>> {
    (0..MATRIX_COUNT)
        .map(|matrix| {
            let mut output = vec![K::ZERO; D.next_power_of_two()];
            let mut power = K::ONE;
            for child in children {
                for (target, value) in output.iter_mut().zip(&child.eval_a[matrix]) {
                    *target += power * *value;
                }
                power *= K::from(F::from_u64(2));
            }
            output
        })
        .collect()
}

fn optimized_eval_a(children: &[Claim]) -> Vec<Vec<K>> {
    let powers = powers_k();
    (0..MATRIX_COUNT)
        .into_par_iter()
        .map(|matrix| {
            (0..D.next_power_of_two())
                .map(|coefficient| {
                    children
                        .iter()
                        .zip(&powers)
                        .fold(K::ZERO, |sum, (child, power)| {
                            sum + *power * child.eval_a[matrix][coefficient]
                        })
                })
                .collect()
        })
        .collect()
}

fn claim_result(claim: &Claim, stage: u64) -> ClaimResult {
    ClaimResult {
        commitment: commitment_words(&claim.c),
        public_input: public_input_words(&claim.X),
        point: claim.r.iter().copied().map(extension_words).collect(),
        eval_k: claim.eval_k[..COEFFICIENT_COUNT]
            .iter()
            .copied()
            .map(extension_words)
            .collect(),
        eval_a: claim
            .eval_a
            .iter()
            .map(|family| {
                family[..COEFFICIENT_COUNT]
                    .iter()
                    .copied()
                    .map(extension_words)
                    .collect()
            })
            .collect(),
        stage,
    }
}

fn raw_claim_result(claim: &RawClaim) -> ClaimResult {
    ClaimResult {
        commitment: claim.0.clone(),
        public_input: claim.1.clone(),
        point: claim.2.clone(),
        eval_k: claim.3.clone(),
        eval_a: claim.4.clone(),
        stage: claim.5,
    }
}

fn finish_result(
    parent: &Claim,
    children: &[Claim],
    state: [u64; 8],
    digits: Vec<Vec<u64>>,
    recomposed_commitment: Commitment,
    recomposed_public_input: Vec<u64>,
    recomposed_eval_k: Vec<K>,
    recomposed_eval_a: Vec<Vec<K>>,
    accepted: bool,
    unbounded_rejected: bool,
) -> PhaseResult {
    let parent_words = public_input_words(&parent.X);
    let parent_bound_results = parent_words
        .iter()
        .map(|word| centered_magnitude(*word) < COMBINED_BOUND)
        .collect::<Vec<_>>();
    let digit_range_results = digits
        .iter()
        .map(|child| {
            child
                .iter()
                .map(|word| centered_magnitude(*word) < 2)
                .collect()
        })
        .collect();
    let recomposed_commitment = commitment_words(&recomposed_commitment);
    let recomposed_eval_k = recomposed_eval_k[..COEFFICIENT_COUNT]
        .iter()
        .copied()
        .map(extension_words)
        .collect::<Vec<_>>();
    let recomposed_eval_a = recomposed_eval_a
        .iter()
        .map(|family| {
            family[..COEFFICIENT_COUNT]
                .iter()
                .copied()
                .map(extension_words)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let parent_eval_k = parent.eval_k[..COEFFICIENT_COUNT]
        .iter()
        .copied()
        .map(extension_words)
        .collect::<Vec<_>>();
    let parent_eval_a = parent
        .eval_a
        .iter()
        .map(|family| {
            family[..COEFFICIENT_COUNT]
                .iter()
                .copied()
                .map(extension_words)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    PhaseResult {
        accepted,
        parent_bounded: parent_bound_results.iter().all(|result| *result),
        digits,
        parent_bound_results,
        digit_range_results,
        commitment_equation: recomposed_commitment == commitment_words(&parent.c),
        recomposed_commitment,
        public_input_equation: recomposed_public_input == parent_words,
        recomposed_public_input,
        eval_k_equation: recomposed_eval_k == parent_eval_k,
        recomposed_eval_k,
        eval_a_equation: recomposed_eval_a == parent_eval_a,
        recomposed_eval_a,
        children: children
            .iter()
            .map(|child| claim_result(child, 0))
            .collect(),
        outgoing_state: state,
        unbounded_rejected,
    }
}

fn unbounded_parent(parent: &Claim) -> Claim {
    let mut output = parent.clone();
    output.X[(0, 0)] = F::from_u64(COMBINED_BOUND);
    output
}

fn paper_result(params: &Params, parent: &Claim, children: &[Claim], state: [u64; 8]) -> PhaseResult {
    let accepted = paper_exact::verify_pi_dec(params, parent, children, ajtai_dec_mixer);
    let unbounded = unbounded_parent(parent);
    let unbounded_words = public_input_words(&unbounded.X);
    let unbounded_rejected = paper_digits(&unbounded_words).is_none()
        && !paper_exact::verify_pi_dec(params, &unbounded, children, ajtai_dec_mixer);
    finish_result(
        parent,
        children,
        state,
        paper_digits(&public_input_words(&parent.X)).expect("PaperExact bounded parent"),
        paper_commitment(children),
        paper_public_input(children),
        paper_eval_k(children),
        paper_eval_a(children),
        accepted,
        unbounded_rejected,
    )
}

fn optimized_result(
    params: &Params,
    structure: &CcsStructure<F>,
    parent: &Claim,
    children: &[Claim],
    state: [u64; 8],
) -> PhaseResult {
    let accepted = optimized::verify_pi_dec(params, structure, parent, children, ajtai_dec_mixer);
    let unbounded = unbounded_parent(parent);
    let unbounded_rejected = optimized_digits(&unbounded.X).is_none()
        && !optimized::verify_pi_dec(params, structure, &unbounded, children, ajtai_dec_mixer);
    finish_result(
        parent,
        children,
        state,
        optimized_digits(&parent.X).expect("optimized bounded parent"),
        optimized_commitment(children),
        optimized_public_input(children),
        optimized_eval_k(children),
        optimized_eval_a(children),
        accepted,
        unbounded_rejected,
    )
}

fn lean_result(result: &RawResult) -> PhaseResult {
    assert_eq!(result.16, vec![1; 6], "Lean PiDEC nonzero assurance");
    PhaseResult {
        accepted: result.0 == 1,
        parent_bounded: result.1 == 1,
        digits: result.2.clone(),
        parent_bound_results: result.3.iter().map(|value| *value == 1).collect(),
        digit_range_results: result
            .4
            .iter()
            .map(|child| child.iter().map(|value| *value == 1).collect())
            .collect(),
        recomposed_commitment: result.5.clone(),
        commitment_equation: result.6 == 1,
        recomposed_public_input: result.7.clone(),
        public_input_equation: result.8 == 1,
        recomposed_eval_k: result.9.clone(),
        eval_k_equation: result.10 == 1,
        recomposed_eval_a: result.11.clone(),
        eval_a_equation: result.12 == 1,
        children: result.13.iter().map(raw_claim_result).collect(),
        outgoing_state: result.14,
        unbounded_rejected: result.15 == 1,
    }
}

fn assert_cumulative_handoff() {
    let pi_ccs: serde_json::Value =
        serde_json::from_slice(&fs::read(pi_ccs_artifact_path()).expect("Lean PiCCS parity bytes"))
            .expect("Lean PiCCS parity JSON");
    let pi_rlc: serde_json::Value =
        serde_json::from_slice(&fs::read(pi_rlc_artifact_path()).expect("Lean PiRLC parity bytes"))
            .expect("Lean PiRLC parity JSON");
    let pi_dec: serde_json::Value =
        serde_json::from_slice(&fs::read(artifact_path()).expect("Lean PiDEC parity bytes"))
            .expect("Lean PiDEC parity JSON");
    assert_eq!(pi_rlc[1][0], pi_ccs[2][14], "PiCCS outgoing-state handoff");
    assert_eq!(pi_rlc[1][1], pi_ccs[2][6], "PiCCS point handoff");
    assert_eq!(pi_rlc[1][2], pi_ccs[2][10], "PiCCS commitment handoff");
    assert_eq!(pi_rlc[1][3], pi_ccs[2][11], "PiCCS public-input handoff");
    assert_eq!(pi_rlc[1][4], pi_ccs[2][12], "PiCCS Eval_K handoff");
    assert_eq!(pi_rlc[1][5], pi_ccs[2][13], "PiCCS Eval_A handoff");
    assert_eq!(pi_dec[1][0][0], pi_rlc[2][3], "PiRLC commitment handoff");
    assert_eq!(pi_dec[1][0][1], pi_rlc[2][4], "PiRLC public-input handoff");
    assert_eq!(pi_dec[1][0][2], pi_rlc[2][5], "PiRLC point handoff");
    assert_eq!(pi_dec[1][0][3], pi_rlc[2][6], "PiRLC Eval_K handoff");
    assert_eq!(pi_dec[1][0][4], pi_rlc[2][7], "PiRLC Eval_A handoff");
    assert_eq!(pi_dec[1][5], pi_rlc[2][9], "PiRLC transcript-state handoff");
}

fn transition_running_words(children: &[RawClaim]) -> Vec<u64> {
    let mut words = Vec::new();
    words.push((children[0].2.len() * 2) as u64);
    for value in &children[0].2 {
        words.extend_from_slice(value);
    }
    for child in children {
        words.push(child.0.len() as u64);
        words.extend_from_slice(&child.0);
        words.push(child.1.len() as u64);
        words.extend_from_slice(&child.1);
        let evaluation_words = child.3.len() * 2 + child.4.iter().map(|matrix| matrix.len() * 2).sum::<usize>();
        words.push(evaluation_words as u64);
        for value in &child.3 {
            words.extend_from_slice(value);
        }
        for matrix in &child.4 {
            for value in matrix {
                words.extend_from_slice(value);
            }
        }
    }
    words
}

fn assert_both_reject(
    params: &Params,
    structure: &CcsStructure<F>,
    parent: &Claim,
    children: &[Claim],
    location: &str,
) {
    assert!(
        !paper_exact::verify_pi_dec(params, parent, children, ajtai_dec_mixer),
        "PaperExact accepted mutated {location}"
    );
    assert!(
        !optimized::verify_pi_dec(params, structure, parent, children, ajtai_dec_mixer),
        "optimized accepted mutated {location}"
    );
}

fn bump_word(word: &mut u64) {
    *word = if *word + 1 == MODULUS { 0 } else { *word + 1 };
}

fn assert_changed(changed: PhaseResult, paper: &PhaseResult, optimized: &PhaseResult, location: &str) {
    assert_ne!(&changed, paper, "PaperExact missed changed {location}");
    assert_ne!(&changed, optimized, "optimized missed changed {location}");
}

#[test]
fn lean_paper_exact_and_optimized_match_complete_nonzero_pi_dec_result() {
    assert_cumulative_handoff();
    let Artifact(schema, input, result) = artifact();
    assert_eq!(schema, 2);
    assert_eq!(input.6, POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY);
    assert_eq!(input.0 .5, 1, "PiDEC parent stage");
    assert_eq!(result.0, 1, "Lean PiDEC acceptance");
    assert_eq!(result.13.len(), CHILD_COUNT);
    assert!(result.13.iter().all(|child| child.5 == 0));
    assert_eq!(result.17.len(), PI_CCS_V1_1_STATE_PREIMAGE_WORDS);
    let running = transition_running_words(&result.13);
    assert_eq!(&result.17[39..39 + running.len()], running.as_slice());
    assert_eq!(result.17.last().copied(), Some(1));
    assert_eq!(
        pi_ccs_v1_1_state_hash(&result.17).expect("transition output hash"),
        result.18
    );

    let structure = relation();
    let params = params(&structure);
    let (parent, children) = claims(&input);
    let lean = lean_result(&result);
    let paper = paper_result(&params, &parent, &children, input.5);
    let optimized = optimized_result(&params, &structure, &parent, &children, input.5);

    assert_eq!(paper, lean);
    assert_eq!(optimized, lean);
    assert_eq!(paper.canonical_bytes(), lean.canonical_bytes());
    assert_eq!(optimized.canonical_bytes(), lean.canonical_bytes());
}

#[test]
fn both_engines_reject_every_indexed_pi_dec_input_family_mutation() {
    let Artifact(_, input, _) = artifact();
    let structure = relation();
    let params = params(&structure);
    let (parent, children) = claims(&input);

    for child in 0..CHILD_COUNT {
        let mut changed = children.clone();
        changed[child].c.data[0] += F::ONE;
        assert_both_reject(
            &params,
            &structure,
            &parent,
            &changed,
            &format!("child {child} commitment"),
        );

        let mut changed = children.clone();
        changed[child].X[(0, 0)] += F::ONE;
        assert_both_reject(
            &params,
            &structure,
            &parent,
            &changed,
            &format!("child {child} public input"),
        );

        let mut changed = children.clone();
        changed[child].eval_k[0] += K::ONE;
        assert_both_reject(&params, &structure, &parent, &changed, &format!("child {child} Eval_K"));

        for matrix in 0..MATRIX_COUNT {
            let mut changed = children.clone();
            changed[child].eval_a[matrix][0] += K::ONE;
            assert_both_reject(
                &params,
                &structure,
                &parent,
                &changed,
                &format!("child {child} Eval_A matrix {matrix}"),
            );
        }

        let mut changed = children.clone();
        changed[child].r[0] += K::ONE;
        assert_both_reject(&params, &structure, &parent, &changed, &format!("child {child} point"));

        let mut changed = children.clone();
        changed[child].fold_digest[0] ^= 1;
        assert_both_reject(&params, &structure, &parent, &changed, &format!("child {child} state"));
    }

    let mut changed = parent.clone();
    changed.c.data[0] += F::ONE;
    assert_both_reject(&params, &structure, &changed, &children, "parent commitment");

    let mut changed = parent.clone();
    changed.X[(0, 0)] += F::ONE;
    assert_both_reject(&params, &structure, &changed, &children, "parent public input");

    let mut changed = parent.clone();
    changed.eval_k[0] += K::ONE;
    assert_both_reject(&params, &structure, &changed, &children, "parent Eval_K");

    for matrix in 0..MATRIX_COUNT {
        let mut changed = parent.clone();
        changed.eval_a[matrix][0] += K::ONE;
        assert_both_reject(
            &params,
            &structure,
            &changed,
            &children,
            &format!("parent Eval_A matrix {matrix}"),
        );
    }

    let mut changed = parent.clone();
    changed.r[0] += K::ONE;
    assert_both_reject(&params, &structure, &changed, &children, "parent point");

    let mut changed = parent.clone();
    changed.fold_digest[0] ^= 1;
    assert_both_reject(&params, &structure, &changed, &children, "parent state");

    let changed = unbounded_parent(&parent);
    assert!(paper_digits(&public_input_words(&changed.X)).is_none());
    assert!(optimized_digits(&changed.X).is_none());
    assert_both_reject(&params, &structure, &changed, &children, "unbounded parent");
}

#[test]
fn complete_pi_dec_result_comparator_detects_every_output_family_mutation() {
    let Artifact(_, input, result) = artifact();
    let structure = relation();
    let params = params(&structure);
    let (parent, children) = claims(&input);
    let paper = paper_result(&params, &parent, &children, input.5);
    let optimized = optimized_result(&params, &structure, &parent, &children, input.5);
    let lean = lean_result(&result);
    assert_eq!(paper, lean);
    assert_eq!(optimized, lean);

    let mut changed = lean.clone();
    changed.accepted = !changed.accepted;
    assert_changed(changed, &paper, &optimized, "acceptance");

    let mut changed = lean.clone();
    changed.parent_bounded = !changed.parent_bounded;
    assert_changed(changed, &paper, &optimized, "parent bound");

    let mut changed = lean.clone();
    changed.parent_bound_results[0] = !changed.parent_bound_results[0];
    assert_changed(changed, &paper, &optimized, "parent-bound result");

    for child in 0..CHILD_COUNT {
        let mut changed = lean.clone();
        bump_word(&mut changed.digits[child][0]);
        assert_changed(changed, &paper, &optimized, &format!("child {child} digit"));

        let mut changed = lean.clone();
        changed.digit_range_results[child][0] = !changed.digit_range_results[child][0];
        assert_changed(changed, &paper, &optimized, &format!("child {child} range"));

        let mut changed = lean.clone();
        bump_word(&mut changed.children[child].commitment[0]);
        assert_changed(changed, &paper, &optimized, &format!("child {child} commitment result"));

        let mut changed = lean.clone();
        bump_word(&mut changed.children[child].public_input[0]);
        assert_changed(changed, &paper, &optimized, &format!("child {child} public result"));

        let mut changed = lean.clone();
        bump_word(&mut changed.children[child].point[0][0]);
        assert_changed(changed, &paper, &optimized, &format!("child {child} point result"));

        let mut changed = lean.clone();
        bump_word(&mut changed.children[child].eval_k[0][0]);
        assert_changed(changed, &paper, &optimized, &format!("child {child} Eval_K result"));

        for matrix in 0..MATRIX_COUNT {
            let mut changed = lean.clone();
            bump_word(&mut changed.children[child].eval_a[matrix][0][0]);
            assert_changed(
                changed,
                &paper,
                &optimized,
                &format!("child {child} Eval_A matrix {matrix} result"),
            );
        }

        let mut changed = lean.clone();
        changed.children[child].stage ^= 1;
        assert_changed(changed, &paper, &optimized, &format!("child {child} stage"));
    }

    let mut changed = lean.clone();
    bump_word(&mut changed.recomposed_commitment[0]);
    assert_changed(changed, &paper, &optimized, "recomposed commitment");

    let mut changed = lean.clone();
    changed.commitment_equation = !changed.commitment_equation;
    assert_changed(changed, &paper, &optimized, "commitment equation");

    let mut changed = lean.clone();
    bump_word(&mut changed.recomposed_public_input[0]);
    assert_changed(changed, &paper, &optimized, "recomposed public input");

    let mut changed = lean.clone();
    changed.public_input_equation = !changed.public_input_equation;
    assert_changed(changed, &paper, &optimized, "public-input equation");

    let mut changed = lean.clone();
    bump_word(&mut changed.recomposed_eval_k[0][0]);
    assert_changed(changed, &paper, &optimized, "recomposed Eval_K");

    let mut changed = lean.clone();
    changed.eval_k_equation = !changed.eval_k_equation;
    assert_changed(changed, &paper, &optimized, "Eval_K equation");

    for matrix in 0..MATRIX_COUNT {
        let mut changed = lean.clone();
        bump_word(&mut changed.recomposed_eval_a[matrix][0][0]);
        assert_changed(
            changed,
            &paper,
            &optimized,
            &format!("recomposed Eval_A matrix {matrix}"),
        );
    }

    let mut changed = lean.clone();
    changed.eval_a_equation = !changed.eval_a_equation;
    assert_changed(changed, &paper, &optimized, "Eval_A equation");

    let mut changed = lean.clone();
    bump_word(&mut changed.outgoing_state[0]);
    assert_changed(changed, &paper, &optimized, "outgoing state");

    let mut changed = lean;
    changed.unbounded_rejected = !changed.unbounded_rejected;
    assert_changed(changed, &paper, &optimized, "unbounded rejection");
}
