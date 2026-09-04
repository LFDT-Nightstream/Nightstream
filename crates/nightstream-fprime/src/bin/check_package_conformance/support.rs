use std::{fs, path::Path};

use nightstream_fprime::{
    load_with_expanded_package, LoadedPackage, PackageSparseMatrix, PiCcsV1_1OutputEvaluations, PiDecV1_1PackageInputs,
    PI_CCS_V1_1_COEFFICIENT_COUNT, PI_CCS_V1_1_FRESH_COMMITMENT_WORDS, PI_CCS_V1_1_MATRIX_COUNT,
    PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_1_ROUND_COUNT,
    PI_CCS_V1_1_SOURCE_COUNT, PI_CCS_V1_1_STATE_PREIMAGE_WORDS, PI_DEC_V1_1_CHILD_COUNT,
    PI_DEC_V1_1_COMMITMENT_WORDS_PER_CHILD, PI_DEC_V1_1_EVAL_A_MATRICES_PER_CHILD, PI_DEC_V1_1_EVAL_K_VALUES_PER_CHILD,
    POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY,
};
use rayon::prelude::*;
use serde::de::IgnoredAny;
use serde::Deserialize;
use serde_json::Value;

#[allow(dead_code)]
mod canonical_assignment;
#[allow(dead_code)]
mod independent_assignment;
mod owner_mutations;
#[allow(dead_code)]
mod raw_assignment;
mod reference;
#[allow(unused_imports)]
pub use canonical_assignment::{evaluate_canonical_assignment, evaluate_pi_ccs_prefix_assignment};
#[allow(unused_imports)]
pub use independent_assignment::{
    check_piccs_owner_mutations, compare_lean_expanded_matrices, compare_sealed_matrices,
};
use reference::*;

const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;

#[derive(Deserialize)]
struct RawPackage(
    u64,
    IgnoredAny,
    RawPoseidonSchedule,
    RawLayout,
    IgnoredAny,
    RawTemplate,
    Vec<RawChain>,
    Vec<RawInvocation>,
    Vec<RawCompactTemplate>,
    Vec<RawCompactInvocation>,
    IgnoredAny,
    Vec<RawInstruction>,
    Vec<RawRow>,
    IgnoredAny,
);

#[derive(Deserialize)]
struct RawPoseidonSchedule(
    IgnoredAny,
    u64,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
);

#[derive(Deserialize)]
struct RawLayout(u64, u64, u64, u64, u64, IgnoredAny, Vec<RawSegment>);

#[derive(Deserialize)]
struct RawSegment(u64, u64, u64);

#[derive(Deserialize)]
struct RawTemplate(u64, u64, u64, Vec<RawTemplateRow>);

#[derive(Deserialize)]
struct RawTemplateRow(
    u64,
    RawTemplateCombination,
    RawTemplateCombination,
    RawTemplateCombination,
);

#[derive(Deserialize)]
struct RawTemplateCombination(u64, Vec<RawTemplateTerm>);

#[derive(Deserialize)]
struct RawTemplateTerm(RawColumnRef, u64);

#[derive(Deserialize)]
struct RawColumnRef(u64, u64);

#[derive(Deserialize)]
struct RawChain(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

#[derive(Deserialize)]
struct RawInvocation(u64, u64, u64, Vec<RawCombination>);

#[derive(Deserialize)]
struct RawCompactTemplate(u64, u64, u64, IgnoredAny, Vec<RawCompactRow>);

#[derive(Deserialize)]
struct RawCompactRow(
    IgnoredAny,
    RawTemplateCombination,
    RawTemplateCombination,
    RawTemplateCombination,
);

#[derive(Deserialize)]
struct RawCompactRange(u64, u64, u64, u64);

#[derive(Deserialize)]
struct RawCompactInvocation(u64, u64, u64, u64, Vec<RawCompactRange>);

#[derive(Deserialize)]
struct RawInstruction(u64, u64, RawCombination, RawCombination);

#[derive(Deserialize)]
struct RawRow(u64, RawCombination, RawCombination, RawCombination);

#[derive(Deserialize)]
struct RawCombination(u64, Vec<RawTerm>);

#[derive(Deserialize)]
struct RawTerm(u64, u64);

#[derive(Clone, Copy)]
enum MatrixSide {
    A,
    B,
    C,
}

#[derive(Clone, Copy, Debug)]
struct OwnerSpan {
    name: &'static str,
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug)]
struct ColumnOwnerSpan {
    name: &'static str,
    rows: OwnerSpan,
    columns: OwnerSpan,
}

#[derive(Debug)]
struct OwnershipInventory {
    row_spans: Vec<OwnerSpan>,
    column_spans: Vec<OwnerSpan>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PiCcsOwnerMutationReport {
    pub row_families: usize,
    pub row_mutations: usize,
    pub column_families: usize,
    pub zero_column_families: usize,
    pub column_mutations: usize,
    pub public_segments: usize,
    pub public_mutations: usize,
}

#[derive(Clone, Copy)]
enum Invocation<'a> {
    Hash { chain: &'a RawChain, ordinal: usize },
    Explicit(&'a RawInvocation),
}

#[derive(Clone, Copy)]
enum Event<'a> {
    Permutation {
        row_start: usize,
        invocation: Invocation<'a>,
    },
    Compact {
        invocation: &'a RawCompactInvocation,
        template: &'a RawCompactTemplate,
    },
    Witness(&'a RawInstruction),
    Assertion(&'a RawRow),
}

impl Event<'_> {
    fn row_start(self) -> usize {
        match self {
            Self::Permutation { row_start, .. } => row_start,
            Self::Compact { invocation, .. } => word(invocation.2),
            Self::Witness(row) => word(row.0),
            Self::Assertion(row) => word(row.0),
        }
    }
}

fn add_term(terms: &mut Vec<(usize, u64)>, column: usize, coefficient: u64) {
    if coefficient != 0 {
        terms.push((column, coefficient));
    }
}

fn canonicalize(mut terms: Vec<(usize, u64)>) -> Vec<(usize, u64)> {
    terms.sort_unstable_by_key(|term| term.0);
    let mut canonical: Vec<(usize, u64)> = Vec::with_capacity(terms.len());
    for (column, coefficient) in terms {
        match canonical.last_mut() {
            Some((last_column, last_coefficient)) if *last_column == column => {
                *last_coefficient = add_mod(*last_coefficient, coefficient);
                if *last_coefficient == 0 {
                    canonical.pop();
                }
            }
            _ => {
                canonical.push((column, coefficient));
            }
        }
    }
    canonical
}

fn sparse_terms(combination: &RawCombination, layout: &ReferenceLayout) -> Vec<(usize, u64)> {
    let mut terms = Vec::with_capacity(combination.1.len() + 1);
    add_term(&mut terms, layout.constant_column(), combination.0);
    for term in &combination.1 {
        add_term(&mut terms, layout.map_column(word(term.0)), term.1);
    }
    canonicalize(terms)
}

fn explicit_input_terms(
    coefficient: u64,
    input: &RawCombination,
    layout: &ReferenceLayout,
    terms: &mut Vec<(usize, u64)>,
) {
    add_term(terms, layout.constant_column(), mul_mod(coefficient, input.0));
    for term in &input.1 {
        add_term(terms, layout.map_column(word(term.0)), mul_mod(coefficient, term.1));
    }
}

fn template_terms(
    combination: &RawTemplateCombination,
    invocation: Invocation<'_>,
    layout: &ReferenceLayout,
) -> Vec<(usize, u64)> {
    let mut terms = Vec::with_capacity(combination.1.len() * 3 + 1);
    add_term(&mut terms, layout.constant_column(), combination.0);
    for term in &combination.1 {
        let lane = word(term.0 .1);
        match term.0 .0 {
            1 => {
                let witness_start = match invocation {
                    Invocation::Hash { chain, ordinal } => word(chain.5) + ordinal * 592,
                    Invocation::Explicit(invocation) => word(invocation.2),
                };
                add_term(&mut terms, witness_start + lane, term.1);
            }
            0 => match invocation {
                Invocation::Hash { chain, ordinal } => {
                    if ordinal > 0 {
                        add_term(&mut terms, word(chain.5) + (ordinal - 1) * 592 + 584 + lane, term.1);
                    }
                    let absorb_count = word(chain.7);
                    if ordinal < absorb_count {
                        let input_offset = ordinal * 4 + lane;
                        if lane < 4 && input_offset < word(chain.4) {
                            add_term(&mut terms, word(chain.3) + input_offset, term.1);
                        }
                    } else if lane == 0 {
                        add_term(&mut terms, layout.constant_column(), term.1);
                    }
                }
                Invocation::Explicit(invocation) => {
                    explicit_input_terms(term.1, &invocation.3[lane], layout, &mut terms);
                }
            },
            _ => panic!("reference template column tag"),
        }
    }
    canonicalize(terms)
}

fn compact_input_column(invocation: &RawCompactInvocation, input: usize) -> usize {
    for range in &invocation.4 {
        let input_start = word(range.0);
        let input_count = word(range.1);
        if input_start <= input && input < input_start + input_count {
            return word(range.2) + (input - input_start) * word(range.3);
        }
    }
    panic!("reference compact input coverage")
}

fn compact_terms(
    combination: &RawTemplateCombination,
    invocation: &RawCompactInvocation,
    layout: &ReferenceLayout,
) -> Vec<(usize, u64)> {
    let mut terms = Vec::with_capacity(combination.1.len() + 1);
    add_term(&mut terms, layout.constant_column(), combination.0);
    for term in &combination.1 {
        let index = word(term.0 .1);
        let column = match term.0 .0 {
            0 => compact_input_column(invocation, index),
            1 => word(invocation.3) + index,
            _ => panic!("reference compact column tag"),
        };
        add_term(&mut terms, layout.map_column(column), term.1);
    }
    canonicalize(terms)
}

fn template_side(row: &RawTemplateRow, side: MatrixSide) -> &RawTemplateCombination {
    match side {
        MatrixSide::A => &row.1,
        MatrixSide::B => &row.2,
        MatrixSide::C => &row.3,
    }
}

fn compact_side(row: &RawCompactRow, side: MatrixSide) -> &RawTemplateCombination {
    match side {
        MatrixSide::A => &row.1,
        MatrixSide::B => &row.2,
        MatrixSide::C => &row.3,
    }
}

fn row_side(row: &RawRow, side: MatrixSide) -> &RawCombination {
    match side {
        MatrixSide::A => &row.1,
        MatrixSide::B => &row.2,
        MatrixSide::C => &row.3,
    }
}

fn expected_row(
    event: Event<'_>,
    template: &RawTemplate,
    template_ordinal: usize,
    side: MatrixSide,
    layout: &ReferenceLayout,
) -> Vec<(usize, u64)> {
    match event {
        Event::Permutation { invocation, .. } => {
            let row = &template.3[template_ordinal];
            assert_eq!(word(row.0), template_ordinal);
            template_terms(template_side(row, side), invocation, layout)
        }
        Event::Compact { invocation, template } => {
            compact_terms(compact_side(&template.4[template_ordinal], side), invocation, layout)
        }
        Event::Witness(instruction) => match side {
            MatrixSide::A => sparse_terms(&instruction.2, layout),
            MatrixSide::B => sparse_terms(&instruction.3, layout),
            MatrixSide::C => vec![(layout.map_column(word(instruction.1)), 1)],
        },
        Event::Assertion(row) => sparse_terms(row_side(row, side), layout),
    }
}

fn actual_row(matrix: &PackageSparseMatrix, row: usize) -> Vec<(usize, u64)> {
    let start = matrix.row_offsets()[row];
    let end = matrix.row_offsets()[row + 1];
    matrix.column_indices()[start..end]
        .iter()
        .copied()
        .zip(matrix.values()[start..end].iter().copied())
        .collect()
}

fn exact_row_accepts(matrix: &PackageSparseMatrix, row: usize, candidate: &[(usize, u64)]) -> bool {
    actual_row(matrix, row) == candidate
}

fn compare_row(matrix: &PackageSparseMatrix, row: usize, expected: &[(usize, u64)], side: &str) {
    if !exact_row_accepts(matrix, row, expected) {
        assert_eq!(actual_row(matrix, row), expected, "{side} row {row}");
    }
}

fn json_words(value: &Value, location: &str) -> Vec<u64> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("{location} array"))
        .iter()
        .map(|word| {
            let word = word
                .as_u64()
                .unwrap_or_else(|| panic!("{location} canonical word"));
            assert!(word < GOLDILOCKS_MODULUS, "{location} canonical word");
            word
        })
        .collect()
}

fn json_extension(value: &Value, location: &str) -> [u64; 2] {
    json_words(value, location)
        .try_into()
        .unwrap_or_else(|_| panic!("{location} extension width"))
}

fn json_extensions(value: &Value, location: &str) -> Vec<[u64; 2]> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("{location} array"))
        .iter()
        .map(|extension| json_extension(extension, location))
        .collect()
}

struct PhaseLocalInputs {
    private_values: Vec<u64>,
    public_values: Vec<u64>,
    fixture_package_identity: [u64; 4],
    pi_dec_starts: [usize; 4],
}

struct PiDecFixtureInputs {
    package: PiDecV1_1PackageInputs,
    output_preimage: Vec<u64>,
    output_digest: [u64; 4],
    package_identity: [u64; 4],
}

fn nonzero_inputs(package: &LoadedPackage, parity_path: &Path, pi_dec_path: &Path) -> PhaseLocalInputs {
    let bytes = fs::read(parity_path).expect("Lean-emitted PiCCS parity bytes");
    let parity: Value = serde_json::from_slice(&bytes).expect("PiCCS parity JSON");
    let parity = parity.as_array().expect("PiCCS parity tuple");
    assert_eq!(parity.len(), 3, "PiCCS parity tuple length");
    assert_eq!(parity[0].as_u64(), Some(8), "PiCCS parity schema");
    let input = parity[1].as_array().expect("PiCCS parity input tuple");
    let result = parity[2].as_array().expect("PiCCS parity result tuple");
    assert_eq!(input.len(), 11, "PiCCS parity input tuple length");
    assert_eq!(result.len(), 16, "PiCCS parity result tuple length");
    assert_eq!(result[0].as_u64(), Some(1), "Lean PiCCS acceptance");
    assert!(json_words(&result[15], "PiCCS parity assurance")
        .iter()
        .all(|flag| *flag == 1));

    let prior_preimage = json_words(&input[0], "PiCCS parity prior preimage");
    let phase_output_preimage = json_words(&input[1], "PiCCS parity output preimage");
    let prior_public_input = json_words(&input[2], "PiCCS parity public input");
    let phase_output_digest: [u64; 4] = json_words(&input[3], "PiCCS parity output digest")
        .try_into()
        .expect("PiCCS parity digest width");
    let verifier_context_digest: [u64; 4] = json_words(&input[4], "PiCCS parity verifier-context digest")
        .try_into()
        .expect("PiCCS parity verifier-context digest width");
    let fresh_commitment = json_words(&input[5], "PiCCS parity fresh commitment");
    assert_eq!(fresh_commitment.len(), PI_CCS_V1_1_FRESH_COMMITMENT_WORDS);
    assert!(fresh_commitment.iter().all(|word| *word != 0));

    let round_messages: Vec<Vec<[u64; 2]>> = input[6]
        .as_array()
        .expect("PiCCS parity round-message array")
        .iter()
        .map(|round| json_extensions(round, "PiCCS parity round message"))
        .collect();
    assert_eq!(round_messages.len(), PI_CCS_V1_1_ROUND_COUNT);
    assert!(round_messages.iter().all(|round| {
        round.len() == PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT && round.iter().all(|value| *value != [0, 0])
    }));

    let eval_k: Vec<Vec<[u64; 2]>> = input[7]
        .as_array()
        .expect("PiCCS parity Eval_K array")
        .iter()
        .map(|source| json_extensions(source, "PiCCS parity Eval_K"))
        .collect();
    assert_eq!(eval_k.len(), PI_CCS_V1_1_SOURCE_COUNT);
    assert!(eval_k.iter().all(|source| {
        source.len() == PI_CCS_V1_1_COEFFICIENT_COUNT && source.iter().all(|value| *value != [0, 0])
    }));

    let eval_a: Vec<Vec<Vec<[u64; 2]>>> = input[8]
        .as_array()
        .expect("PiCCS parity Eval_A array")
        .iter()
        .map(|source| {
            source
                .as_array()
                .expect("PiCCS parity Eval_A source")
                .iter()
                .map(|matrix| json_extensions(matrix, "PiCCS parity Eval_A"))
                .collect()
        })
        .collect();
    assert_eq!(eval_a.len(), PI_CCS_V1_1_SOURCE_COUNT);
    assert!(eval_a.iter().all(|source| {
        source.len() == PI_CCS_V1_1_MATRIX_COUNT
            && source.iter().all(|matrix| {
                matrix.len() == PI_CCS_V1_1_COEFFICIENT_COUNT && matrix.iter().all(|value| *value != [0, 0])
            })
    }));
    let output_evaluations = PiCcsV1_1OutputEvaluations::new(eval_k, eval_a).expect("nonzero PiCCS output evaluations");
    assert_eq!(prior_preimage.len(), PI_CCS_V1_1_STATE_PREIMAGE_WORDS);
    assert_eq!(phase_output_preimage.len(), PI_CCS_V1_1_STATE_PREIMAGE_WORDS);
    assert_eq!(prior_public_input.len(), PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS);

    let PiDecFixtureInputs {
        package: pi_dec,
        output_preimage,
        output_digest,
        package_identity: fixture_package_identity,
    } = pi_dec_inputs(pi_dec_path);
    assert_ne!(
        phase_output_digest, output_digest,
        "recursive transition changes the phase-local output digest"
    );
    let pi_ccs_private_count = 2 * PI_CCS_V1_1_STATE_PREIMAGE_WORDS
        + PI_CCS_V1_1_FRESH_COMMITMENT_WORDS
        + PI_CCS_V1_1_ROUND_COUNT * PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT * 2
        + PI_CCS_V1_1_SOURCE_COUNT
            * (PI_CCS_V1_1_COEFFICIENT_COUNT * 2 + PI_CCS_V1_1_MATRIX_COUNT * PI_CCS_V1_1_COEFFICIENT_COUNT * 2);
    let pi_dec_commitment_start = pi_ccs_private_count;
    let pi_dec_eval_k_start =
        pi_dec_commitment_start + PI_DEC_V1_1_CHILD_COUNT * PI_DEC_V1_1_COMMITMENT_WORDS_PER_CHILD;
    let pi_dec_eval_a_start = pi_dec_eval_k_start + PI_DEC_V1_1_CHILD_COUNT * PI_DEC_V1_1_EVAL_K_VALUES_PER_CHILD * 2;
    let pi_dec_public_input_start = pi_dec_eval_a_start
        + PI_DEC_V1_1_CHILD_COUNT * PI_DEC_V1_1_EVAL_A_MATRICES_PER_CHILD * PI_CCS_V1_1_COEFFICIENT_COUNT * 2;
    let mut private_values = Vec::with_capacity(package.private_input_count());
    private_values.extend_from_slice(&prior_preimage);
    private_values.extend_from_slice(&output_preimage);
    private_values.extend_from_slice(&fresh_commitment);
    for round in &round_messages {
        for value in round {
            private_values.extend_from_slice(value);
        }
    }
    for source in 0..PI_CCS_V1_1_SOURCE_COUNT {
        for value in &output_evaluations.eval_k()[source] {
            private_values.extend_from_slice(value);
        }
        for matrix in &output_evaluations.eval_a()[source] {
            for value in matrix {
                private_values.extend_from_slice(value);
            }
        }
    }
    for commitment in pi_dec.child_commitments() {
        private_values.extend_from_slice(commitment);
    }
    for eval_k in pi_dec.child_eval_k() {
        for value in eval_k {
            private_values.extend_from_slice(value);
        }
    }
    for eval_a in pi_dec.child_eval_a() {
        for matrix in eval_a {
            for value in matrix {
                private_values.extend_from_slice(value);
            }
        }
    }
    for public_input in pi_dec.child_public_inputs() {
        private_values.extend_from_slice(public_input);
    }
    assert_eq!(private_values.len(), package.private_input_count());

    let mut public_values = Vec::with_capacity(package.public_column_count());
    public_values.extend_from_slice(&prior_public_input);
    public_values.extend_from_slice(&output_digest);
    public_values.extend_from_slice(&verifier_context_digest);
    assert_eq!(public_values.len(), package.public_column_count());
    PhaseLocalInputs {
        private_values,
        public_values,
        fixture_package_identity,
        pi_dec_starts: [
            pi_dec_commitment_start,
            pi_dec_eval_k_start,
            pi_dec_eval_a_start,
            pi_dec_public_input_start,
        ],
    }
}

fn pi_dec_inputs(path: &Path) -> PiDecFixtureInputs {
    let bytes = fs::read(path).expect("Lean-emitted PiDEC parity bytes");
    let parity: Value = serde_json::from_slice(&bytes).expect("PiDEC parity JSON");
    let parity = parity.as_array().expect("PiDEC parity tuple");
    assert_eq!(parity.len(), 3, "PiDEC parity tuple length");
    assert_eq!(parity[0].as_u64(), Some(2), "PiDEC parity schema");
    let input = parity[1].as_array().expect("PiDEC parity input tuple");
    let result = parity[2].as_array().expect("PiDEC parity result tuple");
    assert_eq!(input.len(), 7, "PiDEC parity input tuple length");
    assert_eq!(result.len(), 19, "PiDEC parity result tuple length");
    assert_eq!(result[0].as_u64(), Some(1), "Lean PiDEC acceptance");
    assert_eq!(result[1].as_u64(), Some(1), "Lean PiDEC parent bound");
    assert_eq!(result[6].as_u64(), Some(1), "Lean PiDEC commitment equation");
    assert_eq!(result[8].as_u64(), Some(1), "Lean PiDEC public-input equation");
    assert_eq!(result[10].as_u64(), Some(1), "Lean PiDEC Eval_K equation");
    assert_eq!(result[12].as_u64(), Some(1), "Lean PiDEC Eval_A equation");
    assert_eq!(result[15].as_u64(), Some(1), "Lean PiDEC unbounded rejection");
    assert!(json_words(&result[3], "PiDEC parent-bound results")
        .iter()
        .all(|flag| *flag == 1));
    assert!(result[4]
        .as_array()
        .expect("PiDEC digit-range children")
        .iter()
        .all(|child| json_words(child, "PiDEC digit-range results")
            .iter()
            .all(|flag| *flag == 1)));
    assert!(json_words(&result[16], "PiDEC assurance")
        .iter()
        .all(|flag| *flag == 1));
    assert_eq!(result[2], input[4], "PiDEC verifier-computed public digits");
    assert_eq!(result[14], input[5], "PiDEC outgoing state");
    let package_identity = json_words(&input[6], "PiDEC package identity")
        .try_into()
        .expect("PiDEC package identity width");
    assert_eq!(
        result[13].as_array().expect("PiDEC child claims").len(),
        PI_DEC_V1_1_CHILD_COUNT,
        "PiDEC child result count",
    );

    let child_commitments = input[1]
        .as_array()
        .expect("PiDEC child commitments")
        .iter()
        .map(|child| json_words(child, "PiDEC child commitment"))
        .collect();
    let child_eval_k = input[2]
        .as_array()
        .expect("PiDEC child Eval_K")
        .iter()
        .map(|child| json_extensions(child, "PiDEC child Eval_K"))
        .collect();
    let child_eval_a = input[3]
        .as_array()
        .expect("PiDEC child Eval_A")
        .iter()
        .map(|child| {
            child
                .as_array()
                .expect("PiDEC child Eval_A matrices")
                .iter()
                .map(|matrix| json_extensions(matrix, "PiDEC child Eval_A"))
                .collect()
        })
        .collect();
    let child_public_inputs = input[4]
        .as_array()
        .expect("PiDEC child public inputs")
        .iter()
        .map(|child| json_words(child, "PiDEC child public input"))
        .collect();
    let output_preimage = json_words(&result[17], "running-transition output preimage");
    assert_eq!(output_preimage.len(), PI_CCS_V1_1_STATE_PREIMAGE_WORDS);
    let output_digest = json_words(&result[18], "running-transition output digest")
        .try_into()
        .expect("running-transition output digest width");
    PiDecFixtureInputs {
        package: PiDecV1_1PackageInputs::new(child_commitments, child_eval_k, child_eval_a, child_public_inputs)
            .expect("typed PiDEC package inputs"),
        output_preimage,
        output_digest,
        package_identity,
    }
}

fn events(raw: &RawPackage) -> Vec<Event<'_>> {
    let template_rows = raw.5 .3.len();
    let mut events = Vec::new();
    for chain in &raw.6 {
        assert_ne!(chain.0, 0, "reference hash-chain phase");
        assert_eq!(
            word(chain.2),
            word(chain.6) + word(chain.8),
            "reference hash-chain rows",
        );
        assert_eq!(
            word(chain.6),
            (word(chain.7) + 1) * template_rows,
            "reference hash-chain witness rows",
        );
        assert!(
            word(chain.8) == 0
                || (word(chain.9) >= word(raw.3 .2) + 1 && word(chain.9) + word(chain.8) <= word(raw.3 .4)),
            "reference hash-chain digest range",
        );
        for ordinal in 0..=word(chain.7) {
            events.push(Event::Permutation {
                row_start: word(chain.1) + ordinal * template_rows,
                invocation: Invocation::Hash { chain, ordinal },
            });
        }
    }
    events.extend(raw.7.iter().map(|invocation| {
        assert_ne!(invocation.0, 0, "reference invocation phase");
        Event::Permutation {
            row_start: word(invocation.1),
            invocation: Invocation::Explicit(invocation),
        }
    }));
    events.extend(raw.9.iter().map(|invocation| {
        assert_ne!(invocation.0, 0, "reference compact phase");
        let template = &raw.8[word(invocation.1)];
        assert!(word(template.2) < word(template.0), "reference compact output input");
        assert_eq!(template.4.len(), word(template.1) + 1, "reference compact rows");
        let mut input_cursor = 0usize;
        for range in &invocation.4 {
            assert_eq!(word(range.0), input_cursor, "reference compact input order");
            assert_ne!(word(range.1), 0, "reference compact input count");
            assert_ne!(word(range.3), 0, "reference compact column stride");
            input_cursor += word(range.1);
        }
        assert_eq!(input_cursor, word(template.0), "reference compact input coverage");
        Event::Compact { invocation, template }
    }));
    events.extend(raw.11.iter().map(Event::Witness));
    events.extend(raw.12.iter().map(Event::Assertion));
    events.sort_unstable_by_key(|event| event.row_start());
    events
}

fn event_row_count(event: Event<'_>, raw: &RawPackage) -> usize {
    match event {
        Event::Permutation { .. } => raw.5 .3.len(),
        Event::Compact { template, .. } => template.4.len(),
        Event::Witness(_) | Event::Assertion(_) => 1,
    }
}

pub fn run(
    plan_path: &Path,
    reference_path: &Path,
    pi_ccs_parity_path: &Path,
    pi_dec_parity_path: &Path,
    expected_identity: [u64; 4],
) {
    let plan_bytes = fs::read(plan_path).expect("Lean-emitted package-plan bytes");
    let reference_bytes = fs::read(reference_path).expect("Lean-emitted expanded-package bytes");
    let (package, expanded_bytes) = load_with_expanded_package(&plan_bytes, expected_identity)
        .expect("identity-checked candidate package and canonical expansion");
    assert_eq!(
        expanded_bytes.len(),
        reference_bytes.len(),
        "production and Lean expanded-package byte lengths",
    );
    if let Some(index) = expanded_bytes
        .iter()
        .zip(&reference_bytes)
        .position(|(actual, expected)| actual != expected)
    {
        panic!("production and Lean expanded packages differ at byte {index}");
    }
    let matrices = package.r1cs_matrices().expect("final package matrices");
    let raw: RawPackage = serde_json::from_slice(&reference_bytes).expect("independent expanded-package decode");

    assert_eq!(raw.0, 8);
    let cube_variables = package.ccs_relation().cube_variables();
    let domain_size = 1usize << cube_variables;
    let layout = ReferenceLayout {
        unpadded_rows: word(raw.3 .0),
        unpadded_constant: word(raw.3 .2),
        public_columns: word(raw.3 .3),
        domain_size,
        final_columns: domain_size + 1 + word(raw.3 .3),
    };
    assert_eq!(word(raw.3 .1), layout.unpadded_constant);
    assert_eq!(word(raw.3 .4), layout.unpadded_constant + 1 + layout.public_columns);
    assert_eq!((word(raw.5 .0), word(raw.5 .1), word(raw.5 .2)), (8, 592, 584));

    for matrix in [matrices.a(), matrices.b(), matrices.c()] {
        assert_eq!(matrix.rows(), layout.domain_size);
        assert_eq!(matrix.columns(), layout.final_columns);
        assert!(matrix
            .values()
            .iter()
            .all(|value| *value != 0 && *value < GOLDILOCKS_MODULUS));
    }
    let matrix_nonzeros = [
        matrices.a().nonzero_count(),
        matrices.b().nonzero_count(),
        matrices.c().nonzero_count(),
    ];

    let schedule = events(&raw);
    let mut row_cursor = 0usize;
    for &event in &schedule {
        assert_eq!(event.row_start(), row_cursor, "independent row schedule");
        row_cursor += event_row_count(event, &raw);
    }
    assert_eq!(row_cursor, layout.unpadded_rows);
    schedule.par_iter().for_each(|&event| {
        let row_count = event_row_count(event, &raw);
        for ordinal in 0..row_count {
            let row_index = event.row_start() + ordinal;
            let expected_a = expected_row(event, &raw.5, ordinal, MatrixSide::A, &layout);
            let expected_b = expected_row(event, &raw.5, ordinal, MatrixSide::B, &layout);
            let expected_c = expected_row(event, &raw.5, ordinal, MatrixSide::C, &layout);
            compare_row(matrices.a(), row_index, &expected_a, "A");
            compare_row(matrices.b(), row_index, &expected_b, "B");
            compare_row(matrices.c(), row_index, &expected_c, "C");
        }
    });
    println!("matrix_row_equality=passed");

    for matrix in [matrices.a(), matrices.b(), matrices.c()] {
        let final_nonzero = matrix.nonzero_count();
        assert!(matrix.row_offsets()[layout.unpadded_rows..]
            .iter()
            .all(|offset| *offset == final_nonzero));
    }
    drop(matrices);
    let encoded = nonzero_inputs(&package, pi_ccs_parity_path, pi_dec_parity_path);
    assert_eq!(encoded.private_values.len(), package.private_input_count());
    assert_eq!(encoded.public_values.len(), package.public_column_count());
    let assignment = package
        .execute_witness(&encoded.private_values, &encoded.public_values)
        .expect("phase-local PiCCS witness assignment");
    assert_eq!(assignment.private_values().len(), layout.unpadded_constant);
    assert_eq!(assignment.public_values().len(), layout.public_columns);
    assert!(assignment
        .private_values()
        .iter()
        .chain(assignment.public_values())
        .all(|value| *value < GOLDILOCKS_MODULUS));
    let fresh_start = 2 * PI_CCS_V1_1_STATE_PREIMAGE_WORDS;
    let rounds_start = fresh_start + PI_CCS_V1_1_FRESH_COMMITMENT_WORDS;
    let eval_k_start = rounds_start + PI_CCS_V1_1_ROUND_COUNT * PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT * 2;
    let eval_a_start = eval_k_start + PI_CCS_V1_1_COEFFICIENT_COUNT * 2;
    let mut input_mutation_checks = 0;
    for (location, index) in [
        ("prior preimage", 0),
        ("output preimage", PI_CCS_V1_1_STATE_PREIMAGE_WORDS),
        ("fresh commitment", fresh_start),
        ("round messages", rounds_start),
        ("output Eval_K", eval_k_start),
        ("output Eval_A", eval_a_start),
    ] {
        let mut private_values = encoded.private_values.clone();
        private_values[index] = changed_word(private_values[index]);
        assert!(
            package
                .execute_witness(&private_values, &encoded.public_values)
                .is_err(),
            "{location} mutation must reject",
        );
        input_mutation_checks += 1;
    }
    for (location, index) in [
        ("PiDEC commitments", encoded.pi_dec_starts[0]),
        ("PiDEC Eval_K", encoded.pi_dec_starts[1]),
        ("PiDEC Eval_A", encoded.pi_dec_starts[2]),
        ("PiDEC child public inputs", encoded.pi_dec_starts[3]),
    ] {
        let mut private_values = encoded.private_values.clone();
        private_values[index] = changed_word(private_values[index]);
        assert!(
            package
                .execute_witness(&private_values, &encoded.public_values)
                .is_err(),
            "{location} mutation must reject",
        );
        input_mutation_checks += 1;
    }
    for (location, index) in [
        ("prior public input", 0),
        ("output digest", PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS),
    ] {
        let mut public_values = encoded.public_values.clone();
        public_values[index] = changed_word(public_values[index]);
        assert!(
            package
                .execute_witness(&encoded.private_values, &public_values)
                .is_err(),
            "{location} mutation must reject",
        );
        input_mutation_checks += 1;
    }
    let context_start = PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS + 4;
    for lane in 0..4 {
        let mut public_values = encoded.public_values.clone();
        public_values[context_start + lane] = changed_word(public_values[context_start + lane]);
        assert!(
            package
                .execute_witness(&encoded.private_values, &public_values)
                .is_err(),
            "verification-key lane {lane} mutation must reject",
        );
        input_mutation_checks += 1;
    }

    assert_eq!(
        evaluate_canonical_assignment(
            &reference_bytes,
            assignment.private_values(),
            assignment.public_values(),
        )
        .expect("independent canonical assignment evaluation"),
        layout.unpadded_rows,
    );
    println!("expanded_package_bytes={}", expanded_bytes.len());
    println!("relation_identifier={expected_identity:?}");
    assert_eq!(
        encoded.fixture_package_identity, POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY,
        "phase-local fixture package identity",
    );
    println!("matrix_rows={}", layout.domain_size);
    println!("matrix_nonzeros={matrix_nonzeros:?}");
    println!("independent_assignment_rows={}", layout.unpadded_rows);
    println!("input_mutation_checks={input_mutation_checks}");
    println!("mutation_checks={input_mutation_checks}");
    println!("package_matrix_conformance=passed");
}
