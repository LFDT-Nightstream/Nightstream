use std::{fs, path::PathBuf};

use nightstream_fprime::{
    load, PackageError, PackageSparseMatrix, PiCcsV1_1OutputEvaluations, PiCcsV1_1PackageInputs, WitnessAssignment,
    PI_CCS_V1_1_COEFFICIENT_COUNT, PI_CCS_V1_1_FRESH_COMMITMENT_WORDS, PI_CCS_V1_1_MATRIX_COUNT,
    PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_1_ROUND_COUNT,
    PI_CCS_V1_1_SOURCE_COUNT, PI_CCS_V1_1_STATE_PREIMAGE_WORDS,
};
use serde::de::IgnoredAny;
use serde::Deserialize;
use serde_json::Value;

const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;

#[derive(Deserialize)]
struct RawPackage(
    u64,
    IgnoredAny,
    IgnoredAny,
    RawLayout,
    IgnoredAny,
    RawTemplate,
    Vec<RawChain>,
    Vec<RawInvocation>,
    IgnoredAny,
    Vec<RawInstruction>,
    Vec<RawRow>,
    IgnoredAny,
);

#[derive(Deserialize)]
struct RawLayout(u64, u64, u64, u64, u64, IgnoredAny, IgnoredAny);

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
struct RawChain(u64, u64, u64, u64, u64, u64, u64, u64, u64);

#[derive(Deserialize)]
struct RawInvocation(u64, u64, u64, Vec<RawCombination>);

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
    Witness(&'a RawInstruction),
    Assertion(&'a RawRow),
}

impl Event<'_> {
    fn row_start(self) -> usize {
        match self {
            Self::Permutation { row_start, .. } => row_start,
            Self::Witness(row) => word(row.0),
            Self::Assertion(row) => word(row.0),
        }
    }
}

struct ReferenceLayout {
    unpadded_rows: usize,
    unpadded_constant: usize,
    public_columns: usize,
    domain_size: usize,
    final_columns: usize,
}

impl ReferenceLayout {
    fn map_column(&self, column: usize) -> usize {
        if column < self.unpadded_constant {
            column
        } else {
            self.domain_size + (column - self.unpadded_constant)
        }
    }

    fn constant_column(&self) -> usize {
        self.domain_size
    }
}

fn artifact_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-v1.json")
}

fn parity_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-piccs-parity-v1.json")
}

fn word(value: u64) -> usize {
    usize::try_from(value).expect("reference word fits usize")
}

fn add_mod(left: u64, right: u64) -> u64 {
    ((u128::from(left) + u128::from(right)) % u128::from(GOLDILOCKS_MODULUS)) as u64
}

fn mul_mod(left: u64, right: u64) -> u64 {
    ((u128::from(left) * u128::from(right)) % u128::from(GOLDILOCKS_MODULUS)) as u64
}

fn changed_word(value: u64) -> u64 {
    if value + 1 == GOLDILOCKS_MODULUS {
        0
    } else {
        value + 1
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

fn template_side(row: &RawTemplateRow, side: MatrixSide) -> &RawTemplateCombination {
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

fn compare_row(matrix: &PackageSparseMatrix, row: usize, expected: &[(usize, u64)], side: &str) {
    let actual = actual_row(matrix, row);
    assert_eq!(actual, expected, "{side} row {row}");
}

fn assignment_value(column: usize, layout: &ReferenceLayout, assignment: &WitnessAssignment) -> u64 {
    assert!(column < layout.final_columns, "reference assignment column");
    if column < layout.unpadded_constant {
        assignment.private_values()[column]
    } else if column < layout.domain_size {
        0
    } else if column == layout.constant_column() {
        1
    } else {
        assignment.public_values()[column - layout.constant_column() - 1]
    }
}

fn evaluate_reference_combination(
    combination: &[(usize, u64)],
    layout: &ReferenceLayout,
    assignment: &WitnessAssignment,
) -> u64 {
    combination.iter().fold(0, |sum, (column, coefficient)| {
        add_mod(
            sum,
            mul_mod(*coefficient, assignment_value(*column, layout, assignment)),
        )
    })
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

fn nonzero_inputs() -> PiCcsV1_1PackageInputs {
    let bytes = fs::read(parity_path()).expect("Lean-emitted PiCCS parity bytes");
    let parity: Value = serde_json::from_slice(&bytes).expect("PiCCS parity JSON");
    let parity = parity.as_array().expect("PiCCS parity tuple");
    assert_eq!(parity.len(), 3, "PiCCS parity tuple length");
    assert_eq!(parity[0].as_u64(), Some(6), "PiCCS parity schema");
    let input = parity[1].as_array().expect("PiCCS parity input tuple");
    let result = parity[2].as_array().expect("PiCCS parity result tuple");
    assert_eq!(input.len(), 11, "PiCCS parity input tuple length");
    assert_eq!(result.len(), 16, "PiCCS parity result tuple length");
    assert_eq!(result[0].as_u64(), Some(1), "Lean PiCCS acceptance");
    assert!(json_words(&result[15], "PiCCS parity assurance")
        .iter()
        .all(|flag| *flag == 1));

    let prior_preimage = json_words(&input[0], "PiCCS parity prior preimage");
    let output_preimage = json_words(&input[1], "PiCCS parity output preimage");
    let prior_public_input = json_words(&input[2], "PiCCS parity public input");
    let output_digest: [u64; 4] = json_words(&input[3], "PiCCS parity output digest")
        .try_into()
        .expect("PiCCS parity digest width");
    let verifier_context: [u64; 4] = json_words(&input[4], "PiCCS parity verifier context")
        .try_into()
        .expect("PiCCS parity verifier-context width");
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

    PiCcsV1_1PackageInputs::new(
        prior_preimage,
        output_preimage,
        fresh_commitment,
        round_messages,
        output_evaluations,
        prior_public_input,
        output_digest,
        verifier_context,
    )
    .expect("canonical PiCCS package inputs")
}

fn candidate_identity(bytes: &[u8]) -> [u64; 4] {
    match load(bytes, [0; 4]) {
        Err(PackageError::ExpectedIdentityMismatch { computed, .. }) => computed,
        Ok(_) => [0; 4],
        Err(error) => panic!("candidate package does not load: {error}"),
    }
}

fn events(raw: &RawPackage) -> Vec<Event<'_>> {
    let template_rows = raw.5 .3.len();
    let mut events = Vec::new();
    for chain in &raw.6 {
        assert_ne!(chain.0, 0, "reference hash-chain phase");
        assert_eq!(word(chain.2), word(chain.6) + 4, "reference hash-chain rows");
        assert_eq!(
            word(chain.6),
            (word(chain.7) + 1) * template_rows,
            "reference hash-chain witness rows",
        );
        assert!(
            word(chain.8) >= word(raw.3 .2) + 1 && word(chain.8) + 4 <= word(raw.3 .4),
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
    events.extend(raw.9.iter().map(Event::Witness));
    events.extend(raw.10.iter().map(Event::Assertion));
    events.sort_unstable_by_key(|event| event.row_start());
    events
}

#[test]
fn final_rust_matrices_equal_the_lean_padded_rows_entry_for_entry() {
    let bytes = fs::read(artifact_path()).expect("Lean-emitted package bytes");
    let identity = candidate_identity(&bytes);
    let package = load(&bytes, identity).expect("identity-checked candidate package");
    let matrices = package.r1cs_matrices().expect("final package matrices");
    let raw: RawPackage = serde_json::from_slice(&bytes).expect("independent package decode");

    assert_eq!(raw.0, 6);
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

    let schedule = events(&raw);
    let mut row_cursor = 0usize;
    let mut row_mutations_checked = false;
    for &event in &schedule {
        assert_eq!(event.row_start(), row_cursor, "independent row schedule");
        let row_count = match event {
            Event::Permutation { .. } => raw.5 .3.len(),
            Event::Witness(_) | Event::Assertion(_) => 1,
        };
        for ordinal in 0..row_count {
            let expected_a = expected_row(event, &raw.5, ordinal, MatrixSide::A, &layout);
            let expected_b = expected_row(event, &raw.5, ordinal, MatrixSide::B, &layout);
            let expected_c = expected_row(event, &raw.5, ordinal, MatrixSide::C, &layout);
            compare_row(matrices.a(), row_cursor, &expected_a, "A");
            compare_row(matrices.b(), row_cursor, &expected_b, "B");
            compare_row(matrices.c(), row_cursor, &expected_c, "C");
            if !row_mutations_checked && !expected_a.is_empty() {
                let actual = actual_row(matrices.a(), row_cursor);

                let mut changed_row = expected_a.clone();
                changed_row.remove(0);
                assert_ne!(actual, changed_row, "row mutation must fail exact comparison");

                let mut changed_coefficient = expected_a.clone();
                changed_coefficient[0].1 = changed_word(changed_coefficient[0].1);
                changed_coefficient = canonicalize(changed_coefficient);
                assert_ne!(
                    actual, changed_coefficient,
                    "coefficient mutation must fail exact comparison"
                );

                let mut changed_column = expected_a.clone();
                changed_column[0].0 = (changed_column[0].0 + 1) % layout.final_columns;
                changed_column = canonicalize(changed_column);
                assert_ne!(actual, changed_column, "column mutation must fail exact comparison");
                row_mutations_checked = true;
            }
            row_cursor += 1;
        }
    }
    assert_eq!(row_cursor, layout.unpadded_rows);
    assert!(row_mutations_checked);

    for matrix in [matrices.a(), matrices.b(), matrices.c()] {
        let final_nonzero = matrix.nonzero_count();
        assert!(matrix.row_offsets()[layout.unpadded_rows..]
            .iter()
            .all(|offset| *offset == final_nonzero));
    }
    drop(matrices);

    let inputs = nonzero_inputs();
    let encoded = package
        .encode_pi_ccs_v1_1_inputs(&inputs)
        .expect("canonical PiCCS input encoding");
    let assignment = package
        .execute_witness(encoded.private_values(), encoded.public_values())
        .expect("canonical PiCCS witness assignment");
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
    for (location, index) in [
        ("prior preimage", 0),
        ("output preimage", PI_CCS_V1_1_STATE_PREIMAGE_WORDS),
        ("fresh commitment", fresh_start),
        ("round messages", rounds_start),
        ("output Eval_K", eval_k_start),
        ("output Eval_A", eval_a_start),
    ] {
        let mut private_values = encoded.private_values().to_vec();
        private_values[index] = changed_word(private_values[index]);
        assert!(
            package
                .execute_witness(&private_values, encoded.public_values())
                .is_err(),
            "{location} mutation must reject",
        );
    }
    for (location, index) in [
        ("prior public input", 0),
        ("output digest", PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS),
    ] {
        let mut public_values = encoded.public_values().to_vec();
        public_values[index] = changed_word(public_values[index]);
        assert!(
            package
                .execute_witness(encoded.private_values(), &public_values)
                .is_err(),
            "{location} mutation must reject",
        );
    }
    let context_start = PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS + 4;
    for lane in 0..4 {
        let mut public_values = encoded.public_values().to_vec();
        public_values[context_start + lane] = changed_word(public_values[context_start + lane]);
        assert!(
            package
                .execute_witness(encoded.private_values(), &public_values)
                .is_err(),
            "verifier-context lane {lane} mutation must reject",
        );
    }

    let mut checked_rows = 0usize;
    for &event in &schedule {
        assert_eq!(event.row_start(), checked_rows, "independent assignment row schedule");
        let row_count = match event {
            Event::Permutation { .. } => raw.5 .3.len(),
            Event::Witness(_) | Event::Assertion(_) => 1,
        };
        for ordinal in 0..row_count {
            let left = evaluate_reference_combination(
                &expected_row(event, &raw.5, ordinal, MatrixSide::A, &layout),
                &layout,
                &assignment,
            );
            let right = evaluate_reference_combination(
                &expected_row(event, &raw.5, ordinal, MatrixSide::B, &layout),
                &layout,
                &assignment,
            );
            let output = evaluate_reference_combination(
                &expected_row(event, &raw.5, ordinal, MatrixSide::C, &layout),
                &layout,
                &assignment,
            );
            assert_eq!(
                mul_mod(left, right),
                output,
                "independent assignment row {checked_rows}",
            );
            checked_rows += 1;
        }
    }
    assert_eq!(checked_rows, layout.unpadded_rows);

    let empty_row = Vec::new();
    let zero = evaluate_reference_combination(&empty_row, &layout, &assignment);
    assert_eq!(mul_mod(zero, zero), zero, "independent padded zero rows");
}
