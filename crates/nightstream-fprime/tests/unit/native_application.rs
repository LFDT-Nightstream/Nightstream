use super::*;
use crate::application_records::{ApplicationForm, ApplicationRecordsWriter, ApplicationRowHeader, ApplicationTerm};
use crate::package::{PermutationTemplate, Segment};
use serde_json::{json, Value};

fn metadata() -> Value {
    json!([
        1,
        1,
        [0, 1, 2, 3],
        [8],
        [4, 5, 6, 7],
        9,
        1,
        2,
        2,
        [],
        [],
        [],
        [],
        [],
        [],
        []
    ])
}

fn layout() -> Layout {
    Layout {
        row_count: 5,
        private_column_count: 10,
        constant_column: 10,
        public_column_count: 1,
        total_column_count: 12,
        private_segments: vec![
            Segment {
                role: super::super::sealed::APPLICATION_WITNESS_ROLE,
                start: 8,
                length: 1,
            },
            Segment {
                role: super::super::sealed::APPLICATION_LOCAL_ROLE,
                start: 9,
                length: 1,
            },
        ],
        public_segments: Vec::new(),
    }
}

fn term(form: ApplicationForm, variable: usize, coefficient: u64) -> Result<ApplicationTerm, PackageError> {
    Ok(ApplicationTerm {
        form,
        variable,
        coefficient,
    })
}

fn records(variable: usize, recipe_variable: usize) -> Arc<ApplicationRecords> {
    let mut writer = ApplicationRecordsWriter::new().unwrap();
    writer
        .append_row(
            ApplicationRowHeader {
                constants: [0, 1, 0],
                term_counts: [3, 0, 1],
            },
            [
                term(ApplicationForm::A, 0, 1),
                term(ApplicationForm::A, variable, 3),
                term(ApplicationForm::A, 0, super::super::GOLDILOCKS_MODULUS - 1),
                term(ApplicationForm::C, 9, 1),
            ],
        )
        .unwrap();
    writer
        .append_recipe(
            0,
            [
                Ok(ApplicationRecipeNode::Multiply),
                Ok(ApplicationRecipeNode::Constant(3)),
                Ok(ApplicationRecipeNode::Variable(recipe_variable)),
            ],
        )
        .unwrap();
    writer
        .append_row(
            ApplicationRowHeader {
                constants: [0, 1, 0],
                term_counts: [1, 0, 1],
            },
            [term(ApplicationForm::A, 9, 1), term(ApplicationForm::C, 5, 1)],
        )
        .unwrap();
    Arc::new(writer.finish().unwrap())
}

#[test]
fn sealed_native_rows_preserve_raw_terms_and_map_bounded_runtime_coefficients() {
    let records = records(4, 4);
    let application = PreparedApplication::from_metadata(&metadata(), records.clone()).unwrap();
    application.validate(&layout()).unwrap();
    assert_eq!(records.row_header(0).unwrap().term_counts, [3, 0, 1]);
    let mut original = Vec::new();
    let flow = records
        .visit_terms(0, |term| {
            original.push((term.form.index(), term.variable, term.coefficient));
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(flow, ControlFlow::Continue(()));
    assert_eq!(original[0], (0, 0, 1));
    assert_eq!(original[2], (0, 0, super::super::GOLDILOCKS_MODULUS - 1));
    let row = application.assertion(0).unwrap();
    assert_eq!(row.row_index, 2);
    assert_eq!(row.a.terms.len(), 1);
    assert_eq!(
        (row.a.terms[0].column, row.a.terms[0].coefficient.as_canonical_u64()),
        (8, 3)
    );
    assert_eq!(row.c.terms[0].column, 9);
    let copy = application.clone();
    assert!(Arc::ptr_eq(&application.records, &copy.records));
}

#[test]
fn native_recipes_execute_stored_syntax_and_report_global_assertion_failure() {
    let application = PreparedApplication::from_metadata(&metadata(), records(4, 4)).unwrap();
    application.validate(&layout()).unwrap();
    let mut assignment = vec![Goldilocks::ZERO; layout().total_column_count];
    assignment[8] = Goldilocks::from_u64(2);
    assignment[4] = Goldilocks::from_u64(6);
    application.execute_recipes(&mut assignment).unwrap();
    assert_eq!(assignment[9], Goldilocks::from_u64(6));
    application.check_assertions(&assignment).unwrap();
    assignment[4] += Goldilocks::ONE;
    assert!(matches!(
        application.check_assertions(&assignment),
        Err(PackageError::UnsatisfiedAssertionRow { row: 3 })
    ));
    assignment[9] += Goldilocks::ONE;
    assert!(matches!(
        application.check_assertions(&assignment),
        Err(PackageError::UnsatisfiedAssertionRow { row: 2 })
    ));

    // A different, still-causal stored recipe must execute as written and
    // fail its assertion; using the generated row as an unchecked hint would
    // incorrectly replace this value with 3 * assignment[8].
    let changed = PreparedApplication::from_metadata(&metadata(), records(4, 0)).unwrap();
    changed.validate(&layout()).unwrap();
    assignment[0] = Goldilocks::from_u64(5);
    changed.execute_recipes(&mut assignment).unwrap();
    assert_eq!(assignment[9], Goldilocks::from_u64(15));
    assert!(matches!(
        changed.check_assertions(&assignment),
        Err(PackageError::UnsatisfiedAssertionRow { row: 2 })
    ));
}

#[test]
fn precomputed_application_values_keep_input_output_and_constraint_checks() {
    let application = PreparedApplication::from_metadata(&metadata(), records(4, 4)).unwrap();
    application.validate(&layout()).unwrap();
    let mut assignment = vec![Goldilocks::ZERO; layout().total_column_count];
    assignment[8] = Goldilocks::from_u64(2);
    assignment[4] = Goldilocks::from_u64(6);
    let values = [0, 0, 0, 0, 2, 6, 0, 0, 0, 6].map(Goldilocks::from_u64);
    application.apply_values(&values, &mut assignment).unwrap();
    application.check_assertions(&assignment).unwrap();
    for local in [0, 4, 5] {
        let mut changed = values;
        changed[local] += Goldilocks::ONE;
        assert!(application.apply_values(&changed, &mut assignment).is_err());
    }
    assert!(application
        .apply_values(&values[..9], &mut assignment)
        .is_err());
    let mut changed = values;
    changed[9] += Goldilocks::ONE;
    application.apply_values(&changed, &mut assignment).unwrap();
    assert!(matches!(
        application.check_assertions(&assignment),
        Err(PackageError::UnsatisfiedAssertionRow { row: 2 })
    ));
}

#[test]
fn native_recipe_batch_matches_independent_evaluation_with_causal_dependencies() {
    use ApplicationRecipeNode::*;

    let mut writer = ApplicationRecordsWriter::new().unwrap();
    let rows = [
        (
            [2, 0, 0],
            [1, 1, 1],
            vec![
                term(ApplicationForm::A, 4, 1),
                term(ApplicationForm::B, 0, 1),
                term(ApplicationForm::C, 9, 1),
            ],
            vec![Multiply, Add, Variable(4), Constant(2), Variable(0)],
        ),
        (
            [0, 1, 0],
            [1, 0, 1],
            vec![term(ApplicationForm::A, 9, 1), term(ApplicationForm::C, 10, 1)],
            vec![Variable(9)],
        ),
        (
            [0, 1, 0],
            [2, 0, 1],
            vec![
                term(ApplicationForm::A, 10, 1),
                term(ApplicationForm::A, 4, 3),
                term(ApplicationForm::C, 11, 1),
            ],
            vec![Add, Variable(10), Multiply, Constant(3), Variable(4)],
        ),
    ];
    for (constants, term_counts, terms, recipe) in rows {
        let row = writer
            .append_row(ApplicationRowHeader { constants, term_counts }, terms)
            .unwrap();
        writer
            .append_recipe(row, recipe.into_iter().map(Ok))
            .unwrap();
    }
    writer
        .append_row(
            ApplicationRowHeader {
                constants: [0, 1, 0],
                term_counts: [1, 0, 1],
            },
            [term(ApplicationForm::A, 11, 1), term(ApplicationForm::C, 5, 1)],
        )
        .unwrap();
    let records = Arc::new(writer.finish().unwrap());
    let mut metadata = metadata();
    metadata[6] = json!(3);
    metadata[8] = json!(4);
    let mut layout = layout();
    layout.row_count = 7;
    layout.private_column_count = 12;
    layout.constant_column = 12;
    layout.total_column_count = 14;
    layout.private_segments[1].length = 3;
    let application = PreparedApplication::from_metadata(&metadata, records.clone()).unwrap();
    application.validate(&layout).unwrap();
    let mut actual = vec![Goldilocks::ZERO; layout.total_column_count];
    actual[0] = Goldilocks::from_u64(5);
    actual[8] = Goldilocks::from_u64(7);
    actual[4] = Goldilocks::from_u64(66);
    let mut expected = actual.clone();
    for recipe in 0..records.recipe_count() {
        let value = records
            .evaluate_recipe(recipe, |variable| {
                Ok(expected[application.column(variable)?].as_canonical_u64())
            })
            .unwrap();
        expected[application.plan().private_range().start + recipe] = Goldilocks::from_u64(value);
    }
    application.execute_recipes(&mut actual).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual[9..12], [45, 45, 66].map(Goldilocks::from_u64));
    application.check_assertions(&actual).unwrap();
}

#[test]
fn native_records_reject_unmapped_terms_noncausal_recipes_and_dynamic_placeholders() {
    let invalid_column = PreparedApplication::from_metadata(&metadata(), records(10, 4)).unwrap();
    assert!(matches!(
        invalid_column.validate(&layout()),
        Err(PackageError::Invalid("native application variable"))
    ));
    let future_recipe = PreparedApplication::from_metadata(&metadata(), records(4, 9)).unwrap();
    assert!(matches!(
        future_recipe.validate(&layout()),
        Err(PackageError::Invalid("noncausal witness expression"))
    ));
    // Output local five maps to physical column four, before target nine.
    // The native DSL still forbids that dependency while generating locals.
    let output_recipe = PreparedApplication::from_metadata(&metadata(), records(4, 5)).unwrap();
    assert!(matches!(
        output_recipe.validate(&layout()),
        Err(PackageError::Invalid("noncausal witness expression"))
    ));
    for field in 9..16 {
        let mut changed = metadata();
        changed[field] = json!([0]);
        assert!(PreparedApplication::from_metadata(&changed, records(4, 4)).is_err());
    }
    let mut changed = metadata();
    changed[8] = json!(3);
    assert!(matches!(
        PreparedApplication::from_metadata(&changed, records(4, 4)),
        Err(PackageError::Invalid("native application record counts"))
    ));
}

#[test]
fn native_row_coverage_rejects_fixed_rows_inside_the_overlay() {
    let application = PreparedApplication::from_metadata(&metadata(), records(4, 4)).unwrap();
    let permutation = PermutationTemplate {
        input_count: 0,
        local_column_count: 0,
        output_local_start: 0,
        rows: Vec::new(),
    };
    let empty = SparseCombination {
        constant: Goldilocks::ZERO,
        terms: Vec::new(),
    };
    let mut fixed = [0, 1, 4].map(|row_index| SparseRow {
        row_index,
        a: empty.clone(),
        b: empty.clone(),
        c: empty.clone(),
    });
    super::super::validate_row_coverage(
        &layout(),
        &permutation,
        &[],
        &[],
        &[],
        &[],
        &[],
        &fixed,
        Some(&application),
    )
    .unwrap();
    fixed[2].row_index = 2;
    assert!(super::super::validate_row_coverage(
        &layout(),
        &permutation,
        &[],
        &[],
        &[],
        &[],
        &[],
        &fixed,
        Some(&application)
    )
    .is_err());
}

#[test]
fn native_loader_rejects_dynamic_source_payload_before_fixed_decoding() {
    for field in [10, 11, 12] {
        let mut source = vec![json!(null); 14];
        for slot in [10, 11, 12] {
            source[slot] = json!([]);
        }
        source[field] = if field == 10 {
            json!([[9, [], []]])
        } else {
            json!([[2, [], [], []]])
        };
        // Other source fields are deliberately absent: the dynamic-slot
        // error must occur before copying/decoding the fixed source payload.
        let fixed = json!([6, source, [], metadata(), [], [4, 1], 270]);
        assert!(matches!(
            super::super::load_prepared_application_records(fixed, records(4, 4)),
            Err(PackageError::Invalid(
                "native application dynamic source slots are not empty"
            ))
        ));
    }
}
