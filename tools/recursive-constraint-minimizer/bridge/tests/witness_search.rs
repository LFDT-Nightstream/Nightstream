use nightstream_constraint_exporter::find_exclusive_column_witness;
use recursive_constraint_minimizer::{Problem, Row, Scope, Source, Term, GOLDILOCKS_MODULUS, PROBLEM_SCHEMA};

fn term(column: usize) -> Term {
    Term {
        column,
        coefficient: "1".to_owned(),
    }
}

fn problem(rows: Vec<Row>, column_count: usize, complete_families: Vec<&str>) -> Problem {
    Problem {
        schema: PROBLEM_SCHEMA.to_owned(),
        source: Source {
            profile: "witness-search-test".to_owned(),
            artifact_digest: "test-only".to_owned(),
            scope: Scope::Branch,
            total_rows: rows.len(),
        },
        field_modulus: GOLDILOCKS_MODULUS.to_owned(),
        column_count,
        constant_one_column: 0,
        public_input_count: 1,
        complete_families: complete_families.into_iter().map(str::to_owned).collect(),
        rows,
    }
}

fn zero_row(id: &str, source_index: usize, family: &str, column: usize) -> Row {
    Row {
        id: id.to_owned(),
        source_index,
        family: family.to_owned(),
        a: vec![term(column)],
        b: vec![term(0)],
        c: Vec::new(),
    }
}

#[test]
fn finds_a_witness_through_an_exclusive_column() {
    let problem = problem(
        vec![zero_row("shared", 0, "shared", 1), zero_row("gadget", 1, "gadget", 2)],
        3,
        vec!["gadget", "shared"],
    );
    let witness = find_exclusive_column_witness(&problem, &[1, 0, 0], "gadget")
        .expect("search must run")
        .expect("column 2 is exclusive to the gadget family");

    assert_eq!(witness.family(), "gadget");
    assert_eq!(witness.column(), 2);
    assert_eq!(witness.violated_rows(), [1]);
    assert_eq!(witness.model().values()[0], 1);
    assert_eq!(witness.model().values()[1], 0);
    assert_ne!(witness.model().values()[2], 0);
}

#[test]
fn returns_none_when_every_family_column_is_shared() {
    let shared_user = Row {
        id: "user".to_owned(),
        source_index: 1,
        family: "gadget".to_owned(),
        a: vec![term(1)],
        b: vec![term(0)],
        c: Vec::new(),
    };
    let problem = problem(
        vec![zero_row("shared", 0, "shared", 1), shared_user],
        2,
        vec!["gadget", "shared"],
    );
    let witness = find_exclusive_column_witness(&problem, &[1, 0], "gadget").expect("search must run");
    assert!(witness.is_none());
}

#[test]
fn rejects_a_background_that_violates_the_relation() {
    let problem = problem(
        vec![zero_row("shared", 0, "shared", 1), zero_row("gadget", 1, "gadget", 2)],
        3,
        vec!["gadget", "shared"],
    );
    let error = find_exclusive_column_witness(&problem, &[1, 5, 0], "gadget")
        .expect_err("a violating background must be rejected");
    assert!(error.to_string().contains("violates source row"));
}

#[test]
fn rejects_a_background_without_constant_one() {
    let problem = problem(
        vec![zero_row("shared", 0, "shared", 1), zero_row("gadget", 1, "gadget", 2)],
        3,
        vec!["gadget", "shared"],
    );
    let error = find_exclusive_column_witness(&problem, &[0, 0, 0], "gadget")
        .expect_err("the normalized assignment boundary must be checked");
    assert!(error.to_string().contains("constant-one"));
}

#[test]
fn rejects_an_incomplete_family_ledger() {
    let problem = problem(
        vec![zero_row("shared", 0, "shared", 1), zero_row("gadget", 1, "gadget", 2)],
        3,
        vec!["gadget"],
    );
    let error = find_exclusive_column_witness(&problem, &[1, 0, 0], "gadget")
        .expect_err("all source families must be authoritative");
    assert!(error.to_string().contains("exact family ledger"));
}

#[test]
fn rejects_an_unknown_family() {
    let problem = problem(
        vec![zero_row("shared", 0, "shared", 1), zero_row("gadget", 1, "gadget", 2)],
        3,
        vec!["gadget", "shared"],
    );
    let error = find_exclusive_column_witness(&problem, &[1, 0, 0], "missing").expect_err("unknown family must fail");
    assert!(error.to_string().contains("not a complete family"));
}
