use recursive_constraint_minimizer::{render_query, Problem, Selection};

fn fixture() -> Problem {
    serde_json::from_str(include_str!("../examples/known-local.json")).expect("valid fixture")
}

#[test]
fn duplicate_row_query_keeps_the_original_row() {
    let query = render_query(&fixture(), &Selection::Row("zero_copy".to_owned())).expect("valid query");
    assert_eq!(query.retained_rows.len(), 2);
    assert_eq!(query.removed_rows.len(), 1);
    assert_eq!(query.removed_rows[0].problem_index, 2);
    assert_eq!(query.removed_rows[0].source_index, 2);
    assert!(query.smt2.contains(":named keep_1"));
    assert!(query.smt2.contains(":named candidate_violation"));
    assert!(!query.smt2.contains("(or (not"));
}

#[test]
fn family_query_asks_for_any_removed_row_violation() {
    let query = render_query(&fixture(), &Selection::Family("zero".to_owned())).expect("valid query");
    assert_eq!(query.retained_rows.len(), 1);
    assert_eq!(query.removed_rows.len(), 2);
    assert!(query.smt2.contains("(or (not"));
    assert!(query.smt2.contains("(_ FiniteField 18446744069414584321)"));
}

#[test]
fn missing_selection_fails_closed() {
    let error = render_query(&fixture(), &Selection::Row("absent".to_owned())).expect_err("must reject");
    assert!(error.to_string().contains("does not match"));
}

#[test]
fn incomplete_family_selection_fails_closed() {
    let mut problem = fixture();
    problem.complete_families.retain(|family| family != "zero");
    let error = render_query(&problem, &Selection::Family("zero".to_owned())).expect_err("must reject");
    assert!(error.to_string().contains("is not complete"));
}

#[test]
fn duplicate_source_index_fails_closed() {
    let mut problem = fixture();
    problem.rows[2].source_index = problem.rows[1].source_index;
    let error = problem.validate().expect_err("must reject");
    assert!(error.to_string().contains("source_index"));
}

#[test]
fn invalid_public_prefix_fails_closed() {
    let mut problem = fixture();
    problem.public_input_count = problem.column_count + 1;
    let error = problem
        .validate()
        .expect_err("must reject public-prefix drift");
    assert!(error.to_string().contains("public_input_count"));
}

#[test]
fn declares_only_columns_reached_by_the_bounded_rows() {
    let mut problem = fixture();
    problem.column_count = 3;
    let query = render_query(&problem, &Selection::Row("zero_copy".to_owned())).expect("bounded query");
    assert_eq!(query.model_columns, [0, 1]);
    assert!(!query.smt2.contains("declare-const x_2"));
}
