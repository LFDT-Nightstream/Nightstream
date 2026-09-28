use recursive_constraint_minimizer::{derive_scalar_certificate, validate_scalar_certificate, Problem, Selection};

fn fixture() -> Problem {
    serde_json::from_str(include_str!("../examples/known-local.json")).expect("valid fixture")
}

#[test]
fn derives_and_rechecks_duplicate_row_certificate() {
    let problem = fixture();
    let certificate = derive_scalar_certificate(&problem, &Selection::Row("zero_copy".to_owned()))
        .expect("certificate search")
        .expect("duplicate row is in the scalar span");

    assert_eq!(certificate.rows.len(), 1);
    assert_eq!(certificate.rows[0].candidate_source_index, 2);
    assert_eq!(certificate.rows[0].support.len(), 1);
    assert_eq!(certificate.rows[0].support[0].source_index, 1);
    assert_eq!(certificate.rows[0].support[0].coefficient, "1");
    validate_scalar_certificate(&problem, &certificate).expect("valid certificate");
}

#[test]
fn reports_unsupported_certificate_grammar_without_claiming_necessity() {
    let problem = fixture();
    let result =
        derive_scalar_certificate(&problem, &Selection::Family("zero".to_owned())).expect("certificate search");
    assert!(result.is_none());
}

#[test]
fn rejects_changed_certificate_coefficient() {
    let problem = fixture();
    let mut certificate = derive_scalar_certificate(&problem, &Selection::Row("zero_copy".to_owned()))
        .expect("certificate search")
        .expect("duplicate certificate");
    certificate.rows[0].support[0].coefficient = "2".to_owned();
    let error = validate_scalar_certificate(&problem, &certificate).expect_err("must reject drift");
    assert!(error.to_string().contains("identity failed"));
}
