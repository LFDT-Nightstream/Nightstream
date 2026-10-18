use recursive_constraint_minimizer::{
    derive_scalar_certificate, validate_scalar_certificate, Problem, Residue, ScalarSearch, Selection, Term,
};

fn fixture() -> Problem {
    serde_json::from_str(include_str!("../examples/known-local.json")).expect("valid fixture")
}

#[test]
fn derives_and_rechecks_duplicate_row_certificate() {
    let problem = fixture();
    let ScalarSearch::Certified(certificate) =
        derive_scalar_certificate(&problem, &Selection::Row("zero_copy".to_owned())).expect("certificate search")
    else {
        panic!("duplicate row is in the scalar span");
    };

    assert_eq!(certificate.rows.len(), 1);
    assert_eq!(certificate.rows[0].candidate_source_index, 2);
    assert_eq!(certificate.rows[0].support.len(), 1);
    assert_eq!(certificate.rows[0].support[0].source_index, 1);
    assert_eq!(certificate.rows[0].support[0].coefficient.get(), 1);
    validate_scalar_certificate(&problem, &certificate).expect("valid certificate");
}

#[test]
fn reports_unsupported_certificate_grammar_without_claiming_necessity() {
    let problem = fixture();
    let result =
        derive_scalar_certificate(&problem, &Selection::Family("zero".to_owned())).expect("certificate search");
    assert_eq!(result, ScalarSearch::OutsideGrammar);
}

#[test]
fn rejects_changed_certificate_coefficient() {
    let problem = fixture();
    let ScalarSearch::Certified(mut certificate) =
        derive_scalar_certificate(&problem, &Selection::Row("zero_copy".to_owned())).expect("certificate search")
    else {
        panic!("duplicate certificate");
    };
    certificate.rows[0].support[0].coefficient = Residue::new(2).expect("canonical residue");
    let error = validate_scalar_certificate(&problem, &certificate).expect_err("must reject drift");
    assert!(error.to_string().contains("identity failed"));
}

#[test]
fn residue_keeps_the_decimal_string_schema_and_rejects_noncanonical_text() {
    let term: Term = serde_json::from_str(r#"{"column":3,"coefficient":"18446744069414584320"}"#).expect("canonical");
    assert_eq!(term.coefficient.get(), 18446744069414584320);
    assert_eq!(
        serde_json::to_string(&term).expect("serialize"),
        r#"{"column":3,"coefficient":"18446744069414584320"}"#
    );
    for text in ["0", "01", "+1", "18446744069414584321", "-1", "x"] {
        let json = format!(r#"{{"column":3,"coefficient":"{text}"}}"#);
        assert!(serde_json::from_str::<Term>(&json).is_err(), "{text}");
    }
}
