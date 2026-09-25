use recursive_constraint_minimizer::{
    parse_model, parse_model_with_defaults, row_is_satisfied, FieldModel, Problem, GOLDILOCKS_MODULUS,
};

#[test]
fn builds_a_model_only_from_canonical_values() {
    let model = FieldModel::from_canonical_values(vec![1, 2]).expect("canonical model");
    assert_eq!(model.values(), [1, 2]);

    assert!(FieldModel::from_canonical_values(Vec::new()).is_err());
    let modulus = GOLDILOCKS_MODULUS
        .parse::<u64>()
        .expect("Goldilocks modulus");
    assert!(FieldModel::from_canonical_values(vec![modulus]).is_err());
}

fn fixture() -> Problem {
    serde_json::from_str(include_str!("../examples/known-local.json")).expect("valid fixture")
}

#[test]
fn parses_cvc5_hash_finite_field_values() {
    let modulus = recursive_constraint_minimizer::GOLDILOCKS_MODULUS;
    let stdout = format!(
        "sat\n(\n(define-fun x_0 () (_ FiniteField {modulus}) #f1m{modulus})\n\
         (define-fun x_1 () (_ FiniteField {modulus}) #f0m{modulus})\n)\n"
    );
    let model = parse_model(&stdout, 2).expect("valid model");
    assert_eq!(model.values(), [1, 0]);
    assert!(fixture()
        .rows
        .iter()
        .all(|row| row_is_satisfied(row, &model).expect("valid row")));
}

#[test]
fn parses_as_finite_field_values_and_normalizes_negative_residues() {
    let stdout = "sat\n(model (define-fun x_0 () F (as ff1 F)) \
                  (define-fun x_1 () F (as ff-1 F)))\n";
    let model = parse_model(stdout, 2).expect("valid model");
    assert_eq!(model.values()[0], 1);
    assert_eq!(
        model.values()[1],
        recursive_constraint_minimizer::GOLDILOCKS_MODULUS
            .parse::<u64>()
            .expect("u64 modulus")
            - 1
    );
}

#[test]
fn rejects_missing_or_wrong_field_assignments() {
    let missing = parse_model("sat\n(define-fun x_0 () F (as ff1 F))", 2).expect_err("must reject missing column");
    assert!(missing.to_string().contains("does not define x_1"));

    let wrong_modulus = parse_model("sat\n(define-fun x_0 () F #f1m13)", 1).expect_err("must reject wrong field");
    assert!(wrong_modulus.to_string().contains("wrong modulus"));
}

#[test]
fn fills_undeclared_columns_from_an_exact_background() {
    let stdout = "sat\n(model (define-fun x_0 () F (as ff1 F)) \
                  (define-fun x_2 () F (as ff9 F)))\n";
    let model = parse_model_with_defaults(stdout, &[1, 7, 3], &[0, 2]).expect("bounded model");
    assert_eq!(model.values(), [1, 7, 9]);

    let error = parse_model_with_defaults(stdout, &[1, 7, 3], &[0]).expect_err("must reject undeclared output");
    assert!(error.to_string().contains("undeclared source column x_2"));
}
