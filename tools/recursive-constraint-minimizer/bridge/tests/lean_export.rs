use nightstream_constraint_exporter::{
    render_redundancy_certificate_lean, render_removal_counterexample_lean,
    render_terminal_redundancy_certificate_lean, render_terminal_removal_counterexample_lean,
};
use recursive_constraint_minimizer::{
    derive_scalar_certificate, parse_model, Problem, Row, Scope, Selection, Source, Term, GOLDILOCKS_MODULUS,
    PROBLEM_SCHEMA,
};

fn duplicate_problem() -> Problem {
    let row = |id: &str, source_index: usize, family: &str| Row {
        id: id.to_owned(),
        source_index,
        family: family.to_owned(),
        a: vec![Term {
            column: 1,
            coefficient: "1".to_owned(),
        }],
        b: vec![Term {
            column: 0,
            coefficient: "1".to_owned(),
        }],
        c: Vec::new(),
    };
    Problem {
        schema: PROBLEM_SCHEMA.to_owned(),
        source: Source {
            profile: "lean-render-control".to_owned(),
            artifact_digest: "test-only".to_owned(),
            scope: Scope::Branch,
            total_rows: 2,
        },
        field_modulus: GOLDILOCKS_MODULUS.to_owned(),
        column_count: 2,
        constant_one_column: 0,
        public_input_count: 1,
        complete_families: vec!["duplicate".to_owned()],
        rows: vec![row("r0", 0, "retained"), row("r1", 1, "duplicate")],
    }
}

fn complete_duplicate_problem() -> Problem {
    let mut problem = duplicate_problem();
    problem.complete_families = vec!["duplicate".to_owned(), "retained".to_owned()];
    problem
}

#[test]
fn renders_a_rechecked_family_certificate_as_a_lean_proof() {
    let problem = duplicate_problem();
    let complete = complete_duplicate_problem();
    let certificate = derive_scalar_certificate(&problem, &Selection::Family("duplicate".to_owned()))
        .expect("derive scalar certificate")
        .expect("duplicate row has a scalar certificate");
    let lean = render_redundancy_certificate_lean(
        &complete,
        &problem,
        &certificate,
        "Generated.Artifact",
        "Generated.Artifact",
        "Generated.Certificate",
        &["retained".to_owned(), "duplicate".to_owned()],
    )
    .expect("render checked Lean certificate");

    assert!(lean.contains("import Generated.Artifact"));
    assert!(lean.contains("def familyCertificate : FamilyCertificate where"));
    assert!(lean.contains("family := \"duplicate\""));
    assert!(lean.contains("sourceIndex := 1, family := \"duplicate\""));
    assert!(lean.contains("sourceIndex := 0, family := \"retained\""));
    assert!(lean.contains("coefficient := (1 : Field)"));
    assert!(lean.contains("theorem familyCertificate_valid"));
    assert!(lean.contains("theorem redundant"));
    assert!(lean.contains("redundant_of_full_bound_valid"));
    assert!(lean.contains("boundArtifact_coversFullRelation"));
    assert!(lean.contains("theorem normalizedRedundant"));
    assert!(lean.contains("NormalizedFamilyHolds boundArtifact.source"));
    assert!(lean.contains("normalizedRedundant_of_redundant"));
}

#[test]
fn renders_a_rechecked_terminal_family_certificate_as_a_lean_proof() {
    let problem = duplicate_problem();
    let complete = complete_duplicate_problem();
    let certificate = derive_scalar_certificate(&problem, &Selection::Family("duplicate".to_owned()))
        .expect("derive scalar certificate")
        .expect("duplicate row has a scalar certificate");
    let lean = render_terminal_redundancy_certificate_lean(
        &complete,
        &problem,
        &certificate,
        "Generated.TerminalArtifact",
        "Generated.TerminalArtifact",
        "Generated.TerminalCertificate",
        &["retained".to_owned(), "duplicate".to_owned()],
    )
    .expect("render checked terminal Lean certificate");

    assert!(lean.contains("FamilyHolds terminalBoundArtifact.source"));
    assert!(lean.contains("redundant_of_full_terminal_bound_valid"));
    assert!(lean.contains("terminalBoundArtifact_coversFullRelation"));
    assert!(lean.contains("NormalizedFamilyHolds terminalBoundArtifact.source"));
    assert!(lean.contains("normalizedRedundant_of_redundant"));
}

#[test]
fn redundancy_renderer_rejects_a_query_row_that_differs_from_the_complete_relation() {
    let complete = complete_duplicate_problem();
    let mut query = duplicate_problem();
    let certificate = derive_scalar_certificate(&query, &Selection::Family("duplicate".to_owned()))
        .expect("derive scalar certificate")
        .expect("duplicate row has a scalar certificate");
    query.rows[0].id = "changed".to_owned();

    assert!(render_redundancy_certificate_lean(
        &complete,
        &query,
        &certificate,
        "Generated.Artifact",
        "Generated.Artifact",
        "Generated.Certificate",
        &["retained".to_owned(), "duplicate".to_owned()],
    )
    .unwrap_err()
    .to_string()
    .contains("differs from the complete source relation"));
}

fn necessary_problem() -> Problem {
    let one = Term {
        column: 0,
        coefficient: "1".to_owned(),
    };
    let x = Term {
        column: 1,
        coefficient: "1".to_owned(),
    };
    let minus_one = Term {
        column: 0,
        coefficient: "18446744069414584320".to_owned(),
    };
    Problem {
        schema: PROBLEM_SCHEMA.to_owned(),
        source: Source {
            profile: "lean-counterexample-control".to_owned(),
            artifact_digest: "test-only-complete".to_owned(),
            scope: Scope::Branch,
            total_rows: 2,
        },
        field_modulus: GOLDILOCKS_MODULUS.to_owned(),
        column_count: 2,
        constant_one_column: 0,
        public_input_count: 1,
        complete_families: vec!["bitness".to_owned(), "zero".to_owned()],
        rows: vec![
            Row {
                id: "r0".to_owned(),
                source_index: 0,
                family: "bitness".to_owned(),
                a: vec![x.clone()],
                b: vec![minus_one, x.clone()],
                c: Vec::new(),
            },
            Row {
                id: "r1".to_owned(),
                source_index: 1,
                family: "zero".to_owned(),
                a: vec![x],
                b: vec![one],
                c: Vec::new(),
            },
        ],
    }
}

fn one_model() -> recursive_constraint_minimizer::FieldModel {
    parse_model(
        &format!(
            "sat\n(\n(define-fun x_0 () (_ FiniteField {0}) #f1m{0})\n\
             (define-fun x_1 () (_ FiniteField {0}) #f1m{0})\n)\n",
            GOLDILOCKS_MODULUS
        ),
        2,
    )
    .expect("parse complete x = 1 model")
}

#[test]
fn renders_a_complete_checked_removal_counterexample() {
    let problem = necessary_problem();
    let model = one_model();
    let plan = ["bitness".to_owned(), "zero".to_owned()];
    let lean = render_removal_counterexample_lean(
        &problem,
        &model,
        "zero",
        "Generated.Artifact",
        "Generated.Artifact",
        "Generated.Counterexample",
        &plan,
    )
    .expect("render complete checked Lean counterexample");

    assert!(lean.contains("import Generated.Artifact"));
    assert!(lean.contains("removedFamily := \"zero\""));
    assert!(lean.contains("values := [1,1]"));
    assert!(lean.contains("theorem removalCounterexample_valid"));
    assert!(lean.contains("theorem necessary"));
    assert!(lean.contains("necessary_of_full_bound_valid"));
    assert!(lean.contains("boundArtifact_coversFullRelation"));
    assert!(lean.contains("theorem necessaryNormalized"));
    assert!(lean.contains("NormalizedTarget boundArtifact.source"));
    assert!(lean.contains("necessary_normalized_of_full_bound_valid"));

    let terminal = render_terminal_removal_counterexample_lean(
        &problem,
        &model,
        "zero",
        "Generated.TerminalArtifact",
        "Generated.TerminalArtifact",
        "Generated.TerminalCounterexample",
        &plan,
    )
    .expect("render complete checked terminal Lean counterexample");
    assert!(terminal.contains("terminalBoundArtifact.source"));
    assert!(terminal.contains("necessary_of_full_terminal_bound_valid"));
    assert!(terminal.contains("terminalBoundArtifact_coversFullRelation"));
    assert!(terminal.contains("NormalizedTarget terminalBoundArtifact.source"));
    assert!(terminal.contains("necessary_normalized_of_full_terminal_bound_valid"));
}

#[test]
fn removal_counterexample_rejects_an_incomplete_relation() {
    let mut problem = necessary_problem();
    problem.source.total_rows = 3;
    assert!(render_removal_counterexample_lean(
        &problem,
        &one_model(),
        "zero",
        "Generated.Artifact",
        "Generated.Artifact",
        "Generated.Counterexample",
        &["bitness".to_owned(), "zero".to_owned()],
    )
    .unwrap_err()
    .to_string()
    .contains("every source row"));
}
