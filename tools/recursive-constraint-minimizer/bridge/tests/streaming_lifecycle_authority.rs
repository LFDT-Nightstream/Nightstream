//! Exact base-lifecycle authority attack with full cvc5 and Rust replay.

use std::path::PathBuf;

use neo_fold_clean::frontends::nebula::f_prime::{
    prepare_streaming_lifecycle_preprocessing, synthesize_streaming_lifecycle_source_arms,
    NebulaFPrimeStreamingLifecycleArm, NebulaFPrimeStreamingLifecycleSourceArms,
};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::paper::f_prime::stage as fprime_stage;
use neo_fold_clean::paper::params::Params;
use neo_math::F;
use nightstream_constraint_exporter::{export_sparse_problem, sparse_family_census, ExportRequest};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use recursive_constraint_minimizer::{
    audit_complete_typed_candidate, row_is_satisfied, Conclusion, FieldModel, LinearCombination, Scope, Selection,
    SolverConfig, SolverMode, SolverStatus, Term, TypedTarget, TypedTargetRow, GOLDILOCKS_MODULUS,
};

const VERIFIER_ADVICE_COLUMNS: &str = "fprime.streaming.base.verifier_advice";

fn lifecycle_source(seed: u64) -> NebulaFPrimeStreamingLifecycleSourceArms {
    let reference_params = Params::production();
    let memory = NebulaParams::new(0, 0, 1, 2, 1).expect("one-step memory profile");
    let plan = NebulaPlan::new(memory, vec![7], [0xD9; 32], reference_params.kappa() as usize)
        .expect("Nebula plan");
    let params = Params::for_ccs_shape(
        plan.circuit().structure().n,
        plan.circuit().structure().m,
        plan.circuit().structure().t(),
        plan.circuit().structure().max_degree(),
    )
    .expect("shape-specific Appendix B.2 parameters");
    assert!(params.has_production_core());
    let log = neo_fold_clean::frontends::direct_ccs::ajtai::setup_seeded(
        &params,
        plan.circuit().structure(),
        seed,
    );
    let preprocessing =
        neo_fold_clean::lifecycle::preprocess_with_test_log(params, plan.circuit().structure().clone(), log, Some(648))
            .expect("verifier-owned lifecycle preprocessing");
    let preprocessing =
        prepare_streaming_lifecycle_preprocessing(preprocessing, &plan).expect("fixed streaming lifecycle policy");
    synthesize_streaming_lifecycle_source_arms(&preprocessing, &plan).expect("exact streaming lifecycle source rows")
}

fn fixed_value_row(id: String, one: usize, column: usize, value: u64) -> TypedTargetRow {
    let modulus = GOLDILOCKS_MODULUS
        .parse::<u64>()
        .expect("Goldilocks modulus");
    let mut a = LinearCombination::new();
    if value != 0 {
        a.push(Term {
            column: one,
            coefficient: (modulus - value).to_string(),
        });
    }
    a.push(Term {
        column,
        coefficient: "1".to_owned(),
    });
    TypedTargetRow {
        id,
        a,
        b: vec![Term {
            column: one,
            coefficient: "1".to_owned(),
        }],
        c: Vec::new(),
    }
}

#[test]
#[ignore = "expensive exact lifecycle cvc5 authority attack"]
fn private_verifier_advice_admits_a_replayed_base_counterexample() {
    let lifecycle = lifecycle_source(0x5354_5245_414d);
    let source = lifecycle.arm(NebulaFPrimeStreamingLifecycleArm::Base);
    source
        .is_satisfied_by(lifecycle.base_assignment())
        .expect("checked base background");

    let families = sparse_family_census(source).expect("exclusive physical-stage ownership");
    let problem = export_sparse_problem(
        source,
        ExportRequest {
            profile: "nightstream/goldilocks/streaming-base-authority-attack/v1".to_owned(),
            scope: Scope::Lifecycle,
            public_input_count: source.m_in,
            source_rows: (0..source.n).collect(),
            complete_families: families
                .iter()
                .map(|family| family.name().to_owned())
                .collect(),
        },
    )
    .expect("complete exact base source export");

    let advice = source
        .column_family_ranges()
        .iter()
        .find(|range| range.name == VERIFIER_ADVICE_COLUMNS)
        .expect("verifier advice column family");
    assert!(advice.column_end - advice.column_start >= 12);
    let advice_columns = advice.column_start..advice.column_start + 12;
    let background = lifecycle
        .base_assignment()
        .iter()
        .map(|value| value.as_canonical_u64())
        .collect::<Vec<_>>();
    let target = TypedTarget {
        id: "nightstream.streaming.base.verifier_advice.authority".to_owned(),
        column_count: problem.column_count,
        rows: advice_columns
            .clone()
            .enumerate()
            .map(|(index, column)| {
                fixed_value_row(
                    format!("verifier_advice.field.{index}"),
                    problem.constant_one_column,
                    column,
                    background[column],
                )
            })
            .collect(),
    };

    let mut candidate_values = background;
    let mutated_column = advice_columns.start;
    candidate_values[mutated_column] = F::ONE.as_canonical_u64();
    assert_ne!(
        candidate_values[mutated_column],
        lifecycle.base_assignment()[mutated_column].as_canonical_u64()
    );
    let candidate = FieldModel::from_canonical_values(candidate_values).expect("canonical attack assignment");

    let mut violated_selected_rows = Vec::new();
    for row in &problem.rows {
        if row_is_satisfied(row, &candidate).expect("exact Rust row replay") {
            continue;
        }
        assert_eq!(
            row.family,
            fprime_stage::BASE_VERIFIER_KEY,
            "candidate must satisfy every retained physical stage"
        );
        violated_selected_rows.push(row.source_index);
    }
    assert!(!violated_selected_rows.is_empty());

    eprintln!(
        "streaming base authority query: rows={} columns={} family={} violated_selected={:?}",
        problem.rows.len(),
        problem.column_count,
        fprime_stage::BASE_VERIFIER_KEY,
        violated_selected_rows,
    );
    let report = audit_complete_typed_candidate(
        &problem,
        &Selection::Family(fprime_stage::BASE_VERIFIER_KEY.to_owned()),
        &target,
        &candidate,
        &SolverConfig {
            executable: PathBuf::from("/Users/nijaar/.local/bin/cvc5"),
            mode: SolverMode::Split,
            timeout_ms: 120_000,
        },
    )
    .expect("bounded strict cvc5 candidate audit");

    eprintln!(
        "streaming base authority attack: rows={} columns={} family={} cvc5={:?} replayed={} violated_target={:?} violated_selected={:?}",
        problem.rows.len(),
        problem.column_count,
        fprime_stage::BASE_VERIFIER_KEY,
        report.solver_run.status,
        report.retained_rows_replayed.len(),
        report.violated_target_rows,
        violated_selected_rows,
    );
    assert_eq!(report.solver_run.status, SolverStatus::Sat);
    assert_eq!(report.conclusion, Conclusion::CounterexampleCandidate);
    assert_eq!(
        report.retained_rows_replayed.len(),
        problem
            .rows
            .iter()
            .filter(|row| row.family != fprime_stage::BASE_VERIFIER_KEY)
            .count()
    );
    assert_eq!(report.violated_target_rows, [0]);
    assert_eq!(report.model.expect("full cvc5 model"), candidate);
}

#[test]
#[ignore = "expensive exact lifecycle verifier-authority regression"]
fn verifier_advice_is_bound_to_preprocessing_constants() {
    let baseline = lifecycle_source(0x5354_5245_414d);
    let baseline_assignment = baseline.base_assignment().to_vec();
    let [baseline_source, _] = baseline.into_arms();

    let alternate = lifecycle_source(0x5354_5245_414e);
    let alternate_assignment = alternate.base_assignment().to_vec();
    let [alternate_source, _] = alternate.into_arms();

    baseline_source
        .is_satisfied_by(&baseline_assignment)
        .expect("baseline assignment");
    alternate_source
        .is_satisfied_by(&alternate_assignment)
        .expect("alternate assignment");
    assert_eq!(
        (baseline_source.n, baseline_source.m, baseline_source.m_in),
        (alternate_source.n, alternate_source.m, alternate_source.m_in)
    );
    assert_ne!(baseline_assignment, alternate_assignment);
    assert!(
        baseline_source.is_satisfied_by(&alternate_assignment).is_err(),
        "a lifecycle relation must reject advice from a different verifier preprocessing"
    );
}
