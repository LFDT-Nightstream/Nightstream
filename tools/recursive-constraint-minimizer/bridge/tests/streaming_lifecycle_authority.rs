//! Exact base-lifecycle authority attack with full cvc5 and Rust replay.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use neo_fold_clean::frontends::nebula::f_prime::{
    prepare_streaming_lifecycle_preprocessing, production_streaming_lifecycle_profile,
    synthesize_streaming_lifecycle_source_arms, NebulaFPrimeStreamingLifecycleArm,
    NebulaFPrimeStreamingLifecycleSourceArms,
};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::r1cs_f_prime::build_multi_branch_selective_low_norm_r1cs_with_alignment;
use neo_fold_clean::paper::digest::digest32_as_fields;
use neo_fold_clean::paper::f_prime::stage as fprime_stage;
use neo_fold_clean::paper::params::Params;
use neo_math::{D, F};
use nightstream_constraint_exporter::{
    export_complete_streaming_lifecycle_problem, export_sparse_problem, sparse_family_census,
    streaming_lifecycle_x_out_typed_target, ExportRequest,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use recursive_constraint_minimizer::{
    audit_complete_typed_candidate, row_is_satisfied, typed_target_row_is_satisfied, Conclusion, FieldModel,
    LinearCombination, Scope, Selection, SolverConfig, SolverMode, SolverStatus, Term, TypedTarget, TypedTargetRow,
    GOLDILOCKS_MODULUS,
};

const VERIFIER_ADVICE_COLUMNS: &str = "fprime.streaming.base.verifier_advice";
const BASE_VERIFIER_KEY_OMISSION_ARTIFACT_PATH: &str = "../../../formal/nightstream-lean/Nightstream/Implementation/\
R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingLifecycleBaseVerifierKeyOmission.lean";

struct LifecycleFixture {
    lifecycle: NebulaFPrimeStreamingLifecycleSourceArms,
    verifier_digest: [F; 4],
    pi_ccs_header: [F; 4],
}

fn lifecycle_source(seed: u64) -> LifecycleFixture {
    let reference_params = Params::production();
    let memory = NebulaParams::new(0, 0, 1, 2, 1).expect("one-step memory profile");
    let plan = NebulaPlan::new(memory, vec![7], [0xD9; 32], reference_params.kappa() as usize).expect("Nebula plan");
    let params = Params::for_ccs_shape(
        plan.circuit().structure().n,
        plan.circuit().structure().m,
        plan.circuit().structure().t(),
        plan.circuit().structure().max_degree(),
    )
    .expect("shape-specific Nightstream Goldilocks k_rho=16 parameters");
    assert!(params.has_production_core());
    let log = neo_fold_clean::frontends::direct_ccs::ajtai::setup_seeded(&params, plan.circuit().structure(), seed);
    let preprocessing =
        neo_fold_clean::lifecycle::preprocess_with_test_log(params, plan.circuit().structure().clone(), log, Some(648))
            .expect("verifier-owned lifecycle preprocessing");
    let preprocessing =
        prepare_streaming_lifecycle_preprocessing(preprocessing, &plan).expect("fixed streaming lifecycle policy");
    let verifier_digest = digest32_as_fields(preprocessing.vk.digest());
    let pi_ccs_header = preprocessing.pi_ccs_header_bundle();
    let lifecycle = synthesize_streaming_lifecycle_source_arms(&preprocessing, &plan)
        .expect("exact streaming lifecycle source rows");
    LifecycleFixture {
        lifecycle,
        verifier_digest,
        pi_ccs_header,
    }
}

fn after_x_out_verifier_context(fixture: &LifecycleFixture, assignment: &[F]) -> ([F; 4], [F; 4]) {
    let columns = fixture
        .lifecycle
        .x_out_preimage_columns(NebulaFPrimeStreamingLifecycleArm::Base)
        .after();
    let verifier_digest = std::array::from_fn(|index| assignment[columns[1 + index]]);
    let pi_ccs_header = std::array::from_fn(|index| assignment[columns[5 + index]]);
    (verifier_digest, pi_ccs_header)
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

fn lean_range(range: std::ops::Range<usize>) -> String {
    format!("{{ start := {}, stop := {} }}", range.start, range.end)
}

fn lean_option(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_owned(), |value| format!("some {value}"))
}

fn render_base_verifier_key_omission_artifact() -> String {
    let fixture = lifecycle_source(0x5354_5245_414d);
    let lifecycle = &fixture.lifecycle;
    let source = lifecycle.arm(NebulaFPrimeStreamingLifecycleArm::Base);
    let source_arms = [
        source.clone(),
        lifecycle
            .arm(NebulaFPrimeStreamingLifecycleArm::Recursive)
            .clone(),
    ];
    let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&source_arms, 0, D, 0)
        .expect("exact two-arm selective lifecycle relation");
    let profile = production_streaming_lifecycle_profile(lifecycle, &relation)
        .expect("exact lifecycle source-to-selective profile");
    let problem_export = export_complete_streaming_lifecycle_problem(
        lifecycle,
        &relation,
        NebulaFPrimeStreamingLifecycleArm::Base,
        "nightstream/goldilocks/streaming-base-authority-attack/v1",
    )
    .expect("complete base lifecycle problem export");
    let problem = problem_export.problem();
    let arm_profile = profile.arm(NebulaFPrimeStreamingLifecycleArm::Base);
    let stages = arm_profile
        .stages()
        .iter()
        .filter(|stage| stage.path() == fprime_stage::BASE_VERIFIER_KEY)
        .collect::<Vec<_>>();
    let [stage] = stages.as_slice() else {
        panic!("base lifecycle arm must contain exactly one verifier-key stage")
    };
    let advice = source
        .column_family_ranges()
        .iter()
        .find(|range| range.name == VERIFIER_ADVICE_COLUMNS)
        .expect("base verifier advice columns");
    let changed_column = advice.column_start;
    let baseline_value = lifecycle.base_assignment()[changed_column].as_canonical_u64();
    let candidate_value = F::ONE.as_canonical_u64();
    assert_ne!(baseline_value, candidate_value);
    assert_eq!(problem.constant_one_column, 0);

    let selected_row_count = problem
        .rows
        .iter()
        .filter(|row| row.family == fprime_stage::BASE_VERIFIER_KEY)
        .count();
    let retained_row_count = problem.rows.len() - selected_row_count;
    let mut occurrences = Vec::new();
    for row in &problem.rows {
        for (side, terms) in [("a", &row.a), ("b", &row.b), ("c", &row.c)] {
            for term in terms.iter().filter(|term| term.column == changed_column) {
                assert_eq!(
                    row.family,
                    fprime_stage::BASE_VERIFIER_KEY,
                    "changed verifier advice column escapes the selected family",
                );
                occurrences.push((row.source_index, side, term.coefficient.as_str(), row.family.as_str()));
            }
        }
    }
    assert!(!occurrences.is_empty());

    let mut source_runs = String::from("[\n");
    let mut source_run_proof = String::from("by\n  unfold sourceRuns\n  exact ");
    let mut source_cursor = stage.source_rows().start;
    let mut source_run_count = 0usize;
    for run in stage.source_runs() {
        let rows = run.source_rows();
        assert_eq!(rows.start, source_cursor);
        writeln!(
            source_runs,
            "    {{ sourceRows := {}, disposition := \"{:?}\", emittedStart := {} }},",
            lean_range(rows.clone()),
            run.disposition(),
            lean_option(run.emitted_start()),
        )
        .unwrap();
        source_run_proof.push_str("SourceRunChain.cons rfl (by decide)\n    (");
        source_cursor = rows.end;
        source_run_count += 1;
    }
    assert_eq!(source_cursor, stage.source_rows().end);
    source_run_proof.push_str(&format!("SourceRunChain.nil {source_cursor}"));
    source_run_proof.extend(std::iter::repeat_n(')', source_run_count));
    source_runs.push_str("  ]");

    let mut final_runs = String::from("[\n");
    let mut final_run_proof = String::from("by\n  unfold finalRuns\n  exact ");
    let mut final_run_count = 0usize;
    for run in stage.final_row_runs() {
        let rows = run.rows();
        writeln!(
            final_runs,
            "    {{ family := \"{:?}\", rows := {}, rewriteId := {} }},",
            run.family(),
            lean_range(rows),
            lean_option(run.rewrite_id()),
        )
        .unwrap();
        final_run_proof.push_str("FinalRunsWithin.cons (by decide) (by decide)\n    (");
        final_run_count += 1;
    }
    final_run_proof.push_str("FinalRunsWithin.nil");
    final_run_proof.extend(std::iter::repeat_n(')', final_run_count));
    final_runs.push_str("  ]");

    let mut occurrence_values = String::from("[\n");
    let mut occurrence_proof = String::from("by\n  unfold occurrences\n  exact ");
    for (source_row, side, coefficient, family) in &occurrences {
        writeln!(
            occurrence_values,
            "    {{ sourceRow := {source_row}, side := .{side}, coefficient := {coefficient}, family := \"{family}\" }},",
        )
        .unwrap();
        occurrence_proof.push_str("OccurrenceOwnership.cons rfl\n    (");
    }
    occurrence_proof.push_str("OccurrenceOwnership.nil");
    occurrence_proof.extend(std::iter::repeat_n(')', occurrences.len()));
    occurrence_values.push_str("  ]");

    let payload = format!(
        "def sourceRuns : List SourceRun := {source_runs}\n\n\
         def finalRuns : List FinalRun := {final_runs}\n\n\
         def occurrences : List Occurrence := {occurrence_values}\n\n\
         def rawArtifact : Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleBaseVerifierKeyOmission.Artifact.RawArtifact :=\n  \
         {{ schemaVersion := 1,\n    \
            profileId := \"nightstream/goldilocks/streaming-base-authority-attack/v1\",\n    \
            lifecycleScope := \"{}\",\n    \
            sourceArtifactIdentity := \"{}\",\n    \
            finalArtifactIdentity := \"{}\",\n    \
            family := \"{}\", stagePath := \"{}\", occurrence := {},\n    \
            sourceRows := {}, sourceColumns := {},\n    \
            sourceRowCount := {}, selectedRowCount := {}, retainedRowCount := {},\n    \
            columnCount := {}, constantOneColumn := {}, changedColumn := {},\n    \
            baselineValue := {}, candidateValue := {},\n    \
            finalRowCount := {}, sourceRuns := sourceRuns, finalRuns := finalRuns,\n    \
            occurrences := occurrences }}\n\n\
         theorem sourceRuns_cover : SourceRunChain {} sourceRuns {} :=\n{}\n\n\
         theorem finalRuns_inside : FinalRunsWithin {} finalRuns :=\n{}\n\n\
         theorem occurrences_owned : OccurrenceOwnership \"{}\" occurrences :=\n{}\n",
        arm_profile.lifecycle_scope(),
        arm_profile.source_artifact_identity(),
        profile.final_artifact_identity(),
        fprime_stage::BASE_VERIFIER_KEY,
        stage.path(),
        stage.occurrence(),
        lean_range(stage.source_rows()),
        lean_range(stage.source_columns()),
        problem.rows.len(),
        selected_row_count,
        retained_row_count,
        problem.column_count,
        problem.constant_one_column,
        changed_column,
        baseline_value,
        candidate_value,
        profile.final_rows(),
        stage.source_rows().start,
        stage.source_rows().end,
        source_run_proof,
        profile.final_rows(),
        final_run_proof,
        fprime_stage::BASE_VERIFIER_KEY,
        occurrence_proof,
    );
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyOmissionSchema\n\n\
         /-! Generated exhaustive changed-column projection and source-to-final ownership for the base verifier-key omission audit.\n\n\
         Rust remains the authority for the full rows and assignment. This leaf contains no digest-based row claim.\n\n\
         Emits constraints: no.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleBaseVerifierKeyOmission\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRecursiveVerifierKey.Artifact\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleBaseVerifierKeyOmission.Artifact\n\n\
         {payload}\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleBaseVerifierKeyOmission\n",
    )
}

fn base_verifier_key_omission_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(BASE_VERIFIER_KEY_OMISSION_ARTIFACT_PATH)
}

#[test]
#[ignore = "exact base verifier-key omission projection"]
fn base_verifier_key_omission_artifact_is_current() {
    let path = base_verifier_key_omission_artifact_path();
    let rendered = render_base_verifier_key_omission_artifact();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        let expected = path.with_extension("lean.expected");
        std::fs::write(&expected, rendered).expect("write expected base verifier-key omission artifact");
        panic!(
            "base verifier-key omission Lean artifact drifted; inspect {}",
            expected.display()
        );
    }
}

#[test]
#[ignore = "deliberately writes the reviewed base verifier-key omission artifact"]
fn regenerate_base_verifier_key_omission_artifact() {
    std::fs::write(
        base_verifier_key_omission_artifact_path(),
        render_base_verifier_key_omission_artifact(),
    )
    .expect("write generated base verifier-key omission artifact");
}

#[test]
#[ignore = "expensive exact lifecycle target construction"]
fn complete_x_out_typed_targets_match_exact_rust_assignments() {
    let fixture = lifecycle_source(0x5354_5245_414d);
    for arm in [
        NebulaFPrimeStreamingLifecycleArm::Base,
        NebulaFPrimeStreamingLifecycleArm::Recursive,
    ] {
        let target = streaming_lifecycle_x_out_typed_target(&fixture.lifecycle, arm)
            .expect("complete lifecycle XOut target");
        assert_eq!(target.column_count, fixture.lifecycle.arm(arm).m);
        assert_eq!(target.rows.len(), 64);
        let columns = fixture.lifecycle.x_out_preimage_columns(arm);
        let values = fixture.lifecycle.x_out_preimage_values(arm);
        for (state, offset, state_columns, state_values) in [
            ("before", 0, columns.before(), values.before()),
            ("after", 32, columns.after(), values.after()),
        ] {
            for (field, (&column, &value)) in state_columns.iter().zip(state_values).enumerate() {
                assert_eq!(
                    target.rows[offset + field],
                    fixed_value_row(
                        format!("{state}.x_out.field.{field}"),
                        0,
                        column,
                        value.as_canonical_u64(),
                    )
                );
            }
        }
    }

    let base_model = FieldModel::from_canonical_values(
        fixture
            .lifecycle
            .base_assignment()
            .iter()
            .map(|value| value.as_canonical_u64())
            .collect(),
    )
    .expect("canonical base assignment");
    let base_target = streaming_lifecycle_x_out_typed_target(
        &fixture.lifecycle,
        NebulaFPrimeStreamingLifecycleArm::Base,
    )
    .expect("complete base XOut target");
    assert!(base_target
        .rows
        .iter()
        .all(|row| typed_target_row_is_satisfied(row, &base_model).expect("base target replay")));
}

#[test]
#[ignore = "expensive exact lifecycle cvc5 authority attack"]
fn private_verifier_advice_admits_a_replayed_base_counterexample() {
    let fixture = lifecycle_source(0x5354_5245_414d);
    let lifecycle = &fixture.lifecycle;
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
fn verifier_advice_is_bound_by_the_terminal_public_image() {
    let baseline = lifecycle_source(0x5354_5245_414d);
    let alternate = lifecycle_source(0x5354_5245_414e);
    let baseline_assignment = baseline.lifecycle.base_assignment();
    let alternate_assignment = alternate.lifecycle.base_assignment();
    let baseline_source = baseline
        .lifecycle
        .arm(NebulaFPrimeStreamingLifecycleArm::Base);
    let alternate_source = alternate
        .lifecycle
        .arm(NebulaFPrimeStreamingLifecycleArm::Base);

    baseline_source
        .is_satisfied_by(baseline_assignment)
        .expect("baseline assignment");
    alternate_source
        .is_satisfied_by(alternate_assignment)
        .expect("alternate assignment");
    assert_eq!(
        (baseline_source.n, baseline_source.m, baseline_source.m_in),
        (alternate_source.n, alternate_source.m, alternate_source.m_in)
    );
    assert_ne!(baseline_assignment, alternate_assignment);
    baseline_source
        .is_satisfied_by(alternate_assignment)
        .expect("verifier advice must not become self-referential coefficients");

    let baseline_context = after_x_out_verifier_context(&baseline, baseline_assignment);
    let alternate_context = after_x_out_verifier_context(&alternate, alternate_assignment);
    assert_eq!(baseline_context, (baseline.verifier_digest, baseline.pi_ccs_header));
    assert_eq!(alternate_context, (alternate.verifier_digest, alternate.pi_ccs_header));
    assert_ne!(baseline_context, alternate_context);
    assert_ne!(
        baseline.verifier_digest, alternate_context.0,
        "terminal public binding must reject a verifier digest from different preprocessing"
    );
}
