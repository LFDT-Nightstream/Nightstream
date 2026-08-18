//! Bounded cvc5 attack for the compact lifecycle semantic link.

use std::path::PathBuf;

use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::frontends::nebula::f_prime::{
    enforce_streaming_lifecycle_semantic_link, streaming_phase_semantic_digest, StreamingLifecycleSemanticLinkWires,
    STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS, STREAMING_LIFECYCLE_PAYLOAD_DOMAIN_FAMILY,
    STREAMING_LIFECYCLE_SEMANTIC_LINK_FAMILY,
};
use neo_math::F;
use nightstream_constraint_exporter::{export_problem, ExportRequest};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use recursive_constraint_minimizer::{
    audit_complete_typed_candidate, row_is_satisfied, Conclusion, FieldModel, LinearCombination, Problem, Scope,
    Selection, SolverConfig, SolverMode, SolverStatus, Term, TypedTarget, TypedTargetRow, GOLDILOCKS_MODULUS,
};

struct BuiltLink {
    builder: R1csBuilder,
    semantic_columns: Vec<usize>,
    semantic_values: Vec<F>,
    payload_columns: Vec<usize>,
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

fn bit_row(id: String, one: usize, column: usize) -> TypedTargetRow {
    let modulus = GOLDILOCKS_MODULUS
        .parse::<u64>()
        .expect("Goldilocks modulus");
    TypedTargetRow {
        id,
        a: vec![Term {
            column,
            coefficient: "1".to_owned(),
        }],
        b: vec![
            Term {
                column: one,
                coefficient: (modulus - 1).to_string(),
            },
            Term {
                column,
                coefficient: "1".to_owned(),
            },
        ],
        c: Vec::new(),
    }
}

fn build_link(
    before_payload: [F; STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS],
    after_payload: [F; STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS],
) -> BuiltLink {
    let before_local = [F::from_u64(3); 4];
    let after_local = [F::from_u64(5); 4];
    let before_semantic = streaming_phase_semantic_digest(before_local, &before_payload);
    let after_semantic = streaming_phase_semantic_digest(after_local, &after_payload);

    let mut builder = R1csBuilder::new();
    let before_semantic_wires = before_semantic.map(|value| builder.alloc(value));
    let after_semantic_wires = after_semantic.map(|value| builder.alloc(value));
    let before_local_wires = before_local.map(|value| builder.alloc(value));
    let after_local_wires = after_local.map(|value| builder.alloc(value));
    let before_payload_wires = before_payload.map(|value| builder.alloc(value));
    let after_payload_wires = after_payload.map(|value| builder.alloc(value));
    enforce_streaming_lifecycle_semantic_link(
        &mut builder,
        StreamingLifecycleSemanticLinkWires {
            before_semantic_digest: before_semantic_wires,
            after_semantic_digest: after_semantic_wires,
            before_local_state_digest: before_local_wires,
            after_local_state_digest: after_local_wires,
            before_delayed_payload: &before_payload_wires,
            after_delayed_payload: &after_payload_wires,
        },
    );

    BuiltLink {
        builder,
        semantic_columns: before_semantic_wires
            .iter()
            .chain(&after_semantic_wires)
            .map(|wire| wire.col())
            .collect(),
        semantic_values: before_semantic.into_iter().chain(after_semantic).collect(),
        payload_columns: before_payload_wires
            .iter()
            .chain(&after_payload_wires)
            .map(|wire| wire.col())
            .collect(),
    }
}

fn export_link(builder: &R1csBuilder) -> Problem {
    let row_families = builder.row_family_ranges().to_vec();
    assert_eq!(
        row_families
            .iter()
            .map(|family| family.name)
            .collect::<Vec<_>>(),
        [
            STREAMING_LIFECYCLE_PAYLOAD_DOMAIN_FAMILY,
            STREAMING_LIFECYCLE_SEMANTIC_LINK_FAMILY,
        ],
    );
    let source = builder.snapshot();
    export_problem(
        &source,
        &row_families,
        ExportRequest {
            profile: "nightstream/goldilocks/streaming-lifecycle-semantic-link/v1".to_owned(),
            scope: Scope::Lifecycle,
            public_input_count: 1,
            source_rows: (0..source.rows()).collect(),
            complete_families: vec![
                STREAMING_LIFECYCLE_PAYLOAD_DOMAIN_FAMILY.to_owned(),
                STREAMING_LIFECYCLE_SEMANTIC_LINK_FAMILY.to_owned(),
            ],
        },
    )
    .expect("complete compact semantic-link export")
}

#[test]
#[ignore = "requires the installed GPL cvc5 finite-field solver"]
fn installed_cvc5_replays_the_semantic_link_counterexample() {
    let before_payload = std::array::from_fn(|index| F::from_bool(index % 2 == 0));
    let after_payload = std::array::from_fn(|index| F::from_bool(index % 3 == 0));
    let built = build_link(before_payload, after_payload);
    assert!(built.builder.is_satisfied());
    let problem = export_link(&built.builder);
    let target = TypedTarget {
        id: "nightstream.streaming.lifecycle.semantic_link".to_owned(),
        column_count: problem.column_count,
        rows: built
            .semantic_columns
            .iter()
            .copied()
            .zip(built.semantic_values.iter().copied())
            .enumerate()
            .map(|(index, (column, value))| {
                fixed_value_row(
                    format!("semantic_digest.field.{index}"),
                    problem.constant_one_column,
                    column,
                    value.as_canonical_u64(),
                )
            })
            .collect(),
    };

    let mut candidate_values = built
        .builder
        .witness()
        .iter()
        .map(|value| value.as_canonical_u64())
        .collect::<Vec<_>>();
    let changed_column = built.semantic_columns[0];
    candidate_values[changed_column] = (F::from_u64(candidate_values[changed_column]) + F::ONE).as_canonical_u64();
    let candidate = FieldModel::from_canonical_values(candidate_values).expect("canonical semantic-link attack");
    let violated_selected_rows = problem
        .rows
        .iter()
        .filter(|row| !row_is_satisfied(row, &candidate).expect("exact Rust row replay"))
        .map(|row| {
            assert_eq!(
                row.family, STREAMING_LIFECYCLE_SEMANTIC_LINK_FAMILY,
                "candidate must satisfy every retained family",
            );
            row.source_index
        })
        .collect::<Vec<_>>();
    assert!(!violated_selected_rows.is_empty());
    assert!(problem
        .rows
        .iter()
        .any(|row| row.family == STREAMING_LIFECYCLE_PAYLOAD_DOMAIN_FAMILY));

    let report = audit_complete_typed_candidate(
        &problem,
        &Selection::Family(STREAMING_LIFECYCLE_SEMANTIC_LINK_FAMILY.to_owned()),
        &target,
        &candidate,
        &SolverConfig {
            executable: PathBuf::from("/Users/nijaar/.local/bin/cvc5"),
            mode: SolverMode::Split,
            timeout_ms: 120_000,
        },
    )
    .expect("bounded strict semantic-link audit");
    eprintln!(
        "streaming lifecycle semantic-link audit: rows={} columns={} family={} cvc5={:?} replayed={} violated_target={:?} violated_selected={violated_selected_rows:?}",
        problem.rows.len(),
        problem.column_count,
        STREAMING_LIFECYCLE_SEMANTIC_LINK_FAMILY,
        report.solver_run.status,
        report.retained_rows_replayed.len(),
        report.violated_target_rows,
    );
    assert_eq!(report.solver_run.status, SolverStatus::Sat);
    assert_eq!(report.conclusion, Conclusion::CounterexampleCandidate);
    assert_eq!(
        report.retained_rows_replayed.len(),
        problem
            .rows
            .iter()
            .filter(|row| row.family != STREAMING_LIFECYCLE_SEMANTIC_LINK_FAMILY)
            .count(),
    );
    assert_eq!(report.violated_target_rows, [0]);
    assert_eq!(report.model.expect("full cvc5 model"), candidate);
}

#[test]
#[ignore = "requires the installed GPL cvc5 finite-field solver"]
fn installed_cvc5_replays_the_payload_domain_counterexample() {
    let mut before_payload = std::array::from_fn(|index| F::from_bool(index % 2 == 0));
    let after_payload = std::array::from_fn(|index| F::from_bool(index % 3 == 0));
    before_payload[0] = F::from_u64(2);
    let built = build_link(before_payload, after_payload);
    let problem = export_link(&built.builder);
    assert_eq!(built.payload_columns.len(), 2 * STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS);

    let target = TypedTarget {
        id: "nightstream.streaming.lifecycle.payload_domain".to_owned(),
        column_count: problem.column_count,
        rows: built
            .payload_columns
            .iter()
            .copied()
            .enumerate()
            .map(|(index, column)| bit_row(format!("payload.bit.{index}"), problem.constant_one_column, column))
            .collect(),
    };
    let candidate = FieldModel::from_canonical_values(
        built
            .builder
            .witness()
            .iter()
            .map(|value| value.as_canonical_u64())
            .collect(),
    )
    .expect("canonical payload-domain attack");
    let violated_selected_rows = problem
        .rows
        .iter()
        .filter(|row| !row_is_satisfied(row, &candidate).expect("exact Rust row replay"))
        .map(|row| {
            assert_eq!(
                row.family, STREAMING_LIFECYCLE_PAYLOAD_DOMAIN_FAMILY,
                "candidate must satisfy every retained family",
            );
            row.source_index
        })
        .collect::<Vec<_>>();
    assert_eq!(violated_selected_rows, [0]);

    let report = audit_complete_typed_candidate(
        &problem,
        &Selection::Family(STREAMING_LIFECYCLE_PAYLOAD_DOMAIN_FAMILY.to_owned()),
        &target,
        &candidate,
        &SolverConfig {
            executable: PathBuf::from("/Users/nijaar/.local/bin/cvc5"),
            mode: SolverMode::Split,
            timeout_ms: 120_000,
        },
    )
    .expect("bounded strict payload-domain audit");
    eprintln!(
        "streaming lifecycle payload-domain audit: rows={} columns={} family={} cvc5={:?} replayed={} violated_target={:?} violated_selected={violated_selected_rows:?}",
        problem.rows.len(),
        problem.column_count,
        STREAMING_LIFECYCLE_PAYLOAD_DOMAIN_FAMILY,
        report.solver_run.status,
        report.retained_rows_replayed.len(),
        report.violated_target_rows,
    );
    assert_eq!(report.solver_run.status, SolverStatus::Sat);
    assert_eq!(report.conclusion, Conclusion::CounterexampleCandidate);
    assert_eq!(
        report.retained_rows_replayed.len(),
        problem
            .rows
            .iter()
            .filter(|row| row.family != STREAMING_LIFECYCLE_PAYLOAD_DOMAIN_FAMILY)
            .count(),
    );
    assert_eq!(report.violated_target_rows, [0]);
    assert_eq!(report.model.expect("full cvc5 model"), candidate);
}
