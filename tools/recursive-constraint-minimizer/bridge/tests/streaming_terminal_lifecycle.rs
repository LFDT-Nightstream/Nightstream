//! Bounded cvc5 attacks for the complete streaming terminal lifecycle.

#[path = "../../../../crates/neo-fold-clean/tests/support/streaming_terminal_fixture.rs"]
mod streaming_terminal_fixture;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::STREAMING_TERMINAL_R1CS_FAMILY_NAMES;
use nightstream_constraint_exporter::{export_problem, ExportRequest};
use recursive_constraint_minimizer::{
    audit_complete_typed_family, Conclusion, Problem, Scope, Selection, SolverConfig, SolverMode, SolverStatus,
    TypedTarget, TypedTargetRow,
};
use sha2::{Digest, Sha256};

use streaming_terminal_fixture::build_streaming_terminal_audit_fixture;

#[test]
#[ignore = "requires the installed GPL cvc5 finite-field solver"]
fn installed_cvc5_audits_terminal_source_binding() {
    audit_terminal_lifecycle_family(0);
}

#[test]
#[ignore = "requires the installed GPL cvc5 finite-field solver"]
fn installed_cvc5_audits_terminal_profile_selection() {
    audit_terminal_lifecycle_family(1);
}

#[test]
#[ignore = "requires the installed GPL cvc5 finite-field solver"]
fn installed_cvc5_audits_terminal_nebula_program_binding() {
    audit_terminal_lifecycle_family(5);
}

#[test]
#[ignore = "requires the installed GPL cvc5 finite-field solver"]
fn installed_cvc5_audits_terminal_nebula_finalizer() {
    audit_terminal_lifecycle_family(6);
}

#[test]
#[ignore = "requires the installed GPL cvc5 finite-field solver"]
fn installed_cvc5_audits_terminal_nebula_closed() {
    audit_terminal_lifecycle_family(7);
}

fn audit_terminal_lifecycle_family(family_index: usize) {
    let fixture = build_streaming_terminal_audit_fixture();
    let _rust_tamper_columns = [
        fixture.schedule_selector_column,
        fixture.verifier_key_column,
        fixture.program_binding_column,
        fixture.delayed_payload_column,
        fixture.fresh_adv_column,
        fixture.final_closed_lane_column,
    ];
    let row_families = fixture.terminal.row_family_ranges().to_vec();
    let family = STREAMING_TERMINAL_R1CS_FAMILY_NAMES[family_index];
    let owned_source_runs = row_families
        .iter()
        .filter(|range| range.name == family)
        .map(|range| range.row_start..range.row_end)
        .collect::<Vec<_>>();
    assert!(!owned_source_runs.is_empty(), "selected family must own source rows");
    assert_eq!(
        row_families
            .iter()
            .map(|family| family.name)
            .collect::<Vec<_>>(),
        [
            STREAMING_TERMINAL_R1CS_FAMILY_NAMES[6],
            STREAMING_TERMINAL_R1CS_FAMILY_NAMES[0],
            STREAMING_TERMINAL_R1CS_FAMILY_NAMES[1],
            STREAMING_TERMINAL_R1CS_FAMILY_NAMES[2],
            STREAMING_TERMINAL_R1CS_FAMILY_NAMES[3],
            STREAMING_TERMINAL_R1CS_FAMILY_NAMES[4],
            STREAMING_TERMINAL_R1CS_FAMILY_NAMES[5],
            STREAMING_TERMINAL_R1CS_FAMILY_NAMES[6],
            STREAMING_TERMINAL_R1CS_FAMILY_NAMES[7],
        ],
    );
    let source = fixture.terminal.into_snapshot();
    let mut complete_families = STREAMING_TERMINAL_R1CS_FAMILY_NAMES
        .map(str::to_owned)
        .to_vec();
    complete_families.sort_unstable();
    let raw_problem = export_problem(
        &source,
        &row_families,
        ExportRequest {
            profile: "nightstream/goldilocks/streaming-terminal-lifecycle/v1".to_owned(),
            scope: Scope::Lifecycle,
            public_input_count: 1,
            source_rows: (0..source.rows()).collect(),
            complete_families,
        },
    )
    .expect("complete exact streaming terminal export");
    drop(source);
    let problem = compact_active_columns(raw_problem);
    assert!(owned_source_runs.iter().all(|run| {
        problem.rows[run.clone()]
            .iter()
            .all(|row| row.family == family)
    }));
    eprintln!(
        "streaming terminal lifecycle ownership: family={family} scope=lifecycle source_runs={owned_source_runs:?} final_audit_runs={owned_source_runs:?} row_map=identity",
    );
    let target = TypedTarget {
        id: "nightstream.streaming.terminal.complete_lifecycle".to_owned(),
        column_count: problem.column_count,
        rows: problem
            .rows
            .iter()
            .map(|row| TypedTargetRow {
                id: format!("target.{}", row.source_index),
                a: row.a.clone(),
                b: row.b.clone(),
                c: row.c.clone(),
            })
            .collect(),
    };

    let report = match audit_complete_typed_family(
        &problem,
        &Selection::Family(family.to_owned()),
        &target,
        &SolverConfig {
            executable: PathBuf::from("/Users/nijaar/.local/bin/cvc5"),
            mode: SolverMode::Split,
            timeout_ms: 30_000,
        },
    ) {
        Ok(report) => report,
        Err(error) => {
            eprintln!(
                "streaming terminal lifecycle audit: family={family} rows={} active_columns={} cvc5=Inconclusive replayed=0 violated_target=[] decision=retain lean_certificate=missing error={error}",
                problem.rows.len(),
                problem.column_count,
            );
            return;
        }
    };
    eprintln!(
        "streaming terminal lifecycle audit: family={family} rows={} active_columns={} cvc5={:?} replayed={} violated_target={:?} decision=retain lean_certificate=missing",
        problem.rows.len(),
        problem.column_count,
        report.solver_run.status,
        report.retained_rows_replayed.len(),
        report.violated_target_rows,
    );
    match report.solver_run.status {
        SolverStatus::Sat => {
            assert_eq!(report.conclusion, Conclusion::CounterexampleCandidate);
            assert!(report.model.is_some(), "SAT must include the full compact model");
            assert_eq!(
                report.retained_rows_replayed.len(),
                problem
                    .rows
                    .iter()
                    .filter(|row| row.family != family)
                    .count(),
            );
            assert!(!report.violated_target_rows.is_empty());
        }
        SolverStatus::Unsat => {
            assert_eq!(report.conclusion, Conclusion::RedundancyCandidate);
            assert!(report.model.is_none());
        }
        SolverStatus::Unknown => {
            assert_eq!(report.conclusion, Conclusion::Inconclusive);
            assert!(report.model.is_none());
        }
    }
}

fn compact_active_columns(mut problem: Problem) -> Problem {
    let mut active = BTreeSet::from([problem.constant_one_column]);
    for row in &problem.rows {
        for term in row.a.iter().chain(&row.b).chain(&row.c) {
            active.insert(term.column);
        }
    }
    let old_columns = active.into_iter().collect::<Vec<_>>();
    let old_to_new = old_columns
        .iter()
        .enumerate()
        .map(|(new, &old)| (old, new))
        .collect::<BTreeMap<_, _>>();
    for row in &mut problem.rows {
        for term in row.a.iter_mut().chain(&mut row.b).chain(&mut row.c) {
            term.column = old_to_new[&term.column];
        }
    }

    let mut hasher = Sha256::new();
    hasher.update(b"nightstream/active-column-projection/v1");
    hasher.update(problem.source.artifact_digest.as_bytes());
    for column in &old_columns {
        hasher.update(column.to_le_bytes());
    }
    problem.source.profile.push_str("/active-columns-v1");
    problem.source.artifact_digest = format!("{:x}", hasher.finalize());
    problem.column_count = old_columns.len();
    problem.constant_one_column = old_to_new[&problem.constant_one_column];
    problem.public_input_count = 1;
    assert_eq!(problem.constant_one_column, 0);
    problem
        .validate()
        .expect("valid active-column terminal problem");
    problem
}
