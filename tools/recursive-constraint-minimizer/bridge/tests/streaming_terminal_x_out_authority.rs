//! Bounded cvc5 attacks for the exact terminal XOut authority families.

use std::path::PathBuf;

use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::{
    streaming_terminal_x_out_authority_audit, STREAMING_TERMINAL_R1CS_FAMILY_NAMES,
};
use neo_math::F;
use nightstream_constraint_exporter::{export_problem, ExportRequest};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use recursive_constraint_minimizer::{
    audit_complete_typed_candidate, row_is_satisfied, Conclusion, FieldModel, Scope, Selection, SolverConfig,
    SolverMode, SolverStatus, TypedTarget, TypedTargetRow,
};

#[test]
#[ignore = "requires the installed GPL cvc5 finite-field solver"]
fn installed_cvc5_replays_each_terminal_x_out_authority_counterexample() {
    let audit = streaming_terminal_x_out_authority_audit();
    let source = audit.source();
    let family_names = audit
        .row_families()
        .iter()
        .map(|family| family.name.to_owned())
        .collect::<Vec<_>>();
    assert_eq!(family_names, STREAMING_TERMINAL_R1CS_FAMILY_NAMES[2..5]);
    let mut complete_families = family_names;
    complete_families.sort_unstable();
    let problem = export_problem(
        source,
        audit.row_families(),
        ExportRequest {
            profile: "nightstream/goldilocks/streaming-terminal-x-out-authority/v1".to_owned(),
            scope: Scope::Lifecycle,
            public_input_count: 1,
            source_rows: (0..source.rows()).collect(),
            complete_families,
        },
    )
    .expect("complete exact terminal XOut authority export");
    let target = TypedTarget {
        id: "nightstream.streaming.terminal.x_out.authority".to_owned(),
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

    for (family, x_out_index) in [
        (STREAMING_TERMINAL_R1CS_FAMILY_NAMES[2], 1usize),
        (STREAMING_TERMINAL_R1CS_FAMILY_NAMES[3], 19usize),
        (STREAMING_TERMINAL_R1CS_FAMILY_NAMES[4], 28usize),
    ] {
        let mut values = source
            .witness()
            .iter()
            .map(|value| value.as_canonical_u64())
            .collect::<Vec<_>>();
        let changed_column = audit.x_out_columns()[x_out_index];
        values[changed_column] = (F::from_u64(values[changed_column]) + F::ONE).as_canonical_u64();
        let candidate = FieldModel::from_canonical_values(values).expect("canonical terminal attack assignment");
        let violated_selected_rows = problem
            .rows
            .iter()
            .filter(|row| !row_is_satisfied(row, &candidate).expect("exact Rust row replay"))
            .map(|row| {
                assert_eq!(row.family, family, "candidate must satisfy every retained family");
                row.source_index
            })
            .collect::<Vec<_>>();
        assert!(!violated_selected_rows.is_empty());

        let report = audit_complete_typed_candidate(
            &problem,
            &Selection::Family(family.to_owned()),
            &target,
            &candidate,
            &SolverConfig {
                executable: PathBuf::from("/Users/nijaar/.local/bin/cvc5"),
                mode: SolverMode::Split,
                timeout_ms: 30_000,
            },
        )
        .expect("bounded strict terminal authority audit");
        eprintln!(
            "streaming terminal XOut audit: family={family} rows={} columns={} cvc5={:?} replayed={} violated_target={:?} violated_selected={violated_selected_rows:?}",
            problem.rows.len(),
            problem.column_count,
            report.solver_run.status,
            report.retained_rows_replayed.len(),
            report.violated_target_rows,
        );
        assert_eq!(report.solver_run.status, SolverStatus::Sat);
        assert_eq!(report.conclusion, Conclusion::CounterexampleCandidate);
        assert_eq!(
            report.retained_rows_replayed.len(),
            problem.rows.iter().filter(|row| row.family != family).count()
        );
        assert!(!report.violated_target_rows.is_empty());
        assert_eq!(report.model.expect("full cvc5 model"), candidate);
    }
}
