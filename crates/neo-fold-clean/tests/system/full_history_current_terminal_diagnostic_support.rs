//! Bounded current-source capture for the fixed-one terminal shell.
//!
//! This exporter records current terminal owner frontiers and the exact
//! affine rows between `terminal.running_link` and `terminal.latest_link`.
//! It deliberately does not export the terminal NIFS, accumulator, or CE
//! matrices: their ranges and compiler schedules remain diagnostic data until
//! a current semantic decoder exists.

use std::fs;

use neo_fold_clean::engine::r1cs_circuit::builder::{RowFamilyRange, TerminalCeClaimAudit};
use serde_json::{json, Value};

use super::full_history_affine_artifact_support::render_artifact;
use super::full_history_manifest_identity_support::{range_json, source_hash};
use super::*;

const DIAGNOSTIC_PATH: &str = "formal/nightstream-lean/assurance/fprime-current-terminal-diagnostic.json";
const AFFINE_SHELL_PATH: &str = "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
FPrimeFullHistory/Generated/FPrimeFullHistoryCurrentTerminalAffineShell.lean";
const AFFINE_SHELL_NAMESPACE: &str = "FPrimeFullHistoryCurrentTerminalAffineShell";

fn unique_range<'a>(builder: &'a R1csBuilder, name: &str) -> &'a RowFamilyRange {
    let ranges = builder
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == name)
        .collect::<Vec<_>>();
    assert_eq!(ranges.len(), 1, "expected one current terminal range {name}");
    ranges[0]
}

fn sequence_runs(values: &[usize]) -> Vec<Value> {
    let mut runs = Vec::new();
    let mut start = 0;
    while start < values.len() {
        let step = values
            .get(start + 1)
            .and_then(|next| next.checked_sub(values[start]))
            .unwrap_or(0);
        let mut end = start + 1;
        while end < values.len() && values[end] == values[start].saturating_add((end - start).saturating_mul(step)) {
            end += 1;
        }
        runs.push(json!({
            "start": values[start],
            "step": step,
            "count": end - start,
        }));
        start = end;
    }
    runs
}

fn pair_runs(values: &[[usize; 2]]) -> Value {
    let c0 = values.iter().map(|pair| pair[0]).collect::<Vec<_>>();
    let c1 = values.iter().map(|pair| pair[1]).collect::<Vec<_>>();
    json!({
        "c0": sequence_runs(&c0),
        "c1": sequence_runs(&c1),
    })
}

fn terminal_ce_schedule(audit: &TerminalCeClaimAudit) -> Value {
    json!({
        "rows": {
            "start": audit.row_start,
            "end": audit.row_end,
            "count": audit.row_end - audit.row_start,
        },
        "first_allocated_column": audit.first_allocated_column,
        "norm_bound": audit.norm_bound,
        "expected_public_width": audit.expected_public_width,
        "structure": {
            "rows": audit.structure_rows,
            "columns": audit.structure_columns,
        },
        "witness": {
            "rows": audit.witness_rows,
            "columns": audit.witness_columns,
            "column_runs": sequence_runs(&audit.witness_cols),
        },
        "norm_first_allocated_column": audit.norm_first_allocated_column,
        "commitment": {
            "d": audit.commitment_d,
            "kappa": audit.commitment_kappa,
            "column_runs": sequence_runs(&audit.commitment_cols),
        },
        "public": {
            "rows": audit.public_rows,
            "width": audit.public_width,
            "input_len": audit.public_input_len,
            "column_runs": sequence_runs(&audit.public_cols),
        },
        "point": pair_runs(&audit.point_cols),
        "evaluations": audit
            .evaluation_cols
            .iter()
            .map(|columns| sequence_runs(columns))
            .collect::<Vec<_>>(),
        "constant_terms": pair_runs(&audit.constant_term_cols),
        "nc_point": pair_runs(&audit.nc_point_cols),
        "nc_evaluation_column_runs": sequence_runs(&audit.nc_evaluation_cols),
        "nc_evaluation_lanes": audit.nc_evaluation_lanes,
    })
}

fn current_terminal_diagnostic(
    builder: &R1csBuilder,
    affine_shell: &RowFamilyRange,
    relation_matrix_count: usize,
) -> Value {
    let terminal_ce_claims = builder.terminal_ce_claim_audits();
    let terminal_ce_evaluation_count = terminal_ce_claims
        .first()
        .map(|audit| audit.evaluation_cols.len())
        .expect("current terminal diagnostic requires direct CE claims");
    assert_eq!(
        terminal_ce_claims.len(),
        14,
        "current diagnostic fixture must expose fourteen terminal CE claims"
    );
    assert!(
        terminal_ce_claims
            .iter()
            .all(|audit| audit.evaluation_cols.len() == terminal_ce_evaluation_count),
        "every current diagnostic terminal CE claim must use one evaluation count"
    );
    assert_eq!(
        relation_matrix_count, 3,
        "the direct-R1CS diagnostic must retain its standard three-matrix CCS structure"
    );
    assert_eq!(
        terminal_ce_evaluation_count, relation_matrix_count,
        "terminal CE evaluations must match the diagnostic relation matrix count"
    );
    let selected = [
        "terminal.transcript",
        "terminal.nifs",
        "terminal.running_link",
        "terminal.parent_link",
        "terminal.latest_link",
        "terminal.accumulator",
        "terminal.total",
        "decider.terminal_fold",
        "decider.terminal_continuity",
        "decider.public_pins",
        "decider.terminal_ce",
    ];
    let root = formal_repo_root();
    let source_paths = [
        "crates/neo-fold-clean/src/engine/decider/terminal.rs",
        "crates/neo-fold-clean/src/engine/decider.rs",
        "crates/neo-fold-clean/src/engine/r1cs_circuit/decider_audit.rs",
        "crates/neo-fold-clean/src/paper/decider_ce_relation/mod.rs",
        "crates/neo-fold-clean/src/paper/f_prime/public_input_link.rs",
        "crates/neo-fold-clean/tests/system/decider_r1cs_manifest.rs",
        "crates/neo-fold-clean/tests/system/full_history_affine_artifact_support.rs",
        "crates/neo-fold-clean/tests/system/full_history_current_terminal_diagnostic_support.rs",
    ];
    json!({
        "artifact_kind": "r1cs/f-prime-current-terminal-bounded-diagnostic",
        "assurance_tier": "Rust diagnostic plus artifact-checked affine shell",
        "profile": {
            "relation": "direct-r1cs-standard-three-matrix-diagnostic",
            "fixed_one": true,
            "layout": "plain",
            "carrier_width": 270,
            "logical_width": 257,
            "completion_width": 13,
            "batch_schedule": [1, 1],
            "relation_matrix_count": relation_matrix_count,
            "terminal_ce_claim_count": terminal_ce_claims.len(),
            "terminal_ce_evaluation_count": terminal_ce_evaluation_count,
        },
        "builder": {
            "rows": builder.rows(),
            "columns": builder.cols(),
        },
        "terminal_ranges": selected
            .iter()
            .map(|name| range_json(builder, unique_range(builder, name)))
            .collect::<Vec<_>>(),
        "coefficient_complete_capture": {
            "owners": [
                "terminal.running_link",
                "terminal.parent_link",
                "terminal.latest_link",
            ],
            "range": range_json(builder, affine_shell),
            "lean_artifact": AFFINE_SHELL_PATH,
        },
        "terminal_ce_claim_schedules": terminal_ce_claims
            .iter()
            .map(terminal_ce_schedule)
            .collect::<Vec<_>>(),
        "activation": {
            "host_preconditions": [
                "final_fold is present",
                "at least one F-prime step is present",
                "terminal fresh batch is nonempty",
            ],
            "row_selector": null,
            "note": "The terminal owners are emitted unconditionally after host preflight; no selector gates these rows.",
        },
        "semantic_scope": {
            "decoded": [
                "running digest continuity",
                "parent-authority continuity",
                "prior/latest public-input link",
            ],
            "not_decoded": [
                "terminal NIFS",
                "terminal output accumulator",
                "terminal direct CE",
            ],
            "warning": "Range names, hashes, and column schedules are not semantic authority.",
        },
        "source_hashes": source_paths
            .iter()
            .map(|path| source_hash(&root, path))
            .collect::<Vec<_>>(),
    })
}

pub fn compare_current_terminal_diagnostic(builder: &R1csBuilder, relation_matrix_count: usize) {
    let running = unique_range(builder, "terminal.running_link");
    let parent = unique_range(builder, "terminal.parent_link");
    let latest = unique_range(builder, "terminal.latest_link");
    assert_eq!(running.row_end, parent.row_start, "running/parent shell gap");
    assert_eq!(parent.row_end, latest.row_start, "parent/latest shell gap");
    assert_eq!(running.row_end - running.row_start, 4, "running digest width");
    assert_eq!(latest.row_end - latest.row_start, 270, "plain carrier width");

    let affine_shell = RowFamilyRange {
        name: "terminal.current_affine_shell",
        row_start: running.row_start,
        row_end: latest.row_end,
    };
    let (artifact, run_modules) = render_artifact(builder, &affine_shell, AFFINE_SHELL_NAMESPACE);
    assert!(
        run_modules.is_empty(),
        "bounded terminal affine shell must fit in one compact certificate"
    );
    super::compare_full_history_artifact(&formal_repo_root().join(AFFINE_SHELL_PATH), &artifact, "lean.expected");

    let rendered = format!(
        "{}\n",
        serde_json::to_string_pretty(&current_terminal_diagnostic(
            builder,
            &affine_shell,
            relation_matrix_count,
        ))
        .expect("render current terminal diagnostic")
    );
    let path = formal_repo_root().join(DIAGNOSTIC_PATH);
    let committed = fs::read_to_string(&path).unwrap_or_default();
    if committed != rendered {
        fs::write(path.with_extension("json.expected"), &rendered)
            .expect("write current terminal diagnostic candidate");
    }
    assert_eq!(committed, rendered, "current terminal diagnostic drift");
}
