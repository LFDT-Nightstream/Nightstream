use std::collections::HashSet;
use std::fmt::Write as _;

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Pin {
    Zero(usize),
    Constant(usize, u64),
    Equal(usize, usize),
}

#[derive(Clone, Copy)]
pub(super) enum PinRun {
    Zero {
        start: usize,
        step: usize,
        count: usize,
    },
    Constant {
        column_start: usize,
        column_step: usize,
        value_start: u64,
        value_step: u64,
        count: usize,
    },
    Equal {
        left_start: usize,
        right_start: usize,
        left_step: usize,
        right_step: usize,
        count: usize,
    },
}

fn rows_in_range(
    builder: &R1csBuilder,
    range: &RowFamilyRange,
) -> (Vec<Vec<(usize, u64)>>, Vec<Vec<(usize, u64)>>, Vec<Vec<(usize, u64)>>) {
    let count = range.row_end - range.row_start;
    let mut a_rows = vec![Vec::new(); count];
    let mut b_rows = vec![Vec::new(); count];
    let mut c_rows = vec![Vec::new(); count];
    let (a, b, c) = builder.sparse_triplets();
    for (source, rows) in [(a, &mut a_rows), (b, &mut b_rows), (c, &mut c_rows)] {
        for &(row, column, coefficient) in source {
            if range.row_start <= row && row < range.row_end {
                rows[row - range.row_start].push((column, coefficient.as_canonical_u64()));
            }
        }
    }
    (a_rows, b_rows, c_rows)
}

pub(super) fn affine_pins(builder: &R1csBuilder, range: &RowFamilyRange) -> Vec<Pin> {
    let (a_rows, b_rows, c_rows) = rows_in_range(builder, range);
    let minus_one = F::ORDER_U64 - 1;
    a_rows
        .into_iter()
        .zip(b_rows)
        .zip(c_rows)
        .enumerate()
        .map(|(offset, ((a, b), c))| {
            assert_eq!(b, vec![(0, 1)], "affine B row {}", range.row_start + offset);
            assert!(c.is_empty(), "affine C row {}", range.row_start + offset);
            match a.as_slice() {
                [(column, 1)] => Pin::Zero(*column),
                [first, second] => {
                    let (output, other) = if first.1 == 1 {
                        (first.0, *second)
                    } else if second.1 == 1 {
                        (second.0, *first)
                    } else {
                        panic!("affine row {} has no unit output: {a:?}", range.row_start + offset);
                    };
                    if other.0 == 0 {
                        let value = if other.1 == 0 { 0 } else { F::ORDER_U64 - other.1 };
                        if value == 0 {
                            Pin::Zero(output)
                        } else {
                            Pin::Constant(output, value)
                        }
                    } else {
                        assert_eq!(
                            other.1,
                            minus_one,
                            "affine equality coefficient at row {}",
                            range.row_start + offset
                        );
                        Pin::Equal(output, other.0)
                    }
                }
                _ => panic!("row {} is not an affine pin: {a:?}", range.row_start + offset),
            }
        })
        .collect()
}

fn step(left: usize, right: usize) -> Option<usize> {
    right.checked_sub(left)
}

fn value_step(left: u64, right: u64) -> Option<u64> {
    right.checked_sub(left)
}

pub(super) fn pin_runs(pins: &[Pin]) -> Vec<PinRun> {
    let mut runs = Vec::new();
    let mut start = 0;
    while start < pins.len() {
        match pins[start] {
            Pin::Zero(column_start) => {
                let column_step = match pins.get(start + 1) {
                    Some(Pin::Zero(next)) => step(column_start, *next).unwrap_or(0),
                    _ => 0,
                };
                let mut end = start + 1;
                while matches!(pins.get(end), Some(Pin::Zero(column)) if *column == column_start + (end - start) * column_step)
                {
                    end += 1;
                }
                runs.push(PinRun::Zero {
                    start: column_start,
                    step: column_step,
                    count: end - start,
                });
                start = end;
            }
            Pin::Constant(column_start, value_start) => {
                let (column_step, value_step) = match pins.get(start + 1) {
                    Some(Pin::Constant(next_column, next_value)) => (
                        step(column_start, *next_column).unwrap_or(0),
                        value_step(value_start, *next_value).unwrap_or(0),
                    ),
                    _ => (0, 0),
                };
                let mut end = start + 1;
                while matches!(pins.get(end), Some(Pin::Constant(column, value))
                    if *column == column_start + (end - start) * column_step
                        && *value == value_start + (end - start) as u64 * value_step)
                {
                    end += 1;
                }
                runs.push(PinRun::Constant {
                    column_start,
                    column_step,
                    value_start,
                    value_step,
                    count: end - start,
                });
                start = end;
            }
            Pin::Equal(left_start, right_start) => {
                let (left_step, right_step) = match pins.get(start + 1) {
                    Some(Pin::Equal(next_left, next_right)) => (
                        step(left_start, *next_left).unwrap_or(0),
                        step(right_start, *next_right).unwrap_or(0),
                    ),
                    _ => (0, 0),
                };
                let mut end = start + 1;
                while matches!(pins.get(end), Some(Pin::Equal(left, right))
                    if *left == left_start + (end - start) * left_step
                        && *right == right_start + (end - start) * right_step)
                {
                    end += 1;
                }
                runs.push(PinRun::Equal {
                    left_start,
                    right_start,
                    left_step,
                    right_step,
                    count: end - start,
                });
                start = end;
            }
        }
    }
    runs
}

pub(super) fn render_runs(runs: &[PinRun]) -> String {
    let mut rendered = String::new();
    for (index, run) in runs.iter().enumerate() {
        let prefix = if index == 0 { "  [" } else { "  ," };
        match run {
            PinRun::Zero { start, step, count } => {
                writeln!(rendered, "{prefix} .zero {start} {step} {count}").unwrap();
            }
            PinRun::Constant {
                column_start,
                column_step,
                value_start,
                value_step,
                count,
            } => {
                writeln!(
                    rendered,
                    "{prefix} .constant {column_start} {column_step} {value_start} {value_step} {count}"
                )
                .unwrap();
            }
            PinRun::Equal {
                left_start,
                right_start,
                left_step,
                right_step,
                count,
            } => {
                writeln!(
                    rendered,
                    "{prefix} .equal {left_start} {right_start} {left_step} {right_step} {count}"
                )
                .unwrap();
            }
        }
    }
    rendered.push_str("  ]");
    rendered
}

const RUNS_PER_MODULE: usize = 1_000;

fn render_run_module(namespace: &str, part: usize, runs: &[PinRun]) -> String {
    let rendered_runs = render_runs(runs);
    format!(
        "import Nightstream.Implementation.R1CS.Core.AffinePins\n\n\
         /-! Generated exact affine-pin run data. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.{namespace}Runs{part}\n\n\
         open Nightstream.Implementation.R1CS\n\n\
         def pinRuns : List AffinePins.Run :=\n{rendered_runs}\n\n\
         end Nightstream.Implementation.R1CS.{namespace}Runs{part}\n"
    )
}

pub(super) fn render_artifact(builder: &R1csBuilder, range: &RowFamilyRange, namespace: &str) -> (String, Vec<String>) {
    let pins = affine_pins(builder, range);
    let runs = pin_runs(&pins);
    let run_modules = if runs.len() > RUNS_PER_MODULE {
        runs.chunks(RUNS_PER_MODULE)
            .enumerate()
            .map(|(part, chunk)| render_run_module(namespace, part, chunk))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let imports = (0..run_modules.len())
        .map(|part| {
            format!(
                "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.{namespace}Runs{part}\n"
            )
        })
        .collect::<String>();
    let pin_runs = if run_modules.is_empty() {
        format!("def pinRuns : List AffinePins.Run :=\n{}", render_runs(&runs))
    } else {
        let parts = (0..run_modules.len())
            .map(|part| format!("{namespace}Runs{part}.pinRuns"))
            .collect::<Vec<_>>()
            .join(" ++\n    ");
        format!("def pinRuns : List AffinePins.Run :=\n    {parts}")
    };
    let hash = full_history_range_hash(builder, range);
    let artifact = format!(
        "import Nightstream.Implementation.R1CS.Core.AffinePins\n\
         {imports}\n\
         /-! Generated exact affine-pin phase. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.{namespace}\n\n\
         open Nightstream.Implementation.R1CS\n\n\
         def rangeSha256 : String := \"{hash}\"\n\
         def rowStart : Nat := {}\n\
         def rowEnd : Nat := {}\n\
         def rowCount : Nat := {}\n\n\
         {pin_runs}\n\n\
         def pins : List AffinePins.Pin := AffinePins.expandRuns pinRuns\n\
         def rows : List Row := AffinePins.rows pins\n\n\
         theorem pins_canonical : AffinePins.PinsCanonical pins := by native_decide\n\
         theorem pins_length : pins.length = rowCount := by\n\
           rw [pins, AffinePins.expandRuns_length]\n\
           native_decide\n\n\
         theorem rows_length : rows.length = rowCount := by\n\
           simpa [rows, AffinePins.rows] using pins_length\n\n\
         end Nightstream.Implementation.R1CS.{namespace}\n",
        range.row_start,
        range.row_end,
        pins.len(),
    );
    (artifact, run_modules)
}

fn nth_range<'a>(builder: &'a R1csBuilder, name: &str, occurrence: usize) -> &'a RowFamilyRange {
    builder
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == name)
        .nth(occurrence)
        .unwrap_or_else(|| panic!("missing occurrence {occurrence} of {name}"))
}

pub fn compare_affine_artifacts(builder: &R1csBuilder) {
    let specs = [
        ("nifs.pi_ccs.allocation", 0, "FPrimeFullHistoryPiCcsRecursiveAllocation"),
        ("nifs.pi_ccs.authority", 0, "FPrimeFullHistoryPiCcsRecursiveAuthority"),
        (
            "nifs.pi_ccs.output_binding",
            0,
            "FPrimeFullHistoryPiCcsRecursiveOutputBinding",
        ),
        ("nifs.pi_rlc.shape", 0, "FPrimeFullHistoryPiRlcRecursiveShape"),
        (
            "nifs.pi_rlc.linear_folds",
            0,
            "FPrimeFullHistoryPiRlcRecursiveLinearFolds",
        ),
        ("nifs.pi_ccs.allocation", 1, "FPrimeFullHistoryPiCcsTerminalAllocation"),
        (
            "nifs.pi_ccs.output_binding",
            1,
            "FPrimeFullHistoryPiCcsTerminalOutputBinding",
        ),
        ("nifs.pi_rlc.shape", 1, "FPrimeFullHistoryPiRlcTerminalShape"),
        (
            "nifs.pi_rlc.linear_folds",
            1,
            "FPrimeFullHistoryPiRlcTerminalLinearFolds",
        ),
    ];
    let mut paths = HashSet::new();
    for (name, occurrence, namespace) in specs {
        let range = nth_range(builder, name, occurrence);
        let path = formal_repo_root().join(format!(
            "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/{namespace}.lean"
        ));
        assert!(paths.insert(path.clone()), "duplicate affine artifact path");
        let (artifact, run_modules) = render_artifact(builder, range, namespace);
        compare_full_history_artifact(&path, &artifact, "lean.expected");
        for (part, run_module) in run_modules.into_iter().enumerate() {
            let part_path = formal_repo_root().join(format!(
                "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/{namespace}Runs{part}.lean"
            ));
            assert!(paths.insert(part_path.clone()), "duplicate affine artifact path");
            compare_full_history_artifact(&part_path, &run_module, "lean.expected");
        }
    }

    // The terminal Pi_CCS authority phase starts with one exact strict-Pi_DEC
    // replay (exported separately as `terminalCeRows`) and ends with the
    // affine output-ct, y_ring-padding, and authoritative y_zcol-padding
    // checks. Export that suffix independently so
    // the full parent schedule can reuse the checked Pi_DEC compiler without
    // duplicating its 10,597 rows as literals.
    let authority = nth_range(builder, "nifs.pi_ccs.authority", 1);
    let tail = RowFamilyRange {
        name: "nifs.pi_ccs.authority.output_checks",
        row_start: authority.row_start + 10_597,
        row_end: authority.row_end,
    };
    assert_eq!(tail.row_end - tail.row_start, 1_290, "terminal authority affine tail");
    let namespace = "FPrimeFullHistoryPiCcsTerminalAuthorityTail";
    let path = formal_repo_root().join(format!(
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/{namespace}.lean"
    ));
    assert!(paths.insert(path.clone()), "duplicate affine artifact path");
    let (artifact, run_modules) = render_artifact(builder, &tail, namespace);
    compare_full_history_artifact(&path, &artifact, "lean.expected");
    for (part, run_module) in run_modules.into_iter().enumerate() {
        let part_path = formal_repo_root().join(format!(
            "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/{namespace}Runs{part}.lean"
        ));
        assert!(paths.insert(part_path.clone()), "duplicate affine artifact path");
        compare_full_history_artifact(&part_path, &run_module, "lean.expected");
    }
}

/// Export only the current terminal latest-link placement from a live
/// full-history synthesis. This certificate is intentionally independent of
/// the captured aggregate artifact, whose terminal-link range predates the
/// thirteen plain-carrier padding rows.
pub fn compare_current_terminal_link_artifact(builder: &R1csBuilder) {
    let range = nth_range(builder, "terminal.latest_link", 0);
    assert_eq!(
        range.row_end - range.row_start,
        neo_fold_clean::paper::f_prime::r1cs::F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN,
        "current terminal latest-link row count",
    );
    let namespace = "FPrimeFullHistoryCurrentTerminalLinkPlacement";
    let path = formal_repo_root().join(format!(
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/{namespace}.lean"
    ));
    let (artifact, run_modules) = render_artifact(builder, range, namespace);
    assert!(
        run_modules.is_empty(),
        "current terminal-link placement must fit in one bounded certificate",
    );
    compare_full_history_artifact(&path, &artifact, "lean.expected");
}
