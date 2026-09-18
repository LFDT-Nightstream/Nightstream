use std::fmt::Write as _;

use super::*;

#[derive(Clone, Copy)]
struct PairRun {
    left_start: usize,
    right_start: usize,
    left_step: usize,
    right_step: usize,
    count: usize,
}

fn equality_pairs(builder: &R1csBuilder, range: &RowFamilyRange) -> Vec<(usize, usize)> {
    let row_count = range.row_end - range.row_start;
    let mut a_rows = vec![Vec::new(); row_count];
    let mut b_rows = vec![Vec::new(); row_count];
    let mut c_rows = vec![Vec::new(); row_count];
    let (a, b, c) = builder.sparse_triplets();
    for (source, rows) in [(a, &mut a_rows), (b, &mut b_rows), (c, &mut c_rows)] {
        for &(row, column, coefficient) in source {
            if row >= range.row_start && row < range.row_end {
                rows[row - range.row_start].push((column, coefficient.as_canonical_u64()));
            }
        }
    }
    let minus_one = F::ORDER_U64 - 1;
    (0..row_count)
        .map(|row| {
            assert_eq!(b_rows[row], vec![(0, 1)], "equality row B at {row}");
            assert!(c_rows[row].is_empty(), "equality row C at {row}");
            let [(left, 1), (right, coefficient)] = a_rows[row].as_slice() else {
                panic!("row {row} is not a two-term equality: {:?}", a_rows[row]);
            };
            assert_eq!(*coefficient, minus_one, "equality row -1 at {row}");
            (*left, *right)
        })
        .collect()
}

fn pair_runs(pairs: &[(usize, usize)]) -> Vec<PairRun> {
    let mut runs = Vec::new();
    let mut start = 0;
    while start < pairs.len() {
        let (left_step, right_step) = if pairs[start].1 != 0
            && start + 1 < pairs.len()
            && pairs[start + 1].0 >= pairs[start].0
            && pairs[start + 1].1 >= pairs[start].1
        {
            (pairs[start + 1].0 - pairs[start].0, pairs[start + 1].1 - pairs[start].1)
        } else {
            (0, 0)
        };
        let mut end = start + 1;
        while end < pairs.len()
            && pairs[end]
                == (
                    pairs[start].0 + (end - start) * left_step,
                    pairs[start].1 + (end - start) * right_step,
                )
        {
            end += 1;
        }
        runs.push(PairRun {
            left_start: pairs[start].0,
            right_start: pairs[start].1,
            left_step,
            right_step,
            count: end - start,
        });
        start = end;
    }
    runs
}

pub fn render_equality_artifact(
    builder: &R1csBuilder,
    range: &RowFamilyRange,
    namespace: &str,
    title: &str,
    range_hash: &str,
) -> String {
    let pairs = equality_pairs(builder, range);
    let runs = pair_runs(&pairs);
    let mut rendered_runs = String::new();
    for (index, run) in runs.iter().enumerate() {
        let prefix = if index == 0 { "  [" } else { "  ," };
        writeln!(
            rendered_runs,
            "{prefix} ⟨{}, {}, {}, {}, {}⟩",
            run.left_start, run.right_start, run.left_step, run.right_step, run.count
        )
        .expect("render equality run");
    }
    rendered_runs.push_str("  ]");
    format!(
        "import Nightstream.Implementation.R1CS.Core.EqualityPins\n\n\
         /-! Generated {title}. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.{namespace}\n\n\
         open Nightstream.Implementation.R1CS\n\n\
         def rangeSha256 : String := \"{range_hash}\"\n\
         def rowStart : Nat := {}\n\
         def rowEnd : Nat := {}\n\
         def rowCount : Nat := {}\n\n\
         def pairRuns : List EqualityPins.PairRun :=\n{rendered_runs}\n\n\
         def pairs : List (Nat × Nat) := EqualityPins.expandRuns pairRuns\n\
         def rows : List Row := EqualityPins.rows pairs\n\n\
         theorem pairs_length : pairs.length = rowCount := by\n\
           rw [pairs, EqualityPins.expandRuns_length]\n\
           native_decide\n\n\
         theorem rows_length : rows.length = rowCount := by\n\
           simpa [rows, EqualityPins.rows] using pairs_length\n\n\
         end Nightstream.Implementation.R1CS.{namespace}\n",
        range.row_start,
        range.row_end,
        pairs.len(),
    )
}
