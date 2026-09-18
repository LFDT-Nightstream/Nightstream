use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, Var};
use neo_fold_clean::paper::f_prime::r1cs::{
    enforce_f_prime_counter_input_binding, enforce_f_prime_recursive_counter_transition, FPrimeCounterInputWires,
    FPrimeCounterTransitionWires,
};
use neo_fold_clean::paper::f_prime::source_image::FPrimeSourceImage;
use neo_fold_clean::paper::f_prime::source_image_circuit::SourceImageWires;

use super::*;

const ARTIFACT_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryCounterArtifact.lean";
const ROWS_IN_CHUNK: u64 = 1;
const SOURCE_BIT_ROWS: usize = 128;

#[derive(Clone)]
struct CounterLayout {
    chunk_in_var: usize,
    step_in_var: usize,
    chunk_input_canonical_map: Vec<usize>,
    step_input_canonical_map: Vec<usize>,
    chunk_output_canonical_map: Vec<usize>,
    step_output_canonical_map: Vec<usize>,
    increment_map: Vec<usize>,
    add_map: Vec<usize>,
    rows_in_chunk_bits: Vec<usize>,
}

fn canonical_map(field: usize, bits: &[usize; 64]) -> Vec<usize> {
    std::iter::once(0)
        .chain(std::iter::once(field))
        .chain(bits.iter().copied())
        .chain([bits[63] + 1, bits[63] + 2])
        .collect()
}

fn layout_from_wires(
    chunk_in: Var,
    step_in: Var,
    input_aux_start: usize,
    input: &FPrimeCounterInputWires,
    transition: &FPrimeCounterTransitionWires,
) -> CounterLayout {
    let chunk_input_bits = input.chunk_count_bits.map(Var::col);
    let step_input_bits = input.step_count_bits.map(Var::col);
    let chunk_output_bits = transition.chunk_count_out_bits.map(Var::col);
    let step_output_bits = transition.step_count_out_bits.map(Var::col);
    let increment_carry_start = chunk_output_bits[63] + 3;
    let add_carry_start = step_output_bits[63] + 3;
    CounterLayout {
        chunk_in_var: chunk_in.col(),
        step_in_var: step_in.col(),
        chunk_input_canonical_map: std::iter::once(0)
            .chain(std::iter::once(chunk_in.col()))
            .chain(chunk_input_bits)
            .chain([input_aux_start, input_aux_start + 1])
            .collect(),
        step_input_canonical_map: std::iter::once(0)
            .chain(std::iter::once(step_in.col()))
            .chain(step_input_bits)
            .chain([input_aux_start + 2, input_aux_start + 3])
            .collect(),
        chunk_output_canonical_map: canonical_map(transition.chunk_count_out.col(), &chunk_output_bits),
        step_output_canonical_map: canonical_map(transition.step_count_out.col(), &step_output_bits),
        increment_map: std::iter::once(0)
            .chain(chunk_input_bits)
            .chain(chunk_output_bits)
            .chain(increment_carry_start..increment_carry_start + 63)
            .collect(),
        add_map: std::iter::once(0)
            .chain(step_input_bits)
            .chain(transition.rows_in_chunk_bits.map(Var::col))
            .chain(step_output_bits)
            .chain(add_carry_start..add_carry_start + 63)
            .collect(),
        rows_in_chunk_bits: transition
            .rows_in_chunk_bits
            .iter()
            .map(|bit| bit.col())
            .collect(),
    }
}

fn isolated_counter() -> (R1csBuilder, CounterLayout, usize) {
    let mut image = FPrimeSourceImage::new();
    let chunk_word = image.push_u64_le(1);
    let step_word = image.push_u64_le(1);
    let mut builder = R1csBuilder::new();
    let chunk_in = builder.alloc(F::ONE);
    let step_in = builder.alloc(F::ONE);
    let source_wires = SourceImageWires::alloc(&mut builder, &image);
    let input_aux_start = builder.cols();
    let input =
        enforce_f_prime_counter_input_binding(&mut builder, &source_wires, chunk_word, step_word, chunk_in, step_in);
    let input_rows = builder.rows();
    let transition =
        enforce_f_prime_recursive_counter_transition(&mut builder, chunk_in, step_in, &input, ROWS_IN_CHUNK, 2, 2);
    let layout = layout_from_wires(chunk_in, step_in, input_aux_start, &input, &transition);
    (builder, layout, input_rows)
}

fn rows_of(builder: &R1csBuilder, start: usize, end: usize) -> Vec<checked_program_artifact_support::Row> {
    let (a, b, c) = builder.sparse_triplets();
    (start..end)
        .map(|row| checked_program_artifact_support::Row {
            a: a.iter()
                .filter(|entry| entry.0 == row)
                .map(|entry| (entry.1, entry.2.as_canonical_u64()))
                .collect(),
            b: b.iter()
                .filter(|entry| entry.0 == row)
                .map(|entry| (entry.1, entry.2.as_canonical_u64()))
                .collect(),
            c: c.iter()
                .filter(|entry| entry.0 == row)
                .map(|entry| (entry.1, entry.2.as_canonical_u64()))
                .collect(),
        })
        .collect()
}

fn canonicalize_rows(rows: &[checked_program_artifact_support::Row]) -> CanonicalizedProgram {
    canonicalize_program(&NormalizedProgram {
        instructions: rows.iter().cloned().map(Instruction::Check).collect(),
        input_columns: Vec::new(),
        definition_count: 0,
        check_count: rows.len(),
    })
}

fn relabel_rows(
    rows: &[checked_program_artifact_support::Row],
    map: &[usize],
) -> Vec<checked_program_artifact_support::Row> {
    let terms = |source: &[(usize, u64)]| {
        source
            .iter()
            .map(|&(column, coefficient)| (map[column], coefficient))
            .collect()
    };
    rows.iter()
        .map(|row| checked_program_artifact_support::Row {
            a: terms(&row.a),
            b: terms(&row.b),
            c: terms(&row.c),
        })
        .collect()
}

pub fn render_counter_artifact(
    builder: &R1csBuilder,
    audit: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
) -> String {
    let (isolated, local, input_rows) = isolated_counter();
    let local_rows = rows_of(&isolated, 0, isolated.rows());
    let prelude = builder
        .row_family_ranges()
        .iter()
        .find(|range| {
            range.name == "fprime.recursive.prelude"
                && audit.row_start <= range.row_start
                && range.row_end <= audit.row_end
        })
        .expect("recursive prelude range");
    let counter = builder
        .row_family_ranges()
        .iter()
        .find(|range| {
            range.name == "fprime.recursive.counter"
                && audit.row_start <= range.row_start
                && range.row_end <= audit.row_end
        })
        .expect("recursive counter range");
    let prelude_rows = rows_of(builder, prelude.row_start, prelude.row_end);
    let binding_rows = input_rows - SOURCE_BIT_ROWS;
    let local_input_shape = canonicalize_rows(&local_rows[SOURCE_BIT_ROWS..input_rows]);
    let chunk_index = local_input_shape
        .column_map
        .iter()
        .position(|column| *column == local.chunk_in_var)
        .expect("local chunk counter canonical index");
    let step_index = local_input_shape
        .column_map
        .iter()
        .position(|column| *column == local.step_in_var)
        .expect("local step counter canonical index");
    let candidates = prelude_rows
        .windows(binding_rows)
        .enumerate()
        .filter_map(|(start, window)| {
            let candidate = canonicalize_rows(window);
            (candidate.instructions == local_input_shape.instructions
                && candidate.column_map[chunk_index] == audit.state_in_columns[8]
                && candidate.column_map[step_index] == audit.state_in_columns[9])
                .then_some(start)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        candidates.len(),
        1,
        "one exact counter input-binding block in recursive prelude"
    );
    let mut global_rows = prelude_rows[candidates[0]..candidates[0] + binding_rows].to_vec();
    global_rows.extend(rows_of(builder, counter.row_start, counter.row_end));
    let local_canonical = canonicalize_rows(&local_rows[SOURCE_BIT_ROWS..]);
    let global_canonical = canonicalize_rows(&global_rows);
    assert_eq!(
        global_canonical.instructions, local_canonical.instructions,
        "counter input and transition rows share one exact canonical shape"
    );
    assert_eq!(local_canonical.column_map.len(), isolated.cols());
    assert_eq!(global_canonical.column_map.len(), isolated.cols());
    let mut column_map = vec![usize::MAX; isolated.cols()];
    for (&local_column, &global_column) in local_canonical
        .column_map
        .iter()
        .zip(&global_canonical.column_map)
    {
        column_map[local_column] = global_column;
    }
    assert!(column_map.iter().all(|column| *column != usize::MAX));
    for bit_row in relabel_rows(&local_rows[..SOURCE_BIT_ROWS], &column_map) {
        assert!(
            prelude_rows.contains(&bit_row),
            "mapped counter source-bit row occurs in recursive prelude"
        );
    }
    format!(
        "import Nightstream.Implementation.R1CS.Core.Semantics\n\n\
         /-! Generated exact counter program embedded across recursive prelude and transition owners. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryCounter\n\n\
         def rowsInChunk : Nat := {ROWS_IN_CHUNK}\n\
         def chunkInputVarCol : Nat := {}\n\
         def stepInputVarCol : Nat := {}\n\
         def chunkInputCanonicalMap : List Nat := {}\n\
         def stepInputCanonicalMap : List Nat := {}\n\
         def chunkOutputCanonicalMap : List Nat := {}\n\
         def stepOutputCanonicalMap : List Nat := {}\n\
         def incrementMap : List Nat := {}\n\
         def addMap : List Nat := {}\n\
         def rowsInChunkBitCols : List Nat := {}\n\
         def globalColumnMap : List Nat := {}\n\
         def inputRowCount : Nat := {input_rows}\n\
         def transitionRowStart : Nat := {}\n\
         def transitionRowEnd : Nat := {}\n\
         def rowCount : Nat := {}\n\n\
         def rows : List Row :=\n  {}\n\n\
         theorem rows_length : rows.length = rowCount := by native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryCounter\n",
        local.chunk_in_var,
        local.step_in_var,
        lean_nat_list(local.chunk_input_canonical_map),
        lean_nat_list(local.step_input_canonical_map),
        lean_nat_list(local.chunk_output_canonical_map),
        lean_nat_list(local.step_output_canonical_map),
        lean_nat_list(local.increment_map),
        lean_nat_list(local.add_map),
        lean_nat_list(local.rows_in_chunk_bits),
        lean_nat_list(column_map),
        counter.row_start,
        counter.row_end,
        isolated.rows(),
        lean_artifact_support::lean_rows(&isolated),
    )
}

pub fn compare_counter_artifact(builder: &R1csBuilder, audit: &neo_fold_clean::engine::decider::FPrimeStepWireAudit) {
    let path = formal_repo_root().join(ARTIFACT_PATH);
    let rendered = render_counter_artifact(builder, audit);
    compare_full_history_artifact(&path, &rendered, "lean.expected");
}
