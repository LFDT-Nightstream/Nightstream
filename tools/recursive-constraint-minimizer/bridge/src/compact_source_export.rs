//! String-payload emission for complete source artifacts.
//!
//! Owns: the compact wire construction (shared value table, per-matrix CSR
//! payloads, compact seeded blocks, family row ranges), a complete chunked
//! replay that compares the wire expansion against the independent sparse
//! row recovery before anything is rendered, and the Lean module rendering
//! (one string payload part per data module plus one assembly module).
//!
//! Does not own: expansion semantics (Lean `CompactSourceArtifact.expand`),
//! sampler conformance (the mirror gate), the minimization theorems, or any
//! removal authority.

use std::collections::BTreeMap;

use neo_ccs::{CcsMatrix, CscMat, SeededPhi81LinearBlock};
use neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs;
use neo_math::F;
use p3_field::PrimeField64;

use recursive_constraint_minimizer::GOLDILOCKS_MODULUS;

use crate::lean_export::GeneratedLeanModule;
use crate::{recover_sparse_rows, sparse_family_census, ExportError, SparseProblemExporter};

const PART_CHAR_BUDGET: usize = 7_800_000;
const REPLAY_CHUNK_ROWS: usize = 131_072;

struct MatrixPayload {
    row_counts: Vec<u16>,
    columns: Vec<u32>,
    value_indexes: Vec<u16>,
    blocks: Vec<SeededPhi81LinearBlock>,
}

struct WirePayloads {
    value_table: Vec<u64>,
    matrices: [MatrixPayload; 3],
}

fn matrix_parts(matrix: &CcsMatrix<F>) -> Result<(&CscMat<F>, &[SeededPhi81LinearBlock]), ExportError> {
    match matrix {
        CcsMatrix::Csc(csc) => Ok((csc, &[])),
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            if !geometric_runs.is_empty() {
                return Err(ExportError::new(
                    "compact source emission does not support geometric runs yet",
                ));
            }
            Ok((csc, blocks))
        }
        _ => Err(ExportError::new(
            "compact source emission requires materialized CSC matrices",
        )),
    }
}

fn build_payloads(arm: &SparseR1cs) -> Result<WirePayloads, ExportError> {
    let mut value_index = BTreeMap::<u64, u16>::new();
    for matrix in [&arm.a, &arm.b, &arm.c] {
        let (csc, _) = matrix_parts(matrix)?;
        for value in &csc.vals {
            let canonical = value.as_canonical_u64();
            if canonical == 0 {
                return Err(ExportError::new("compact source payload holds a zero coefficient"));
            }
            let next = value_index.len();
            value_index
                .entry(canonical)
                .or_insert_with(|| u16::try_from(next).expect("value table exceeds u16 index range"));
        }
    }
    // BTreeMap iteration is sorted; re-number so indices follow sorted values.
    let value_table: Vec<u64> = value_index.keys().copied().collect();
    for (position, value) in value_table.iter().enumerate() {
        value_index.insert(*value, u16::try_from(position).expect("value index fits u16"));
    }

    let mut matrices = Vec::with_capacity(3);
    for matrix in [&arm.a, &arm.b, &arm.c] {
        let (csc, blocks) = matrix_parts(matrix)?;
        for block in blocks {
            if block.has_superneo_transformed_columns() {
                return Err(ExportError::new(
                    "compact source emission does not support transformed seeded columns",
                ));
            }
        }
        if csc.nrows != arm.n || csc.ncols != arm.m {
            return Err(ExportError::new("sparse matrix geometry differs from its arm"));
        }
        let mut row_counts = vec![0u32; arm.n];
        for column in 0..csc.ncols {
            for entry in csc.column_range(column) {
                row_counts[csc.row_index(entry)] += 1;
            }
        }
        let mut offsets = vec![0usize; arm.n + 1];
        for row in 0..arm.n {
            offsets[row + 1] = offsets[row] + row_counts[row] as usize;
        }
        let nnz = offsets[arm.n];
        let mut columns = vec![0u32; nnz];
        let mut value_indexes = vec![0u16; nnz];
        let mut cursor = offsets.clone();
        for column in 0..csc.ncols {
            let column_u32 = u32::try_from(column).map_err(|_| ExportError::new("column exceeds u32"))?;
            for entry in csc.column_range(column) {
                let row = csc.row_index(entry);
                let slot = cursor[row];
                columns[slot] = column_u32;
                value_indexes[slot] = value_index[&csc.vals[entry].as_canonical_u64()];
                cursor[row] += 1;
            }
        }
        let row_counts = row_counts
            .into_iter()
            .map(|count| u16::try_from(count).map_err(|_| ExportError::new("row term count exceeds u16")))
            .collect::<Result<Vec<_>, _>>()?;
        matrices.push(MatrixPayload {
            row_counts,
            columns,
            value_indexes,
            blocks: blocks.to_vec(),
        });
    }
    let matrices: [MatrixPayload; 3] = matrices
        .try_into()
        .map_err(|_| ExportError::new("exactly three matrices"))?;
    Ok(WirePayloads { value_table, matrices })
}

/// Replay every source row: the payload expansion (CSR terms merged with the
/// production seeded expansion) must equal the independent sparse recovery.
fn replay_against_recovery(arm: &SparseR1cs, payloads: &WirePayloads) -> Result<usize, ExportError> {
    let mut offsets = [
        vec![0usize; arm.n + 1],
        vec![0usize; arm.n + 1],
        vec![0usize; arm.n + 1],
    ];
    for (matrix_index, payload) in payloads.matrices.iter().enumerate() {
        for row in 0..arm.n {
            offsets[matrix_index][row + 1] = offsets[matrix_index][row] + payload.row_counts[row] as usize;
        }
    }
    let mut replayed = 0usize;
    let mut chunk_start = 0usize;
    while chunk_start < arm.n {
        let chunk_end = (chunk_start + REPLAY_CHUNK_ROWS).min(arm.n);
        let selected: Vec<usize> = (chunk_start..chunk_end).collect();
        let recovered = recover_sparse_rows(arm, &selected)?;
        for (offset, row) in selected.iter().copied().enumerate() {
            for matrix_index in 0..3 {
                let payload = &payloads.matrices[matrix_index];
                let start = offsets[matrix_index][row];
                let stop = offsets[matrix_index][row + 1];
                let mut terms = Vec::with_capacity(stop - start);
                for slot in start..stop {
                    terms.push((
                        payload.columns[slot] as usize,
                        payloads.value_table[payload.value_indexes[slot] as usize],
                    ));
                }
                for block in &payload.blocks {
                    if row >= block.row_start() && row < block.row_end() {
                        let before = terms.len();
                        block.for_each_row_term::<F, _>(row, |column, coefficient| {
                            terms.push((column, coefficient.as_canonical_u64()));
                        });
                        let seeded = terms.split_off(before);
                        let mut seeded = seeded;
                        seeded.sort_unstable();
                        let plain = terms.split_off(0);
                        terms = merge_disjoint(plain, seeded).ok_or_else(|| {
                            ExportError::new(format!("row {row}: payload and seeded columns overlap"))
                        })?;
                    }
                }
                let expected = recovered[offset][matrix_index]
                    .iter()
                    .map(|(&column, value)| (column, value.as_canonical_u64()))
                    .collect::<Vec<_>>();
                if terms != expected {
                    return Err(ExportError::new(format!(
                        "row {row} matrix {matrix_index}: payload expansion differs from sparse recovery"
                    )));
                }
            }
            replayed += 1;
        }
        chunk_start = chunk_end;
    }
    Ok(replayed)
}

fn merge_disjoint(left: Vec<(usize, u64)>, right: Vec<(usize, u64)>) -> Option<Vec<(usize, u64)>> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    let mut l = left.into_iter().peekable();
    let mut r = right.into_iter().peekable();
    loop {
        match (l.peek(), r.peek()) {
            (None, _) => {
                out.extend(r);
                return Some(out);
            }
            (_, None) => {
                out.extend(l);
                return Some(out);
            }
            (Some(lh), Some(rh)) => {
                if lh.0 < rh.0 {
                    out.push(l.next().expect("peeked"));
                } else if rh.0 < lh.0 {
                    out.push(r.next().expect("peeked"));
                } else {
                    return None;
                }
            }
        }
    }
}

fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let word = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(word >> 18) as usize & 63] as char);
        out.push(ALPHABET[(word >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[(word >> 6) as usize & 63] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[word as usize & 63] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn le_bytes_u16(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn le_bytes_u32(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn le_bytes_u64(values: &[u64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

struct PayloadRef {
    first_part: usize,
    part_count: usize,
}

struct PartWriter {
    parts: Vec<String>,
}

impl PartWriter {
    fn push_payload(&mut self, bytes: &[u8]) -> PayloadRef {
        let encoded = base64_encode(bytes);
        let first_part = self.parts.len();
        if encoded.is_empty() {
            self.parts.push(String::new());
            return PayloadRef {
                first_part,
                part_count: 1,
            };
        }
        let mut cursor = 0;
        while cursor < encoded.len() {
            let stop = (cursor + PART_CHAR_BUDGET).min(encoded.len());
            self.parts.push(encoded[cursor..stop].to_owned());
            cursor = stop;
        }
        PayloadRef {
            first_part,
            part_count: self.parts.len() - first_part,
        }
    }
}

fn payload_expression(namespace: &str, payload: &PayloadRef) -> String {
    (payload.first_part..payload.first_part + payload.part_count)
        .map(|part| format!("{namespace}Data{part}.part"))
        .collect::<Vec<_>>()
        .join(" ++ ")
}

fn lean_string_literal(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn render_block(out: &mut String, name: &str, block: &SeededPhi81LinearBlock) {
    let rows = block.row_end() - block.row_start();
    out.push_str(&format!("def {name} : SeededPhi81.Block :=\n"));
    out.push_str(&format!("  {{ rowStart := {}\n", block.row_start()));
    out.push_str(&format!(
        "    wordStarts := [{}]\n",
        block
            .word_starts()
            .iter()
            .map(|start| start.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    out.push_str(&format!("    wordWidth := {}\n", block.word_width()));
    out.push_str(&format!("    kappa := {}\n", block.kappa()));
    out.push_str(&format!("    messageCols := {}\n", block.message_cols()));
    // The artifact expansion never reads output columns; `Block.Valid` only
    // constrains their count.
    out.push_str(&format!("    outputColumns := List.range {rows}\n"));
    out.push_str("    superneoTransformedColumns := false\n");
    out.push_str("    schedule :=\n");
    out.push_str(&format!("      {{ chunkSize := {}\n", block.chunk_size()));
    out.push_str("        seedsByOutput := [");
    for (row_index, seeds) in block.chunk_seeds_by_row().iter().enumerate() {
        if row_index != 0 {
            out.push_str(", ");
        }
        out.push('[');
        for (seed_index, seed) in seeds.iter().enumerate() {
            if seed_index != 0 {
                out.push_str(", ");
            }
            out.push('[');
            out.push_str(
                &seed
                    .iter()
                    .map(|byte| byte.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            out.push(']');
        }
        out.push(']');
    }
    out.push_str("]\n");
    out.push_str("        rejectionFuel := 4 } }\n\n");
}

fn family_ranges(source_rows: &[usize]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut iter = source_rows.iter().copied();
    let Some(first) = iter.next() else {
        return ranges;
    };
    let mut start = first;
    let mut stop = first + 1;
    for row in iter {
        if row == stop {
            stop += 1;
        } else {
            ranges.push((start, stop));
            start = row;
            stop = row + 1;
        }
    }
    ranges.push((start, stop));
    ranges
}

/// Complete compact-source emission for one physical arm.
pub struct CompactSourceEmission {
    /// Data modules (one payload part each) followed by the assembly module.
    pub modules: Vec<GeneratedLeanModule>,
    /// Rows replayed against the independent sparse recovery.
    pub replayed_rows: usize,
}

/// Render the complete source relation of one arm as string-payload Lean
/// modules. `module_namespace` is the assembly module name; data modules are
/// `<module_namespace>Data<i>`. When `committed_equality_namespace` is given,
/// the assembly also pins `sourceArtifact` equal to that module's committed
/// literal artifact.
pub fn render_compact_source_artifact_modules(
    arm: &SparseR1cs,
    profile: &str,
    module_namespace: &str,
    committed_equality_namespace: Option<&str>,
) -> Result<CompactSourceEmission, ExportError> {
    let exporter = SparseProblemExporter::new(arm)?;
    let digest = exporter.artifact_digest().to_owned();
    let census = sparse_family_census(arm)?;
    let payloads = build_payloads(arm)?;
    let replayed_rows = replay_against_recovery(arm, &payloads)?;
    if replayed_rows != arm.n {
        return Err(ExportError::new("compact source replay did not cover every row"));
    }

    let mut writer = PartWriter { parts: Vec::new() };
    let value_table_ref = writer.push_payload(&le_bytes_u64(&payloads.value_table));
    let mut matrix_refs = Vec::with_capacity(3);
    for payload in &payloads.matrices {
        let row_counts = writer.push_payload(&le_bytes_u16(&payload.row_counts));
        let columns = writer.push_payload(&le_bytes_u32(&payload.columns));
        let value_indexes = writer.push_payload(&le_bytes_u16(&payload.value_indexes));
        matrix_refs.push((row_counts, columns, value_indexes));
    }

    let mut modules = Vec::with_capacity(writer.parts.len() + 1);
    for (index, part) in writer.parts.iter().enumerate() {
        let module_name = format!("{module_namespace}Data{index}");
        let mut content = String::with_capacity(part.len() + 256);
        content.push_str("/-!\nGENERATED FILE - do not edit by hand.\n\n");
        content.push_str("One base64 payload part of the compact source artifact.\n-/\n\n");
        content.push_str(&format!("namespace {module_name}\n\n"));
        content.push_str("def part : String :=\n  ");
        content.push_str(&lean_string_literal(part));
        content.push_str(&format!("\n\nend {module_name}\n"));
        modules.push(GeneratedLeanModule { module_name, content });
    }

    let mut out = String::new();
    out.push_str("import Nightstream.Assurance.CompactSourceArtifact\n");
    if let Some(committed) = committed_equality_namespace {
        out.push_str(&format!("import {committed}\n"));
    }
    for index in 0..writer.parts.len() {
        out.push_str(&format!("import {module_namespace}Data{index}\n"));
    }
    out.push_str("\n/-!\nGENERATED FILE - do not edit by hand.\n\n");
    out.push_str("Assembly of the complete string-payload source artifact. The\n");
    out.push_str("emitter replayed every payload row against the independent sparse\n");
    out.push_str("recovery before rendering; `expand` re-derives the artifact\n");
    out.push_str("natively and fails closed on any malformation.\n-/\n\n");
    out.push_str(&format!("namespace {module_namespace}\n\n"));
    out.push_str("open Nightstream.Assurance.CompactSourceArtifact\n");
    out.push_str("open Nightstream.Assurance.ConstraintMinimization\n");
    out.push_str("open Nightstream.Implementation.R1CS (SeededPhi81)\n\n");
    out.push_str("set_option maxHeartbeats 2000000\nset_option maxRecDepth 65536\n\n");

    let mut block_names: [Vec<String>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (matrix_index, payload) in payloads.matrices.iter().enumerate() {
        for (block_index, block) in payload.blocks.iter().enumerate() {
            let name = format!("seededBlock{}{}", ["A", "B", "C"][matrix_index], block_index);
            render_block(&mut out, &name, block);
            block_names[matrix_index].push(name);
        }
    }

    out.push_str("def families : List FamilyRanges :=\n  [");
    for (family_index, family) in census.iter().enumerate() {
        if family_index != 0 {
            out.push_str(",\n   ");
        }
        let ranges = family_ranges(family.source_rows())
            .into_iter()
            .map(|(start, stop)| format!("({start}, {stop})"))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!("⟨{}, [{ranges}]⟩", lean_string_literal(family.name())));
    }
    out.push_str("]\n\n");

    for (matrix_index, (row_counts, columns, value_indexes)) in matrix_refs.iter().enumerate() {
        let label = ["A", "B", "C"][matrix_index];
        out.push_str(&format!(
            "def matrix{label} : MatrixWire where\n  rowCounts := {}\n  columns := {}\n  valueIndexes := {}\n  seededBlocks := [{}]\n\n",
            payload_expression(module_namespace, row_counts),
            payload_expression(module_namespace, columns),
            payload_expression(module_namespace, value_indexes),
            block_names[matrix_index].join(", "),
        ));
    }

    out.push_str("def wire : Wire where\n");
    out.push_str("  schema := \"nightstream/r1cs-redundancy-problem/v3\"\n");
    out.push_str(&format!("  profile := {}\n", lean_string_literal(profile)));
    out.push_str("  scope := \"branch\"\n");
    out.push_str(&format!("  diagnosticDigest := {}\n", lean_string_literal(&digest)));
    out.push_str(&format!("  fieldModulus := \"{GOLDILOCKS_MODULUS}\"\n"));
    out.push_str(&format!("  totalRows := {}\n", arm.n));
    out.push_str(&format!("  columnCount := {}\n", arm.m));
    out.push_str("  constantOneColumn := 0\n");
    out.push_str(&format!("  publicInputCount := {}\n", arm.m_in));
    out.push_str(&format!(
        "  completeFamilies := [{}]\n",
        census
            .iter()
            .map(|family| lean_string_literal(family.name()))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    out.push_str(&format!(
        "  valueTable := {}\n",
        payload_expression(module_namespace, &value_table_ref)
    ));
    out.push_str("  families := families\n");
    out.push_str("  a := matrixA\n  b := matrixB\n  c := matrixC\n\n");

    out.push_str("theorem expand_succeeds : (expand wire).isSome := by native_decide\n\n");
    out.push_str("def sourceArtifact : Artifact := (expand wire).get expand_succeeds\n\n");
    out.push_str("theorem sourceArtifact_coversFullRelation :\n");
    out.push_str("    sourceArtifact.CoversFullRelation := by native_decide\n\n");
    out.push_str("theorem sourceArtifact_exactValidation :\n");
    out.push_str("    Artifact.ExactValidation sourceArtifact sourceArtifact = true := by\n");
    out.push_str("  native_decide\n");
    if let Some(committed) = committed_equality_namespace {
        out.push_str(&format!(
            "\ntheorem sourceArtifact_matches_committed :\n    sourceArtifact = {committed}.sourceArtifact := by native_decide\n"
        ));
    }
    out.push_str(&format!("\nend {module_namespace}\n"));
    modules.push(GeneratedLeanModule {
        module_name: module_namespace.to_owned(),
        content: out,
    });
    Ok(CompactSourceEmission { modules, replayed_rows })
}

/// Render one shared background assignment as string-payload modules. The
/// assembly decodes it once; family modules apply per-column overrides.
pub fn render_assignment_payload_modules(
    values: &[u64],
    module_namespace: &str,
) -> Result<Vec<GeneratedLeanModule>, ExportError> {
    let mut writer = PartWriter { parts: Vec::new() };
    let payload = writer.push_payload(&le_bytes_u64(values));
    let mut modules = Vec::with_capacity(writer.parts.len() + 1);
    for (index, part) in writer.parts.iter().enumerate() {
        let module_name = format!("{module_namespace}Data{index}");
        let mut content = String::with_capacity(part.len() + 256);
        content.push_str("/-!\nGENERATED FILE - do not edit by hand.\n\n");
        content.push_str("One base64 payload part of a shared background assignment.\n-/\n\n");
        content.push_str(&format!("namespace {module_name}\n\n"));
        content.push_str("def part : String :=\n  ");
        content.push_str(&lean_string_literal(part));
        content.push_str(&format!("\n\nend {module_name}\n"));
        modules.push(GeneratedLeanModule { module_name, content });
    }
    let mut out = String::new();
    out.push_str("import Nightstream.Assurance.CompactSourceArtifact\n");
    for index in 0..writer.parts.len() {
        out.push_str(&format!("import {module_namespace}Data{index}\n"));
    }
    out.push_str("\n/-!\nGENERATED FILE - do not edit by hand.\n\n");
    out.push_str("Shared accepted background assignment, decoded once. Removal\n");
    out.push_str("counterexamples apply per-column overrides to these values.\n-/\n\n");
    out.push_str(&format!("namespace {module_namespace}\n\n"));
    out.push_str("open Nightstream.Assurance.CompactSourceArtifact\n\n");
    out.push_str(&format!(
        "def payload : String := {}\n\n",
        payload_expression(module_namespace, &payload)
    ));
    out.push_str("theorem decode_succeeds : (decodeAssignment payload).isSome := by\n");
    out.push_str("  native_decide\n\n");
    out.push_str("def values : Array Nat := (decodeAssignment payload).get decode_succeeds\n\n");
    out.push_str(&format!("end {module_namespace}\n"));
    modules.push(GeneratedLeanModule {
        module_name: module_namespace.to_owned(),
        content: out,
    });
    Ok(modules)
}

/// Render one removal-counterexample module against a string-payload source
/// artifact and a shared background assignment. The witness must come from
/// `find_exclusive_column_witness` over the same complete problem and
/// background; every Lean premise is re-checked here before rendering.
pub fn render_compact_removal_counterexample_lean(
    problem: &recursive_constraint_minimizer::Problem,
    background: &[u64],
    witness: &crate::ExclusiveColumnWitness,
    artifact_namespace: &str,
    background_namespace: &str,
    namespace: &str,
    reviewed_plan: &[String],
) -> Result<String, ExportError> {
    if background.len() != problem.column_count {
        return Err(ExportError::new("background length differs from the relation"));
    }
    if witness.column() >= background.len() {
        return Err(ExportError::new("witness column exceeds the relation"));
    }
    let modulus = recursive_constraint_minimizer::GOLDILOCKS_MODULUS
        .parse::<u128>()
        .expect("fixed Goldilocks modulus");
    let mutated = ((u128::from(background[witness.column()]) + u128::from(witness.delta())) % modulus) as u64;
    let mut expected = background.to_vec();
    expected[witness.column()] = mutated;
    if witness.model().values() != expected.as_slice() {
        return Err(ExportError::new(
            "witness model differs from the background plus its override",
        ));
    }
    let mut plan_names = problem.complete_families.clone();
    plan_names.sort();
    let mut reviewed_sorted = reviewed_plan.to_vec();
    reviewed_sorted.sort();
    if plan_names != reviewed_sorted {
        return Err(ExportError::new(
            "reviewed plan differs from the problem's complete families",
        ));
    }
    if !reviewed_plan.iter().any(|name| name == witness.family()) {
        return Err(ExportError::new("witness family is not in the reviewed plan"));
    }

    let mut out = String::new();
    out.push_str(&format!("import {artifact_namespace}\n"));
    out.push_str(&format!("import {background_namespace}\n"));
    out.push_str("set_option maxHeartbeats 2000000\n");
    out.push_str("set_option maxRecDepth 65536\n\n");
    out.push_str(&format!("namespace {namespace}\n\n"));
    out.push_str("open Nightstream.Assurance.ConstraintMinimization\n");
    out.push_str("open Nightstream.Assurance.CompactSourceArtifact\n");
    out.push_str(&format!("open {artifact_namespace}\n\n"));
    out.push_str(&format!(
        "def reviewedPlan : List String := [{}]\n\n",
        reviewed_plan
            .iter()
            .map(|name| lean_string_literal(name))
            .collect::<Vec<_>>()
            .join(",")
    ));
    out.push_str(&format!(
        "def overrides : List (Nat × Nat) := [({}, {})]\n\n",
        witness.column(),
        mutated
    ));
    out.push_str(&format!(
        "theorem overrides_apply :\n    (applyOverrides {background_namespace}.values overrides).isSome := by\n  native_decide\n\n"
    ));
    out.push_str(&format!(
        "def removalCounterexampleValues : List Field :=\n  ((applyOverrides {background_namespace}.values overrides).get overrides_apply).toList.map\n    (fun value => (value : Field))\n\n"
    ));
    out.push_str("def removalCounterexample : RemovalCounterexample where\n");
    out.push_str(&format!(
        "  removedFamily := {}\n",
        lean_string_literal(witness.family())
    ));
    out.push_str("  values := removalCounterexampleValues\n\n");
    out.push_str("theorem removalCounterexample_valid :\n");
    out.push_str("    removalCounterexample.Valid sourceArtifact reviewedPlan := by\n");
    out.push_str("  native_decide\n\n");
    out.push_str("theorem necessary :\n");
    out.push_str("    NecessaryForSoundness (FamilyHolds sourceArtifact)\n");
    out.push_str(&format!(
        "      (Target sourceArtifact) reviewedPlan {} :=\n",
        lean_string_literal(witness.family())
    ));
    out.push_str("  removalCounterexample.necessary_of_full_valid\n");
    out.push_str("    sourceArtifact sourceArtifact reviewedPlan\n");
    out.push_str("    sourceArtifact_coversFullRelation sourceArtifact_exactValidation\n");
    out.push_str("    removalCounterexample_valid\n\n");
    out.push_str("theorem necessaryNormalized :\n");
    out.push_str("    NecessaryForSoundness\n");
    out.push_str("      (NormalizedFamilyHolds sourceArtifact)\n");
    out.push_str(&format!(
        "      (NormalizedTarget sourceArtifact) reviewedPlan {} :=\n",
        lean_string_literal(witness.family())
    ));
    out.push_str("  removalCounterexample.necessary_normalized_of_full_valid\n");
    out.push_str("    sourceArtifact sourceArtifact reviewedPlan\n");
    out.push_str("    sourceArtifact_coversFullRelation sourceArtifact_exactValidation\n");
    out.push_str("    removalCounterexample_valid\n\n");
    out.push_str(&format!("end {namespace}\n"));
    Ok(out)
}
