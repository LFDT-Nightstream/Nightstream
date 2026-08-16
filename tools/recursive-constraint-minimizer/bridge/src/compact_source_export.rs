//! String-payload emission for complete source artifacts (v3, chunked).
//!
//! Owns: the chunk-aligned wire construction (shared value table, per-chunk
//! CSR payloads, clipped seeded blocks, family row ranges), a complete
//! replay that compares the wire expansion against the independent sparse
//! row recovery before anything is rendered, and the Lean module rendering:
//! one data module per row chunk and one assembly module whose per-chunk
//! leaf certificates are bounded `native_decide` facts glued by the
//! structural theorems of `Nightstream.Assurance.CompactSourceArtifact`.
//! No generated proof obligation evaluates the whole artifact.
//!
//! Does not own: expansion semantics (Lean `CompactSourceArtifact`), sampler
//! conformance (the mirror gate), the minimization theorems, or any removal
//! authority.

use std::collections::BTreeMap;

use neo_ccs::{CcsMatrix, CscMat, SeededPhi81LinearBlock};
use neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs;
use neo_math::F;
use p3_field::PrimeField64;

use recursive_constraint_minimizer::GOLDILOCKS_MODULUS;

use crate::lean_export::GeneratedLeanModule;
use crate::{recover_sparse_rows, sparse_family_census, ExportError, SparseProblemExporter};

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
                        let mut seeded = terms.split_off(before);
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

fn lean_string_literal(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn render_block(out: &mut String, block: &SeededPhi81LinearBlock) {
    let rows = block.row_end() - block.row_start();
    out.push_str("      { rowStart := ");
    out.push_str(&block.row_start().to_string());
    out.push_str("\n        wordStarts := [");
    out.push_str(
        &block
            .word_starts()
            .iter()
            .map(|start| start.to_string())
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push_str("]\n");
    out.push_str(&format!("        wordWidth := {}\n", block.word_width()));
    out.push_str(&format!("        kappa := {}\n", block.kappa()));
    out.push_str(&format!("        messageCols := {}\n", block.message_cols()));
    // The artifact expansion never reads output columns; `Block.Valid` only
    // constrains their count.
    out.push_str(&format!("        outputColumns := List.range {rows}\n"));
    out.push_str("        superneoTransformedColumns := false\n");
    out.push_str("        schedule :=\n");
    out.push_str(&format!("          {{ chunkSize := {}\n", block.chunk_size()));
    out.push_str("            seedsByOutput := [");
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
    out.push_str("            rejectionFuel := 4 } }");
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

/// Emit a bounded `match` dispatcher from per-chunk leaf theorems to the
/// universally quantified hypothesis the assembly theorems consume.
fn render_dispatcher(out: &mut String, theorem_name: &str, statement: &str, leaf_prefix: &str, chunk_count: usize) {
    out.push_str(&format!("theorem {theorem_name} :\n    {statement} := by\n"));
    out.push_str("  intro k bound\n  match k with\n");
    for chunk in 0..chunk_count {
        out.push_str(&format!("  | {chunk} => exact {leaf_prefix}{chunk}\n"));
    }
    out.push_str(&format!("  | n + {chunk_count} => exact absurd bound (by omega)\n\n"));
}

/// Complete compact-source emission for one physical arm.
pub struct CompactSourceEmission {
    /// Chunk data modules followed by the assembly module.
    pub modules: Vec<GeneratedLeanModule>,
    /// Rows replayed against the independent sparse recovery.
    pub replayed_rows: usize,
    /// Chunk grain used by the wire.
    pub chunk_rows: usize,
    /// Number of row chunks.
    pub chunk_count: usize,
}

/// Optional per-chunk equality pin against a committed literal artifact.
pub struct CommittedEquality {
    /// Namespace holding the literal `sourceArtifact` and its chunk defs.
    pub namespace: String,
    /// Chunk-def prefix, e.g. `sourceArtifactRowsChunk`.
    pub chunk_prefix: String,
}

/// Render the complete source relation of one arm as chunk-aligned Lean
/// modules with bounded leaf certificates. `module_namespace` is the
/// assembly module name; chunk data modules are `<module_namespace>Data<k>`.
pub fn render_compact_source_artifact_modules(
    arm: &SparseR1cs,
    profile: &str,
    module_namespace: &str,
    chunk_rows: usize,
    committed_equality: Option<&CommittedEquality>,
) -> Result<CompactSourceEmission, ExportError> {
    if chunk_rows == 0 {
        return Err(ExportError::new("chunk grain must be positive"));
    }
    let exporter = SparseProblemExporter::new(arm)?;
    let digest = exporter.artifact_digest().to_owned();
    let census = sparse_family_census(arm)?;
    let payloads = build_payloads(arm)?;
    let replayed_rows = replay_against_recovery(arm, &payloads)?;
    if replayed_rows != arm.n {
        return Err(ExportError::new("compact source replay did not cover every row"));
    }
    let chunk_count = arm.n.div_ceil(chunk_rows);

    // Per-matrix term offsets for chunk slicing.
    let mut offsets = Vec::with_capacity(3);
    for payload in &payloads.matrices {
        let mut acc = vec![0usize; arm.n + 1];
        for row in 0..arm.n {
            acc[row + 1] = acc[row] + payload.row_counts[row] as usize;
        }
        offsets.push(acc);
    }

    let mut modules = Vec::with_capacity(chunk_count + 1);
    for chunk in 0..chunk_count {
        let start = chunk * chunk_rows;
        let stop = ((chunk + 1) * chunk_rows).min(arm.n);
        let module_name = format!("{module_namespace}Data{chunk}");
        let mut content = String::new();
        content.push_str("import Nightstream.Assurance.CompactSourceArtifact\n\n");
        content.push_str("/-!\nGENERATED FILE - do not edit by hand.\n\n");
        content.push_str("One row chunk of the compact source artifact.\n-/\n\n");
        content.push_str(&format!("namespace {module_name}\n\n"));
        content.push_str("open Nightstream.Assurance.CompactSourceArtifact\n");
        content.push_str("open Nightstream.Implementation.R1CS\n\n");
        content.push_str("set_option maxHeartbeats 2000000\n");
        content.push_str("set_option maxRecDepth 65536\n\n");
        content.push_str("def chunk : ChunkWire :=\n");
        for (matrix_index, label) in ["a", "b", "c"].iter().enumerate() {
            let payload = &payloads.matrices[matrix_index];
            let term_start = offsets[matrix_index][start];
            let term_stop = offsets[matrix_index][stop];
            let counts = base64_encode(&le_bytes_u16(&payload.row_counts[start..stop]));
            let columns = base64_encode(&le_bytes_u32(&payload.columns[term_start..term_stop]));
            let indexes = base64_encode(&le_bytes_u16(&payload.value_indexes[term_start..term_stop]));
            if matrix_index == 0 {
                content.push_str("  { ");
            } else {
                content.push_str("    ");
            }
            content.push_str(&format!(
                "{label} := {{ rowCounts := {}, columns := {}, valueIndexes := {} }},\n",
                lean_string_literal(&counts),
                lean_string_literal(&columns),
                lean_string_literal(&indexes),
            ));
        }
        content.push_str("    seededBlocksA := [");
        let mut first_block = true;
        for block in &payloads.matrices[0].blocks {
            if block.row_start() < stop && block.row_end() > start {
                if !first_block {
                    content.push(',');
                }
                content.push('\n');
                render_block(&mut content, block);
                first_block = false;
            }
        }
        if first_block {
            content.push_str("] }\n");
        } else {
            content.push_str("\n    ] }\n");
        }
        content.push_str(&format!("\nend {module_name}\n"));
        modules.push(GeneratedLeanModule { module_name, content });
    }

    let mut out = String::new();
    out.push_str("import Nightstream.Assurance.CompactSourceArtifact\n");
    if let Some(committed) = committed_equality {
        out.push_str(&format!("import {}\n", committed.namespace));
    }
    for chunk in 0..chunk_count {
        out.push_str(&format!("import {module_namespace}Data{chunk}\n"));
    }
    out.push_str("\n/-!\nGENERATED FILE - do not edit by hand.\n\n");
    out.push_str("Assembly of the chunk-aligned compact source artifact. Every leaf\n");
    out.push_str("below is a bounded per-chunk fact; the artifact-level properties\n");
    out.push_str("come from the structural composition theorems, and the exact\n");
    out.push_str("validation is discharged by proof, never by evaluation. The\n");
    out.push_str("emitter replayed every payload row against the independent sparse\n");
    out.push_str("recovery before rendering.\n-/\n\n");
    out.push_str(&format!("namespace {module_namespace}\n\n"));
    out.push_str("open Nightstream.Assurance.CompactSourceArtifact\n");
    out.push_str("open Nightstream.Assurance.ConstraintMinimization\n");
    out.push_str("open Nightstream.Implementation.R1CS\n\n");
    out.push_str("set_option maxHeartbeats 2000000\n");
    out.push_str("set_option maxRecDepth 65536\n\n");

    out.push_str("def families : List FamilyRanges :=\n  [");
    for (family_index, family) in census.iter().enumerate() {
        if family_index != 0 {
            out.push_str(",\n   ");
        }
        let ranges = family_ranges(family.source_rows())
            .into_iter()
            .map(|(range_start, range_stop)| format!("({range_start}, {range_stop})"))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!("⟨{}, [{ranges}]⟩", lean_string_literal(family.name())));
    }
    out.push_str("]\n\n");

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
        lean_string_literal(&base64_encode(&le_bytes_u64(&payloads.value_table)))
    ));
    out.push_str("  families := families\n");
    out.push_str(&format!("  chunkRows := {chunk_rows}\n"));
    out.push_str("  chunks := #[");
    for chunk in 0..chunk_count {
        if chunk != 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("{module_namespace}Data{chunk}.chunk"));
    }
    out.push_str("]\n\n");
    out.push_str("def sourceArtifact : Artifact := sourceArtifactOf wire\n\n");
    out.push_str("def reviewedPlan : List String := sourceArtifact.completeFamilies\n\n");
    out.push_str("theorem reviewedPlan_subset :\n");
    out.push_str("    ∀ family ∈ reviewedPlan, family ∈ sourceArtifact.completeFamilies :=\n");
    out.push_str("  fun _ membership => membership\n\n");

    // Bounded per-chunk leaves.
    for chunk in 0..chunk_count {
        let start = chunk * chunk_rows;
        let length = (arm.n - start).min(chunk_rows);
        out.push_str(&format!(
            "theorem censusLeaf{chunk} :\n    (rowsChunk wire {chunk}).map (fun row => row.sourceIndex) =\n      List.range' {start} {length} := by\n  native_decide\n\n"
        ));
        out.push_str(&format!(
            "theorem rowsWfLeaf{chunk} :\n    (rowsChunk wire {chunk}).all (rowWellFormedAt {} {}) = true := by\n  native_decide\n\n",
            arm.n, arm.m
        ));
        out.push_str(&format!(
            "theorem familiesLeaf{chunk} :\n    (rowsChunk wire {chunk}).all\n      (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by\n  native_decide\n\n"
        ));
    }

    // Family presence leaves: one bounded fact per family.
    for (family_index, family) in census.iter().enumerate() {
        let first_row = family.source_rows()[0];
        let chunk = first_row / chunk_rows;
        out.push_str(&format!(
            "theorem presenceLeaf{family_index} :\n    (rowsChunk wire {chunk}).any\n      (fun row => decide (row.family = {})) = true := by\n  native_decide\n\n",
            lean_string_literal(family.name())
        ));
    }

    // Dispatchers.
    render_dispatcher(
        &mut out,
        "censuses",
        "∀ k, k < wire.chunkCount →\n      (rowsChunk wire k).map (fun row => row.sourceIndex) =\n        List.range' (wire.chunkStart k) (wire.chunkLength k)",
        "censusLeaf",
        chunk_count,
    );
    render_dispatcher(
        &mut out,
        "rowsWf",
        &format!(
            "∀ k, k < wire.chunkCount →\n      (rowsChunk wire k).all (rowWellFormedAt {} {}) = true",
            arm.n, arm.m
        ),
        "rowsWfLeaf",
        chunk_count,
    );
    render_dispatcher(
        &mut out,
        "familiesCovered",
        "∀ k, k < wire.chunkCount →\n      (rowsChunk wire k).all\n        (fun row => decide (row.family ∈ wire.completeFamilies)) = true",
        "familiesLeaf",
        chunk_count,
    );

    // Chunk arithmetic: bounded scalar facts.
    out.push_str("theorem chunkArithmeticFull :\n    ∀ k, k + 1 < wire.chunkCount → wire.chunkLength k = wire.chunkRows := by\n  native_decide\n\n");
    out.push_str("theorem chunkArithmeticLast :\n    wire.chunkCount ≠ 0 →\n      (wire.chunkCount - 1) * wire.chunkRows +\n        wire.chunkLength (wire.chunkCount - 1) = wire.totalRows := by\n  native_decide\n\n");
    out.push_str("theorem chunkArithmeticLead :\n    wire.chunkCount ≠ 0 →\n      (wire.chunkCount - 1) * wire.chunkRows ≤ wire.totalRows := by\n  native_decide\n\n");
    out.push_str(
        "theorem chunkArithmeticEmpty :\n    wire.chunkCount = 0 → wire.totalRows = 0 := by\n  native_decide\n\n",
    );

    // Family presence assembled from the per-family leaves.
    out.push_str("theorem familyPresence :\n    sourceArtifact.completeFamilies.all\n      (fun family =>\n        sourceArtifact.rows.any\n          (fun row => decide (row.family = family))) = true := by\n");
    out.push_str("  rw [List.all_eq_true]\n  intro family membership\n");
    out.push_str("  have present : ∃ chunk, chunk < wire.chunkCount ∧\n      (rowsChunk wire chunk).any\n        (fun row => decide (row.family = family)) = true := by\n");
    out.push_str("    fin_cases membership\n");
    for (family_index, family) in census.iter().enumerate() {
        let first_row = family.source_rows()[0];
        let chunk = first_row / chunk_rows;
        out.push_str(&format!(
            "    · exact ⟨{chunk}, by native_decide, presenceLeaf{family_index}⟩\n"
        ));
    }
    out.push_str("  rcases present with ⟨chunk, chunkBound, chunkAny⟩\n");
    out.push_str("  rw [List.any_eq_true] at chunkAny ⊢\n");
    out.push_str("  rcases chunkAny with ⟨row, rowMember, rowFamily⟩\n");
    out.push_str("  refine ⟨row, ?_, rowFamily⟩\n");
    out.push_str("  show row ∈ artifactRows wire\n");
    out.push_str("  unfold artifactRows\n");
    out.push_str("  exact List.mem_flatMap.mpr ⟨chunk, List.mem_range.mpr chunkBound, rowMember⟩\n\n");

    // Scalar conjuncts: one small closed pack.
    out.push_str("theorem scalarFacts :\n    sourceArtifact.schema = Artifact.supportedSchema ∧\n      sourceArtifact.profile ≠ \"\" ∧\n      sourceArtifact.scope ∈ Artifact.scopes ∧\n      sourceArtifact.diagnosticDigest ≠ \"\" ∧\n      sourceArtifact.fieldModulus = Artifact.goldilocksModulusDecimal ∧\n      0 < sourceArtifact.totalRows ∧\n      0 < sourceArtifact.columnCount ∧\n      0 < sourceArtifact.publicInputCount ∧\n      sourceArtifact.publicInputCount ≤ sourceArtifact.columnCount ∧\n      sourceArtifact.constantOneColumn < sourceArtifact.publicInputCount ∧\n      sourceArtifact.completeFamilies.Nodup ∧\n      sourceArtifact.completeFamilies.all\n        (fun family => decide (family ≠ \"\")) = true := by\n  native_decide\n\n");

    // Assembly.
    out.push_str("theorem sourceArtifact_indexCensus :\n    (artifactRows wire).map (fun row => row.sourceIndex) =\n      List.range wire.totalRows :=\n");
    out.push_str("  covers_indexes_of_chunks wire censuses chunkArithmeticFull\n    chunkArithmeticLast chunkArithmeticLead chunkArithmeticEmpty\n\n");
    out.push_str("theorem sourceArtifact_coversFullRelation :\n    sourceArtifact.CoversFullRelation :=\n");
    out.push_str("  coversFullRelation_of_chunks wire censuses chunkArithmeticFull\n    chunkArithmeticLast chunkArithmeticLead chunkArithmeticEmpty familiesCovered\n\n");
    out.push_str("theorem sourceArtifact_wellFormed : sourceArtifact.WellFormed :=\n");
    out.push_str("  wellFormed_of_chunks wire scalarFacts sourceArtifact_indexCensus\n    rowsWf familyPresence\n\n");
    out.push_str("theorem sourceArtifact_exactValidation :\n    Artifact.ExactValidation sourceArtifact sourceArtifact = true :=\n");
    out.push_str("  exactValidation_self sourceArtifact_wellFormed\n");

    if let Some(committed) = committed_equality {
        out.push_str("\n-- Per-chunk equality pins against the committed literal artifact.\n");
        out.push_str("-- Together with both artifacts' coverage theorems these leaves make\n");
        out.push_str("-- any divergence impossible without a leaf failing.\n");
        for chunk in 0..chunk_count {
            out.push_str(&format!(
                "theorem equalityLeaf{chunk} :\n    rowsChunk wire {chunk} = {}.{}{chunk} := by\n  native_decide\n\n",
                committed.namespace, committed.chunk_prefix
            ));
        }
    }
    out.push_str(&format!("\nend {module_namespace}\n"));
    modules.push(GeneratedLeanModule {
        module_name: module_namespace.to_owned(),
        content: out,
    });
    Ok(CompactSourceEmission {
        modules,
        replayed_rows,
        chunk_rows,
        chunk_count,
    })
}

/// Render one shared background assignment as string-payload modules with
/// its two bounded leaves: the decoded size and the constant-one entry.
pub fn render_assignment_payload_modules(
    values: &[u64],
    module_namespace: &str,
) -> Result<Vec<GeneratedLeanModule>, ExportError> {
    const PART_CHAR_BUDGET: usize = 7_800_000;
    let encoded = base64_encode(&le_bytes_u64(values));
    let mut parts = Vec::new();
    if encoded.is_empty() {
        parts.push(String::new());
    } else {
        let mut cursor = 0;
        while cursor < encoded.len() {
            let stop = (cursor + PART_CHAR_BUDGET).min(encoded.len());
            parts.push(encoded[cursor..stop].to_owned());
            cursor = stop;
        }
    }
    let mut modules = Vec::with_capacity(parts.len() + 1);
    for (index, part) in parts.iter().enumerate() {
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
    for index in 0..parts.len() {
        out.push_str(&format!("import {module_namespace}Data{index}\n"));
    }
    out.push_str("\n/-!\nGENERATED FILE - do not edit by hand.\n\n");
    out.push_str("Shared accepted background assignment, decoded once. Removal\n");
    out.push_str("counterexamples override single columns of these values. The two\n");
    out.push_str("leaves below are the only facts that force the decode.\n-/\n\n");
    out.push_str(&format!("namespace {module_namespace}\n\n"));
    out.push_str("open Nightstream.Assurance.CompactSourceArtifact\n\n");
    out.push_str("def payload : String :=\n  ");
    out.push_str(
        &(0..parts.len())
            .map(|index| format!("{module_namespace}Data{index}.part"))
            .collect::<Vec<_>>()
            .join(" ++ "),
    );
    out.push_str("\n\ndef values : Array Nat := (decodeAssignment payload).getD #[]\n\n");
    out.push_str(&format!(
        "theorem values_size : values.size = {} := by\n  native_decide\n\n",
        values.len()
    ));
    out.push_str("theorem values_one : values.getD 0 0 = 1 := by\n  native_decide\n\n");
    out.push_str(&format!("end {module_namespace}\n"));
    modules.push(GeneratedLeanModule {
        module_name: module_namespace.to_owned(),
        content: out,
    });
    Ok(modules)
}

/// One override of a classification batch: the family, its exclusive
/// column, and the mutated canonical value.
pub struct ClassificationOverride {
    /// Family the override removes.
    pub family: String,
    /// Exclusive column the witness mutates.
    pub column: usize,
    /// Mutated canonical residue at that column.
    pub value: u64,
}

/// Render the shared classification-leaves module of one batch: per-chunk
/// background-holds leaves and override-guard leaves (each bounded), their
/// dispatchers, and the override pair table every family module cites.
pub fn render_classification_leaves_module(
    artifact_namespace: &str,
    assignment_namespace: &str,
    module_namespace: &str,
    chunk_count: usize,
    overrides: &[ClassificationOverride],
) -> Result<GeneratedLeanModule, ExportError> {
    if overrides.is_empty() {
        return Err(ExportError::new("a classification batch needs at least one override"));
    }
    let mut out = String::new();
    out.push_str(&format!("import {artifact_namespace}\n"));
    out.push_str(&format!("import {assignment_namespace}\n"));
    out.push_str("\n/-!\nGENERATED FILE - do not edit by hand.\n\n");
    out.push_str("Shared classification leaves: every chunk proves once that the\n");
    out.push_str("background satisfies its rows and that each row either belongs to\n");
    out.push_str("an override's family or avoids the override's column. All family\n");
    out.push_str("necessity modules of the batch reuse these leaves.\n-/\n\n");
    out.push_str(&format!("namespace {module_namespace}\n\n"));
    out.push_str("open Nightstream.Assurance.CompactSourceArtifact\n");
    out.push_str("open Nightstream.Assurance.ConstraintMinimization\n");
    out.push_str(&format!("open {artifact_namespace}\n\n"));
    out.push_str("set_option maxHeartbeats 2000000\n");
    out.push_str("set_option maxRecDepth 65536\n\n");
    out.push_str(&format!(
        "def background : Nat → Field := backgroundFn {assignment_namespace}.values\n\n"
    ));
    out.push_str("def overridePairs : List (Nat × String) :=\n  [");
    for (index, override_entry) in overrides.iter().enumerate() {
        if index != 0 {
            out.push_str(",\n   ");
        }
        out.push_str(&format!(
            "({}, {})",
            override_entry.column,
            lean_string_literal(&override_entry.family)
        ));
    }
    out.push_str("]\n\n");
    for chunk in 0..chunk_count {
        out.push_str(&format!(
            "theorem holdsLeaf{chunk} :\n    (rowsChunk wire {chunk}).all\n      (fun row => decide (Algebraic.Holds background row.row)) = true := by\n  native_decide\n\n"
        ));
        out.push_str(&format!(
            "theorem guardsLeaf{chunk} :\n    chunkGuardsOverrides overridePairs (rowsChunk wire {chunk}) = true := by\n  native_decide\n\n"
        ));
    }
    render_dispatcher(
        &mut out,
        "holdsAll",
        "∀ k, k < wire.chunkCount →\n      (rowsChunk wire k).all\n        (fun row => decide (Algebraic.Holds background row.row)) = true",
        "holdsLeaf",
        chunk_count,
    );
    render_dispatcher(
        &mut out,
        "guardsAll",
        "∀ k, k < wire.chunkCount →\n      chunkGuardsOverrides overridePairs (rowsChunk wire k) = true",
        "guardsLeaf",
        chunk_count,
    );
    out.push_str(&format!("end {module_namespace}\n"));
    Ok(GeneratedLeanModule {
        module_name: module_namespace.to_owned(),
        content: out,
    })
}

/// Render one family's compact necessity module: two bounded leaves (the
/// violated row's membership in its chunk and its violation under the
/// override) plus structural glue through `mkCounterexample_valid` and the
/// Artifact-level full theorems.
#[allow(clippy::too_many_arguments)]
pub fn render_compact_removal_counterexample_lean(
    artifact_namespace: &str,
    assignment_namespace: &str,
    leaves_namespace: &str,
    namespace: &str,
    override_entry: &ClassificationOverride,
    violated_row: &recursive_constraint_minimizer::Row,
    violated_chunk: usize,
    chunk_count: usize,
) -> Result<String, ExportError> {
    if violated_chunk >= chunk_count {
        return Err(ExportError::new("violated chunk exceeds the wire"));
    }
    let render_terms = |terms: &[recursive_constraint_minimizer::Term]| -> Result<String, ExportError> {
        let rendered = terms
            .iter()
            .map(|term| {
                term.coefficient
                    .parse::<u64>()
                    .map(|value| format!("({}, {})", term.column, value))
                    .map_err(|_| ExportError::new("noncanonical coefficient in the violated row"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(format!("[{}]", rendered.join(", ")))
    };
    let mut out = String::new();
    out.push_str(&format!("import {artifact_namespace}\n"));
    out.push_str(&format!("import {leaves_namespace}\n"));
    out.push_str("\n/-!\nGENERATED FILE - do not edit by hand.\n-/\n\n");
    out.push_str(&format!("namespace {namespace}\n\n"));
    out.push_str("open Nightstream.Assurance.CompactSourceArtifact\n");
    out.push_str("open Nightstream.Assurance.ConstraintMinimization\n");
    out.push_str(&format!("open {artifact_namespace}\n\n"));
    out.push_str("set_option maxHeartbeats 2000000\n");
    out.push_str("set_option maxRecDepth 65536\n\n");
    out.push_str(&format!("def column : Nat := {}\n\n", override_entry.column));
    out.push_str(&format!("def value : Nat := {}\n\n", override_entry.value));
    out.push_str(&format!(
        "def removedFamily : String := {}\n\n",
        lean_string_literal(&override_entry.family)
    ));
    out.push_str("def violatedRow : IndexedRow :=\n");
    out.push_str(&format!(
        "  ⟨{}, {}, ⟨{}, {}, {}⟩⟩\n\n",
        violated_row.source_index,
        lean_string_literal(&violated_row.family),
        render_terms(&violated_row.a)?,
        render_terms(&violated_row.b)?,
        render_terms(&violated_row.c)?,
    ));
    out.push_str(&format!(
        "theorem violated_mem : violatedRow ∈ rowsChunk wire {violated_chunk} := by\n  native_decide\n\n"
    ));
    out.push_str(&format!(
        "theorem violation :\n    ¬ Algebraic.Holds\n      (overrideAt {leaves_namespace}.background column (value : Field))\n      violatedRow.row := by\n  native_decide\n\n"
    ));
    out.push_str(&format!(
        "theorem pair_member : (column, removedFamily) ∈ {leaves_namespace}.overridePairs := by\n  native_decide\n\n"
    ));
    out.push_str(&format!(
        "theorem column_inRange : column < {assignment_namespace}.values.size := by\n  rw [{assignment_namespace}.values_size]\n  decide\n\n"
    ));
    out.push_str(&format!(
        "theorem constant_one :\n    overrideAt {leaves_namespace}.background column (value : Field)\n      wire.constantOneColumn = 1 := by\n  have distinct : wire.constantOneColumn ≠ column := by native_decide\n  show overrideAt _ _ _ wire.constantOneColumn = 1\n  unfold overrideAt\n  rw [if_neg distinct]\n  show {leaves_namespace}.background wire.constantOneColumn = 1\n  have zero : wire.constantOneColumn = 0 := by native_decide\n  rw [zero]\n  show ((({assignment_namespace}.values.getD 0 0 : Nat)) : Field) = 1\n  rw [{assignment_namespace}.values_one]\n  norm_num\n\n"
    ));
    out.push_str("def removalCounterexample : RemovalCounterexample :=\n");
    out.push_str(&format!(
        "  mkCounterexample {assignment_namespace}.values column value removedFamily\n\n"
    ));
    out.push_str(
        "theorem removalCounterexample_valid :\n    removalCounterexample.Valid sourceArtifact reviewedPlan :=\n",
    );
    out.push_str(&format!(
        "  mkCounterexample_valid wire {assignment_namespace}.values\n    {leaves_namespace}.overridePairs column value removedFamily reviewedPlan\n    reviewedPlan_subset\n    (by rw [{assignment_namespace}.values_size]; native_decide)\n    column_inRange constant_one pair_member {leaves_namespace}.guardsAll\n    {leaves_namespace}.holdsAll violatedRow {violated_chunk}\n    ⟨by native_decide, violated_mem⟩ violation\n\n"
    ));
    out.push_str("theorem necessary :\n    NecessaryForSoundness (FamilyHolds sourceArtifact)\n      (Target sourceArtifact) reviewedPlan removedFamily :=\n");
    out.push_str("  removalCounterexample.necessary_of_full_valid\n    sourceArtifact sourceArtifact reviewedPlan\n    sourceArtifact_coversFullRelation sourceArtifact_exactValidation\n    removalCounterexample_valid\n\n");
    out.push_str("theorem necessaryNormalized :\n    NecessaryForSoundness\n      (NormalizedFamilyHolds sourceArtifact)\n      (NormalizedTarget sourceArtifact) reviewedPlan removedFamily :=\n");
    out.push_str("  removalCounterexample.necessary_normalized_of_full_valid\n    sourceArtifact sourceArtifact reviewedPlan\n    sourceArtifact_coversFullRelation sourceArtifact_exactValidation\n    removalCounterexample_valid\n\n");
    out.push_str(&format!("end {namespace}\n"));
    Ok(out)
}
