//! Exact S-box ownership census for the outgoing-accumulator aggregate hash.
//!
//! Owns: the compact call manifest, isolated-permutation replay, sponge
//! schedule replay, prehash boundary ownership checks, and whole-matrix use
//! census for every candidate `x^7` output in the selected aggregate stage.
//!
//! Does not own: a compact S-box encoding, centered substitution, semantic
//! accumulator validity, or authority for any carried digest.
//!
//! Emits constraints: no.
//!
//! Authority boundary: production source rows are the local implementation
//! arithmetic reference. Encoding traces only locate candidates; this
//! validator re-emits the production permutation, renames it over every call,
//! compares all A/B/C rows, and scans the entire matrix before returning a
//! manifest. The manifest does not establish protocol necessity.
//!
//! | Stage path | Function | Equation | Multiplicity | Source rows/formula | Lowered gate | Lean theorem |
//! |---|---|---|---:|---|---|---|
//! | `output_authority.prehash` | domain and child-count constants | eight fresh affine bindings | 8 | eight affine rows / columns | none | open |
//! | `output_authority.poseidon2.sponge` | absorb | `next_i = state_i + input_i` | 64 | one affine row per input | none | open |
//! | `output_authority.poseidon2.sponge` | pad | `next_0 = state_0 + 1` | 1 | one affine row | none | open |
//! | `output_authority.poseidon2_sbox.definition` | S-box | `x2=x*x; x4=x2*x2; x6=x2*x4; x7=x*x6` | 1,462 | four product rows | none | `Sbox7Compact` |
//! | `output_authority.poseidon2_sbox.consumers` | linear layers | exact uses of each `x7` output | 1,462 | eight A-uses plus one C-definition | none | `Sbox7OutputLayout` |
//! | `output_authority.digest_binding` | outgoing authority | `claimed_digest = computed_digest` | 4 | four affine rows | none | open |

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::{
    enforce_poseidon2_permutation, Lc, PoseidonPermutationTraceEntry, R1csBuilder, R1csEncodingTrace, R1csSnapshot,
    Sbox7TraceEntry, Var,
};
use crate::paper::f_prime::stage;

const WIDTH: usize = 8;
const RATE: usize = 4;
const DIGEST_LEN: usize = 4;
const EXPECTED_HASH_INPUTS: usize = 64;
const EXPECTED_PREHASH_ROWS: usize = 8;
const EXPECTED_PREHASH_COLUMNS: usize = 8;
const EXPECTED_FULL_ABSORBS: usize = 16;
const EXPECTED_PARTIAL_ABSORB_FIELDS: usize = 0;
const EXPECTED_PERMUTATIONS: usize = 17;
const SBOXES_PER_PERMUTATION: usize = 86;
const EXPECTED_SBOXES: usize = 1_462;
const EXPECTED_STAGE_ROWS: usize = 10_278;
const EXPECTED_STAGE_COLUMNS: usize = 10_278;
const EXPECTED_PERMUTATION_ROWS: usize = 600;
const EXPECTED_PERMUTATION_COLUMNS: usize = 600;
const EXPECTED_INITIAL_SBOXES_PER_PERMUTATION: usize = 32;
const EXPECTED_PARTIAL_SBOXES_PER_PERMUTATION: usize = 22;
const EXPECTED_TERMINAL_SBOXES_PER_PERMUTATION: usize = 32;
const EXPECTED_A_USES_PER_SBOX_OUTPUT: usize = 8;
const EXPECTED_C_USES_PER_SBOX_OUTPUT: usize = 1;

/// One affine renaming of the isolated WIDTH-8 production permutation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poseidon2PermutationCall {
    pub trace_index: usize,
    pub source_rows: Range<usize>,
    pub input_columns: [usize; WIDTH],
    pub first_allocated_column: usize,
    pub allocated_column_count: usize,
    pub output_columns: [usize; WIDTH],
}

/// Exact census proven by [`audit_output_authority_poseidon2_sboxes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputAuthorityPoseidon2SboxCensus {
    pub scanned_source_rows: usize,
    pub scanned_source_columns: usize,
    pub stage_rows: usize,
    pub stage_columns: usize,
    pub prehash_binding_rows: usize,
    pub prehash_fresh_columns: usize,
    pub hash_input_fields: usize,
    pub full_absorb_rounds: usize,
    pub partial_absorb_fields: usize,
    pub pad_rounds: usize,
    pub permutations: usize,
    pub initial_external_sboxes: usize,
    pub partial_sboxes: usize,
    pub terminal_external_sboxes: usize,
    pub candidate_sbox_outputs: usize,
    pub definition_uses: usize,
    pub linear_consumer_uses: usize,
    pub total_matrix_uses: usize,
}

/// S-box order inside one isolated production permutation.
///
/// These ranges index
/// [`OutputAuthorityPoseidon2SboxManifest::isolated_sbox_output_offsets`].
/// They are suitable for a generated Lean manifest because they derive from
/// production call order and contain no absolute production columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputAuthorityPoseidon2SboxFamilyLayout {
    pub initial_external: Range<usize>,
    pub partial: Range<usize>,
    pub terminal_external: Range<usize>,
}

/// Compact, exact call geometry for all output-authority S-box candidates.
///
/// The manifest stores 17 call records plus 86 isolated output offsets, not
/// a handwritten list of 1,462 columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputAuthorityPoseidon2SboxManifest {
    pub stage_rows: Range<usize>,
    pub stage_columns: Range<usize>,
    pub prehash_rows: Range<usize>,
    pub prehash_columns: Range<usize>,
    pub hash_index: usize,
    pub hash_rows: Range<usize>,
    pub hash_input_columns: Vec<usize>,
    pub hash_output_columns: [usize; DIGEST_LEN],
    pub claimed_digest_columns: [usize; DIGEST_LEN],
    pub semantic_state_output_columns: [usize; DIGEST_LEN],
    pub permutation_trace_range: Range<usize>,
    pub sbox_trace_range: Range<usize>,
    pub calls: Vec<Poseidon2PermutationCall>,
    pub census: OutputAuthorityPoseidon2SboxCensus,
    pub family_layout: OutputAuthorityPoseidon2SboxFamilyLayout,
    isolated_sbox_output_offsets: Vec<usize>,
}

impl OutputAuthorityPoseidon2SboxManifest {
    /// Resolve one candidate without materializing the full 36,292-column list.
    pub fn candidate_column(&self, call: usize, sbox: usize) -> Option<usize> {
        let call = self.calls.get(call)?;
        let offset = *self.isolated_sbox_output_offsets.get(sbox)?;
        Some(call.first_allocated_column + offset)
    }

    pub fn first_candidate_column(&self) -> usize {
        self.candidate_column(0, 0)
            .expect("validated output-authority manifest is nonempty")
    }

    /// The 86 fresh-column offsets for one isolated production permutation.
    /// Combining this slice with each call's `first_allocated_column` yields
    /// the full census without storing 36,292 absolute columns.
    pub fn isolated_sbox_output_offsets(&self) -> &[usize] {
        &self.isolated_sbox_output_offsets
    }
}

#[derive(Debug, thiserror::Error)]
#[error("output-authority Poseidon2 S-box manifest: {scope}: {detail}")]
pub struct OutputAuthorityPoseidon2SboxManifestError {
    scope: &'static str,
    detail: String,
}

fn invalid(scope: &'static str, detail: impl Into<String>) -> OutputAuthorityPoseidon2SboxManifestError {
    OutputAuthorityPoseidon2SboxManifestError {
        scope,
        detail: detail.into(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Matrix {
    A,
    B,
    C,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MatrixUse {
    row: usize,
    matrix: Matrix,
    coefficient: F,
}

struct IsolatedPermutation {
    source: R1csSnapshot,
    permutation: PoseidonPermutationTraceEntry,
    sboxes: Vec<Sbox7TraceEntry>,
    output_offsets: Vec<usize>,
    uses_by_sbox: Vec<Vec<MatrixUse>>,
}

fn isolated_permutation() -> Result<IsolatedPermutation, OutputAuthorityPoseidon2SboxManifestError> {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    let inputs: [Var; WIDTH] = std::array::from_fn(|lane| builder.alloc(F::from_u64(lane as u64 + 1)));
    enforce_poseidon2_permutation(&mut builder, &inputs);
    let source = builder.snapshot();
    let trace = builder.encoding_trace();
    if trace.poseidon_permutations().len() != 1 || trace.sbox7().len() != SBOXES_PER_PERMUTATION {
        return Err(invalid("isolated", "production emitter has unexpected trace census"));
    }
    let permutation = trace.poseidon_permutations()[0].clone();
    if permutation.source_rows.len() != EXPECTED_PERMUTATION_ROWS
        || permutation.allocated_columns.len() != EXPECTED_PERMUTATION_COLUMNS
        || permutation.source_rows != (0..source.rows())
    {
        return Err(invalid("isolated", "production permutation geometry drifted"));
    }
    let output_offsets = trace
        .sbox7()
        .iter()
        .map(|sbox| {
            sbox.output
                .col()
                .checked_sub(permutation.allocated_columns.start)
                .ok_or_else(|| invalid("isolated", "S-box output precedes fresh permutation columns"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut by_column = vec![usize::MAX; source.cols()];
    for (index, sbox) in trace.sbox7().iter().enumerate() {
        let column = sbox.output.col();
        if column >= source.cols() || by_column[column] != usize::MAX {
            return Err(invalid("isolated", "S-box output columns are not unique and in range"));
        }
        by_column[column] = index;
    }
    let uses_by_sbox = collect_uses(&source, &by_column, SBOXES_PER_PERMUTATION);
    for uses in &uses_by_sbox {
        let a_uses = uses
            .iter()
            .filter(|entry| entry.matrix == Matrix::A)
            .count();
        let b_uses = uses
            .iter()
            .filter(|entry| entry.matrix == Matrix::B)
            .count();
        let c_uses = uses
            .iter()
            .filter(|entry| entry.matrix == Matrix::C)
            .count();
        if (a_uses, b_uses, c_uses) != (EXPECTED_A_USES_PER_SBOX_OUTPUT, 0, EXPECTED_C_USES_PER_SBOX_OUTPUT) {
            return Err(invalid("isolated", "S-box output use geometry drifted"));
        }
    }
    Ok(IsolatedPermutation {
        source,
        permutation,
        sboxes: trace.sbox7().to_vec(),
        output_offsets,
        uses_by_sbox,
    })
}

fn ranges_overlap(left: &Range<usize>, right: &Range<usize>) -> bool {
    left.start < right.end && right.start < left.end
}

fn find_stage(
    trace: &R1csEncodingTrace,
) -> Result<(Range<usize>, Range<usize>, [usize; DIGEST_LEN]), OutputAuthorityPoseidon2SboxManifestError> {
    let one = |label: &'static str| {
        let matches = trace
            .stages()
            .iter()
            .enumerate()
            .filter(|(_, checkpoint)| checkpoint.label == label)
            .collect::<Vec<_>>();
        let [(index, checkpoint)] = matches.as_slice() else {
            return Err(invalid(
                "stage",
                format!("expected one {label} checkpoint, found {}", matches.len()),
            ));
        };
        Ok((*index, (*checkpoint).clone()))
    };
    let (_, root) = one(stage::RECURSIVE_ACCUMULATOR_OUTPUT)?;
    let (_, children) = one(stage::RECURSIVE_ACCUMULATOR_OUTPUT_CHILD_DIGESTS)?;
    let (aggregate_index, start) = one(stage::RECURSIVE_ACCUMULATOR_OUTPUT_AGGREGATE)?;
    let (_, end) = one(stage::RECURSIVE_COUNTERS)?;
    if children.row != root.row
        || children.col != root.col + DIGEST_LEN
        || start.row < children.row
        || start.col < children.col
        || aggregate_index + 1 >= trace.stages().len()
        || trace.stages()[aggregate_index + 1].label != stage::RECURSIVE_COUNTERS
    {
        return Err(invalid("stage", "selected output-authority stage order drifted"));
    }
    let rows = start.row..end.row;
    let columns = start.col..end.col;
    if rows.len() != EXPECTED_STAGE_ROWS || columns.len() != EXPECTED_STAGE_COLUMNS {
        return Err(invalid(
            "stage",
            format!(
                "expected {EXPECTED_STAGE_ROWS} rows/{EXPECTED_STAGE_COLUMNS} columns, found {}/{}",
                rows.len(),
                columns.len()
            ),
        ));
    }
    Ok((rows, columns, std::array::from_fn(|lane| root.col + lane)))
}

fn normalized_terms(terms: impl IntoIterator<Item = (usize, F)>) -> Vec<(usize, F)> {
    let mut normalized = BTreeMap::<usize, F>::new();
    for (column, coefficient) in terms {
        *normalized.entry(column).or_insert(F::ZERO) += coefficient;
    }
    normalized
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != F::ZERO)
        .collect()
}

fn affine_row(terms: impl IntoIterator<Item = (usize, F)>) -> (Vec<(usize, F)>, Vec<(usize, F)>, Vec<(usize, F)>) {
    (normalized_terms(terms), vec![(Var::ONE.col(), F::ONE)], Vec::new())
}

fn expect_row(
    source: &R1csSnapshot,
    row: usize,
    expected: &(Vec<(usize, F)>, Vec<(usize, F)>, Vec<(usize, F)>),
    scope: &'static str,
) -> Result<(), OutputAuthorityPoseidon2SboxManifestError> {
    if row >= source.rows() {
        return Err(invalid(scope, format!("row {row} is out of range")));
    }
    if source.a_row(row) != expected.0.as_slice()
        || source.b_row(row) != expected.1.as_slice()
        || source.c_row(row) != expected.2.as_slice()
    {
        return Err(invalid(
            scope,
            format!("row {row} does not match the exact source program"),
        ));
    }
    Ok(())
}

fn eval_sparse_row(row: &[(usize, F)], witness: &[F]) -> F {
    row.iter().fold(F::ZERO, |value, &(column, coefficient)| {
        value + coefficient * witness[column]
    })
}

fn validate_prehash_bindings(
    source: &R1csSnapshot,
    rows: Range<usize>,
    columns: Range<usize>,
) -> Result<(), OutputAuthorityPoseidon2SboxManifestError> {
    if rows.len() != EXPECTED_PREHASH_ROWS || columns.len() != EXPECTED_PREHASH_COLUMNS {
        return Err(invalid("prehash", "prehash row/column census drifted"));
    }
    let mut next_fresh = columns.start;
    for row in rows {
        if source.b_row(row) != [(Var::ONE.col(), F::ONE)] || !source.c_row(row).is_empty() {
            return Err(invalid("prehash", format!("row {row} is not an affine binding")));
        }
        let a = source.a_row(row);
        if eval_sparse_row(a, source.witness()) != F::ZERO {
            return Err(invalid(
                "prehash",
                format!("row {row} is not satisfied by its bound value"),
            ));
        }
        let fresh_terms = a
            .iter()
            .filter(|&&(column, _)| columns.contains(&column))
            .collect::<Vec<_>>();
        if fresh_terms.len() != 1
            || *fresh_terms[0] != (next_fresh, F::ONE)
            || a.iter()
                .any(|&(column, _)| column != Var::ONE.col() && column != next_fresh)
        {
            return Err(invalid(
                "prehash",
                format!("row {row} does not bind fresh column {next_fresh}"),
            ));
        }
        next_fresh += 1;
    }
    if next_fresh != columns.end {
        return Err(invalid(
            "prehash",
            "prehash fresh-column ownership has a gap or duplicate",
        ));
    }
    Ok(())
}

fn map_column(
    isolated: &PoseidonPermutationTraceEntry,
    call: &PoseidonPermutationTraceEntry,
    column: usize,
) -> Result<usize, OutputAuthorityPoseidon2SboxManifestError> {
    if column == Var::ONE.col() {
        return Ok(Var::ONE.col());
    }
    if let Some(lane) = isolated
        .input_columns
        .iter()
        .position(|&input| input == column)
    {
        return Ok(call.input_columns[lane]);
    }
    if isolated.allocated_columns.contains(&column) {
        return Ok(call.allocated_columns.start + column - isolated.allocated_columns.start);
    }
    Err(invalid(
        "permutation",
        format!("isolated column {column} has no call-site mapping"),
    ))
}

fn map_row(
    row: &[(usize, F)],
    isolated: &PoseidonPermutationTraceEntry,
    call: &PoseidonPermutationTraceEntry,
) -> Result<Vec<(usize, F)>, OutputAuthorityPoseidon2SboxManifestError> {
    Ok(normalized_terms(
        row.iter()
            .map(|&(column, coefficient)| Ok((map_column(isolated, call, column)?, coefficient)))
            .collect::<Result<Vec<_>, OutputAuthorityPoseidon2SboxManifestError>>()?,
    ))
}

fn map_columns<const N: usize>(
    columns: [usize; N],
    isolated: &PoseidonPermutationTraceEntry,
    call: &PoseidonPermutationTraceEntry,
) -> Result<[usize; N], OutputAuthorityPoseidon2SboxManifestError> {
    let mut mapped = [0usize; N];
    for (index, column) in columns.into_iter().enumerate() {
        mapped[index] = map_column(isolated, call, column)?;
    }
    Ok(mapped)
}

fn map_lc(
    lc: &Lc,
    isolated: &PoseidonPermutationTraceEntry,
    call: &PoseidonPermutationTraceEntry,
) -> Result<(Vec<(usize, F)>, F), OutputAuthorityPoseidon2SboxManifestError> {
    let terms = lc
        .terms
        .iter()
        .map(|&(column, coefficient)| Ok((map_column(isolated, call, column)?, coefficient)))
        .collect::<Result<Vec<_>, OutputAuthorityPoseidon2SboxManifestError>>()?;
    Ok((normalized_terms(terms), lc.constant))
}

fn canonical_lc(lc: &Lc) -> (Vec<(usize, F)>, F) {
    (normalized_terms(lc.terms.iter().copied()), lc.constant)
}

fn validate_permutation_rows(
    source: &R1csSnapshot,
    reference: &IsolatedPermutation,
    call: &PoseidonPermutationTraceEntry,
) -> Result<(), OutputAuthorityPoseidon2SboxManifestError> {
    let expected_outputs = map_columns(reference.permutation.output_columns, &reference.permutation, call)?;
    if call.source_rows.len() != reference.permutation.source_rows.len()
        || call.allocated_columns.len() != reference.permutation.allocated_columns.len()
        || call.output_columns != expected_outputs
    {
        return Err(invalid(
            "permutation",
            "call geometry does not rename the isolated program",
        ));
    }
    for local_row in reference.permutation.source_rows.clone() {
        let call_row = call.source_rows.start + local_row - reference.permutation.source_rows.start;
        let expected_a = map_row(reference.source.a_row(local_row), &reference.permutation, call)?;
        let expected_b = map_row(reference.source.b_row(local_row), &reference.permutation, call)?;
        let expected_c = map_row(reference.source.c_row(local_row), &reference.permutation, call)?;
        if source.a_row(call_row) != expected_a
            || source.b_row(call_row) != expected_b
            || source.c_row(call_row) != expected_c
        {
            return Err(invalid("permutation", format!("mapped row {call_row} drifted")));
        }
    }
    Ok(())
}

fn validate_sbox_trace(
    reference: &IsolatedPermutation,
    call: &PoseidonPermutationTraceEntry,
    actual: &[Sbox7TraceEntry],
) -> Result<(), OutputAuthorityPoseidon2SboxManifestError> {
    if actual.len() != reference.sboxes.len() {
        return Err(invalid("sbox trace", "permutation has an unexpected S-box census"));
    }
    for (local, actual) in reference.sboxes.iter().zip(actual) {
        let expected_rows = (call.source_rows.start + local.source_rows.start - reference.permutation.source_rows.start)
            ..(call.source_rows.start + local.source_rows.end - reference.permutation.source_rows.start);
        let expected_intermediate_columns =
            map_columns(local.intermediates.map(Var::col), &reference.permutation, call)?;
        let expected_intermediates = expected_intermediate_columns.map(Var::from_column_for_trace);
        let expected_output = map_column(&reference.permutation, call, local.output.col())?;
        if actual.source_rows != expected_rows
            || actual.intermediates != expected_intermediates
            || actual.output.col() != expected_output
            || canonical_lc(&actual.input) != map_lc(&local.input, &reference.permutation, call)?
        {
            return Err(invalid(
                "sbox trace",
                "S-box provenance does not rename the isolated program",
            ));
        }
    }
    Ok(())
}

fn collect_uses(source: &R1csSnapshot, by_column: &[usize], candidate_count: usize) -> Vec<Vec<MatrixUse>> {
    let mut uses = vec![Vec::new(); candidate_count];
    for row in 0..source.rows() {
        for (matrix, entries) in [
            (Matrix::A, source.a_row(row)),
            (Matrix::B, source.b_row(row)),
            (Matrix::C, source.c_row(row)),
        ] {
            for &(column, coefficient) in entries {
                let index = by_column.get(column).copied().unwrap_or(usize::MAX);
                if index != usize::MAX {
                    uses[index].push(MatrixUse {
                        row,
                        matrix,
                        coefficient,
                    });
                }
            }
        }
    }
    uses
}

fn validate_whole_matrix_uses(
    source: &R1csSnapshot,
    reference: &IsolatedPermutation,
    calls: &[PoseidonPermutationTraceEntry],
    candidate_columns: &[usize],
) -> Result<(), OutputAuthorityPoseidon2SboxManifestError> {
    let mut by_column = vec![usize::MAX; source.cols()];
    for (index, &column) in candidate_columns.iter().enumerate() {
        if column >= source.cols() || by_column[column] != usize::MAX {
            return Err(invalid("whole matrix", "candidate columns are not unique and in range"));
        }
        by_column[column] = index;
    }
    let actual = collect_uses(source, &by_column, candidate_columns.len());
    let mut expected = vec![Vec::new(); candidate_columns.len()];
    for (call_index, call) in calls.iter().enumerate() {
        for (sbox_index, local_uses) in reference.uses_by_sbox.iter().enumerate() {
            let index = call_index * SBOXES_PER_PERMUTATION + sbox_index;
            expected[index].extend(local_uses.iter().map(|entry| MatrixUse {
                row: call.source_rows.start + entry.row - reference.permutation.source_rows.start,
                matrix: entry.matrix,
                coefficient: entry.coefficient,
            }));
        }
    }
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        if actual != expected {
            return Err(invalid(
                "whole matrix",
                format!(
                    "candidate {} has unexpected, missing, duplicate, or coefficient-drifted uses",
                    candidate_columns[index]
                ),
            ));
        }
    }
    Ok(())
}

/// Validate and compactly describe every Poseidon2 S-box output owned by the
/// outgoing accumulator-authority hash.
pub fn audit_output_authority_poseidon2_sboxes(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    public_columns: &[usize],
) -> Result<OutputAuthorityPoseidon2SboxManifest, OutputAuthorityPoseidon2SboxManifestError> {
    let reference = isolated_permutation()?;
    let (stage_rows, stage_columns, claimed_digest_columns) = find_stage(trace)?;
    if stage_rows.end > source.rows() || stage_columns.end > source.cols() {
        return Err(invalid("stage", "stage range escapes the source relation"));
    }

    let overlapping_hashes = trace
        .poseidon_hashes()
        .iter()
        .enumerate()
        .filter_map(|(index, hash)| ranges_overlap(&hash.source_rows, &stage_rows).then_some(index))
        .collect::<Vec<_>>();
    if overlapping_hashes.len() != 1 {
        return Err(invalid(
            "hash",
            format!("expected one overlapping hash, found {}", overlapping_hashes.len()),
        ));
    }
    let hash_index = overlapping_hashes[0];
    let hash = &trace.poseidon_hashes()[hash_index];
    if hash.input_len != EXPECTED_HASH_INPUTS || hash.input_columns.len() != EXPECTED_HASH_INPUTS {
        return Err(invalid("hash", "authoritative hash input census drifted"));
    }
    if hash.permutation_range.len() != EXPECTED_PERMUTATIONS
        || hash.source_rows.start != stage_rows.start + EXPECTED_PREHASH_ROWS
    {
        return Err(invalid(
            "hash",
            format!(
                "expected {EXPECTED_PERMUTATIONS} permutations and row start {}, found {} and {}",
                stage_rows.start + EXPECTED_PREHASH_ROWS,
                hash.permutation_range.len(),
                hash.source_rows.start,
            ),
        ));
    }
    if hash
        .input_columns
        .iter()
        .any(|&column| column >= source.cols())
    {
        return Err(invalid("hash", "hash input column escapes the source relation"));
    }

    let semantic_state_output_columns = std::array::from_fn(|lane| stage_columns.end - DIGEST_LEN + lane);
    let prehash_rows = stage_rows.start..hash.source_rows.start;
    let prehash_columns = stage_columns.start..hash.zero_column;
    validate_prehash_bindings(source, prehash_rows.clone(), prehash_columns.clone())?;
    if hash.zero_column != stage_columns.start + EXPECTED_PREHASH_COLUMNS
        || hash
            .output_columns
            .iter()
            .any(|&column| column >= source.cols())
    {
        return Err(invalid("hash", "zero or output column geometry drifted"));
    }

    let selected_permutations = trace
        .poseidon_permutations()
        .get(hash.permutation_range.clone())
        .ok_or_else(|| invalid("hash", "permutation range escapes trace"))?;
    let overlapping_permutations = trace
        .poseidon_permutations()
        .iter()
        .enumerate()
        .filter_map(|(index, permutation)| ranges_overlap(&permutation.source_rows, &stage_rows).then_some(index))
        .collect::<Vec<_>>();
    if overlapping_permutations != hash.permutation_range.clone().collect::<Vec<_>>() {
        return Err(invalid(
            "permutation",
            "stage permutation ownership has gaps, extras, or escapes",
        ));
    }

    let overlapping_sboxes = trace
        .sbox7()
        .iter()
        .enumerate()
        .filter_map(|(index, sbox)| ranges_overlap(&sbox.source_rows, &stage_rows).then_some(index))
        .collect::<Vec<_>>();
    if overlapping_sboxes.len() != EXPECTED_SBOXES
        || overlapping_sboxes
            .windows(2)
            .any(|window| window[1] != window[0] + 1)
    {
        return Err(invalid(
            "sbox trace",
            "expected one contiguous 36,292-entry stage census",
        ));
    }
    let sbox_trace_range = overlapping_sboxes[0]..overlapping_sboxes[0] + overlapping_sboxes.len();

    let mut cursor_row = hash.source_rows.start;
    let mut cursor_column = hash.zero_column;
    expect_row(
        source,
        cursor_row,
        &affine_row([(hash.zero_column, F::ONE)]),
        "hash zero",
    )?;
    cursor_row += 1;
    cursor_column += 1;
    let mut state_columns = [hash.zero_column; WIDTH];
    let mut calls = Vec::with_capacity(EXPECTED_PERMUTATIONS);
    let mut full_absorb_rounds = 0usize;
    let mut partial_absorb_fields = 0usize;
    let mut stage_sbox_cursor = sbox_trace_range.start;

    for (round, permutation) in selected_permutations.iter().enumerate() {
        let is_pad = round + 1 == EXPECTED_PERMUTATIONS;
        if is_pad {
            let next = cursor_column;
            expect_row(
                source,
                cursor_row,
                &affine_row([(next, F::ONE), (state_columns[0], -F::ONE), (Var::ONE.col(), -F::ONE)]),
                "hash padding",
            )?;
            state_columns[0] = next;
            cursor_row += 1;
            cursor_column += 1;
        } else {
            let start = round * RATE;
            let remaining = EXPECTED_HASH_INPUTS - start;
            let chunk_len = remaining.min(RATE);
            if chunk_len == RATE {
                full_absorb_rounds += 1;
            } else {
                partial_absorb_fields += chunk_len;
            }
            for lane in 0..chunk_len {
                let next = cursor_column;
                expect_row(
                    source,
                    cursor_row,
                    &affine_row([
                        (next, F::ONE),
                        (state_columns[lane], -F::ONE),
                        (hash.input_columns[start + lane], -F::ONE),
                    ]),
                    "hash absorb",
                )?;
                state_columns[lane] = next;
                cursor_row += 1;
                cursor_column += 1;
            }
        }
        if permutation.input_columns != state_columns
            || permutation.source_rows.start != cursor_row
            || permutation.allocated_columns.start != cursor_column
            || permutation.source_rows.end > stage_rows.end
            || permutation.allocated_columns.end > stage_columns.end
        {
            return Err(invalid(
                "hash schedule",
                format!("permutation round {round} is disconnected"),
            ));
        }
        validate_permutation_rows(source, &reference, permutation)?;
        let next_sbox_cursor = stage_sbox_cursor + SBOXES_PER_PERMUTATION;
        validate_sbox_trace(
            &reference,
            permutation,
            trace
                .sbox7()
                .get(stage_sbox_cursor..next_sbox_cursor)
                .ok_or_else(|| invalid("sbox trace", "permutation S-box range escapes trace"))?,
        )?;
        calls.push(Poseidon2PermutationCall {
            trace_index: hash.permutation_range.start + round,
            source_rows: permutation.source_rows.clone(),
            input_columns: permutation.input_columns,
            first_allocated_column: permutation.allocated_columns.start,
            allocated_column_count: permutation.allocated_columns.len(),
            output_columns: permutation.output_columns,
        });
        cursor_row = permutation.source_rows.end;
        cursor_column = permutation.allocated_columns.end;
        state_columns = permutation.output_columns;
        stage_sbox_cursor = next_sbox_cursor;
    }
    if full_absorb_rounds != EXPECTED_FULL_ABSORBS
        || partial_absorb_fields != EXPECTED_PARTIAL_ABSORB_FIELDS
        || cursor_row != hash.source_rows.end
        || cursor_column != stage_columns.end - DIGEST_LEN
        || stage_sbox_cursor != sbox_trace_range.end
        || hash.output_columns != state_columns[..DIGEST_LEN]
    {
        return Err(invalid("hash schedule", "terminal sponge geometry or census drifted"));
    }

    if hash.source_rows.end + DIGEST_LEN != stage_rows.end {
        return Err(invalid(
            "digest binding",
            "computed digest is not followed by exactly four binding rows",
        ));
    }
    for lane in 0..DIGEST_LEN {
        expect_row(
            source,
            hash.source_rows.end + lane,
            &affine_row([
                (claimed_digest_columns[lane], F::ONE),
                (hash.output_columns[lane], -F::ONE),
            ]),
            "digest binding",
        )?;
    }

    let mut protected = BTreeSet::new();
    for &column in public_columns {
        if column >= source.cols() || !protected.insert(column) {
            return Err(invalid(
                "public boundary",
                "public columns are duplicated or out of range",
            ));
        }
    }
    protected.extend(hash.input_columns.iter().copied());
    protected.extend(hash.output_columns);
    protected.insert(hash.zero_column);
    protected.extend(claimed_digest_columns);
    protected.extend(semantic_state_output_columns);
    for permutation in selected_permutations {
        protected.extend(permutation.input_columns);
        protected.extend(permutation.output_columns);
    }

    let mut candidate_columns = Vec::with_capacity(EXPECTED_SBOXES);
    for permutation in selected_permutations {
        for &offset in &reference.output_offsets {
            let column = permutation.allocated_columns.start + offset;
            if protected.contains(&column) {
                return Err(invalid(
                    "authority alias",
                    format!("candidate column {column} aliases a protected boundary"),
                ));
            }
            candidate_columns.push(column);
        }
    }
    if candidate_columns.len() != EXPECTED_SBOXES {
        return Err(invalid("census", "candidate S-box output census drifted"));
    }
    validate_whole_matrix_uses(source, &reference, selected_permutations, &candidate_columns)?;

    let initial_external_sboxes = EXPECTED_PERMUTATIONS * EXPECTED_INITIAL_SBOXES_PER_PERMUTATION;
    let partial_sboxes = EXPECTED_PERMUTATIONS * EXPECTED_PARTIAL_SBOXES_PER_PERMUTATION;
    let terminal_external_sboxes = EXPECTED_PERMUTATIONS * EXPECTED_TERMINAL_SBOXES_PER_PERMUTATION;
    let definition_uses = EXPECTED_SBOXES * EXPECTED_C_USES_PER_SBOX_OUTPUT;
    let linear_consumer_uses = EXPECTED_SBOXES * EXPECTED_A_USES_PER_SBOX_OUTPUT;
    Ok(OutputAuthorityPoseidon2SboxManifest {
        stage_rows,
        stage_columns,
        prehash_rows,
        prehash_columns,
        hash_index,
        hash_rows: hash.source_rows.clone(),
        hash_input_columns: hash.input_columns.clone(),
        hash_output_columns: hash.output_columns,
        claimed_digest_columns,
        semantic_state_output_columns,
        permutation_trace_range: hash.permutation_range.clone(),
        sbox_trace_range,
        calls,
        census: OutputAuthorityPoseidon2SboxCensus {
            scanned_source_rows: source.rows(),
            scanned_source_columns: source.cols(),
            stage_rows: EXPECTED_STAGE_ROWS,
            stage_columns: EXPECTED_STAGE_COLUMNS,
            prehash_binding_rows: EXPECTED_PREHASH_ROWS,
            prehash_fresh_columns: EXPECTED_PREHASH_COLUMNS,
            hash_input_fields: EXPECTED_HASH_INPUTS,
            full_absorb_rounds,
            partial_absorb_fields,
            pad_rounds: 1,
            permutations: EXPECTED_PERMUTATIONS,
            initial_external_sboxes,
            partial_sboxes,
            terminal_external_sboxes,
            candidate_sbox_outputs: EXPECTED_SBOXES,
            definition_uses,
            linear_consumer_uses,
            total_matrix_uses: definition_uses + linear_consumer_uses,
        },
        family_layout: OutputAuthorityPoseidon2SboxFamilyLayout {
            initial_external: 0..EXPECTED_INITIAL_SBOXES_PER_PERMUTATION,
            partial: EXPECTED_INITIAL_SBOXES_PER_PERMUTATION
                ..EXPECTED_INITIAL_SBOXES_PER_PERMUTATION + EXPECTED_PARTIAL_SBOXES_PER_PERMUTATION,
            terminal_external: EXPECTED_INITIAL_SBOXES_PER_PERMUTATION + EXPECTED_PARTIAL_SBOXES_PER_PERMUTATION
                ..SBOXES_PER_PERMUTATION,
        },
        isolated_sbox_output_offsets: reference.output_offsets,
    })
}
