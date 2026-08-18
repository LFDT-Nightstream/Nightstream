//! Compact Rust-owned recipe data for the lifecycle verifier-key hashes.
//!
//! This module copies exact normalized columns and assigned constants before
//! shape-only synthesis discards its incomplete recursive assignment. The
//! emitted sparse rows remain the relation authority.

use std::collections::BTreeMap;
use std::ops::Range;

use neo_ccs::CcsMatrix;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::NebulaFPrimeRelationError;
use crate::engine::r1cs_circuit::builder::{
    Poseidon2HashAudit, Poseidon2HashRoundAuditKind, Poseidon2PermutationTrace,
};
use crate::engine::r1cs_circuit::{
    enforce_poseidon2_permutation, PoseidonPermutationTraceEntry, R1csBuilder, R1csSnapshot, Var,
};
use crate::frontends::r1cs_f_prime::SparseR1cs;
use crate::paper::f_prime::stage as fprime_stage;
use neo_math::F;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingVerifierKeyHashBlock {
    source_rows: Range<usize>,
    constant_values: Vec<u64>,
    constant_start_column: usize,
    local_columns: Vec<usize>,
    ordered_input_columns: Vec<usize>,
    output_columns: [usize; 4],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingVerifierKeyDigestBinding {
    source_rows: Range<usize>,
    left_columns: [usize; 4],
    right_columns: [usize; 4],
}

impl NebulaFPrimeStreamingVerifierKeyDigestBinding {
    pub fn source_rows(&self) -> Range<usize> {
        self.source_rows.clone()
    }

    pub fn left_columns(&self) -> &[usize; 4] {
        &self.left_columns
    }

    pub fn right_columns(&self) -> &[usize; 4] {
        &self.right_columns
    }
}

impl NebulaFPrimeStreamingVerifierKeyHashBlock {
    pub fn source_rows(&self) -> Range<usize> {
        self.source_rows.clone()
    }

    pub fn constant_values(&self) -> &[u64] {
        &self.constant_values
    }

    pub fn constant_start_column(&self) -> usize {
        self.constant_start_column
    }

    pub fn local_columns(&self) -> &[usize] {
        &self.local_columns
    }

    pub fn ordered_input_columns(&self) -> &[usize] {
        &self.ordered_input_columns
    }

    pub fn output_columns(&self) -> &[usize; 4] {
        &self.output_columns
    }

    #[doc(hidden)]
    pub fn apply_constant_value_test_mutation(&mut self, index: usize, value: u64) {
        self.constant_values[index] = value;
    }

    #[doc(hidden)]
    pub fn validate_source_rows_for_test(&self, source: &SparseR1cs) -> Result<(), NebulaFPrimeRelationError> {
        let hashes = source
            .poseidon2_hash_audits()
            .iter()
            .filter(|audit| {
                audit.row_end == self.source_rows.end
                    && audit.zero_row == self.source_rows.start + self.constant_values.len()
            })
            .collect::<Vec<_>>();
        let [audit] = hashes.as_slice() else {
            return Err(geometry("verifier-key hash block does not select one source hash"));
        };
        let rows = SourceRowWindow::new(source, self.source_rows.clone())?;
        validate_hash_block_rows(source, &rows, audit, self, &isolated_permutation()?)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingVerifierKeyHashRecipes {
    base_verifier_key: NebulaFPrimeStreamingVerifierKeyHashBlock,
    policy_verifier_key: NebulaFPrimeStreamingVerifierKeyHashBlock,
    policy_digest_binding: NebulaFPrimeStreamingVerifierKeyDigestBinding,
    initial_boundary: NebulaFPrimeStreamingVerifierKeyHashBlock,
    initial_boundary_binding: NebulaFPrimeStreamingVerifierKeyDigestBinding,
    public_trace_binding: NebulaFPrimeStreamingVerifierKeyDigestBinding,
}

impl NebulaFPrimeStreamingVerifierKeyHashRecipes {
    pub fn base_verifier_key(&self) -> &NebulaFPrimeStreamingVerifierKeyHashBlock {
        &self.base_verifier_key
    }

    pub fn policy_verifier_key(&self) -> &NebulaFPrimeStreamingVerifierKeyHashBlock {
        &self.policy_verifier_key
    }

    pub fn policy_digest_binding(&self) -> &NebulaFPrimeStreamingVerifierKeyDigestBinding {
        &self.policy_digest_binding
    }

    pub fn initial_boundary(&self) -> &NebulaFPrimeStreamingVerifierKeyHashBlock {
        &self.initial_boundary
    }

    pub fn initial_boundary_binding(&self) -> &NebulaFPrimeStreamingVerifierKeyDigestBinding {
        &self.initial_boundary_binding
    }

    pub fn public_trace_binding(&self) -> &NebulaFPrimeStreamingVerifierKeyDigestBinding {
        &self.public_trace_binding
    }
}

pub(super) fn extract_recursive_verifier_key_hash_recipes(
    source: &SparseR1cs,
    assignment: &[F],
) -> Result<NebulaFPrimeStreamingVerifierKeyHashRecipes, NebulaFPrimeRelationError> {
    let stages = source
        .physical_stage_ranges()
        .iter()
        .filter(|stage| stage.path() == fprime_stage::RECURSIVE_VERIFIER_KEY)
        .collect::<Vec<_>>();
    let [stage] = stages.as_slice() else {
        return Err(geometry(
            "recursive lifecycle arm must contain exactly one verifier-key stage",
        ));
    };
    let stage_rows = stage.rows();
    let hashes = source
        .poseidon2_hash_audits()
        .iter()
        .filter(|audit| stage_rows.start <= audit.zero_row && audit.row_end <= stage_rows.end)
        .collect::<Vec<_>>();
    let [base_hash, policy_hash, initial_boundary_hash] = hashes.as_slice() else {
        return Err(geometry(
            "recursive verifier-key stage must contain exactly three Poseidon2 hashes",
        ));
    };
    if [
        base_hash.rounds.len(),
        policy_hash.rounds.len(),
        initial_boundary_hash.rounds.len(),
    ] != [11, 5, 4]
    {
        return Err(geometry("recursive verifier-key Poseidon2 round counts differ"));
    }
    let isolated = isolated_permutation()?;
    let source_rows = SourceRowWindow::new(source, stage_rows.clone())?;

    let build = |audit: &Poseidon2HashAudit,
                 constant_count: usize|
     -> Result<NebulaFPrimeStreamingVerifierKeyHashBlock, NebulaFPrimeRelationError> {
        let constant_start_column = audit
            .zero_col
            .checked_sub(constant_count)
            .ok_or_else(|| geometry("verifier-key hash constant columns underflow"))?;
        let source_start = audit
            .zero_row
            .checked_sub(constant_count)
            .ok_or_else(|| geometry("verifier-key hash constant rows underflow"))?;
        let constant_columns = constant_start_column..audit.zero_col;
        if !constant_columns
            .clone()
            .all(|column| audit.input_cols.contains(&column))
        {
            return Err(geometry("verifier-key hash input omits an assigned constant"));
        }
        let constant_values = constant_columns
            .clone()
            .map(|column| {
                assignment
                    .get(column)
                    .copied()
                    .map(|value| value.as_canonical_u64())
                    .ok_or_else(|| geometry("verifier-key hash constant column escapes the assignment"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let local_columns = audit
            .input_cols
            .iter()
            .copied()
            .filter(|column| !constant_columns.contains(column))
            .collect();
        let block = NebulaFPrimeStreamingVerifierKeyHashBlock {
            source_rows: source_start..audit.row_end,
            constant_values,
            constant_start_column,
            local_columns,
            ordered_input_columns: audit.input_cols.clone(),
            output_columns: audit.output_cols,
        };
        validate_hash_block_rows(source, &source_rows, audit, &block, &isolated)?;
        Ok(block)
    };

    let base_verifier_key = build(base_hash, 21)?;
    let policy_verifier_key = build(policy_hash, 9)?;
    let initial_boundary = build(initial_boundary_hash, 8)?;
    if base_verifier_key.source_rows.start != stage_rows.start
        || base_verifier_key.source_rows.end != policy_verifier_key.source_rows.start
        || policy_verifier_key.source_rows.end + 4 != initial_boundary.source_rows.start
        || initial_boundary.source_rows.end + 8 != stage_rows.end
    {
        return Err(geometry("recursive verifier-key hash block geometry differs"));
    }
    let policy_digest_binding = extract_digest_binding(
        &source_rows,
        policy_verifier_key.source_rows.end..initial_boundary.source_rows.start,
        Some(policy_verifier_key.output_columns),
        "recursive verifier-key policy digest binding",
    )?;
    let initial_boundary_binding = extract_digest_binding(
        &source_rows,
        initial_boundary.source_rows.end..initial_boundary.source_rows.end + 4,
        Some(initial_boundary.output_columns),
        "recursive verifier-key initial-boundary binding",
    )?;
    let public_trace_binding = extract_digest_binding(
        &source_rows,
        initial_boundary.source_rows.end + 4..stage_rows.end,
        None,
        "recursive verifier-key public-trace binding",
    )?;
    Ok(NebulaFPrimeStreamingVerifierKeyHashRecipes {
        base_verifier_key,
        policy_verifier_key,
        policy_digest_binding,
        initial_boundary,
        initial_boundary_binding,
        public_trace_binding,
    })
}

struct IsolatedPermutation {
    source: R1csSnapshot,
    trace: PoseidonPermutationTraceEntry,
}

type Term = (usize, F);

struct SourceRowWindow {
    start: usize,
    a: Vec<Vec<Term>>,
    b: Vec<Vec<Term>>,
    c: Vec<Vec<Term>>,
}

impl SourceRowWindow {
    fn new(source: &SparseR1cs, rows: Range<usize>) -> Result<Self, NebulaFPrimeRelationError> {
        if rows.start > rows.end || rows.end > source.n {
            return Err(geometry("verifier-key source row window escapes the relation"));
        }
        let length = rows.len();
        Ok(Self {
            start: rows.start,
            a: materialize_row_window(&source.a, rows.start, length)?,
            b: materialize_row_window(&source.b, rows.start, length)?,
            c: materialize_row_window(&source.c, rows.start, length)?,
        })
    }

    fn row(&self, row: usize) -> Option<(&[Term], &[Term], &[Term])> {
        let offset = row.checked_sub(self.start)?;
        Some((self.a.get(offset)?, self.b.get(offset)?, self.c.get(offset)?))
    }
}

fn materialize_row_window(
    matrix: &CcsMatrix<F>,
    start: usize,
    length: usize,
) -> Result<Vec<Vec<Term>>, NebulaFPrimeRelationError> {
    let stop = start
        .checked_add(length)
        .ok_or_else(|| geometry("verifier-key row window overflows"))?;
    let mut rows = vec![Vec::new(); length];
    match matrix {
        CcsMatrix::Identity { n } => {
            if stop > *n {
                return Err(geometry("verifier-key row window escapes an identity matrix"));
            }
            for (offset, row) in rows.iter_mut().enumerate() {
                row.push((start + offset, F::ONE));
            }
        }
        CcsMatrix::Csc(csc) => materialize_csc_window(csc, start, stop, &mut rows)?,
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            if blocks
                .iter()
                .any(|block| block.row_start() < stop && start < block.row_end())
                || geometric_runs
                    .iter()
                    .any(|run| (start..stop).contains(&run.row()))
            {
                return Err(geometry("verifier-key row window overlaps a compact matrix component"));
            }
            materialize_csc_window(csc, start, stop, &mut rows)?;
        }
        CcsMatrix::VerifierArtifact { .. } => {
            return Err(geometry("verifier-key source matrix content is unavailable"));
        }
    }
    Ok(rows)
}

fn materialize_csc_window(
    csc: &neo_ccs::CscMat<F>,
    start: usize,
    stop: usize,
    rows: &mut [Vec<Term>],
) -> Result<(), NebulaFPrimeRelationError> {
    if stop > csc.nrows || !csc.is_canonical() {
        return Err(geometry("verifier-key source CSC is noncanonical or too short"));
    }
    for column in 0..csc.ncols {
        for entry in csc.column_range(column) {
            let row = csc.row_index(entry);
            if (start..stop).contains(&row) {
                rows[row - start].push((column, csc.vals[entry]));
            }
        }
    }
    Ok(())
}

fn isolated_permutation() -> Result<IsolatedPermutation, NebulaFPrimeRelationError> {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    let inputs: [Var; 8] = std::array::from_fn(|lane| builder.alloc(F::from_u64(lane as u64 + 1)));
    enforce_poseidon2_permutation(&mut builder, &inputs);
    let source = builder.snapshot();
    let traces = builder.encoding_trace().poseidon_permutations();
    let [trace] = traces else {
        return Err(geometry("isolated Poseidon2 emitter did not produce one permutation"));
    };
    if trace.source_rows != (0..source.rows()) || trace.allocated_columns.len() != 600 {
        return Err(geometry("isolated Poseidon2 permutation geometry differs"));
    }
    Ok(IsolatedPermutation {
        source,
        trace: trace.clone(),
    })
}

fn validate_hash_block_rows(
    source: &SparseR1cs,
    rows: &SourceRowWindow,
    audit: &Poseidon2HashAudit,
    block: &NebulaFPrimeStreamingVerifierKeyHashBlock,
    isolated: &IsolatedPermutation,
) -> Result<(), NebulaFPrimeRelationError> {
    if audit.row_start != audit.zero_row
        || block.source_rows.start + block.constant_values.len() != audit.zero_row
        || block.constant_start_column + block.constant_values.len() != audit.zero_col
        || block.source_rows.end != audit.row_end
    {
        return Err(geometry("verifier-key hash constant and zero-row geometry differs"));
    }
    for (offset, &value) in block.constant_values.iter().enumerate() {
        expect_affine_row(
            rows,
            block.source_rows.start + offset,
            [
                (block.constant_start_column + offset, F::ONE),
                (Var::ONE.col(), -F::from_u64(value)),
            ],
            "verifier-key constant",
        )?;
    }
    expect_affine_row(
        rows,
        audit.zero_row,
        [(audit.zero_col, F::ONE)],
        "verifier-key hash zero",
    )?;

    let permutations = source
        .poseidon2_traces()
        .iter()
        .filter(|trace| audit.row_start <= trace.row_start && trace.row_end <= audit.row_end)
        .collect::<Vec<_>>();
    if permutations.len() != audit.rounds.len() {
        return Err(geometry("verifier-key hash permutation census differs"));
    }

    let mut state = [audit.zero_col; 8];
    let mut input_cursor = 0usize;
    let mut row_cursor = audit.zero_row + 1;
    for (round_index, (round, permutation)) in audit.rounds.iter().zip(permutations).enumerate() {
        if round.state_before_cols != state {
            return Err(geometry(format!(
                "verifier-key hash round {round_index} does not continue its state"
            )));
        }
        let defining_count = match &round.kind {
            Poseidon2HashRoundAuditKind::Absorb { chunk_cols } => {
                if chunk_cols.is_empty()
                    || chunk_cols.len() > 4
                    || input_cursor + chunk_cols.len() > audit.input_cols.len()
                    || chunk_cols.as_slice() != &audit.input_cols[input_cursor..input_cursor + chunk_cols.len()]
                {
                    return Err(geometry(format!(
                        "verifier-key hash absorb {round_index} input order differs"
                    )));
                }
                for (lane, &input) in chunk_cols.iter().enumerate() {
                    expect_affine_row(
                        rows,
                        row_cursor + lane,
                        [
                            (round.permutation_input_cols[lane], F::ONE),
                            (state[lane], -F::ONE),
                            (input, -F::ONE),
                        ],
                        "verifier-key hash absorb",
                    )?;
                }
                if round.permutation_input_cols[chunk_cols.len()..] != state[chunk_cols.len()..] {
                    return Err(geometry(format!(
                        "verifier-key hash absorb {round_index} changes an untouched lane"
                    )));
                }
                input_cursor += chunk_cols.len();
                chunk_cols.len()
            }
            Poseidon2HashRoundAuditKind::Pad => {
                if input_cursor != audit.input_cols.len() || round.permutation_input_cols[1..] != state[1..] {
                    return Err(geometry(format!(
                        "verifier-key hash padding round {round_index} is disconnected"
                    )));
                }
                expect_affine_row(
                    rows,
                    row_cursor,
                    [
                        (round.permutation_input_cols[0], F::ONE),
                        (state[0], -F::ONE),
                        (Var::ONE.col(), -F::ONE),
                    ],
                    "verifier-key hash padding",
                )?;
                1
            }
        };
        if round.defining_rows != (row_cursor..row_cursor + defining_count).collect::<Vec<_>>()
            || permutation.row_start != row_cursor + defining_count
            || permutation.input_cols != round.permutation_input_cols
            || permutation.output_cols != round.permutation_output_cols
        {
            return Err(geometry(format!(
                "verifier-key hash round {round_index} row ownership differs"
            )));
        }
        validate_permutation_rows(rows, isolated, permutation)?;
        row_cursor = permutation.row_end;
        state = permutation.output_cols;
    }
    if input_cursor != audit.input_cols.len() || row_cursor != audit.row_end || audit.output_cols != state[..4] {
        return Err(geometry("verifier-key hash rows do not close exactly"));
    }
    Ok(())
}

fn validate_permutation_rows(
    rows: &SourceRowWindow,
    isolated: &IsolatedPermutation,
    call: &Poseidon2PermutationTrace,
) -> Result<(), NebulaFPrimeRelationError> {
    if call.row_end - call.row_start != isolated.trace.source_rows.len()
        || call.allocated_columns.len() != isolated.trace.allocated_columns.len()
    {
        return Err(geometry("mapped Poseidon2 permutation geometry differs"));
    }
    let expected_outputs = map_columns(isolated.trace.output_columns, &isolated.trace, call)?;
    if call.output_cols != expected_outputs {
        return Err(geometry("mapped Poseidon2 permutation outputs differ"));
    }
    for local_row in isolated.trace.source_rows.clone() {
        let source_row = call.row_start + local_row - isolated.trace.source_rows.start;
        expect_row(
            rows,
            source_row,
            &map_row(isolated.source.a_row(local_row), &isolated.trace, call)?,
            &map_row(isolated.source.b_row(local_row), &isolated.trace, call)?,
            &map_row(isolated.source.c_row(local_row), &isolated.trace, call)?,
            "mapped Poseidon2 permutation",
        )?;
    }
    Ok(())
}

fn map_columns<const N: usize>(
    columns: [usize; N],
    isolated: &PoseidonPermutationTraceEntry,
    call: &Poseidon2PermutationTrace,
) -> Result<[usize; N], NebulaFPrimeRelationError> {
    let mut mapped = [0usize; N];
    for (index, column) in columns.into_iter().enumerate() {
        mapped[index] = map_column(column, isolated, call)?;
    }
    Ok(mapped)
}

fn map_row(
    row: &[(usize, F)],
    isolated: &PoseidonPermutationTraceEntry,
    call: &Poseidon2PermutationTrace,
) -> Result<Vec<(usize, F)>, NebulaFPrimeRelationError> {
    let terms = row
        .iter()
        .map(|&(column, coefficient)| Ok((map_column(column, isolated, call)?, coefficient)))
        .collect::<Result<Vec<_>, NebulaFPrimeRelationError>>()?;
    Ok(normalized_terms(terms))
}

fn map_column(
    column: usize,
    isolated: &PoseidonPermutationTraceEntry,
    call: &Poseidon2PermutationTrace,
) -> Result<usize, NebulaFPrimeRelationError> {
    if column == Var::ONE.col() {
        return Ok(column);
    }
    if let Some(lane) = isolated
        .input_columns
        .iter()
        .position(|&input| input == column)
    {
        return Ok(call.input_cols[lane]);
    }
    if isolated.allocated_columns.contains(&column) {
        return call
            .allocated_columns
            .get(column - isolated.allocated_columns.start)
            .copied()
            .ok_or_else(|| geometry("mapped Poseidon2 column escapes the call"));
    }
    Err(geometry("isolated Poseidon2 column has no call-site mapping"))
}

fn expect_affine_row<const N: usize>(
    rows: &SourceRowWindow,
    row: usize,
    terms: [(usize, F); N],
    scope: &'static str,
) -> Result<(), NebulaFPrimeRelationError> {
    expect_row(
        rows,
        row,
        &normalized_terms(terms),
        &vec![(Var::ONE.col(), F::ONE)],
        &Vec::new(),
        scope,
    )
}

fn extract_digest_binding(
    rows: &SourceRowWindow,
    source_rows: Range<usize>,
    expected_right_columns: Option<[usize; 4]>,
    scope: &'static str,
) -> Result<NebulaFPrimeStreamingVerifierKeyDigestBinding, NebulaFPrimeRelationError> {
    if source_rows.len() != 4 {
        return Err(geometry(format!("{scope} does not contain four rows")));
    }
    let mut left_columns = [0usize; 4];
    let mut right_columns = [0usize; 4];
    for lane in 0..4 {
        let row = source_rows.start + lane;
        let Some((a, b, c)) = rows.row(row) else {
            return Err(geometry(format!("{scope} row {row} escapes the captured window")));
        };
        if b != [(Var::ONE.col(), F::ONE)] || !c.is_empty() || a.len() != 2 {
            return Err(geometry(format!("{scope} row {row} is not an affine equality")));
        }
        let left = a
            .iter()
            .find_map(|&(column, coefficient)| (coefficient == F::ONE).then_some(column))
            .ok_or_else(|| geometry(format!("{scope} row {row} has no left column")))?;
        let right = a
            .iter()
            .find_map(|&(column, coefficient)| (coefficient == -F::ONE).then_some(column))
            .ok_or_else(|| geometry(format!("{scope} row {row} has no right column")))?;
        if left == right {
            return Err(geometry(format!("{scope} row {row} aliases both sides")));
        }
        expect_affine_row(rows, row, [(left, F::ONE), (right, -F::ONE)], scope)?;
        left_columns[lane] = left;
        right_columns[lane] = right;
    }
    if expected_right_columns.is_some_and(|expected| expected != right_columns) {
        return Err(geometry(format!("{scope} does not consume the computed digest")));
    }
    Ok(NebulaFPrimeStreamingVerifierKeyDigestBinding {
        source_rows,
        left_columns,
        right_columns,
    })
}

fn expect_row(
    rows: &SourceRowWindow,
    row: usize,
    expected_a: &[(usize, F)],
    expected_b: &[(usize, F)],
    expected_c: &[(usize, F)],
    scope: &'static str,
) -> Result<(), NebulaFPrimeRelationError> {
    let Some((actual_a, actual_b, actual_c)) = rows.row(row) else {
        return Err(geometry(format!("{scope} row {row} escapes the captured window")));
    };
    if actual_a != expected_a || actual_b != expected_b || actual_c != expected_c {
        return Err(geometry(format!("{scope} row {row} differs from the source matrix")));
    }
    Ok(())
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

fn geometry(message: impl Into<String>) -> NebulaFPrimeRelationError {
    NebulaFPrimeRelationError::Geometry(message.into())
}
