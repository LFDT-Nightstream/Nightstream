//! Diagnostic export from exact Nightstream R1CS rows.
//!
//! This crate reads the protocol crate. The protocol crate does not depend on
//! this tool. SHA-256 identifies diagnostic artifacts only; it is not a
//! protocol hash or an authority for a circuit change.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{self, Write as _};
use std::io;

use neo_ccs::CcsMatrix;
use neo_fold_clean::engine::r1cs_circuit::builder::RowFamilyRange;
use neo_fold_clean::engine::r1cs_circuit::{PhysicalStageRange, R1csSnapshot, Var};
use neo_fold_clean::frontends::r1cs_f_prime::ivc::{R1csIvcBranch, R1csIvcConstraintSourceAudit};
use neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use recursive_constraint_minimizer::{Problem, Row, Scope, Source, Term, GOLDILOCKS_MODULUS, PROBLEM_SCHEMA};
use sha2::{Digest, Sha256};

mod refinement;
mod selective_binding;

pub use refinement::{refine_with_cvc5, RefinementError, RefinementReport, MAX_REFINEMENT_ITERATIONS};
pub use selective_binding::{
    FixedPointProblemExport, SelectiveRetainedRowBinding, SelectiveRewriteBinding, SelectiveSliceBinding,
};

const DIGEST_DOMAIN: &[u8] = b"nightstream/r1cs-source-artifact/v1";
const SPARSE_DIGEST_DOMAIN: &[u8] = b"nightstream/sparse-r1cs-source-artifact/v2";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportRequest {
    pub profile: String,
    pub scope: Scope,
    /// Exclusive end of the normalized public-column prefix.
    pub public_input_count: usize,
    /// Strictly increasing row indices in the complete source relation.
    pub source_rows: Vec<usize>,
    /// Strictly ordered family names that are complete in `source_rows`.
    pub complete_families: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportError(String);

impl ExportError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ExportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ExportError {}

/// Rows owned by one physical-stage family in a sparse source relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseOwnedFamily {
    name: &'static str,
    source_rows: Vec<usize>,
}

impl SparseOwnedFamily {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn source_rows(&self) -> &[usize] {
        &self.source_rows
    }
}

/// Enumerate every nonempty physical-stage family in name order.
///
/// This generic function checks the exact row and private-column partition.
/// It does not validate the caller-supplied path vocabulary.
pub fn sparse_family_census(arm: &SparseR1cs) -> Result<Vec<SparseOwnedFamily>, ExportError> {
    arm.validate_shape()
        .map_err(|error| ExportError::new(format!("invalid sparse source shape: {error}")))?;
    validate_physical_stages(arm)?;
    let owners = stage_owners(arm.n, arm.physical_stage_ranges());
    let mut rows_by_family = BTreeMap::<&'static str, Vec<usize>>::new();
    for (row, owner) in owners.into_iter().enumerate() {
        let name = owner.ok_or_else(|| ExportError::new(format!("sparse source row {row} has no row-family owner")))?;
        rows_by_family.entry(name).or_default().push(row);
    }
    Ok(rows_by_family
        .into_iter()
        .map(|(name, source_rows)| SparseOwnedFamily { name, source_rows })
        .collect())
}

/// Enumerate a fixed-point arm after checking its complete reviewed stage
/// vocabulary.
pub fn fixed_point_family_census(
    audit: &R1csIvcConstraintSourceAudit,
    branch: R1csIvcBranch,
) -> Result<Vec<SparseOwnedFamily>, ExportError> {
    let arm = audit.arm(branch);
    validate_fixed_point_stage_vocabulary(arm, branch)?;
    sparse_family_census(arm)
}

/// Export selected rows and bind them to the complete source relation.
pub fn export_problem(
    snapshot: &R1csSnapshot,
    ranges: &[RowFamilyRange],
    request: ExportRequest,
) -> Result<Problem, ExportError> {
    validate_request(snapshot.rows(), &request)?;
    validate_public_input_count(snapshot.cols(), &request)?;
    let ranges = validate_ranges(snapshot.rows(), ranges)?;
    let mut hasher = Sha256::new();
    hasher.update(DIGEST_DOMAIN);
    hash_usize(&mut hasher, snapshot.rows())?;
    hash_usize(&mut hasher, snapshot.cols())?;
    hash_usize(&mut hasher, Var::ONE.col())?;
    hash_usize(&mut hasher, request.public_input_count)?;
    hash_bytes(&mut hasher, GOLDILOCKS_MODULUS.as_bytes())?;

    let mut exported_rows = Vec::with_capacity(request.source_rows.len());
    let mut total_by_family = BTreeMap::<&str, usize>::new();
    let mut selected_by_family = BTreeMap::<&str, usize>::new();
    let mut selected_cursor = 0usize;
    let mut range_cursor = 0usize;
    let mut active_ranges = Vec::<usize>::new();

    for source_index in 0..snapshot.rows() {
        while active_ranges
            .last()
            .is_some_and(|&range_index| ranges[range_index].row_end == source_index)
        {
            active_ranges.pop();
        }
        while range_cursor < ranges.len() && ranges[range_cursor].row_start == source_index {
            active_ranges.push(range_cursor);
            range_cursor += 1;
        }
        let owner = active_ranges
            .last()
            .map(|&range_index| ranges[range_index].name);
        hash_row(&mut hasher, snapshot, source_index, owner)?;

        if let Some(family) = owner {
            *total_by_family.entry(family).or_default() += 1;
        }
        if request.source_rows.get(selected_cursor) == Some(&source_index) {
            let family = owner.ok_or_else(|| {
                ExportError::new(format!("selected source row {source_index} has no row-family owner"))
            })?;
            *selected_by_family.entry(family).or_default() += 1;
            exported_rows.push(export_row(snapshot, source_index, family));
            selected_cursor += 1;
        }
    }
    debug_assert_eq!(selected_cursor, request.source_rows.len());

    for family in &request.complete_families {
        let total = total_by_family.get(family.as_str()).copied().unwrap_or(0);
        let selected = selected_by_family
            .get(family.as_str())
            .copied()
            .unwrap_or(0);
        if total == 0 {
            return Err(ExportError::new(format!(
                "complete family {family:?} owns no source rows"
            )));
        }
        if selected != total {
            return Err(ExportError::new(format!(
                "complete family {family:?} has {selected} of {total} source rows"
            )));
        }
    }

    let digest = hasher.finalize();
    let mut artifact_digest = String::with_capacity(7 + digest.len() * 2);
    artifact_digest.push_str("sha256:");
    for byte in digest {
        write!(artifact_digest, "{byte:02x}").expect("writing to String cannot fail");
    }
    let problem = Problem {
        schema: PROBLEM_SCHEMA.to_owned(),
        source: Source {
            profile: request.profile,
            artifact_digest,
            scope: request.scope,
            total_rows: snapshot.rows(),
        },
        field_modulus: GOLDILOCKS_MODULUS.to_owned(),
        column_count: snapshot.cols(),
        constant_one_column: Var::ONE.col(),
        public_input_count: request.public_input_count,
        complete_families: request.complete_families,
        rows: exported_rows,
    };
    problem
        .validate()
        .map_err(|error| ExportError::new(format!("exported problem is invalid: {error}")))?;
    Ok(problem)
}

/// Export selected rows from one exact stabilized field-R1CS arm.
pub fn export_sparse_problem(arm: &SparseR1cs, request: ExportRequest) -> Result<Problem, ExportError> {
    arm.validate_shape()
        .map_err(|error| ExportError::new(format!("invalid sparse source shape: {error}")))?;
    if arm.m_in == 0 || Var::ONE.col() >= arm.m_in {
        return Err(ExportError::new(
            "sparse source has no normalized constant-one public prefix",
        ));
    }
    if request.public_input_count != arm.m_in {
        return Err(ExportError::new(format!(
            "requested public prefix {} differs from sparse source prefix {}",
            request.public_input_count, arm.m_in
        )));
    }
    if [&arm.a, &arm.b, &arm.c]
        .into_iter()
        .any(|matrix| !matrix.has_canonical_csc())
    {
        return Err(ExportError::new(
            "sparse source matrices must have canonical materialized components",
        ));
    }
    validate_request(arm.n, &request)?;
    validate_physical_stages(arm)?;
    let owners = stage_owners(arm.n, arm.physical_stage_ranges());
    let recovered = recover_sparse_rows(arm, &request.source_rows)?;
    let mut hasher = Sha256::new();
    hasher.update(SPARSE_DIGEST_DOMAIN);
    hash_usize(&mut hasher, arm.n)?;
    hash_usize(&mut hasher, arm.m)?;
    hash_usize(&mut hasher, arm.m_in)?;
    hash_usize(&mut hasher, Var::ONE.col())?;
    hash_bytes(&mut hasher, GOLDILOCKS_MODULUS.as_bytes())?;
    hash_physical_stages(&mut hasher, arm.physical_stage_ranges())?;
    hash_sparse_matrix(&mut hasher, 0, &arm.a)?;
    hash_sparse_matrix(&mut hasher, 1, &arm.b)?;
    hash_sparse_matrix(&mut hasher, 2, &arm.c)?;

    let mut exported_rows = Vec::with_capacity(request.source_rows.len());
    let mut total_by_family = BTreeMap::<&str, usize>::new();
    let mut selected_by_family = BTreeMap::<&str, usize>::new();
    let mut selected_cursor = 0usize;
    for (source_index, owner) in owners.into_iter().enumerate() {
        hash_usize(&mut hasher, source_index)?;
        match owner {
            Some(name) => {
                hasher.update([1]);
                hash_bytes(&mut hasher, name.as_bytes())?;
                *total_by_family.entry(name).or_default() += 1;
            }
            None => hasher.update([0]),
        }

        if request.source_rows.get(selected_cursor) == Some(&source_index) {
            let family = owner.ok_or_else(|| {
                ExportError::new(format!("selected source row {source_index} has no row-family owner"))
            })?;
            *selected_by_family.entry(family).or_default() += 1;
            let [a, b, c] = &recovered[selected_cursor];
            exported_rows.push(Row {
                id: format!("r1cs.row.{source_index}"),
                source_index,
                family: family.to_owned(),
                a: export_sparse_terms(a),
                b: export_sparse_terms(b),
                c: export_sparse_terms(c),
            });
            selected_cursor += 1;
        }
    }
    debug_assert_eq!(selected_cursor, request.source_rows.len());

    validate_complete_families(&request.complete_families, &total_by_family, &selected_by_family)?;
    let artifact_digest = finish_digest(hasher);
    let problem = Problem {
        schema: PROBLEM_SCHEMA.to_owned(),
        source: Source {
            profile: request.profile,
            artifact_digest,
            scope: request.scope,
            total_rows: arm.n,
        },
        field_modulus: GOLDILOCKS_MODULUS.to_owned(),
        column_count: arm.m,
        constant_one_column: Var::ONE.col(),
        public_input_count: arm.m_in,
        complete_families: request.complete_families,
        rows: exported_rows,
    };
    problem
        .validate()
        .map_err(|error| ExportError::new(format!("exported problem is invalid: {error}")))?;
    Ok(problem)
}

/// Export one fixed-point arm after checking its complete reviewed stage
/// vocabulary.
pub fn export_fixed_point_problem(
    audit: &R1csIvcConstraintSourceAudit,
    branch: R1csIvcBranch,
    request: ExportRequest,
) -> Result<FixedPointProblemExport, ExportError> {
    let arm = audit.arm(branch);
    validate_fixed_point_stage_vocabulary(arm, branch)?;
    let problem = export_sparse_problem(arm, request)?;
    selective_binding::bind_fixed_point_problem(audit, branch, problem)
}

fn stage_owners(row_count: usize, stages: &[PhysicalStageRange]) -> Vec<Option<&'static str>> {
    let mut owners = vec![None; row_count];
    for stage in stages {
        for owner in &mut owners[stage.rows()] {
            debug_assert!(owner.is_none());
            *owner = Some(stage.path());
        }
    }
    owners
}

fn validate_physical_stages(arm: &SparseR1cs) -> Result<(), ExportError> {
    let stages = arm.physical_stage_ranges();
    if stages.is_empty() {
        return Err(ExportError::new("sparse source has no physical-stage schedule"));
    }
    let mut row_cursor = 0usize;
    let mut column_cursor = arm.m_in;
    for (occurrence, stage) in stages.iter().enumerate() {
        if stage.path().trim().is_empty() {
            return Err(ExportError::new(format!(
                "physical stage {occurrence} has an empty path"
            )));
        }
        if stage.row_start() != row_cursor || stage.row_end() < stage.row_start() {
            return Err(ExportError::new(format!(
                "physical stage {occurrence} does not continue the exact row partition"
            )));
        }
        if stage.column_start() != column_cursor || stage.column_end() < stage.column_start() {
            return Err(ExportError::new(format!(
                "physical stage {occurrence} does not continue the exact private-column partition"
            )));
        }
        row_cursor = stage.row_end();
        column_cursor = stage.column_end();
    }
    if row_cursor != arm.n || column_cursor != arm.m {
        return Err(ExportError::new(format!(
            "physical stages end at rows {row_cursor} and columns {column_cursor}; expected {} and {}",
            arm.n, arm.m
        )));
    }
    Ok(())
}

fn validate_fixed_point_stage_vocabulary(arm: &SparseR1cs, branch: R1csIvcBranch) -> Result<(), ExportError> {
    validate_physical_stages(arm)?;
    let expected = fixed_point_stage_vocabulary(branch);
    let actual = arm
        .physical_stage_ranges()
        .iter()
        .map(PhysicalStageRange::path)
        .collect::<BTreeSet<_>>();
    if actual != expected {
        let missing = expected.difference(&actual).copied().collect::<Vec<_>>();
        let unexpected = actual.difference(&expected).copied().collect::<Vec<_>>();
        return Err(ExportError::new(format!(
            "fixed-point {branch:?} stage vocabulary drifted; missing {missing:?}, unexpected {unexpected:?}"
        )));
    }
    Ok(())
}

fn fixed_point_stage_vocabulary(branch: R1csIvcBranch) -> BTreeSet<&'static str> {
    use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::pi_rlc_challenge_stage;
    use neo_fold_clean::paper::f_prime::stage as fprime_stage;
    use neo_fold_clean::paper::nifs::circuit::stage as nifs_stage;
    use neo_fold_clean::paper::reductions::pi_ccs_circuit::stage as pi_ccs_stage;
    use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage as pi_rlc_stage;

    match branch {
        R1csIvcBranch::Base => fprime_stage::BASE_ALL
            .iter()
            .copied()
            .filter(|path| {
                ![
                    fprime_stage::BASE_VERIFIER_KEY,
                    fprime_stage::BASE_FINALIZE,
                    fprime_stage::BASE_CONTEXT_LINK,
                ]
                .contains(path)
            })
            .collect(),
        R1csIvcBranch::BootstrapRecursive | R1csIvcBranch::Recursive => {
            let mut expected = fprime_stage::RECURSIVE_ALL
                .iter()
                .chain(nifs_stage::ALL)
                .chain(pi_ccs_stage::ALL)
                .chain(pi_rlc_stage::LIFECYCLE_ALL)
                .chain(pi_rlc_challenge_stage::ALL)
                .chain(pi_rlc_stage::ALL)
                .copied()
                .collect::<BTreeSet<_>>();
            for nonphysical in [
                fprime_stage::RECURSIVE_VERIFIER_KEY,
                fprime_stage::RECURSIVE_FINALIZE,
                fprime_stage::RECURSIVE_CONTEXT_LINK,
                pi_rlc_challenge_stage::CHUNK_ACCEPT_PACKED,
                pi_rlc_challenge_stage::ACCEPT_TREE_BIT_PAIRS,
                pi_rlc_challenge_stage::ACCEPT_PRODUCT_AGGREGATE,
                pi_rlc_challenge_stage::ACCEPT_ROOT_BINDING,
                pi_rlc_challenge_stage::CHUNK_MOD5_PACKED,
                pi_rlc_challenge_stage::LOW_BIT_PAIRS,
                pi_rlc_challenge_stage::HIGH_BIT_PAIR,
                pi_rlc_challenge_stage::RESIDUE_PAIR,
                pi_rlc_challenge_stage::SELECT_BIND,
                pi_rlc_challenge_stage::SELECT_BIND_ACCEPT,
                pi_rlc_challenge_stage::SELECT_BIND_PREFIX,
                pi_rlc_challenge_stage::SELECT_BIND_SYMBOL,
                pi_rlc_stage::PADDING,
            ] {
                expected.remove(nonphysical);
            }
            expected
        }
    }
}

fn validate_request(row_count: usize, request: &ExportRequest) -> Result<(), ExportError> {
    if row_count == 0 {
        return Err(ExportError::new("source snapshot has no rows"));
    }
    if request.profile.trim().is_empty() {
        return Err(ExportError::new("profile must not be empty"));
    }
    if request.source_rows.is_empty() {
        return Err(ExportError::new("source_rows must not be empty"));
    }
    let mut prior_row = None;
    for &row in &request.source_rows {
        if row >= row_count {
            return Err(ExportError::new(format!(
                "source row {row} is out of range for {row_count} rows",
            )));
        }
        if prior_row.is_some_and(|prior| row <= prior) {
            return Err(ExportError::new("source_rows must be strictly ordered and unique"));
        }
        prior_row = Some(row);
    }
    let mut prior_family: Option<&str> = None;
    for family in &request.complete_families {
        if family.trim().is_empty() {
            return Err(ExportError::new("complete_families contains an empty name"));
        }
        if prior_family.is_some_and(|prior| family.as_str() <= prior) {
            return Err(ExportError::new(
                "complete_families must be strictly ordered and unique",
            ));
        }
        prior_family = Some(family);
    }
    Ok(())
}

fn validate_public_input_count(column_count: usize, request: &ExportRequest) -> Result<(), ExportError> {
    if request.public_input_count == 0 || request.public_input_count > column_count {
        return Err(ExportError::new(format!(
            "public_input_count {} is out of range for {column_count} columns",
            request.public_input_count
        )));
    }
    if Var::ONE.col() >= request.public_input_count {
        return Err(ExportError::new(
            "constant-one column is outside the requested public prefix",
        ));
    }
    Ok(())
}

fn validate_complete_families<'a>(
    families: &[String],
    total_by_family: &BTreeMap<&'a str, usize>,
    selected_by_family: &BTreeMap<&'a str, usize>,
) -> Result<(), ExportError> {
    for family in families {
        let total = total_by_family.get(family.as_str()).copied().unwrap_or(0);
        let selected = selected_by_family
            .get(family.as_str())
            .copied()
            .unwrap_or(0);
        if total == 0 {
            return Err(ExportError::new(format!(
                "complete family {family:?} owns no source rows"
            )));
        }
        if selected != total {
            return Err(ExportError::new(format!(
                "complete family {family:?} has {selected} of {total} source rows"
            )));
        }
    }
    Ok(())
}

fn recover_sparse_rows(arm: &SparseR1cs, selected_rows: &[usize]) -> Result<Vec<[BTreeMap<usize, F>; 3]>, ExportError> {
    let mut positions = vec![usize::MAX; arm.n];
    for (position, &row) in selected_rows.iter().enumerate() {
        positions[row] = position;
    }
    let mut rows = (0..selected_rows.len())
        .map(|_| std::array::from_fn(|_| BTreeMap::new()))
        .collect::<Vec<[BTreeMap<usize, F>; 3]>>();
    for (port, matrix) in [&arm.a, &arm.b, &arm.c].into_iter().enumerate() {
        recover_sparse_port(matrix, selected_rows, &positions, &mut rows, port)?;
    }
    Ok(rows)
}

fn recover_sparse_port(
    matrix: &CcsMatrix<F>,
    selected_rows: &[usize],
    positions: &[usize],
    rows: &mut [[BTreeMap<usize, F>; 3]],
    port: usize,
) -> Result<(), ExportError> {
    match matrix {
        CcsMatrix::Identity { n } => {
            for &row in selected_rows {
                if row >= *n {
                    return Err(ExportError::new("selected row exceeds an identity matrix"));
                }
                accumulate_sparse_term(rows, positions[row], port, row, F::ONE);
            }
        }
        CcsMatrix::Csc(csc) | CcsMatrix::CscWithSeededPhi81 { csc, .. } => {
            for column in 0..csc.ncols {
                for entry in csc.column_range(column) {
                    let row = csc.row_index(entry);
                    let position = positions[row];
                    if position != usize::MAX {
                        accumulate_sparse_term(rows, position, port, column, csc.vals[entry]);
                    }
                }
            }
        }
        CcsMatrix::VerifierArtifact { .. } => {
            return Err(ExportError::new(
                "sparse source export requires materialized matrix content",
            ));
        }
    }
    if let CcsMatrix::CscWithSeededPhi81 {
        blocks, geometric_runs, ..
    } = matrix
    {
        for block in blocks {
            let start = selected_rows.partition_point(|&row| row < block.row_start());
            let end = selected_rows.partition_point(|&row| row < block.row_end());
            if end - start == block.row_end() - block.row_start() {
                block.for_each_term::<F, _>(|row, column, coefficient| {
                    accumulate_sparse_term(rows, positions[row], port, column, coefficient);
                });
            } else {
                for &row in &selected_rows[start..end] {
                    block.for_each_row_term::<F, _>(row, |column, coefficient| {
                        accumulate_sparse_term(rows, positions[row], port, column, coefficient);
                    });
                }
            }
        }
        for run in geometric_runs {
            let position = positions[run.row()];
            if position != usize::MAX {
                run.for_each_term(|_, column, coefficient| {
                    accumulate_sparse_term(rows, position, port, column, coefficient);
                });
            }
        }
    }
    Ok(())
}

fn accumulate_sparse_term(
    rows: &mut [[BTreeMap<usize, F>; 3]],
    position: usize,
    port: usize,
    column: usize,
    coefficient: F,
) {
    if coefficient != F::ZERO {
        *rows[position][port].entry(column).or_insert(F::ZERO) += coefficient;
    }
}

fn export_sparse_terms(terms: &BTreeMap<usize, F>) -> Vec<Term> {
    terms
        .iter()
        .filter(|(_, coefficient)| **coefficient != F::ZERO)
        .map(|(&column, coefficient)| Term {
            column,
            coefficient: coefficient.as_canonical_u64().to_string(),
        })
        .collect()
}

fn hash_sparse_matrix(hasher: &mut Sha256, port: u8, matrix: &CcsMatrix<F>) -> Result<(), ExportError> {
    hasher.update([port]);
    bincode::serialize_into(HashWriter(hasher), matrix)
        .map_err(|error| ExportError::new(format!("cannot hash sparse matrix: {error}")))
}

fn hash_physical_stages(hasher: &mut Sha256, stages: &[PhysicalStageRange]) -> Result<(), ExportError> {
    hash_usize(hasher, stages.len())?;
    for (occurrence, stage) in stages.iter().enumerate() {
        hash_usize(hasher, occurrence)?;
        hash_bytes(hasher, stage.path().as_bytes())?;
        hash_usize(hasher, stage.row_start())?;
        hash_usize(hasher, stage.row_end())?;
        hash_usize(hasher, stage.column_start())?;
        hash_usize(hasher, stage.column_end())?;
    }
    Ok(())
}

struct HashWriter<'a>(&'a mut Sha256);

impl io::Write for HashWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn finish_digest(hasher: Sha256) -> String {
    let digest = hasher.finalize();
    let mut artifact_digest = String::with_capacity(7 + digest.len() * 2);
    artifact_digest.push_str("sha256:");
    for byte in digest {
        write!(artifact_digest, "{byte:02x}").expect("writing to String cannot fail");
    }
    artifact_digest
}

fn validate_ranges(row_count: usize, input: &[RowFamilyRange]) -> Result<Vec<RowFamilyRange>, ExportError> {
    if input.is_empty() {
        return Err(ExportError::new("row-family ranges must not be empty"));
    }
    let mut ranges = input.to_vec();
    for range in &ranges {
        if range.name.trim().is_empty() {
            return Err(ExportError::new("row-family range has an empty name"));
        }
        if range.row_start > range.row_end {
            return Err(ExportError::new(format!(
                "row-family {:?} has a reversed range {}..{}",
                range.name, range.row_start, range.row_end
            )));
        }
        if range.row_end > row_count {
            return Err(ExportError::new(format!(
                "row-family {:?} ends at {}, after source row count {row_count}",
                range.name, range.row_end
            )));
        }
    }
    ranges.retain(|range| range.row_start < range.row_end);
    if ranges.is_empty() {
        return Err(ExportError::new("row-family ranges own no rows"));
    }
    ranges.sort_unstable_by(|left, right| {
        left.row_start
            .cmp(&right.row_start)
            .then_with(|| right.row_end.cmp(&left.row_end))
            .then_with(|| left.name.cmp(right.name))
    });

    let mut active = Vec::<usize>::new();
    for index in 0..ranges.len() {
        while active
            .last()
            .is_some_and(|&active_index| ranges[index].row_start >= ranges[active_index].row_end)
        {
            active.pop();
        }
        if let Some(&parent_index) = active.last() {
            let parent = ranges[parent_index];
            let current = ranges[index];
            if current.row_start == parent.row_start && current.row_end == parent.row_end {
                return Err(ExportError::new(format!(
                    "row-family ownership is ambiguous for range {}..{}",
                    current.row_start, current.row_end
                )));
            }
            if current.row_end > parent.row_end {
                return Err(ExportError::new(format!(
                    "row-family ranges {:?} and {:?} partially overlap",
                    parent.name, current.name
                )));
            }
        }
        active.push(index);
    }
    Ok(ranges)
}

fn export_row(snapshot: &R1csSnapshot, source_index: usize, family: &str) -> Row {
    Row {
        id: format!("r1cs.row.{source_index}"),
        source_index,
        family: family.to_owned(),
        a: export_terms(snapshot.a_row(source_index)),
        b: export_terms(snapshot.b_row(source_index)),
        c: export_terms(snapshot.c_row(source_index)),
    }
}

fn export_terms(terms: &[(usize, F)]) -> Vec<Term> {
    terms
        .iter()
        .map(|&(column, coefficient)| Term {
            column,
            coefficient: coefficient.as_canonical_u64().to_string(),
        })
        .collect()
}

fn hash_row(hasher: &mut Sha256, snapshot: &R1csSnapshot, row: usize, owner: Option<&str>) -> Result<(), ExportError> {
    hash_usize(hasher, row)?;
    hash_terms(hasher, snapshot.a_row(row))?;
    hash_terms(hasher, snapshot.b_row(row))?;
    hash_terms(hasher, snapshot.c_row(row))?;
    match owner {
        Some(name) => {
            hasher.update([1]);
            hash_bytes(hasher, name.as_bytes())?;
        }
        None => hasher.update([0]),
    }
    Ok(())
}

fn hash_terms(hasher: &mut Sha256, terms: &[(usize, F)]) -> Result<(), ExportError> {
    hash_usize(hasher, terms.len())?;
    for &(column, coefficient) in terms {
        hash_usize(hasher, column)?;
        hasher.update(coefficient.as_canonical_u64().to_le_bytes());
    }
    Ok(())
}

fn hash_bytes(hasher: &mut Sha256, bytes: &[u8]) -> Result<(), ExportError> {
    hash_usize(hasher, bytes.len())?;
    hasher.update(bytes);
    Ok(())
}

fn hash_usize(hasher: &mut Sha256, value: usize) -> Result<(), ExportError> {
    let value = u64::try_from(value).map_err(|_| ExportError::new("artifact dimension does not fit in u64"))?;
    hasher.update(value.to_le_bytes());
    Ok(())
}
