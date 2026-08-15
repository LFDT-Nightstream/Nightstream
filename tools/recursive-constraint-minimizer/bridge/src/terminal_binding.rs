//! Exact terminal source-family binding to the padded Spartan relation.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::ops::Range;

use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::{
    TerminalR1csConstraintAudit, TERMINAL_CONTEXT_GUARD_NAMES, TERMINAL_PROOF_GUARD_NAMES, TERMINAL_R1CS_FAMILY_NAMES,
    TERMINAL_STATEMENT_GUARD_NAMES,
};
use recursive_constraint_minimizer::{Problem, Row, Term};
use sha2::{Digest, Sha256};

use super::{export_problem, hash_bytes, hash_usize, ExportError, ExportRequest};

const TERMINAL_BINDING_DIGEST_DOMAIN: &[u8] = b"nightstream/terminal-spartan-binding/v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalOwnedFamily {
    name: &'static str,
    source_rows: Vec<usize>,
}

impl TerminalOwnedFamily {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn source_rows(&self) -> &[usize] {
        &self.source_rows
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalProjectedRowArtifact {
    source_row: usize,
    spartan_row: usize,
    a: Vec<Term>,
    b: Vec<Term>,
    c: Vec<Term>,
}

impl TerminalProjectedRowArtifact {
    pub fn source_row(&self) -> usize {
        self.source_row
    }

    pub fn spartan_row(&self) -> usize {
        self.spartan_row
    }

    pub fn a(&self) -> &[Term] {
        &self.a
    }

    pub fn b(&self) -> &[Term] {
        &self.b
    }

    pub fn c(&self) -> &[Term] {
        &self.c
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalColumnLayout {
    source_public_columns: usize,
    source_private_columns: usize,
    spartan_private_columns: usize,
}

impl TerminalColumnLayout {
    pub fn new(
        source_public_columns: usize,
        source_private_columns: usize,
        spartan_private_columns: usize,
    ) -> Result<Self, ExportError> {
        if source_public_columns == 0 || spartan_private_columns < source_private_columns {
            return Err(ExportError::new("invalid terminal column layout"));
        }
        Ok(Self {
            source_public_columns,
            source_private_columns,
            spartan_private_columns,
        })
    }

    pub fn source_public_columns(&self) -> usize {
        self.source_public_columns
    }

    pub fn source_private_columns(&self) -> usize {
        self.source_private_columns
    }

    pub fn spartan_private_columns(&self) -> usize {
        self.spartan_private_columns
    }

    pub fn source_to_spartan_column(&self, source_column: usize) -> Option<usize> {
        if source_column == 0 {
            Some(self.spartan_private_columns)
        } else if source_column < self.source_public_columns {
            Some(self.spartan_private_columns + source_column)
        } else if source_column < self.source_public_columns + self.source_private_columns {
            Some(source_column - self.source_public_columns)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalSpartanBinding {
    requested_source_rows: Vec<usize>,
    verifier_native_guards: Vec<&'static str>,
    column_layout: TerminalColumnLayout,
    projected_rows: Vec<TerminalProjectedRowArtifact>,
    spartan_rows: usize,
    spartan_columns: usize,
    spartan_padding_rows: Range<usize>,
    spartan_private_padding_columns: usize,
    diagnostic_digest: String,
}

impl TerminalSpartanBinding {
    pub fn requested_source_rows(&self) -> &[usize] {
        &self.requested_source_rows
    }

    /// Verifier acceptance guards that are retained outside the polynomial
    /// R1CS family plan and must never enter a cvc5 removal query.
    pub fn verifier_native_guards(&self) -> &[&'static str] {
        &self.verifier_native_guards
    }

    pub fn column_layout(&self) -> &TerminalColumnLayout {
        &self.column_layout
    }

    pub fn projected_rows(&self) -> &[TerminalProjectedRowArtifact] {
        &self.projected_rows
    }

    pub fn spartan_rows(&self) -> usize {
        self.spartan_rows
    }

    pub fn spartan_columns(&self) -> usize {
        self.spartan_columns
    }

    pub fn spartan_padding_rows(&self) -> Range<usize> {
        self.spartan_padding_rows.clone()
    }

    pub fn spartan_private_padding_columns(&self) -> usize {
        self.spartan_private_padding_columns
    }

    /// SHA-256 is diagnostic here. Exact source rows and maps are authority.
    pub fn diagnostic_digest(&self) -> &str {
        &self.diagnostic_digest
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalProblemExport {
    problem: Problem,
    binding: TerminalSpartanBinding,
}

impl TerminalProblemExport {
    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    pub fn binding(&self) -> &TerminalSpartanBinding {
        &self.binding
    }

    pub fn into_problem(self) -> Problem {
        self.problem
    }
}

pub fn terminal_family_census(audit: &TerminalR1csConstraintAudit) -> Result<Vec<TerminalOwnedFamily>, ExportError> {
    validate_terminal_audit(audit)?;
    let mut rows_by_family = BTreeMap::<&'static str, Vec<usize>>::new();
    for range in audit.row_families() {
        rows_by_family
            .entry(range.name)
            .or_default()
            .extend(range.row_start..range.row_end);
    }
    Ok(rows_by_family
        .into_iter()
        .map(|(name, source_rows)| TerminalOwnedFamily { name, source_rows })
        .collect())
}

pub fn terminal_verifier_native_guard_names() -> Vec<&'static str> {
    TERMINAL_CONTEXT_GUARD_NAMES
        .into_iter()
        .chain(TERMINAL_STATEMENT_GUARD_NAMES)
        .chain(TERMINAL_PROOF_GUARD_NAMES)
        .collect()
}

pub fn export_terminal_problem(
    audit: &TerminalR1csConstraintAudit,
    request: ExportRequest,
) -> Result<TerminalProblemExport, ExportError> {
    validate_terminal_audit(audit)?;
    if request.public_input_count != audit.source_public_columns() {
        return Err(ExportError::new(
            "terminal export public prefix differs from the compiler audit",
        ));
    }
    let problem = export_problem(audit.source(), audit.row_families(), request)?;
    bind_terminal_problem(audit, problem)
}

/// Export every terminal source row and polynomial family.
///
/// This complete artifact is the Lean authority for terminal removal
/// counterexamples. Native terminal guards remain in the separate retained
/// guard ledger.
pub fn export_complete_terminal_problem(
    audit: &TerminalR1csConstraintAudit,
    profile: &str,
) -> Result<TerminalProblemExport, ExportError> {
    let families = terminal_family_census(audit)?;
    export_terminal_problem(
        audit,
        ExportRequest {
            profile: profile.to_owned(),
            scope: recursive_constraint_minimizer::Scope::Branch,
            public_input_count: audit.source_public_columns(),
            source_rows: (0..audit.source().rows()).collect(),
            complete_families: families
                .into_iter()
                .map(|family| family.name().to_owned())
                .collect(),
        },
    )
}

pub(crate) fn bind_terminal_problem(
    audit: &TerminalR1csConstraintAudit,
    problem: Problem,
) -> Result<TerminalProblemExport, ExportError> {
    if problem.source.total_rows != audit.source().rows()
        || problem.column_count != audit.source().cols()
        || problem.constant_one_column != 0
        || problem.public_input_count != audit.source_public_columns()
    {
        return Err(ExportError::new(
            "terminal problem geometry differs from its source audit",
        ));
    }
    let requested_source_rows = problem
        .rows
        .iter()
        .map(|row| row.source_index)
        .collect::<Vec<_>>();
    if requested_source_rows.is_empty() {
        return Err(ExportError::new("terminal problem has no source rows"));
    }
    let column_layout = TerminalColumnLayout::new(
        audit.source_public_columns(),
        audit.source_private_columns(),
        audit.spartan_private_columns(),
    )?;
    let projected_rows = problem
        .rows
        .iter()
        .map(|row| project_row(row, &column_layout))
        .collect::<Result<Vec<_>, _>>()?;
    let spartan_private_padding_columns = audit
        .spartan_columns()
        .checked_sub(audit.source().cols())
        .ok_or_else(|| ExportError::new("terminal Spartan relation is narrower than its source"))?;
    let diagnostic_digest = binding_digest(&problem, audit, &projected_rows)?;
    let binding = TerminalSpartanBinding {
        requested_source_rows,
        verifier_native_guards: terminal_verifier_native_guard_names(),
        column_layout,
        projected_rows,
        spartan_rows: audit.spartan_rows(),
        spartan_columns: audit.spartan_columns(),
        spartan_padding_rows: audit.source().rows()..audit.spartan_rows(),
        spartan_private_padding_columns,
        diagnostic_digest,
    };
    validate_binding(&problem, &binding)?;
    Ok(TerminalProblemExport { problem, binding })
}

fn project_row(row: &Row, layout: &TerminalColumnLayout) -> Result<TerminalProjectedRowArtifact, ExportError> {
    Ok(TerminalProjectedRowArtifact {
        source_row: row.source_index,
        spartan_row: row.source_index,
        a: project_terms(&row.a, layout)?,
        b: project_terms(&row.b, layout)?,
        c: project_terms(&row.c, layout)?,
    })
}

fn project_terms(terms: &[Term], layout: &TerminalColumnLayout) -> Result<Vec<Term>, ExportError> {
    let mut projected = terms
        .iter()
        .map(|term| {
            let column = layout
                .source_to_spartan_column(term.column)
                .ok_or_else(|| ExportError::new("terminal source term exceeds the Spartan column map"))?;
            Ok(Term {
                column,
                coefficient: term.coefficient.clone(),
            })
        })
        .collect::<Result<Vec<_>, ExportError>>()?;
    projected.sort_unstable_by_key(|term| term.column);
    Ok(projected)
}

fn validate_terminal_audit(audit: &TerminalR1csConstraintAudit) -> Result<(), ExportError> {
    let source = audit.source();
    if source.rows() == 0
        || source.cols() == 0
        || audit.source_public_columns() == 0
        || audit.source_public_columns() + audit.source_private_columns() != source.cols()
        || audit.spartan_rows() < source.rows()
        || audit.spartan_columns() < source.cols()
        || audit.spartan_private_columns() < audit.source_private_columns()
    {
        return Err(ExportError::new("invalid terminal constraint-audit geometry"));
    }
    if audit.spartan_columns() != audit.spartan_private_columns() + audit.source_public_columns()
        || (0..source.cols()).any(|column| {
            audit
                .source_to_spartan_column(column)
                .is_none_or(|mapped| mapped >= audit.spartan_columns())
        })
    {
        return Err(ExportError::new(
            "terminal source-to-Spartan map is not injective and in range",
        ));
    }
    let reviewed = TERMINAL_R1CS_FAMILY_NAMES
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    let mut cursor = 0usize;
    for range in audit.row_families() {
        if range.row_start != cursor || range.row_end <= range.row_start || !reviewed.contains(range.name) {
            return Err(ExportError::new(
                "terminal row-family ledger is not an exclusive reviewed partition",
            ));
        }
        seen.insert(range.name);
        cursor = range.row_end;
    }
    if cursor != source.rows() || seen != reviewed {
        return Err(ExportError::new(
            "terminal row-family ledger does not cover the source relation",
        ));
    }
    Ok(())
}

fn validate_binding(problem: &Problem, binding: &TerminalSpartanBinding) -> Result<(), ExportError> {
    if problem
        .rows
        .iter()
        .map(|row| row.source_index)
        .ne(binding.requested_source_rows.iter().copied())
        || binding
            .projected_rows
            .iter()
            .map(|row| row.source_row)
            .ne(binding.requested_source_rows.iter().copied())
        || binding
            .projected_rows
            .iter()
            .any(|row| row.spartan_row != row.source_row || row.spartan_row >= binding.spartan_rows)
        || binding.spartan_padding_rows.start != problem.source.total_rows
        || binding.spartan_padding_rows.end != binding.spartan_rows
        || binding.column_layout.source_public_columns + binding.column_layout.source_private_columns
            != problem.column_count
        || binding.column_layout.spartan_private_columns < binding.column_layout.source_private_columns
        || binding.column_layout.spartan_private_columns + binding.column_layout.source_public_columns
            != binding.spartan_columns
        || binding.spartan_private_padding_columns
            != binding
                .column_layout
                .spartan_private_columns
                .saturating_sub(binding.column_layout.source_private_columns)
    {
        return Err(ExportError::new("terminal source-to-Spartan binding is incoherent"));
    }
    let expected_guards = terminal_verifier_native_guard_names();
    let unique_guards = binding
        .verifier_native_guards
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if binding.verifier_native_guards != expected_guards
        || unique_guards.len() != binding.verifier_native_guards.len()
        || binding
            .verifier_native_guards
            .iter()
            .any(|guard| guard.is_empty())
        || problem
            .complete_families
            .iter()
            .any(|family| unique_guards.contains(family.as_str()))
    {
        return Err(ExportError::new(
            "terminal verifier-native guards are not an exact disjoint ledger",
        ));
    }
    Ok(())
}

fn binding_digest(
    problem: &Problem,
    audit: &TerminalR1csConstraintAudit,
    projected_rows: &[TerminalProjectedRowArtifact],
) -> Result<String, ExportError> {
    let mut hasher = Sha256::new();
    hasher.update(TERMINAL_BINDING_DIGEST_DOMAIN);
    hash_bytes(&mut hasher, problem.source.artifact_digest.as_bytes())?;
    hash_usize(&mut hasher, audit.spartan_rows())?;
    hash_usize(&mut hasher, audit.spartan_columns())?;
    hash_usize(&mut hasher, audit.source_public_columns())?;
    hash_usize(&mut hasher, audit.source_private_columns())?;
    hash_usize(&mut hasher, audit.spartan_private_columns())?;
    let native_guards = terminal_verifier_native_guard_names();
    hash_usize(&mut hasher, native_guards.len())?;
    for guard in native_guards {
        hash_bytes(&mut hasher, guard.as_bytes())?;
    }
    hash_usize(&mut hasher, projected_rows.len())?;
    for row in projected_rows {
        hash_usize(&mut hasher, row.source_row)?;
        hash_usize(&mut hasher, row.spartan_row)?;
        hash_terms(&mut hasher, &row.a)?;
        hash_terms(&mut hasher, &row.b)?;
        hash_terms(&mut hasher, &row.c)?;
    }
    let digest = hasher.finalize();
    let mut output = String::with_capacity(7 + digest.len() * 2);
    output.push_str("sha256:");
    for byte in digest {
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(output)
}

fn hash_terms(hasher: &mut Sha256, terms: &[Term]) -> Result<(), ExportError> {
    hash_usize(hasher, terms.len())?;
    for term in terms {
        hash_usize(hasher, term.column)?;
        hash_bytes(hasher, term.coefficient.as_bytes())?;
    }
    Ok(())
}
