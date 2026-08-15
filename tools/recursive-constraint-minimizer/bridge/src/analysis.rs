//! Fail-closed fixed-point family analysis.

use neo_fold_clean::frontends::nebula::f_prime::{NebulaFPrimeBranch, NebulaFPrimeConstraintSourceAudit};
use neo_fold_clean::frontends::r1cs_f_prime::ivc::{R1csIvcBranch, R1csIvcConstraintSourceAudit};
use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::TerminalR1csConstraintAudit;
use neo_math::F;
use recursive_constraint_minimizer::{
    derive_scalar_certificate, validate_scalar_certificate, Conclusion, ScalarCertificate, Scope, Selection,
    SolverConfig,
};

use super::{
    export_fixed_point_problem, export_nebula_problem, export_terminal_problem, fixed_point_family_census,
    nebula_family_census, refine_fixed_point_with_cvc5, refine_nebula_with_cvc5, refine_terminal_with_cvc5,
    terminal_family_census, ExportError, ExportRequest, FixedPointRefinementReport, RefinementError, SparseOwnedFamily,
    TerminalOwnedFamily, TerminalRefinementReport,
};

/// One bounded search result. Only `RedundancyCertificate` has a universal
/// algebraic certificate. Neither variant authorizes a circuit change until
/// its generated data is checked in Lean.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedPointFamilySearch {
    RedundancyCertificate {
        report: FixedPointRefinementReport,
        certificate: ScalarCertificate,
    },
    RustCounterexampleCandidate {
        report: FixedPointRefinementReport,
    },
    Inconclusive {
        family: String,
        reason: String,
    },
}

impl FixedPointFamilySearch {
    pub fn family(&self) -> &str {
        match self {
            Self::RedundancyCertificate { report, .. } | Self::RustCounterexampleCandidate { report } => report
                .refinement()
                .problem
                .complete_families
                .first()
                .expect("family searches always select one complete family"),
            Self::Inconclusive { family, .. } => family,
        }
    }
}

/// One exact family and its fail-closed bounded search result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedPointFamilySearchRecord {
    family: SparseOwnedFamily,
    search: FixedPointFamilySearch,
}

impl FixedPointFamilySearchRecord {
    pub fn name(&self) -> &'static str {
        self.family.name()
    }

    pub fn source_rows(&self) -> &[usize] {
        self.family.source_rows()
    }

    pub fn search(&self) -> &FixedPointFamilySearch {
        &self.search
    }
}

/// Exhaustive bounded search ledger for one exact fixed-point branch.
///
/// The source and final-plan digests are diagnostic identities. The exact
/// rows and bindings in each successful family report remain authoritative.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedPointBranchSearchReport {
    profile: String,
    branch: R1csIvcBranch,
    source_artifact_digest: String,
    final_plan_digest: String,
    source_rows: usize,
    source_columns: usize,
    source_public_columns: usize,
    final_rows: usize,
    final_columns: usize,
    final_public_columns: usize,
    families: Vec<FixedPointFamilySearchRecord>,
}

/// Exhaustive bounded search ledger for one exact Nebula F-prime branch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaBranchSearchReport {
    profile: String,
    branch: NebulaFPrimeBranch,
    source_artifact_digest: String,
    final_plan_digest: String,
    source_rows: usize,
    source_columns: usize,
    source_public_columns: usize,
    final_rows: usize,
    final_columns: usize,
    final_public_columns: usize,
    families: Vec<FixedPointFamilySearchRecord>,
}

/// One bounded search result for a terminal polynomial family.
///
/// Native context, statement, and proof guards are not part of this enum.
/// They remain outside the cvc5 removal plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TerminalFamilySearch {
    RedundancyCertificate {
        report: TerminalRefinementReport,
        certificate: ScalarCertificate,
    },
    RustCounterexampleCandidate {
        report: TerminalRefinementReport,
    },
    Inconclusive {
        family: String,
        reason: String,
    },
}

impl TerminalFamilySearch {
    pub fn family(&self) -> &str {
        match self {
            Self::RedundancyCertificate { report, .. } | Self::RustCounterexampleCandidate { report } => report
                .refinement()
                .problem
                .complete_families
                .first()
                .expect("family searches always select one complete family"),
            Self::Inconclusive { family, .. } => family,
        }
    }
}

/// One exact terminal polynomial family and its bounded search result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalFamilySearchRecord {
    family: TerminalOwnedFamily,
    search: TerminalFamilySearch,
}

impl TerminalFamilySearchRecord {
    pub fn name(&self) -> &'static str {
        self.family.name()
    }

    pub fn source_rows(&self) -> &[usize] {
        self.family.source_rows()
    }

    pub fn search(&self) -> &TerminalFamilySearch {
        &self.search
    }
}

/// Exhaustive bounded search ledger for the terminal polynomial relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalFamilySearchReport {
    profile: String,
    source_artifact_digest: String,
    source_rows: usize,
    source_columns: usize,
    source_public_columns: usize,
    source_private_columns: usize,
    spartan_rows: usize,
    spartan_columns: usize,
    spartan_private_columns: usize,
    families: Vec<TerminalFamilySearchRecord>,
}

impl TerminalFamilySearchReport {
    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn source_artifact_digest(&self) -> &str {
        &self.source_artifact_digest
    }

    pub fn source_rows(&self) -> usize {
        self.source_rows
    }

    pub fn source_columns(&self) -> usize {
        self.source_columns
    }

    pub fn source_public_columns(&self) -> usize {
        self.source_public_columns
    }

    pub fn source_private_columns(&self) -> usize {
        self.source_private_columns
    }

    pub fn spartan_rows(&self) -> usize {
        self.spartan_rows
    }

    pub fn spartan_columns(&self) -> usize {
        self.spartan_columns
    }

    pub fn spartan_private_columns(&self) -> usize {
        self.spartan_private_columns
    }

    pub fn families(&self) -> &[TerminalFamilySearchRecord] {
        &self.families
    }
}

impl FixedPointBranchSearchReport {
    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn branch(&self) -> R1csIvcBranch {
        self.branch
    }

    pub fn source_artifact_digest(&self) -> &str {
        &self.source_artifact_digest
    }

    pub fn final_plan_digest(&self) -> &str {
        &self.final_plan_digest
    }

    pub fn source_rows(&self) -> usize {
        self.source_rows
    }

    pub fn source_columns(&self) -> usize {
        self.source_columns
    }

    pub fn source_public_columns(&self) -> usize {
        self.source_public_columns
    }

    pub fn final_rows(&self) -> usize {
        self.final_rows
    }

    pub fn final_columns(&self) -> usize {
        self.final_columns
    }

    pub fn final_public_columns(&self) -> usize {
        self.final_public_columns
    }

    pub fn families(&self) -> &[FixedPointFamilySearchRecord] {
        &self.families
    }
}

impl NebulaBranchSearchReport {
    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn branch(&self) -> NebulaFPrimeBranch {
        self.branch
    }

    pub fn source_artifact_digest(&self) -> &str {
        &self.source_artifact_digest
    }

    pub fn final_plan_digest(&self) -> &str {
        &self.final_plan_digest
    }

    pub fn source_rows(&self) -> usize {
        self.source_rows
    }

    pub fn source_columns(&self) -> usize {
        self.source_columns
    }

    pub fn source_public_columns(&self) -> usize {
        self.source_public_columns
    }

    pub fn final_rows(&self) -> usize {
        self.final_rows
    }

    pub fn final_columns(&self) -> usize {
        self.final_columns
    }

    pub fn final_public_columns(&self) -> usize {
        self.final_public_columns
    }

    pub fn families(&self) -> &[FixedPointFamilySearchRecord] {
        &self.families
    }
}

/// Analyze every reviewed family in one exact base or recursive branch.
///
/// A source or binding error prevents the ledger from being built. Once the
/// exact ledger exists, each family gets one result. Per-family solver,
/// parser, replay, or certificate failures become `Inconclusive` and retain
/// that family.
#[allow(clippy::too_many_arguments)]
pub fn analyze_fixed_point_branch(
    audit: &R1csIvcConstraintSourceAudit,
    branch: R1csIvcBranch,
    background_assignment: &[F],
    profile: &str,
    solver: &SolverConfig,
    max_iterations: usize,
) -> Result<FixedPointBranchSearchReport, ExportError> {
    let census = fixed_point_family_census(audit, branch)?;
    let identity_family = census
        .first()
        .ok_or_else(|| ExportError::new("fixed-point branch has no reviewed family"))?;
    let arm = audit.arm(branch);
    let identity = export_fixed_point_problem(
        audit,
        branch,
        ExportRequest {
            profile: profile.to_owned(),
            scope: Scope::Branch,
            public_input_count: arm.m_in,
            source_rows: identity_family.source_rows().to_vec(),
            complete_families: vec![identity_family.name().to_owned()],
        },
    )?;
    let source_artifact_digest = identity.problem().source.artifact_digest.clone();
    let final_plan_digest = identity.binding().final_plan_digest().to_owned();
    let final_rows = identity.binding().final_rows();
    let final_columns = identity.binding().final_columns();
    let final_public_columns = identity.binding().final_public_input_count();

    let families = census
        .into_iter()
        .map(|family| {
            let search = analyze_fixed_point_family(
                audit,
                branch,
                background_assignment,
                profile,
                &family,
                solver,
                max_iterations,
            );
            debug_assert_eq!(search.family(), family.name());
            FixedPointFamilySearchRecord { family, search }
        })
        .collect();

    Ok(FixedPointBranchSearchReport {
        profile: profile.to_owned(),
        branch,
        source_artifact_digest,
        final_plan_digest,
        source_rows: arm.n,
        source_columns: arm.m,
        source_public_columns: arm.m_in,
        final_rows,
        final_columns,
        final_public_columns,
        families,
    })
}

/// Analyze every reviewed family in one exact Nebula F-prime branch.
///
/// Each family keeps its own fail-closed result. A failed solver or replay
/// cannot omit the family from the returned ledger.
#[allow(clippy::too_many_arguments)]
pub fn analyze_nebula_branch(
    audit: &NebulaFPrimeConstraintSourceAudit,
    branch: NebulaFPrimeBranch,
    background_assignment: &[F],
    profile: &str,
    solver: &SolverConfig,
    max_iterations: usize,
) -> Result<NebulaBranchSearchReport, ExportError> {
    let census = nebula_family_census(audit, branch)?;
    let identity_family = census
        .first()
        .ok_or_else(|| ExportError::new("Nebula branch has no reviewed family"))?;
    let arm = audit.arm(branch);
    let identity = export_nebula_problem(
        audit,
        branch,
        ExportRequest {
            profile: profile.to_owned(),
            scope: Scope::Branch,
            public_input_count: arm.m_in,
            source_rows: identity_family.source_rows().to_vec(),
            complete_families: vec![identity_family.name().to_owned()],
        },
    )?;
    let source_artifact_digest = identity.problem().source.artifact_digest.clone();
    let final_plan_digest = identity.binding().final_plan_digest().to_owned();
    let final_rows = identity.binding().final_rows();
    let final_columns = identity.binding().final_columns();
    let final_public_columns = identity.binding().final_public_input_count();

    let families = census
        .into_iter()
        .map(|family| {
            let search = analyze_nebula_family(
                audit,
                branch,
                background_assignment,
                profile,
                &family,
                solver,
                max_iterations,
            );
            debug_assert_eq!(search.family(), family.name());
            FixedPointFamilySearchRecord { family, search }
        })
        .collect();

    Ok(NebulaBranchSearchReport {
        profile: profile.to_owned(),
        branch,
        source_artifact_digest,
        final_plan_digest,
        source_rows: arm.n,
        source_columns: arm.m,
        source_public_columns: arm.m_in,
        final_rows,
        final_columns,
        final_public_columns,
        families,
    })
}

/// Analyze every reviewed terminal polynomial family.
///
/// The compiler audit supplies the exact satisfying background assignment.
/// Per-family solver, parser, replay, or certificate failures become
/// `Inconclusive` and retain that family.
pub fn analyze_terminal_families(
    audit: &TerminalR1csConstraintAudit,
    profile: &str,
    solver: &SolverConfig,
    max_iterations: usize,
) -> Result<TerminalFamilySearchReport, ExportError> {
    let census = terminal_family_census(audit)?;
    let identity_family = census
        .first()
        .ok_or_else(|| ExportError::new("terminal relation has no reviewed polynomial family"))?;
    let identity = export_terminal_problem(
        audit,
        ExportRequest {
            profile: profile.to_owned(),
            scope: Scope::Branch,
            public_input_count: audit.source_public_columns(),
            source_rows: identity_family.source_rows().to_vec(),
            complete_families: vec![identity_family.name().to_owned()],
        },
    )?;
    let source_artifact_digest = identity.problem().source.artifact_digest.clone();

    let families = census
        .into_iter()
        .map(|family| {
            let search = analyze_terminal_family(audit, profile, &family, solver, max_iterations);
            debug_assert_eq!(search.family(), family.name());
            TerminalFamilySearchRecord { family, search }
        })
        .collect();

    Ok(TerminalFamilySearchReport {
        profile: profile.to_owned(),
        source_artifact_digest,
        source_rows: audit.source().rows(),
        source_columns: audit.source().cols(),
        source_public_columns: audit.source_public_columns(),
        source_private_columns: audit.source_private_columns(),
        spartan_rows: audit.spartan_rows(),
        spartan_columns: audit.spartan_columns(),
        spartan_private_columns: audit.spartan_private_columns(),
        families,
    })
}

/// Analyze one complete reviewed physical-stage family.
///
/// Timeout, solver failure, parser failure, incomplete scalar certificates,
/// and all other uncertain results return `Inconclusive` and retain the
/// family.
#[allow(clippy::too_many_arguments)]
pub fn analyze_fixed_point_family(
    audit: &R1csIvcConstraintSourceAudit,
    branch: R1csIvcBranch,
    background_assignment: &[F],
    profile: &str,
    family_record: &SparseOwnedFamily,
    solver: &SolverConfig,
    max_iterations: usize,
) -> FixedPointFamilySearch {
    let family = family_record.name();
    let selection = Selection::Family(family.to_owned());
    let request = ExportRequest {
        profile: profile.to_owned(),
        scope: Scope::Branch,
        public_input_count: audit.arm(branch).m_in,
        source_rows: family_record.source_rows().to_vec(),
        complete_families: vec![family.to_owned()],
    };
    classify_fixed_point_family(
        family,
        &selection,
        refine_fixed_point_with_cvc5(
            audit,
            branch,
            background_assignment,
            request,
            &selection,
            solver,
            max_iterations,
        ),
    )
}

/// Analyze one complete reviewed Nebula physical-stage family.
#[allow(clippy::too_many_arguments)]
pub fn analyze_nebula_family(
    audit: &NebulaFPrimeConstraintSourceAudit,
    branch: NebulaFPrimeBranch,
    background_assignment: &[F],
    profile: &str,
    family_record: &SparseOwnedFamily,
    solver: &SolverConfig,
    max_iterations: usize,
) -> FixedPointFamilySearch {
    let family = family_record.name();
    let selection = Selection::Family(family.to_owned());
    let request = ExportRequest {
        profile: profile.to_owned(),
        scope: Scope::Branch,
        public_input_count: audit.arm(branch).m_in,
        source_rows: family_record.source_rows().to_vec(),
        complete_families: vec![family.to_owned()],
    };
    classify_fixed_point_family(
        family,
        &selection,
        refine_nebula_with_cvc5(
            audit,
            branch,
            background_assignment,
            request,
            &selection,
            solver,
            max_iterations,
        ),
    )
}

fn classify_fixed_point_family(
    family: &str,
    selection: &Selection,
    report: Result<FixedPointRefinementReport, RefinementError>,
) -> FixedPointFamilySearch {
    let report = match report {
        Ok(report) => report,
        Err(error) => return inconclusive(family, format!("bounded refinement failed: {error}")),
    };
    match report.refinement().conclusion {
        Conclusion::CounterexampleCandidate => FixedPointFamilySearch::RustCounterexampleCandidate { report },
        Conclusion::Inconclusive => inconclusive(family, "cvc5 returned no checked conclusion"),
        Conclusion::RedundancyCandidate => {
            let certificate = match derive_scalar_certificate(&report.refinement().problem, selection) {
                Ok(Some(certificate)) => certificate,
                Ok(None) => {
                    return inconclusive(
                        family,
                        "the checked scalar certificate grammar cannot prove this implication",
                    );
                }
                Err(error) => {
                    return inconclusive(family, format!("scalar certificate search failed: {error}"));
                }
            };
            if let Err(error) = validate_scalar_certificate(&report.refinement().problem, &certificate) {
                return inconclusive(family, format!("scalar certificate replay failed: {error}"));
            }
            FixedPointFamilySearch::RedundancyCertificate { report, certificate }
        }
    }
}

/// Analyze one complete terminal polynomial family.
pub fn analyze_terminal_family(
    audit: &TerminalR1csConstraintAudit,
    profile: &str,
    family_record: &TerminalOwnedFamily,
    solver: &SolverConfig,
    max_iterations: usize,
) -> TerminalFamilySearch {
    let family = family_record.name();
    let selection = Selection::Family(family.to_owned());
    let request = ExportRequest {
        profile: profile.to_owned(),
        scope: Scope::Branch,
        public_input_count: audit.source_public_columns(),
        source_rows: family_record.source_rows().to_vec(),
        complete_families: vec![family.to_owned()],
    };
    let report = match refine_terminal_with_cvc5(audit, request, &selection, solver, max_iterations) {
        Ok(report) => report,
        Err(error) => return terminal_inconclusive(family, format!("bounded refinement failed: {error}")),
    };

    match report.refinement().conclusion {
        Conclusion::CounterexampleCandidate => TerminalFamilySearch::RustCounterexampleCandidate { report },
        Conclusion::Inconclusive => terminal_inconclusive(family, "cvc5 returned no checked conclusion"),
        Conclusion::RedundancyCandidate => {
            let certificate = match derive_scalar_certificate(&report.refinement().problem, &selection) {
                Ok(Some(certificate)) => certificate,
                Ok(None) => {
                    return terminal_inconclusive(
                        family,
                        "the checked scalar certificate grammar cannot prove this implication",
                    );
                }
                Err(error) => {
                    return terminal_inconclusive(family, format!("scalar certificate search failed: {error}"));
                }
            };
            if let Err(error) = validate_scalar_certificate(&report.refinement().problem, &certificate) {
                return terminal_inconclusive(family, format!("scalar certificate replay failed: {error}"));
            }
            TerminalFamilySearch::RedundancyCertificate { report, certificate }
        }
    }
}

fn inconclusive(family: &str, reason: impl Into<String>) -> FixedPointFamilySearch {
    FixedPointFamilySearch::Inconclusive {
        family: family.to_owned(),
        reason: reason.into(),
    }
}

fn terminal_inconclusive(family: &str, reason: impl Into<String>) -> TerminalFamilySearch {
    TerminalFamilySearch::Inconclusive {
        family: family.to_owned(),
        reason: reason.into(),
    }
}
