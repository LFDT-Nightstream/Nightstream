//! Direct Spartan relation for the Lean-owned terminal R1CS.
//!
//! Owns: statement validation, exact terminal row synthesis, the public and
//! private Spartan column permutation, and canonical sparse matrices.
//!
//! Does not own: the recursive F' relation, terminal statements, Ajtai setup,
//! Spartan setup, proofs, WHIR, or manifest generation.

mod compiler;
mod lane_opening;
#[cfg(test)]
mod lifecycle;
mod streaming_lifecycle;
#[cfg(test)]
mod tests;

use neo_ajtai::AjtaiSModule;
use thiserror::Error;
use wip_spartan::{
    provider::{goldi::F as SpartanF, GoldilocksWhirEngine},
    SplitR1CSShape,
};

use crate::engine::r1cs_circuit::builder::RowFamilyRange;
use crate::engine::r1cs_circuit::R1csSnapshot;
use crate::paper::relations::{CcsClaim, CcsInstance, CeClaim, LaneScheme, WitnessMat};

use super::{
    lean_native_ccs_manifest::LeanNativeCcsManifest, lean_nebula_combined_manifest::LeanNebulaCombinedManifest,
};

pub use streaming_lifecycle::{
    enforce_streaming_terminal_lifecycle, streaming_terminal_x_out_authority_audit, StreamingTerminalLifecycleError,
    StreamingTerminalLifecycleOutput, StreamingTerminalPublicWires, StreamingTerminalXOutAuthorityAudit,
    STREAMING_TERMINAL_R1CS_FAMILY_NAMES,
};

/// Direct Spartan engine used by the terminal reference relation.
pub type TerminalSpartanEngine = GoldilocksWhirEngine;

#[repr(u8)]
#[derive(Clone, Copy)]
enum TerminalContextGuard {
    Induction,
    PlainChain,
    PublicWidth,
    RelationStructure,
}

impl TerminalContextGuard {
    const fn name(self) -> &'static str {
        match self {
            Self::Induction => "terminal.context.induction",
            Self::PlainChain => "terminal.context.plain_chain",
            Self::PublicWidth => "terminal.context.public_width",
            Self::RelationStructure => "terminal.context.relation_structure",
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy)]
enum TerminalStatementGuard {
    RunningClaimCount,
    VerifierKey,
    InitialSemanticState,
    InitialBoundary,
    ProgramCounter,
    Counters,
    FreshBoundary,
    RunningAccumulator,
    SemanticState,
    StateXOut,
    FreshPublicLink,
}

impl TerminalStatementGuard {
    const fn name(self) -> &'static str {
        match self {
            Self::RunningClaimCount => "terminal.statement.running_claim_count",
            Self::VerifierKey => "terminal.statement.verifier_key",
            Self::InitialSemanticState => "terminal.statement.initial_semantic_state",
            Self::InitialBoundary => "terminal.statement.initial_boundary",
            Self::ProgramCounter => "terminal.statement.program_counter",
            Self::Counters => "terminal.statement.counters",
            Self::FreshBoundary => "terminal.statement.fresh_boundary",
            Self::RunningAccumulator => "terminal.statement.running_accumulator",
            Self::SemanticState => "terminal.statement.semantic_state",
            Self::StateXOut => "terminal.statement.state_x_out",
            Self::FreshPublicLink => "terminal.statement.fresh_public_link",
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy)]
enum TerminalProofGuard {
    ExpectedPublicImage,
    SpartanVerification,
    PublicStatement,
}

impl TerminalProofGuard {
    const fn name(self) -> &'static str {
        match self {
            Self::ExpectedPublicImage => "terminal.proof.expected_public_image",
            Self::SpartanVerification => "terminal.proof.spartan_verification",
            Self::PublicStatement => "terminal.proof.public_statement",
        }
    }
}

/// Reviewed semantic family vocabulary for the direct terminal R1CS.
pub const TERMINAL_R1CS_FAMILY_NAMES: [&str; 8] = [
    "terminal.fresh.commitment",
    "terminal.fresh.norm",
    "terminal.fresh.public_projection",
    "terminal.fresh.selected_relation",
    "terminal.running.commitment",
    "terminal.running.evaluations",
    "terminal.running.norm",
    "terminal.running.public_projection",
];

/// Verifier-owned context guards outside the terminal R1CS.
pub const TERMINAL_CONTEXT_GUARD_NAMES: [&str; 4] = [
    TerminalContextGuard::Induction.name(),
    TerminalContextGuard::PlainChain.name(),
    TerminalContextGuard::PublicWidth.name(),
    TerminalContextGuard::RelationStructure.name(),
];

/// Verifier-native terminal statement guards outside the terminal R1CS.
///
/// cvc5 must not classify these names as polynomial row families. Their
/// authority comes from verifier recomputation and separate Lean proofs.
pub const TERMINAL_STATEMENT_GUARD_NAMES: [&str; 11] = [
    TerminalStatementGuard::RunningClaimCount.name(),
    TerminalStatementGuard::VerifierKey.name(),
    TerminalStatementGuard::InitialSemanticState.name(),
    TerminalStatementGuard::InitialBoundary.name(),
    TerminalStatementGuard::ProgramCounter.name(),
    TerminalStatementGuard::Counters.name(),
    TerminalStatementGuard::FreshBoundary.name(),
    TerminalStatementGuard::RunningAccumulator.name(),
    TerminalStatementGuard::SemanticState.name(),
    TerminalStatementGuard::StateXOut.name(),
    TerminalStatementGuard::FreshPublicLink.name(),
];

/// Cryptographic proof-boundary guards outside the terminal R1CS.
pub const TERMINAL_PROOF_GUARD_NAMES: [&str; 3] = [
    TerminalProofGuard::ExpectedPublicImage.name(),
    TerminalProofGuard::SpartanVerification.name(),
    TerminalProofGuard::PublicStatement.name(),
];

/// Exact terminal relation and one satisfying assignment.
pub struct CompiledTerminalR1cs {
    shape: SplitR1CSShape<TerminalSpartanEngine>,
    private_values: Vec<SpartanF>,
    public_values: Vec<SpartanF>,
    lean_public_columns: usize,
    constraint_audit: TerminalR1csConstraintAudit,
}

/// Exact unpadded terminal R1CS and its map into the padded Spartan shape.
///
/// Source columns use `[one, public, private]`. Spartan uses
/// `[padded private, one, public]`. The explicit map binds both layouts.
#[derive(Clone, Debug)]
pub struct TerminalR1csConstraintAudit {
    source: R1csSnapshot,
    row_families: Vec<RowFamilyRange>,
    reviewed_family_names: Vec<&'static str>,
    source_public_columns: usize,
    source_private_columns: usize,
    spartan_private_columns: usize,
    spartan_rows: usize,
    spartan_columns: usize,
}

impl TerminalR1csConstraintAudit {
    pub fn source(&self) -> &R1csSnapshot {
        &self.source
    }

    pub fn row_families(&self) -> &[RowFamilyRange] {
        &self.row_families
    }

    /// Exact family vocabulary that the compiler reviewed for this relation.
    pub fn reviewed_family_names(&self) -> &[&'static str] {
        &self.reviewed_family_names
    }

    /// Source public prefix length, including the constant-one column.
    pub fn source_public_columns(&self) -> usize {
        self.source_public_columns
    }

    pub fn source_private_columns(&self) -> usize {
        self.source_private_columns
    }

    pub fn spartan_private_columns(&self) -> usize {
        self.spartan_private_columns
    }

    /// Map `[one, public, private]` source columns into
    /// `[padded private, one, public]` Spartan columns.
    pub fn source_to_spartan_column(&self, source_column: usize) -> Option<usize> {
        if source_column == 0 {
            Some(self.spartan_private_columns)
        } else if source_column < self.source_public_columns {
            Some(self.spartan_private_columns + source_column)
        } else if source_column < self.source.cols() {
            Some(source_column - self.source_public_columns)
        } else {
            None
        }
    }

    pub fn spartan_rows(&self) -> usize {
        self.spartan_rows
    }

    pub fn spartan_columns(&self) -> usize {
        self.spartan_columns
    }
}

/// Verifier-reconstructible terminal relation and its explicit public values.
///
/// This value contains no private witness. The verifier derives it from the
/// Lean manifest, its Ajtai key, and the public terminal claims.
pub struct CompiledTerminalR1csStatement {
    shape: SplitR1CSShape<TerminalSpartanEngine>,
    public_values: Vec<SpartanF>,
    lean_public_columns: usize,
}

impl CompiledTerminalR1csStatement {
    pub fn shape(&self) -> &SplitR1CSShape<TerminalSpartanEngine> {
        &self.shape
    }

    /// Public values excluding Spartan's implicit constant-one column.
    pub fn public_values(&self) -> &[SpartanF] {
        &self.public_values
    }

    pub fn lean_public_columns(&self) -> usize {
        self.lean_public_columns
    }

    pub fn into_parts(self) -> (SplitR1CSShape<TerminalSpartanEngine>, Vec<SpartanF>) {
        (self.shape, self.public_values)
    }
}

impl CompiledTerminalR1cs {
    pub fn shape(&self) -> &SplitR1CSShape<TerminalSpartanEngine> {
        &self.shape
    }

    pub fn private_values(&self) -> &[SpartanF] {
        &self.private_values
    }

    /// Public values excluding Spartan's implicit constant-one column.
    pub fn public_values(&self) -> &[SpartanF] {
        &self.public_values
    }

    /// Lean counts the verifier-owned constant-one column as public.
    pub fn lean_public_columns(&self) -> usize {
        self.lean_public_columns
    }

    pub fn constraint_audit(&self) -> &TerminalR1csConstraintAudit {
        &self.constraint_audit
    }

    pub fn into_parts(self) -> (SplitR1CSShape<TerminalSpartanEngine>, Vec<SpartanF>, Vec<SpartanF>) {
        (self.shape, self.private_values, self.public_values)
    }
}

/// Terminal inputs. Claims are verifier-visible; witness matrices are private.
pub struct TerminalR1csInput<'a> {
    pub running_claims: &'a [CeClaim],
    pub running_witnesses: &'a [WitnessMat],
    pub fresh: &'a CcsInstance,
}

/// Public terminal statement used to reconstruct Spartan verification keys.
pub struct TerminalR1csStatement<'a> {
    pub running_claims: &'a [CeClaim],
    pub fresh_claim: &'a CcsClaim,
}

#[derive(Debug, Error)]
pub enum TerminalR1csError {
    #[error("terminal R1CS manifest emission failed: {0}")]
    Manifest(String),
    #[error("terminal R1CS shape mismatch for {what}: expected {expected}, got {got}")]
    Shape {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("terminal R1CS does not support {0}")]
    Unsupported(&'static str),
    #[error("terminal R1CS coefficient construction failed: {0}")]
    Coefficients(String),
    #[error("terminal R1CS assignment is unsatisfied at row {0}")]
    Unsatisfied(usize),
    #[error("terminal R1CS Spartan shape failed: {0}")]
    Spartan(String),
    #[error("terminal R1CS carrier materialization failed: {0}")]
    Carrier(String),
    #[error("terminal R1CS preprocessing relation differs from the Lean manifest")]
    RelationMismatch,
    #[error("terminal R1CS Ajtai setup differs from the Lean manifest")]
    SetupMismatch,
    #[error("terminal R1CS preprocessing does not certify the complete recursive F-prime induction")]
    UncertifiedInduction,
    #[error("terminal R1CS public image differs from the expected image")]
    PublicImageMismatch,
    #[error("terminal R1CS proof public statement differs from verifier reconstruction")]
    PublicStatementMismatch,
    #[error("terminal R1CS state is invalid: {0}")]
    InvalidState(&'static str),
}

/// Compile the exact Lean-owned direct terminal relation.
pub fn compile_terminal_r1cs(
    manifest: &LeanNativeCcsManifest,
    log: &AjtaiSModule,
    input: TerminalR1csInput<'_>,
) -> Result<CompiledTerminalR1cs, TerminalR1csError> {
    compiler::compile(manifest, log, input)
}

/// Rebuild the exact terminal R1CS shape and public vector without private
/// witnesses. This is the verifier-side companion to [`compile_terminal_r1cs`].
pub fn compile_terminal_r1cs_statement(
    manifest: &LeanNativeCcsManifest,
    log: &AjtaiSModule,
    statement: TerminalR1csStatement<'_>,
) -> Result<CompiledTerminalR1csStatement, TerminalR1csError> {
    compiler::compile_statement(manifest, log, statement)
}

/// Compile the native terminal relation with three verifier-owned Nebula
/// lane-slice openings for every running and fresh claim. The slice rows are
/// owned by the existing terminal commitment families.
pub fn compile_terminal_r1cs_with_nebula_lanes(
    manifest: &LeanNativeCcsManifest,
    log: &AjtaiSModule,
    lanes: &LaneScheme,
    input: TerminalR1csInput<'_>,
) -> Result<CompiledTerminalR1cs, TerminalR1csError> {
    compiler::compile_with_nebula_lanes(manifest, log, lanes, input)
}

/// Rebuild the Nebula lane-opening terminal shape and public statement
/// without private witnesses.
pub fn compile_terminal_r1cs_statement_with_nebula_lanes(
    manifest: &LeanNativeCcsManifest,
    log: &AjtaiSModule,
    lanes: &LaneScheme,
    statement: TerminalR1csStatement<'_>,
) -> Result<CompiledTerminalR1csStatement, TerminalR1csError> {
    compiler::compile_statement_with_nebula_lanes(manifest, log, lanes, statement)
}

/// Compile the exact Lean-owned native F-prime plus Nebula terminal relation.
pub fn compile_combined_terminal_r1cs(
    manifest: &LeanNebulaCombinedManifest,
    log: &AjtaiSModule,
    input: TerminalR1csInput<'_>,
) -> Result<CompiledTerminalR1cs, TerminalR1csError> {
    compiler::compile_combined(manifest, log, input)
}

/// Rebuild the combined terminal shape and public vector without private
/// witnesses.
pub fn compile_combined_terminal_r1cs_statement(
    manifest: &LeanNebulaCombinedManifest,
    log: &AjtaiSModule,
    statement: TerminalR1csStatement<'_>,
) -> Result<CompiledTerminalR1csStatement, TerminalR1csError> {
    compiler::compile_combined_statement(manifest, log, statement)
}

#[cfg(test)]
pub use lifecycle::{
    audit_combined_terminal_context_guards, audit_combined_terminal_statement_guards, audit_terminal_context_guards,
    audit_terminal_statement_guards, finish_combined_with_spartan, finish_with_spartan, verify_combined_spartan,
    verify_spartan, TerminalContextGuardAudit, TerminalRunningStatement, TerminalSpartanProof,
    TerminalSpartanStatement, TerminalStatementGuardAudit,
};
