//! Direct Spartan relation for the Lean-owned terminal R1CS.
//!
//! Owns: statement validation, exact terminal row synthesis, the public and
//! private Spartan column permutation, and canonical sparse matrices.
//!
//! Does not own: the recursive F' relation, terminal statements, Ajtai setup,
//! Spartan setup, proofs, WHIR, or manifest generation.

mod compiler;
mod lifecycle;

use neo_ajtai::AjtaiSModule;
use thiserror::Error;
use wip_spartan::{
    provider::{goldi::F as SpartanF, GoldilocksWhirEngine},
    SplitR1CSShape,
};

use crate::paper::relations::{CcsClaim, CcsInstance, CeClaim, WitnessMat};

use super::{
    lean_native_ccs_manifest::LeanNativeCcsManifest, lean_nebula_combined_manifest::LeanNebulaCombinedManifest,
};

/// Direct Spartan engine used by the terminal reference relation.
pub type TerminalSpartanEngine = GoldilocksWhirEngine;

/// Exact terminal relation and one satisfying assignment.
pub struct CompiledTerminalR1cs {
    shape: SplitR1CSShape<TerminalSpartanEngine>,
    private_values: Vec<SpartanF>,
    public_values: Vec<SpartanF>,
    lean_public_columns: usize,
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

pub use lifecycle::{
    finish_with_spartan, verify_spartan, TerminalRunningStatement, TerminalSpartanProof, TerminalSpartanStatement,
};
