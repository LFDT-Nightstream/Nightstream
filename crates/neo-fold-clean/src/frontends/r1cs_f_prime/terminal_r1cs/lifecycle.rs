//! Spartan/WHIR terminal lifecycle for the Lean-owned CCS relations.
//!
//! Owns: terminal statement binding, verifier-side relation reconstruction,
//! direct Spartan setup, proving, and verification.
//!
//! Does not own: recursive F' compilation, the Lean manifest bytes, Ajtai
//! setup, application execution, Nebula finalization, or a cryptographic
//! reduction for Spartan, Fiat--Shamir, or WHIR.

use neo_ccs::CcsMatrix;
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};
use wip_spartan::spartan::{RepeatedR1CSSNARK, R1CSSNARK};

use crate::lifecycle::{Preprocessing, PublicImage, Uncompressed};
use crate::paper::construction2::{self, ProofState, RunningInstance, SemanticStateMode, State};
use crate::paper::digest::{digest_fields_as_digest32, initial_boundary_digest};
use crate::paper::f_prime::r1cs::{f_prime_public_input_link_matches, FPrimePublicInputLayout};
use crate::paper::relations::{CcsClaim, CeClaim, Structure};

use super::super::LeanNebulaCombinedPreprocessing;

use super::{
    compile_combined_terminal_r1cs, compile_combined_terminal_r1cs_statement, compile_terminal_r1cs,
    compile_terminal_r1cs_statement, CompiledTerminalR1cs, CompiledTerminalR1csStatement, LeanNativeCcsManifest,
    LeanNebulaCombinedManifest, TerminalContextGuard, TerminalProofGuard, TerminalR1csError, TerminalR1csInput,
    TerminalR1csStatement, TerminalSpartanEngine, TerminalStatementGuard, TERMINAL_CONTEXT_GUARD_NAMES,
    TERMINAL_PROOF_GUARD_NAMES, TERMINAL_STATEMENT_GUARD_NAMES,
};

#[derive(Clone, Copy)]
enum TerminalRelation<'a> {
    Native(&'a LeanNativeCcsManifest),
    Combined(&'a LeanNebulaCombinedManifest),
}

impl TerminalRelation<'_> {
    fn running_claim_count(self) -> usize {
        match self {
            Self::Native(manifest) => manifest.running_claim_count(),
            Self::Combined(manifest) => manifest.running_claim_count(),
        }
    }

    fn public_carrier_width(self) -> usize {
        match self {
            Self::Native(manifest) => manifest.public_carrier_width(),
            Self::Combined(manifest) => manifest.public_carrier_width(),
        }
    }

    fn public_input_layout(self) -> FPrimePublicInputLayout {
        match self {
            Self::Native(_) => FPrimePublicInputLayout::plain(),
            Self::Combined(manifest) => manifest.public_input_layout(),
        }
    }

    fn structure(self) -> Result<Structure, TerminalR1csError> {
        match self {
            Self::Native(manifest) => manifest
                .emit_phi81_step(|_| Some(neo_math::F::ZERO))
                .map(|emission| emission.structure().clone())
                .map_err(|error| TerminalR1csError::Manifest(error.to_string())),
            Self::Combined(manifest) => manifest
                .terminal_structure()
                .map_err(|error| TerminalR1csError::Manifest(error.to_string())),
        }
    }

    fn compile(
        self,
        log: &neo_ajtai::AjtaiSModule,
        input: TerminalR1csInput<'_>,
    ) -> Result<CompiledTerminalR1cs, TerminalR1csError> {
        match self {
            Self::Native(manifest) => compile_terminal_r1cs(manifest, log, input),
            Self::Combined(manifest) => compile_combined_terminal_r1cs(manifest, log, input),
        }
    }

    fn compile_statement(
        self,
        log: &neo_ajtai::AjtaiSModule,
        statement: TerminalR1csStatement<'_>,
    ) -> Result<CompiledTerminalR1csStatement, TerminalR1csError> {
        match self {
            Self::Native(manifest) => compile_terminal_r1cs_statement(manifest, log, statement),
            Self::Combined(manifest) => compile_combined_terminal_r1cs_statement(manifest, log, statement),
        }
    }
}

#[derive(Default)]
struct TerminalGuardTracker {
    accepted_mask: u16,
}

impl TerminalGuardTracker {
    fn accept(&mut self, guard: u8) -> Result<(), TerminalR1csError> {
        let bit = 1u16 << guard;
        if self.accepted_mask & bit != 0 {
            return Err(TerminalR1csError::InvalidState(
                "terminal verifier guard was recorded more than once",
            ));
        }
        self.accepted_mask |= bit;
        Ok(())
    }

    fn require(&mut self, guard: u8, accepted: bool, error: TerminalR1csError) -> Result<(), TerminalR1csError> {
        if !accepted {
            return Err(error);
        }
        self.accept(guard)
    }

    fn finish(self, expected_mask: u16) -> Result<u16, TerminalR1csError> {
        if self.accepted_mask != expected_mask {
            return Err(TerminalR1csError::InvalidState(
                "terminal verifier guard ledger is incomplete",
            ));
        }
        Ok(self.accepted_mask)
    }
}

const TERMINAL_CONTEXT_GUARD_MASK: u16 = (1u16 << TERMINAL_CONTEXT_GUARD_NAMES.len()) - 1;

const TERMINAL_STATEMENT_GUARD_MASK: u16 = (1u16 << TERMINAL_STATEMENT_GUARD_NAMES.len()) - 1;

const TERMINAL_PROOF_GUARD_MASK: u16 = (1u16 << TERMINAL_PROOF_GUARD_NAMES.len()) - 1;

/// Successful evaluation of all verifier-owned terminal context guards.
///
/// This diagnostic value is not a proof and does not cover the terminal R1CS,
/// statement guards, or Spartan verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalContextGuardAudit {
    accepted_mask: u16,
}

impl TerminalContextGuardAudit {
    pub fn guard_names(&self) -> &'static [&'static str] {
        debug_assert_eq!(self.accepted_mask, TERMINAL_CONTEXT_GUARD_MASK);
        &TERMINAL_CONTEXT_GUARD_NAMES
    }
}

/// Successful evaluation of every verifier-native terminal statement guard.
///
/// This diagnostic value is not a proof and does not cover the terminal R1CS
/// or Spartan verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalStatementGuardAudit {
    accepted_mask: u16,
}

impl TerminalStatementGuardAudit {
    pub fn guard_names(&self) -> &'static [&'static str] {
        debug_assert_eq!(self.accepted_mask, TERMINAL_STATEMENT_GUARD_MASK);
        &TERMINAL_STATEMENT_GUARD_NAMES
    }
}

/// Public terminal statement for the direct Spartan relation.
///
/// The verifier replays these exact claims into the Lean-owned terminal R1CS.
/// They are separate from [`TerminalSpartanProof`] so the proof cannot select
/// its own statement implicitly.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TerminalSpartanStatement {
    public_image: PublicImage,
    running: TerminalRunningStatement,
    fresh_claim: CcsClaim,
}

impl TerminalSpartanStatement {
    pub fn new(public_image: PublicImage, running: TerminalRunningStatement, fresh_claim: CcsClaim) -> Self {
        Self {
            public_image,
            running,
            fresh_claim,
        }
    }

    pub fn public_image(&self) -> &PublicImage {
        &self.public_image
    }

    pub fn running(&self) -> &TerminalRunningStatement {
        &self.running
    }

    pub fn fresh_claim(&self) -> &CcsClaim {
        &self.fresh_claim
    }
}

/// Verifier-visible part of one running accumulator.
///
/// Witness matrices are deliberately absent. The checked parent cache remains
/// so the verifier can reproduce the selected accumulator digest.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TerminalRunningStatement {
    claims: Vec<CeClaim>,
    parent_authority: Option<CeClaim>,
}

impl TerminalRunningStatement {
    pub fn from_running(running: &RunningInstance) -> Self {
        Self {
            claims: running.claims.clone(),
            parent_authority: running.parent_authority.clone(),
        }
    }

    pub fn claims(&self) -> &[CeClaim] {
        &self.claims
    }

    fn as_running(&self) -> Result<RunningInstance, TerminalR1csError> {
        Ok(RunningInstance::new(
            self.claims.clone(),
            Vec::new(),
            self.parent_authority.clone(),
        ))
    }
}

/// Succinct terminal proof for one explicit [`TerminalSpartanStatement`].
#[derive(Deserialize, Serialize)]
pub struct TerminalSpartanProof {
    snark: RepeatedR1CSSNARK<TerminalSpartanEngine>,
}

/// Finish a plain authoritative F' proof with direct Spartan and WHIR.
///
/// This consumes HyperNova's pre-final `(running, latest)` pair. It does not
/// perform the legacy extra final fold.
pub fn finish_with_spartan(
    prep: &Preprocessing,
    manifest: &LeanNativeCcsManifest,
    proof: Uncompressed,
) -> Result<(TerminalSpartanStatement, TerminalSpartanProof), TerminalR1csError> {
    finish_relation_with_spartan(prep, TerminalRelation::Native(manifest), proof)
}

/// Finish one combined F' plus Nebula proof with direct Spartan and WHIR.
///
/// The combined fresh instance already contains both row programs. This
/// consumes the plain pre-final `(running, latest)` pair and does not accept
/// the delayed-Nebula final-fold representation.
pub fn finish_combined_with_spartan(
    setup: &LeanNebulaCombinedPreprocessing,
    proof: Uncompressed,
) -> Result<(TerminalSpartanStatement, TerminalSpartanProof), TerminalR1csError> {
    finish_relation_with_spartan(
        setup.preprocessing(),
        TerminalRelation::Combined(setup.manifest()),
        proof,
    )
}

fn finish_relation_with_spartan(
    prep: &Preprocessing,
    relation: TerminalRelation<'_>,
    proof: Uncompressed,
) -> Result<(TerminalSpartanStatement, TerminalSpartanProof), TerminalR1csError> {
    validate_context(prep, relation)?;
    if proof.final_fold.is_some() {
        return Err(TerminalR1csError::InvalidState(
            "plain HyperNova terminal proof must not contain a final fold",
        ));
    }
    let ProofState::Active { running, latest } = &proof.state.proof else {
        return Err(TerminalR1csError::InvalidState(
            "terminal compression needs an active running/latest pair",
        ));
    };
    if latest.instances.len() != 1 {
        return Err(TerminalR1csError::Shape {
            what: "terminal fresh claim count",
            expected: 1,
            got: latest.instances.len(),
        });
    }
    let fresh = &latest.instances[0];
    let public_image = public_image_from_state(prep, &proof.state);
    let running_statement = TerminalRunningStatement::from_running(running);
    validate_public_statement(prep, relation, &public_image, &running_statement, &fresh.claim)?;

    let compiled = relation.compile(
        &prep.log,
        TerminalR1csInput {
            running_claims: &running.claims,
            running_witnesses: &running.witnesses,
            fresh,
        },
    )?;
    let (shape, witness, public_values) = compiled.into_parts();
    let (prover_key, _) = R1CSSNARK::<TerminalSpartanEngine>::setup_direct(shape)
        .map_err(|error| TerminalR1csError::Spartan(error.to_string()))?;
    let snark = RepeatedR1CSSNARK::<TerminalSpartanEngine>::prove_direct(&prover_key, &witness, &public_values, true)
        .map_err(|error| TerminalR1csError::Spartan(error.to_string()))?;

    Ok((
        TerminalSpartanStatement::new(public_image, running_statement, fresh.claim.clone()),
        TerminalSpartanProof { snark },
    ))
}

/// Verify a terminal Spartan/WHIR proof against verifier-owned context and
/// the expected public image.
pub fn verify_spartan(
    prep: &Preprocessing,
    manifest: &LeanNativeCcsManifest,
    expected: &PublicImage,
    statement: &TerminalSpartanStatement,
    proof: &TerminalSpartanProof,
) -> Result<(), TerminalR1csError> {
    verify_relation_spartan(prep, TerminalRelation::Native(manifest), expected, statement, proof)
}

/// Verify one combined F' plus Nebula Spartan proof against its exact setup.
pub fn verify_combined_spartan(
    setup: &LeanNebulaCombinedPreprocessing,
    expected: &PublicImage,
    statement: &TerminalSpartanStatement,
    proof: &TerminalSpartanProof,
) -> Result<(), TerminalR1csError> {
    verify_relation_spartan(
        setup.preprocessing(),
        TerminalRelation::Combined(setup.manifest()),
        expected,
        statement,
        proof,
    )
}

fn verify_relation_spartan(
    prep: &Preprocessing,
    relation: TerminalRelation<'_>,
    expected: &PublicImage,
    statement: &TerminalSpartanStatement,
    proof: &TerminalSpartanProof,
) -> Result<(), TerminalR1csError> {
    validate_context(prep, relation)?;
    let mut proof_guards = TerminalGuardTracker::default();
    proof_guards.require(
        TerminalProofGuard::ExpectedPublicImage as u8,
        statement.public_image() == expected,
        TerminalR1csError::PublicImageMismatch,
    )?;
    validate_public_statement(
        prep,
        relation,
        statement.public_image(),
        statement.running(),
        statement.fresh_claim(),
    )?;

    let compiled = relation.compile_statement(
        &prep.log,
        TerminalR1csStatement {
            running_claims: statement.running().claims(),
            fresh_claim: statement.fresh_claim(),
        },
    )?;
    let (shape, expected_public) = compiled.into_parts();
    let (_, verifier_key) = R1CSSNARK::<TerminalSpartanEngine>::setup_direct(shape)
        .map_err(|error| TerminalR1csError::Spartan(error.to_string()))?;
    let verified_public = proof
        .snark
        .verify(&verifier_key)
        .map_err(|error| TerminalR1csError::Spartan(error.to_string()))?;
    proof_guards.accept(TerminalProofGuard::SpartanVerification as u8)?;
    proof_guards.require(
        TerminalProofGuard::PublicStatement as u8,
        verified_public == expected_public,
        TerminalR1csError::PublicStatementMismatch,
    )?;
    proof_guards.finish(TERMINAL_PROOF_GUARD_MASK)?;
    Ok(())
}

fn validate_context(
    prep: &Preprocessing,
    relation: TerminalRelation<'_>,
) -> Result<TerminalContextGuardAudit, TerminalR1csError> {
    let mut guards = TerminalGuardTracker::default();
    guards.require(
        TerminalContextGuard::Induction as u8,
        prep.enforces_terminal_induction(),
        TerminalR1csError::UncertifiedInduction,
    )?;
    guards.require(
        TerminalContextGuard::PlainChain as u8,
        prep.nebula().is_none(),
        TerminalR1csError::Unsupported("Nebula needs its delayed terminal memory finalization"),
    )?;
    guards.require(
        TerminalContextGuard::PublicWidth as u8,
        prep.public_input_len == Some(relation.public_carrier_width()),
        TerminalR1csError::Shape {
            what: "terminal public input width",
            expected: relation.public_carrier_width(),
            got: prep.public_input_len.unwrap_or(0),
        },
    )?;
    let expected_structure = relation.structure()?;
    guards.require(
        TerminalContextGuard::RelationStructure as u8,
        same_structure(prep.structure(), &expected_structure),
        TerminalR1csError::RelationMismatch,
    )?;
    Ok(TerminalContextGuardAudit {
        accepted_mask: guards.finish(TERMINAL_CONTEXT_GUARD_MASK)?,
    })
}

fn validate_public_statement(
    prep: &Preprocessing,
    relation: TerminalRelation<'_>,
    image: &PublicImage,
    running: &TerminalRunningStatement,
    fresh_claim: &CcsClaim,
) -> Result<TerminalStatementGuardAudit, TerminalR1csError> {
    let mut guards = TerminalGuardTracker::default();
    guards.require(
        TerminalStatementGuard::RunningClaimCount as u8,
        running.claims().len() == relation.running_claim_count(),
        TerminalR1csError::Shape {
            what: "terminal running claim count",
            expected: relation.running_claim_count(),
            got: running.claims().len(),
        },
    )?;
    guards.require(
        TerminalStatementGuard::VerifierKey as u8,
        image.vk_fs_digest == prep.vk.digest(),
        TerminalR1csError::InvalidState("public image verifier-key digest differs from preprocessing"),
    )?;
    guards.require(
        TerminalStatementGuard::InitialSemanticState as u8,
        image.initial_semantic_state_digest == prep.initial_semantic_state_digest(),
        TerminalR1csError::InvalidState("public image initial semantic state differs from preprocessing"),
    )?;
    let expected_z_0 = initial_boundary_digest(prep.structure_digest(), prep.public_input_len);
    guards.require(
        TerminalStatementGuard::InitialBoundary as u8,
        image.z_0 == expected_z_0,
        TerminalR1csError::InvalidState("public image initial boundary differs from preprocessing"),
    )?;
    guards.require(
        TerminalStatementGuard::ProgramCounter as u8,
        image.pc == construction2::TRIVIAL_PC,
        TerminalR1csError::InvalidState("public image program counter is outside the fixed-one profile"),
    )?;
    guards.require(
        TerminalStatementGuard::Counters as u8,
        image.chunk_count != 0 && image.step_count != 0 && image.chunk_count == image.step_count,
        TerminalR1csError::InvalidState("fixed-one terminal counters must be equal and nonzero"),
    )?;
    let start_index = image
        .step_count
        .checked_sub(1)
        .ok_or(TerminalR1csError::InvalidState("terminal step counter underflow"))?;
    let expected_boundary = digest_fields_as_digest32(construction2::f_prime_chunk_public_digest_from_claims(
        start_index,
        std::slice::from_ref(fresh_claim),
    ));
    guards.require(
        TerminalStatementGuard::FreshBoundary as u8,
        image.z_i == expected_boundary && image.public_trace == expected_boundary,
        TerminalR1csError::InvalidState("public image terminal chunk boundary is not the fresh claim boundary"),
    )?;
    let expected_acc = running
        .as_running()?
        .accumulator_digest(prep.params.b(), prep.structure())
        .map_err(|error| TerminalR1csError::Carrier(error.to_string()))?;
    guards.require(
        TerminalStatementGuard::RunningAccumulator as u8,
        image.acc_digest == expected_acc,
        TerminalR1csError::InvalidState("public image accumulator digest is not the running claim digest"),
    )?;
    guards.require(
        TerminalStatementGuard::SemanticState as u8,
        !matches!(prep.semantic_state_mode(), SemanticStateMode::Stateless)
            || image.semantic_state_digest == image.acc_digest,
        TerminalR1csError::InvalidState("stateless terminal semantic digest is not the accumulator digest"),
    )?;

    let state = state_from_public_image(image);
    let expected_x_out = construction2::compute_x_out(
        &prep.vk,
        &prep.params,
        prep.structure_digest(),
        &state,
        prep.semantic_state_mode(),
    );
    guards.require(
        TerminalStatementGuard::StateXOut as u8,
        image.x_out == expected_x_out,
        TerminalR1csError::InvalidState("public image x_out is not the Poseidon2 digest of its state"),
    )?;
    let layout = relation.public_input_layout();
    guards.require(
        TerminalStatementGuard::FreshPublicLink as u8,
        f_prime_public_input_link_matches(
            layout,
            &expected_x_out,
            relation.public_carrier_width(),
            fresh_claim.m_in,
            &fresh_claim.x,
        ),
        TerminalR1csError::InvalidState("fresh public input does not encode the terminal state x_out"),
    )?;
    Ok(TerminalStatementGuardAudit {
        accepted_mask: guards.finish(TERMINAL_STATEMENT_GUARD_MASK)?,
    })
}

/// Recompute every verifier-owned terminal context guard without compiling
/// the terminal R1CS or running Spartan.
pub fn audit_terminal_context_guards(
    prep: &Preprocessing,
    manifest: &LeanNativeCcsManifest,
) -> Result<TerminalContextGuardAudit, TerminalR1csError> {
    validate_context(prep, TerminalRelation::Native(manifest))
}

/// Recompute every combined-relation context guard without compiling the
/// terminal R1CS or running Spartan.
pub fn audit_combined_terminal_context_guards(
    setup: &LeanNebulaCombinedPreprocessing,
) -> Result<TerminalContextGuardAudit, TerminalR1csError> {
    validate_context(setup.preprocessing(), TerminalRelation::Combined(setup.manifest()))
}

/// Recompute every verifier-native terminal statement guard without running
/// the terminal R1CS or Spartan verifier.
pub fn audit_terminal_statement_guards(
    prep: &Preprocessing,
    manifest: &LeanNativeCcsManifest,
    statement: &TerminalSpartanStatement,
) -> Result<TerminalStatementGuardAudit, TerminalR1csError> {
    validate_public_statement(
        prep,
        TerminalRelation::Native(manifest),
        statement.public_image(),
        statement.running(),
        statement.fresh_claim(),
    )
}

/// Recompute every combined-relation statement guard without running the
/// terminal R1CS or Spartan verifier.
pub fn audit_combined_terminal_statement_guards(
    setup: &LeanNebulaCombinedPreprocessing,
    statement: &TerminalSpartanStatement,
) -> Result<TerminalStatementGuardAudit, TerminalR1csError> {
    validate_public_statement(
        setup.preprocessing(),
        TerminalRelation::Combined(setup.manifest()),
        statement.public_image(),
        statement.running(),
        statement.fresh_claim(),
    )
}

fn public_image_from_state(prep: &Preprocessing, state: &State) -> PublicImage {
    let x_out = construction2::compute_x_out(
        &prep.vk,
        &prep.params,
        prep.structure_digest(),
        state,
        prep.semantic_state_mode(),
    );
    PublicImage {
        vk_fs_digest: prep.vk.digest(),
        chunk_count: state.chunk_count,
        step_count: state.step_count,
        z_0: state.z_0,
        z_i: state.z_i,
        pc: state.pc,
        initial_semantic_state_digest: state.initial_semantic_state_digest,
        semantic_state_digest: state.semantic_state_digest,
        acc_digest: state.acc_digest,
        public_trace: state.public_trace,
        x_out,
    }
}

fn state_from_public_image(image: &PublicImage) -> State {
    State {
        chunk_count: image.chunk_count,
        step_count: image.step_count,
        z_0: image.z_0,
        z_i: image.z_i,
        pc: image.pc,
        initial_semantic_state_digest: image.initial_semantic_state_digest,
        semantic_state_digest: image.semantic_state_digest,
        acc_digest: image.acc_digest,
        public_trace: image.public_trace,
        proof: ProofState::Initial,
        nebula: None,
    }
}

fn same_structure(left: &Structure, right: &Structure) -> bool {
    if left.n != right.n
        || left.m != right.m
        || left.matrices.len() != right.matrices.len()
        || left.f.arity() != right.f.arity()
        || left.f.terms().len() != right.f.terms().len()
    {
        return false;
    }
    if left
        .f
        .terms()
        .iter()
        .zip(right.f.terms())
        .any(|(left, right)| left.coeff != right.coeff || left.exps != right.exps)
    {
        return false;
    }
    left.matrices.iter().zip(&right.matrices).all(same_matrix)
}

fn same_matrix(left: (&CcsMatrix<neo_math::F>, &CcsMatrix<neo_math::F>)) -> bool {
    let (left, right) = left;
    if left.rows() != right.rows() || left.cols() != right.cols() {
        return false;
    }
    (0..left.rows()).all(|row| left.materialize_row(row) == right.materialize_row(row))
}
