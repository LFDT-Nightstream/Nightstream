//! Spartan/WHIR terminal lifecycle for the Lean-owned native CCS relation.
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

use super::{
    compile_terminal_r1cs, compile_terminal_r1cs_statement, LeanNativeCcsManifest, TerminalR1csError,
    TerminalR1csInput, TerminalR1csStatement, TerminalSpartanEngine,
};

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
    validate_context(prep, manifest)?;
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
    validate_public_statement(prep, manifest, &public_image, &running_statement, &fresh.claim)?;

    let relation = compile_terminal_r1cs(
        manifest,
        &prep.log,
        TerminalR1csInput {
            running_claims: &running.claims,
            running_witnesses: &running.witnesses,
            fresh,
        },
    )?;
    let (shape, witness, public_values) = relation.into_parts();
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
    validate_context(prep, manifest)?;
    if statement.public_image() != expected {
        return Err(TerminalR1csError::PublicImageMismatch);
    }
    validate_public_statement(
        prep,
        manifest,
        statement.public_image(),
        statement.running(),
        statement.fresh_claim(),
    )?;

    let relation = compile_terminal_r1cs_statement(
        manifest,
        &prep.log,
        TerminalR1csStatement {
            running_claims: statement.running().claims(),
            fresh_claim: statement.fresh_claim(),
        },
    )?;
    let (shape, expected_public) = relation.into_parts();
    let (_, verifier_key) = R1CSSNARK::<TerminalSpartanEngine>::setup_direct(shape)
        .map_err(|error| TerminalR1csError::Spartan(error.to_string()))?;
    let verified_public = proof
        .snark
        .verify(&verifier_key)
        .map_err(|error| TerminalR1csError::Spartan(error.to_string()))?;
    if verified_public != expected_public {
        return Err(TerminalR1csError::PublicStatementMismatch);
    }
    Ok(())
}

fn validate_context(prep: &Preprocessing, manifest: &LeanNativeCcsManifest) -> Result<(), TerminalR1csError> {
    if !prep.enforces_terminal_induction() {
        return Err(TerminalR1csError::UncertifiedInduction);
    }
    if prep.nebula().is_some() {
        return Err(TerminalR1csError::Unsupported(
            "Nebula needs its delayed terminal memory finalization",
        ));
    }
    if prep.public_input_len != Some(manifest.public_carrier_width()) {
        return Err(TerminalR1csError::Shape {
            what: "terminal public input width",
            expected: manifest.public_carrier_width(),
            got: prep.public_input_len.unwrap_or(0),
        });
    }
    let emitted = manifest
        .emit_phi81_step(|_| Some(neo_math::F::ZERO))
        .map_err(|error| TerminalR1csError::Manifest(error.to_string()))?;
    if !same_structure(prep.structure(), emitted.structure()) {
        return Err(TerminalR1csError::RelationMismatch);
    }
    Ok(())
}

fn validate_public_statement(
    prep: &Preprocessing,
    manifest: &LeanNativeCcsManifest,
    image: &PublicImage,
    running: &TerminalRunningStatement,
    fresh_claim: &CcsClaim,
) -> Result<(), TerminalR1csError> {
    if running.claims().len() != manifest.running_claim_count() {
        return Err(TerminalR1csError::Shape {
            what: "terminal running claim count",
            expected: manifest.running_claim_count(),
            got: running.claims().len(),
        });
    }
    if image.vk_fs_digest != prep.vk.digest() {
        return Err(TerminalR1csError::InvalidState(
            "public image verifier-key digest differs from preprocessing",
        ));
    }
    if image.initial_semantic_state_digest != prep.initial_semantic_state_digest() {
        return Err(TerminalR1csError::InvalidState(
            "public image initial semantic state differs from preprocessing",
        ));
    }
    let expected_z_0 = initial_boundary_digest(prep.structure_digest(), prep.public_input_len);
    if image.z_0 != expected_z_0 {
        return Err(TerminalR1csError::InvalidState(
            "public image initial boundary differs from preprocessing",
        ));
    }
    if image.pc != construction2::TRIVIAL_PC {
        return Err(TerminalR1csError::InvalidState(
            "public image program counter is outside the fixed-one profile",
        ));
    }
    if image.chunk_count == 0 || image.step_count == 0 || image.chunk_count != image.step_count {
        return Err(TerminalR1csError::InvalidState(
            "fixed-one terminal counters must be equal and nonzero",
        ));
    }
    let start_index = image
        .step_count
        .checked_sub(1)
        .ok_or(TerminalR1csError::InvalidState("terminal step counter underflow"))?;
    let expected_boundary = digest_fields_as_digest32(construction2::f_prime_chunk_public_digest_from_claims(
        start_index,
        std::slice::from_ref(fresh_claim),
    ));
    if image.z_i != expected_boundary || image.public_trace != expected_boundary {
        return Err(TerminalR1csError::InvalidState(
            "public image terminal chunk boundary is not the fresh claim boundary",
        ));
    }
    let expected_acc = running
        .as_running()?
        .accumulator_digest(prep.params.b(), prep.structure())
        .map_err(|error| TerminalR1csError::Carrier(error.to_string()))?;
    if image.acc_digest != expected_acc {
        return Err(TerminalR1csError::InvalidState(
            "public image accumulator digest is not the running claim digest",
        ));
    }
    if matches!(prep.semantic_state_mode(), SemanticStateMode::Stateless)
        && image.semantic_state_digest != image.acc_digest
    {
        return Err(TerminalR1csError::InvalidState(
            "stateless terminal semantic digest is not the accumulator digest",
        ));
    }

    let state = state_from_public_image(image);
    let expected_x_out = construction2::compute_x_out(
        &prep.vk,
        &prep.params,
        prep.structure_digest(),
        &state,
        prep.semantic_state_mode(),
    );
    if image.x_out != expected_x_out {
        return Err(TerminalR1csError::InvalidState(
            "public image x_out is not the Poseidon2 digest of its state",
        ));
    }
    let layout = FPrimePublicInputLayout::plain();
    if !f_prime_public_input_link_matches(
        layout,
        &expected_x_out,
        manifest.public_carrier_width(),
        fresh_claim.m_in,
        &fresh_claim.x,
    ) {
        return Err(TerminalR1csError::InvalidState(
            "fresh public input does not encode the terminal state x_out",
        ));
    }
    Ok(())
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
