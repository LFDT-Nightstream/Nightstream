//! Owns timing breakdowns for the published RV64IM Nightstream verifier path.

use std::time::Instant;

use crate::nightstream::NightstreamStatement;
use crate::rv64im::{Rv64imProofStatement, SimpleKernelError};

use super::{
    rv64im_verifier_context_digest, verify_rv64im_main_proof, verify_rv64im_nightstream_carried_boundary,
    Rv64imNightstreamProof, Rv64imSideBindingVerifierKey, Rv64imSideOpeningSpartanVerifierKey,
};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rv64imNightstreamVerifyPerf {
    pub carried_boundary_ms: f64,
    pub statement_binding_ms: f64,
    pub side_proof_ms: f64,
    pub remaining_side_surfaces_ms: f64,
    pub main_proof_ms: f64,
    pub total_ms: f64,
}

impl Rv64imNightstreamVerifyPerf {
    pub fn before_main_proof_ms(&self) -> f64 {
        self.total_ms - self.main_proof_ms
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

pub fn verify_rv64im_nightstream_with_perf(
    statement: &NightstreamStatement,
    proof: &Rv64imNightstreamProof,
    trusted_root_params_id: [u8; 32],
    side_opening_vk: &Rv64imSideOpeningSpartanVerifierKey,
    side_binding_vk: &Rv64imSideBindingVerifierKey,
    public_statement: &Rv64imProofStatement,
) -> Result<Rv64imNightstreamVerifyPerf, SimpleKernelError> {
    let total_started = Instant::now();

    let started = Instant::now();
    verify_rv64im_nightstream_carried_boundary(statement, proof)?;
    let carried_boundary_ms = elapsed_ms(started);

    let started = Instant::now();
    let expected_verifier_context_digest = rv64im_verifier_context_digest(trusted_root_params_id);
    if statement.verifier_context_digest != expected_verifier_context_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream verifier context digest does not match the trusted root parameters".into(),
        ));
    }

    let expected_public_statement_digest = public_statement.recompute_digest();
    if public_statement.digest != expected_public_statement_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream public statement digest does not match the carried public statement fields".into(),
        ));
    }
    if statement.public_io_digest != proof.main_proof().published_statement().expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream public IO digest does not match the carried published main statement".into(),
        ));
    }
    let statement_binding_ms = elapsed_ms(started);

    let started = Instant::now();
    super::verify_rv64im_side_proof(
        side_opening_vk,
        side_binding_vk,
        statement,
        public_statement,
        &proof.side_proof,
    )?;
    let side_proof_ms = elapsed_ms(started);

    let remaining_side_surfaces_ms = 0.0;

    let started = Instant::now();
    verify_rv64im_main_proof(&proof.main_proof)?;
    let main_proof_ms = elapsed_ms(started);

    Ok(Rv64imNightstreamVerifyPerf {
        carried_boundary_ms,
        statement_binding_ms,
        side_proof_ms,
        remaining_side_surfaces_ms,
        main_proof_ms,
        total_ms: elapsed_ms(total_started),
    })
}
