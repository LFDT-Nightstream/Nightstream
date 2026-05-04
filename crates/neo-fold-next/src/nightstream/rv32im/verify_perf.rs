//! Owns timing breakdowns for the published RV32IM Nightstream verifier path.

use std::time::Instant;

use crate::nightstream::NightstreamStatement;
use crate::rv32im::{Rv32imIvcSnarkVerifierKey, Rv32imProofStatement, SimpleKernelError};

use super::{
    rv32im_verifier_context_digest, verify_rv32im_nightstream_carried_boundary, Rv32imNightstreamProof,
    Rv32imSideBindingVerifierKey, Rv32imSideOpeningSpartanVerifierKey,
};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rv32imNightstreamVerifyPerf {
    pub carried_boundary_ms: f64,
    pub statement_binding_ms: f64,
    pub side_proof_ms: f64,
    pub remaining_side_surfaces_ms: f64,
    pub main_proof_ms: f64,
    pub total_ms: f64,
}

impl Rv32imNightstreamVerifyPerf {
    pub fn before_main_proof_ms(&self) -> f64 {
        self.total_ms - self.main_proof_ms
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

pub fn verify_rv32im_nightstream_with_perf(
    statement: &NightstreamStatement,
    proof: &Rv32imNightstreamProof,
    trusted_root_params_id: [u8; 32],
    ivc_recursion_snark_vk: &Rv32imIvcSnarkVerifierKey,
    side_opening_vk: &Rv32imSideOpeningSpartanVerifierKey,
    side_binding_vk: &Rv32imSideBindingVerifierKey,
    public_statement: &Rv32imProofStatement,
) -> Result<Rv32imNightstreamVerifyPerf, SimpleKernelError> {
    let total_started = Instant::now();

    let started = Instant::now();
    let expected_verifier_context_digest = rv32im_verifier_context_digest(
        trusted_root_params_id,
        proof.main_proof().published_statement(),
        ivc_recursion_snark_vk,
    )?;
    if statement.verifier_context_digest != expected_verifier_context_digest {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream verifier context digest does not match the trusted root parameters and IVC verifier key"
                .into(),
        ));
    }

    let expected_public_statement_digest = public_statement.recompute_digest();
    if public_statement.digest != expected_public_statement_digest {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream public statement digest does not match the carried public statement fields".into(),
        ));
    }
    if statement.public_io_digest != proof.main_proof().published_statement().expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream public IO digest does not match the carried published main statement".into(),
        ));
    }
    let statement_binding_ms = elapsed_ms(started);

    let started = Instant::now();
    verify_rv32im_nightstream_carried_boundary(statement, proof, expected_public_statement_digest)?;
    let carried_boundary_ms = elapsed_ms(started);

    let started = Instant::now();
    super::verify_rv32im_side_proof(
        side_opening_vk,
        side_binding_vk,
        statement,
        public_statement,
        &proof.side_proof,
    )?;
    let side_proof_ms = elapsed_ms(started);

    let remaining_side_surfaces_ms = 0.0;

    let started = Instant::now();
    let main_public_image = proof.main_proof().expected_ivc_public_image()?;
    proof
        .main_proof()
        .ivc_snark()
        .verify(ivc_recursion_snark_vk, &main_public_image)?;
    let main_proof_ms = elapsed_ms(started);

    Ok(Rv32imNightstreamVerifyPerf {
        carried_boundary_ms,
        statement_binding_ms,
        side_proof_ms,
        remaining_side_surfaces_ms,
        main_proof_ms,
        total_ms: elapsed_ms(total_started),
    })
}
