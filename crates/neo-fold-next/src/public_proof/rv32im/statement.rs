//! Builds and verifies the RV32IM Nightstream statement boundary.

use neo_transcript::{Poseidon2Transcript, Transcript};

use crate::public_proof::{nightstream_proof_binding_root, NightstreamProofBindingInputs, NightstreamStatement};
use crate::rv32im::final_relation::{
    audit_check_rv32im_final_statement_with_output, Rv32imFinalBuildProof, Rv32imFinalStatement,
};
use crate::rv32im::kernel::SimpleKernelError;
use crate::rv32im::main_proof::Rv32imPublishedStatement;
use crate::rv32im::Rv32imIvcSnarkVerifierKey;

use super::proof::{rv32im_main_nightstream_proof_digest, Rv32imNightstreamProof};

pub fn rv32im_verifier_context_digest(
    root_params_id: [u8; 32],
    published_statement: &Rv32imPublishedStatement,
    ivc_recursion_snark_vk: &Rv32imIvcSnarkVerifierKey,
) -> Result<[u8; 32], SimpleKernelError> {
    Ok(rv32im_verifier_context_digest_from_key_digest(
        root_params_id,
        published_statement,
        ivc_recursion_snark_vk.expected_digest()?,
    ))
}

fn rv32im_verifier_context_digest_from_key_digest(
    root_params_id: [u8; 32],
    published_statement: &Rv32imPublishedStatement,
    ivc_recursion_snark_vk_digest: [u8; 32],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/verifier_context");
    tr.append_message(b"neo.fold.next/nightstream/rv32im/verifier_context/version", b"v3");
    tr.append_message(
        b"neo.fold.next/nightstream/rv32im/verifier_context/root_params_id",
        &root_params_id,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/rv32im/verifier_context/main_recursion_vk_fs",
        &published_statement.vk_fs().expected_digest(),
    );
    tr.append_message(
        b"neo.fold.next/nightstream/rv32im/verifier_context/main_recursion_shape",
        &published_statement.shape_digest(),
    );
    tr.append_message(
        b"neo.fold.next/nightstream/rv32im/verifier_context/ivc_recursion_snark_vk",
        &ivc_recursion_snark_vk_digest,
    );
    tr.digest32()
}

pub fn build_rv32im_nightstream_statement_from_final(
    public_io_digest: [u8; 32],
    verifier_context_digest: [u8; 32],
    statement: &Rv32imFinalStatement,
    proof: &Rv32imFinalBuildProof,
    proof_binding_root: [u8; 32],
) -> Result<NightstreamStatement, SimpleKernelError> {
    audit_check_rv32im_final_statement_with_output(statement, proof)?;
    Ok(NightstreamStatement {
        public_io_digest,
        verifier_context_digest,
        fold_schedule: statement.folded.fold_schedule,
        semantic_step_count: statement.folded.semantic_step_count,
        proof_binding_root,
    })
}

pub fn build_rv32im_nightstream_statement_from_published_statement(
    verifier_context_digest: [u8; 32],
    published_statement: &Rv32imPublishedStatement,
    proof_binding_root: [u8; 32],
) -> Result<NightstreamStatement, SimpleKernelError> {
    Ok(NightstreamStatement {
        public_io_digest: published_statement.expected_digest(),
        verifier_context_digest,
        fold_schedule: published_statement.fold_schedule(),
        semantic_step_count: published_statement.step_count(),
        proof_binding_root,
    })
}

pub(super) fn verify_rv32im_nightstream_carried_boundary(
    statement: &NightstreamStatement,
    proof: &Rv32imNightstreamProof,
    public_statement_digest: [u8; 32],
) -> Result<(), SimpleKernelError> {
    let mut expected_statement = build_rv32im_nightstream_statement_from_published_statement(
        statement.verifier_context_digest,
        proof.main_proof().published_statement(),
        [0; 32],
    )?;
    let proof_binding_inputs = NightstreamProofBindingInputs {
        main_proof_digest: rv32im_main_nightstream_proof_digest(proof.main_proof()),
        side_proof_digest: proof.side_proof().expected_digest(),
        public_statement_digest,
    };
    expected_statement.proof_binding_root =
        nightstream_proof_binding_root(expected_statement.core_digest(), &proof_binding_inputs);
    if &expected_statement != statement {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream statement does not match the verified proof boundary".into(),
        ));
    }
    Ok(())
}
