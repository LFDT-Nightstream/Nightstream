//! Owns the CHIP-8 published Nightstream boundary above the current recursive/final seam.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::chip8::decider::build_chip8_spartan2_decider_target;
use crate::chip8::final_relation::{final_proof_component_digests, folded_statement_digest};
use crate::chip8::kernel::{SimpleKernelError, CHIP8_BRIDGE_FOLD_SCHEDULE, CHIP8_BRIDGE_ROWS_PER_CHUNK};
use crate::chip8::proof::{
    public_io_digest, statement_digest, verify_recursive, Chip8FinalProof, Chip8Statement, CHIP8_MAIN_CARRY_WIDTH,
};
use crate::nightstream::{nightstream_proof_binding_root, NightstreamProofBindingInputs, NightstreamStatement};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chip8MainDeciderProof {
    pub decider_target_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chip8MainResidualProof {
    pub statement_digest: [u8; 32],
    pub folded_statement_digest: [u8; 32],
    pub final_proof_digest: [u8; 32],
    pub kernel_export_proof_digest: [u8; 32],
    pub chunk_transition_digests: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chip8NightstreamProof {
    pub main_decider_proof: Chip8MainDeciderProof,
    pub main_residual_proof: Chip8MainResidualProof,
}

impl Chip8MainDeciderProof {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/chip8/main_decider_proof");
        tr.append_message(b"neo.fold.next/nightstream/chip8/main_decider_proof/version", b"v1");
        tr.append_message(
            b"neo.fold.next/nightstream/chip8/main_decider_proof/decider_target_digest",
            &self.decider_target_digest,
        );
        tr.digest32()
    }
}

impl Chip8MainResidualProof {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/chip8/main_residual_proof");
        tr.append_message(b"neo.fold.next/nightstream/chip8/main_residual_proof/version", b"v1");
        tr.append_message(
            b"neo.fold.next/nightstream/chip8/main_residual_proof/statement_digest",
            &self.statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/chip8/main_residual_proof/folded_statement_digest",
            &self.folded_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/chip8/main_residual_proof/final_proof_digest",
            &self.final_proof_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/chip8/main_residual_proof/kernel_export_proof_digest",
            &self.kernel_export_proof_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/chip8/main_residual_proof/chunk_transition_count",
            &[self.chunk_transition_digests.len() as u64],
        );
        for digest in &self.chunk_transition_digests {
            tr.append_message(
                b"neo.fold.next/nightstream/chip8/main_residual_proof/chunk_transition_digest",
                digest,
            );
        }
        tr.digest32()
    }
}

fn chip8_main_proof_digest(
    main_decider_proof: &Chip8MainDeciderProof,
    main_residual_proof: &Chip8MainResidualProof,
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/chip8/main_proof");
    tr.append_message(b"neo.fold.next/nightstream/chip8/main_proof/version", b"v1");
    tr.append_message(
        b"neo.fold.next/nightstream/chip8/main_proof/main_decider_proof_digest",
        &main_decider_proof.expected_digest(),
    );
    tr.append_message(
        b"neo.fold.next/nightstream/chip8/main_proof/main_residual_proof_digest",
        &main_residual_proof.expected_digest(),
    );
    tr.digest32()
}

pub fn chip8_verifier_context_digest() -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/chip8/verifier_context");
    tr.append_message(b"neo.fold.next/nightstream/chip8/verifier_context/version", b"v1");
    tr.append_u64s(
        b"neo.fold.next/nightstream/chip8/verifier_context/fold_schedule",
        &CHIP8_BRIDGE_FOLD_SCHEDULE.meta_words(),
    );
    tr.append_u64s(
        b"neo.fold.next/nightstream/chip8/verifier_context/layout",
        &[CHIP8_BRIDGE_ROWS_PER_CHUNK as u64, CHIP8_MAIN_CARRY_WIDTH as u64],
    );
    tr.digest32()
}

fn chip8_absent_artifact_digest(label: &'static [u8]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/chip8/absent_artifact");
    tr.append_message(b"neo.fold.next/nightstream/chip8/absent_artifact/version", b"v1");
    tr.append_message(b"neo.fold.next/nightstream/chip8/absent_artifact/label", label);
    tr.digest32()
}

pub fn chip8_absent_side_proof_digest() -> [u8; 32] {
    chip8_absent_artifact_digest(b"side_proof")
}

pub fn chip8_absent_linkage_binding_digest() -> [u8; 32] {
    chip8_absent_artifact_digest(b"linkage_binding")
}

pub fn chip8_nightstream_linkage_root(kernel_export_anchor_digest: [u8; 32]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/chip8/linkage_root");
    tr.append_message(b"neo.fold.next/nightstream/chip8/linkage_root/version", b"v1");
    tr.append_message(
        b"neo.fold.next/nightstream/chip8/linkage_root/kernel_export_anchor_digest",
        &kernel_export_anchor_digest,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/chip8/linkage_root/linkage_binding_digest",
        &chip8_absent_linkage_binding_digest(),
    );
    tr.digest32()
}

pub fn build_chip8_main_decider_proof(
    statement: &Chip8Statement,
    proof: &Chip8FinalProof,
) -> Result<Chip8MainDeciderProof, SimpleKernelError> {
    Ok(Chip8MainDeciderProof {
        decider_target_digest: build_chip8_spartan2_decider_target(statement, proof)?.digest(),
    })
}

pub fn verify_chip8_main_decider_proof(
    statement: &Chip8Statement,
    proof: &Chip8FinalProof,
    main_decider_proof: &Chip8MainDeciderProof,
) -> Result<(), SimpleKernelError> {
    let expected = build_chip8_main_decider_proof(statement, proof)?;
    if &expected != main_decider_proof {
        return Err(SimpleKernelError::BridgeFailed(
            "CHIP-8 Nightstream main decider proof does not match the verified decider target".into(),
        ));
    }
    Ok(())
}

pub fn build_chip8_main_residual_proof(
    statement: &Chip8Statement,
    proof: &Chip8FinalProof,
) -> Result<Chip8MainResidualProof, SimpleKernelError> {
    verify_recursive(statement, proof)?;
    let component_digests = final_proof_component_digests(proof);
    Ok(Chip8MainResidualProof {
        statement_digest: statement.digest,
        folded_statement_digest: statement.folded.digest,
        final_proof_digest: proof.proof_digest,
        kernel_export_proof_digest: component_digests.kernel_export_digest,
        chunk_transition_digests: component_digests.chunk_transition_digests,
    })
}

pub fn verify_chip8_main_residual_proof(
    statement: &Chip8Statement,
    proof: &Chip8FinalProof,
    residual: &Chip8MainResidualProof,
) -> Result<(), SimpleKernelError> {
    let expected = build_chip8_main_residual_proof(statement, proof)?;
    if &expected != residual {
        return Err(SimpleKernelError::BridgeFailed(
            "CHIP-8 Nightstream main residual proof does not match the carried final proof seam".into(),
        ));
    }
    Ok(())
}

pub fn build_chip8_nightstream_statement(
    statement: &Chip8Statement,
    proof: &Chip8FinalProof,
    proof_binding_root: [u8; 32],
) -> Result<NightstreamStatement, SimpleKernelError> {
    verify_recursive(statement, proof)?;
    Ok(NightstreamStatement {
        public_io_digest: public_io_digest(&statement.public, &statement.final_state),
        verifier_context_digest: chip8_verifier_context_digest(),
        fold_schedule: statement.folded.fold_schedule,
        semantic_step_count: statement.folded.semantic_step_count,
        chunk_summaries: proof.chunk_summaries.clone(),
        linkage_root: chip8_nightstream_linkage_root(proof.kernel_export.digest),
        proof_binding_root,
    })
}

pub fn build_chip8_nightstream_from_recursive_proof(
    statement: &Chip8Statement,
    proof: &Chip8FinalProof,
) -> Result<(NightstreamStatement, Chip8NightstreamProof), SimpleKernelError> {
    let main_decider_proof = build_chip8_main_decider_proof(statement, proof)?;
    let main_residual_proof = build_chip8_main_residual_proof(statement, proof)?;
    let mut nightstream_statement = build_chip8_nightstream_statement(statement, proof, [0; 32])?;
    let proof_binding_inputs = NightstreamProofBindingInputs {
        main_proof_digest: chip8_main_proof_digest(&main_decider_proof, &main_residual_proof),
        side_proof_digest: chip8_absent_side_proof_digest(),
        linkage_binding_digest: chip8_absent_linkage_binding_digest(),
    };
    nightstream_statement.proof_binding_root =
        nightstream_proof_binding_root(nightstream_statement.core_digest(), &proof_binding_inputs);
    Ok((
        nightstream_statement,
        Chip8NightstreamProof {
            main_decider_proof,
            main_residual_proof,
        },
    ))
}

pub fn verify_chip8_nightstream_from_recursive_proof(
    recursive_statement: &Chip8Statement,
    final_proof: &Chip8FinalProof,
    nightstream_statement: &NightstreamStatement,
    nightstream_proof: &Chip8NightstreamProof,
) -> Result<(), SimpleKernelError> {
    if recursive_statement.folded.digest != folded_statement_digest(&recursive_statement.folded) {
        return Err(SimpleKernelError::BridgeFailed(
            "CHIP-8 folded statement digest mismatch".into(),
        ));
    }
    if recursive_statement.digest != statement_digest(recursive_statement) {
        return Err(SimpleKernelError::BridgeFailed(
            "CHIP-8 statement digest mismatch".into(),
        ));
    }
    verify_chip8_main_decider_proof(recursive_statement, final_proof, &nightstream_proof.main_decider_proof)?;
    verify_chip8_main_residual_proof(recursive_statement, final_proof, &nightstream_proof.main_residual_proof)?;
    let expected = build_chip8_nightstream_from_recursive_proof(recursive_statement, final_proof)?;
    if &expected.0 != nightstream_statement {
        return Err(SimpleKernelError::BridgeFailed(
            "CHIP-8 Nightstream statement does not match the verified recursive seam".into(),
        ));
    }
    if &expected.1 != nightstream_proof {
        return Err(SimpleKernelError::BridgeFailed(
            "CHIP-8 Nightstream proof does not match the verified recursive seam".into(),
        ));
    }
    Ok(())
}
