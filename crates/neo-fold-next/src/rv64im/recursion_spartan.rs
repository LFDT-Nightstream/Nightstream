//! Owns the current terminal recursion relation above the recursive-step proof-chain backend.
//!
//! This module owns the theorem-facing closure checks that bind the published
//! RV64IM accumulator statement to the current recursion proof carrier. It does
//! not own recursion-proof packaging or the recursive-step backend circuit.

use crate::chunk_relation::ChunkReplayWitness;
use crate::rv64im::chunk_fold_step::{Rv64imChunkFoldCarry, Rv64imChunkFoldFresh, Rv64imChunkStepPublic};
use crate::rv64im::chunk_step_ivc::{
    build_rv64im_chunk_step_ivc_relations, rv64im_bridge_handoff_chain_digest_from_digests,
    rv64im_bridge_handoff_chain_digest_init, rv64im_chunk_step_ivc_initial_state,
    rv64im_recursion_step_statement_chain_digest, rv64im_step_statement_chain_digest_init,
};
use crate::rv64im::final_relation::{
    build_rv64im_terminal_chunk_fold_witness, rv64im_chunk_fold_carry_recursive_accumulator_digest,
    verify_rv64im_final_statement_with_output, verify_rv64im_terminal_chunk_fold_witness,
    Rv64imChunkFoldTranscriptSnapshot, Rv64imFinalBuildProof, Rv64imFinalStatement, Rv64imRecursiveAccumulator,
    Rv64imTerminalChunkFoldWitness,
};
use crate::rv64im::kernel::Rv64imVerifiedKernelChunkHandoff;
use crate::rv64im::main_proof::Rv64imAccumulatorPublicStatement;
use crate::rv64im::main_recursion::{
    build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs, build_rv64im_main_recursion_f_prime_advices,
    build_rv64im_main_recursion_verifier_key_fs, Rv64imEncodedPublicInput, Rv64imMainRecursionFPrimeAdvice,
    Rv64imVerifierKeyFs,
};
use crate::rv64im::main_relation_spartan::{
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    prove_rv64im_main_recursion_step_spartan_chain, validate_rv64im_main_recursion_step_spartan_chain_shape,
    verify_rv64im_main_recursion_step_spartan_chain_and_extract_published_targets,
    Rv64imMainRecursionStepSpartanChainProof, Rv64imMainRecursionStepSpartanPublishedTarget,
    Rv64imMainRecursionStepSpartanShape, Rv64imMainRecursionStepSpartanStatement,
};
use crate::rv64im::recursion_shape::{build_rv64im_recursion_shape, RecursionShape};
use crate::rv64im::SimpleKernelError;
use neo_transcript::{Poseidon2Transcript, Transcript};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct Rv64imRecursionBackendProof {
    final_public_image: Rv64imMainRecursionFinalRelationPublicImage,
    spartan_shape: Rv64imMainRecursionStepSpartanShape,
    chain_proof: Rv64imMainRecursionStepSpartanChainProof,
}

pub(crate) fn audit_empty_step_proof_chain() -> Rv64imMainRecursionStepSpartanChainProof {
    Vec::new()
}

fn extracted_backend_statement_from_chain_with_vk_fs(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    chain_proof: &Rv64imMainRecursionStepSpartanChainProof,
    vk_fs: &Rv64imVerifierKeyFs,
) -> Result<Rv64imMainRecursionStepSpartanStatement, SimpleKernelError> {
    let published_targets =
        verify_rv64im_main_recursion_step_spartan_chain_and_extract_published_targets(spartan_shape, chain_proof)
            .map_err(|err| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM main recursion fixed-step proof-chain verify failed: {err}"
                ))
            })?;
    if let Some(last_target) = published_targets.last() {
        Ok(last_target.output_statement())
    } else {
        let seed_state = rv64im_chunk_step_ivc_initial_state();
        Ok(build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs(
            vk_fs,
            0,
            rv64im_chunk_fold_carry_recursive_accumulator_digest(&seed_state.carry),
            rv64im_step_statement_chain_digest_init(),
            rv64im_bridge_handoff_chain_digest_init(),
            seed_state.carry.terminal_handle.0,
        ))
    }
}

impl Rv64imRecursionBackendProof {
    fn proof_step_count(&self) -> u64 {
        self.chain_proof.len() as u64
    }

    fn extracted_published_targets(
        &self,
    ) -> Result<Vec<Rv64imMainRecursionStepSpartanPublishedTarget>, SimpleKernelError> {
        verify_rv64im_main_recursion_step_spartan_chain_and_extract_published_targets(
            self.spartan_shape(),
            &self.chain_proof,
        )
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM main recursion fixed-step proof-chain verify failed: {err}"
            ))
        })
    }

    fn extracted_backend_statement_with_vk_fs(
        &self,
        vk_fs: &Rv64imVerifierKeyFs,
    ) -> Result<Rv64imMainRecursionStepSpartanStatement, SimpleKernelError> {
        extracted_backend_statement_from_chain_with_vk_fs(self.spartan_shape(), &self.chain_proof, vk_fs)
    }

    fn extracted_public_image_with_vk_fs(
        &self,
        vk_fs: &Rv64imVerifierKeyFs,
    ) -> Result<Rv64imMainRecursionFinalRelationPublicImage, SimpleKernelError> {
        Ok(Rv64imMainRecursionFinalRelationPublicImage::from_backend_statement(
            &self.extracted_backend_statement_with_vk_fs(vk_fs)?,
        ))
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_proof");
        tr.append_message(b"neo.fold.next/rv64im/main_recursion_proof/version", b"v17");
        let total_step_bytes: u64 = self
            .chain_proof
            .iter()
            .map(|proof| proof.snark_data.len() as u64)
            .sum();
        tr.append_u64s(
            b"neo.fold.next/rv64im/main_recursion_proof/meta",
            &[self.chain_proof.len() as u64, total_step_bytes],
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_proof/spartan_shape_digest",
            &self.spartan_shape.expected_digest(),
        );
        for step_proof in &self.chain_proof {
            tr.append_message(
                b"neo.fold.next/rv64im/main_recursion_proof/step_proof",
                &step_proof.snark_data,
            );
        }
        tr.digest32()
    }

    pub fn final_public_image_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_final_public_image");
        tr.append_message(b"neo.fold.next/rv64im/main_recursion_final_public_image/version", b"v1");
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_final_public_image/public_image_digest",
            &self.final_public_image.expected_digest(),
        );
        tr.digest32()
    }

    fn spartan_shape(&self) -> &Rv64imMainRecursionStepSpartanShape {
        &self.spartan_shape
    }

    pub(crate) fn first_step_proof_snark_bytes_mut(&mut self) -> &mut Vec<u8> {
        if self.chain_proof.is_empty() {
            self.chain_proof.push(
                crate::rv64im::main_relation_spartan::Rv64imMainRecursionStepSpartanProof { snark_data: Vec::new() },
            );
        }
        &mut self.chain_proof[0].snark_data
    }

    pub(crate) fn x_last_bytes_mut(&mut self) -> &mut [u8; 32] {
        self.final_public_image.x_last.bytes_mut()
    }

    pub(crate) fn pop_last_step_proof(&mut self) -> Result<(), SimpleKernelError> {
        self.chain_proof.pop().ok_or_else(|| {
            SimpleKernelError::Bridge("RV64IM main recursion proof chain does not contain any step proofs".into())
        })?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Rv64imRecursionProof {
    backend_proof: Rv64imRecursionBackendProof,
}

impl Rv64imRecursionProof {
    pub fn expected_digest(&self) -> [u8; 32] {
        self.backend_proof.expected_digest()
    }

    pub fn final_public_image_digest(&self) -> [u8; 32] {
        self.backend_proof.final_public_image_digest()
    }

    pub(crate) fn first_step_proof_snark_bytes_mut(&mut self) -> &mut Vec<u8> {
        self.backend_proof.first_step_proof_snark_bytes_mut()
    }

    pub(crate) fn x_last_bytes_mut(&mut self) -> &mut [u8; 32] {
        self.backend_proof.x_last_bytes_mut()
    }

    pub(crate) fn pop_last_step_proof(&mut self) -> Result<(), SimpleKernelError> {
        self.backend_proof.pop_last_step_proof()
    }
}

pub(crate) fn build_rv64im_recursion_proof_from_parts(
    spartan_shape: Rv64imMainRecursionStepSpartanShape,
    chain_proof: Rv64imMainRecursionStepSpartanChainProof,
) -> Result<Rv64imRecursionProof, SimpleKernelError> {
    let vk_fs = build_rv64im_main_recursion_verifier_key_fs()?;
    let backend_proof = Rv64imRecursionBackendProof {
        final_public_image: Rv64imMainRecursionFinalRelationPublicImage::from_backend_statement(
            &extracted_backend_statement_from_chain_with_vk_fs(&spartan_shape, &chain_proof, &vk_fs)?,
        ),
        spartan_shape,
        chain_proof,
    };
    Ok(Rv64imRecursionProof { backend_proof })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv64imRecursionProverKey {
    pub shape: RecursionShape,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv64imRecursionVerifierKey {
    pub shape_digest: [u8; 32],
    pub vk_fs: Rv64imVerifierKeyFs,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Rv64imMainRecursionFinalRelationPublicImage {
    x_last: Rv64imEncodedPublicInput,
    folded_accumulator_digest: [u8; 32],
    terminal_handle_digest: [u8; 32],
}

impl Rv64imMainRecursionFinalRelationPublicImage {
    fn from_backend_statement(backend_statement: &Rv64imMainRecursionStepSpartanStatement) -> Self {
        Self {
            x_last: backend_statement.x_out.clone(),
            folded_accumulator_digest: backend_statement.folded_accumulator_digest,
            terminal_handle_digest: backend_statement.terminal_handle_digest,
        }
    }

    fn from_recursion_proof(proof: &Rv64imRecursionProof) -> Result<Self, SimpleKernelError> {
        let vk_fs = build_rv64im_main_recursion_verifier_key_fs()?;
        proof
            .backend_proof
            .extracted_public_image_with_vk_fs(&vk_fs)
    }

    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_final_public_image");
        tr.append_message(b"neo.fold.next/rv64im/main_recursion_final_public_image/version", b"v1");
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_final_public_image/x_last",
            &self.x_last.bytes(),
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_final_public_image/folded_accumulator_digest",
            &self.folded_accumulator_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_final_public_image/terminal_handle_digest",
            &self.terminal_handle_digest,
        );
        tr.digest32()
    }

    fn from_accumulator_witness_with_vk_fs(
        vk_fs: &Rv64imVerifierKeyFs,
        accumulator_witness: &Rv64imMainRecursionAccumulatorWitness,
    ) -> Result<Self, SimpleKernelError> {
        let backend_statement = accumulator_witness.backend_statement_with_vk_fs(vk_fs)?;
        Ok(Self {
            x_last: backend_statement.x_out,
            folded_accumulator_digest: backend_statement.folded_accumulator_digest,
            terminal_handle_digest: backend_statement.terminal_handle_digest,
        })
    }

    fn validate_against_published_statement(
        &self,
        published_statement: &Rv64imAccumulatorPublicStatement,
    ) -> Result<(), SimpleKernelError> {
        if &self.x_last != published_statement.x_last() {
            return Err(SimpleKernelError::Bridge(
                "RV64IM main recursion published x_last does not match the canonical final-relation public image"
                    .into(),
            ));
        }
        if self.folded_accumulator_digest != published_statement.canonical_folded_accumulator_digest() {
            return Err(SimpleKernelError::Bridge(
                "RV64IM main recursion published final accumulator does not match the canonical final-relation public image"
                    .into(),
            ));
        }
        if self.terminal_handle_digest != published_statement.canonical_terminal_handle_digest() {
            return Err(SimpleKernelError::Bridge(
                "RV64IM main recursion published terminal handle does not match the canonical final-relation public image"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rv64imMainRecursionFinalRelationStatement {
    shape_digest: [u8; 32],
    vk_fs: Rv64imVerifierKeyFs,
    chunk_count: u64,
    pc_final: u64,
    accumulator_final: Rv64imRecursiveAccumulator,
    x_last: Rv64imEncodedPublicInput,
}

impl Rv64imMainRecursionFinalRelationStatement {
    fn from_published_statement(
        published_statement: &Rv64imAccumulatorPublicStatement,
    ) -> Result<Self, SimpleKernelError> {
        published_statement.validate()?;
        Ok(Self {
            shape_digest: published_statement.shape_digest(),
            vk_fs: published_statement.vk_fs().clone(),
            chunk_count: published_statement.expected_chunk_count()?,
            pc_final: published_statement.pc_final(),
            accumulator_final: published_statement.accumulator_final().clone(),
            x_last: published_statement.x_last().clone(),
        })
    }

    fn canonical_public_image(&self) -> Rv64imMainRecursionFinalRelationPublicImage {
        Rv64imMainRecursionFinalRelationPublicImage {
            x_last: self.x_last.clone(),
            folded_accumulator_digest: rv64im_chunk_fold_carry_recursive_accumulator_digest(&Rv64imChunkFoldCarry {
                main: crate::proof::Carry {
                    claims: self.accumulator_final.final_main_claims.clone(),
                    witnesses: Vec::new(),
                },
                terminal_handle: self.accumulator_final.terminal_handle,
            }),
            terminal_handle_digest: self.accumulator_final.terminal_handle.0,
        }
    }

    pub fn shape_digest(&self) -> [u8; 32] {
        self.shape_digest
    }

    pub fn vk_fs(&self) -> &Rv64imVerifierKeyFs {
        &self.vk_fs
    }

    pub fn chunk_count(&self) -> u64 {
        self.chunk_count
    }

    pub fn pc_final(&self) -> u64 {
        self.pc_final
    }

    pub fn accumulator_final(&self) -> &Rv64imRecursiveAccumulator {
        &self.accumulator_final
    }

    pub fn x_last(&self) -> &Rv64imEncodedPublicInput {
        &self.x_last
    }
}

#[derive(Clone, Debug)]
pub struct Rv64imMainRecursionAccumulatorWitness {
    public_statement_digest: [u8; 32],
    handoff: Rv64imVerifiedKernelChunkHandoff,
    running_last: Rv64imChunkFoldCarry,
    transcript_in: Rv64imChunkFoldTranscriptSnapshot,
    fresh_last: Rv64imChunkFoldFresh,
    final_fold_witness: ChunkReplayWitness,
    running_final: Rv64imChunkFoldCarry,
    transcript_out: Rv64imChunkFoldTranscriptSnapshot,
    step_public: Rv64imChunkStepPublic,
    halted_out: bool,
    chunk_count: u64,
    pc_final: u64,
    step_statement_chain_digest: [u8; 32],
    bridge_handoff_chain_digest: [u8; 32],
}

impl Rv64imMainRecursionAccumulatorWitness {
    fn terminal_chunk_fold_witness(&self) -> Rv64imTerminalChunkFoldWitness {
        Rv64imTerminalChunkFoldWitness {
            public_statement_digest: self.public_statement_digest,
            handoff: self.handoff.clone(),
            running_last: self.running_last.clone(),
            transcript_in: self.transcript_in.clone(),
            fresh_last: self.fresh_last.clone(),
            final_fold_witness: self.final_fold_witness.clone(),
            running_final: self.running_final.clone(),
            transcript_out: self.transcript_out.clone(),
            step_public: self.step_public.clone(),
            halted_out: self.halted_out,
        }
    }

    pub fn public_statement_digest(&self) -> [u8; 32] {
        self.public_statement_digest
    }

    pub fn handoff(&self) -> &Rv64imVerifiedKernelChunkHandoff {
        &self.handoff
    }

    pub fn running_last(&self) -> &Rv64imChunkFoldCarry {
        &self.running_last
    }

    pub fn transcript_in(&self) -> &Rv64imChunkFoldTranscriptSnapshot {
        &self.transcript_in
    }

    pub fn fresh_last(&self) -> &Rv64imChunkFoldFresh {
        &self.fresh_last
    }

    pub fn final_fold_witness(&self) -> &ChunkReplayWitness {
        &self.final_fold_witness
    }

    pub(crate) fn final_fold_witness_mut(&mut self) -> &mut ChunkReplayWitness {
        &mut self.final_fold_witness
    }

    pub fn running_final(&self) -> &Rv64imChunkFoldCarry {
        &self.running_final
    }

    pub(crate) fn running_final_mut(&mut self) -> &mut Rv64imChunkFoldCarry {
        &mut self.running_final
    }

    pub fn transcript_out(&self) -> &Rv64imChunkFoldTranscriptSnapshot {
        &self.transcript_out
    }

    pub fn step_public(&self) -> &Rv64imChunkStepPublic {
        &self.step_public
    }

    pub fn halted_out(&self) -> bool {
        self.halted_out
    }

    pub fn chunk_count(&self) -> u64 {
        self.chunk_count
    }

    pub fn pc_final(&self) -> u64 {
        self.pc_final
    }

    pub fn step_statement_chain_digest(&self) -> [u8; 32] {
        self.step_statement_chain_digest
    }

    pub fn bridge_handoff_chain_digest(&self) -> [u8; 32] {
        self.bridge_handoff_chain_digest
    }

    pub fn accumulator_final(&self) -> Rv64imRecursiveAccumulator {
        self.terminal_chunk_fold_witness().accumulator_final()
    }

    fn backend_statement_with_vk_fs(
        &self,
        vk_fs: &Rv64imVerifierKeyFs,
    ) -> Result<Rv64imMainRecursionStepSpartanStatement, SimpleKernelError> {
        Ok(build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs(
            vk_fs,
            self.chunk_count,
            rv64im_chunk_fold_carry_recursive_accumulator_digest(self.running_final()),
            self.step_statement_chain_digest,
            self.bridge_handoff_chain_digest,
            self.running_final().terminal_handle.0,
        ))
    }
}

impl Rv64imRecursionBackendProof {
    fn verify_against_final_relation_statement(
        &self,
        final_relation_statement: &Rv64imMainRecursionFinalRelationStatement,
    ) -> Result<(), SimpleKernelError> {
        validate_rv64im_main_recursion_final_relation_surface(final_relation_statement, self)?;
        let expected_public_image = self.extracted_public_image_with_vk_fs(final_relation_statement.vk_fs())?;
        if self.final_public_image != expected_public_image {
            return Err(SimpleKernelError::Bridge(
                "RV64IM main recursion proof carrier final public image does not match the final published target"
                    .into(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn build_rv64im_main_recursion_x_last_from_accumulator_with_vk_fs(
    vk_fs: &Rv64imVerifierKeyFs,
    chunk_count: u64,
    accumulator_final: &Rv64imRecursiveAccumulator,
    step_statement_chain_digest: [u8; 32],
    bridge_handoff_chain_digest: [u8; 32],
) -> Result<Rv64imEncodedPublicInput, SimpleKernelError> {
    let folded_accumulator_digest = rv64im_chunk_fold_carry_recursive_accumulator_digest(&Rv64imChunkFoldCarry {
        main: crate::proof::Carry {
            claims: accumulator_final.final_main_claims.clone(),
            witnesses: Vec::new(),
        },
        terminal_handle: accumulator_final.terminal_handle,
    });
    Ok(build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs(
        vk_fs,
        chunk_count,
        folded_accumulator_digest,
        step_statement_chain_digest,
        bridge_handoff_chain_digest,
        accumulator_final.terminal_handle.0,
    )
    .x_out)
}

pub fn build_rv64im_main_recursion_accumulator_witness(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<Rv64imMainRecursionAccumulatorWitness, SimpleKernelError> {
    let verified_kernel = verify_rv64im_final_statement_with_output(statement, proof)?;
    let relations = build_rv64im_chunk_step_ivc_relations(statement, proof)?;
    let step_statement_chain_digest = rv64im_recursion_step_statement_chain_digest(&relations);
    let bridge_handoff_chain_digest = rv64im_bridge_handoff_chain_digest_from_digests(
        &relations
            .iter()
            .map(|relation| relation.witness.handoff.bridge_handoff.digest)
            .collect::<Vec<_>>(),
    );
    let terminal_chunk = build_rv64im_terminal_chunk_fold_witness(statement, proof)?;
    let Rv64imTerminalChunkFoldWitness {
        public_statement_digest,
        handoff,
        running_last,
        transcript_in,
        fresh_last,
        final_fold_witness,
        running_final,
        transcript_out,
        step_public,
        halted_out,
    } = terminal_chunk;
    Ok(Rv64imMainRecursionAccumulatorWitness {
        public_statement_digest,
        handoff,
        running_last,
        transcript_in,
        fresh_last,
        final_fold_witness,
        running_final,
        transcript_out,
        step_public,
        halted_out,
        chunk_count: relations.len() as u64,
        pc_final: verified_kernel.final_pc,
        step_statement_chain_digest,
        bridge_handoff_chain_digest,
    })
}

pub fn prove_rv64im_recursion_proof(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<Rv64imRecursionProof, SimpleKernelError> {
    let relations = build_rv64im_chunk_step_ivc_relations(statement, proof)?;
    let advices = build_rv64im_main_recursion_f_prime_advices(&relations)?;
    prove_rv64im_recursion_proof_from_advices(&relations, &advices)
}

pub fn setup_rv64im_recursion() -> Result<(Rv64imRecursionProverKey, Rv64imRecursionVerifierKey), SimpleKernelError> {
    let shape = build_rv64im_recursion_shape()?;
    let shape_digest = shape.canonical_digest();
    let vk_fs = build_rv64im_main_recursion_verifier_key_fs()?;
    Ok((
        Rv64imRecursionProverKey { shape },
        Rv64imRecursionVerifierKey { shape_digest, vk_fs },
    ))
}

pub(crate) fn prove_rv64im_recursion_proof_from_advices(
    relations: &[crate::rv64im::chunk_step_ivc::Rv64imChunkStepIvcRelation],
    advices: &[Rv64imMainRecursionFPrimeAdvice],
) -> Result<Rv64imRecursionProof, SimpleKernelError> {
    let (spartan_shape, backend_relations) =
        build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(relations, advices)
            .map_err(|err| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM recursion proof recursive-step relation build failed: {err}"
                ))
            })?;
    validate_rv64im_main_recursion_step_spartan_chain_shape(&spartan_shape, &backend_relations).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM recursion proof shared step-shape validation failed: {err}"
        ))
    })?;
    let chain_proof = prove_rv64im_main_recursion_step_spartan_chain(&spartan_shape, &backend_relations)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM recursion prove failed: {err}")))?;
    build_rv64im_recursion_proof_from_parts(spartan_shape, chain_proof)
}

fn validate_rv64im_main_recursion_final_relation_surface(
    final_relation_statement: &Rv64imMainRecursionFinalRelationStatement,
    backend_proof: &Rv64imRecursionBackendProof,
) -> Result<(), SimpleKernelError> {
    let expected_vk_fs = build_rv64im_main_recursion_verifier_key_fs()?;
    if *final_relation_statement.vk_fs() != expected_vk_fs {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion published statement verifier key fs does not match the canonical recursion verifier key"
                .into(),
        ));
    }
    if final_relation_statement.vk_fs().main_lane_shape_digest != final_relation_statement.shape_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion published statement shape_digest does not match the canonical recursion shape"
                .into(),
        ));
    }
    let chain_shape_chunk_count = backend_proof.proof_step_count();
    if chain_shape_chunk_count != final_relation_statement.chunk_count() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion proof chain step count does not match the published statement chunk schedule".into(),
        ));
    }
    if backend_proof.extracted_public_image_with_vk_fs(final_relation_statement.vk_fs())?
        != final_relation_statement.canonical_public_image()
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion proof surface does not match the canonical final-relation statement public image"
                .into(),
        ));
    }
    Ok(())
}

pub fn build_rv64im_main_recursion_final_relation_statement(
    published_statement: &Rv64imAccumulatorPublicStatement,
) -> Result<Rv64imMainRecursionFinalRelationStatement, SimpleKernelError> {
    Rv64imMainRecursionFinalRelationStatement::from_published_statement(published_statement)
}

pub(crate) fn verify_rv64im_recursion_proof_against_final_relation_statement(
    final_relation_statement: &Rv64imMainRecursionFinalRelationStatement,
    main_recursion_proof: &Rv64imRecursionProof,
) -> Result<(), SimpleKernelError> {
    main_recursion_proof
        .backend_proof
        .verify_against_final_relation_statement(final_relation_statement)
}

pub fn verify_rv64im_recursion(
    recursion_vk: &Rv64imRecursionVerifierKey,
    published_statement: &Rv64imAccumulatorPublicStatement,
    main_recursion_proof: &Rv64imRecursionProof,
) -> Result<(), SimpleKernelError> {
    let final_relation_statement = build_rv64im_main_recursion_final_relation_statement(published_statement)?;
    validate_rv64im_recursion_verifier_key_against_final_relation_statement(recursion_vk, &final_relation_statement)?;
    verify_rv64im_recursion_proof_against_final_relation_statement(&final_relation_statement, main_recursion_proof)
}

pub(crate) fn validate_rv64im_recursion_verifier_key_against_final_relation_statement(
    recursion_vk: &Rv64imRecursionVerifierKey,
    final_relation_statement: &Rv64imMainRecursionFinalRelationStatement,
) -> Result<(), SimpleKernelError> {
    if recursion_vk.shape_digest != final_relation_statement.shape_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM recursion verifier key shape_digest does not match the published statement".into(),
        ));
    }
    if recursion_vk.vk_fs != *final_relation_statement.vk_fs() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM recursion verifier key fs does not match the published statement".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_rv64im_recursion_verifier_key_against_published_statement(
    recursion_vk: &Rv64imRecursionVerifierKey,
    published_statement: &Rv64imAccumulatorPublicStatement,
) -> Result<(), SimpleKernelError> {
    let final_relation_statement = build_rv64im_main_recursion_final_relation_statement(published_statement)?;
    validate_rv64im_recursion_verifier_key_against_final_relation_statement(recursion_vk, &final_relation_statement)
}

pub(crate) fn validate_rv64im_main_recursion_public_surface_against_published_statement(
    published_statement: &Rv64imAccumulatorPublicStatement,
    main_recursion_proof: &Rv64imRecursionProof,
) -> Result<(), SimpleKernelError> {
    let final_relation_statement = build_rv64im_main_recursion_final_relation_statement(published_statement)?;
    validate_rv64im_main_recursion_final_relation_surface(
        &final_relation_statement,
        &main_recursion_proof.backend_proof,
    )
}

pub fn verify_rv64im_main_recursion_final_relation_native_against_statement(
    final_relation_statement: &Rv64imMainRecursionFinalRelationStatement,
    accumulator_witness: &Rv64imMainRecursionAccumulatorWitness,
) -> Result<(), SimpleKernelError> {
    verify_rv64im_terminal_chunk_fold_witness(&accumulator_witness.terminal_chunk_fold_witness())?;
    if accumulator_witness.chunk_count() != final_relation_statement.chunk_count() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion accumulator witness chunk count does not match the published statement schedule"
                .into(),
        ));
    }
    if accumulator_witness.pc_final() != final_relation_statement.pc_final() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion accumulator witness pc_final does not match the published statement".into(),
        ));
    }
    if accumulator_witness.step_public().chunk_index + 1 != final_relation_statement.chunk_count() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion final relation terminal chunk index does not close the published statement chunk schedule"
                .into(),
        ));
    }
    if !accumulator_witness.halted_out() || !accumulator_witness.step_public().halted_out {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion accumulator witness does not close on a terminal halted chunk".into(),
        ));
    }
    if accumulator_witness.accumulator_final() != *final_relation_statement.accumulator_final() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion accumulator witness final accumulator does not match the published statement".into(),
        ));
    }
    if Rv64imMainRecursionFinalRelationPublicImage::from_accumulator_witness_with_vk_fs(
        final_relation_statement.vk_fs(),
        accumulator_witness,
    )? != final_relation_statement.canonical_public_image()
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion accumulator witness does not match the canonical final-relation statement public image"
                .into(),
        ));
    }
    if accumulator_witness.step_public().halted_out != accumulator_witness.halted_out() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion accumulator witness halted flag does not match the terminal step public".into(),
        ));
    }
    Ok(())
}

pub fn verify_rv64im_main_recursion_final_relation_native(
    published_statement: &Rv64imAccumulatorPublicStatement,
    accumulator_witness: &Rv64imMainRecursionAccumulatorWitness,
) -> Result<(), SimpleKernelError> {
    let final_relation_statement = build_rv64im_main_recursion_final_relation_statement(published_statement)?;
    verify_rv64im_main_recursion_final_relation_native_against_statement(&final_relation_statement, accumulator_witness)
}

pub fn verify_rv64im_main_recursion_accumulator_witness(
    published_statement: &Rv64imAccumulatorPublicStatement,
    accumulator_witness: &Rv64imMainRecursionAccumulatorWitness,
) -> Result<(), SimpleKernelError> {
    verify_rv64im_main_recursion_final_relation_native(published_statement, accumulator_witness)
}

pub(crate) fn audit_rv64im_main_recursion_final_relation_public_images_match(
    published_statement: &Rv64imAccumulatorPublicStatement,
    accumulator_witness: &Rv64imMainRecursionAccumulatorWitness,
    recursion_proof: &Rv64imRecursionProof,
) -> Result<(), SimpleKernelError> {
    let witness_image = Rv64imMainRecursionFinalRelationPublicImage::from_accumulator_witness_with_vk_fs(
        published_statement.vk_fs(),
        accumulator_witness,
    )?;
    witness_image.validate_against_published_statement(published_statement)?;
    let proof_image = Rv64imMainRecursionFinalRelationPublicImage::from_recursion_proof(recursion_proof)?;
    proof_image.validate_against_published_statement(published_statement)?;
    if witness_image != proof_image {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion proof surface does not match the native terminal accumulator witness public image"
                .into(),
        ));
    }
    Ok(())
}

pub(crate) fn audit_rv64im_main_recursion_terminal_published_target_matches_native_witness(
    published_statement: &Rv64imAccumulatorPublicStatement,
    accumulator_witness: &Rv64imMainRecursionAccumulatorWitness,
    recursion_proof: &Rv64imRecursionProof,
) -> Result<(), SimpleKernelError> {
    let final_relation_statement = build_rv64im_main_recursion_final_relation_statement(published_statement)?;
    let published_targets = recursion_proof
        .backend_proof
        .extracted_published_targets()?;
    let last_target = published_targets.last().ok_or_else(|| {
        SimpleKernelError::Bridge(
            "RV64IM main recursion terminal published-target parity requires a non-empty recursion proof chain".into(),
        )
    })?;

    let expected_backend_statement =
        accumulator_witness.backend_statement_with_vk_fs(final_relation_statement.vk_fs())?;
    if last_target.output_statement() != expected_backend_statement {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion terminal published target did not match the native terminal accumulator witness backend statement"
                .into(),
        ));
    }

    let expected_x_last = build_rv64im_main_recursion_x_last_from_accumulator_with_vk_fs(
        final_relation_statement.vk_fs(),
        accumulator_witness.chunk_count(),
        &accumulator_witness.accumulator_final(),
        accumulator_witness.step_statement_chain_digest(),
        accumulator_witness.bridge_handoff_chain_digest(),
    )?;
    if last_target.x_out != expected_x_last {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion terminal published target x_last did not match the canonical terminal Construction-2 state image"
                .into(),
        ));
    }
    if &last_target.x_out != final_relation_statement.x_last() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion terminal published target x_last did not match the published terminal relation statement"
                .into(),
        ));
    }
    if last_target.chunk_index + 1 != accumulator_witness.chunk_count() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion terminal published target chunk index did not close the native terminal witness schedule"
                .into(),
        ));
    }
    if last_target.folded_accumulator_out_digest
        != rv64im_chunk_fold_carry_recursive_accumulator_digest(accumulator_witness.running_final())
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion terminal published target folded accumulator did not match the native terminal witness"
                .into(),
        ));
    }
    if last_target.step_statement_chain_digest != accumulator_witness.step_statement_chain_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion terminal published target step-statement chain digest did not match the native terminal witness"
                .into(),
        ));
    }
    if last_target.bridge_handoff_chain_digest != accumulator_witness.bridge_handoff_chain_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion terminal published target bridge-handoff chain digest did not match the native terminal witness"
                .into(),
        ));
    }
    if last_target.terminal_handle_digest != accumulator_witness.running_final().terminal_handle.0 {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion terminal published target terminal handle did not match the native terminal witness"
                .into(),
        ));
    }

    Ok(())
}
