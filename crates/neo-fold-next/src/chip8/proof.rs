//! Owns CHIP-8 proof construction and local audit replay checks.
//! Does not own final proof acceptance; that belongs to a compressed decider relation.

use neo_ajtai::Commitment;
use neo_ccs::CcsWitness;
use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};

use crate::chip8::chunk_relation::Chip8ChunkReplayWitness;
use crate::chip8::final_relation::{
    audit_check_final_statement_with_output, audit_check_folded_statement_with_output, build_final_proof,
    build_recursive_proof, folded_statement_digest,
};
use crate::chip8::kernel::{
    build_kernel_bridge_binding_summary, build_kernel_exact_frames,
    build_kernel_export_proof_from_verified_execution_relation,
    build_kernel_export_relation_digest_from_verified_execution_relation, build_kernel_row_projection_summary,
    build_kernel_semantic_evidence_summary, prove_simple_kernel, verify_kernel_bridge_binding_summary,
    verify_kernel_execution_relation, verify_kernel_export_relation, verify_kernel_row_projection_summary,
    verify_kernel_semantic_evidence_summary, verify_simple_kernel, Chip8PreparedStepBridgeBinding,
    KernelBridgeBindingSummary, KernelExecutionRelationResult, KernelExecutionRelationWitness, KernelExportProof,
    KernelRowProjectionSummary, KernelSemanticEvidenceInputs, KernelSemanticEvidenceSummary, SimpleKernelError,
    SimpleKernelProof, SimpleKernelProverInput, SimpleKernelPublicInput, SimpleKernelRootContext,
    SimpleKernelVerifierInput, CHIP8_BRIDGE_FOLD_SCHEDULE, CHIP8_BRIDGE_ROWS_PER_CHUNK,
};
use crate::chip8::{Chip8State, Chip8VmSpec};
use crate::finalize::FixedShapeChunkSummary;
use crate::proof::FoldSchedule;
use crate::vm::VmSpec;

pub(crate) const CHIP8_MAIN_CARRY_WIDTH: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccumulatorHandle(pub [u8; 32]);

#[derive(Clone, Debug, PartialEq)]
pub struct Chip8AuditBundle {
    pub row_projection_summary: KernelRowProjectionSummary,
    pub bridge_binding_summary: KernelBridgeBindingSummary,
    pub semantic_evidence_summary: KernelSemanticEvidenceSummary,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Chip8PublicMachineState {
    pub pc_word: u16,
    pub registers: [u8; 16],
    pub i: u16,
    pub ram: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Chip8MainChunkTransitionWitness {
    pub step_witness_slots: [Option<CcsWitness<F>>; CHIP8_BRIDGE_ROWS_PER_CHUNK],
    pub replay_witness: Chip8ChunkReplayWitness,
}

#[derive(Clone, Debug)]
pub struct Chip8ChunkTransitionWitness {
    pub main_transition: Chip8MainChunkTransitionWitness,
    pub bridge_bindings: [Option<Chip8PreparedStepBridgeBinding>; CHIP8_BRIDGE_ROWS_PER_CHUNK],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Chip8RecursiveAccumulator {
    pub final_main_claims: [neo_ccs::CeClaim<Commitment, F, neo_math::K>; CHIP8_MAIN_CARRY_WIDTH],
    pub bridge_state: [u8; 32],
    pub terminal_handle: AccumulatorHandle,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Chip8FoldedStatement {
    pub fold_schedule: FoldSchedule,
    pub chunk_count: u64,
    pub semantic_step_count: u64,
    pub kernel_relation_digest: [u8; 32],
    pub final_accumulator: Chip8RecursiveAccumulator,
    pub digest: [u8; 32],
}

pub struct Chip8FoldedProof {
    pub kernel_export: KernelExportProof,
    pub steps: Vec<Chip8ChunkTransitionWitness>,
}

pub struct Chip8FinalProof {
    pub proof_digest: [u8; 32],
    pub kernel_export: KernelExportProof,
    pub chunk_summaries: Vec<FixedShapeChunkSummary>,
    pub steps: Vec<Chip8ChunkTransitionWitness>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Chip8Statement {
    pub public: SimpleKernelPublicInput,
    pub final_state: Chip8PublicMachineState,
    pub folded: Chip8FoldedStatement,
    pub digest: [u8; 32],
}

struct Chip8FoldedBuildOutput {
    final_state: Chip8PublicMachineState,
    folded: Chip8FoldedStatement,
    chunk_summaries: Vec<FixedShapeChunkSummary>,
    proof: Chip8FoldedProof,
}

struct BuiltKernelExport {
    relation_digest: [u8; 32],
    proof: KernelExportProof,
    verified_native: KernelExecutionRelationResult,
}

pub fn prove_kernel_export(
    input: &SimpleKernelProverInput,
) -> Result<([u8; 32], KernelExecutionRelationWitness), SimpleKernelError> {
    let (_output, native_proof) = prove_simple_kernel(input)?;
    let witness = split_kernel_export_witness(native_proof)?;
    let verifier_input = SimpleKernelVerifierInput {
        public: input.public.clone(),
    };
    let verified_relation = verify_kernel_execution_relation(&verifier_input, &witness)?;
    let relation_digest =
        build_kernel_export_relation_digest_from_verified_execution_relation(&verified_relation, &witness);
    Ok((relation_digest, witness))
}

pub fn verify_kernel_export(
    public: &SimpleKernelPublicInput,
    expected_schedule: FoldSchedule,
    relation_digest: [u8; 32],
    proof: &KernelExecutionRelationWitness,
) -> Result<(), SimpleKernelError> {
    verify_kernel_export_relation(public, expected_schedule, relation_digest, proof)?;
    Ok(())
}

pub fn prove_folded_statement(
    input: &SimpleKernelProverInput,
) -> Result<(Chip8FoldedStatement, Chip8FoldedProof), SimpleKernelError> {
    let built = build_folded_statement(input)?;
    Ok((built.folded, built.proof))
}

/// Local replay check for construction tests; not a final proof verifier.
pub fn audit_check_folded_statement_replay(
    public: &SimpleKernelPublicInput,
    folded: &Chip8FoldedStatement,
    proof: &Chip8FoldedProof,
) -> Result<(), SimpleKernelError> {
    audit_check_folded_statement_with_output(public, folded, proof)?;
    Ok(())
}

pub fn prove_final_statement(
    input: &SimpleKernelProverInput,
) -> Result<(Chip8FoldedStatement, Chip8FinalProof), SimpleKernelError> {
    let built = build_folded_statement(input)?;
    let final_proof = build_final_proof(&built.folded, built.chunk_summaries, built.proof)?;
    Ok((built.folded, final_proof))
}

/// Local replay check for construction tests; not a final proof verifier.
pub fn audit_check_final_statement_replay(
    public: &SimpleKernelPublicInput,
    folded: &Chip8FoldedStatement,
    proof: &Chip8FinalProof,
) -> Result<(), SimpleKernelError> {
    audit_check_final_statement_with_output(public, folded, proof)?;
    Ok(())
}

pub fn prove_audit(
    input: &SimpleKernelProverInput,
) -> Result<(Chip8AuditBundle, SimpleKernelProof), SimpleKernelError> {
    let (output, proof) = prove_simple_kernel(input)?;
    let audit = build_audit_bundle(&input.public, &proof, &output)?;
    Ok((audit, proof))
}

pub fn verify_audit(
    input: &SimpleKernelVerifierInput,
    proof: &SimpleKernelProof,
    audit: &Chip8AuditBundle,
) -> Result<(), SimpleKernelError> {
    let output = verify_simple_kernel(input, proof)?;
    let frames = build_kernel_exact_frames(&input.public, proof)?;
    let semantic_rows = frames.iter().map(|frame| frame.row).collect::<Vec<_>>();
    verify_kernel_row_projection_summary(
        &proof.kernel_opening_manifest,
        &proof.opening_refinement_summary,
        &proof.stage3.row_bindings,
        &semantic_rows,
        &audit.row_projection_summary,
    )?;
    verify_kernel_bridge_binding_summary(
        &proof.kernel_opening_manifest,
        &proof.opening_refinement_summary,
        &proof.stage3.row_bindings,
        &output.prepared_steps,
        &audit.bridge_binding_summary,
    )?;
    verify_kernel_semantic_evidence_summary(
        KernelSemanticEvidenceInputs {
            stage1: &proof.stage1,
            stage2: &proof.stage2,
            stage3: &proof.stage3,
            kernel_opening_manifest: &proof.kernel_opening_manifest,
            root_opening_manifest: &proof.root_opening_manifest,
            time_opening_summary: &proof.time_opening_summary,
            opening_refinement_summary: &proof.opening_refinement_summary,
            joint_opening_summary: &proof.joint_opening_summary,
            joint_opening_fold_bucket_proofs: &proof.joint_opening_fold_bucket_proofs,
            row_projection_summary: &audit.row_projection_summary,
            bridge_binding_summary: &audit.bridge_binding_summary,
        },
        &audit.semantic_evidence_summary,
    )?;
    Ok(())
}

pub fn prove_recursive(
    input: &SimpleKernelProverInput,
) -> Result<(Chip8Statement, Chip8FinalProof), SimpleKernelError> {
    let built = build_folded_statement(input)?;
    let final_proof = build_final_proof(&built.folded, built.chunk_summaries, built.proof)?;
    let mut statement = Chip8Statement {
        public: input.public.clone(),
        final_state: built.final_state,
        folded: built.folded,
        digest: [0; 32],
    };
    statement.digest = statement_digest(&statement);
    Ok((statement, final_proof))
}

/// Local replay check for construction tests; not a final proof verifier.
pub fn audit_check_recursive_artifact_replay(
    statement: &Chip8Statement,
    proof: &Chip8FinalProof,
) -> Result<(), SimpleKernelError> {
    if statement.folded.fold_schedule != CHIP8_BRIDGE_FOLD_SCHEDULE {
        return Err(SimpleKernelError::BridgeFailed(
            "folded statement schedule does not match frozen CHIP-8 export schedule".into(),
        ));
    }
    if statement.folded.final_accumulator.final_main_claims.len() != CHIP8_MAIN_CARRY_WIDTH {
        return Err(SimpleKernelError::BridgeFailed(
            "folded statement main-claim width mismatch".into(),
        ));
    }
    if statement.folded.digest != folded_statement_digest(&statement.folded) {
        return Err(SimpleKernelError::BridgeFailed(
            "folded statement digest mismatch".into(),
        ));
    }
    if statement.digest != statement_digest(statement) {
        return Err(SimpleKernelError::BridgeFailed("statement digest mismatch".into()));
    }
    let verified_kernel = audit_check_final_statement_with_output(&statement.public, &statement.folded, proof)?;
    if statement.final_state != public_machine_state(&verified_kernel.final_state) {
        return Err(SimpleKernelError::BridgeFailed(
            "final public machine state mismatch".into(),
        ));
    }
    Ok(())
}

fn build_audit_bundle(
    public: &SimpleKernelPublicInput,
    proof: &SimpleKernelProof,
    output: &crate::chip8::kernel::SimpleKernelOutput,
) -> Result<Chip8AuditBundle, SimpleKernelError> {
    let frames = build_kernel_exact_frames(public, proof)?;
    let semantic_rows = frames.iter().map(|frame| frame.row).collect::<Vec<_>>();
    let row_projection_summary = build_kernel_row_projection_summary(
        &proof.kernel_opening_manifest,
        &proof.opening_refinement_summary,
        &proof.stage3.row_bindings,
        &semantic_rows,
    )?;
    let bridge_binding_summary = build_kernel_bridge_binding_summary(
        &proof.kernel_opening_manifest,
        &proof.opening_refinement_summary,
        &proof.stage3.row_bindings,
        &output.prepared_steps,
    )?;
    let semantic_evidence_summary = build_kernel_semantic_evidence_summary(KernelSemanticEvidenceInputs {
        stage1: &proof.stage1,
        stage2: &proof.stage2,
        stage3: &proof.stage3,
        kernel_opening_manifest: &proof.kernel_opening_manifest,
        root_opening_manifest: &proof.root_opening_manifest,
        time_opening_summary: &proof.time_opening_summary,
        opening_refinement_summary: &proof.opening_refinement_summary,
        joint_opening_summary: &proof.joint_opening_summary,
        joint_opening_fold_bucket_proofs: &proof.joint_opening_fold_bucket_proofs,
        row_projection_summary: &row_projection_summary,
        bridge_binding_summary: &bridge_binding_summary,
    })?;
    Ok(Chip8AuditBundle {
        row_projection_summary,
        bridge_binding_summary,
        semantic_evidence_summary,
    })
}

fn build_folded_statement(input: &SimpleKernelProverInput) -> Result<Chip8FoldedBuildOutput, SimpleKernelError> {
    let (_output, native_kernel_proof) = prove_simple_kernel(input)?;
    let built_kernel_export = build_kernel_export(&input.public, native_kernel_proof)?;
    let kernel_relation_digest = built_kernel_export.relation_digest;
    let kernel_export = built_kernel_export.proof;
    let verified_kernel = built_kernel_export.verified_native;
    let final_state = public_machine_state(&verified_kernel.final_state);
    let vm = Chip8VmSpec::default();
    let root_context = SimpleKernelRootContext::new()?;
    let structure = &vm.core_ccs_spec().structure;
    let chunk_handoffs = verified_kernel.chunk_handoffs.clone();
    let (recursive, chunk_summaries, final_accumulator) =
        build_recursive_proof(&chunk_handoffs, root_context.params(), structure, root_context.log())?;
    let mut folded = Chip8FoldedStatement {
        fold_schedule: CHIP8_BRIDGE_FOLD_SCHEDULE,
        chunk_count: chunk_handoffs.len() as u64,
        semantic_step_count: chunk_handoffs
            .iter()
            .map(|handoff| handoff.chunk_input.steps.len() as u64)
            .sum(),
        kernel_relation_digest,
        final_accumulator,
        digest: [0; 32],
    };
    folded.digest = folded_statement_digest(&folded);
    Ok(Chip8FoldedBuildOutput {
        final_state,
        folded,
        chunk_summaries,
        proof: Chip8FoldedProof {
            kernel_export,
            steps: recursive,
        },
    })
}

fn split_kernel_export_witness(
    native_proof: SimpleKernelProof,
) -> Result<KernelExecutionRelationWitness, SimpleKernelError> {
    KernelExecutionRelationWitness::from_simple_kernel_proof(native_proof)
}

fn build_kernel_export(
    public: &SimpleKernelPublicInput,
    native_proof: SimpleKernelProof,
) -> Result<BuiltKernelExport, SimpleKernelError> {
    let witness = split_kernel_export_witness(native_proof)?;
    let verifier_input = SimpleKernelVerifierInput { public: public.clone() };
    let verified_native = verify_kernel_execution_relation(&verifier_input, &witness)?;
    let (proof, _verified_compact, relation_digest) =
        build_kernel_export_proof_from_verified_execution_relation(public, &verified_native);
    Ok(BuiltKernelExport {
        relation_digest,
        proof,
        verified_native,
    })
}

fn public_machine_state(state: &Chip8State) -> Chip8PublicMachineState {
    Chip8PublicMachineState {
        pc_word: state.pc / 2,
        registers: state.v,
        i: state.i,
        ram: state.memory.to_vec(),
    }
}

pub(crate) fn statement_digest(statement: &Chip8Statement) -> [u8; 32] {
    let public_io_digest = public_io_digest(&statement.public, &statement.final_state);
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/chip8/statement");
    tr.append_message(b"neo.fold.next/chip8/statement/public_io_digest", &public_io_digest);
    tr.append_message(b"neo.fold.next/chip8/statement/folded_digest", &statement.folded.digest);
    tr.digest32()
}

pub(crate) fn public_io_digest(public: &SimpleKernelPublicInput, final_state: &Chip8PublicMachineState) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/chip8/public_io");
    append_public_input(&mut tr, public);
    append_public_machine_state(&mut tr, final_state);
    tr.digest32()
}

fn append_public_input(tr: &mut Poseidon2Transcript, public: &SimpleKernelPublicInput) {
    tr.append_u64s(
        b"neo.fold.next/chip8/public_input/meta",
        &[
            public.program_image.len() as u64,
            public.initial_pc_word as u64,
            public.initial_i as u64,
            public.initial_ram.len() as u64,
            public.transcript_seed.len() as u64,
        ],
    );
    tr.append_message(b"neo.fold.next/chip8/public_input/program", &public.program_image);
    tr.append_message(b"neo.fold.next/chip8/public_input/registers", &public.initial_registers);
    tr.append_message(b"neo.fold.next/chip8/public_input/ram", &public.initial_ram);
    tr.append_message(
        b"neo.fold.next/chip8/public_input/transcript_seed",
        &public.transcript_seed,
    );
}

fn append_public_machine_state(tr: &mut Poseidon2Transcript, state: &Chip8PublicMachineState) {
    tr.append_u64s(
        b"neo.fold.next/chip8/public_machine_state/meta",
        &[state.pc_word as u64, state.i as u64, state.ram.len() as u64],
    );
    tr.append_message(b"neo.fold.next/chip8/public_machine_state/registers", &state.registers);
    tr.append_message(b"neo.fold.next/chip8/public_machine_state/ram", &state.ram);
}
