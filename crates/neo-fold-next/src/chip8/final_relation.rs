//! Owns the folded/final CHIP-8 relation replay, accumulator replay, and proof-binding digests.

use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, CcsWitness, Mat};
use neo_math::{KExtensions, F};
use neo_params::NeoParams;
use neo_reductions::engines::utils::me_digest_poseidon_into;
use neo_reductions::optimized_engine::OptimizedStructureCache;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

use crate::chip8::chunk_relation::{
    chip8_chunk_relation_digest, public_chunk_digest, step_handle, synthesize_chip8_chunk_relation_artifacts,
    verify_chip8_chunk_relation_with_witness, Chip8ChunkReplayWitness, Chip8ReplayRoundWitness,
};
use crate::chip8::kernel::{
    chip8_bridge_state_seed, verify_kernel_export_proof, KernelExportChunkHandoff, KernelExportProof,
    KernelExportRelationResult, SimpleKernelError, SimpleKernelRootContext, VerifiedKernelChunkHandoff,
    CHIP8_BRIDGE_FOLD_SCHEDULE, CHIP8_BRIDGE_ROWS_PER_CHUNK,
};
use crate::chip8::proof::{
    AccumulatorHandle, Chip8ChunkTransitionWitness, Chip8FinalProof, Chip8FoldedProof, Chip8FoldedStatement,
    Chip8MainChunkTransitionWitness, Chip8RecursiveAccumulator, CHIP8_MAIN_CARRY_WIDTH,
};
use crate::chip8::Chip8VmSpec;
use crate::chunk_relation::{
    compute_chunk_replay_witness_and_relation_with_perf, ChunkReplayWitness, CommitmentMixers,
};
use crate::finalize::{digest_fixed_shape_final_proof, fixed_shape_recursive_seed, FixedShapeChunkSummary};
use crate::proof::{Carry, ChunkInput, PublicChunk, StepInput};
use crate::vm::VmSpec;

struct Chip8RecursiveAccumulatorState {
    main: Carry,
    bridge_state: [u8; 32],
    terminal_handle: AccumulatorHandle,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Chip8FinalProofComponentDigests {
    pub kernel_export_digest: [u8; 32],
    pub chunk_transition_digests: Vec<[u8; 32]>,
}

impl Chip8RecursiveAccumulatorState {
    fn seed() -> Self {
        Self {
            main: Carry::default(),
            bridge_state: chip8_bridge_state_seed(),
            terminal_handle: AccumulatorHandle(recursive_seed()),
        }
    }

    fn into_public(self) -> Result<Chip8RecursiveAccumulator, SimpleKernelError> {
        Ok(Chip8RecursiveAccumulator {
            final_main_claims: chip8_main_carry_claims(self.main.claims)?,
            bridge_state: self.bridge_state,
            terminal_handle: self.terminal_handle,
        })
    }
}

pub(crate) fn build_recursive_proof<L>(
    chunk_handoffs: &[VerifiedKernelChunkHandoff],
    params: &NeoParams,
    structure: &CcsStructure<F>,
    log: &L,
) -> Result<
    (
        Vec<Chip8ChunkTransitionWitness>,
        Vec<FixedShapeChunkSummary>,
        Chip8RecursiveAccumulator,
    ),
    SimpleKernelError,
>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
{
    let mut accumulator = Chip8RecursiveAccumulatorState::seed();
    let mut transcript = Poseidon2Transcript::new(b"neo.fold.next/session");
    let optimized_cache = OptimizedStructureCache::build(structure).map_err(run_error)?;
    let mut steps = Vec::with_capacity(chunk_handoffs.len());
    let mut chunk_summaries = Vec::with_capacity(chunk_handoffs.len());
    for (chunk_index, handoff) in chunk_handoffs.iter().enumerate() {
        let chunk_input = &handoff.chunk_input;
        let public_chunk = &handoff.public_chunk;
        let ((replay_witness, proved), _perf) = compute_chunk_replay_witness_and_relation_with_perf(
            &mut transcript,
            params,
            structure,
            chunk_input,
            &accumulator.main,
            log,
            ajtai_mixers(),
            &optimized_cache,
        )
        .map_err(run_error)?;
        let main_transition = chunk_transition_main_witness(params, structure, chunk_input, replay_witness)?;
        let relation_artifacts = synthesize_chip8_chunk_relation_artifacts(
            public_chunk,
            proved.artifacts,
            handoff.bridge_handoff.witness_digest,
        );
        if handoff.bridge_handoff.previous_state != accumulator.bridge_state {
            return Err(SimpleKernelError::BridgeFailed(format!(
                "verified kernel handoff {} bridge previous state mismatch",
                chunk_index
            )));
        }
        let chunk_relation_digest = chip8_chunk_relation_digest(public_chunk, &relation_artifacts);
        let next_handle = AccumulatorHandle(step_handle(
            accumulator.terminal_handle.0,
            chunk_index,
            public_chunk.start_index,
            public_chunk.steps.len(),
            chunk_relation_digest,
        ));
        chunk_summaries.push(FixedShapeChunkSummary::from_public_chunk(
            public_chunk,
            public_chunk_digest(public_chunk),
            chunk_relation_digest,
        ));
        let step = Chip8ChunkTransitionWitness {
            main_transition,
            bridge_bindings: handoff.bridge_handoff.step_bindings.clone(),
        };
        accumulator = Chip8RecursiveAccumulatorState {
            main: proved.next_main,
            bridge_state: handoff.bridge_handoff.next_state,
            terminal_handle: next_handle,
        };
        steps.push(step);
        transcript.append_message(b"neo.fold.next/chunk_done", &[1]);
    }
    Ok((steps, chunk_summaries, accumulator.into_public()?))
}

pub(crate) fn audit_check_folded_statement_with_output(
    public: &crate::chip8::kernel::SimpleKernelPublicInput,
    folded: &Chip8FoldedStatement,
    proof: &Chip8FoldedProof,
) -> Result<KernelExportRelationResult, SimpleKernelError> {
    let (verified_kernel, _) =
        audit_check_folded_statement_components_with_output(public, folded, &proof.kernel_export, &proof.steps)?;
    Ok(verified_kernel)
}

pub(crate) fn audit_check_final_statement_with_output(
    public: &crate::chip8::kernel::SimpleKernelPublicInput,
    folded: &Chip8FoldedStatement,
    proof: &Chip8FinalProof,
) -> Result<KernelExportRelationResult, SimpleKernelError> {
    if proof.proof_digest != final_proof_digest(folded, &proof.kernel_export, &proof.chunk_summaries, &proof.steps) {
        return Err(SimpleKernelError::BridgeFailed("final proof digest mismatch".into()));
    }
    let (verified_kernel, expected_chunk_summaries) =
        audit_check_folded_statement_components_with_output(public, folded, &proof.kernel_export, &proof.steps)?;
    if proof.chunk_summaries != expected_chunk_summaries {
        return Err(SimpleKernelError::BridgeFailed(
            "final proof chunk summaries do not match the checked CHIP-8 recursive boundary".into(),
        ));
    }
    Ok(verified_kernel)
}

pub(crate) fn build_final_proof(
    folded: &Chip8FoldedStatement,
    chunk_summaries: Vec<FixedShapeChunkSummary>,
    proof: Chip8FoldedProof,
) -> Result<Chip8FinalProof, SimpleKernelError> {
    let proof_digest = final_proof_digest(folded, &proof.kernel_export, &chunk_summaries, &proof.steps);
    Ok(Chip8FinalProof {
        proof_digest,
        kernel_export: proof.kernel_export,
        chunk_summaries,
        steps: proof.steps,
    })
}

pub(crate) fn folded_statement_digest(folded: &Chip8FoldedStatement) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/chip8/folded_statement");
    tr.append_u64s(
        b"neo.fold.next/chip8/folded_statement/meta",
        &[folded.chunk_count, folded.semantic_step_count],
    );
    tr.append_u64s(
        b"neo.fold.next/chip8/folded_statement/schedule",
        &folded.fold_schedule.meta_words(),
    );
    tr.append_message(
        b"neo.fold.next/chip8/folded_statement/kernel_relation_digest",
        &folded.kernel_relation_digest,
    );
    append_recursive_accumulator(&mut tr, &folded.final_accumulator);
    tr.digest32()
}

fn audit_check_folded_statement_components_with_output(
    public: &crate::chip8::kernel::SimpleKernelPublicInput,
    folded: &Chip8FoldedStatement,
    kernel_export: &KernelExportProof,
    steps: &[Chip8ChunkTransitionWitness],
) -> Result<(KernelExportRelationResult, Vec<FixedShapeChunkSummary>), SimpleKernelError> {
    if folded.fold_schedule != CHIP8_BRIDGE_FOLD_SCHEDULE {
        return Err(SimpleKernelError::BridgeFailed(
            "folded statement schedule does not match frozen CHIP-8 export schedule".into(),
        ));
    }
    if folded.final_accumulator.final_main_claims.len() != CHIP8_MAIN_CARRY_WIDTH {
        return Err(SimpleKernelError::BridgeFailed(
            "folded statement main-claim width mismatch".into(),
        ));
    }
    if folded.digest != folded_statement_digest(folded) {
        return Err(SimpleKernelError::BridgeFailed(
            "folded statement digest mismatch".into(),
        ));
    }
    let verified_kernel = verify_kernel_export_proof(
        public,
        folded.fold_schedule,
        folded.kernel_relation_digest,
        kernel_export,
    )?;
    let verified_semantic_step_count: usize = verified_kernel
        .chunk_handoffs
        .iter()
        .map(|handoff| handoff.public_chunk.steps.len())
        .sum();
    if folded.semantic_step_count as usize != verified_semantic_step_count {
        return Err(SimpleKernelError::BridgeFailed(
            "folded statement semantic step count does not match checked native handoff".into(),
        ));
    }
    if folded.chunk_count as usize != verified_kernel.chunk_handoffs.len() {
        return Err(SimpleKernelError::BridgeFailed(
            "folded statement chunk count does not match checked native handoff".into(),
        ));
    }
    if folded.chunk_count as usize != steps.len() {
        return Err(SimpleKernelError::BridgeFailed(
            "folded statement chunk count does not match recursive steps".into(),
        ));
    }
    let chunk_summaries = audit_check_recursive_steps(folded, &verified_kernel, steps)?;
    Ok((verified_kernel, chunk_summaries))
}

fn audit_check_recursive_steps(
    folded: &Chip8FoldedStatement,
    verified_kernel: &KernelExportRelationResult,
    steps: &[Chip8ChunkTransitionWitness],
) -> Result<Vec<FixedShapeChunkSummary>, SimpleKernelError> {
    let vm = Chip8VmSpec::default();
    let root_context = SimpleKernelRootContext::new()?;
    let structure = &vm.core_ccs_spec().structure;
    let params = root_context.params();
    let optimized_cache = OptimizedStructureCache::build(structure).map_err(run_error)?;
    let mut transcript = Poseidon2Transcript::new(b"neo.fold.next/session");
    let mut accumulator = Chip8RecursiveAccumulatorState::seed();
    let mut chunk_summaries = Vec::with_capacity(steps.len());

    for (chunk_index, step_witness) in steps.iter().enumerate() {
        let (next_main, next_handle, next_bridge_state, chunk_relation_digest) = audit_check_chunk_transition(
            step_witness,
            chunk_index,
            accumulator.terminal_handle,
            accumulator.bridge_state,
            &accumulator.main,
            &mut transcript,
            &params,
            structure,
            &optimized_cache,
            root_context.log(),
            &verified_kernel.chunk_handoffs,
        )?;
        let public_chunk = &verified_kernel.chunk_handoffs[chunk_index].public_chunk;
        chunk_summaries.push(FixedShapeChunkSummary::from_public_chunk(
            public_chunk,
            public_chunk_digest(public_chunk),
            chunk_relation_digest,
        ));

        accumulator = Chip8RecursiveAccumulatorState {
            main: next_main,
            bridge_state: next_bridge_state,
            terminal_handle: next_handle,
        };
    }

    let final_accumulator = accumulator.into_public()?;
    if final_accumulator.terminal_handle != folded.final_accumulator.terminal_handle {
        return Err(SimpleKernelError::BridgeFailed(
            "recursive terminal handle mismatch".into(),
        ));
    }
    if final_accumulator.final_main_claims.as_slice() != folded.final_accumulator.final_main_claims.as_slice() {
        return Err(SimpleKernelError::BridgeFailed(
            "recursive final main claims mismatch".into(),
        ));
    }
    if final_accumulator.bridge_state != folded.final_accumulator.bridge_state {
        return Err(SimpleKernelError::BridgeFailed(
            "recursive final bridge state mismatch".into(),
        ));
    }
    if final_accumulator.bridge_state != verified_kernel.bridge_final_state {
        return Err(SimpleKernelError::BridgeFailed(
            "statement bridge final state does not match checked kernel handoff".into(),
        ));
    }
    Ok(chunk_summaries)
}

fn audit_check_chunk_transition<L>(
    step_witness: &Chip8ChunkTransitionWitness,
    expected_chunk_index: usize,
    previous_handle: AccumulatorHandle,
    previous_bridge_state: [u8; 32],
    main_carry: &Carry,
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    optimized_cache: &OptimizedStructureCache,
    log: &L,
    chunk_handoffs: &[KernelExportChunkHandoff],
) -> Result<(Carry, AccumulatorHandle, [u8; 32], [u8; 32]), SimpleKernelError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
{
    let handoff = chunk_handoffs.get(expected_chunk_index).ok_or_else(|| {
        SimpleKernelError::BridgeFailed(format!(
            "recursive step {} chunk handoff missing from verified kernel export",
            expected_chunk_index
        ))
    })?;
    let public_chunk = &handoff.public_chunk;
    let chunk_input = main_transition_chunk_input(public_chunk, &step_witness.main_transition)?;
    let proved = verify_chip8_chunk_relation_with_witness(
        expected_chunk_index,
        &chunk_input,
        main_carry,
        &step_witness.main_transition.replay_witness,
        &handoff.bridge_handoff,
        &step_witness.bridge_bindings,
        previous_bridge_state,
        transcript,
        params,
        structure,
        log,
        ajtai_mixers(),
        optimized_cache,
    )?;
    let chunk_relation_digest = chip8_chunk_relation_digest(public_chunk, &proved.artifacts);
    transcript.append_message(b"neo.fold.next/chunk_done", &[1]);

    let next_handle = AccumulatorHandle(step_handle(
        previous_handle.0,
        expected_chunk_index,
        public_chunk.start_index,
        public_chunk.steps.len(),
        chunk_relation_digest,
    ));
    Ok((
        proved.next_main,
        next_handle,
        proved.next_bridge_state,
        chunk_relation_digest,
    ))
}

fn chunk_transition_main_witness(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    chunk_input: &ChunkInput,
    replay_witness: ChunkReplayWitness,
) -> Result<Chip8MainChunkTransitionWitness, SimpleKernelError> {
    Ok(Chip8MainChunkTransitionWitness {
        step_witness_slots: std::array::from_fn(|slot| chunk_input.steps.get(slot).map(|step| step.witness.clone())),
        replay_witness: Chip8ChunkReplayWitness::from_chunk_replay_witness(params, structure, replay_witness)?,
    })
}

fn main_transition_chunk_input(
    public_chunk: &PublicChunk,
    witness: &Chip8MainChunkTransitionWitness,
) -> Result<ChunkInput, SimpleKernelError> {
    let mut steps = Vec::with_capacity(public_chunk.steps.len());
    for slot_index in 0..CHIP8_BRIDGE_ROWS_PER_CHUNK {
        if slot_index < public_chunk.steps.len() {
            let public_step = public_chunk.steps[slot_index].clone();
            let step_witness = witness.step_witness_slots[slot_index]
                .clone()
                .ok_or_else(|| {
                    SimpleKernelError::BridgeFailed(format!(
                        "public chunk active slot {} missing step witness",
                        slot_index
                    ))
                })?;
            steps.push(StepInput {
                label: public_step.label,
                mcs: public_step.mcs,
                witness: step_witness,
            });
        } else if witness.step_witness_slots[slot_index].is_some() {
            return Err(SimpleKernelError::BridgeFailed(format!(
                "public chunk inactive slot {} must not carry a step witness",
                slot_index
            )));
        }
    }
    Ok(ChunkInput {
        start_index: public_chunk.start_index,
        steps,
    })
}

fn chip8_main_carry_claims(
    claims: Vec<neo_ccs::CeClaim<Commitment, F, neo_math::K>>,
) -> Result<[neo_ccs::CeClaim<Commitment, F, neo_math::K>; CHIP8_MAIN_CARRY_WIDTH], SimpleKernelError> {
    claims.try_into().map_err(|claims: Vec<_>| {
        SimpleKernelError::BridgeFailed(format!(
            "recursive final main claim width {} != frozen CHIP-8 carry width {}",
            claims.len(),
            CHIP8_MAIN_CARRY_WIDTH
        ))
    })
}

pub(crate) fn final_proof_digest(
    folded: &Chip8FoldedStatement,
    kernel_export: &KernelExportProof,
    chunk_summaries: &[FixedShapeChunkSummary],
    steps: &[Chip8ChunkTransitionWitness],
) -> [u8; 32] {
    let component_digests = chip8_final_proof_component_digests_from_parts(kernel_export, steps);
    digest_fixed_shape_final_proof(
        &folded.digest,
        folded.chunk_count,
        chunk_summaries,
        &[component_digests.kernel_export_digest],
        &component_digests.chunk_transition_digests,
    )
}

pub(crate) fn kernel_export_proof_digest(proof: &KernelExportProof) -> [u8; 32] {
    proof.digest
}

pub(crate) fn final_proof_component_digests(proof: &Chip8FinalProof) -> Chip8FinalProofComponentDigests {
    chip8_final_proof_component_digests_from_parts(&proof.kernel_export, &proof.steps)
}

fn chip8_final_proof_component_digests_from_parts(
    kernel_export: &KernelExportProof,
    steps: &[Chip8ChunkTransitionWitness],
) -> Chip8FinalProofComponentDigests {
    Chip8FinalProofComponentDigests {
        kernel_export_digest: kernel_export_proof_digest(kernel_export),
        chunk_transition_digests: steps.iter().map(chunk_transition_witness_digest).collect(),
    }
}

pub(crate) fn chunk_transition_witness_digest(step: &Chip8ChunkTransitionWitness) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/chip8/chunk_transition_witness");
    tr.append_message(
        b"neo.fold.next/chip8/chunk_transition_witness/header_digest",
        &step.main_transition.replay_witness.header_digest,
    );
    tr.append_u64s(
        b"neo.fold.next/chip8/chunk_transition_witness/counts",
        &[
            step.main_transition.step_witness_slots.len() as u64,
            step.main_transition.replay_witness.ccs_output_slots.len() as u64,
            step.main_transition.replay_witness.fe_rounds.len() as u64,
            step.main_transition.replay_witness.nc_rounds.len() as u64,
            step.bridge_bindings.len() as u64,
        ],
    );
    for witness in &step.main_transition.step_witness_slots {
        match witness {
            Some(witness) => {
                tr.append_u64s(b"neo.fold.next/chip8/chunk_transition_witness/step_flag", &[1]);
                tr.append_message(
                    b"neo.fold.next/chip8/chunk_transition_witness/step_digest",
                    &ccs_witness_digest(witness),
                );
            }
            None => {
                tr.append_u64s(b"neo.fold.next/chip8/chunk_transition_witness/step_flag", &[0]);
            }
        }
    }
    let mut me_scratch = Vec::<F>::with_capacity(2048);
    for output in &step.main_transition.replay_witness.ccs_output_slots {
        match output {
            Some(output) => {
                tr.append_u64s(b"neo.fold.next/chip8/chunk_transition_witness/output_flag", &[1]);
                let digest = me_digest_poseidon_into(&mut me_scratch, output);
                tr.append_fields_iter(
                    b"neo.fold.next/chip8/chunk_transition_witness/output_digest",
                    digest.len(),
                    digest.iter().copied(),
                );
            }
            None => {
                tr.append_u64s(b"neo.fold.next/chip8/chunk_transition_witness/output_flag", &[0]);
            }
        }
    }
    append_replay_round_witnesses(
        &mut tr,
        b"neo.fold.next/chip8/chunk_transition_witness/fe_round",
        &step.main_transition.replay_witness.fe_rounds,
    );
    append_replay_round_witnesses(
        &mut tr,
        b"neo.fold.next/chip8/chunk_transition_witness/nc_round",
        &step.main_transition.replay_witness.nc_rounds,
    );
    for binding in &step.bridge_bindings {
        match binding {
            Some(binding) => {
                tr.append_u64s(
                    b"neo.fold.next/chip8/chunk_transition_witness/bridge_binding_flag",
                    &[1],
                );
                tr.append_message(
                    b"neo.fold.next/chip8/chunk_transition_witness/bridge_binding_digest",
                    &binding.digest,
                );
            }
            None => {
                tr.append_u64s(
                    b"neo.fold.next/chip8/chunk_transition_witness/bridge_binding_flag",
                    &[0],
                );
            }
        }
    }
    tr.digest32()
}

fn ccs_witness_digest(witness: &CcsWitness<F>) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/chip8/ccs_witness");
    tr.append_u64s(
        b"neo.fold.next/chip8/ccs_witness/meta",
        &[witness.w.len() as u64, witness.Z.rows() as u64, witness.Z.cols() as u64],
    );
    tr.append_fields_iter(
        b"neo.fold.next/chip8/ccs_witness/w",
        witness.w.len(),
        witness.w.iter().copied(),
    );
    tr.append_fields_iter(
        b"neo.fold.next/chip8/ccs_witness/z",
        witness.Z.as_slice().len(),
        witness.Z.as_slice().iter().copied(),
    );
    tr.digest32()
}

fn append_replay_round_witnesses<const N: usize>(
    tr: &mut Poseidon2Transcript,
    label: &'static [u8],
    rounds: &[Chip8ReplayRoundWitness; N],
) {
    for round in rounds {
        tr.append_u64s(label, &[round.coeff_len as u64]);
        for coeff in &round.coeffs[..round.coeff_len as usize] {
            tr.append_fields(label, &coeff.as_coeffs());
        }
    }
}

fn append_recursive_accumulator(tr: &mut Poseidon2Transcript, accumulator: &Chip8RecursiveAccumulator) {
    let final_main_claim_digests = final_main_claim_digests(&accumulator.final_main_claims);
    tr.append_message(
        b"neo.fold.next/chip8/statement/final_bridge_state",
        &accumulator.bridge_state,
    );
    tr.append_message(
        b"neo.fold.next/chip8/statement/terminal_handle",
        &accumulator.terminal_handle.0,
    );
    tr.append_u64s(
        b"neo.fold.next/chip8/statement/final_main_claim_count",
        &[final_main_claim_digests.len() as u64],
    );
    tr.append_fields_iter(
        b"neo.fold.next/chip8/statement/final_main_claim_digest",
        final_main_claim_digests.len() * 4,
        final_main_claim_digests
            .iter()
            .flat_map(|digest| digest.iter().copied()),
    );
}

fn final_main_claim_digests(
    final_main_claims: &[neo_ccs::CeClaim<Commitment, F, neo_math::K>; CHIP8_MAIN_CARRY_WIDTH],
) -> Vec<[F; 4]> {
    let mut digests = Vec::with_capacity(final_main_claims.len());
    let mut scratch = Vec::<F>::with_capacity(2048);
    for claim in final_main_claims {
        digests.push(me_digest_poseidon_into(&mut scratch, claim));
    }
    digests
}

pub(crate) fn recursive_seed() -> [u8; 32] {
    fixed_shape_recursive_seed(b"neo.fold.next/chip8/recursive_seed")
}

fn run_error(err: impl ToString) -> SimpleKernelError {
    SimpleKernelError::BridgeFailed(err.to_string())
}

fn add_commitments(lhs: &Commitment, rhs: &Commitment) -> Commitment {
    let mut out = lhs.clone();
    out.add_inplace(rhs);
    out
}

fn scale_commitment_by_rho(rho: &Mat<F>, commitment: &Commitment) -> Commitment {
    let mut out = Commitment::zeros(commitment.d, commitment.kappa);
    for col in 0..commitment.kappa {
        for row in 0..commitment.d {
            let mut acc = F::ZERO;
            for idx in 0..commitment.d {
                acc += rho[(row, idx)] * commitment.col(col)[idx];
            }
            out.col_mut(col)[row] = acc;
        }
    }
    out
}

fn mix_rhos_commits(rhos: &[Mat<F>], commitments: &[Commitment]) -> Commitment {
    let mut acc = Commitment::zeros(commitments[0].d, commitments[0].kappa);
    for (rho, commitment) in rhos.iter().zip(commitments.iter()) {
        acc = add_commitments(&acc, &scale_commitment_by_rho(rho, commitment));
    }
    acc
}

fn combine_b_pows(commitments: &[Commitment], b: u32) -> Commitment {
    let mut acc = Commitment::zeros(commitments[0].d, commitments[0].kappa);
    let mut pow = F::ONE;
    let base = F::from_u64(b as u64);
    for commitment in commitments {
        let mut term = commitment.clone();
        for value in &mut term.data {
            *value *= pow;
        }
        acc = add_commitments(&acc, &term);
        pow *= base;
    }
    acc
}

fn ajtai_mixers() -> CommitmentMixers<fn(&[Mat<F>], &[Commitment]) -> Commitment, fn(&[Commitment], u32) -> Commitment>
{
    CommitmentMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}
