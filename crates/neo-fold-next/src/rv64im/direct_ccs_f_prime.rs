//! Owns single-terminal-step direct-CCS SuperNeo compression for non-VM diagnostics.
//!
//! This is a relation-owned diagnostic path for one terminal F' step: the probe
//! supplies direct CCS steps, while this module synthesizes the shared `NIFS.V`
//! replay and terminal post-DEC CE consistency inside Spartan. It deliberately
//! rejects multi-chunk runs because that would check historical chunks instead
//! of proving a folded Construction-2/IVC state.

use std::time::Instant;

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::optimized_engine::{OptimizedStructureCache, PiCcsReplayProofWitness};
use neo_reductions::engines::utils::{build_dims_and_policy, digest_ccs_matrices_with_sparse_cache, Dims};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::ivc_snark::{
    R1CSSNARKTrait, Rv64imDeciderEngine, Rv64imDeciderProverKey, Rv64imDeciderSnark, Rv64imDeciderVerifierKey, ShapeCS,
    SpartanCircuit, SpartanF,
};
use super::main_relation_circuit::ce_consistency::{
    debug_enforce_paper_ce_claim_consistency_with_breakdown, enforce_paper_ce_claim_consistency,
    PaperCeRelationConstraintBreakdown,
};
use super::main_relation_circuit::claim::alloc_ce_claim;
use super::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use super::main_relation_circuit::witness::alloc_packed_witness;
use super::main_relation_spartan::{synthesize_direct_ccs_nifs_chunk, Rv64imClaimBundle};
use super::main_relation_trace::{
    build_rv64im_main_circuit_chunk_replay_surface, build_rv64im_main_circuit_pi_ccs_replay_surface,
    Rv64imMainCircuitChunkCover, Rv64imMainCircuitChunkReplaySurface, Rv64imMainCircuitHandoff,
    Rv64imMainCircuitPublicInputLayout,
};
use crate::chunk_relation::trace_chunk_relation_with_witness_and_instance_digest;
use crate::finalize::{digest_fields_as_digest32, public_chunk_digest};
use crate::ivc::{SuperNeoIvcState, SuperNeoIvcStepRelation, SuperNeoIvcTranscriptSnapshot};
use crate::proof::{partition_prover_step_inputs, Carry, PackagedProof, ProverChunkInput, StepInput};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectCcsFPrimeSnarkProof {
    pub snark_data: Vec<u8>,
}

impl DirectCcsFPrimeSnarkProof {
    pub fn snark_bytes_len(&self) -> usize {
        self.snark_data.len()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DirectCcsFPrimeSnarkPerf {
    pub setup_ms: f64,
    pub prep_ms: f64,
    pub prove_ms: f64,
    pub encode_ms: f64,
    pub verify_ms: f64,
    pub total_prove_ms: f64,
    pub total_verify_ms: f64,
    pub r1cs_sizes: [usize; 10],
    pub r1cs_nnz: usize,
    pub pcs_ms: f64,
    pub final_proof_bytes: usize,
    pub snark_bytes: usize,
    pub public_inputs: usize,
    pub chunk_constraints_first4: [usize; 4],
    pub chunk_constraints_by_chunk: Vec<usize>,
    pub chunk_count: usize,
    pub public_link_constraints: usize,
    pub chunk_done_constraints: usize,
    pub final_ce_relation_constraints: usize,
    pub final_ce_relation_breakdown: PaperCeRelationConstraintBreakdown,
}

#[derive(Debug, Error)]
pub enum DirectCcsFPrimeSnarkError {
    #[error("direct CCS F' input error: {0}")]
    Input(String),
    #[error("direct CCS F' synthesis failed: {0}")]
    Synthesis(String),
    #[error("direct CCS F' setup failed: {0}")]
    Setup(String),
    #[error("direct CCS F' prepare failed: {0}")]
    Prepare(String),
    #[error("direct CCS F' prove failed: {0}")]
    Prove(String),
    #[error("direct CCS F' verify failed: {0}")]
    Verify(String),
    #[error("direct CCS F' proof encoding failed: {0}")]
    Encode(String),
    #[error("direct CCS F' proof decoding failed: {0}")]
    Decode(String),
    #[error("direct CCS F' public IO mismatch")]
    PublicIoMismatch,
}

struct DirectCcsFPrimeKeys {
    pk: Rv64imDeciderProverKey,
    vk: Rv64imDeciderVerifierKey,
}

#[derive(Clone, Debug)]
pub struct DirectCcsLatestFPrimeSummary {
    pub chunk_index: u64,
    pub fresh_claims: usize,
    pub incoming_ce_claims: usize,
    pub output_ce_claims: usize,
    pub final_ce_claims: usize,
}

#[derive(Clone)]
pub struct DirectCcsIvcState {
    params: NeoParams,
    structure: CcsStructure<F>,
    dims: Dims,
    mat_digest: [Goldilocks; 4],
    optimized_cache: OptimizedStructureCache,
    state: SuperNeoIvcState,
    last_step: Option<DirectCcsIvcStepRecord>,
}

#[derive(Clone)]
struct DirectCcsIvcStepRecord {
    relation: SuperNeoIvcStepRelation,
    surface: DirectCcsChunkCircuitSurface,
}

#[derive(Clone)]
struct DirectCcsChunkCircuitSurface {
    cover: Rv64imMainCircuitChunkCover,
    replay: Rv64imMainCircuitChunkReplaySurface,
}

#[derive(Clone)]
struct DirectCcsFPrimeCircuit {
    params: NeoParams,
    structure: CcsStructure<F>,
    dims: Dims,
    mat_digest: [Goldilocks; 4],
    chunks: Vec<DirectCcsChunkCircuitSurface>,
    initial_claims: Vec<CeClaim<Commitment, F, K>>,
    initial_transcript: Option<SuperNeoIvcTranscriptSnapshot>,
    final_claims: Vec<CeClaim<Commitment, F, K>>,
    final_witnesses: Vec<CcsWitness<F>>,
}

impl DirectCcsFPrimeCircuit {
    fn chunk_digest_public_values(&self) -> Vec<SpartanF> {
        self.chunks
            .iter()
            .flat_map(|chunk| {
                chunk
                    .replay
                    .handoff
                    .public_chunk_instance_digest
                    .into_iter()
                    .map(field_to_spartan)
            })
            .collect()
    }
}

impl DirectCcsIvcState {
    pub fn new(params: &NeoParams, structure: &CcsStructure<F>) -> Result<Self, DirectCcsFPrimeSnarkError> {
        let dims = build_dims_and_policy(params, structure)
            .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
        let optimized_cache = OptimizedStructureCache::build(structure)
            .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
        let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, None)
            .try_into()
            .map_err(|digest: Vec<Goldilocks>| {
                DirectCcsFPrimeSnarkError::Input(format!("expected 4 matrix digest limbs, got {}", digest.len()))
            })?;
        Ok(Self {
            params: params.clone(),
            structure: structure.clone(),
            dims,
            mat_digest,
            optimized_cache,
            state: SuperNeoIvcState::seed(),
            last_step: None,
        })
    }

    pub fn append<L, MR, MB>(
        &self,
        relation: &SuperNeoIvcStepRelation,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        if !superneo_ivc_states_match(&self.state, &relation.state_in) {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct CCS IVC append relation does not start from the carried state".into(),
            ));
        }
        relation
            .verify(&self.params, &self.structure, log, mixers, &self.optimized_cache)
            .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
        let surface = build_direct_ccs_chunk_surface_from_ivc_relation(
            &self.params,
            &self.structure,
            self.dims,
            relation,
            log,
            mixers,
            &self.optimized_cache,
        )?;
        Ok(Self {
            params: self.params.clone(),
            structure: self.structure.clone(),
            dims: self.dims,
            mat_digest: self.mat_digest,
            optimized_cache: self.optimized_cache.clone(),
            state: relation.state_out.clone(),
            last_step: Some(DirectCcsIvcStepRecord {
                relation: relation.clone(),
                surface,
            }),
        })
    }

    pub fn append_all<L, MR, MB>(
        params: &NeoParams,
        structure: &CcsStructure<F>,
        relations: &[SuperNeoIvcStepRelation],
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let mut state = Self::new(params, structure)?;
        for relation in relations {
            state = state.append(relation, log, mixers)?;
        }
        Ok(state)
    }

    pub fn final_state(&self) -> &SuperNeoIvcState {
        &self.state
    }

    pub fn latest_relation_and_advice(&self) -> Result<DirectCcsLatestFPrimeSummary, DirectCcsFPrimeSnarkError> {
        let last = self.last_step.as_ref().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input(
                "direct CCS folded compression requires at least one appended SuperNeo relation".into(),
            )
        })?;
        Ok(DirectCcsLatestFPrimeSummary {
            chunk_index: last.relation.chunk_index,
            fresh_claims: last.relation.chunk.steps.len(),
            incoming_ce_claims: last.relation.state_in.carry.claims.len(),
            output_ce_claims: last.relation.replay_witness.ccs_outputs.len(),
            final_ce_claims: self.state.carry.claims.len(),
        })
    }

    pub fn compress_with_trace(
        &self,
        emit: &mut dyn FnMut(&str),
    ) -> Result<(DirectCcsFPrimeSnarkProof, DirectCcsFPrimeSnarkPerf), DirectCcsFPrimeSnarkError> {
        emit("direct_ccs_ivc.phase=latest_relation_and_advice.start");
        let circuit = self.latest_circuit()?;
        emit("direct_ccs_ivc.phase=latest_relation_and_advice.done");
        emit("direct_ccs_ivc.phase=spartan_terminal_prove.start");
        let proved = prove_direct_ccs_f_prime_circuit(circuit);
        emit("direct_ccs_ivc.phase=spartan_terminal_prove.done");
        proved
    }

    pub fn compress(&self) -> Result<(DirectCcsFPrimeSnarkProof, DirectCcsFPrimeSnarkPerf), DirectCcsFPrimeSnarkError> {
        let mut emit = |_message: &str| {};
        self.compress_with_trace(&mut emit)
    }

    fn latest_circuit(&self) -> Result<DirectCcsFPrimeCircuit, DirectCcsFPrimeSnarkError> {
        let last = self.last_step.as_ref().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input(
                "direct CCS folded compression requires at least one appended SuperNeo relation".into(),
            )
        })?;
        Ok(DirectCcsFPrimeCircuit {
            params: self.params.clone(),
            structure: self.structure.clone(),
            dims: self.dims,
            mat_digest: self.mat_digest,
            chunks: vec![last.surface.clone()],
            initial_claims: last.relation.state_in.carry.claims.clone(),
            initial_transcript: Some(last.relation.state_in.transcript.clone()),
            final_claims: self.state.carry.claims.clone(),
            final_witnesses: final_carry_witnesses(&self.state.carry.witnesses)?,
        })
    }
}

impl SpartanCircuit<Rv64imDeciderEngine> for DirectCcsFPrimeCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(self.chunk_digest_public_values())
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        let public_values = self.public_values()?;
        let public_inputs = public_values
            .into_iter()
            .enumerate()
            .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("public_{idx}")), || Ok(value)))
            .collect::<Result<Vec<_>, _>>()?;
        let mut public_cursor = 0usize;
        let mut transcript = alloc_initial_transcript(cs, self.initial_transcript.as_ref())?;
        let mut carried = alloc_initial_claim_bundle(cs, &self.initial_claims)?;

        for (chunk_index, chunk) in self.chunks.iter().enumerate() {
            let (next, chunk_digest) = synthesize_direct_ccs_nifs_chunk(
                &self.params,
                &self.structure,
                self.dims,
                &self.mat_digest,
                &mut cs.namespace(|| format!("chunk_{chunk_index}")),
                chunk_index,
                &chunk.cover,
                &chunk.replay,
                &mut transcript,
                carried,
            )?;
            enforce_digest_public_io(
                &mut cs.namespace(|| format!("chunk_{chunk_index}_public_digest")),
                &chunk_digest,
                &public_inputs,
                &mut public_cursor,
                &format!("chunk_{chunk_index}_public_digest"),
            )?;
            transcript.append_message(
                cs.namespace(|| format!("chunk_{chunk_index}_done")),
                b"neo.fold.next/chunk_done",
                &[1],
            )?;
            carried = next;
        }

        if carried.effective_count() != self.final_claims.len()
            || self.final_witnesses.len() != carried.effective_count()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        for (claim_index, (claim, witness)) in carried
            .effective_claims()
            .iter()
            .zip(self.final_witnesses.iter())
            .enumerate()
        {
            let witness = alloc_packed_witness(
                &mut cs.namespace(|| format!("final_claim_{claim_index}_witness")),
                witness,
                &format!("final_claim_{claim_index}_witness"),
            )?;
            enforce_paper_ce_claim_consistency(
                &mut cs.namespace(|| format!("final_claim_{claim_index}_ce_consistency")),
                &self.params,
                &self.structure,
                &self.structure,
                &witness,
                claim,
                SpartanF::from_canonical_u64(7),
                &format!("final_claim_{claim_index}_ce_consistency"),
            )?;
        }
        if public_cursor != public_inputs.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        Ok(())
    }
}

pub fn prove_direct_ccs_f_prime_snark_with_perf(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    packaged: &PackagedProof,
    final_carry: &Carry,
    steps: &[StepInput],
) -> Result<(DirectCcsFPrimeSnarkProof, DirectCcsFPrimeSnarkPerf), DirectCcsFPrimeSnarkError> {
    if packaged.statement.chunks.len() != 1 {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct CCS terminal replay compressor is only allowed for one terminal F' step; got {} chunks. \
             Multi-chunk direct CCS compression must use a folded Construction-2/IVC state, not replay every chunk in Spartan",
            packaged.statement.chunks.len()
        )));
    }
    if packaged.statement.final_main_claims != final_carry.claims {
        return Err(DirectCcsFPrimeSnarkError::Input(
            "packaged final claims do not match final carry claims".into(),
        ));
    }
    if final_carry.claims.len() != final_carry.witnesses.len() {
        return Err(DirectCcsFPrimeSnarkError::Input(
            "final carry requires one witness per claim".into(),
        ));
    }
    let final_witnesses = final_carry_witnesses(&final_carry.witnesses)?;
    let input_chunks = partition_prover_step_inputs(packaged.statement.fold_schedule, steps.to_vec())
        .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
    let circuit = build_direct_ccs_f_prime_circuit(params, structure, packaged, &input_chunks, final_witnesses)?;
    prove_direct_ccs_f_prime_circuit(circuit)
}

fn prove_direct_ccs_f_prime_circuit(
    circuit: DirectCcsFPrimeCircuit,
) -> Result<(DirectCcsFPrimeSnarkProof, DirectCcsFPrimeSnarkPerf), DirectCcsFPrimeSnarkError> {
    let breakdown = measure_direct_ccs_f_prime_constraints(&circuit)?;

    let setup_started = Instant::now();
    let (pk, vk) =
        Rv64imDeciderSnark::setup(circuit.clone()).map_err(|err| DirectCcsFPrimeSnarkError::Setup(err.to_string()))?;
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;
    let r1cs_sizes = pk.sizes();
    let r1cs_nnz = pk.shape_debug_stats().total_nnz;
    let keys = DirectCcsFPrimeKeys { pk, vk };

    let prep_started = Instant::now();
    let prep = Rv64imDeciderSnark::prep_prove(&keys.pk, circuit.clone(), false)
        .map_err(|err| DirectCcsFPrimeSnarkError::Prepare(err.to_string()))?;
    let prep_ms = prep_started.elapsed().as_secs_f64() * 1_000.0;

    let prove_started = Instant::now();
    let (spartan_proof, spartan_perf) = Rv64imDeciderSnark::prove_with_perf(&keys.pk, circuit.clone(), &prep, false)
        .map_err(|err| DirectCcsFPrimeSnarkError::Prove(err.to_string()))?;
    let prove_ms = prove_started.elapsed().as_secs_f64() * 1_000.0;

    let encode_started = Instant::now();
    let snark_data =
        bincode::serialize(&spartan_proof).map_err(|err| DirectCcsFPrimeSnarkError::Encode(err.to_string()))?;
    let encode_ms = encode_started.elapsed().as_secs_f64() * 1_000.0;
    let proof = DirectCcsFPrimeSnarkProof { snark_data };

    let verify_started = Instant::now();
    verify_direct_ccs_f_prime_proof(&keys.vk, &circuit, &proof.snark_data)?;
    let verify_ms = verify_started.elapsed().as_secs_f64() * 1_000.0;

    let final_proof_bytes = bincode::serialize(&proof)
        .map_err(|err| DirectCcsFPrimeSnarkError::Encode(err.to_string()))?
        .len();
    let snark_bytes = proof.snark_bytes_len();
    Ok((
        proof,
        DirectCcsFPrimeSnarkPerf {
            setup_ms,
            prep_ms,
            prove_ms,
            encode_ms,
            verify_ms,
            total_prove_ms: prep_ms + prove_ms + encode_ms,
            total_verify_ms: verify_ms,
            r1cs_sizes,
            r1cs_nnz,
            pcs_ms: spartan_perf.pcs_prove_ms,
            final_proof_bytes,
            snark_bytes,
            public_inputs: breakdown.public_inputs,
            chunk_constraints_first4: breakdown.chunk_constraints_first4,
            chunk_constraints_by_chunk: breakdown.chunk_constraints_by_chunk,
            chunk_count: breakdown.chunk_count,
            public_link_constraints: breakdown.public_link_constraints,
            chunk_done_constraints: breakdown.chunk_done_constraints,
            final_ce_relation_constraints: breakdown.final_ce_relation_constraints,
            final_ce_relation_breakdown: breakdown.final_ce_relation_breakdown,
        },
    ))
}

fn build_direct_ccs_f_prime_circuit(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    packaged: &PackagedProof,
    input_chunks: &[ProverChunkInput],
    final_witnesses: Vec<CcsWitness<F>>,
) -> Result<DirectCcsFPrimeCircuit, DirectCcsFPrimeSnarkError> {
    if packaged.proof.session.chunks.len() != input_chunks.len() {
        return Err(DirectCcsFPrimeSnarkError::Input(
            "packaged chunk count does not match direct step partition".into(),
        ));
    }
    let dims =
        build_dims_and_policy(params, structure).map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, None)
        .try_into()
        .map_err(|digest: Vec<Goldilocks>| {
            DirectCcsFPrimeSnarkError::Input(format!("expected 4 matrix digest limbs, got {}", digest.len()))
        })?;
    let mut chunks = Vec::with_capacity(input_chunks.len());
    for (chunk_index, ((statement_chunk, proof_chunk), input_chunk)) in packaged
        .statement
        .chunks
        .iter()
        .zip(packaged.proof.session.chunks.iter())
        .zip(input_chunks.iter())
        .enumerate()
    {
        if !public_chunks_match(statement_chunk, &proof_chunk.chunk)
            || !public_chunks_match(statement_chunk, &input_chunk.public_chunk)
        {
            return Err(DirectCcsFPrimeSnarkError::Input(format!(
                "public chunk {chunk_index} differs between statement, proof, and direct step partition"
            )));
        }
        chunks.push(build_direct_ccs_chunk_surface(dims, proof_chunk, input_chunk)?);
    }
    Ok(DirectCcsFPrimeCircuit {
        params: params.clone(),
        structure: structure.clone(),
        dims,
        mat_digest,
        chunks,
        initial_claims: Vec::new(),
        initial_transcript: None,
        final_claims: packaged.statement.final_main_claims.clone(),
        final_witnesses,
    })
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct DirectCcsFPrimeConstraintBreakdown {
    public_inputs: usize,
    chunk_count: usize,
    chunk_constraints_first4: [usize; 4],
    chunk_constraints_by_chunk: Vec<usize>,
    public_link_constraints: usize,
    chunk_done_constraints: usize,
    final_ce_relation_constraints: usize,
    final_ce_relation_breakdown: PaperCeRelationConstraintBreakdown,
}

fn measure_direct_ccs_f_prime_constraints(
    circuit: &DirectCcsFPrimeCircuit,
) -> Result<DirectCcsFPrimeConstraintBreakdown, DirectCcsFPrimeSnarkError> {
    let mut cs = ShapeCS::<Rv64imDeciderEngine>::new();
    let public_values = circuit
        .public_values()
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    let public_inputs = public_values
        .into_iter()
        .enumerate()
        .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("public_{idx}")), || Ok(value)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    let mut out = DirectCcsFPrimeConstraintBreakdown {
        public_inputs: public_inputs.len(),
        chunk_count: circuit.chunks.len(),
        ..DirectCcsFPrimeConstraintBreakdown::default()
    };
    let mut public_cursor = 0usize;
    let mut transcript = alloc_initial_transcript(&mut cs, circuit.initial_transcript.as_ref())
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    let mut carried = alloc_initial_claim_bundle(&mut cs, &circuit.initial_claims)
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;

    for (chunk_index, chunk) in circuit.chunks.iter().enumerate() {
        let before_chunk = cs.num_constraints();
        let (next, chunk_digest) = synthesize_direct_ccs_nifs_chunk(
            &circuit.params,
            &circuit.structure,
            circuit.dims,
            &circuit.mat_digest,
            &mut cs.namespace(|| format!("chunk_{chunk_index}")),
            chunk_index,
            &chunk.cover,
            &chunk.replay,
            &mut transcript,
            carried,
        )
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
        let chunk_constraints = cs.num_constraints() - before_chunk;
        if chunk_index < out.chunk_constraints_first4.len() {
            out.chunk_constraints_first4[chunk_index] = chunk_constraints;
        }
        out.chunk_constraints_by_chunk.push(chunk_constraints);

        let before_link = cs.num_constraints();
        enforce_digest_public_io(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_public_digest")),
            &chunk_digest,
            &public_inputs,
            &mut public_cursor,
            &format!("chunk_{chunk_index}_public_digest"),
        )
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
        out.public_link_constraints += cs.num_constraints() - before_link;

        let before_done = cs.num_constraints();
        transcript
            .append_message(
                cs.namespace(|| format!("chunk_{chunk_index}_done")),
                b"neo.fold.next/chunk_done",
                &[1],
            )
            .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
        out.chunk_done_constraints += cs.num_constraints() - before_done;
        carried = next;
    }

    if carried.effective_count() != circuit.final_claims.len()
        || circuit.final_witnesses.len() != carried.effective_count()
    {
        return Err(DirectCcsFPrimeSnarkError::Synthesis(
            "measured direct F' carried claim count does not match final claims".into(),
        ));
    }
    let before_final_ce = cs.num_constraints();
    for (claim_index, (claim, witness)) in carried
        .effective_claims()
        .iter()
        .zip(circuit.final_witnesses.iter())
        .enumerate()
    {
        let witness = alloc_packed_witness(
            &mut cs.namespace(|| format!("final_claim_{claim_index}_witness")),
            witness,
            &format!("final_claim_{claim_index}_witness"),
        )
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
        let breakdown = debug_enforce_paper_ce_claim_consistency_with_breakdown(
            &mut cs,
            &circuit.params,
            &circuit.structure,
            &circuit.structure,
            &witness,
            claim,
            SpartanF::from_canonical_u64(7),
            &format!("final_claim_{claim_index}_ce_consistency"),
        )
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
        out.final_ce_relation_breakdown.add_assign(breakdown);
    }
    out.final_ce_relation_constraints = cs.num_constraints() - before_final_ce;
    if public_cursor != public_inputs.len() {
        return Err(DirectCcsFPrimeSnarkError::Synthesis(
            "measured direct F' public cursor mismatch".into(),
        ));
    }
    Ok(out)
}

fn public_chunks_match(left: &crate::proof::PublicChunk, right: &crate::proof::PublicChunk) -> bool {
    left.start_index == right.start_index
        && left.steps.len() == right.steps.len()
        && left
            .steps
            .iter()
            .zip(right.steps.iter())
            .all(|(left_step, right_step)| public_steps_match(left_step, right_step))
}

fn public_steps_match(left: &crate::proof::PublicStep, right: &crate::proof::PublicStep) -> bool {
    left.label == right.label && ccs_claims_match(&left.mcs, &right.mcs)
}

fn ccs_claims_match(left: &CcsClaim<Commitment, F>, right: &CcsClaim<Commitment, F>) -> bool {
    left.c == right.c && left.x == right.x && left.m_in == right.m_in
}

fn build_direct_ccs_chunk_surface(
    dims: Dims,
    proof_chunk: &crate::proof::ChunkProof,
    input_chunk: &ProverChunkInput,
) -> Result<DirectCcsChunkCircuitSurface, DirectCcsFPrimeSnarkError> {
    let replay_proof = PiCcsReplayProofWitness::from_proof(&proof_chunk.ccs_proof)
        .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
    let (row_chals, alpha_prime) = split_challenges(&proof_chunk.ccs_proof.sumcheck_challenges, dims.ell_n, "FE")?;
    let (s_col, alpha_prime_nc) = split_challenges(&proof_chunk.ccs_proof.sumcheck_challenges_nc, dims.ell_m, "NC")?;
    let public_chunk_instance_digest = public_chunk_digest(&input_chunk.public_chunk);
    let handoff = Rv64imMainCircuitHandoff {
        public_chunk: input_chunk.public_chunk.clone(),
        public_chunk_instance_digest,
        public_chunk_digest: digest_fields_as_digest32(public_chunk_instance_digest),
        bridge_handoff_digest: [0u8; 32],
        chunk_relation_digest: proof_chunk.relation_digest,
        public_input_layout: Rv64imMainCircuitPublicInputLayout::PackedPrefix,
    };
    let pi_ccs = build_rv64im_main_circuit_pi_ccs_replay_surface(
        proof_chunk.ccs_outputs.clone(),
        replay_proof,
        proof_chunk.ccs_proof.challenges_public.clone(),
        row_chals,
        alpha_prime,
        s_col,
        alpha_prime_nc,
    );
    let replay = build_rv64im_main_circuit_chunk_replay_surface(
        &handoff,
        &input_chunk.fresh_claims,
        &input_chunk.fresh_witnesses,
        pi_ccs,
        proof_chunk.rlc.parent.clone(),
        proof_chunk.dec.children.clone(),
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Input(format!("failed to build direct chunk replay surface: {err}")))?;
    let cover = Rv64imMainCircuitChunkCover::from_replay_surface(&replay);
    Ok(DirectCcsChunkCircuitSurface { cover, replay })
}

fn build_direct_ccs_chunk_surface_from_ivc_relation<L, MR, MB>(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    relation: &SuperNeoIvcStepRelation,
    log: &L,
    mixers: crate::prover::CommitmentMixers<MR, MB>,
    optimized_cache: &OptimizedStructureCache,
) -> Result<DirectCcsChunkCircuitSurface, DirectCcsFPrimeSnarkError>
where
    L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let public_chunk = relation.chunk.public();
    let public_chunk_instance_digest = public_chunk_digest(&public_chunk);
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(
        relation.state_in.transcript.state,
        relation.state_in.transcript.absorbed,
    );
    let trace = trace_chunk_relation_with_witness_and_instance_digest(
        &mut transcript,
        params,
        structure,
        &relation.chunk,
        &relation.state_in.carry,
        &relation.replay_witness,
        log,
        mixers,
        optimized_cache,
        public_chunk_instance_digest,
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
    let handoff = Rv64imMainCircuitHandoff {
        public_chunk,
        public_chunk_instance_digest,
        public_chunk_digest: digest_fields_as_digest32(public_chunk_instance_digest),
        bridge_handoff_digest: [0u8; 32],
        chunk_relation_digest: relation.chunk_relation_digest,
        public_input_layout: Rv64imMainCircuitPublicInputLayout::PackedPrefix,
    };
    let pi_ccs = build_rv64im_main_circuit_pi_ccs_replay_surface(
        trace.ccs_outputs,
        trace.ccs_replay_proof,
        trace.terminal_state.challenges_public,
        trace.terminal_state.row_chals,
        trace.terminal_state.alpha_prime,
        trace.terminal_state.s_col,
        trace.terminal_state.alpha_prime_nc,
    );
    let fresh_claims = relation
        .chunk
        .steps
        .iter()
        .map(|step| step.mcs.clone())
        .collect::<Vec<_>>();
    let fresh_witnesses = relation
        .chunk
        .steps
        .iter()
        .map(|step| step.witness.clone())
        .collect::<Vec<_>>();
    let replay = build_rv64im_main_circuit_chunk_replay_surface(
        &handoff,
        &fresh_claims,
        &fresh_witnesses,
        pi_ccs,
        trace.parent,
        trace.children,
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Input(format!("failed to build latest direct chunk surface: {err}")))?;
    let cover = Rv64imMainCircuitChunkCover::from_replay_surface(&replay);
    let _ = dims;
    Ok(DirectCcsChunkCircuitSurface { cover, replay })
}

fn split_challenges(
    values: &[K],
    prefix_len: usize,
    label: &str,
) -> Result<(Vec<K>, Vec<K>), DirectCcsFPrimeSnarkError> {
    if values.len() < prefix_len {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "{label} sumcheck challenge vector too short: got {}, need prefix {prefix_len}",
            values.len()
        )));
    }
    Ok((values[..prefix_len].to_vec(), values[prefix_len..].to_vec()))
}

fn final_carry_witnesses(zs: &[Mat<F>]) -> Result<Vec<CcsWitness<F>>, DirectCcsFPrimeSnarkError> {
    zs.iter()
        .enumerate()
        .map(|(idx, z)| {
            if z.rows() != D {
                return Err(DirectCcsFPrimeSnarkError::Input(format!(
                    "final CE witness {idx} has {} rows, expected {D}",
                    z.rows()
                )));
            }
            Ok(CcsWitness {
                w: Vec::new(),
                Z: z.clone(),
            })
        })
        .collect()
}

fn alloc_initial_transcript<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    snapshot: Option<&SuperNeoIvcTranscriptSnapshot>,
) -> Result<Poseidon2TranscriptCircuit, SynthesisError> {
    match snapshot {
        Some(snapshot) => {
            let _ = cs;
            Poseidon2TranscriptCircuit::from_constant_state(snapshot.state.map(field_to_spartan), snapshot.absorbed)
        }
        None => Poseidon2TranscriptCircuit::new(cs.namespace(|| "session_transcript"), b"neo.fold.next/session"),
    }
}

fn alloc_initial_claim_bundle<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claims: &[CeClaim<Commitment, F, K>],
) -> Result<Rv64imClaimBundle, SynthesisError> {
    claims
        .iter()
        .enumerate()
        .map(|(idx, claim)| {
            alloc_ce_claim(
                &mut cs.namespace(|| format!("initial_carry_claim_{idx}")),
                claim,
                &format!("initial_carry_claim_{idx}"),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Rv64imClaimBundle::from_effective_claims)
}

fn verify_direct_ccs_f_prime_proof(
    vk: &Rv64imDeciderVerifierKey,
    circuit: &DirectCcsFPrimeCircuit,
    snark_data: &[u8],
) -> Result<(), DirectCcsFPrimeSnarkError> {
    let proof: Rv64imDeciderSnark =
        bincode::deserialize(snark_data).map_err(|err| DirectCcsFPrimeSnarkError::Decode(err.to_string()))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| DirectCcsFPrimeSnarkError::Verify(err.to_string()))?;
    let expected = circuit
        .public_values()
        .map_err(|err| DirectCcsFPrimeSnarkError::Verify(err.to_string()))?;
    if public_values != expected {
        return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
    }
    Ok(())
}

fn enforce_digest_public_io<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    digest: &[AllocatedNum<SpartanF>; 4],
    public_inputs: &[AllocatedNum<SpartanF>],
    cursor: &mut usize,
    label: &str,
) -> Result<(), SynthesisError> {
    if *cursor + digest.len() > public_inputs.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, lane) in digest.iter().enumerate() {
        let expected = &public_inputs[*cursor + idx];
        cs.enforce(
            || format!("{label}_{idx}"),
            |lc| lc + lane.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + expected.get_variable(),
        );
    }
    *cursor += digest.len();
    Ok(())
}

fn field_to_spartan(value: F) -> SpartanF {
    SpartanF::from_canonical_u64(value.as_canonical_u64())
}

fn superneo_ivc_states_match(left: &SuperNeoIvcState, right: &SuperNeoIvcState) -> bool {
    left.chunk_count == right.chunk_count
        && left.step_count == right.step_count
        && left.transcript == right.transcript
        && left.carry.claims == right.carry.claims
        && left.carry.witnesses == right.carry.witnesses
}
