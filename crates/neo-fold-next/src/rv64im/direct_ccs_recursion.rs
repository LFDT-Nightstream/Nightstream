//! Owns the direct-CCS diagnostic Spartan bridge.
//!
//! This module reuses the existing RV64IM NIFS.V gadgets for non-VM probes.
//! It does not define a second folding verifier.

use std::time::Instant;

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::optimized_engine::PiCcsReplayProofWitness;
use neo_reductions::engines::utils::{build_dims_and_policy, digest_ccs_matrices_with_sparse_cache, Dims};
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
use super::main_relation_circuit::claim::{me_input_projection_digest_poseidon, packed_bytes_field_values};
use super::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use super::main_relation_circuit::witness::alloc_packed_witness;
use super::main_relation_spartan::{synthesize_direct_ccs_nifs_chunk, Rv64imClaimBundle};
use super::main_relation_trace::{
    build_rv64im_main_circuit_chunk_replay_surface, build_rv64im_main_circuit_pi_ccs_replay_surface,
    Rv64imMainCircuitChunkCover, Rv64imMainCircuitChunkReplaySurface, Rv64imMainCircuitHandoff,
    Rv64imMainCircuitPublicInputLayout,
};
use crate::finalize::{digest_fields_as_digest32, public_chunk_digest};
use crate::proof::{partition_prover_step_inputs, Carry, PackagedProof, ProverChunkInput, StepInput};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectCcsRecursionSnarkProof {
    pub nifs_snark_data: Vec<u8>,
}

impl DirectCcsRecursionSnarkProof {
    pub fn snark_bytes_len(&self) -> usize {
        self.nifs_snark_data.len()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DirectCcsRecursionSnarkPerf {
    pub nifs_setup_ms: f64,
    pub nifs_prep_ms: f64,
    pub nifs_prove_ms: f64,
    pub nifs_encode_ms: f64,
    pub nifs_verify_ms: f64,
    pub final_ce_setup_ms: f64,
    pub final_ce_prove_ms: f64,
    pub final_ce_verify_ms: f64,
    pub total_prove_ms: f64,
    pub total_verify_ms: f64,
    pub nifs_r1cs_sizes: [usize; 10],
    pub final_ce_r1cs_sizes: [usize; 10],
    pub nifs_r1cs_nnz: usize,
    pub final_ce_r1cs_nnz: usize,
    pub nifs_pcs_ms: f64,
    pub final_proof_bytes: usize,
    pub snark_bytes: usize,
    pub nifs_public_inputs: usize,
    pub nifs_chunk_constraints_first4: [usize; 4],
    pub nifs_chunk_constraints_by_chunk: Vec<usize>,
    pub nifs_chunk_count: usize,
    pub nifs_public_link_constraints: usize,
    pub nifs_chunk_done_constraints: usize,
    pub nifs_final_claim_digest_constraints: usize,
    pub skipped_final_claim_digest_constraints: usize,
    pub final_ce_digest_attribution: FinalCeProjectionDigestAttribution,
    pub final_ce_digest_constraints: usize,
    pub final_ce_relation_constraints: usize,
    pub final_ce_relation_breakdown: PaperCeRelationConstraintBreakdown,
    pub final_ce_digest_match_constraints: usize,
    pub final_ce_public_inputs: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FinalCeProjectionDigestAttribution {
    pub children: usize,
    pub fields_per_child: FinalCeProjectionDigestFields,
    pub total_fields: usize,
    pub explicit_public_fields: usize,
    pub poseidon_rate: usize,
    pub poseidon_width: usize,
    pub permutations_per_child: usize,
    pub total_permutations: usize,
    pub constraints_per_child: f64,
    pub effective_rows_per_permutation: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FinalCeProjectionDigestFields {
    pub domain_tag: usize,
    pub commitment_c: usize,
    pub x: usize,
    pub r: usize,
    pub y_ring: usize,
    pub aux: usize,
    pub total: usize,
}

#[derive(Debug, Error)]
pub enum DirectCcsRecursionSnarkError {
    #[error("direct CCS recursion input error: {0}")]
    Input(String),
    #[error("direct CCS recursion synthesis failed: {0}")]
    Synthesis(String),
    #[error("direct CCS recursion setup failed: {0}")]
    Setup(String),
    #[error("direct CCS recursion prepare failed: {0}")]
    Prepare(String),
    #[error("direct CCS recursion prove failed: {0}")]
    Prove(String),
    #[error("direct CCS recursion verify failed: {0}")]
    Verify(String),
    #[error("direct CCS recursion proof encoding failed: {0}")]
    Encode(String),
    #[error("direct CCS recursion proof decoding failed: {0}")]
    Decode(String),
    #[error("direct CCS recursion public IO mismatch")]
    PublicIoMismatch,
}

struct DirectCcsRecursionKeys {
    nifs_pk: Rv64imDeciderProverKey,
    nifs_vk: Rv64imDeciderVerifierKey,
}

#[derive(Clone)]
struct DirectCcsChunkCircuitSurface {
    cover: Rv64imMainCircuitChunkCover,
    replay: Rv64imMainCircuitChunkReplaySurface,
}

#[derive(Clone)]
struct DirectCcsNifsCircuit {
    params: NeoParams,
    structure: CcsStructure<F>,
    dims: Dims,
    mat_digest: [Goldilocks; 4],
    chunks: Vec<DirectCcsChunkCircuitSurface>,
    final_claims: Vec<CeClaim<Commitment, F, K>>,
    final_witnesses: Vec<CcsWitness<F>>,
}

impl DirectCcsNifsCircuit {
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

impl SpartanCircuit<Rv64imDeciderEngine> for DirectCcsNifsCircuit {
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
        let mut transcript =
            Poseidon2TranscriptCircuit::new(cs.namespace(|| "session_transcript"), b"neo.fold.next/session")?;
        let mut carried = Rv64imClaimBundle::from_effective_claims(Vec::new());

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

        if carried.effective_count() != self.final_claims.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        if self.final_witnesses.len() != carried.effective_count() {
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

pub fn prove_direct_ccs_recursion_snark_with_perf(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    packaged: &PackagedProof,
    final_carry: &Carry,
    steps: &[StepInput],
) -> Result<(DirectCcsRecursionSnarkProof, DirectCcsRecursionSnarkPerf), DirectCcsRecursionSnarkError> {
    if packaged.statement.final_main_claims != final_carry.claims {
        return Err(DirectCcsRecursionSnarkError::Input(
            "packaged final claims do not match final carry claims".into(),
        ));
    }
    if final_carry.claims.len() != final_carry.witnesses.len() {
        return Err(DirectCcsRecursionSnarkError::Input(
            "final carry requires one witness per claim".into(),
        ));
    }
    let final_witnesses = final_carry_witnesses(&final_carry.witnesses)?;
    let input_chunks = partition_prover_step_inputs(packaged.statement.fold_schedule, steps.to_vec())
        .map_err(|err| DirectCcsRecursionSnarkError::Input(err.to_string()))?;
    let circuit = build_direct_ccs_nifs_circuit(params, structure, packaged, &input_chunks, final_witnesses)?;
    let nifs_constraint_breakdown = measure_direct_ccs_nifs_constraints(&circuit)?;

    let setup_started = Instant::now();
    let (nifs_pk, nifs_vk) = Rv64imDeciderSnark::setup(circuit.clone())
        .map_err(|err| DirectCcsRecursionSnarkError::Setup(err.to_string()))?;
    let nifs_setup_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;
    let nifs_sizes = nifs_pk.sizes();
    let nifs_stats = nifs_pk.shape_debug_stats();

    let keys = DirectCcsRecursionKeys { nifs_pk, nifs_vk };
    let mut perf = DirectCcsRecursionSnarkPerf {
        nifs_setup_ms,
        final_ce_setup_ms: 0.0,
        nifs_r1cs_sizes: nifs_sizes,
        final_ce_r1cs_sizes: [0; 10],
        nifs_r1cs_nnz: nifs_stats.total_nnz,
        final_ce_r1cs_nnz: 0,
        nifs_public_inputs: nifs_constraint_breakdown.public_inputs,
        nifs_chunk_constraints_first4: nifs_constraint_breakdown.chunk_constraints_first4,
        nifs_chunk_constraints_by_chunk: nifs_constraint_breakdown.chunk_constraints_by_chunk.clone(),
        nifs_chunk_count: nifs_constraint_breakdown.chunk_count,
        nifs_public_link_constraints: nifs_constraint_breakdown.public_link_constraints,
        nifs_chunk_done_constraints: nifs_constraint_breakdown.chunk_done_constraints,
        nifs_final_claim_digest_constraints: nifs_constraint_breakdown.final_claim_digest_constraints,
        skipped_final_claim_digest_constraints: nifs_constraint_breakdown.skipped_final_claim_digest_constraints,
        final_ce_digest_attribution: final_ce_projection_digest_attribution(
            &circuit.final_claims,
            nifs_constraint_breakdown.skipped_final_claim_digest_constraints,
        ),
        final_ce_digest_constraints: 0,
        final_ce_relation_constraints: nifs_constraint_breakdown.final_ce_relation_constraints,
        final_ce_relation_breakdown: nifs_constraint_breakdown.final_ce_relation_breakdown,
        final_ce_digest_match_constraints: 0,
        final_ce_public_inputs: 0,
        ..DirectCcsRecursionSnarkPerf::default()
    };

    let prep_started = Instant::now();
    let prep = Rv64imDeciderSnark::prep_prove(&keys.nifs_pk, circuit.clone(), false)
        .map_err(|err| DirectCcsRecursionSnarkError::Prepare(err.to_string()))?;
    perf.nifs_prep_ms = prep_started.elapsed().as_secs_f64() * 1_000.0;

    let nifs_prove_started = Instant::now();
    let (nifs_proof, nifs_spartan_perf) =
        Rv64imDeciderSnark::prove_with_perf(&keys.nifs_pk, circuit.clone(), &prep, false)
            .map_err(|err| DirectCcsRecursionSnarkError::Prove(err.to_string()))?;
    perf.nifs_prove_ms = nifs_prove_started.elapsed().as_secs_f64() * 1_000.0;
    perf.nifs_pcs_ms = nifs_spartan_perf.pcs_prove_ms;

    let encode_started = Instant::now();
    let nifs_snark_data =
        bincode::serialize(&nifs_proof).map_err(|err| DirectCcsRecursionSnarkError::Encode(err.to_string()))?;
    perf.nifs_encode_ms = encode_started.elapsed().as_secs_f64() * 1_000.0;

    let proof = DirectCcsRecursionSnarkProof { nifs_snark_data };

    let nifs_verify_started = Instant::now();
    verify_direct_ccs_nifs_proof(&keys.nifs_vk, &circuit, &proof.nifs_snark_data)?;
    perf.nifs_verify_ms = nifs_verify_started.elapsed().as_secs_f64() * 1_000.0;

    perf.total_prove_ms = perf.nifs_prep_ms + perf.nifs_prove_ms + perf.nifs_encode_ms + perf.final_ce_prove_ms;
    perf.total_verify_ms = perf.nifs_verify_ms + perf.final_ce_verify_ms;
    perf.snark_bytes = proof.snark_bytes_len();
    perf.final_proof_bytes = bincode::serialize(&proof)
        .map_err(|err| DirectCcsRecursionSnarkError::Encode(err.to_string()))?
        .len();
    Ok((proof, perf))
}

fn build_direct_ccs_nifs_circuit(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    packaged: &PackagedProof,
    input_chunks: &[ProverChunkInput],
    final_witnesses: Vec<CcsWitness<F>>,
) -> Result<DirectCcsNifsCircuit, DirectCcsRecursionSnarkError> {
    if packaged.proof.session.chunks.len() != input_chunks.len() {
        return Err(DirectCcsRecursionSnarkError::Input(
            "packaged chunk count does not match direct step partition".into(),
        ));
    }
    let dims =
        build_dims_and_policy(params, structure).map_err(|err| DirectCcsRecursionSnarkError::Input(err.to_string()))?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, None)
        .try_into()
        .map_err(|digest: Vec<Goldilocks>| {
            DirectCcsRecursionSnarkError::Input(format!("expected 4 matrix digest limbs, got {}", digest.len()))
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
            return Err(DirectCcsRecursionSnarkError::Input(format!(
                "public chunk {chunk_index} differs between statement, proof, and direct step partition"
            )));
        }
        chunks.push(build_direct_ccs_chunk_surface(
            dims,
            chunk_index,
            proof_chunk,
            input_chunk,
        )?);
    }
    Ok(DirectCcsNifsCircuit {
        params: params.clone(),
        structure: structure.clone(),
        dims,
        mat_digest,
        chunks,
        final_claims: packaged.statement.final_main_claims.clone(),
        final_witnesses,
    })
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct DirectCcsNifsConstraintBreakdown {
    public_inputs: usize,
    chunk_count: usize,
    chunk_constraints_first4: [usize; 4],
    chunk_constraints_by_chunk: Vec<usize>,
    public_link_constraints: usize,
    chunk_done_constraints: usize,
    final_claim_digest_constraints: usize,
    skipped_final_claim_digest_constraints: usize,
    final_ce_relation_constraints: usize,
    final_ce_relation_breakdown: PaperCeRelationConstraintBreakdown,
}

fn measure_direct_ccs_nifs_constraints(
    circuit: &DirectCcsNifsCircuit,
) -> Result<DirectCcsNifsConstraintBreakdown, DirectCcsRecursionSnarkError> {
    let mut cs = ShapeCS::<Rv64imDeciderEngine>::new();
    let public_values = circuit
        .public_values()
        .map_err(|err| DirectCcsRecursionSnarkError::Synthesis(err.to_string()))?;
    let public_inputs = public_values
        .into_iter()
        .enumerate()
        .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("public_{idx}")), || Ok(value)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| DirectCcsRecursionSnarkError::Synthesis(err.to_string()))?;
    let mut out = DirectCcsNifsConstraintBreakdown {
        public_inputs: public_inputs.len(),
        chunk_count: circuit.chunks.len(),
        ..DirectCcsNifsConstraintBreakdown::default()
    };
    let mut public_cursor = 0usize;
    let mut transcript =
        Poseidon2TranscriptCircuit::new(cs.namespace(|| "session_transcript"), b"neo.fold.next/session")
            .map_err(|err| DirectCcsRecursionSnarkError::Synthesis(err.to_string()))?;
    let mut carried = Rv64imClaimBundle::from_effective_claims(Vec::new());

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
        .map_err(|err| DirectCcsRecursionSnarkError::Synthesis(err.to_string()))?;
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
        .map_err(|err| DirectCcsRecursionSnarkError::Synthesis(err.to_string()))?;
        out.public_link_constraints += cs.num_constraints() - before_link;

        let before_done = cs.num_constraints();
        transcript
            .append_message(
                cs.namespace(|| format!("chunk_{chunk_index}_done")),
                b"neo.fold.next/chunk_done",
                &[1],
            )
            .map_err(|err| DirectCcsRecursionSnarkError::Synthesis(err.to_string()))?;
        out.chunk_done_constraints += cs.num_constraints() - before_done;
        carried = next;
    }

    if carried.effective_count() != circuit.final_claims.len() {
        return Err(DirectCcsRecursionSnarkError::Synthesis(
            "measured direct NIFS carried claim count does not match final claims".into(),
        ));
    }
    if circuit.final_witnesses.len() != carried.effective_count() {
        return Err(DirectCcsRecursionSnarkError::Synthesis(
            "measured direct NIFS final witness count does not match final claims".into(),
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
        .map_err(|err| DirectCcsRecursionSnarkError::Synthesis(err.to_string()))?;
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
        .map_err(|err| DirectCcsRecursionSnarkError::Synthesis(err.to_string()))?;
        out.final_ce_relation_breakdown.add_assign(breakdown);
    }
    out.final_ce_relation_constraints = cs.num_constraints() - before_final_ce;
    if public_cursor != public_inputs.len() {
        return Err(DirectCcsRecursionSnarkError::Synthesis(
            "measured direct NIFS public cursor mismatch".into(),
        ));
    }
    let before_skipped_digest = cs.num_constraints();
    for (claim_index, claim) in carried.effective_claims().iter().enumerate() {
        let _ = me_input_projection_digest_poseidon(
            &mut cs.namespace(|| format!("skipped_final_claim_{claim_index}_digest")),
            claim,
            &format!("skipped_final_claim_{claim_index}_digest"),
        )
        .map_err(|err| DirectCcsRecursionSnarkError::Synthesis(err.to_string()))?;
    }
    out.skipped_final_claim_digest_constraints = cs.num_constraints() - before_skipped_digest;
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

fn final_ce_projection_digest_attribution(
    claims: &[CeClaim<Commitment, F, K>],
    total_constraints: usize,
) -> FinalCeProjectionDigestAttribution {
    let fields_per_child = claims
        .first()
        .map(final_ce_projection_digest_fields)
        .unwrap_or_default();
    let total_fields = claims
        .iter()
        .map(final_ce_projection_digest_fields)
        .map(|fields| fields.total)
        .sum();
    let explicit_public_fields = claims.iter().map(final_ce_explicit_public_fields).sum();
    let poseidon_rate = neo_params::poseidon2_goldilocks::RATE;
    let poseidon_width = neo_params::poseidon2_goldilocks::WIDTH;
    let permutations_per_child = if fields_per_child.total == 0 {
        0
    } else {
        fields_per_child.total.div_ceil(poseidon_rate) + 1
    };
    let total_permutations = claims
        .iter()
        .map(final_ce_projection_digest_fields)
        .map(|fields| fields.total.div_ceil(poseidon_rate) + 1)
        .sum::<usize>();
    let children = claims.len();
    let constraints_per_child = if children == 0 {
        0.0
    } else {
        total_constraints as f64 / children as f64
    };
    let effective_rows_per_permutation = if total_permutations == 0 {
        0.0
    } else {
        total_constraints as f64 / total_permutations as f64
    };
    FinalCeProjectionDigestAttribution {
        children,
        fields_per_child,
        total_fields,
        explicit_public_fields,
        poseidon_rate,
        poseidon_width,
        permutations_per_child,
        total_permutations,
        constraints_per_child,
        effective_rows_per_permutation,
    }
}

fn final_ce_projection_digest_fields(claim: &CeClaim<Commitment, F, K>) -> FinalCeProjectionDigestFields {
    let domain_tag = packed_bytes_field_values(b"neo/ccs/me_input_projection_digest_poseidon/v2").len();
    let commitment_c = 1 + claim.c.data.len();
    let x = 1 + claim.m_in;
    let r = k_slice_digest_fields(claim.r.len());
    let y_ring = 1 + claim
        .y_ring
        .iter()
        .map(|row| k_slice_digest_fields(row.len()))
        .sum::<usize>();
    let aux = 0;
    let total = domain_tag + commitment_c + x + r + y_ring + aux;
    FinalCeProjectionDigestFields {
        domain_tag,
        commitment_c,
        x,
        r,
        y_ring,
        aux,
        total,
    }
}

fn final_ce_explicit_public_fields(claim: &CeClaim<Commitment, F, K>) -> usize {
    claim.c.data.len() + claim.m_in + (2 * claim.r.len()) + claim.y_ring.iter().map(|row| 2 * row.len()).sum::<usize>()
}

fn k_slice_digest_fields(len: usize) -> usize {
    2 + (2 * len)
}

fn build_direct_ccs_chunk_surface(
    dims: Dims,
    chunk_index: usize,
    proof_chunk: &crate::proof::ChunkProof,
    input_chunk: &ProverChunkInput,
) -> Result<DirectCcsChunkCircuitSurface, DirectCcsRecursionSnarkError> {
    let replay_proof = PiCcsReplayProofWitness::from_proof(&proof_chunk.ccs_proof)
        .map_err(|err| DirectCcsRecursionSnarkError::Input(err.to_string()))?;
    let (row_chals, alpha_prime) = split_challenges(&proof_chunk.ccs_proof.sumcheck_challenges, dims.ell_n, "FE")?;
    let (s_col, alpha_prime_nc) = split_challenges(&proof_chunk.ccs_proof.sumcheck_challenges_nc, dims.ell_m, "NC")?;
    let public_chunk_instance_digest = public_chunk_digest(&input_chunk.public_chunk);
    let public_chunk_digest32 = digest_fields_as_digest32(public_chunk_instance_digest);
    let handoff = Rv64imMainCircuitHandoff {
        public_chunk: input_chunk.public_chunk.clone(),
        public_chunk_instance_digest,
        public_chunk_digest: public_chunk_digest32,
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
    .map_err(|err| {
        DirectCcsRecursionSnarkError::Input(format!(
            "failed to build direct chunk {chunk_index} replay surface: {err}"
        ))
    })?;
    let cover = Rv64imMainCircuitChunkCover::from_replay_surface(&replay);
    Ok(DirectCcsChunkCircuitSurface { cover, replay })
}

fn split_challenges(
    values: &[K],
    prefix_len: usize,
    label: &str,
) -> Result<(Vec<K>, Vec<K>), DirectCcsRecursionSnarkError> {
    if values.len() < prefix_len {
        return Err(DirectCcsRecursionSnarkError::Input(format!(
            "{label} sumcheck challenge vector too short: got {}, need prefix {prefix_len}",
            values.len()
        )));
    }
    Ok((values[..prefix_len].to_vec(), values[prefix_len..].to_vec()))
}

fn final_carry_witnesses(zs: &[Mat<F>]) -> Result<Vec<CcsWitness<F>>, DirectCcsRecursionSnarkError> {
    zs.iter()
        .enumerate()
        .map(|(idx, z)| {
            if z.rows() != D {
                return Err(DirectCcsRecursionSnarkError::Input(format!(
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

fn verify_direct_ccs_nifs_proof(
    vk: &Rv64imDeciderVerifierKey,
    circuit: &DirectCcsNifsCircuit,
    snark_data: &[u8],
) -> Result<(), DirectCcsRecursionSnarkError> {
    let proof: Rv64imDeciderSnark =
        bincode::deserialize(snark_data).map_err(|err| DirectCcsRecursionSnarkError::Decode(err.to_string()))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| DirectCcsRecursionSnarkError::Verify(err.to_string()))?;
    let expected = circuit
        .public_values()
        .map_err(|err| DirectCcsRecursionSnarkError::Verify(err.to_string()))?;
    if public_values != expected {
        return Err(DirectCcsRecursionSnarkError::PublicIoMismatch);
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
