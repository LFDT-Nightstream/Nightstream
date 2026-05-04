//! Owns Spartan compression for the native RV32IM IVC carrier.
//!
//! The compressed verifier boundary is the final Construction-2 verifier
//! surface: final `U_i` CE satisfiability plus the terminal `F'` committed-step
//! relation. The terminal `F'` proof binds `u_i.C` to the same SuperNeo-packed
//! low-norm R2 assignment whose rows it checks. No terminal chunk-step proof or
//! native replay is accepted as a verifier fallback.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};

mod terminal_f_prime_committed;

use self::terminal_f_prime_committed::{
    debug_check_rv32im_terminal_f_prime_r1cs_ccs_relation, terminal_f_prime_committed_step_boundary_public_values,
    Rv32imTerminalFPrimeCommittedRelation, Rv32imTerminalFPrimeCommittedStepSetup,
};
use crate::rv32im::chunk_step_ivc::build_rv32im_chunk_step_ivc_relations;
use crate::rv32im::final_relation::{
    rv32im_recursive_accumulator_instance_digest_from_parts, Rv32imFinalBuildProof, Rv32imFinalStatement,
};
use crate::rv32im::ivc::{
    build_rv32im_ivc_prover_state_from_relations, derive_rv32im_ivc_step_cap, Rv32imIvcPublicImage, Rv32imIvcState,
};
use crate::rv32im::main_relation_spartan::{
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    build_rv32im_main_recursion_step_spartan_published_target, debug_check_rv32im_main_recursion_step_spartan_circuit,
    terminal_f_prime_r2_public_values_from_parts, Rv32imMainRecursionFPrimeBackendRelation,
    Rv32imMainRecursionStepSpartanShape,
};
use crate::rv32im::SimpleKernelError;
#[allow(unused_imports)]
pub(crate) use crate::spartan_backend::{
    hash_packed_goldilocks_fields, DigestHelperTrait, GoldilocksP3MerkleMleEngine, R1CSSNARKTrait, Rv32imDeciderEngine,
    Rv32imDeciderProverKey, Rv32imDeciderSnark, Rv32imDeciderVerifierKey, ShapeCS, SpartanCircuit, SpartanF,
    SpartanProverKey, SpartanShape, SpartanVerifierKey, SplitR1CSShape, R1CSSNARK,
};
use crate::superneo_circuit::ce_spartan::{
    prove_rv32im_ce_bundle_relation, setup_rv32im_ce_bundle_relation, verify_rv32im_ce_bundle_relation,
    Rv32imCeBundleProof,
};
use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{F, K};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

pub struct Rv32imIvcSnarkProverKey {
    terminal_f_prime: Rv32imDeciderProverKey,
    final_ce: Rv32imDeciderProverKey,
}

pub struct Rv32imIvcSnarkVerifierKey {
    terminal_f_prime: Rv32imDeciderVerifierKey,
    final_ce: Rv32imDeciderVerifierKey,
}

impl Rv32imIvcSnarkVerifierKey {
    pub fn expected_digest(&self) -> Result<[u8; 32], SimpleKernelError> {
        let terminal_f_prime_digest = self.terminal_f_prime.digest().map_err(|err| {
            SimpleKernelError::Bridge(format!("RV32IM IVC terminal F' verifier key digest failed: {err}"))
        })?;
        let final_ce_digest = self.final_ce.digest().map_err(|err| {
            SimpleKernelError::Bridge(format!("RV32IM IVC final CE verifier key digest failed: {err}"))
        })?;

        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/ivc_snark_verifier_key");
        tr.append_message(b"neo.fold.next/rv32im/ivc_snark_verifier_key/version", b"v1");
        tr.append_message(
            b"neo.fold.next/rv32im/ivc_snark_verifier_key/terminal_f_prime",
            &terminal_f_prime_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv32im/ivc_snark_verifier_key/final_ce",
            &final_ce_digest,
        );
        Ok(tr.digest32())
    }
}

impl Rv32imIvcSnarkProverKey {
    pub fn sizes(&self) -> [usize; 10] {
        self.terminal_f_prime.sizes()
    }

    pub fn shape_debug_stats(&self) -> spartan2::SplitR1CSShapeDebugStats {
        self.terminal_f_prime.shape_debug_stats()
    }
}

pub type Rv32imIvcSnarkKeyPair = Arc<(Rv32imIvcSnarkProverKey, Rv32imIvcSnarkVerifierKey)>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imIvcRecursionSnarkSetupShape {
    main_recursion_step_shape: Rv32imMainRecursionStepSpartanShape,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imTerminalFPrimeCommittedStepShape {
    pub terminal_r2_source_ccs_rows: usize,
    pub terminal_r2_source_ccs_cols: usize,
    pub terminal_r2_source_ccs_nnz: usize,
    pub terminal_r2_public_inputs: usize,
    pub terminal_r2_witness_inputs: usize,
    pub terminal_r2_private_padding_inputs: usize,
    pub terminal_r2_private_bit_inputs: usize,
    pub terminal_r2_private_u32_inputs: usize,
    pub terminal_r2_private_u64_inputs: usize,
    pub terminal_r2_private_low_norm_bit_inputs: usize,
    pub terminal_r2_committed_low_norm_width: usize,
    pub terminal_r2_superneo_packed_cols: usize,
    pub terminal_r2_commitment_words: usize,
    pub terminal_committed_step_public_inputs: usize,
    pub terminal_committed_step_constraints: usize,
    pub terminal_f_prime_r1cs_public_inputs: usize,
    pub terminal_f_prime_r1cs_challenges: usize,
    pub terminal_f_prime_r1cs_variables: usize,
    pub terminal_f_prime_r1cs_constraints: usize,
    pub terminal_f_prime_r1cs_nnz: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv32imTerminalFPrimeCommittedStepProof {
    pub snark_data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv32imIvcSnarkProof {
    pub terminal_f_prime_committed_step_proof: Rv32imTerminalFPrimeCommittedStepProof,
    pub final_main_claims: Vec<CeClaim<Commitment, F, K>>,
    pub final_ce_proof: Rv32imCeBundleProof,
}

impl Rv32imIvcSnarkProof {
    pub fn snark_bytes_len(&self) -> usize {
        self.terminal_f_prime_committed_step_proof.snark_data.len() + self.final_ce_proof.snark_data.len()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv32imIvcSnark {
    proof: Rv32imIvcSnarkProof,
    public_image: Rv32imIvcPublicImage,
}

static RV32IM_IVC_SNARK_PROOF_CACHE: OnceLock<Mutex<HashMap<[u8; 32], Arc<Rv32imIvcSnarkProof>>>> = OnceLock::new();
static RV32IM_IVC_SNARK_SETUP_CACHE: OnceLock<Mutex<HashMap<[u8; 32], Rv32imIvcSnarkKeyPair>>> = OnceLock::new();

type Rv32imIvcSnarkTrace<'a> = Option<&'a mut dyn FnMut(&str)>;

fn trace_emit(trace: &mut Rv32imIvcSnarkTrace<'_>, message: &str) {
    if let Some(emit) = trace.as_deref_mut() {
        emit(message);
    }
}

fn trace_emit_owned(trace: &mut Rv32imIvcSnarkTrace<'_>, message: String) {
    trace_emit(trace, &message);
}

fn trace_start(trace: &mut Rv32imIvcSnarkTrace<'_>, phase: &str) -> Instant {
    trace_emit_owned(trace, format!("{phase}.start"));
    Instant::now()
}

fn trace_done(trace: &mut Rv32imIvcSnarkTrace<'_>, phase: &str, started: Instant) {
    trace_emit_owned(
        trace,
        format!("{phase}.done_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0),
    );
}

fn ccs_structure_nnz(structure: &CcsStructure<F>) -> usize {
    structure
        .matrices
        .iter()
        .map(|matrix| match matrix.as_csc() {
            Some(csc) => csc.vals.len(),
            None => matrix.rows(),
        })
        .sum()
}

fn rv32im_ivc_snark_setup_cache_key(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    final_ce_claims: &[CeClaim<Commitment, F, K>],
    terminal_setup: &Rv32imTerminalFPrimeCommittedStepSetup,
) -> Result<[u8; 32], SimpleKernelError> {
    let terminal_r2 = terminal_setup.r1cs_ccs();
    let (private_bit_inputs, private_u32_inputs, private_u64_inputs) =
        terminal_setup.terminal_r2_private_encoding_counts();
    let private_padding_inputs = terminal_setup.terminal_r2_private_padding_inputs();
    let private_low_norm_bit_inputs = private_bit_inputs + (private_u32_inputs * 32) + (private_u64_inputs * 64);
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/ivc_recursion_snark_setup_cache");
    tr.append_message(b"neo.fold.next/rv32im/ivc_recursion_snark_setup_cache/version", b"v10");
    tr.append_message(
        b"neo.fold.next/rv32im/ivc_recursion_snark_setup_cache/recursive_shape",
        &spartan_shape.expected_digest(),
    );
    let mut final_ce_shape = Vec::with_capacity(1 + final_ce_claims.len() * 10);
    final_ce_shape.push(final_ce_claims.len() as u64);
    for claim in final_ce_claims {
        final_ce_shape.extend([
            claim.c.data.len() as u64,
            claim.X.rows() as u64,
            claim.X.cols() as u64,
            claim.r.len() as u64,
            claim.y_ring.len() as u64,
            claim.m_in as u64,
        ]);
        final_ce_shape.extend(claim.y_ring.iter().map(|row| row.len() as u64));
    }
    tr.append_u64s(
        b"neo.fold.next/rv32im/ivc_recursion_snark_setup_cache/final_ce_shape",
        &final_ce_shape,
    );
    tr.append_u64s(
        b"neo.fold.next/rv32im/ivc_recursion_snark_setup_cache/terminal_committed_step_shape",
        &[
            terminal_setup.step_cap() as u64,
            terminal_r2.structure().n as u64,
            terminal_r2.structure().m as u64,
            ccs_structure_nnz(terminal_r2.structure()) as u64,
            terminal_r2.total_nnz() as u64,
            terminal_r2.num_public() as u64,
            terminal_setup.r2_witness_len() as u64,
            private_padding_inputs as u64,
            private_bit_inputs as u64,
            private_u32_inputs as u64,
            private_u64_inputs as u64,
            private_low_norm_bit_inputs as u64,
            terminal_setup.terminal_r2_committed_low_norm_width()? as u64,
            terminal_setup.terminal_r2_superneo_packed_cols() as u64,
            terminal_setup.terminal_r2_commitment_words() as u64,
        ],
    );
    Ok(tr.digest32())
}

fn rv32im_ivc_snark_cache_key(statement: &Rv32imFinalStatement, proof: &Rv32imFinalBuildProof) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/ivc_recursion_snark_cache");
    tr.append_message(b"neo.fold.next/rv32im/ivc_recursion_snark_cache/version", b"v3");
    tr.append_message(
        b"neo.fold.next/rv32im/ivc_recursion_snark_cache/final_statement_digest",
        &statement.digest,
    );
    tr.append_message(
        b"neo.fold.next/rv32im/ivc_recursion_snark_cache/final_proof_digest",
        &proof.proof_digest,
    );
    tr.digest32()
}

fn rv32im_terminal_f_prime_r2_public_values_from_public_image(public_image: &Rv32imIvcPublicImage) -> Vec<SpartanF> {
    terminal_f_prime_r2_public_values_from_parts(
        public_image.vk_fs_digest,
        public_image.chunk_count,
        public_image.z_0,
        public_image.z_i,
        public_image.pc,
        &public_image.x_i,
        public_image.folded_accumulator_digest,
        public_image.terminal_bridge_handoff_digest,
        public_image.terminal_verified_step_statement_digest,
    )
}

fn build_rv32im_ivc_prover_state_from_final(
    statement: &Rv32imFinalStatement,
    proof: &Rv32imFinalBuildProof,
) -> Result<Rv32imIvcState, SimpleKernelError> {
    let relations = build_rv32im_chunk_step_ivc_relations(statement, proof)?;
    let step_cap = derive_rv32im_ivc_step_cap(
        statement.folded.fold_schedule,
        usize::try_from(statement.folded.semantic_step_count).map_err(|_| {
            SimpleKernelError::Bridge(
                "RV32IM IVC SNARK recursion step_count does not fit into the native step-cap model".into(),
            )
        })?,
    )?;
    build_rv32im_ivc_prover_state_from_relations(&relations, step_cap)
}

fn build_rv32im_ivc_recursion_backend_from_state(
    state: &Rv32imIvcState,
) -> Result<
    (
        Rv32imMainRecursionStepSpartanShape,
        Rv32imMainRecursionFPrimeBackendRelation,
    ),
    SimpleKernelError,
> {
    let mut trace = None;
    build_rv32im_ivc_recursion_backend_from_state_with_trace(state, &mut trace)
}

fn build_rv32im_ivc_recursion_backend_from_state_with_trace(
    state: &Rv32imIvcState,
    trace: &mut Rv32imIvcSnarkTrace<'_>,
) -> Result<
    (
        Rv32imMainRecursionStepSpartanShape,
        Rv32imMainRecursionFPrimeBackendRelation,
    ),
    SimpleKernelError,
> {
    let started = trace_start(trace, "ivc_backend.phase=validate_current_surface");
    state.validate_current_surface_for_compression()?;
    trace_done(trace, "ivc_backend.phase=validate_current_surface", started);

    let started = trace_start(trace, "ivc_backend.phase=latest_relation_and_advice");
    let (relation, advice) = state.latest_relation_and_advice()?;
    trace_done(trace, "ivc_backend.phase=latest_relation_and_advice", started);

    let started = trace_start(trace, "ivc_backend.phase=build_spartan_shape_and_backend");
    let relations = [relation];
    let advices = [advice];
    let (spartan_shape, mut backend_relations) =
        build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(&relations, &advices)?;
    let backend_relation = backend_relations.pop().ok_or_else(|| {
        SimpleKernelError::Bridge("RV32IM IVC SNARK requires a latest recursive-step backend relation".into())
    })?;
    trace_done(trace, "ivc_backend.phase=build_spartan_shape_and_backend", started);
    Ok((spartan_shape, backend_relation))
}

fn ccs_witness_from_packed_z(z: &Mat<F>) -> Result<CcsWitness<F>, SimpleKernelError> {
    if z.rows() != neo_math::D {
        return Err(SimpleKernelError::Bridge(
            "RV32IM IVC final CE witness must use D rows".into(),
        ));
    }
    Ok(CcsWitness {
        w: Vec::new(),
        Z: z.clone(),
    })
}

fn final_ce_context(
    claim_count: usize,
) -> Result<(neo_params::NeoParams, &'static neo_ccs::CcsStructure<F>), SimpleKernelError> {
    let (params, _, structure) = crate::rv32im::kernel::rv32im_root_main_lane_context_for_claim_count(claim_count)?;
    Ok((params, structure))
}

fn canonical_final_ce_claim(claim: &CeClaim<Commitment, F, K>) -> CeClaim<Commitment, F, K> {
    CeClaim {
        c: claim.c.clone(),
        X: claim.X.clone(),
        r: claim.r.clone(),
        s_col: Vec::new(),
        y_ring: claim
            .y_ring
            .iter()
            .map(|row| row.iter().copied().take(neo_math::D).collect())
            .collect(),
        ct: Vec::new(),
        aux_openings: Vec::new(),
        y_zcol: Vec::new(),
        m_in: claim.m_in,
        fold_digest: [0; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}

fn canonical_final_ce_claims(claims: &[CeClaim<Commitment, F, K>]) -> Vec<CeClaim<Commitment, F, K>> {
    claims.iter().map(canonical_final_ce_claim).collect()
}

fn ensure_final_ce_claims_are_canonical(claims: &[CeClaim<Commitment, F, K>]) -> Result<(), SimpleKernelError> {
    for (idx, claim) in claims.iter().enumerate() {
        if !claim.s_col.is_empty()
            || !claim.ct.is_empty()
            || !claim.aux_openings.is_empty()
            || !claim.y_zcol.is_empty()
            || claim.fold_digest != [0; 32]
            || !claim.c_step_coords.is_empty()
            || claim.u_offset != 0
            || claim.u_len != 0
        {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM IVC final CE claim {idx} carries non-authoritative transport fields"
            )));
        }
        let expected_commitment_words = claim.c.kappa.checked_mul(neo_math::D).ok_or_else(|| {
            SimpleKernelError::Bridge(format!("RV32IM IVC final CE claim {idx} commitment shape overflows"))
        })?;
        if claim.c.d != neo_math::D || claim.c.kappa == 0 || claim.c.data.len() != expected_commitment_words {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM IVC final CE claim {idx} commitment shape is not canonical SuperNeo D x kappa"
            )));
        }
        if claim.X.rows() != neo_math::D || claim.X.cols() != claim.m_in {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM IVC final CE claim {idx} must use SuperNeo X shape {} x m_in, got {} x {} for m_in={}",
                neo_math::D,
                claim.X.rows(),
                claim.X.cols(),
                claim.m_in
            )));
        }
        for (matrix_idx, row) in claim.y_ring.iter().enumerate() {
            if row.len() != neo_math::D {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV32IM IVC final CE claim {idx} y_ring[{matrix_idx}] must carry exactly D SuperNeo coefficients"
                )));
            }
        }
    }
    Ok(())
}

fn trace_terminal_f_prime_committed_step_shape(
    trace: &mut Rv32imIvcSnarkTrace<'_>,
    terminal_setup: &Rv32imTerminalFPrimeCommittedStepSetup,
) -> Result<(), SimpleKernelError> {
    if trace.is_none() {
        return Ok(());
    }
    let shape = measure_terminal_f_prime_committed_step_shape_from_setup(terminal_setup)?;
    let r1cs_ccs = terminal_setup.r1cs_ccs();
    trace_emit_owned(
        trace,
        format!(
            "setup.shape.terminal_committed_step_public_inputs={}",
            shape.terminal_committed_step_public_inputs
        ),
    );
    trace_emit_owned(
        trace,
        format!(
            "setup.shape.terminal_committed_step_constraints={}",
            shape.terminal_committed_step_constraints
        ),
    );
    trace_emit_owned(
        trace,
        format!(
            "setup.shape.terminal_f_prime_r1cs_public_inputs={}",
            r1cs_ccs.num_spartan_public()
        ),
    );
    trace_emit_owned(
        trace,
        format!(
            "setup.shape.terminal_f_prime_r1cs_challenges={}",
            r1cs_ccs.num_challenges()
        ),
    );
    trace_emit_owned(
        trace,
        format!(
            "setup.shape.terminal_f_prime_r1cs_variables={}",
            r1cs_ccs.num_variables()
        ),
    );
    trace_emit_owned(
        trace,
        format!(
            "setup.shape.terminal_f_prime_r1cs_constraints={}",
            r1cs_ccs.num_constraints()
        ),
    );
    trace_emit_owned(
        trace,
        format!("setup.shape.terminal_f_prime_r1cs_nnz={}", r1cs_ccs.total_nnz()),
    );
    trace_emit_owned(
        trace,
        format!("setup.shape.terminal_r2_source_ccs_rows={}", r1cs_ccs.structure().n),
    );
    trace_emit_owned(
        trace,
        format!("setup.shape.terminal_r2_source_ccs_cols={}", r1cs_ccs.structure().m),
    );
    trace_emit_owned(
        trace,
        format!(
            "setup.shape.terminal_r2_source_ccs_nnz={}",
            ccs_structure_nnz(r1cs_ccs.structure())
        ),
    );
    trace_emit_owned(
        trace,
        format!(
            "setup.shape.terminal_r2_source_ccs_matrices={}",
            terminal_setup.r1cs_ccs().structure().t()
        ),
    );
    Ok(())
}

fn measure_terminal_f_prime_committed_step_shape_from_setup(
    terminal_setup: &Rv32imTerminalFPrimeCommittedStepSetup,
) -> Result<Rv32imTerminalFPrimeCommittedStepShape, SimpleKernelError> {
    let circuit = terminal_setup.committed_step_circuit();
    let committed_step_shape = ShapeCS::<Rv32imDeciderEngine>::r1cs_shape(&circuit)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM terminal F' committed-step shape failed: {err}")))?;
    let public_values = circuit.public_values().map_err(|err| {
        SimpleKernelError::Bridge(format!("RV32IM terminal F' committed-step public IO failed: {err}"))
    })?;
    let r1cs_ccs = terminal_setup.r1cs_ccs();
    let (private_bit_inputs, private_u32_inputs, private_u64_inputs) =
        terminal_setup.terminal_r2_private_encoding_counts();
    let private_padding_inputs = terminal_setup.terminal_r2_private_padding_inputs();
    let private_low_norm_bit_inputs = private_bit_inputs + (private_u32_inputs * 32) + (private_u64_inputs * 64);
    Ok(Rv32imTerminalFPrimeCommittedStepShape {
        terminal_r2_source_ccs_rows: r1cs_ccs.structure().n,
        terminal_r2_source_ccs_cols: r1cs_ccs.structure().m,
        terminal_r2_source_ccs_nnz: ccs_structure_nnz(r1cs_ccs.structure()),
        terminal_r2_public_inputs: r1cs_ccs.num_public(),
        terminal_r2_witness_inputs: terminal_setup.r2_witness_len(),
        terminal_r2_private_padding_inputs: private_padding_inputs,
        terminal_r2_private_bit_inputs: private_bit_inputs,
        terminal_r2_private_u32_inputs: private_u32_inputs,
        terminal_r2_private_u64_inputs: private_u64_inputs,
        terminal_r2_private_low_norm_bit_inputs: private_low_norm_bit_inputs,
        terminal_r2_committed_low_norm_width: terminal_setup.terminal_r2_committed_low_norm_width()?,
        terminal_r2_superneo_packed_cols: terminal_setup.terminal_r2_superneo_packed_cols(),
        terminal_r2_commitment_words: terminal_setup.terminal_r2_commitment_words(),
        terminal_committed_step_public_inputs: public_values.len(),
        terminal_committed_step_constraints: committed_step_shape.num_constraints(),
        terminal_f_prime_r1cs_public_inputs: r1cs_ccs.num_spartan_public(),
        terminal_f_prime_r1cs_challenges: r1cs_ccs.num_challenges(),
        terminal_f_prime_r1cs_variables: r1cs_ccs.num_variables(),
        terminal_f_prime_r1cs_constraints: r1cs_ccs.num_constraints(),
        terminal_f_prime_r1cs_nnz: r1cs_ccs.total_nnz(),
    })
}

pub(crate) fn debug_measure_rv32im_terminal_f_prime_committed_step_shape(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imTerminalFPrimeCommittedStepShape, SimpleKernelError> {
    let terminal_setup = Rv32imTerminalFPrimeCommittedStepSetup::from_backend_shape(spartan_shape, backend_relation)?;
    measure_terminal_f_prime_committed_step_shape_from_setup(&terminal_setup)
}

fn setup_rv32im_terminal_f_prime_committed_step_relation(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<(Rv32imDeciderProverKey, Rv32imDeciderVerifierKey), SimpleKernelError> {
    let terminal_setup = Rv32imTerminalFPrimeCommittedStepSetup::from_backend_shape(spartan_shape, backend_relation)?;
    setup_rv32im_terminal_f_prime_committed_step_relation_from_shape(&terminal_setup)
}

fn setup_rv32im_terminal_f_prime_committed_step_relation_from_shape(
    terminal_setup: &Rv32imTerminalFPrimeCommittedStepSetup,
) -> Result<(Rv32imDeciderProverKey, Rv32imDeciderVerifierKey), SimpleKernelError> {
    let circuit = terminal_setup.committed_step_circuit();
    Rv32imDeciderSnark::setup(circuit)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM terminal F' committed-step setup failed: {err}")))
}

fn prove_rv32im_terminal_f_prime_committed_step_relation_from_relation(
    pk: &Rv32imDeciderProverKey,
    terminal_relation: &Rv32imTerminalFPrimeCommittedRelation,
) -> Result<Rv32imTerminalFPrimeCommittedStepProof, SimpleKernelError> {
    let circuit = terminal_relation.committed_step_circuit()?;
    let prep = Rv32imDeciderSnark::prep_prove(pk, circuit.clone(), false)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM terminal F' committed-step prepare failed: {err}")))?;
    let proof = Rv32imDeciderSnark::prove(pk, circuit, &prep, false)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM terminal F' committed-step prove failed: {err}")))?;
    let snark_data = bincode::serialize(&proof).map_err(|err| {
        SimpleKernelError::Bridge(format!("RV32IM terminal F' committed-step encoding failed: {err}"))
    })?;
    Ok(Rv32imTerminalFPrimeCommittedStepProof { snark_data })
}

fn verify_rv32im_terminal_f_prime_committed_step_relation(
    vk: &Rv32imDeciderVerifierKey,
    proof: &Rv32imTerminalFPrimeCommittedStepProof,
) -> Result<Vec<SpartanF>, SimpleKernelError> {
    let proof: Rv32imDeciderSnark = bincode::deserialize(&proof.snark_data).map_err(|err| {
        SimpleKernelError::Bridge(format!("RV32IM terminal F' committed-step decoding failed: {err}"))
    })?;
    proof
        .verify(vk)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM terminal F' committed-step verify failed: {err}")))
}

fn setup_rv32im_final_ce_keys(
    claims: &[CeClaim<Commitment, F, K>],
    witnesses: &[Mat<F>],
) -> Result<(Rv32imDeciderProverKey, Rv32imDeciderVerifierKey), SimpleKernelError> {
    if claims.len() != witnesses.len() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM IVC final CE setup requires one witness per final carried claim".into(),
        ));
    }
    let (params, structure) = final_ce_context(claims.len())?;
    let claims = canonical_final_ce_claims(claims);
    let witnesses = witnesses
        .iter()
        .map(ccs_witness_from_packed_z)
        .collect::<Result<Vec<_>, _>>()?;
    setup_rv32im_ce_bundle_relation(&params, structure, &claims, &witnesses, F::from_u64(7))
        .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM IVC final CE bundle setup failed: {err}")))
}

fn verify_rv32im_final_ce_bundle(
    key: &Rv32imDeciderVerifierKey,
    claims: &[CeClaim<Commitment, F, K>],
    proof: &Rv32imCeBundleProof,
) -> Result<(), SimpleKernelError> {
    ensure_final_ce_claims_are_canonical(claims)?;
    verify_rv32im_ce_bundle_relation(key, claims, proof)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM IVC final CE bundle verify failed: {err}")))
}

fn ensure_final_claims_bind_public_image(
    public_image: &Rv32imIvcPublicImage,
    claims: &[CeClaim<Commitment, F, K>],
) -> Result<(), SimpleKernelError> {
    let digest = rv32im_recursive_accumulator_instance_digest_from_parts(claims, public_image.z_i);
    if digest != public_image.folded_accumulator_digest {
        return Err(SimpleKernelError::Bridge(
            "RV32IM IVC final CE claims do not bind to the folded accumulator digest in the public image".into(),
        ));
    }
    Ok(())
}

fn ensure_terminal_committed_step_public_values_bind_public_image(
    public_image: &Rv32imIvcPublicImage,
    public_values: &[SpartanF],
) -> Result<(), SimpleKernelError> {
    let mut expected_terminal_values = rv32im_terminal_f_prime_r2_public_values_from_public_image(public_image);
    expected_terminal_values.extend(terminal_f_prime_committed_step_boundary_public_values(
        &public_image.construction2_u_i,
    ));
    if public_values.len() != expected_terminal_values.len() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM terminal F' public vector length does not match the compressed public image".into(),
        ));
    }
    if public_values != expected_terminal_values.as_slice() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM terminal F' committed-step public IO does not match the compressed public image".into(),
        ));
    }
    Ok(())
}

fn ensure_recursion_backend_matches_public_image(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
    public_image: &Rv32imIvcPublicImage,
) -> Result<(), SimpleKernelError> {
    let actual_target = build_rv32im_main_recursion_step_spartan_published_target(backend_relation)
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    if actual_target.terminal_f_prime_r2_public_values()
        != rv32im_terminal_f_prime_r2_public_values_from_public_image(public_image)
    {
        return Err(SimpleKernelError::Bridge(
            "RV32IM IVC SNARK public image does not match the terminal recursive-step public IO".into(),
        ));
    }
    Ok(())
}

fn public_image_with_terminal_r2_boundary(
    public_image: Rv32imIvcPublicImage,
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imIvcPublicImage, SimpleKernelError> {
    let terminal_relation = terminal_committed_step_inputs_from_backend(spartan_shape, backend_relation)?;
    public_image_with_terminal_r2_boundary_from_relation(public_image, backend_relation, &terminal_relation)
}

fn public_image_with_terminal_r2_boundary_from_relation(
    mut public_image: Rv32imIvcPublicImage,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
    terminal_relation: &Rv32imTerminalFPrimeCommittedRelation,
) -> Result<Rv32imIvcPublicImage, SimpleKernelError> {
    let terminal_target = build_rv32im_main_recursion_step_spartan_published_target(backend_relation)
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    public_image.construction2_u_i = terminal_relation.public_boundary().clone();
    public_image.terminal_verified_step_statement_digest = terminal_target.terminal_verified_step_statement_digest;
    Ok(public_image)
}

fn terminal_committed_step_inputs_from_backend(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imTerminalFPrimeCommittedRelation, SimpleKernelError> {
    let relation = Rv32imTerminalFPrimeCommittedRelation::from_backend(spartan_shape, backend_relation)?;
    relation.validate_shape()?;
    relation.require_superneo_assignment_commitment()?;
    Ok(relation)
}

/// Verifies the compressed RV32IM final Construction-2 boundary.
///
/// Coverage:
/// - HyperNova `V` steps 3-4: public-image `x_i`, `pc_i`, and terminal metadata.
/// - HyperNova `V` step 5 / `R1`: final carried CE claims satisfy SuperNeo CE.
/// - HyperNova `V` step 5 / `R2`: terminal `F'` rows, `u_i.C = Commit(Z)`,
///   and the SuperNeo low-norm bound for the same packed `Z`.
fn verify_rv32im_final_construction2_boundary(
    proof: &Rv32imIvcSnarkProof,
    public_image: &Rv32imIvcPublicImage,
    vk: &Rv32imIvcSnarkVerifierKey,
) -> Result<(), SimpleKernelError> {
    public_image.validate_final_construction2_public_boundary()?;
    ensure_final_ce_claims_are_canonical(&proof.final_main_claims)?;
    ensure_final_claims_bind_public_image(public_image, &proof.final_main_claims)?;
    let terminal_public_values = verify_rv32im_terminal_f_prime_committed_step_relation(
        &vk.terminal_f_prime,
        &proof.terminal_f_prime_committed_step_proof,
    )?;
    ensure_terminal_committed_step_public_values_bind_public_image(public_image, &terminal_public_values)?;
    verify_rv32im_final_ce_bundle(&vk.final_ce, &proof.final_main_claims, &proof.final_ce_proof)
}

fn setup_rv32im_ivc_snark_from_recursion_backend_cached(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imIvcSnarkKeyPair, SimpleKernelError> {
    let mut trace = None;
    setup_rv32im_ivc_snark_from_recursion_backend_cached_with_trace(spartan_shape, backend_relation, &mut trace)
}

fn setup_rv32im_ivc_snark_from_recursion_backend_cached_with_trace(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
    trace: &mut Rv32imIvcSnarkTrace<'_>,
) -> Result<Rv32imIvcSnarkKeyPair, SimpleKernelError> {
    trace_emit_owned(
        trace,
        format!(
            "setup.terminal_backend.chunk_count_in={} terminal_step={} fresh_claim_count={} state_in_claim_count={} pc_i={} pc_next={}",
            backend_relation.f_prime_advice.chunk_count_in(),
            backend_relation.f_prime_advice.bridge_handoff_halted_out(),
            backend_relation.payload.step_shape.fresh_claim_count,
            backend_relation.payload.step_shape.state_in_claim_count,
            backend_relation.payload.pc_i(),
            backend_relation.payload.pc_next(),
        ),
    );
    let started = trace_start(trace, "setup.phase=terminal_committed_step_shape_inputs");
    let terminal_setup = Rv32imTerminalFPrimeCommittedStepSetup::from_backend_shape(spartan_shape, backend_relation)?;
    trace_done(trace, "setup.phase=terminal_committed_step_shape_inputs", started);
    setup_rv32im_ivc_snark_from_terminal_setup_cached_with_trace(
        spartan_shape,
        backend_relation,
        &terminal_setup,
        trace,
    )
}

fn setup_rv32im_ivc_snark_from_terminal_relation_cached_with_trace(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
    terminal_relation: &Rv32imTerminalFPrimeCommittedRelation,
    trace: &mut Rv32imIvcSnarkTrace<'_>,
) -> Result<Rv32imIvcSnarkKeyPair, SimpleKernelError> {
    let started = trace_start(trace, "setup.phase=terminal_committed_step_reuse");
    let step_cap = backend_relation
        .f_prime_advice
        .verifier_key_fs()
        .step_cap()?;
    let terminal_setup = terminal_relation.committed_step_setup(step_cap)?;
    trace_done(trace, "setup.phase=terminal_committed_step_reuse", started);
    setup_rv32im_ivc_snark_from_terminal_setup_cached_with_trace(
        spartan_shape,
        backend_relation,
        &terminal_setup,
        trace,
    )
}

fn setup_rv32im_ivc_snark_from_terminal_setup_cached_with_trace(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
    terminal_setup: &Rv32imTerminalFPrimeCommittedStepSetup,
    trace: &mut Rv32imIvcSnarkTrace<'_>,
) -> Result<Rv32imIvcSnarkKeyPair, SimpleKernelError> {
    let step_cap = terminal_setup.step_cap();
    let terminal_r2 = terminal_setup.r1cs_ccs();
    let final_carry = &backend_relation.f_prime_advice.fresh_state_out().carry.main;
    trace_emit_owned(
        trace,
        format!("setup.shape.final_ce_claim_count={}", final_carry.claims.len()),
    );
    trace_emit_owned(
        trace,
        format!("setup.shape.final_ce_witness_count={}", final_carry.witnesses.len()),
    );
    trace_emit_owned(trace, format!("setup.shape.construction2_step_cap={step_cap}"));
    trace_emit_owned(
        trace,
        format!("setup.shape.terminal_r2_source_ccs_rows={}", terminal_r2.structure().n),
    );
    trace_emit_owned(
        trace,
        format!("setup.shape.terminal_r2_source_ccs_cols={}", terminal_r2.structure().m),
    );
    trace_emit_owned(
        trace,
        format!(
            "setup.shape.terminal_r2_source_ccs_nnz={}",
            ccs_structure_nnz(terminal_r2.structure())
        ),
    );
    trace_emit_owned(
        trace,
        format!("setup.shape.terminal_r2_public_inputs={}", terminal_r2.num_public()),
    );
    trace_emit_owned(
        trace,
        format!(
            "setup.shape.terminal_r2_witness_inputs={}",
            terminal_setup.r2_witness_len()
        ),
    );
    let (private_bit_inputs, private_u32_inputs, private_u64_inputs) =
        terminal_setup.terminal_r2_private_encoding_counts();
    let private_padding_inputs = terminal_setup.terminal_r2_private_padding_inputs();
    let private_low_norm_bit_inputs = private_bit_inputs + (private_u32_inputs * 32) + (private_u64_inputs * 64);
    trace_emit_owned(
        trace,
        format!("setup.shape.terminal_r2_private_padding_inputs={private_padding_inputs}"),
    );
    trace_emit_owned(
        trace,
        format!("setup.shape.terminal_r2_private_bit_inputs={private_bit_inputs}"),
    );
    trace_emit_owned(
        trace,
        format!("setup.shape.terminal_r2_private_u32_inputs={private_u32_inputs}"),
    );
    trace_emit_owned(
        trace,
        format!("setup.shape.terminal_r2_private_u64_inputs={private_u64_inputs}"),
    );
    trace_emit_owned(
        trace,
        format!("setup.shape.terminal_r2_private_low_norm_bit_inputs={private_low_norm_bit_inputs}"),
    );
    trace_emit_owned(
        trace,
        format!(
            "setup.shape.terminal_r2_committed_low_norm_width={}",
            terminal_setup.terminal_r2_committed_low_norm_width()?
        ),
    );
    trace_emit(trace, "setup.shape.terminal_r2_superneo_packable=true");
    trace_emit_owned(
        trace,
        format!(
            "setup.shape.terminal_r2_superneo_packed_cols={}",
            terminal_setup.terminal_r2_superneo_packed_cols()
        ),
    );
    trace_emit_owned(
        trace,
        format!(
            "setup.shape.terminal_committed_step_commitment_words={}",
            terminal_setup.terminal_r2_commitment_words()
        ),
    );
    let started = trace_start(trace, "setup.phase=terminal_f_prime_shape");
    trace_terminal_f_prime_committed_step_shape(trace, &terminal_setup)?;
    trace_done(trace, "setup.phase=terminal_f_prime_shape", started);

    let started = trace_start(trace, "setup.phase=cache_key");
    let cache_key = rv32im_ivc_snark_setup_cache_key(spartan_shape, &final_carry.claims, terminal_setup)?;
    trace_done(trace, "setup.phase=cache_key", started);

    let cache = RV32IM_IVC_SNARK_SETUP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(keys) = cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV32IM IVC recursion SNARK setup cache poisoned".into()))?
        .get(&cache_key)
        .cloned()
    {
        trace_emit(trace, "setup.cache=hit");
        return Ok(keys);
    }

    trace_emit(trace, "setup.cache=miss");
    let started = trace_start(trace, "setup.phase=final_ce_setup");
    let (final_ce_pk, final_ce_vk) = setup_rv32im_final_ce_keys(&final_carry.claims, &final_carry.witnesses)?;
    trace_done(trace, "setup.phase=final_ce_setup", started);

    let started = trace_start(trace, "setup.phase=terminal_committed_step_setup");
    let (terminal_pk, terminal_vk) = setup_rv32im_terminal_f_prime_committed_step_relation_from_shape(&terminal_setup)?;
    trace_done(trace, "setup.phase=terminal_committed_step_setup", started);

    let keys = Arc::new((
        Rv32imIvcSnarkProverKey {
            terminal_f_prime: terminal_pk,
            final_ce: final_ce_pk,
        },
        Rv32imIvcSnarkVerifierKey {
            terminal_f_prime: terminal_vk,
            final_ce: final_ce_vk,
        },
    ));
    cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV32IM IVC recursion SNARK setup cache poisoned".into()))?
        .insert(cache_key, keys.clone());
    Ok(keys)
}

fn prove_rv32im_ivc_snark_on_recursion_backend(
    pk: &Rv32imIvcSnarkProverKey,
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imIvcSnarkProof, SimpleKernelError> {
    let mut trace = None;
    prove_rv32im_ivc_snark_on_recursion_backend_with_trace(pk, spartan_shape, backend_relation, &mut trace)
}

fn prove_rv32im_ivc_snark_on_recursion_backend_with_trace(
    pk: &Rv32imIvcSnarkProverKey,
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
    trace: &mut Rv32imIvcSnarkTrace<'_>,
) -> Result<Rv32imIvcSnarkProof, SimpleKernelError> {
    let terminal_relation = terminal_committed_step_inputs_from_backend(spartan_shape, backend_relation)?;
    prove_rv32im_ivc_snark_on_recursion_backend_with_terminal_relation(pk, &terminal_relation, backend_relation, trace)
}

fn prove_rv32im_ivc_snark_on_recursion_backend_with_terminal_relation(
    pk: &Rv32imIvcSnarkProverKey,
    terminal_relation: &Rv32imTerminalFPrimeCommittedRelation,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
    trace: &mut Rv32imIvcSnarkTrace<'_>,
) -> Result<Rv32imIvcSnarkProof, SimpleKernelError> {
    let started = trace_start(trace, "compress.phase=terminal_committed_step_prove");
    let terminal_f_prime_committed_step_proof =
        prove_rv32im_terminal_f_prime_committed_step_relation_from_relation(&pk.terminal_f_prime, terminal_relation)?;
    trace_done(trace, "compress.phase=terminal_committed_step_prove", started);

    let final_carry = &backend_relation.f_prime_advice.fresh_state_out().carry.main;
    let final_main_claims = canonical_final_ce_claims(&final_carry.claims);
    let final_ce_witnesses = final_carry
        .witnesses
        .iter()
        .map(ccs_witness_from_packed_z)
        .collect::<Result<Vec<_>, _>>()?;
    let (params, structure) = final_ce_context(final_main_claims.len())?;
    let started = trace_start(trace, "compress.phase=final_ce_prove");
    let final_ce_proof = prove_rv32im_ce_bundle_relation(
        &pk.final_ce,
        &params,
        structure,
        &final_main_claims,
        &final_ce_witnesses,
        F::from_u64(7),
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM IVC final CE bundle prove failed: {err}")))?;
    trace_done(trace, "compress.phase=final_ce_prove", started);

    Ok(Rv32imIvcSnarkProof {
        terminal_f_prime_committed_step_proof,
        final_main_claims,
        final_ce_proof,
    })
}

impl Rv32imIvcSnark {
    pub(crate) fn from_parts(proof: Rv32imIvcSnarkProof, public_image: Rv32imIvcPublicImage) -> Self {
        Self { proof, public_image }
    }

    pub fn proof(&self) -> &Rv32imIvcSnarkProof {
        &self.proof
    }

    pub fn proof_mut(&mut self) -> &mut Rv32imIvcSnarkProof {
        &mut self.proof
    }

    pub fn public_image(&self) -> &Rv32imIvcPublicImage {
        &self.public_image
    }

    pub fn public_image_mut(&mut self) -> &mut Rv32imIvcPublicImage {
        &mut self.public_image
    }

    pub fn verify(
        &self,
        vk: &Rv32imIvcSnarkVerifierKey,
        expected_public_image: &Rv32imIvcPublicImage,
    ) -> Result<(), SimpleKernelError> {
        if &self.public_image != expected_public_image {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC SNARK public image does not match the caller-supplied public image".into(),
            ));
        }
        verify_rv32im_final_construction2_boundary(&self.proof, &self.public_image, vk)
    }
}

pub(crate) fn build_rv32im_ivc_recursion_snark_setup_shape_from_components(
    statement: &Rv32imFinalStatement,
    proof_digest: [u8; 32],
    kernel_export: &crate::rv32im::kernel::Rv32imKernelExportProof,
    chunk_summaries: &[crate::finalize::FixedShapeChunkSummary],
    steps: &[crate::rv32im::final_relation::Rv32imChunkTransitionWitness],
) -> Result<Rv32imIvcRecursionSnarkSetupShape, SimpleKernelError> {
    let proof = Rv32imFinalBuildProof {
        proof_digest,
        kernel_export: kernel_export.clone(),
        chunk_summaries: chunk_summaries.to_vec(),
        steps: steps.to_vec(),
    };
    let state = build_rv32im_ivc_prover_state_from_final(statement, &proof)?;
    let (main_recursion_step_shape, backend_relation) = build_rv32im_ivc_recursion_backend_from_state(&state)?;
    let _terminal_setup =
        Rv32imTerminalFPrimeCommittedStepSetup::from_backend_shape(&main_recursion_step_shape, &backend_relation)?;
    Ok(Rv32imIvcRecursionSnarkSetupShape {
        main_recursion_step_shape,
    })
}

pub(crate) fn debug_check_rv32im_ivc_recursion_snark_circuit(
    statement: &Rv32imFinalStatement,
    proof: &Rv32imFinalBuildProof,
) -> Result<(), SimpleKernelError> {
    let state = build_rv32im_ivc_prover_state_from_final(statement, proof)?;
    let (spartan_shape, backend_relation) = build_rv32im_ivc_recursion_backend_from_state(&state)?;
    debug_check_rv32im_main_recursion_step_spartan_circuit(&spartan_shape, &backend_relation)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM IVC recursion SNARK circuit failed: {err}")))?;
    debug_check_rv32im_terminal_f_prime_r1cs_ccs_relation(&spartan_shape, &backend_relation)?;
    let keys = setup_rv32im_ivc_snark_from_recursion_backend_cached(&spartan_shape, &backend_relation)?;
    let snark = prove_rv32im_ivc_snark_on_recursion_backend(&keys.as_ref().0, &spartan_shape, &backend_relation)?;
    let public_image = public_image_with_terminal_r2_boundary(state.public_image(), &spartan_shape, &backend_relation)?;
    verify_rv32im_final_construction2_boundary(&snark, &public_image, &keys.as_ref().1)?;
    Ok(())
}

pub fn setup_rv32im_ivc_snark_from_final(
    statement: &Rv32imFinalStatement,
    proof: &Rv32imFinalBuildProof,
) -> Result<(Rv32imIvcSnarkProverKey, Rv32imIvcSnarkVerifierKey), SimpleKernelError> {
    let state = build_rv32im_ivc_prover_state_from_final(statement, proof)?;
    let (spartan_shape, backend_relation) = build_rv32im_ivc_recursion_backend_from_state(&state)?;
    let final_carry = &backend_relation.f_prime_advice.fresh_state_out().carry.main;
    let (final_ce_pk, final_ce_vk) = setup_rv32im_final_ce_keys(&final_carry.claims, &final_carry.witnesses)?;
    let (terminal_pk, terminal_vk) =
        setup_rv32im_terminal_f_prime_committed_step_relation(&spartan_shape, &backend_relation)?;
    Ok((
        Rv32imIvcSnarkProverKey {
            terminal_f_prime: terminal_pk,
            final_ce: final_ce_pk,
        },
        Rv32imIvcSnarkVerifierKey {
            terminal_f_prime: terminal_vk,
            final_ce: final_ce_vk,
        },
    ))
}

pub fn setup_rv32im_ivc_snark_from_final_cached(
    statement: &Rv32imFinalStatement,
    proof: &Rv32imFinalBuildProof,
) -> Result<Rv32imIvcSnarkKeyPair, SimpleKernelError> {
    let state = build_rv32im_ivc_prover_state_from_final(statement, proof)?;
    let (spartan_shape, backend_relation) = build_rv32im_ivc_recursion_backend_from_state(&state)?;
    setup_rv32im_ivc_snark_from_recursion_backend_cached(&spartan_shape, &backend_relation)
}

pub(crate) fn prove_rv32im_ivc_snark_from_final_cached(
    statement: &Rv32imFinalStatement,
    proof: &Rv32imFinalBuildProof,
) -> Result<Rv32imIvcSnark, SimpleKernelError> {
    let ivc_state = build_rv32im_ivc_prover_state_from_final(statement, proof)?;
    let (spartan_shape, backend_relation) = build_rv32im_ivc_recursion_backend_from_state(&ivc_state)?;
    let terminal_relation = terminal_committed_step_inputs_from_backend(&spartan_shape, &backend_relation)?;
    let expected_public_image = public_image_with_terminal_r2_boundary_from_relation(
        ivc_state.public_image(),
        &backend_relation,
        &terminal_relation,
    )?;
    let cache_key = rv32im_ivc_snark_cache_key(statement, proof);
    let cache = RV32IM_IVC_SNARK_PROOF_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(proof) = cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV32IM IVC recursion SNARK proof cache poisoned".into()))?
        .get(&cache_key)
        .cloned()
    {
        return Ok(Rv32imIvcSnark::from_parts((*proof).clone(), expected_public_image));
    }

    ensure_recursion_backend_matches_public_image(&backend_relation, &expected_public_image)?;
    let mut trace = None;
    let keys = setup_rv32im_ivc_snark_from_terminal_relation_cached_with_trace(
        &spartan_shape,
        &backend_relation,
        &terminal_relation,
        &mut trace,
    )?;
    let proof = Arc::new(prove_rv32im_ivc_snark_on_recursion_backend_with_terminal_relation(
        &keys.as_ref().0,
        &terminal_relation,
        &backend_relation,
        &mut trace,
    )?);
    cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV32IM IVC recursion SNARK proof cache poisoned".into()))?
        .insert(cache_key, proof.clone());
    Ok(Rv32imIvcSnark::from_parts((*proof).clone(), expected_public_image))
}

impl Rv32imIvcState {
    pub fn compress(&self) -> Result<Rv32imIvcSnark, SimpleKernelError> {
        let mut trace = None;
        self.compress_with_optional_trace(&mut trace)
    }

    pub fn compress_with_trace(&self, emit: &mut dyn FnMut(&str)) -> Result<Rv32imIvcSnark, SimpleKernelError> {
        let mut trace = Some(emit);
        self.compress_with_optional_trace(&mut trace)
    }

    fn compress_with_optional_trace(
        &self,
        trace: &mut Rv32imIvcSnarkTrace<'_>,
    ) -> Result<Rv32imIvcSnark, SimpleKernelError> {
        let started = trace_start(trace, "compress.phase=public_image");
        let mut public_image = self.public_image();
        trace_done(trace, "compress.phase=public_image", started);

        let (spartan_shape, backend_relation) = build_rv32im_ivc_recursion_backend_from_state_with_trace(self, trace)?;

        let started = trace_start(trace, "compress.phase=terminal_r2_public_boundary");
        let terminal_relation = terminal_committed_step_inputs_from_backend(&spartan_shape, &backend_relation)?;
        public_image =
            public_image_with_terminal_r2_boundary_from_relation(public_image, &backend_relation, &terminal_relation)?;
        trace_done(trace, "compress.phase=terminal_r2_public_boundary", started);

        let started = trace_start(trace, "compress.phase=match_public_image");
        ensure_recursion_backend_matches_public_image(&backend_relation, &public_image)?;
        trace_done(trace, "compress.phase=match_public_image", started);

        let started = trace_start(trace, "compress.phase=setup_cached");
        let keys = setup_rv32im_ivc_snark_from_terminal_relation_cached_with_trace(
            &spartan_shape,
            &backend_relation,
            &terminal_relation,
            trace,
        )?;
        trace_done(trace, "compress.phase=setup_cached", started);

        let started = trace_start(trace, "compress.phase=prove");
        let proof = prove_rv32im_ivc_snark_on_recursion_backend_with_terminal_relation(
            &keys.as_ref().0,
            &terminal_relation,
            &backend_relation,
            trace,
        )?;
        trace_done(trace, "compress.phase=prove", started);

        Ok(Rv32imIvcSnark { proof, public_image })
    }
}

pub fn setup_rv32im_ivc_snark_cached(state: &Rv32imIvcState) -> Result<Rv32imIvcSnarkKeyPair, SimpleKernelError> {
    let mut trace = None;
    setup_rv32im_ivc_snark_cached_with_optional_trace(state, &mut trace)
}

pub fn setup_rv32im_ivc_snark_cached_with_trace(
    state: &Rv32imIvcState,
    emit: &mut dyn FnMut(&str),
) -> Result<Rv32imIvcSnarkKeyPair, SimpleKernelError> {
    let mut trace = Some(emit);
    setup_rv32im_ivc_snark_cached_with_optional_trace(state, &mut trace)
}

fn setup_rv32im_ivc_snark_cached_with_optional_trace(
    state: &Rv32imIvcState,
    trace: &mut Rv32imIvcSnarkTrace<'_>,
) -> Result<Rv32imIvcSnarkKeyPair, SimpleKernelError> {
    let (spartan_shape, backend_relation) = build_rv32im_ivc_recursion_backend_from_state_with_trace(state, trace)?;
    setup_rv32im_ivc_snark_from_recursion_backend_cached_with_trace(&spartan_shape, &backend_relation, trace)
}
