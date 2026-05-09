//! Owns native RV32IM F' semantics and the recursion hash-image boundary.

mod accumulator;
mod side;

use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use neo_ajtai::Commitment;
use neo_ccs::{check_ccs_rowwise_zero, check_ce_consistency, CeClaim, CeWitness, Mat};
use neo_math::{F, K};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

use self::accumulator::Rv32imMainRecursionAccumulatorSurface;
pub use self::accumulator::RV32IM_MAIN_RECURSION_ACCUMULATOR_SLOTS;
pub use self::side::{
    build_rv32im_main_recursion_side_lane_from_side_opening_public, Rv32imMainRecursionPhiSide,
    Rv32imMainRecursionSideClaim, Rv32imMainRecursionSideLaneWitness,
};
use crate::chunk_folding::ChunkReplayWitness;
use crate::construction2::{
    Construction2EncodedPublicInput, CONSTRUCTION2_ENC_INST_BITS, CONSTRUCTION2_ENC_INST_RING_DEGREE,
    CONSTRUCTION2_ENC_INST_RING_SLOTS,
};
use crate::proof::Carry;
use crate::public_proof::rv32im::Rv32imSideOpeningPublic;
use crate::rv32im::chunk::fold::adapt_rv32im_chunk_to_fresh_ccs;
use crate::rv32im::chunk::step_ivc::{rv32im_chunk_step_ivc_initial_state_for_step_cap, Rv32imChunkStepIvcRelation};
use crate::rv32im::construction2::{
    build_rv32im_main_recursion_construction2_default_fresh_instance,
    build_rv32im_main_recursion_construction2_fresh_instance_with_input_and_x_i_with_perf,
    build_rv32im_main_recursion_construction2_nifs_bridge_with_trace,
    build_rv32im_main_recursion_construction2_pi_fold_from_replay_witness,
    build_rv32im_main_recursion_construction2_verified_step_statement_from_summary,
    debug_trace_build_rv32im_main_recursion_construction2_fresh_instance_with_input_and_x_i,
    verify_rv32im_main_recursion_construction2_nifs_step_with_perf_and_trace,
    Rv32imMainRecursionConstruction2FreshInstance, Rv32imMainRecursionConstruction2PiFoldProof,
    Rv32imMainRecursionConstruction2StateImage, Rv32imMainRecursionConstruction2VerifiedStepStatement,
};
use crate::rv32im::final_relation::{rv32im_chunk_fold_carry_recursive_accumulator_digest, Rv32imChunkFoldState};
use crate::rv32im::kernel::Rv32imVerifiedKernelChunkHandoff;
use crate::rv32im::main_relation_trace::{
    build_rv32im_main_circuit_chunk_trace_from_authoritative_parts, Rv32imMainCircuitChunkTrace,
};
use crate::rv32im::SimpleKernelError;

pub type Rv32imEncodedPublicInput = Construction2EncodedPublicInput;

pub const RV32IM_ENC_INST_BITS: usize = CONSTRUCTION2_ENC_INST_BITS;
pub const RV32IM_ENC_INST_RING_DEGREE: usize = CONSTRUCTION2_ENC_INST_RING_DEGREE;
pub const RV32IM_ENC_INST_RING_SLOTS: usize = CONSTRUCTION2_ENC_INST_RING_SLOTS;
pub const RV32IM_MAIN_RECURSION_TRIVIAL_PC: u64 = 1;
pub const RV32IM_MAIN_RECURSION_SIDE_WITNESS_ACTIVE: bool = false;
pub const RV32IM_MAIN_RECURSION_PHI_SIDE_ACTIVE: bool = true;
pub const RV32IM_MAIN_RECURSION_SIDE_LANE_ACTIVE: bool =
    RV32IM_MAIN_RECURSION_SIDE_WITNESS_ACTIVE || RV32IM_MAIN_RECURSION_PHI_SIDE_ACTIVE;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imMainRecursionStepStatement {
    pub x_out: Rv32imEncodedPublicInput,
    pub folded_accumulator_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imMainRecursionBackendStepStatement {
    pub x_out: Rv32imEncodedPublicInput,
    pub folded_accumulator_digest: [u8; 32],
}

impl Rv32imMainRecursionBackendStepStatement {
    pub fn native_statement(&self) -> Rv32imMainRecursionStepStatement {
        Rv32imMainRecursionStepStatement {
            x_out: self.x_out.clone(),
            folded_accumulator_digest: self.folded_accumulator_digest,
        }
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/main_recursion_step_spartan_statement");
        tr.append_message(
            b"neo.fold.next/rv32im/main_recursion_step_spartan_statement/version",
            b"v8",
        );
        tr.append_message(
            b"neo.fold.next/rv32im/main_recursion_step_spartan_statement/x_out",
            &self.x_out.bytes(),
        );
        tr.append_message(
            b"neo.fold.next/rv32im/main_recursion_step_spartan_statement/folded_accumulator_digest",
            &self.folded_accumulator_digest,
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imVerifierKeyFs {
    pub domain_tag_digest: [u8; 32],
    pub main_lane_shape_digest: [u8; 32],
    pub step_cap: u64,
}

impl Rv32imVerifierKeyFs {
    pub fn step_cap(&self) -> Result<usize, SimpleKernelError> {
        usize::try_from(self.step_cap).map_err(|_| {
            SimpleKernelError::Bridge(
                "RV32IM recursion verifier-key step_cap does not fit into the local native-width model".into(),
            )
        })
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/main_recursion_verifier_key_fs");
        tr.append_message(b"neo.fold.next/rv32im/main_recursion_verifier_key_fs/version", b"v1");
        tr.append_message(
            b"neo.fold.next/rv32im/main_recursion_verifier_key_fs/domain_tag_digest",
            &self.domain_tag_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv32im/main_recursion_verifier_key_fs/main_lane_shape_digest",
            &self.main_lane_shape_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/rv32im/main_recursion_verifier_key_fs/step_cap",
            &[self.step_cap],
        );
        tr.digest32()
    }
}

pub fn build_rv32im_main_recursion_verifier_key_fs() -> Result<Rv32imVerifierKeyFs, SimpleKernelError> {
    build_rv32im_main_recursion_verifier_key_fs_for_step_cap(
        crate::rv32im::recursion_shape::RV32IM_RECURSION_DEFAULT_STEP_CAP as usize,
    )
}

pub fn build_rv32im_main_recursion_verifier_key_fs_for_step_cap(
    step_cap: usize,
) -> Result<Rv32imVerifierKeyFs, SimpleKernelError> {
    static VK_FS_CACHE: OnceLock<Mutex<HashMap<usize, Rv32imVerifierKeyFs>>> = OnceLock::new();
    if let Some(cached) = VK_FS_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("rv32im main recursion vk_fs cache mutex poisoned")
        .get(&step_cap)
        .cloned()
    {
        return Ok(cached);
    }

    let mut domain_tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/main_recursion_verifier_key_fs/domain");
    domain_tr.append_message(
        b"neo.fold.next/rv32im/main_recursion_verifier_key_fs/domain/version",
        b"v1",
    );
    domain_tr.append_message(
        b"neo.fold.next/rv32im/main_recursion_verifier_key_fs/domain/tag",
        b"neo.fold.next/rv32im/main_recursion_f_prime_x_out",
    );

    let vk_fs = Rv32imVerifierKeyFs {
        domain_tag_digest: domain_tr.digest32(),
        main_lane_shape_digest: crate::rv32im::recursion_shape::build_rv32im_recursion_shape_for_step_cap(step_cap)?
            .canonical_digest(),
        step_cap: step_cap as u64,
    };
    VK_FS_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("rv32im main recursion vk_fs cache mutex poisoned")
        .insert(step_cap, vk_fs.clone());
    Ok(vk_fs)
}

fn rv32im_main_recursion_initial_z_for_step_cap(step_cap: usize) -> [u8; 32] {
    crate::rv32im::chunk::step_ivc::rv32im_chunk_step_ivc_initial_state_for_step_cap(step_cap)
        .carry
        .terminal_handle
        .0
}

pub(crate) fn build_rv32im_main_recursion_backend_statement_from_parts_with_vk_fs(
    vk_fs: &Rv32imVerifierKeyFs,
    chunk_count: u64,
    folded_accumulator_digest: [u8; 32],
    terminal_handle_digest: [u8; 32],
) -> Rv32imMainRecursionBackendStepStatement {
    Rv32imMainRecursionBackendStepStatement {
        x_out: rv32im_main_recursion_x_out(
            vk_fs,
            chunk_count,
            rv32im_main_recursion_initial_z_for_step_cap(vk_fs.step_cap().expect("canonical vk_fs step_cap")),
            terminal_handle_digest,
            RV32IM_MAIN_RECURSION_TRIVIAL_PC,
            folded_accumulator_digest,
        ),
        folded_accumulator_digest,
    }
}

pub(crate) fn build_rv32im_main_recursion_backend_statement_from_advice(
    advice: &Rv32imMainRecursionFPrimeAdvice,
) -> Result<Rv32imMainRecursionBackendStepStatement, SimpleKernelError> {
    let step_image = evaluate_rv32im_main_recursion_f_prime_advice(advice)?;
    Ok(build_rv32im_main_recursion_backend_statement_from_parts_with_vk_fs(
        advice.verifier_key_fs(),
        step_image.chunk_count(),
        step_image.folded_accumulator_digest(),
        step_image.next_state.carry.terminal_handle.0,
    ))
}

#[derive(Clone, Debug)]
pub(crate) struct Rv32imMainRecursionAccumulator {
    chunk_count: u64,
    state: Rv32imChunkFoldState,
    folded_accumulator_digest: [u8; 32],
}

impl Rv32imMainRecursionAccumulator {
    fn seed(step_cap: usize) -> Self {
        let state = rv32im_chunk_step_ivc_initial_state_for_step_cap(step_cap);
        Self {
            chunk_count: 0,
            folded_accumulator_digest: rv32im_chunk_fold_carry_recursive_accumulator_digest(&state.carry),
            state,
        }
    }

    fn apply_verified_step_image(
        self,
        output: Rv32imMainRecursionFPrimeStepImage,
    ) -> Result<(Self, Rv32imMainRecursionConstruction2FreshInstance), SimpleKernelError> {
        let output = output.into_parts();
        if output.chunk_count != self.chunk_count + 1 {
            return Err(SimpleKernelError::Bridge(
                "RV32IM main recursion step image chunk_count does not advance the carried recursive position".into(),
            ));
        }
        if output.folded_accumulator_digest
            != rv32im_chunk_fold_carry_recursive_accumulator_digest(&output.next_state.carry)
        {
            return Err(SimpleKernelError::Bridge(
                "RV32IM main recursion step image folded accumulator digest does not match next_state".into(),
            ));
        }
        if output.z_next != output.next_state.carry.terminal_handle.0 {
            return Err(SimpleKernelError::Bridge(
                "RV32IM main recursion step image z_next does not match next_state".into(),
            ));
        }
        if output.pc_next != RV32IM_MAIN_RECURSION_TRIVIAL_PC {
            return Err(SimpleKernelError::Bridge(
                "RV32IM main recursion step image pc_next does not match the trivial RV32IM recursion control lane"
                    .into(),
            ));
        }
        Ok((
            Self {
                chunk_count: output.chunk_count,
                state: output.next_state,
                folded_accumulator_digest: output.folded_accumulator_digest,
            },
            output.construction2_u_next,
        ))
    }

    fn x_i(&self, vk_fs: &Rv32imVerifierKeyFs) -> Rv32imEncodedPublicInput {
        rv32im_main_recursion_x_out(
            vk_fs,
            self.chunk_count,
            rv32im_main_recursion_initial_z_for_step_cap(vk_fs.step_cap().expect("canonical vk_fs step_cap")),
            self.state.carry.terminal_handle.0,
            RV32IM_MAIN_RECURSION_TRIVIAL_PC,
            self.folded_accumulator_digest,
        )
    }
}

fn validate_rv32im_main_recursion_base_case_accumulator(
    accumulator: &Rv32imMainRecursionAccumulator,
    advice: &Rv32imMainRecursionFPrimeAdvice,
    construction2_u_i: &Rv32imMainRecursionConstruction2FreshInstance,
) -> Result<(), SimpleKernelError> {
    if accumulator.chunk_count != 0 {
        return Err(SimpleKernelError::Bridge(
            "RV32IM main recursion F' base case must begin at recursive position i = 0".into(),
        ));
    }
    if advice.z_0() != advice.z_i() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM main recursion F' base case does not satisfy z_0 == z_i".into(),
        ));
    }
    let canonical_full_width =
        crate::rv32im::construction2::default::build_rv32im_main_recursion_construction2_canonical_full_width(
            advice.verifier_key_fs(),
            advice.phi_side(),
        )?;
    let expected_default = build_rv32im_main_recursion_construction2_default_fresh_instance(
        advice.verifier_key_fs(),
        canonical_full_width,
    )?;
    if construction2_u_i != &expected_default {
        return Err(SimpleKernelError::Bridge(
            "RV32IM main recursion F' base case does not carry the canonical default fresh instance u_perp".into(),
        ));
    }
    Ok(())
}

fn zero_ce_claim_like(claim: &CeClaim<Commitment, F, K>) -> CeClaim<Commitment, F, K> {
    CeClaim {
        c: Commitment::zeros(claim.c.d, claim.c.kappa),
        X: Mat::zero(claim.X.rows(), claim.X.cols(), F::ZERO),
        r: vec![K::ZERO; claim.r.len()],
        s_col: vec![K::ZERO; claim.s_col.len()],
        y_ring: claim
            .y_ring
            .iter()
            .map(|row| vec![K::ZERO; row.len()])
            .collect(),
        ct: vec![K::ZERO; claim.ct.len()],
        aux_openings: vec![K::ZERO; claim.aux_openings.len()],
        y_zcol: vec![K::ZERO; claim.y_zcol.len()],
        m_in: claim.m_in,
        fold_digest: [0; 32],
        c_step_coords: vec![F::ZERO; claim.c_step_coords.len()],
        u_offset: 0,
        u_len: 0,
    }
}

fn zero_ce_witness_like(witness: &Mat<F>) -> Mat<F> {
    Mat::zero(witness.rows(), witness.cols(), F::ZERO)
}

pub(crate) fn build_rv32im_main_recursion_base_case_default_carry(
    state_like: &Rv32imChunkFoldState,
) -> Result<Carry, SimpleKernelError> {
    let carried_bundle =
        Rv32imMainRecursionAccumulatorSurface::try_from_carry(&state_like.carry.main, "F' base-case output")?;
    let (params, log, structure) =
        crate::rv32im::kernel::rv32im_root_main_lane_context_for_claim_count(state_like.carry.main.claims.len())?;
    let carried_main = carried_bundle.slot(0)?.carry();
    if carried_main.claims.is_empty() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM main recursion F' base-case default carry requires a non-empty carried CE bundle template".into(),
        ));
    }

    let mut default_claims = Vec::with_capacity(carried_main.claims.len());
    let mut default_witnesses = Vec::with_capacity(carried_main.witnesses.len());
    for (claim_index, (claim, witness)) in carried_main
        .claims
        .iter()
        .zip(carried_main.witnesses.iter())
        .enumerate()
    {
        let zero_claim = zero_ce_claim_like(claim);
        let zero_witness = zero_ce_witness_like(witness);
        let zero_x = vec![F::ZERO; zero_claim.m_in];
        let zero_w = vec![F::ZERO; structure.m.saturating_sub(zero_claim.m_in)];
        check_ccs_rowwise_zero(structure, &zero_x, &zero_w).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV32IM main recursion F' base-case zero witness failed CCS row-wise zero for carried CE claim {claim_index}: {err}"
            ))
        })?;
        check_ce_consistency(
            &params,
            structure,
            log,
            &zero_claim,
            &CeWitness {
                Z: zero_witness.clone(),
            },
        )
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV32IM main recursion F' base-case default CE claim failed consistency for carried CE claim {claim_index}: {err}"
            ))
        })?;
        default_claims.push(zero_claim);
        default_witnesses.push(zero_witness);
    }

    Ok(Carry {
        claims: default_claims,
        witnesses: default_witnesses,
    })
}

fn rv32im_main_recursion_fresh_instance_digest(
    x_i: &Rv32imEncodedPublicInput,
    construction2_u_i: Option<&Rv32imMainRecursionConstruction2FreshInstance>,
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/main_recursion_fresh_instance");
    tr.append_message(b"neo.fold.next/rv32im/main_recursion_fresh_instance/version", b"v4");
    tr.append_message(b"neo.fold.next/rv32im/main_recursion_fresh_instance/x_i", &x_i.bytes());
    if let Some(construction2_u_i) = construction2_u_i {
        tr.append_message(
            b"neo.fold.next/rv32im/main_recursion_fresh_instance/construction2_u_i",
            &construction2_u_i.expected_digest(),
        );
    }
    tr.digest32()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imMainRecursionFPrimeInput;

pub trait Rv32imMainRecursionFPrimeBody {
    fn step(
        &self,
        input: &Rv32imMainRecursionFPrimeInput,
        advice: &Rv32imMainRecursionFPrimeAdvice,
    ) -> Result<Rv32imMainRecursionFPrimeStepImage, SimpleKernelError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CanonicalRv32imMainRecursionFPrimeBody;

impl Rv32imMainRecursionFPrimeBody for CanonicalRv32imMainRecursionFPrimeBody {
    fn step(
        &self,
        input: &Rv32imMainRecursionFPrimeInput,
        advice: &Rv32imMainRecursionFPrimeAdvice,
    ) -> Result<Rv32imMainRecursionFPrimeStepImage, SimpleKernelError> {
        evaluate_rv32im_main_recursion_f_prime_step(input, advice)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imMainRecursionFPrimePublicOutput {
    x_out: Rv32imEncodedPublicInput,
}

impl Rv32imMainRecursionFPrimePublicOutput {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/main_recursion_f_prime_public_output");
        tr.append_message(
            b"neo.fold.next/rv32im/main_recursion_f_prime_public_output/version",
            b"v1",
        );
        tr.append_message(
            b"neo.fold.next/rv32im/main_recursion_f_prime_public_output/x_out",
            &self.x_out.bytes(),
        );
        tr.digest32()
    }

    pub fn x_out(&self) -> &Rv32imEncodedPublicInput {
        &self.x_out
    }

    pub fn x_out_mut(&mut self) -> &mut Rv32imEncodedPublicInput {
        &mut self.x_out
    }
}

#[derive(Clone, Debug)]
pub struct Rv32imMainRecursionFPrimeAdvice {
    vk_fs: Rv32imVerifierKeyFs,
    chunk_count_in: u64,
    z_0: [u8; 32],
    z_i: [u8; 32],
    pc_i: u64,
    side_witness: Rv32imMainRecursionSideLaneWitness,
    phi_side: Rv32imMainRecursionPhiSide,
    pub(crate) state_in: Rv32imChunkFoldState,
    folded_accumulator_in_digest: [u8; 32],
    x_i: Rv32imEncodedPublicInput,
    construction2_input_u_i: Option<Rv32imMainRecursionConstruction2FreshInstance>,
    native_verified_step_statement: Rv32imMainRecursionConstruction2VerifiedStepStatement,
    terminal_step: bool,
    verified_kernel_handoff: Rv32imVerifiedKernelChunkHandoff,
    state_out: Rv32imChunkFoldState,
    main_circuit_replay_witness: ChunkReplayWitness,
    construction2_pi_fold: Rv32imMainRecursionConstruction2PiFoldProof,
}

impl Rv32imMainRecursionFPrimeAdvice {
    pub(crate) fn from_parts(
        vk_fs: Rv32imVerifierKeyFs,
        chunk_count_in: u64,
        z_0: [u8; 32],
        z_i: [u8; 32],
        pc_i: u64,
        side_witness: Rv32imMainRecursionSideLaneWitness,
        phi_side: Rv32imMainRecursionPhiSide,
        state_in: Rv32imChunkFoldState,
        x_i: Rv32imEncodedPublicInput,
        construction2_input_u_i: Option<Rv32imMainRecursionConstruction2FreshInstance>,
        native_verified_step_statement: Rv32imMainRecursionConstruction2VerifiedStepStatement,
        terminal_step: bool,
        verified_kernel_handoff: Rv32imVerifiedKernelChunkHandoff,
        state_out: Rv32imChunkFoldState,
        main_circuit_replay_witness: ChunkReplayWitness,
        construction2_pi_fold: Rv32imMainRecursionConstruction2PiFoldProof,
    ) -> Result<Self, SimpleKernelError> {
        let folded_accumulator_in_digest = rv32im_chunk_fold_carry_recursive_accumulator_digest(&state_in.carry);
        Self::from_parts_with_folded_accumulator_in_digest(
            vk_fs,
            chunk_count_in,
            z_0,
            z_i,
            pc_i,
            side_witness,
            phi_side,
            state_in,
            folded_accumulator_in_digest,
            x_i,
            construction2_input_u_i,
            native_verified_step_statement,
            terminal_step,
            verified_kernel_handoff,
            state_out,
            main_circuit_replay_witness,
            construction2_pi_fold,
        )
    }

    pub(crate) fn from_parts_with_folded_accumulator_in_digest(
        vk_fs: Rv32imVerifierKeyFs,
        chunk_count_in: u64,
        z_0: [u8; 32],
        z_i: [u8; 32],
        pc_i: u64,
        side_witness: Rv32imMainRecursionSideLaneWitness,
        phi_side: Rv32imMainRecursionPhiSide,
        state_in: Rv32imChunkFoldState,
        folded_accumulator_in_digest: [u8; 32],
        x_i: Rv32imEncodedPublicInput,
        construction2_input_u_i: Option<Rv32imMainRecursionConstruction2FreshInstance>,
        native_verified_step_statement: Rv32imMainRecursionConstruction2VerifiedStepStatement,
        terminal_step: bool,
        verified_kernel_handoff: Rv32imVerifiedKernelChunkHandoff,
        state_out: Rv32imChunkFoldState,
        main_circuit_replay_witness: ChunkReplayWitness,
        construction2_pi_fold: Rv32imMainRecursionConstruction2PiFoldProof,
    ) -> Result<Self, SimpleKernelError> {
        state_in.carry.validate_projection_digests("state_in")?;
        state_out.carry.validate_projection_digests("state_out")?;
        let expected_folded_accumulator_in_digest =
            rv32im_chunk_fold_carry_recursive_accumulator_digest(&state_in.carry);
        if folded_accumulator_in_digest != expected_folded_accumulator_in_digest {
            return Err(SimpleKernelError::Bridge(
                "RV32IM main recursion advice folded accumulator input digest does not match state_in".into(),
            ));
        }
        if let Some(construction2_input_u_i) = construction2_input_u_i.as_ref() {
            if chunk_count_in != 0 && construction2_input_u_i.x_i() != &x_i {
                return Err(SimpleKernelError::Bridge(
                    "RV32IM main recursion advice cannot bind a Construction-2 u_i whose x_i disagrees with the carried native image"
                        .into(),
                ));
            }
        }
        if native_verified_step_statement.chunk_index != verified_kernel_handoff.bridge_handoff.chunk_index {
            return Err(SimpleKernelError::Bridge(
                "RV32IM main recursion advice carries a native verified-step chunk_index that disagrees with the verified bridge handoff"
                    .into(),
            ));
        }
        if native_verified_step_statement.state_in != state_in.carry.terminal_handle.0 {
            return Err(SimpleKernelError::Bridge(
                "RV32IM main recursion advice carries a native verified-step state_in that disagrees with the running recursive state"
                    .into(),
            ));
        }
        if native_verified_step_statement.state_out != state_out.carry.terminal_handle.0 {
            return Err(SimpleKernelError::Bridge(
                "RV32IM main recursion advice carries a native verified-step state_out that disagrees with the next recursive state"
                    .into(),
            ));
        }
        if native_verified_step_statement.public_chunk_digest != verified_kernel_handoff.public_chunk_digest {
            return Err(SimpleKernelError::Bridge(
                "RV32IM main recursion advice carries a native verified-step public chunk digest that disagrees with the verified bridge handoff"
                    .into(),
            ));
        }
        Ok(Self {
            vk_fs,
            chunk_count_in,
            z_0,
            z_i,
            pc_i,
            side_witness,
            phi_side,
            folded_accumulator_in_digest,
            state_in,
            x_i,
            construction2_input_u_i,
            native_verified_step_statement,
            terminal_step,
            verified_kernel_handoff,
            state_out,
            main_circuit_replay_witness,
            construction2_pi_fold,
        })
    }

    pub fn chunk_index(&self) -> u64 {
        self.native_verified_step_statement.chunk_index
    }

    pub fn chunk_count_in(&self) -> u64 {
        self.chunk_count_in
    }

    pub fn z_0(&self) -> &[u8; 32] {
        &self.z_0
    }

    pub fn z_i(&self) -> &[u8; 32] {
        &self.z_i
    }

    pub fn pc_i(&self) -> u64 {
        self.pc_i
    }

    pub fn x_i(&self) -> &Rv32imEncodedPublicInput {
        &self.x_i
    }

    pub fn x_hash(&self) -> &Rv32imEncodedPublicInput {
        self.x_i()
    }

    pub fn construction2_input_fresh_instance(&self) -> Option<&Rv32imMainRecursionConstruction2FreshInstance> {
        self.construction2_input_u_i.as_ref()
    }

    pub fn side_witness(&self) -> &Rv32imMainRecursionSideLaneWitness {
        &self.side_witness
    }

    pub fn phi_side(&self) -> &Rv32imMainRecursionPhiSide {
        &self.phi_side
    }

    pub fn folded_accumulator_in_digest(&self) -> [u8; 32] {
        self.folded_accumulator_in_digest
    }

    pub(crate) fn folded_accumulator_in_digest_mut(&mut self) -> &mut [u8; 32] {
        &mut self.folded_accumulator_in_digest
    }

    pub fn verifier_key_fs(&self) -> &Rv32imVerifierKeyFs {
        &self.vk_fs
    }

    pub fn running_state(&self) -> &Rv32imChunkFoldState {
        &self.state_in
    }

    pub(crate) fn running_state_mut(&mut self) -> &mut Rv32imChunkFoldState {
        &mut self.state_in
    }

    pub fn step_statement_digest(&self) -> [u8; 32] {
        self.native_verified_step_statement.expected_digest()
    }

    pub fn bridge_handoff_digest(&self) -> [u8; 32] {
        self.verified_kernel_handoff
            .bridge_handoff
            .expected_digest()
    }

    pub(crate) fn bridge_handoff_halted_out(&self) -> bool {
        self.terminal_step
    }

    pub fn fresh_instance_digest(&self) -> [u8; 32] {
        rv32im_main_recursion_fresh_instance_digest(&self.x_i, self.construction2_input_u_i.as_ref())
    }

    pub fn fresh_state_out(&self) -> &Rv32imChunkFoldState {
        &self.state_out
    }

    pub(crate) fn main_circuit_chunk_trace(&self) -> Result<Rv32imMainCircuitChunkTrace, SimpleKernelError> {
        let main_circuit_chunk_summary = self
            .native_verified_step_statement
            .fixed_shape_chunk_summary()?;
        build_rv32im_main_circuit_chunk_trace_from_authoritative_parts(
            self.verified_kernel_handoff.bridge_handoff.chunk_index as usize,
            &self.verified_kernel_handoff,
            &main_circuit_chunk_summary,
            &self.state_in.carry,
            &self.state_out.carry,
            &self.state_in.transcript,
            &self.state_out.transcript,
            &self.main_circuit_replay_witness,
        )
    }

    pub(crate) fn fresh_state_out_mut(&mut self) -> &mut Rv32imChunkFoldState {
        &mut self.state_out
    }

    pub(crate) fn verified_kernel_handoff(&self) -> &Rv32imVerifiedKernelChunkHandoff {
        &self.verified_kernel_handoff
    }

    pub(crate) fn main_circuit_replay_witness(&self) -> &ChunkReplayWitness {
        &self.main_circuit_replay_witness
    }

    pub(crate) fn main_circuit_replay_witness_mut(&mut self) -> &mut ChunkReplayWitness {
        &mut self.main_circuit_replay_witness
    }

    pub(crate) fn verified_kernel_handoff_mut(&mut self) -> &mut Rv32imVerifiedKernelChunkHandoff {
        &mut self.verified_kernel_handoff
    }

    pub(crate) fn verifier_key_fs_mut(&mut self) -> &mut Rv32imVerifierKeyFs {
        &mut self.vk_fs
    }

    pub(crate) fn chunk_count_in_mut(&mut self) -> &mut u64 {
        &mut self.chunk_count_in
    }

    pub(crate) fn z_i_mut(&mut self) -> &mut [u8; 32] {
        &mut self.z_i
    }

    pub(crate) fn pc_i_mut(&mut self) -> &mut u64 {
        &mut self.pc_i
    }

    pub(crate) fn side_witness_mut(&mut self) -> &mut Rv32imMainRecursionSideLaneWitness {
        &mut self.side_witness
    }

    pub(crate) fn x_i_mut(&mut self) -> &mut Rv32imEncodedPublicInput {
        &mut self.x_i
    }

    pub(crate) fn construction2_input_fresh_instance_mut(
        &mut self,
    ) -> Option<&mut Rv32imMainRecursionConstruction2FreshInstance> {
        self.construction2_input_u_i.as_mut()
    }

    pub(crate) fn chunk_index_mut(&mut self) -> &mut u64 {
        &mut self.native_verified_step_statement.chunk_index
    }

    pub(crate) fn construction2_pi_fold(&self) -> &Rv32imMainRecursionConstruction2PiFoldProof {
        &self.construction2_pi_fold
    }

    pub(crate) fn construction2_pi_fold_mut(&mut self) -> &mut Rv32imMainRecursionConstruction2PiFoldProof {
        &mut self.construction2_pi_fold
    }

    pub(crate) fn terminal_step_mut(&mut self) -> &mut bool {
        &mut self.terminal_step
    }
}

#[derive(Clone, Debug)]
pub struct Rv32imMainRecursionFPrimeStepImage {
    chunk_count: u64,
    z_next: [u8; 32],
    pc_next: u64,
    phi_side: Rv32imMainRecursionPhiSide,
    construction2_u_next: Rv32imMainRecursionConstruction2FreshInstance,
    pub(crate) next_state: Rv32imChunkFoldState,
    folded_accumulator_digest: [u8; 32],
    x_out: Rv32imEncodedPublicInput,
}

pub(crate) struct Rv32imMainRecursionFPrimeStepImageParts {
    pub(crate) chunk_count: u64,
    pub(crate) z_next: [u8; 32],
    pub(crate) pc_next: u64,
    pub(crate) phi_side: Rv32imMainRecursionPhiSide,
    pub(crate) construction2_u_next: Rv32imMainRecursionConstruction2FreshInstance,
    pub(crate) next_state: Rv32imChunkFoldState,
    pub(crate) folded_accumulator_digest: [u8; 32],
    pub(crate) x_out: Rv32imEncodedPublicInput,
}

impl Rv32imMainRecursionFPrimeStepImage {
    pub fn chunk_count(&self) -> u64 {
        self.chunk_count
    }

    pub fn z_next(&self) -> &[u8; 32] {
        &self.z_next
    }

    pub fn pc_next(&self) -> u64 {
        self.pc_next
    }

    pub fn phi_side(&self) -> &Rv32imMainRecursionPhiSide {
        &self.phi_side
    }

    pub fn construction2_u_next(&self) -> &Rv32imMainRecursionConstruction2FreshInstance {
        &self.construction2_u_next
    }

    pub fn folded_accumulator_digest(&self) -> [u8; 32] {
        self.folded_accumulator_digest
    }

    pub fn x_out(&self) -> &Rv32imEncodedPublicInput {
        &self.x_out
    }

    pub fn running_out_state(&self) -> &Rv32imChunkFoldState {
        &self.next_state
    }

    pub(crate) fn into_parts(self) -> Rv32imMainRecursionFPrimeStepImageParts {
        Rv32imMainRecursionFPrimeStepImageParts {
            chunk_count: self.chunk_count,
            z_next: self.z_next,
            pc_next: self.pc_next,
            phi_side: self.phi_side,
            construction2_u_next: self.construction2_u_next,
            next_state: self.next_state,
            folded_accumulator_digest: self.folded_accumulator_digest,
            x_out: self.x_out,
        }
    }
}

pub(crate) fn rv32im_main_recursion_x_out(
    vk_fs: &Rv32imVerifierKeyFs,
    chunk_count: u64,
    z_0: [u8; 32],
    z_i: [u8; 32],
    pc_i: u64,
    folded_accumulator_digest: [u8; 32],
) -> Rv32imEncodedPublicInput {
    Rv32imMainRecursionConstruction2StateImage::from_parts(
        vk_fs.clone(),
        chunk_count,
        z_0,
        z_i,
        pc_i,
        folded_accumulator_digest,
    )
    .encoded_public_input()
}

pub(crate) fn build_rv32im_main_recursion_x_i_from_advice(
    advice: &Rv32imMainRecursionFPrimeAdvice,
) -> Rv32imEncodedPublicInput {
    rv32im_main_recursion_accumulator_from_f_prime_advice(advice).x_i(advice.verifier_key_fs())
}

pub(crate) fn build_rv32im_main_recursion_x_hash_from_advice(
    advice: &Rv32imMainRecursionFPrimeAdvice,
) -> Rv32imEncodedPublicInput {
    build_rv32im_main_recursion_x_i_from_advice(advice)
}

#[derive(Clone, Debug, Default)]
pub struct Rv32imMainRecursionFPrimeAdviceStepBuildPerf {
    pub build_advice_ms: f64,
    pub evaluate_step_ms: f64,
    pub apply_step_image_ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Rv32imMainRecursionFPrimeAdviceBuildPerf {
    pub verifier_key_ms: f64,
    pub relation_validation_ms: f64,
    pub canonical_full_width_ms: f64,
    pub canonical_u_perp_ms: f64,
    pub total_ms: f64,
    pub step_count: usize,
    pub per_step: Vec<Rv32imMainRecursionFPrimeAdviceStepBuildPerf>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv32imMainRecursionConstruction2NifsVerifyPerf {
    pub chunk_relation_verify_ms: f64,
    pub derive_next_state_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv32imMainRecursionFPrimeEvalPerf {
    pub prechecks_ms: f64,
    pub base_case_validation_ms: f64,
    pub x_i_check_ms: f64,
    pub build_nifs_bridge_ms: f64,
    pub verify_nifs_step_ms: f64,
    pub verify_nifs_chunk_relation_ms: f64,
    pub verify_nifs_derive_next_state_ms: f64,
    pub next_state_surface_check_ms: f64,
    pub derive_public_outputs_ms: f64,
    pub build_construction2_u_next_ms: f64,
    pub build_construction2_u_next_pack_image_ms: f64,
    pub build_construction2_u_next_commit_ms: f64,
    pub total_ms: f64,
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn emit_debug_timing(trace_prefix: Option<&str>, label: &str, elapsed_ms: f64) {
    if let Some(prefix) = trace_prefix {
        eprintln!("{prefix}.{label}={elapsed_ms:.2}ms");
        let _ = io::stderr().flush();
    }
}

fn rv32im_main_recursion_accumulator_from_f_prime_advice(
    advice: &Rv32imMainRecursionFPrimeAdvice,
) -> Rv32imMainRecursionAccumulator {
    Rv32imMainRecursionAccumulator {
        chunk_count: advice.chunk_count_in(),
        state: advice.state_in.clone(),
        folded_accumulator_digest: advice.folded_accumulator_in_digest,
    }
}

pub fn build_rv32im_main_recursion_f_prime_advices(
    relations: &[Rv32imChunkStepIvcRelation],
) -> Result<Vec<Rv32imMainRecursionFPrimeAdvice>, SimpleKernelError> {
    build_rv32im_main_recursion_f_prime_advices_with_phi_side(relations, &Rv32imMainRecursionPhiSide::zero())
}

pub fn build_rv32im_main_recursion_f_prime_advices_with_perf(
    relations: &[Rv32imChunkStepIvcRelation],
) -> Result<
    (
        Vec<Rv32imMainRecursionFPrimeAdvice>,
        Rv32imMainRecursionFPrimeAdviceBuildPerf,
    ),
    SimpleKernelError,
> {
    let step_cap = rv32im_main_recursion_step_cap_from_relations(relations)?;
    build_rv32im_main_recursion_f_prime_advices_with_phi_side_and_perf(
        relations,
        &Rv32imMainRecursionPhiSide::zero(),
        step_cap,
        None,
    )
}

fn validate_rv32im_main_recursion_single_step_relation(
    relation: &Rv32imChunkStepIvcRelation,
) -> Result<(), SimpleKernelError> {
    if relation.witness.handoff.public_chunk.steps.len() != 1 {
        return Err(SimpleKernelError::Bridge(
            "RV32IM native F' single-step path requires one public step per recursive relation".into(),
        ));
    }
    let fresh = adapt_rv32im_chunk_to_fresh_ccs(&relation.witness.handoff);
    if fresh.fresh_claims.len() != 1 || fresh.fresh_witnesses.len() != 1 {
        return Err(SimpleKernelError::Bridge(
            "RV32IM native F' single-step path requires exactly one fresh CCS instance per recursive relation".into(),
        ));
    }
    Ok(())
}

fn rv32im_main_recursion_step_cap_from_relations(
    relations: &[Rv32imChunkStepIvcRelation],
) -> Result<usize, SimpleKernelError> {
    let step_cap = relations
        .iter()
        .map(|relation| relation.statement.chunk_summary.public_step_count as usize)
        .max()
        .ok_or_else(|| {
            SimpleKernelError::Bridge(
                "RV32IM main recursion F' advice builder requires at least one relation to derive a native step_cap"
                    .into(),
            )
        })?;
    if step_cap == 0 {
        return Err(SimpleKernelError::Bridge(
            "RV32IM main recursion F' advice builder cannot derive a zero-width native step_cap".into(),
        ));
    }
    Ok(step_cap)
}

pub fn build_rv32im_main_recursion_f_prime_advices_single_step(
    relations: &[Rv32imChunkStepIvcRelation],
) -> Result<Vec<Rv32imMainRecursionFPrimeAdvice>, SimpleKernelError> {
    build_rv32im_main_recursion_f_prime_advices_with_phi_side_single_step(
        relations,
        &Rv32imMainRecursionPhiSide::zero(),
    )
}

pub fn build_rv32im_main_recursion_f_prime_advices_single_step_with_perf(
    relations: &[Rv32imChunkStepIvcRelation],
) -> Result<
    (
        Vec<Rv32imMainRecursionFPrimeAdvice>,
        Rv32imMainRecursionFPrimeAdviceBuildPerf,
    ),
    SimpleKernelError,
> {
    build_rv32im_main_recursion_f_prime_advices_with_phi_side_single_step_and_perf(
        relations,
        &Rv32imMainRecursionPhiSide::zero(),
        None,
    )
}

pub fn debug_trace_rv32im_main_recursion_f_prime_advices_single_step_build(
    relations: &[Rv32imChunkStepIvcRelation],
    trace_prefix: &str,
) -> Result<
    (
        Vec<Rv32imMainRecursionFPrimeAdvice>,
        Rv32imMainRecursionFPrimeAdviceBuildPerf,
    ),
    SimpleKernelError,
> {
    build_rv32im_main_recursion_f_prime_advices_with_phi_side_single_step_and_perf(
        relations,
        &Rv32imMainRecursionPhiSide::zero(),
        Some(trace_prefix),
    )
}

pub fn build_rv32im_main_recursion_f_prime_advices_with_side_opening_public(
    relations: &[Rv32imChunkStepIvcRelation],
    side_opening_public: &Rv32imSideOpeningPublic,
) -> Result<Vec<Rv32imMainRecursionFPrimeAdvice>, SimpleKernelError> {
    let (_, phi_side) = build_rv32im_main_recursion_side_lane_from_side_opening_public(side_opening_public)?;
    build_rv32im_main_recursion_f_prime_advices_with_phi_side(relations, &phi_side)
}

pub fn build_rv32im_main_recursion_f_prime_advices_with_side_opening_public_single_step(
    relations: &[Rv32imChunkStepIvcRelation],
    side_opening_public: &Rv32imSideOpeningPublic,
) -> Result<Vec<Rv32imMainRecursionFPrimeAdvice>, SimpleKernelError> {
    let (_, phi_side) = build_rv32im_main_recursion_side_lane_from_side_opening_public(side_opening_public)?;
    build_rv32im_main_recursion_f_prime_advices_with_phi_side_single_step(relations, &phi_side)
}

fn build_rv32im_main_recursion_f_prime_advices_with_phi_side(
    relations: &[Rv32imChunkStepIvcRelation],
    phi_side: &Rv32imMainRecursionPhiSide,
) -> Result<Vec<Rv32imMainRecursionFPrimeAdvice>, SimpleKernelError> {
    let step_cap = rv32im_main_recursion_step_cap_from_relations(relations)?;
    Ok(build_rv32im_main_recursion_f_prime_advices_with_phi_side_and_perf(relations, phi_side, step_cap, None)?.0)
}

fn build_rv32im_main_recursion_f_prime_advices_with_phi_side_and_perf(
    relations: &[Rv32imChunkStepIvcRelation],
    phi_side: &Rv32imMainRecursionPhiSide,
    step_cap: usize,
    trace_prefix: Option<&str>,
) -> Result<
    (
        Vec<Rv32imMainRecursionFPrimeAdvice>,
        Rv32imMainRecursionFPrimeAdviceBuildPerf,
    ),
    SimpleKernelError,
> {
    let total_started = Instant::now();
    let started = Instant::now();
    let vk_fs = build_rv32im_main_recursion_verifier_key_fs_for_step_cap(step_cap)?;
    let verifier_key_ms = elapsed_ms(started);
    emit_debug_timing(trace_prefix, "verifier_key", verifier_key_ms);
    let mut accumulator = Rv32imMainRecursionAccumulator::seed(step_cap);
    let mut current_construction2_u_i: Option<Rv32imMainRecursionConstruction2FreshInstance> = None;
    let mut out = Vec::with_capacity(relations.len());
    let mut perf = Rv32imMainRecursionFPrimeAdviceBuildPerf {
        verifier_key_ms,
        step_count: relations.len(),
        ..Rv32imMainRecursionFPrimeAdviceBuildPerf::default()
    };
    let build_advice = |relation: &Rv32imChunkStepIvcRelation,
                        accumulator: &Rv32imMainRecursionAccumulator,
                        current_construction2_u_i: &Rv32imMainRecursionConstruction2FreshInstance|
     -> Result<Rv32imMainRecursionFPrimeAdvice, SimpleKernelError> {
        let main_circuit_chunk_summary = relation.statement.chunk_summary.clone();
        let native_verified_step_statement =
            build_rv32im_main_recursion_construction2_verified_step_statement_from_summary(
                relation.witness.handoff.bridge_handoff.chunk_index,
                relation.witness.terminal_step,
                &main_circuit_chunk_summary,
                &relation.witness.state_in,
                &relation.witness.state_out,
            );
        let construction2_pi_fold =
            build_rv32im_main_recursion_construction2_pi_fold_from_replay_witness(&relation.witness.replay_witness);
        let x_i = accumulator.x_i(&vk_fs);
        Rv32imMainRecursionFPrimeAdvice::from_parts(
            vk_fs.clone(),
            accumulator.chunk_count,
            rv32im_main_recursion_initial_z_for_step_cap(vk_fs.step_cap()?),
            accumulator.state.carry.terminal_handle.0,
            RV32IM_MAIN_RECURSION_TRIVIAL_PC,
            Rv32imMainRecursionSideLaneWitness::zero(),
            phi_side.clone(),
            accumulator.state.clone(),
            x_i,
            Some(current_construction2_u_i.clone()),
            native_verified_step_statement,
            relation.witness.terminal_step,
            relation.witness.handoff.clone(),
            relation.witness.state_out.clone(),
            relation.witness.replay_witness.clone(),
            construction2_pi_fold,
        )
    };
    let canonical_u_perp = if relations.is_empty() {
        perf.total_ms = elapsed_ms(total_started);
        emit_debug_timing(trace_prefix, "total", perf.total_ms);
        return Ok((out, perf));
    } else {
        let started = Instant::now();
        let canonical_full_width =
            crate::rv32im::construction2::default::build_rv32im_main_recursion_construction2_canonical_full_width(
                &vk_fs, phi_side,
            )?;
        perf.canonical_full_width_ms = elapsed_ms(started);
        emit_debug_timing(trace_prefix, "canonical_full_width", perf.canonical_full_width_ms);
        let started = Instant::now();
        let canonical_u_perp =
            build_rv32im_main_recursion_construction2_default_fresh_instance(&vk_fs, canonical_full_width)?;
        perf.canonical_u_perp_ms = elapsed_ms(started);
        emit_debug_timing(trace_prefix, "canonical_u_perp", perf.canonical_u_perp_ms);
        canonical_u_perp
    };
    for (step_index, relation) in relations.iter().enumerate() {
        let mut step_perf = Rv32imMainRecursionFPrimeAdviceStepBuildPerf::default();
        let started = Instant::now();
        let advice = if accumulator.chunk_count == 0 {
            build_advice(relation, &accumulator, &canonical_u_perp)?
        } else {
            let current_construction2_u_i = current_construction2_u_i.as_ref().ok_or_else(|| {
                SimpleKernelError::Bridge(
                    "RV32IM main recursion F' inductive advice builder is missing the prior-step Construction-2 u_i"
                        .into(),
                )
            })?;
            build_advice(relation, &accumulator, current_construction2_u_i)?
        };
        step_perf.build_advice_ms = elapsed_ms(started);
        emit_debug_timing(
            trace_prefix,
            &format!("step_{step_index}_build_advice"),
            step_perf.build_advice_ms,
        );
        let started = Instant::now();
        let step_trace_prefix = trace_prefix.map(|prefix| format!("{prefix}.step_{step_index}.evaluate"));
        let step_image =
            evaluate_rv32im_main_recursion_f_prime_advice_with_trace(&advice, step_trace_prefix.as_deref())?;
        step_perf.evaluate_step_ms = elapsed_ms(started);
        emit_debug_timing(
            trace_prefix,
            &format!("step_{step_index}_evaluate_step"),
            step_perf.evaluate_step_ms,
        );
        let started = Instant::now();
        let (next_accumulator, next_construction2_u_i) = accumulator.apply_verified_step_image(step_image)?;
        accumulator = next_accumulator;
        step_perf.apply_step_image_ms = elapsed_ms(started);
        emit_debug_timing(
            trace_prefix,
            &format!("step_{step_index}_apply_step_image"),
            step_perf.apply_step_image_ms,
        );
        current_construction2_u_i = Some(next_construction2_u_i);
        out.push(advice);
        perf.per_step.push(step_perf);
    }
    perf.total_ms = elapsed_ms(total_started);
    emit_debug_timing(trace_prefix, "total", perf.total_ms);
    Ok((out, perf))
}

fn build_rv32im_main_recursion_f_prime_advices_with_phi_side_single_step(
    relations: &[Rv32imChunkStepIvcRelation],
    phi_side: &Rv32imMainRecursionPhiSide,
) -> Result<Vec<Rv32imMainRecursionFPrimeAdvice>, SimpleKernelError> {
    Ok(build_rv32im_main_recursion_f_prime_advices_with_phi_side_single_step_and_perf(relations, phi_side, None)?.0)
}

fn build_rv32im_main_recursion_f_prime_advices_with_phi_side_single_step_and_perf(
    relations: &[Rv32imChunkStepIvcRelation],
    phi_side: &Rv32imMainRecursionPhiSide,
    trace_prefix: Option<&str>,
) -> Result<
    (
        Vec<Rv32imMainRecursionFPrimeAdvice>,
        Rv32imMainRecursionFPrimeAdviceBuildPerf,
    ),
    SimpleKernelError,
> {
    let started = Instant::now();
    for relation in relations {
        validate_rv32im_main_recursion_single_step_relation(relation)?;
    }
    let mut built =
        build_rv32im_main_recursion_f_prime_advices_with_phi_side_and_perf(relations, phi_side, 1, trace_prefix)?;
    built.1.relation_validation_ms = elapsed_ms(started);
    emit_debug_timing(trace_prefix, "relation_validation", built.1.relation_validation_ms);
    Ok(built)
}

pub fn build_rv32im_main_recursion_f_prime_public_output(
    advice: &Rv32imMainRecursionFPrimeAdvice,
) -> Result<Rv32imMainRecursionFPrimePublicOutput, SimpleKernelError> {
    let output = CanonicalRv32imMainRecursionFPrimeBody.step(&Rv32imMainRecursionFPrimeInput, advice)?;
    Ok(Rv32imMainRecursionFPrimePublicOutput {
        x_out: output.x_out().clone(),
    })
}

pub fn evaluate_rv32im_main_recursion_f_prime_advice(
    advice: &Rv32imMainRecursionFPrimeAdvice,
) -> Result<Rv32imMainRecursionFPrimeStepImage, SimpleKernelError> {
    Ok(evaluate_rv32im_main_recursion_f_prime_advice_with_perf_and_trace(advice, None)?.0)
}

pub fn evaluate_rv32im_main_recursion_f_prime_advice_with_perf(
    advice: &Rv32imMainRecursionFPrimeAdvice,
) -> Result<(Rv32imMainRecursionFPrimeStepImage, Rv32imMainRecursionFPrimeEvalPerf), SimpleKernelError> {
    evaluate_rv32im_main_recursion_f_prime_advice_with_perf_and_trace(advice, None)
}

fn evaluate_rv32im_main_recursion_f_prime_advice_with_trace(
    advice: &Rv32imMainRecursionFPrimeAdvice,
    trace_prefix: Option<&str>,
) -> Result<Rv32imMainRecursionFPrimeStepImage, SimpleKernelError> {
    Ok(evaluate_rv32im_main_recursion_f_prime_advice_with_perf_and_trace(advice, trace_prefix)?.0)
}

fn evaluate_rv32im_main_recursion_f_prime_advice_with_perf_and_trace(
    advice: &Rv32imMainRecursionFPrimeAdvice,
    trace_prefix: Option<&str>,
) -> Result<(Rv32imMainRecursionFPrimeStepImage, Rv32imMainRecursionFPrimeEvalPerf), SimpleKernelError> {
    evaluate_rv32im_main_recursion_f_prime_step_with_perf_and_trace(
        &Rv32imMainRecursionFPrimeInput,
        advice,
        trace_prefix,
    )
}

fn evaluate_rv32im_main_recursion_f_prime_step(
    _input: &Rv32imMainRecursionFPrimeInput,
    advice: &Rv32imMainRecursionFPrimeAdvice,
) -> Result<Rv32imMainRecursionFPrimeStepImage, SimpleKernelError> {
    Ok(evaluate_rv32im_main_recursion_f_prime_step_with_perf_and_trace(_input, advice, None)?.0)
}

fn evaluate_rv32im_main_recursion_f_prime_step_with_perf_and_trace(
    _input: &Rv32imMainRecursionFPrimeInput,
    advice: &Rv32imMainRecursionFPrimeAdvice,
    trace_prefix: Option<&str>,
) -> Result<(Rv32imMainRecursionFPrimeStepImage, Rv32imMainRecursionFPrimeEvalPerf), SimpleKernelError> {
    let total_started = Instant::now();
    let mut perf = Rv32imMainRecursionFPrimeEvalPerf::default();
    let started = Instant::now();
    let accumulator_in = rv32im_main_recursion_accumulator_from_f_prime_advice(advice);
    let expected_vk_fs =
        build_rv32im_main_recursion_verifier_key_fs_for_step_cap(advice.verifier_key_fs().step_cap()?)?;
    if advice.verifier_key_fs() != &expected_vk_fs {
        return Err(SimpleKernelError::Bridge(
            "RV32IM main recursion F' advice vk_fs does not match the canonical deployed verifier-key context".into(),
        ));
    }
    if advice.chunk_index() != advice.chunk_count_in() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM main recursion F' advice chunk_index does not match chunk_count_in".into(),
        ));
    }
    if advice.z_0() != &rv32im_main_recursion_initial_z_for_step_cap(advice.verifier_key_fs().step_cap()?) {
        return Err(SimpleKernelError::Bridge(
            "RV32IM main recursion F' advice z_0 does not match the canonical initial recursion state".into(),
        ));
    }
    if advice.z_i() != &advice.state_in.carry.terminal_handle.0 {
        return Err(SimpleKernelError::Bridge(
            "RV32IM main recursion F' advice z_i does not match the carried recursive state handle".into(),
        ));
    }
    if advice.pc_i() != RV32IM_MAIN_RECURSION_TRIVIAL_PC {
        return Err(SimpleKernelError::Bridge(
            "RV32IM main recursion F' advice pc_i does not match the trivial RV32IM recursion control lane".into(),
        ));
    }
    if !RV32IM_MAIN_RECURSION_SIDE_WITNESS_ACTIVE && !advice.side_witness().is_zero() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM main recursion F' advice side_witness is non-zero before phi_side is wired".into(),
        ));
    }
    let expected_folded_accumulator_in_digest =
        rv32im_chunk_fold_carry_recursive_accumulator_digest(&advice.state_in.carry);
    if advice.folded_accumulator_in_digest != expected_folded_accumulator_in_digest {
        return Err(SimpleKernelError::Bridge(
            "RV32IM main recursion F' advice folded accumulator input digest does not match state_in".into(),
        ));
    }
    let Some(construction2_u_i) = advice.construction2_input_fresh_instance() else {
        return Err(SimpleKernelError::Bridge(
            "RV32IM main recursion F' advice is missing the threaded HyperNova Construction-2 fresh input u_i".into(),
        ));
    };
    perf.prechecks_ms = elapsed_ms(started);
    emit_debug_timing(trace_prefix, "prechecks", perf.prechecks_ms);
    if advice.chunk_count_in() == 0 {
        let started = Instant::now();
        validate_rv32im_main_recursion_base_case_accumulator(&accumulator_in, advice, construction2_u_i)?;
        perf.base_case_validation_ms = elapsed_ms(started);
        emit_debug_timing(trace_prefix, "base_case_validation", perf.base_case_validation_ms);
    }
    let started = Instant::now();
    let expected_x_i = accumulator_in.x_i(advice.verifier_key_fs());
    if advice.x_i() != &expected_x_i {
        return Err(SimpleKernelError::Bridge(
            "RV32IM main recursion F' advice x_i does not match the carried recursive accumulator image".into(),
        ));
    }
    perf.x_i_check_ms = elapsed_ms(started);
    emit_debug_timing(trace_prefix, "x_i_check", perf.x_i_check_ms);
    let bridge_trace_prefix = trace_prefix.map(|prefix| format!("{prefix}.nifs_bridge"));
    let started = Instant::now();
    let construction2_nifs_bridge = build_rv32im_main_recursion_construction2_nifs_bridge_with_trace(
        advice,
        construction2_u_i,
        bridge_trace_prefix.as_deref(),
    )?;
    perf.build_nifs_bridge_ms = elapsed_ms(started);
    emit_debug_timing(trace_prefix, "build_nifs_bridge", perf.build_nifs_bridge_ms);
    let verify_trace_prefix = trace_prefix.map(|prefix| format!("{prefix}.nifs_verify"));
    let started = Instant::now();
    let (verified_step, verify_perf) = verify_rv32im_main_recursion_construction2_nifs_step_with_perf_and_trace(
        &construction2_nifs_bridge,
        verify_trace_prefix.as_deref(),
    )?;
    perf.verify_nifs_step_ms = elapsed_ms(started);
    perf.verify_nifs_chunk_relation_ms = verify_perf.chunk_relation_verify_ms;
    perf.verify_nifs_derive_next_state_ms = verify_perf.derive_next_state_ms;
    emit_debug_timing(trace_prefix, "verify_nifs_step", perf.verify_nifs_step_ms);
    let next_state = verified_step.state;
    let started = Instant::now();
    let _ = Rv32imMainRecursionAccumulatorSurface::try_from_carry(&next_state.carry.main, "F' next-state accumulator")?;
    perf.next_state_surface_check_ms = elapsed_ms(started);
    emit_debug_timing(
        trace_prefix,
        "next_state_surface_check",
        perf.next_state_surface_check_ms,
    );
    let chunk_count_out = accumulator_in.chunk_count + 1;
    let folded_accumulator_digest = rv32im_chunk_fold_carry_recursive_accumulator_digest(&next_state.carry);
    let started = Instant::now();
    let z_next = next_state.carry.terminal_handle.0;
    let pc_next = RV32IM_MAIN_RECURSION_TRIVIAL_PC;
    let phi_side = advice.phi_side().clone();
    let x_out = rv32im_main_recursion_x_out(
        advice.verifier_key_fs(),
        chunk_count_out,
        *advice.z_0(),
        z_next,
        pc_next,
        folded_accumulator_digest,
    );
    perf.derive_public_outputs_ms = elapsed_ms(started);
    emit_debug_timing(trace_prefix, "derive_public_outputs", perf.derive_public_outputs_ms);
    let started = Instant::now();
    let u_next_trace_prefix = trace_prefix.map(|prefix| format!("{prefix}.construction2_u_next"));
    let construction2_u_next = if let Some(prefix) = u_next_trace_prefix.as_deref() {
        debug_trace_build_rv32im_main_recursion_construction2_fresh_instance_with_input_and_x_i(
            advice,
            construction2_u_i,
            x_out.clone(),
            prefix,
        )?
    } else {
        let (fresh_instance, fresh_instance_perf) =
            build_rv32im_main_recursion_construction2_fresh_instance_with_input_and_x_i_with_perf(
                advice,
                construction2_u_i,
                x_out.clone(),
            )?;
        perf.build_construction2_u_next_pack_image_ms = fresh_instance_perf.pack_image_ms;
        perf.build_construction2_u_next_commit_ms = fresh_instance_perf.commit_ms;
        fresh_instance
    };
    perf.build_construction2_u_next_ms = elapsed_ms(started);
    emit_debug_timing(
        trace_prefix,
        "build_construction2_u_next",
        perf.build_construction2_u_next_ms,
    );
    if construction2_u_next.x_i() != &x_out {
        return Err(SimpleKernelError::Bridge(
            "RV32IM main recursion F' produced a Construction-2 output u_{i+1} whose x_i does not match x_{i+1}".into(),
        ));
    }
    perf.total_ms = elapsed_ms(total_started);
    emit_debug_timing(trace_prefix, "total", perf.total_ms);
    Ok((
        Rv32imMainRecursionFPrimeStepImage {
            chunk_count: chunk_count_out,
            z_next,
            pc_next,
            phi_side,
            construction2_u_next,
            next_state,
            folded_accumulator_digest,
            x_out,
        },
        perf,
    ))
}

pub fn verify_rv32im_main_recursion_f_prime_public_output(
    public_output: &Rv32imMainRecursionFPrimePublicOutput,
    advice: &Rv32imMainRecursionFPrimeAdvice,
) -> Result<Rv32imMainRecursionFPrimeStepImage, SimpleKernelError> {
    let output = CanonicalRv32imMainRecursionFPrimeBody.step(&Rv32imMainRecursionFPrimeInput, advice)?;
    if public_output.x_out != *output.x_out() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM main recursion F' public output x_out does not match the verified recursive step".into(),
        ));
    }
    Ok(output)
}
