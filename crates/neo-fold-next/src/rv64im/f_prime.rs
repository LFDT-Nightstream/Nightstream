//! Owns native RV64IM F' semantics and the recursion hash-image boundary.

use std::collections::BTreeMap;
use std::io::{self, Write};
use std::time::Instant;

use neo_ajtai::Commitment;
use neo_ccs::{check_ccs_rowwise_zero, check_ce_consistency, CeClaim, CeWitness, Mat};
use neo_math::{ring::D, KExtensions, F, K};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};

use crate::finalize::digest32_as_fields;
use crate::nightstream::rv64im::{Rv64imEvalPublic, Rv64imOpenedObjectPublic, Rv64imSideOpeningPublic};
use crate::proof::Carry;
use crate::rv64im::chunk_fold_step::adapt_rv64im_chunk_to_fresh_ccs;
use crate::rv64im::chunk_step_ivc::{rv64im_chunk_step_ivc_initial_state, Rv64imChunkStepIvcRelation};
use crate::rv64im::construction2::{
    build_rv64im_main_recursion_construction2_default_fresh_instance,
    build_rv64im_main_recursion_construction2_fresh_instance_with_input_and_x_i,
    build_rv64im_main_recursion_construction2_nifs_bridge_with_trace,
    build_rv64im_main_recursion_construction2_pi_fold_from_relation,
    build_rv64im_main_recursion_construction2_verified_step_statement_from_relation,
    debug_trace_build_rv64im_main_recursion_construction2_fresh_instance_with_input_and_x_i,
    verify_rv64im_main_recursion_construction2_nifs_step_with_trace, Rv64imMainRecursionConstruction2FreshInstance,
    Rv64imMainRecursionConstruction2PiFoldProof, Rv64imMainRecursionConstruction2StateImage,
    Rv64imMainRecursionConstruction2VerifiedStepStatement,
};
use crate::rv64im::final_relation::{rv64im_chunk_fold_carry_recursive_accumulator_digest, Rv64imChunkFoldState};
use crate::rv64im::kernel::{
    rv64im_cached_root_main_lane_context, FamilyEvalSchemaId, PackedColumnEval, Rv64imVerifiedKernelChunkHandoff,
};
use crate::rv64im::main_relation_trace::{
    build_rv64im_main_circuit_chunk_trace_from_authoritative_parts, Rv64imMainCircuitChunkTrace,
};
use crate::rv64im::recursion_shape::build_rv64im_recursion_shape;
use crate::rv64im::SimpleKernelError;

/// Canonical recursion public input image for the current stack.
///
/// This is intentionally a named boundary type, not a bare `[u8; 32]`.
/// The semantic `enc_inst` image is the digest bit-decomposition in little-
/// endian bit order, one low-norm field element per bit:
///   bit_j = (digest_bytes[j / 8] >> (j % 8)) & 1
/// This keeps `x` binary and therefore `||x||_∞ < 2`.
///
/// The current backend still also needs a legacy 4-limb digest packing bridge:
///   limb_j = u64::from_le_bytes(digest_bytes[8*j .. 8*(j+1)])
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imEncodedPublicInput {
    digest_bytes: [u8; 32],
}

pub const RV64IM_ENC_INST_BITS: usize = 256;
pub const RV64IM_ENC_INST_RING_DEGREE: usize = D;
pub const RV64IM_ENC_INST_RING_SLOTS: usize =
    (RV64IM_ENC_INST_BITS + RV64IM_ENC_INST_RING_DEGREE - 1) / RV64IM_ENC_INST_RING_DEGREE;
pub const RV64IM_MAIN_RECURSION_TRIVIAL_PC: u64 = 1;
pub const RV64IM_MAIN_RECURSION_ACCUMULATOR_SLOTS: usize = 1;
pub const RV64IM_MAIN_RECURSION_SIDE_WITNESS_ACTIVE: bool = false;
pub const RV64IM_MAIN_RECURSION_PHI_SIDE_ACTIVE: bool = true;
pub const RV64IM_MAIN_RECURSION_SIDE_LANE_ACTIVE: bool =
    RV64IM_MAIN_RECURSION_SIDE_WITNESS_ACTIVE || RV64IM_MAIN_RECURSION_PHI_SIDE_ACTIVE;

#[derive(Clone, Debug)]
pub(crate) struct Rv64imMainRecursionAccumulatorBundle {
    main: Carry,
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imMainRecursionAccumulatorArray<const SLOTS: usize> {
    slots: [Rv64imMainRecursionAccumulatorBundle; SLOTS],
}

pub(crate) type Rv64imMainRecursionAccumulatorSurface =
    Rv64imMainRecursionAccumulatorArray<RV64IM_MAIN_RECURSION_ACCUMULATOR_SLOTS>;

impl<const SLOTS: usize> Rv64imMainRecursionAccumulatorArray<SLOTS> {
    pub(crate) fn try_from_carry(main: &Carry, label: &str) -> Result<Self, SimpleKernelError> {
        if main.claims.len() != main.witnesses.len() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM main recursion {label} requires one witness per carried CE claim"
            )));
        }
        if SLOTS != 1 {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM main recursion {label} only supports the current single-PC specialization"
            )));
        }
        Ok(Self {
            slots: core::array::from_fn(|_| Rv64imMainRecursionAccumulatorBundle { main: main.clone() }),
        })
    }

    pub(crate) fn slot(&self, slot: usize) -> Result<&Rv64imMainRecursionAccumulatorBundle, SimpleKernelError> {
        self.slots.get(slot).ok_or_else(|| {
            SimpleKernelError::Bridge(format!(
                "RV64IM main recursion accumulator slot {slot} is out of bounds for {SLOTS} slots"
            ))
        })
    }
}

impl Rv64imMainRecursionAccumulatorBundle {
    pub(crate) fn carry(&self) -> &Carry {
        &self.main
    }
}

fn rv64im_enc_inst_bit_image_le(digest_bytes: [u8; 32]) -> [u8; RV64IM_ENC_INST_BITS] {
    core::array::from_fn(|bit_index| {
        let byte = digest_bytes[bit_index / 8];
        (byte >> (bit_index % 8)) & 1
    })
}

impl Rv64imEncodedPublicInput {
    pub fn from_digest_bytes(digest_bytes: [u8; 32]) -> Self {
        Self { digest_bytes }
    }

    pub fn bytes(&self) -> [u8; 32] {
        self.digest_bytes
    }

    pub fn bytes_mut(&mut self) -> &mut [u8; 32] {
        &mut self.digest_bytes
    }

    pub fn bit_image(&self) -> [u8; RV64IM_ENC_INST_BITS] {
        rv64im_enc_inst_bit_image_le(self.digest_bytes)
    }

    pub fn field_image(&self) -> [F; RV64IM_ENC_INST_BITS] {
        self.bit_image().map(|bit| F::from_u64(bit as u64))
    }

    /// Canonical ring-coordinate image of `enc_inst(h)`.
    ///
    /// This is the module-boundary packing rule the checklist relies on:
    /// `x_ring[q].coeff[r] = x_F[q * D + r]` with zero-padding after bit 255.
    pub fn ring_image(&self) -> [[F; RV64IM_ENC_INST_RING_DEGREE]; RV64IM_ENC_INST_RING_SLOTS] {
        let field_image = self.field_image();
        core::array::from_fn(|ring_slot| {
            core::array::from_fn(|coeff_index| {
                field_image
                    .get(ring_slot * RV64IM_ENC_INST_RING_DEGREE + coeff_index)
                    .copied()
                    .unwrap_or(F::ZERO)
            })
        })
    }

    pub fn is_binary_low_norm(&self) -> bool {
        self.bit_image().into_iter().all(|bit| bit <= 1)
    }

    /// Legacy backend bridge: 4 little-endian u64 limbs of the digest bytes.
    /// This is not the semantic `enc_inst` image; it remains only for the
    /// current backend surfaces that have not yet been rewritten.
    pub fn digest_fields(&self) -> [F; 4] {
        digest32_as_fields(self.digest_bytes)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionStepStatement {
    pub x_out: Rv64imEncodedPublicInput,
    pub folded_accumulator_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionBackendStepStatement {
    pub x_out: Rv64imEncodedPublicInput,
    pub folded_accumulator_digest: [u8; 32],
}

impl Rv64imMainRecursionBackendStepStatement {
    pub fn native_statement(&self) -> Rv64imMainRecursionStepStatement {
        Rv64imMainRecursionStepStatement {
            x_out: self.x_out.clone(),
            folded_accumulator_digest: self.folded_accumulator_digest,
        }
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_step_spartan_statement");
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_spartan_statement/version",
            b"v8",
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_spartan_statement/x_out",
            &self.x_out.bytes(),
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_spartan_statement/folded_accumulator_digest",
            &self.folded_accumulator_digest,
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imVerifierKeyFs {
    pub domain_tag_digest: [u8; 32],
    pub main_lane_shape_digest: [u8; 32],
}

impl Rv64imVerifierKeyFs {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_verifier_key_fs");
        tr.append_message(b"neo.fold.next/rv64im/main_recursion_verifier_key_fs/version", b"v1");
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_verifier_key_fs/domain_tag_digest",
            &self.domain_tag_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_verifier_key_fs/main_lane_shape_digest",
            &self.main_lane_shape_digest,
        );
        tr.digest32()
    }
}

pub fn build_rv64im_main_recursion_verifier_key_fs() -> Result<Rv64imVerifierKeyFs, SimpleKernelError> {
    let mut domain_tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_verifier_key_fs/domain");
    domain_tr.append_message(
        b"neo.fold.next/rv64im/main_recursion_verifier_key_fs/domain/version",
        b"v1",
    );
    domain_tr.append_message(
        b"neo.fold.next/rv64im/main_recursion_verifier_key_fs/domain/tag",
        b"neo.fold.next/rv64im/main_recursion_f_prime_x_out",
    );

    Ok(Rv64imVerifierKeyFs {
        domain_tag_digest: domain_tr.digest32(),
        main_lane_shape_digest: build_rv64im_recursion_shape()?.canonical_digest(),
    })
}

fn rv64im_main_recursion_initial_z() -> [u8; 32] {
    crate::rv64im::chunk_step_ivc::rv64im_chunk_step_ivc_initial_state()
        .carry
        .terminal_handle
        .0
}

pub(crate) fn build_rv64im_main_recursion_backend_statement_from_parts(
    chunk_count: u64,
    folded_accumulator_digest: [u8; 32],
    terminal_handle_digest: [u8; 32],
) -> Result<Rv64imMainRecursionBackendStepStatement, SimpleKernelError> {
    let vk_fs = build_rv64im_main_recursion_verifier_key_fs()?;
    Ok(build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs(
        &vk_fs,
        chunk_count,
        folded_accumulator_digest,
        terminal_handle_digest,
    ))
}

pub(crate) fn build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs(
    vk_fs: &Rv64imVerifierKeyFs,
    chunk_count: u64,
    folded_accumulator_digest: [u8; 32],
    terminal_handle_digest: [u8; 32],
) -> Rv64imMainRecursionBackendStepStatement {
    Rv64imMainRecursionBackendStepStatement {
        x_out: rv64im_main_recursion_x_out(
            vk_fs,
            chunk_count,
            rv64im_main_recursion_initial_z(),
            terminal_handle_digest,
            RV64IM_MAIN_RECURSION_TRIVIAL_PC,
            folded_accumulator_digest,
        ),
        folded_accumulator_digest,
    }
}

pub(crate) fn build_rv64im_main_recursion_backend_statement_from_advice(
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Result<Rv64imMainRecursionBackendStepStatement, SimpleKernelError> {
    let step_image = evaluate_rv64im_main_recursion_f_prime_advice(advice)?;
    Ok(build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs(
        advice.verifier_key_fs(),
        step_image.chunk_count(),
        step_image.folded_accumulator_digest(),
        step_image.next_state.carry.terminal_handle.0,
    ))
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imMainRecursionAccumulator {
    chunk_count: u64,
    state: Rv64imChunkFoldState,
}

impl Rv64imMainRecursionAccumulator {
    fn seed() -> Self {
        Self {
            chunk_count: 0,
            state: rv64im_chunk_step_ivc_initial_state(),
        }
    }

    fn apply_verified_step_image(
        &self,
        output: &Rv64imMainRecursionFPrimeStepImage,
    ) -> Result<Self, SimpleKernelError> {
        if output.chunk_count != self.chunk_count + 1 {
            return Err(SimpleKernelError::Bridge(
                "RV64IM main recursion step image chunk_count does not advance the carried recursive position".into(),
            ));
        }
        if output.folded_accumulator_digest
            != rv64im_chunk_fold_carry_recursive_accumulator_digest(&output.next_state.carry)
        {
            return Err(SimpleKernelError::Bridge(
                "RV64IM main recursion step image folded accumulator digest does not match next_state".into(),
            ));
        }
        if output.z_next != output.next_state.carry.terminal_handle.0 {
            return Err(SimpleKernelError::Bridge(
                "RV64IM main recursion step image z_next does not match next_state".into(),
            ));
        }
        if output.pc_next != RV64IM_MAIN_RECURSION_TRIVIAL_PC {
            return Err(SimpleKernelError::Bridge(
                "RV64IM main recursion step image pc_next does not match the trivial RV64IM recursion control lane"
                    .into(),
            ));
        }
        Ok(Self {
            chunk_count: output.chunk_count,
            state: output.next_state.clone(),
        })
    }

    fn x_i(&self, vk_fs: &Rv64imVerifierKeyFs) -> Rv64imEncodedPublicInput {
        rv64im_main_recursion_x_out(
            vk_fs,
            self.chunk_count,
            rv64im_main_recursion_initial_z(),
            self.state.carry.terminal_handle.0,
            RV64IM_MAIN_RECURSION_TRIVIAL_PC,
            rv64im_chunk_fold_carry_recursive_accumulator_digest(&self.state.carry),
        )
    }
}

fn validate_rv64im_main_recursion_base_case_accumulator(
    accumulator: &Rv64imMainRecursionAccumulator,
    advice: &Rv64imMainRecursionFPrimeAdvice,
    construction2_u_i: &Rv64imMainRecursionConstruction2FreshInstance,
) -> Result<(), SimpleKernelError> {
    if accumulator.chunk_count != 0 {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion F' base case must begin at recursive position i = 0".into(),
        ));
    }
    if advice.z_0() != advice.z_i() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion F' base case does not satisfy z_0 == z_i".into(),
        ));
    }
    let canonical_full_width =
        crate::rv64im::construction2_default::build_rv64im_main_recursion_construction2_canonical_full_width(
            advice.verifier_key_fs(),
            advice.phi_side(),
        )?;
    let expected_default = build_rv64im_main_recursion_construction2_default_fresh_instance(
        advice.verifier_key_fs(),
        canonical_full_width,
    )?;
    if construction2_u_i != &expected_default {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion F' base case does not carry the canonical default fresh instance u_perp".into(),
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

pub(crate) fn build_rv64im_main_recursion_base_case_default_carry(
    state_like: &Rv64imChunkFoldState,
) -> Result<Carry, SimpleKernelError> {
    let carried_bundle =
        Rv64imMainRecursionAccumulatorSurface::try_from_carry(&state_like.carry.main, "F' base-case output")?;
    let (params, log, structure) = rv64im_cached_root_main_lane_context()?;
    let carried_main = carried_bundle.slot(0)?.carry();
    if carried_main.claims.is_empty() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion F' base-case default carry requires a non-empty carried CE bundle template".into(),
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
                "RV64IM main recursion F' base-case zero witness failed CCS row-wise zero for carried CE claim {claim_index}: {err}"
            ))
        })?;
        check_ce_consistency(
            params,
            structure,
            log,
            &zero_claim,
            &CeWitness {
                Z: zero_witness.clone(),
            },
        )
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM main recursion F' base-case default CE claim failed consistency for carried CE claim {claim_index}: {err}"
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

fn rv64im_main_recursion_fresh_instance_digest(
    x_i: &Rv64imEncodedPublicInput,
    construction2_u_i: Option<&Rv64imMainRecursionConstruction2FreshInstance>,
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_fresh_instance");
    tr.append_message(b"neo.fold.next/rv64im/main_recursion_fresh_instance/version", b"v4");
    tr.append_message(b"neo.fold.next/rv64im/main_recursion_fresh_instance/x_i", &x_i.bytes());
    if let Some(construction2_u_i) = construction2_u_i {
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_fresh_instance/construction2_u_i",
            &construction2_u_i.expected_digest(),
        );
    }
    tr.digest32()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionSideClaim {
    pub schema: FamilyEvalSchemaId,
    pub slot: u32,
    pub point_words: Vec<u64>,
    pub payload_words: Vec<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionSideLaneWitness {
    pub(crate) claims: Vec<Rv64imMainRecursionSideClaim>,
}

impl Rv64imMainRecursionSideLaneWitness {
    pub fn zero() -> Self {
        Self { claims: Vec::new() }
    }

    pub fn claims(&self) -> &[Rv64imMainRecursionSideClaim] {
        &self.claims
    }

    pub fn claim_count(&self) -> u64 {
        self.claims.len() as u64
    }

    pub fn is_zero(&self) -> bool {
        self.claims.is_empty()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionPhiSide {
    pub(crate) commitment_words: Vec<Vec<u64>>,
}

impl Rv64imMainRecursionPhiSide {
    pub fn zero() -> Self {
        Self {
            commitment_words: Vec::new(),
        }
    }

    pub fn commitment_words(&self) -> &[Vec<u64>] {
        &self.commitment_words
    }

    pub fn commitment_count(&self) -> u64 {
        self.commitment_words.len() as u64
    }

    pub fn is_zero(&self) -> bool {
        self.commitment_words.is_empty()
    }
}

fn digest32_as_u64_words(digest: [u8; 32]) -> [u64; 4] {
    core::array::from_fn(|limb| {
        let start = limb * 8;
        u64::from_le_bytes(digest[start..start + 8].try_into().expect("digest limb"))
    })
}

fn k_slice_as_u64_words(values: &[K]) -> Vec<u64> {
    values
        .iter()
        .flat_map(|&value| value.as_coeffs().map(|coeff| coeff.as_canonical_u64()))
        .collect()
}

fn packed_column_evals_as_u64_words(values: &[PackedColumnEval]) -> Vec<u64> {
    values
        .iter()
        .flat_map(|column_eval| k_slice_as_u64_words(&column_eval.coeffs))
        .collect()
}

fn build_rv64im_main_recursion_side_claim_from_eval_public(
    eval: &Rv64imEvalPublic,
) -> Result<Rv64imMainRecursionSideClaim, SimpleKernelError> {
    eval.claim.validate().map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM main recursion side-lane adapter carries an internally inconsistent {:?}/{} eval: {err}",
            eval.claim.payload.schema, eval.claim.id.slot
        ))
    })?;
    if eval.digest != eval.expected_digest() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM main recursion side-lane adapter carries a stale {:?}/{} eval digest",
            eval.claim.payload.schema, eval.claim.id.slot
        )));
    }
    Ok(Rv64imMainRecursionSideClaim {
        schema: eval.claim.payload.schema,
        slot: eval.claim.id.slot,
        point_words: k_slice_as_u64_words(&eval.claim.point),
        payload_words: packed_column_evals_as_u64_words(&eval.claim.payload.column_evals),
    })
}

fn build_rv64im_main_recursion_phi_side_commitment_words(
    opened_object: &Rv64imOpenedObjectPublic,
) -> Result<Vec<u64>, SimpleKernelError> {
    let Some(expected_schema) = FamilyEvalSchemaId::from_family(opened_object.opened_object.family) else {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM main recursion side-lane adapter carries unsupported opened-object family {:?}",
            opened_object.opened_object.family
        )));
    };
    if expected_schema != opened_object.schema {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM main recursion side-lane adapter opened-object schema mismatch: expected {:?}, got {:?}",
            expected_schema, opened_object.schema
        )));
    }
    if opened_object.digest != opened_object.expected_digest() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM main recursion side-lane adapter carries a stale {:?} opened-object digest",
            opened_object.schema
        )));
    }

    let mut words = Vec::with_capacity(15);
    words.push(opened_object.schema.tag());
    words.push(opened_object.opened_object.layout_version);
    words.push(opened_object.opened_object.row_domain_log_size as u64);
    words.extend(digest32_as_u64_words(
        opened_object.opened_object.commitment_root_digest,
    ));
    words.extend(digest32_as_u64_words(opened_object.commitment_context.pp_seed_digest));
    words.extend(digest32_as_u64_words(
        opened_object.commitment_context.module_shape_digest,
    ));
    Ok(words)
}

pub fn build_rv64im_main_recursion_side_lane_from_side_opening_public(
    public: &Rv64imSideOpeningPublic,
) -> Result<(Rv64imMainRecursionSideLaneWitness, Rv64imMainRecursionPhiSide), SimpleKernelError> {
    if public.digest != public.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion side-lane adapter carries a stale side-opening public digest".into(),
        ));
    }

    let mut opened_object_by_schema = BTreeMap::<FamilyEvalSchemaId, &Rv64imOpenedObjectPublic>::new();
    let mut previous_schema = None;
    let mut commitment_words = Vec::with_capacity(public.opened_objects.len());
    for opened_object in &public.opened_objects {
        if let Some(previous_schema) = previous_schema {
            if previous_schema >= opened_object.schema {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM main recursion side-lane adapter requires strict canonical opened-object schema order"
                        .into(),
                ));
            }
        }
        if opened_object_by_schema
            .insert(opened_object.schema, opened_object)
            .is_some()
        {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM main recursion side-lane adapter carries duplicate {:?} opened objects",
                opened_object.schema
            )));
        }
        commitment_words.push(build_rv64im_main_recursion_phi_side_commitment_words(opened_object)?);
        previous_schema = Some(opened_object.schema);
    }

    let mut previous_key = None;
    let mut claims = Vec::with_capacity(public.evals.len());
    for eval in &public.evals {
        let key = (eval.claim.payload.schema, eval.claim.id.slot);
        if let Some(previous_key) = previous_key {
            if previous_key >= key {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM main recursion side-lane adapter requires strict canonical eval order".into(),
                ));
            }
        }
        let Some(opened_object) = opened_object_by_schema.get(&eval.claim.payload.schema) else {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM main recursion side-lane adapter is missing the {:?} opened object for slot {}",
                eval.claim.payload.schema, eval.claim.id.slot
            )));
        };
        if eval.claim.opened_object != opened_object.opened_object
            || eval.claim.commitment_context != opened_object.commitment_context
        {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM main recursion side-lane adapter {:?}/{} eval does not match the opened-object public",
                eval.claim.payload.schema, eval.claim.id.slot
            )));
        }
        claims.push(build_rv64im_main_recursion_side_claim_from_eval_public(eval)?);
        previous_key = Some(key);
    }

    Ok((
        Rv64imMainRecursionSideLaneWitness { claims },
        Rv64imMainRecursionPhiSide { commitment_words },
    ))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionFPrimeInput;

pub trait Rv64imMainRecursionFPrimeBody {
    fn step(
        &self,
        input: &Rv64imMainRecursionFPrimeInput,
        advice: &Rv64imMainRecursionFPrimeAdvice,
    ) -> Result<Rv64imMainRecursionFPrimeStepImage, SimpleKernelError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CanonicalRv64imMainRecursionFPrimeBody;

impl Rv64imMainRecursionFPrimeBody for CanonicalRv64imMainRecursionFPrimeBody {
    fn step(
        &self,
        input: &Rv64imMainRecursionFPrimeInput,
        advice: &Rv64imMainRecursionFPrimeAdvice,
    ) -> Result<Rv64imMainRecursionFPrimeStepImage, SimpleKernelError> {
        evaluate_rv64im_main_recursion_f_prime_step(input, advice)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionFPrimePublicOutput {
    x_out: Rv64imEncodedPublicInput,
}

impl Rv64imMainRecursionFPrimePublicOutput {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_f_prime_public_output");
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_f_prime_public_output/version",
            b"v1",
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_f_prime_public_output/x_out",
            &self.x_out.bytes(),
        );
        tr.digest32()
    }

    pub fn x_out(&self) -> &Rv64imEncodedPublicInput {
        &self.x_out
    }

    pub fn x_out_mut(&mut self) -> &mut Rv64imEncodedPublicInput {
        &mut self.x_out
    }
}

#[derive(Clone, Debug)]
pub struct Rv64imMainRecursionFPrimeAdvice {
    vk_fs: Rv64imVerifierKeyFs,
    chunk_count_in: u64,
    z_0: [u8; 32],
    z_i: [u8; 32],
    pc_i: u64,
    side_witness: Rv64imMainRecursionSideLaneWitness,
    phi_side: Rv64imMainRecursionPhiSide,
    pub(crate) state_in: Rv64imChunkFoldState,
    x_i: Rv64imEncodedPublicInput,
    construction2_input_u_i: Option<Rv64imMainRecursionConstruction2FreshInstance>,
    native_verified_step_statement: Rv64imMainRecursionConstruction2VerifiedStepStatement,
    terminal_step: bool,
    verified_kernel_handoff: Rv64imVerifiedKernelChunkHandoff,
    state_out: Rv64imChunkFoldState,
    main_circuit_chunk_trace: Rv64imMainCircuitChunkTrace,
    construction2_pi_fold: Rv64imMainRecursionConstruction2PiFoldProof,
}

impl Rv64imMainRecursionFPrimeAdvice {
    pub(crate) fn from_parts(
        vk_fs: Rv64imVerifierKeyFs,
        chunk_count_in: u64,
        z_0: [u8; 32],
        z_i: [u8; 32],
        pc_i: u64,
        side_witness: Rv64imMainRecursionSideLaneWitness,
        phi_side: Rv64imMainRecursionPhiSide,
        state_in: Rv64imChunkFoldState,
        x_i: Rv64imEncodedPublicInput,
        construction2_input_u_i: Option<Rv64imMainRecursionConstruction2FreshInstance>,
        native_verified_step_statement: Rv64imMainRecursionConstruction2VerifiedStepStatement,
        terminal_step: bool,
        verified_kernel_handoff: Rv64imVerifiedKernelChunkHandoff,
        state_out: Rv64imChunkFoldState,
        main_circuit_chunk_trace: Rv64imMainCircuitChunkTrace,
        construction2_pi_fold: Rv64imMainRecursionConstruction2PiFoldProof,
    ) -> Result<Self, SimpleKernelError> {
        if let Some(construction2_input_u_i) = construction2_input_u_i.as_ref() {
            if chunk_count_in != 0 && construction2_input_u_i.x_i() != &x_i {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM main recursion advice cannot bind a Construction-2 u_i whose x_i disagrees with the carried native image"
                        .into(),
                ));
            }
        }
        if native_verified_step_statement.chunk_index != verified_kernel_handoff.bridge_handoff.chunk_index {
            return Err(SimpleKernelError::Bridge(
                "RV64IM main recursion advice carries a native verified-step chunk_index that disagrees with the verified bridge handoff"
                    .into(),
            ));
        }
        if native_verified_step_statement.state_in != state_in.carry.terminal_handle.0 {
            return Err(SimpleKernelError::Bridge(
                "RV64IM main recursion advice carries a native verified-step state_in that disagrees with the running recursive state"
                    .into(),
            ));
        }
        if native_verified_step_statement.state_out != state_out.carry.terminal_handle.0 {
            return Err(SimpleKernelError::Bridge(
                "RV64IM main recursion advice carries a native verified-step state_out that disagrees with the next recursive state"
                    .into(),
            ));
        }
        if native_verified_step_statement.public_chunk_digest != verified_kernel_handoff.public_chunk_digest {
            return Err(SimpleKernelError::Bridge(
                "RV64IM main recursion advice carries a native verified-step public chunk digest that disagrees with the verified bridge handoff"
                    .into(),
            ));
        }
        if native_verified_step_statement.chunk_relation_digest
            != main_circuit_chunk_trace.ccs_trace.chunk_relation_digest
        {
            return Err(SimpleKernelError::Bridge(
                "RV64IM main recursion advice carries a native verified-step relation digest that disagrees with the replayed main-circuit chunk trace"
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
            state_in,
            x_i,
            construction2_input_u_i,
            native_verified_step_statement,
            terminal_step,
            verified_kernel_handoff,
            state_out,
            main_circuit_chunk_trace,
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

    pub fn x_i(&self) -> &Rv64imEncodedPublicInput {
        &self.x_i
    }

    pub fn x_hash(&self) -> &Rv64imEncodedPublicInput {
        self.x_i()
    }

    pub fn construction2_input_fresh_instance(&self) -> Option<&Rv64imMainRecursionConstruction2FreshInstance> {
        self.construction2_input_u_i.as_ref()
    }

    pub fn side_witness(&self) -> &Rv64imMainRecursionSideLaneWitness {
        &self.side_witness
    }

    pub fn phi_side(&self) -> &Rv64imMainRecursionPhiSide {
        &self.phi_side
    }

    pub fn folded_accumulator_in_digest(&self) -> [u8; 32] {
        rv64im_chunk_fold_carry_recursive_accumulator_digest(&self.state_in.carry)
    }

    pub fn verifier_key_fs(&self) -> &Rv64imVerifierKeyFs {
        &self.vk_fs
    }

    pub fn running_state(&self) -> &Rv64imChunkFoldState {
        &self.state_in
    }

    pub(crate) fn running_state_mut(&mut self) -> &mut Rv64imChunkFoldState {
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
        rv64im_main_recursion_fresh_instance_digest(&self.x_i, self.construction2_input_u_i.as_ref())
    }

    pub fn fresh_state_out(&self) -> &Rv64imChunkFoldState {
        &self.state_out
    }

    pub(crate) fn main_circuit_chunk_trace(&self) -> &Rv64imMainCircuitChunkTrace {
        &self.main_circuit_chunk_trace
    }

    pub(crate) fn fresh_state_out_mut(&mut self) -> &mut Rv64imChunkFoldState {
        &mut self.state_out
    }

    pub(crate) fn verified_kernel_handoff(&self) -> &Rv64imVerifiedKernelChunkHandoff {
        &self.verified_kernel_handoff
    }

    pub(crate) fn verified_kernel_handoff_mut(&mut self) -> &mut Rv64imVerifiedKernelChunkHandoff {
        &mut self.verified_kernel_handoff
    }

    pub(crate) fn verifier_key_fs_mut(&mut self) -> &mut Rv64imVerifierKeyFs {
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

    pub(crate) fn side_witness_mut(&mut self) -> &mut Rv64imMainRecursionSideLaneWitness {
        &mut self.side_witness
    }

    pub(crate) fn x_i_mut(&mut self) -> &mut Rv64imEncodedPublicInput {
        &mut self.x_i
    }

    pub(crate) fn construction2_input_fresh_instance_mut(
        &mut self,
    ) -> Option<&mut Rv64imMainRecursionConstruction2FreshInstance> {
        self.construction2_input_u_i.as_mut()
    }

    pub(crate) fn chunk_index_mut(&mut self) -> &mut u64 {
        &mut self.native_verified_step_statement.chunk_index
    }

    pub(crate) fn construction2_pi_fold(&self) -> &Rv64imMainRecursionConstruction2PiFoldProof {
        &self.construction2_pi_fold
    }

    pub(crate) fn construction2_pi_fold_mut(&mut self) -> &mut Rv64imMainRecursionConstruction2PiFoldProof {
        &mut self.construction2_pi_fold
    }

    pub(crate) fn terminal_step_mut(&mut self) -> &mut bool {
        &mut self.terminal_step
    }
}

#[derive(Clone, Debug)]
pub struct Rv64imMainRecursionFPrimeStepImage {
    chunk_count: u64,
    z_next: [u8; 32],
    pc_next: u64,
    phi_side: Rv64imMainRecursionPhiSide,
    construction2_u_next: Rv64imMainRecursionConstruction2FreshInstance,
    pub(crate) next_state: Rv64imChunkFoldState,
    folded_accumulator_digest: [u8; 32],
    x_out: Rv64imEncodedPublicInput,
}

impl Rv64imMainRecursionFPrimeStepImage {
    pub fn chunk_count(&self) -> u64 {
        self.chunk_count
    }

    pub fn z_next(&self) -> &[u8; 32] {
        &self.z_next
    }

    pub fn pc_next(&self) -> u64 {
        self.pc_next
    }

    pub fn phi_side(&self) -> &Rv64imMainRecursionPhiSide {
        &self.phi_side
    }

    pub fn construction2_u_next(&self) -> &Rv64imMainRecursionConstruction2FreshInstance {
        &self.construction2_u_next
    }

    pub fn folded_accumulator_digest(&self) -> [u8; 32] {
        self.folded_accumulator_digest
    }

    pub fn x_out(&self) -> &Rv64imEncodedPublicInput {
        &self.x_out
    }

    pub fn running_out_state(&self) -> &Rv64imChunkFoldState {
        &self.next_state
    }
}

pub(crate) fn rv64im_main_recursion_x_out(
    vk_fs: &Rv64imVerifierKeyFs,
    chunk_count: u64,
    z_0: [u8; 32],
    z_i: [u8; 32],
    pc_i: u64,
    folded_accumulator_digest: [u8; 32],
) -> Rv64imEncodedPublicInput {
    Rv64imMainRecursionConstruction2StateImage::from_parts(
        vk_fs.clone(),
        chunk_count,
        z_0,
        z_i,
        pc_i,
        folded_accumulator_digest,
    )
    .encoded_public_input()
}

pub(crate) fn build_rv64im_main_recursion_x_i_from_advice(
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Rv64imEncodedPublicInput {
    rv64im_main_recursion_accumulator_from_f_prime_advice(advice).x_i(advice.verifier_key_fs())
}

pub(crate) fn build_rv64im_main_recursion_x_hash_from_advice(
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Rv64imEncodedPublicInput {
    build_rv64im_main_recursion_x_i_from_advice(advice)
}

#[derive(Clone, Debug, Default)]
pub struct Rv64imMainRecursionFPrimeAdviceStepBuildPerf {
    pub build_advice_ms: f64,
    pub evaluate_step_ms: f64,
    pub apply_step_image_ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Rv64imMainRecursionFPrimeAdviceBuildPerf {
    pub verifier_key_ms: f64,
    pub relation_validation_ms: f64,
    pub canonical_full_width_ms: f64,
    pub canonical_u_perp_ms: f64,
    pub total_ms: f64,
    pub step_count: usize,
    pub per_step: Vec<Rv64imMainRecursionFPrimeAdviceStepBuildPerf>,
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

fn rv64im_main_recursion_accumulator_from_f_prime_advice(
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Rv64imMainRecursionAccumulator {
    Rv64imMainRecursionAccumulator {
        chunk_count: advice.chunk_count_in(),
        state: advice.state_in.clone(),
    }
}

pub fn build_rv64im_main_recursion_f_prime_advices(
    relations: &[Rv64imChunkStepIvcRelation],
) -> Result<Vec<Rv64imMainRecursionFPrimeAdvice>, SimpleKernelError> {
    build_rv64im_main_recursion_f_prime_advices_with_phi_side(relations, &Rv64imMainRecursionPhiSide::zero())
}

pub fn build_rv64im_main_recursion_f_prime_advices_with_perf(
    relations: &[Rv64imChunkStepIvcRelation],
) -> Result<
    (
        Vec<Rv64imMainRecursionFPrimeAdvice>,
        Rv64imMainRecursionFPrimeAdviceBuildPerf,
    ),
    SimpleKernelError,
> {
    build_rv64im_main_recursion_f_prime_advices_with_phi_side_and_perf(
        relations,
        &Rv64imMainRecursionPhiSide::zero(),
        None,
    )
}

fn validate_rv64im_main_recursion_single_step_relation(
    relation: &Rv64imChunkStepIvcRelation,
) -> Result<(), SimpleKernelError> {
    if relation.witness.handoff.public_chunk.steps.len() != 1 {
        return Err(SimpleKernelError::Bridge(
            "RV64IM native F' single-step path requires one public step per recursive relation".into(),
        ));
    }
    let fresh = adapt_rv64im_chunk_to_fresh_ccs(&relation.witness.handoff);
    if fresh.fresh_claims.len() != 1 || fresh.fresh_witnesses.len() != 1 {
        return Err(SimpleKernelError::Bridge(
            "RV64IM native F' single-step path requires exactly one fresh CCS instance per recursive relation".into(),
        ));
    }
    Ok(())
}

pub fn build_rv64im_main_recursion_f_prime_advices_single_step(
    relations: &[Rv64imChunkStepIvcRelation],
) -> Result<Vec<Rv64imMainRecursionFPrimeAdvice>, SimpleKernelError> {
    build_rv64im_main_recursion_f_prime_advices_with_phi_side_single_step(
        relations,
        &Rv64imMainRecursionPhiSide::zero(),
    )
}

pub fn build_rv64im_main_recursion_f_prime_advices_single_step_with_perf(
    relations: &[Rv64imChunkStepIvcRelation],
) -> Result<
    (
        Vec<Rv64imMainRecursionFPrimeAdvice>,
        Rv64imMainRecursionFPrimeAdviceBuildPerf,
    ),
    SimpleKernelError,
> {
    build_rv64im_main_recursion_f_prime_advices_with_phi_side_single_step_and_perf(
        relations,
        &Rv64imMainRecursionPhiSide::zero(),
        None,
    )
}

pub fn debug_trace_rv64im_main_recursion_f_prime_advices_single_step_build(
    relations: &[Rv64imChunkStepIvcRelation],
    trace_prefix: &str,
) -> Result<
    (
        Vec<Rv64imMainRecursionFPrimeAdvice>,
        Rv64imMainRecursionFPrimeAdviceBuildPerf,
    ),
    SimpleKernelError,
> {
    build_rv64im_main_recursion_f_prime_advices_with_phi_side_single_step_and_perf(
        relations,
        &Rv64imMainRecursionPhiSide::zero(),
        Some(trace_prefix),
    )
}

pub fn build_rv64im_main_recursion_f_prime_advices_with_side_opening_public(
    relations: &[Rv64imChunkStepIvcRelation],
    side_opening_public: &Rv64imSideOpeningPublic,
) -> Result<Vec<Rv64imMainRecursionFPrimeAdvice>, SimpleKernelError> {
    let (_, phi_side) = build_rv64im_main_recursion_side_lane_from_side_opening_public(side_opening_public)?;
    build_rv64im_main_recursion_f_prime_advices_with_phi_side(relations, &phi_side)
}

pub fn build_rv64im_main_recursion_f_prime_advices_with_side_opening_public_single_step(
    relations: &[Rv64imChunkStepIvcRelation],
    side_opening_public: &Rv64imSideOpeningPublic,
) -> Result<Vec<Rv64imMainRecursionFPrimeAdvice>, SimpleKernelError> {
    let (_, phi_side) = build_rv64im_main_recursion_side_lane_from_side_opening_public(side_opening_public)?;
    build_rv64im_main_recursion_f_prime_advices_with_phi_side_single_step(relations, &phi_side)
}

fn build_rv64im_main_recursion_f_prime_advices_with_phi_side(
    relations: &[Rv64imChunkStepIvcRelation],
    phi_side: &Rv64imMainRecursionPhiSide,
) -> Result<Vec<Rv64imMainRecursionFPrimeAdvice>, SimpleKernelError> {
    Ok(build_rv64im_main_recursion_f_prime_advices_with_phi_side_and_perf(relations, phi_side, None)?.0)
}

fn build_rv64im_main_recursion_f_prime_advices_with_phi_side_and_perf(
    relations: &[Rv64imChunkStepIvcRelation],
    phi_side: &Rv64imMainRecursionPhiSide,
    trace_prefix: Option<&str>,
) -> Result<
    (
        Vec<Rv64imMainRecursionFPrimeAdvice>,
        Rv64imMainRecursionFPrimeAdviceBuildPerf,
    ),
    SimpleKernelError,
> {
    let total_started = Instant::now();
    let started = Instant::now();
    let vk_fs = build_rv64im_main_recursion_verifier_key_fs()?;
    let verifier_key_ms = elapsed_ms(started);
    emit_debug_timing(trace_prefix, "verifier_key", verifier_key_ms);
    let mut accumulator = Rv64imMainRecursionAccumulator::seed();
    let mut current_construction2_u_i: Option<Rv64imMainRecursionConstruction2FreshInstance> = None;
    let mut out = Vec::with_capacity(relations.len());
    let mut perf = Rv64imMainRecursionFPrimeAdviceBuildPerf {
        verifier_key_ms,
        step_count: relations.len(),
        ..Rv64imMainRecursionFPrimeAdviceBuildPerf::default()
    };
    let build_advice = |relation: &Rv64imChunkStepIvcRelation,
                        accumulator: &Rv64imMainRecursionAccumulator,
                        current_construction2_u_i: &Rv64imMainRecursionConstruction2FreshInstance|
     -> Result<Rv64imMainRecursionFPrimeAdvice, SimpleKernelError> {
        let native_verified_step_statement =
            build_rv64im_main_recursion_construction2_verified_step_statement_from_relation(relation)?;
        let main_circuit_chunk_summary = native_verified_step_statement.fixed_shape_chunk_summary()?;
        let main_circuit_chunk_trace = build_rv64im_main_circuit_chunk_trace_from_authoritative_parts(
            relation.witness.handoff.bridge_handoff.chunk_index as usize,
            &relation.witness.handoff,
            &main_circuit_chunk_summary,
            &relation.witness.state_in.carry,
            &relation.witness.state_out.carry,
            &relation.witness.state_in.transcript,
            &relation.witness.state_out.transcript,
            &relation.witness.replay_witness,
        )?;
        let construction2_pi_fold = build_rv64im_main_recursion_construction2_pi_fold_from_relation(relation)?;
        let x_i = accumulator.x_i(&vk_fs);
        Rv64imMainRecursionFPrimeAdvice::from_parts(
            vk_fs.clone(),
            accumulator.chunk_count,
            rv64im_main_recursion_initial_z(),
            accumulator.state.carry.terminal_handle.0,
            RV64IM_MAIN_RECURSION_TRIVIAL_PC,
            Rv64imMainRecursionSideLaneWitness::zero(),
            phi_side.clone(),
            accumulator.state.clone(),
            x_i,
            Some(current_construction2_u_i.clone()),
            native_verified_step_statement,
            relation.witness.terminal_step,
            relation.witness.handoff.clone(),
            relation.witness.state_out.clone(),
            main_circuit_chunk_trace,
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
            crate::rv64im::construction2_default::build_rv64im_main_recursion_construction2_canonical_full_width(
                &vk_fs, phi_side,
            )?;
        perf.canonical_full_width_ms = elapsed_ms(started);
        emit_debug_timing(trace_prefix, "canonical_full_width", perf.canonical_full_width_ms);
        let started = Instant::now();
        let canonical_u_perp =
            build_rv64im_main_recursion_construction2_default_fresh_instance(&vk_fs, canonical_full_width)?;
        perf.canonical_u_perp_ms = elapsed_ms(started);
        emit_debug_timing(trace_prefix, "canonical_u_perp", perf.canonical_u_perp_ms);
        canonical_u_perp
    };
    for (step_index, relation) in relations.iter().enumerate() {
        let mut step_perf = Rv64imMainRecursionFPrimeAdviceStepBuildPerf::default();
        let started = Instant::now();
        let advice = if accumulator.chunk_count == 0 {
            build_advice(relation, &accumulator, &canonical_u_perp)?
        } else {
            let current_construction2_u_i = current_construction2_u_i.as_ref().ok_or_else(|| {
                SimpleKernelError::Bridge(
                    "RV64IM main recursion F' inductive advice builder is missing the prior-step Construction-2 u_i"
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
            evaluate_rv64im_main_recursion_f_prime_advice_with_trace(&advice, step_trace_prefix.as_deref())?;
        step_perf.evaluate_step_ms = elapsed_ms(started);
        emit_debug_timing(
            trace_prefix,
            &format!("step_{step_index}_evaluate_step"),
            step_perf.evaluate_step_ms,
        );
        let started = Instant::now();
        accumulator = accumulator.apply_verified_step_image(&step_image)?;
        step_perf.apply_step_image_ms = elapsed_ms(started);
        emit_debug_timing(
            trace_prefix,
            &format!("step_{step_index}_apply_step_image"),
            step_perf.apply_step_image_ms,
        );
        current_construction2_u_i = Some(step_image.construction2_u_next().clone());
        out.push(advice);
        perf.per_step.push(step_perf);
    }
    perf.total_ms = elapsed_ms(total_started);
    emit_debug_timing(trace_prefix, "total", perf.total_ms);
    Ok((out, perf))
}

fn build_rv64im_main_recursion_f_prime_advices_with_phi_side_single_step(
    relations: &[Rv64imChunkStepIvcRelation],
    phi_side: &Rv64imMainRecursionPhiSide,
) -> Result<Vec<Rv64imMainRecursionFPrimeAdvice>, SimpleKernelError> {
    Ok(build_rv64im_main_recursion_f_prime_advices_with_phi_side_single_step_and_perf(relations, phi_side, None)?.0)
}

fn build_rv64im_main_recursion_f_prime_advices_with_phi_side_single_step_and_perf(
    relations: &[Rv64imChunkStepIvcRelation],
    phi_side: &Rv64imMainRecursionPhiSide,
    trace_prefix: Option<&str>,
) -> Result<
    (
        Vec<Rv64imMainRecursionFPrimeAdvice>,
        Rv64imMainRecursionFPrimeAdviceBuildPerf,
    ),
    SimpleKernelError,
> {
    let started = Instant::now();
    for relation in relations {
        validate_rv64im_main_recursion_single_step_relation(relation)?;
    }
    let mut built =
        build_rv64im_main_recursion_f_prime_advices_with_phi_side_and_perf(relations, phi_side, trace_prefix)?;
    built.1.relation_validation_ms = elapsed_ms(started);
    emit_debug_timing(trace_prefix, "relation_validation", built.1.relation_validation_ms);
    Ok(built)
}

pub fn build_rv64im_main_recursion_f_prime_public_output(
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Result<Rv64imMainRecursionFPrimePublicOutput, SimpleKernelError> {
    let output = CanonicalRv64imMainRecursionFPrimeBody.step(&Rv64imMainRecursionFPrimeInput, advice)?;
    Ok(Rv64imMainRecursionFPrimePublicOutput {
        x_out: output.x_out().clone(),
    })
}

pub fn evaluate_rv64im_main_recursion_f_prime_advice(
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Result<Rv64imMainRecursionFPrimeStepImage, SimpleKernelError> {
    evaluate_rv64im_main_recursion_f_prime_advice_with_trace(advice, None)
}

fn evaluate_rv64im_main_recursion_f_prime_advice_with_trace(
    advice: &Rv64imMainRecursionFPrimeAdvice,
    trace_prefix: Option<&str>,
) -> Result<Rv64imMainRecursionFPrimeStepImage, SimpleKernelError> {
    evaluate_rv64im_main_recursion_f_prime_step_with_trace(&Rv64imMainRecursionFPrimeInput, advice, trace_prefix)
}

fn evaluate_rv64im_main_recursion_f_prime_step(
    _input: &Rv64imMainRecursionFPrimeInput,
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Result<Rv64imMainRecursionFPrimeStepImage, SimpleKernelError> {
    evaluate_rv64im_main_recursion_f_prime_step_with_trace(_input, advice, None)
}

fn evaluate_rv64im_main_recursion_f_prime_step_with_trace(
    _input: &Rv64imMainRecursionFPrimeInput,
    advice: &Rv64imMainRecursionFPrimeAdvice,
    trace_prefix: Option<&str>,
) -> Result<Rv64imMainRecursionFPrimeStepImage, SimpleKernelError> {
    let total_started = Instant::now();
    let started = Instant::now();
    let accumulator_in = rv64im_main_recursion_accumulator_from_f_prime_advice(advice);
    let expected_vk_fs = build_rv64im_main_recursion_verifier_key_fs()?;
    if advice.verifier_key_fs() != &expected_vk_fs {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion F' advice vk_fs does not match the canonical deployed verifier-key context".into(),
        ));
    }
    if advice.chunk_index() != advice.chunk_count_in() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion F' advice chunk_index does not match chunk_count_in".into(),
        ));
    }
    if advice.z_0() != &rv64im_main_recursion_initial_z() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion F' advice z_0 does not match the canonical initial recursion state".into(),
        ));
    }
    if advice.z_i() != &advice.state_in.carry.terminal_handle.0 {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion F' advice z_i does not match the carried recursive state handle".into(),
        ));
    }
    if advice.pc_i() != RV64IM_MAIN_RECURSION_TRIVIAL_PC {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion F' advice pc_i does not match the trivial RV64IM recursion control lane".into(),
        ));
    }
    if !RV64IM_MAIN_RECURSION_SIDE_WITNESS_ACTIVE && !advice.side_witness().is_zero() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion F' advice side_witness is non-zero before phi_side is wired".into(),
        ));
    }
    let Some(construction2_u_i) = advice.construction2_input_fresh_instance() else {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion F' advice is missing the threaded HyperNova Construction-2 fresh input u_i".into(),
        ));
    };
    emit_debug_timing(trace_prefix, "prechecks", elapsed_ms(started));
    if advice.chunk_count_in() == 0 {
        let started = Instant::now();
        validate_rv64im_main_recursion_base_case_accumulator(&accumulator_in, advice, construction2_u_i)?;
        emit_debug_timing(trace_prefix, "base_case_validation", elapsed_ms(started));
    }
    let started = Instant::now();
    let expected_x_i = accumulator_in.x_i(advice.verifier_key_fs());
    if advice.x_i() != &expected_x_i {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion F' advice x_i does not match the carried recursive accumulator image".into(),
        ));
    }
    emit_debug_timing(trace_prefix, "x_i_check", elapsed_ms(started));
    let bridge_trace_prefix = trace_prefix.map(|prefix| format!("{prefix}.nifs_bridge"));
    let started = Instant::now();
    let construction2_nifs_bridge = build_rv64im_main_recursion_construction2_nifs_bridge_with_trace(
        advice,
        construction2_u_i,
        bridge_trace_prefix.as_deref(),
    )?;
    emit_debug_timing(trace_prefix, "build_nifs_bridge", elapsed_ms(started));
    let verify_trace_prefix = trace_prefix.map(|prefix| format!("{prefix}.nifs_verify"));
    let started = Instant::now();
    let verified_step = verify_rv64im_main_recursion_construction2_nifs_step_with_trace(
        &construction2_nifs_bridge,
        verify_trace_prefix.as_deref(),
    )?;
    emit_debug_timing(trace_prefix, "verify_nifs_step", elapsed_ms(started));
    let next_state = verified_step.state;
    let started = Instant::now();
    let _ = Rv64imMainRecursionAccumulatorSurface::try_from_carry(&next_state.carry.main, "F' next-state accumulator")?;
    emit_debug_timing(trace_prefix, "next_state_surface_check", elapsed_ms(started));
    let accumulator_out = Rv64imMainRecursionAccumulator {
        chunk_count: accumulator_in.chunk_count + 1,
        state: next_state.clone(),
    };
    let started = Instant::now();
    let folded_accumulator_digest = rv64im_chunk_fold_carry_recursive_accumulator_digest(&accumulator_out.state.carry);
    let z_next = accumulator_out.state.carry.terminal_handle.0;
    let pc_next = RV64IM_MAIN_RECURSION_TRIVIAL_PC;
    let phi_side = advice.phi_side().clone();
    let x_out = rv64im_main_recursion_x_out(
        advice.verifier_key_fs(),
        accumulator_out.chunk_count,
        *advice.z_0(),
        z_next,
        pc_next,
        folded_accumulator_digest,
    );
    emit_debug_timing(trace_prefix, "derive_public_outputs", elapsed_ms(started));
    let started = Instant::now();
    let u_next_trace_prefix = trace_prefix.map(|prefix| format!("{prefix}.construction2_u_next"));
    let construction2_u_next = if let Some(prefix) = u_next_trace_prefix.as_deref() {
        debug_trace_build_rv64im_main_recursion_construction2_fresh_instance_with_input_and_x_i(
            advice,
            construction2_u_i,
            x_out.clone(),
            prefix,
        )?
    } else {
        build_rv64im_main_recursion_construction2_fresh_instance_with_input_and_x_i(
            advice,
            construction2_u_i,
            x_out.clone(),
        )?
    };
    emit_debug_timing(trace_prefix, "build_construction2_u_next", elapsed_ms(started));
    if construction2_u_next.x_i() != &x_out {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion F' produced a Construction-2 output u_{i+1} whose x_i does not match x_{i+1}".into(),
        ));
    }
    emit_debug_timing(trace_prefix, "total", elapsed_ms(total_started));
    Ok(Rv64imMainRecursionFPrimeStepImage {
        chunk_count: accumulator_out.chunk_count,
        z_next,
        pc_next,
        phi_side,
        construction2_u_next,
        next_state,
        folded_accumulator_digest,
        x_out,
    })
}

pub fn verify_rv64im_main_recursion_f_prime_public_output(
    public_output: &Rv64imMainRecursionFPrimePublicOutput,
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Result<Rv64imMainRecursionFPrimeStepImage, SimpleKernelError> {
    let output = CanonicalRv64imMainRecursionFPrimeBody.step(&Rv64imMainRecursionFPrimeInput, advice)?;
    if public_output.x_out != *output.x_out() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion F' public output x_out does not match the verified recursive step".into(),
        ));
    }
    Ok(output)
}
