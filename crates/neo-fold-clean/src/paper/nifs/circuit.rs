//! NIFS.V composition in-circuit — Π_CCS.V → Π_RLC.V → Π_DEC.V.
//!
//! Production NIFS.V verifier: accepts proofs produced by the native
//! [`crate::paper::nifs::prove`] path bit-for-bit (transcript-state parity)
//! and identity-for-identity (algebraic-check parity). F' R1CS embeds this
//! gadget verbatim in its recursive step.
//!
//! ## Composition
//!
//! ```text
//! 1. enforce_split_nc_pi_ccs_v(transcript, fresh, running, proof.pi_ccs)
//!      → derived.{r_prime, s_col_prime, outputs, fresh_x, running_c_data}
//!    (Internally: header_digest catch-up squeeze, so transcript now
//!     matches what native engine::optimized::verify_pi_ccs leaves.)
//!
//! 2. Absorb the full Π_CCS output claims, then sample
//!    ρ ← enforce_pi_rlc_rhos_from_transcript(transcript, k_total).
//!    This mirrors native `pi_rlc::verify`: Π_CCS output messages must be
//!    Fiat-Shamir input before Π_RLC derives its random coefficients.
//!
//! 3. Allocate parent + children CE wires (msg.combined, msg.children).
//!    Pin the Π_RLC parent shape (`c.kappa`, `m_in`) to the Π_CCS outputs
//!    before folding; then enforce_split_nc_d_pad_shape verifies
//!    y_ring[j]/y_zcol lane counts.
//!
//! 4. Π_RLC.V folds:
//!      - commitment:  parent.c    = Σ_i ρ_i · output_i.c
//!      - X:           parent.X    = Σ_i ρ_i · output_i.X
//!      - per j:       parent.y_ring[j] = Σ_i ρ_i · output_i.y_ring[j]
//!                     (padded helper: rotation on first D, zero on tail)
//!      - y_zcol:      parent.y_zcol = Σ_i ρ_i · output_i.y_zcol
//!                     (same padded shape)
//!      - s_col:       output_i.s_col == parent.s_col (consistency)
//!      - fold_digest: parent.fold_digest == output_i.fold_digest for every i
//!
//! 5. Π_DEC.V strict: b-ary recomposition for c/X/y_ring plus r/s_col
//!    consistency, ct consistency, and fold_digest consistency
//!    parent↔child. No y_zcol recomposition and no unsigned X bitness
//!    check; this mirrors native `verify_dec_public`.
//!
//! 6. Point binding: dec_wires.parent.r == ccs.r_prime.
//! ```
//!
//! All wires consumed downstream by F' R1CS (parent commitment, fresh X,
//! running commitment data) are surfaced via [`NifsVOutputs`].

use neo_ccs::LaneCommitments;
use neo_math::ring::D;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::alphabet_sampling::enforce_pi_rlc_rhos_from_transcript;
use crate::engine::r1cs_circuit::builder::{
    Lc, ProjectionGlueRole, ProjectionIdentityRole, ProjectionNebulaCoordinate, Var,
};
use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::ring_action::{
    enforce_beta_ladder, enforce_polynomial_evaluations_at_beta, PROJECTION_QUOTIENT_LEN,
};
use crate::engine::r1cs_circuit::transcript::TranscriptGadget;
use crate::engine::r1cs_circuit::R1csBuilder;
use crate::paper::f_prime::nebula_lane_circuit::enforce_nebula_lane_leaf_digests_circuit;
use crate::paper::params::Params;
use crate::paper::reductions::accumulator_sis_circuit::{
    enforce_accumulator_digest as enforce_sis_accumulator_digest, PI_RLC_PROJECTION_SIS_CONFIG,
};
use crate::paper::reductions::pi_ccs_split_nc_circuit::{
    enforce_split_nc_pi_ccs_v, enforce_split_nc_pi_ccs_v_with_header_bundle_wires, SplitNcPiCcsOutputWires,
    SplitNcPiCcsVConfig, SplitNcPiCcsVMessages,
};
use crate::paper::reductions::pi_dec_circuit::{
    self, alloc_dec_inputs, enforce_dec_v_strict, enforce_split_nc_d_pad_shape, DecInputWires,
};
use crate::paper::reductions::pi_rlc_circuit::{
    alloc_rlc_projection_quotient_advice as alloc_vector_projection_quotient_advice,
    enforce_rlc_commitment_combination_projection_with_quotient_wires,
    enforce_rlc_padded_k_vector_combination_projection_with_quotient_wires, enforce_rlc_s_col_consistency,
    enforce_rlc_x_combination_projection_with_quotient_wires, rlc_projection_quotients, RlcCommitmentWires,
    RlcPaddedKVectorPairWires, RlcPaddedKVectorWires, RlcPairWires, RlcXPairWires, RlcXWires,
};
use crate::paper::reductions::{pi_ccs, pi_ccs_split_nc_circuit, pi_rlc};
use crate::paper::relations::product_commitment_circuit::{validate_adv_shape, AdvCommitmentWires, CommitmentWires};
use crate::paper::relations::{superneo_public_x_cols, CcsClaim, CeClaim};

/// Configuration for one NIFS.V step.
pub struct NifsVCircuitConfig<'a> {
    pub pi_ccs: SplitNcPiCcsVConfig<'a>,
}

/// Witness/protocol messages from a real native `nifs::prove` proof.
///
/// `pi_ccs` carries the Π_CCS sumcheck transcript + outputs; `combined`
/// is the Π_RLC parent CE claim; `children` is the Π_DEC k-fold output
/// (the new running accumulator).
pub struct NifsVCircuitMessages<'a> {
    pub fresh: &'a [CcsClaim],
    pub running: &'a [CeClaim],
    pub running_parent_authority: Option<&'a CeClaim>,
    pub pi_ccs: &'a pi_ccs::Proof,
    pub combined: &'a CeClaim,
    pub children: &'a [CeClaim],
}

/// Output wires from one NIFS.V step that F' R1CS composition consumes.
pub struct NifsVOutputs {
    /// `Σ b^i · child_i.c.data` — the Π_DEC parent commitment's data wires.
    /// Kept for callers that need the commitment projection; F' binds the
    /// outgoing accumulator via the full `(children, parent)` CE-claim
    /// digest below, not by hashing this commitment projection alone.
    pub parent_c_data: Vec<Var>,
    /// Fresh CCS instances' public-input wires `[fresh_idx][x_lane]`.
    /// F' R1CS uses these to enforce the HyperNova recursive link.
    pub fresh_x: Vec<Vec<Var>>,
    /// Product-commitment coordinates paired with [`Self::fresh_x`].
    pub fresh_adv: Vec<Option<AdvCommitmentWires>>,
    /// Per-running-claim commitment data wires `[running_idx][lane]`.
    /// Kept for callers that need the per-claim view; F' R1CS binds the
    /// running accumulator via the digest below, not via re-hashing
    /// these wires.
    pub running_c_data: Vec<Vec<Var>>,
    /// Four-lane Poseidon2 handle of the strict Π_DEC parent authority.
    /// The SplitNc/NIFS composition checks every running child against that
    /// parent before deriving this value. F' reuses it for `acc_digest_in`,
    /// so the state link and Π_CCS transcript share one authority boundary.
    pub running_acc_digest: [Var; 4],
    /// Per-running-claim CE-claim wire bundles — the running input to
    /// this NIFS.V step. The decider's CE-continuity gate compares the
    /// *previous* step's [`Self::children`] against this step's
    /// `running` to enforce step-to-step accumulator equality without
    /// relying on digest equality alone.
    pub running: Vec<SplitNcPiCcsOutputWires>,
    /// Π_RLC parent authority for [`Self::running`], if running is non-empty.
    pub running_parent_authority: Option<SplitNcPiCcsOutputWires>,
    /// Current step's Π_RLC parent, checked by Π_DEC against
    /// [`Self::children`] and carried as the next step's parent authority.
    pub parent: pi_dec_circuit::CeClaimWires,
    /// Per-child CE-claim wires emitted by Π_DEC — the claims that
    /// become the *next* step's `running`. The decider's
    /// CE-continuity gate pins `prev.children == next.running` element
    /// by element.
    pub children: Vec<pi_dec_circuit::CeClaimWires>,
    /// Π_RLC projection β (two K limbs), squeezed by the Lemma 5
    /// schedule replay. The commitment projection already consumes this
    /// wire; it remains surfaced for audit and eventual low-norm lowering.
    pub projection_beta: [Var; 2],
    /// Per-κ-lane projection quotient advice absorbed before β and consumed
    /// by the commitment identities. Surfaced for audit/low-norm lowering.
    pub projection_q_lanes: Vec<[Var; PROJECTION_QUOTIENT_LEN]>,
    /// Projection quotient advice for the `(ops, is, fs)` coordinates of
    /// the product commitment. Present exactly for Nebula folds.
    pub projection_adv_q_lanes: Option<LaneCommitments<Vec<[Var; PROJECTION_QUOTIENT_LEN]>>>,
    /// One transcript-bound quotient per active X ring column.
    pub projection_x_q_lanes: Vec<[Var; PROJECTION_QUOTIENT_LEN]>,
    /// Two transcript-bound quotients per y_ring row (c0, c1).
    pub projection_y_ring_q_lanes: Vec<[[Var; PROJECTION_QUOTIENT_LEN]; 2]>,
    /// Two transcript-bound quotients for y_zcol (c0, c1).
    pub projection_y_zcol_q_lanes: [[Var; PROJECTION_QUOTIENT_LEN]; 2],
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("NIFS.V circuit: {0}")]
    Inner(String),
    #[error(transparent)]
    PiCcs(#[from] pi_ccs_split_nc_circuit::Error),
    #[error(transparent)]
    PiDec(#[from] crate::paper::reductions::pi_dec_circuit::Error),
    #[error(transparent)]
    PiRlc(#[from] crate::paper::reductions::pi_rlc_circuit::Error),
}

/// Enforce NIFS.V on top of `transcript`. See module docstring for the
/// composition order.
pub fn enforce_nifs_v_circuit_with_transcript(
    builder: &mut R1csBuilder,
    pp: &Params,
    cfg: &NifsVCircuitConfig<'_>,
    transcript: &mut TranscriptGadget,
    msg: &NifsVCircuitMessages<'_>,
) -> Result<NifsVOutputs, Error> {
    enforce_nifs_v_circuit_with_transcript_inner(builder, pp, cfg, transcript, msg, None)
}

/// Folded-F' entrypoint using the verifier header carried by F' state.
pub fn enforce_nifs_v_circuit_with_transcript_and_header_bundle(
    builder: &mut R1csBuilder,
    pp: &Params,
    cfg: &NifsVCircuitConfig<'_>,
    transcript: &mut TranscriptGadget,
    msg: &NifsVCircuitMessages<'_>,
    header_bundle: [Var; 4],
) -> Result<NifsVOutputs, Error> {
    enforce_nifs_v_circuit_with_transcript_inner(builder, pp, cfg, transcript, msg, Some(header_bundle))
}

/// Fixed-relation alias with the header-first argument order used by the
/// backend-oriented F' builder.
pub fn enforce_nifs_v_circuit_with_transcript_and_header_bundle_wires(
    builder: &mut R1csBuilder,
    pp: &Params,
    cfg: &NifsVCircuitConfig<'_>,
    transcript: &mut TranscriptGadget,
    header_bundle: [Var; 4],
    msg: &NifsVCircuitMessages<'_>,
) -> Result<NifsVOutputs, Error> {
    enforce_nifs_v_circuit_with_transcript_inner(builder, pp, cfg, transcript, msg, Some(header_bundle))
}

fn enforce_nifs_v_circuit_with_transcript_inner(
    builder: &mut R1csBuilder,
    pp: &Params,
    cfg: &NifsVCircuitConfig<'_>,
    transcript: &mut TranscriptGadget,
    msg: &NifsVCircuitMessages<'_>,
    header_bundle: Option<[Var; 4]>,
) -> Result<NifsVOutputs, Error> {
    let nifs_start = builder.rows();
    let pi_ccs_start = builder.rows();
    let pi_ccs_first_column = builder.cols();
    // ── 1. Π_CCS.V SplitNc verifier ───────────────────────────────────────
    let pi_ccs_messages = SplitNcPiCcsVMessages {
        fresh: msg.fresh,
        running: msg.running,
        running_parent_authority: msg.running_parent_authority,
        outputs: &msg.pi_ccs.outputs,
        outputs_digest: msg.pi_ccs.outputs_digest,
        sc_initial_sum: msg.pi_ccs.sumcheck.sc_initial_sum,
        sumcheck_rounds_fe: &msg.pi_ccs.sumcheck.sumcheck_rounds,
        sumcheck_rounds_nc: &msg.pi_ccs.sumcheck.sumcheck_rounds_nc,
        header_digest: &msg.pi_ccs.sumcheck.header_digest,
    };
    let ccs = match header_bundle {
        Some(header_bundle) => enforce_split_nc_pi_ccs_v_with_header_bundle_wires(
            builder,
            transcript,
            &cfg.pi_ccs,
            &pi_ccs_messages,
            header_bundle,
        )?,
        None => enforce_split_nc_pi_ccs_v(builder, transcript, &cfg.pi_ccs, &pi_ccs_messages)?,
    };
    builder.record_row_family("nifs.pi_ccs", pi_ccs_start);
    builder.record_program_range("nifs.pi_ccs", pi_ccs_start, pi_ccs_first_column);

    let pi_rlc_start = builder.rows();
    let pi_rlc_first_column = builder.cols();
    let d_pad = 1usize << cfg.pi_ccs.ell_d;
    let k_total = ccs.outputs.len();
    if k_total == 0 {
        return Err(Error::Inner("Π_CCS.V emitted zero outputs".into()));
    }
    let kappa = ccs.outputs[0].c_kappa;
    let m_in = ccs.outputs[0].x_cols;

    // Native parity: Π_RLC's Definition-14 bound `count·T·(b−1) < B` is a
    // structural constraint on `count = K + k` (fixed at gadget-emit
    // time). Native `pi_rlc::enforce_rlc_bound` rejects RLC folds that
    // violate it; the in-circuit path must match (cheap, no constraints —
    // pure native validation, same call as the native verifier).
    crate::paper::sampling::check_rlc_bound(pp, k_total, pp.T() as u128)
        .map_err(|e| Error::Inner(format!("Π_RLC bound: {e}")))?;

    // ── 2. Bind Π_CCS output messages, then sample Π_RLC ρ ────────────────
    let pi_rlc_transcript_start = builder.rows();
    builder.begin_encoding_stage("nifs.pi_rlc.challenge");
    transcript.append_fields(
        builder,
        pi_rlc::PI_RLC_INPUT_CLAIMS_DIGEST_LABEL,
        &ccs.output_claims_digest,
    );

    let rho_wires = enforce_pi_rlc_rhos_from_transcript(builder, transcript, k_total);
    builder.record_row_family("nifs.pi_rlc.transcript_rhos", pi_rlc_transcript_start);

    // ── 3. Parent + children DEC wires + SplitNc shape check ──────────────
    let pi_rlc_shape_start = builder.rows();
    builder.begin_encoding_stage("nifs.pi_dec.allocate");
    let dec_wires = alloc_dec_inputs(builder, msg.combined, msg.children);
    enforce_rlc_output_shape_parity(builder, &ccs.outputs)?;
    enforce_rlc_parent_shape(builder, &dec_wires, &ccs.outputs, kappa, m_in)?;
    enforce_split_nc_d_pad_shape(&dec_wires, cfg.pi_ccs.structure.t(), d_pad)?;
    builder.record_row_family("nifs.pi_rlc.shape", pi_rlc_shape_start);

    // ── 4. Π_RLC.V folds: c, X, per-j y_ring, y_zcol, s_col ──────────────
    let pi_rlc_folds_start = builder.rows();
    builder.begin_encoding_stage("nifs.pi_rlc.verify");
    let commitment_wires = rlc_commitment_fold_wires(&rho_wires, &ccs.outputs, &dec_wires, kappa)?;
    let adv_commitment_wires = rlc_adv_commitment_fold_wires(&rho_wires, &ccs.outputs, &dec_wires)?;
    let x_wires = rlc_x_fold_wires(&rho_wires, &ccs.outputs, &dec_wires, m_in)?;
    let t = cfg.pi_ccs.structure.t();
    let mut y_ring_wires = Vec::with_capacity(t);
    for j in 0..t {
        y_ring_wires.push(rlc_y_ring_row_fold_wires(
            &rho_wires,
            &ccs.outputs,
            &dec_wires,
            j,
            d_pad,
        )?);
    }
    let y_zcol_wires = rlc_y_zcol_fold_wires(&rho_wires, &ccs.outputs, &dec_wires, d_pad)?;
    let input_s_cols: Vec<Vec<KVar>> = ccs.outputs.iter().map(|o| o.s_col.clone()).collect();
    enforce_rlc_s_col_consistency(builder, &input_s_cols, &dec_wires.parent.s_col)?;
    enforce_rlc_fold_digest_consistency(builder, &ccs.outputs, &dec_wires)?;
    builder.record_row_family("nifs.pi_rlc.linear_folds", pi_rlc_folds_start);

    // ── 4b. Π_RLC β schedule replay (Lemma 5; Road A) ────────────────────
    // Native `pi_rlc` absorbs the combined commitment and the per-lane
    // division quotients, then squeezes β — mirror it bit-for-bit so
    // every later challenge stays in lockstep. q enters as advice
    // recomputed from the same wires native uses. The exact q wires absorbed
    // here are consumed by the projection identity below; no duplicate advice
    // can drift away from the Fiat-Shamir schedule.
    let projection_binding_start = builder.rows();
    let mut projection_binding = alloc_projection_binding_preimage(builder);
    append_projection_binding_wires(
        builder,
        &mut projection_binding,
        pi_rlc::PI_RLC_PROJECTION_COMBINED_C_LABEL,
        &dec_wires.parent.c_data[..D * kappa],
    );
    let projection_q_lanes = alloc_projection_quotient_advice(builder, &commitment_wires)?;
    for q_lane in &projection_q_lanes {
        append_projection_binding_wires(
            builder,
            &mut projection_binding,
            pi_rlc::PI_RLC_PROJECTION_QUOTIENTS_LABEL,
            q_lane,
        );
    }
    let projection_adv_q_lanes = if let Some(adv) = &adv_commitment_wires {
        let combined = dec_wires
            .parent
            .adv
            .as_ref()
            .ok_or_else(|| Error::Inner("Pi_RLC adv projection has inputs but no combined coordinate".into()))?;
        for leaf in enforce_nebula_lane_leaf_digests_circuit(
            builder,
            combined.ops.d,
            combined.ops.kappa,
            &combined.ops.data,
            &combined.is.data,
            &combined.fs.data,
        ) {
            append_projection_binding_wires(
                builder,
                &mut projection_binding,
                pi_rlc::PI_RLC_PROJECTION_COMBINED_ADV_LABEL,
                &leaf,
            );
        }
        let ops = alloc_projection_quotient_advice(builder, &adv.ops)?;
        let is = alloc_projection_quotient_advice(builder, &adv.is)?;
        let fs = alloc_projection_quotient_advice(builder, &adv.fs)?;
        for q_lane in ops.iter().chain(is.iter()).chain(fs.iter()) {
            append_projection_binding_wires(
                builder,
                &mut projection_binding,
                pi_rlc::PI_RLC_PROJECTION_ADV_QUOTIENTS_LABEL,
                q_lane,
            );
        }
        Some(LaneCommitments { ops, is, fs })
    } else {
        None
    };
    let projection_x_q_lanes = alloc_x_projection_advice(builder, &mut projection_binding, &x_wires)?;
    let mut projection_y_ring_q_lanes = Vec::with_capacity(y_ring_wires.len());
    for wires in &y_ring_wires {
        projection_y_ring_q_lanes.push(alloc_padded_projection_advice(
            builder,
            &mut projection_binding,
            wires,
            pi_rlc::PI_RLC_PROJECTION_COMBINED_Y_RING_LABEL,
            pi_rlc::PI_RLC_PROJECTION_Y_RING_QUOTIENTS_LABEL,
        )?);
    }
    let projection_y_zcol_q_lanes = alloc_padded_projection_advice(
        builder,
        &mut projection_binding,
        &y_zcol_wires,
        pi_rlc::PI_RLC_PROJECTION_COMBINED_Y_ZCOL_LABEL,
        pi_rlc::PI_RLC_PROJECTION_Y_ZCOL_QUOTIENTS_LABEL,
    )?;
    let projection_binding_digest =
        enforce_sis_accumulator_digest(builder, PI_RLC_PROJECTION_SIS_CONFIG, &projection_binding)
            .map_err(|error| Error::Inner(format!("Pi_RLC projection SIS binding: {error}")))?
            .digest;
    transcript.append_fields(
        builder,
        pi_rlc::PI_RLC_PROJECTION_BINDING_DIGEST_LABEL,
        &projection_binding_digest,
    );
    let beta = transcript.challenge_fields(builder, pi_rlc::PI_RLC_PROJECTION_BETA_LABEL, 2);
    let projection_beta = [beta[0], beta[1]];
    builder.record_row_family("nifs.pi_rlc.projection_binding", projection_binding_start);
    let projection_shared_start = builder.rows();
    let powers = enforce_beta_ladder(builder, KVar::new(beta[0], beta[1]), D);
    let rho_evaluations = enforce_polynomial_evaluations_at_beta(builder, &rho_wires, &powers);
    builder.record_row_family("nifs.pi_rlc.projection_shared", projection_shared_start);
    let projection_identities_start = builder.rows();
    let projection_identity_audit_start = builder.projection_identity_audits().len();
    enforce_rlc_commitment_combination_projection_with_quotient_wires(
        builder,
        &powers,
        &rho_evaluations,
        &commitment_wires,
        &projection_q_lanes,
    )?;
    if let (Some(adv), Some(q)) = (&adv_commitment_wires, &projection_adv_q_lanes) {
        for (coordinate, quotients) in [(&adv.ops, &q.ops), (&adv.is, &q.is), (&adv.fs, &q.fs)] {
            enforce_rlc_commitment_combination_projection_with_quotient_wires(
                builder,
                &powers,
                &rho_evaluations,
                coordinate,
                quotients,
            )?;
        }
    }
    let x_projection_start = builder.rows();
    enforce_rlc_x_combination_projection_with_quotient_wires(
        builder,
        &powers,
        &rho_evaluations,
        &x_wires,
        &projection_x_q_lanes,
    )?;
    let x_glue_start = projection_glue_start(builder, x_projection_start);
    builder.record_projection_glue(ProjectionGlueRole::InactiveXZero, x_glue_start);
    for (row, (wires, quotients)) in y_ring_wires
        .iter()
        .zip(projection_y_ring_q_lanes.iter())
        .enumerate()
    {
        let row_projection_start = builder.rows();
        enforce_rlc_padded_k_vector_combination_projection_with_quotient_wires(
            builder,
            &powers,
            &rho_evaluations,
            wires,
            &quotients[0],
            &quotients[1],
        )?;
        let row_glue_start = projection_glue_start(builder, row_projection_start);
        builder.record_projection_glue(ProjectionGlueRole::YRingPaddingZero { row }, row_glue_start);
    }
    let y_zcol_projection_start = builder.rows();
    enforce_rlc_padded_k_vector_combination_projection_with_quotient_wires(
        builder,
        &powers,
        &rho_evaluations,
        &y_zcol_wires,
        &projection_y_zcol_q_lanes[0],
        &projection_y_zcol_q_lanes[1],
    )?;
    let y_zcol_glue_start = projection_glue_start(builder, y_zcol_projection_start);
    builder.record_projection_glue(ProjectionGlueRole::YZColPaddingZero, y_zcol_glue_start);
    let mut projection_roles = Vec::with_capacity(builder.projection_identity_audits().len());
    projection_roles.extend((0..kappa).map(|lane| ProjectionIdentityRole::CommitmentLane { lane }));
    if projection_adv_q_lanes.is_some() {
        for coordinate in [
            ProjectionNebulaCoordinate::Ops,
            ProjectionNebulaCoordinate::Is,
            ProjectionNebulaCoordinate::Fs,
        ] {
            projection_roles
                .extend((0..kappa).map(|lane| ProjectionIdentityRole::NebulaCommitmentLane { coordinate, lane }));
        }
    }
    projection_roles
        .extend((0..superneo_public_x_cols(m_in)).map(|column| ProjectionIdentityRole::ActiveXColumn { column }));
    for row in 0..t {
        projection_roles.extend((0..2).map(|limb| ProjectionIdentityRole::YRingLimb { row, limb }));
    }
    projection_roles.extend((0..2).map(|limb| ProjectionIdentityRole::YZColLimb { limb }));
    builder.assign_projection_identity_roles(projection_identity_audit_start, &projection_roles);
    builder.record_row_family("nifs.pi_rlc.projection_identities", projection_identities_start);

    // ── 5. Π_DEC.V strict ────────────────────────────────────────────────
    builder.record_row_family("nifs.pi_rlc", pi_rlc_start);
    builder.record_program_range("nifs.pi_rlc", pi_rlc_start, pi_rlc_first_column);

    let pi_dec_start = builder.rows();
    let pi_dec_first_column = builder.cols();
    builder.begin_encoding_stage("nifs.pi_dec.verify");
    enforce_dec_v_strict(builder, pp, &dec_wires)?;
    builder.record_row_family("nifs.pi_dec", pi_dec_start);
    builder.record_program_range("nifs.pi_dec", pi_dec_start, pi_dec_first_column);

    // ── 6. Point binding: dec parent.r == ccs r_prime ────────────────────
    let point_binding_start = builder.rows();
    let point_binding_first_column = builder.cols();
    enforce_kvar_vec_eq(builder, &dec_wires.parent.r, &ccs.r_prime)?;
    builder.record_row_family("nifs.point_binding", point_binding_start);
    builder.record_program_range("nifs.point_binding", point_binding_start, point_binding_first_column);
    builder.record_row_family("nifs.total", nifs_start);

    let running = ccs.running.clone();
    let running_parent_authority = ccs.running_parent_authority.clone();
    let parent = dec_wires.parent.clone();
    let children = dec_wires.children.clone();
    Ok(NifsVOutputs {
        parent_c_data: dec_wires.parent.c_data.clone(),
        fresh_x: ccs.fresh_x,
        fresh_adv: ccs.fresh_adv,
        running_c_data: ccs.running_c_data,
        running_acc_digest: ccs.running_acc_digest,
        running,
        running_parent_authority,
        parent,
        children,
        projection_beta,
        projection_q_lanes,
        projection_adv_q_lanes,
        projection_x_q_lanes,
        projection_y_ring_q_lanes,
        projection_y_zcol_q_lanes,
    })
}

fn projection_glue_start(builder: &R1csBuilder, projection_start: usize) -> usize {
    builder
        .projection_identity_audits()
        .last()
        .filter(|identity| identity.row_start >= projection_start)
        .map_or(builder.rows(), |identity| identity.row_end)
}

/// Native recomputation of the Π_RLC projection quotients as advice
/// wires (Lemma 5 schedule replay). The values mirror what native
/// `pi_rlc::projection_schedule` absorbs — recomputed from the ρ and
/// input-commitment wires' values, never read from the proof.
fn alloc_projection_quotient_advice(
    builder: &mut R1csBuilder,
    wires: &RlcCommitmentWires,
) -> Result<Vec<[Var; PROJECTION_QUOTIENT_LEN]>, Error> {
    let rho_vals: Vec<[F; D]> = wires
        .inputs
        .iter()
        .map(|pair| core::array::from_fn(|i| builder.witness()[pair.rho_coeffs[i].col()]))
        .collect();
    let input_cs: Vec<neo_ajtai::Commitment> = wires
        .inputs
        .iter()
        .map(|pair| neo_ajtai::Commitment {
            d: D,
            kappa: pair.kappa,
            data: pair
                .c_data
                .iter()
                .map(|v| builder.witness()[v.col()])
                .collect(),
        })
        .collect();
    let lanes = rlc_projection_quotients(&rho_vals, &input_cs)?;
    Ok(lanes
        .iter()
        .map(|lane| core::array::from_fn(|i| builder.alloc(lane.q[i])))
        .collect())
}

fn alloc_x_projection_advice(
    builder: &mut R1csBuilder,
    projection_binding: &mut Vec<Var>,
    wires: &RlcXWires,
) -> Result<Vec<[Var; PROJECTION_QUOTIENT_LEN]>, Error> {
    let active_cols = superneo_public_x_cols(wires.m_in);
    let rho_wires: Vec<[Var; D]> = wires.inputs.iter().map(|pair| pair.rho_coeffs).collect();
    let mut quotients = Vec::with_capacity(active_cols);
    for col in 0..active_cols {
        let inputs: Vec<[Var; D]> = wires
            .inputs
            .iter()
            .map(|pair| core::array::from_fn(|row| pair.x_flat[row * wires.m_in + col]))
            .collect();
        let output: [Var; D] = core::array::from_fn(|row| wires.combined_x_flat[row * wires.m_in + col]);
        append_projection_binding_wires(
            builder,
            projection_binding,
            pi_rlc::PI_RLC_PROJECTION_COMBINED_X_LABEL,
            &output,
        );
        let quotient = alloc_vector_projection_quotient_advice(builder, &rho_wires, &inputs)?;
        append_projection_binding_wires(
            builder,
            projection_binding,
            pi_rlc::PI_RLC_PROJECTION_X_QUOTIENTS_LABEL,
            &quotient,
        );
        quotients.push(quotient);
    }
    Ok(quotients)
}

fn alloc_padded_projection_advice(
    builder: &mut R1csBuilder,
    projection_binding: &mut Vec<Var>,
    wires: &RlcPaddedKVectorWires,
    output_label: &'static [u8],
    quotient_label: &'static [u8],
) -> Result<[[Var; PROJECTION_QUOTIENT_LEN]; 2], Error> {
    let rho_wires: Vec<[Var; D]> = wires.inputs.iter().map(|pair| pair.rho_coeffs).collect();
    let inputs_c0: Vec<[Var; D]> = wires
        .inputs
        .iter()
        .map(|pair| core::array::from_fn(|lane| pair.y_c0[lane]))
        .collect();
    let inputs_c1: Vec<[Var; D]> = wires
        .inputs
        .iter()
        .map(|pair| core::array::from_fn(|lane| pair.y_c1[lane]))
        .collect();
    let output_c0: [Var; D] = core::array::from_fn(|lane| wires.combined_c0[lane]);
    let output_c1: [Var; D] = core::array::from_fn(|lane| wires.combined_c1[lane]);
    append_projection_binding_wires(builder, projection_binding, output_label, &output_c0);
    let quotient_c0 = alloc_vector_projection_quotient_advice(builder, &rho_wires, &inputs_c0)?;
    append_projection_binding_wires(builder, projection_binding, quotient_label, &quotient_c0);
    append_projection_binding_wires(builder, projection_binding, output_label, &output_c1);
    let quotient_c1 = alloc_vector_projection_quotient_advice(builder, &rho_wires, &inputs_c1)?;
    append_projection_binding_wires(builder, projection_binding, quotient_label, &quotient_c1);
    Ok([quotient_c0, quotient_c1])
}

fn alloc_projection_binding_preimage(builder: &mut R1csBuilder) -> Vec<Var> {
    crate::paper::digest::pack_bytes_as_fields(pi_rlc::PI_RLC_PROJECTION_BINDING_DOMAIN)
        .into_iter()
        .map(|value| alloc_constant(builder, value))
        .collect()
}

fn append_projection_binding_wires(builder: &mut R1csBuilder, preimage: &mut Vec<Var>, label: &[u8], fields: &[Var]) {
    preimage.extend(
        crate::paper::digest::pack_bytes_as_fields(label)
            .into_iter()
            .map(|value| alloc_constant(builder, value)),
    );
    preimage.push(alloc_constant(builder, F::from_u64(fields.len() as u64)));
    preimage.extend_from_slice(fields);
}

fn alloc_constant(builder: &mut R1csBuilder, value: F) -> Var {
    let var = builder.alloc(value);
    builder.enforce_eq(&Lc::from_var(var), &Lc::from_const(value));
    var
}

// ── Private RLC fold helpers ──────────────────────────────────────────────

fn enforce_rlc_parent_shape(
    builder: &mut R1csBuilder,
    dec_wires: &DecInputWires,
    outputs: &[SplitNcPiCcsOutputWires],
    kappa: usize,
    m_in: usize,
) -> Result<(), Error> {
    let first_output = outputs
        .first()
        .ok_or_else(|| Error::Inner("Π_RLC parent shape requires at least one Π_CCS output".into()))?;
    let parent = &dec_wires.parent;
    let expected_c_len = D * kappa;
    if parent.c_data.len() < expected_c_len {
        return Err(Error::Inner(format!(
            "Π_RLC parent commitment lane count {} < D*kappa {expected_c_len}",
            parent.c_data.len()
        )));
    }
    let expected_x_len = D * m_in;
    if parent.x.len() < expected_x_len {
        return Err(Error::Inner(format!(
            "Π_RLC parent X lane count {} < D*m_in {expected_x_len}",
            parent.x.len()
        )));
    }
    enforce_var_eq(builder, parent.c_d_var, first_output.c_d_var);
    enforce_var_eq(builder, parent.c_kappa_var, first_output.c_kappa_var);
    enforce_var_eq(builder, parent.x_rows_var, first_output.x_rows_var);
    enforce_var_eq(builder, parent.x_cols_var, first_output.x_cols_var);
    enforce_var_eq(builder, parent.m_in_var, first_output.m_in_var);
    Ok(())
}

fn enforce_rlc_output_shape_parity(
    builder: &mut R1csBuilder,
    outputs: &[SplitNcPiCcsOutputWires],
) -> Result<(), Error> {
    let first = outputs
        .first()
        .ok_or_else(|| Error::Inner("Π_RLC output shape parity requires at least one Π_CCS output".into()))?;
    for output in outputs.iter().skip(1) {
        enforce_var_eq(builder, output.c_d_var, first.c_d_var);
        enforce_var_eq(builder, output.c_kappa_var, first.c_kappa_var);
        enforce_var_eq(builder, output.x_rows_var, first.x_rows_var);
        enforce_var_eq(builder, output.x_cols_var, first.x_cols_var);
        enforce_var_eq(builder, output.m_in_var, first.m_in_var);
    }
    Ok(())
}

fn enforce_var_eq(builder: &mut R1csBuilder, a: Var, b: Var) {
    builder.enforce_eq(&Lc::from_var(a), &Lc::from_var(b));
}

fn rlc_commitment_fold_wires(
    rho_wires: &[[Var; D]],
    outputs: &[SplitNcPiCcsOutputWires],
    dec_wires: &DecInputWires,
    kappa: usize,
) -> Result<RlcCommitmentWires, Error> {
    if rho_wires.len() != outputs.len() {
        return Err(Error::Inner(format!(
            "ρ count {} != outputs count {}",
            rho_wires.len(),
            outputs.len()
        )));
    }
    let inputs: Vec<RlcPairWires> = rho_wires
        .iter()
        .zip(outputs.iter())
        .map(|(rho, o)| RlcPairWires {
            rho_coeffs: *rho,
            c_data: o.c_data.clone(),
            kappa: o.c_kappa,
        })
        .collect();
    Ok(RlcCommitmentWires {
        inputs,
        // Π_RLC is defined over the κ fixed by the Π_CCS outputs. A
        // self-consistently wider DEC parent is rejected by the c_kappa
        // wire equality above; keep only the authoritative prefix here so
        // the algebra emits and the rejection remains an unsatisfied row.
        combined_c_data: dec_wires.parent.c_data[..D * kappa].to_vec(),
        kappa,
    })
}

fn rlc_adv_commitment_fold_wires(
    rho_wires: &[[Var; D]],
    outputs: &[SplitNcPiCcsOutputWires],
    dec_wires: &DecInputWires,
) -> Result<Option<LaneCommitments<RlcCommitmentWires>>, Error> {
    let present = outputs.iter().filter(|output| output.adv.is_some()).count();
    match (dec_wires.parent.adv.as_ref(), present) {
        (None, 0) => Ok(None),
        (Some(_), 0) | (None, _) => Err(Error::Inner(
            "Pi_RLC product-commitment adv presence differs between inputs and parent".into(),
        )),
        (Some(parent), count) if count == outputs.len() => {
            validate_adv_shape(Some(parent), parent.ops.d, parent.ops.kappa, "Pi_RLC parent").map_err(Error::Inner)?;
            let output_advs: Vec<&AdvCommitmentWires> = outputs
                .iter()
                .map(|output| output.adv.as_ref().unwrap())
                .collect();
            for (idx, adv) in output_advs.iter().enumerate() {
                validate_adv_shape(
                    Some(adv),
                    parent.ops.d,
                    parent.ops.kappa,
                    &format!("Pi_RLC output[{idx}]"),
                )
                .map_err(Error::Inner)?;
            }
            let coordinate = |select: fn(&AdvCommitmentWires) -> &CommitmentWires,
                              combined: &CommitmentWires|
             -> Result<RlcCommitmentWires, Error> {
                let inputs = rho_wires
                    .iter()
                    .zip(output_advs.iter())
                    .map(|(rho, adv)| {
                        let commitment = select(adv);
                        RlcPairWires {
                            rho_coeffs: *rho,
                            c_data: commitment.data.clone(),
                            kappa: commitment.kappa,
                        }
                    })
                    .collect();
                Ok(RlcCommitmentWires {
                    inputs,
                    combined_c_data: combined.data.clone(),
                    kappa: combined.kappa,
                })
            };
            Ok(Some(LaneCommitments {
                ops: coordinate(|adv| &adv.ops, &parent.ops)?,
                is: coordinate(|adv| &adv.is, &parent.is)?,
                fs: coordinate(|adv| &adv.fs, &parent.fs)?,
            }))
        }
        (Some(_), count) => Err(Error::Inner(format!(
            "Pi_RLC product-commitment adv presence is mixed ({count}/{})",
            outputs.len()
        ))),
    }
}

fn rlc_x_fold_wires(
    rho_wires: &[[Var; D]],
    outputs: &[SplitNcPiCcsOutputWires],
    dec_wires: &DecInputWires,
    m_in: usize,
) -> Result<RlcXWires, Error> {
    let inputs: Vec<RlcXPairWires> = rho_wires
        .iter()
        .zip(outputs.iter())
        .map(|(rho, o)| RlcXPairWires {
            rho_coeffs: *rho,
            x_flat: o.x.clone(),
            m_in: o.x_cols,
        })
        .collect();
    Ok(RlcXWires {
        inputs,
        // The Pi_CCS outputs fix the authoritative X width. A malformed
        // wider parent is rejected by the shape-equality rows above; keep
        // the algebra on the authoritative prefix so synthesis remains a
        // fail-closed unsatisfied circuit instead of returning early.
        combined_x_flat: dec_wires.parent.x[..D * m_in].to_vec(),
        m_in,
    })
}

fn rlc_y_ring_row_fold_wires(
    rho_wires: &[[Var; D]],
    outputs: &[SplitNcPiCcsOutputWires],
    dec_wires: &DecInputWires,
    j: usize,
    d_pad: usize,
) -> Result<RlcPaddedKVectorWires, Error> {
    let inputs: Vec<Vec<KVar>> = outputs.iter().map(|o| o.y_ring[j].clone()).collect();
    let combined = kvars_from_flat_dec(&dec_wires.parent.y_ring[j])?;
    padded_k_vector_wires_from_existing(rho_wires, &inputs, &combined, d_pad)
}

fn rlc_y_zcol_fold_wires(
    rho_wires: &[[Var; D]],
    outputs: &[SplitNcPiCcsOutputWires],
    dec_wires: &DecInputWires,
    d_pad: usize,
) -> Result<RlcPaddedKVectorWires, Error> {
    let inputs: Vec<Vec<KVar>> = outputs.iter().map(|o| o.y_zcol.clone()).collect();
    let combined = kvars_from_flat_dec(&dec_wires.parent.y_zcol)?;
    padded_k_vector_wires_from_existing(rho_wires, &inputs, &combined, d_pad)
}

fn enforce_rlc_fold_digest_consistency(
    builder: &mut R1csBuilder,
    outputs: &[SplitNcPiCcsOutputWires],
    dec_wires: &DecInputWires,
) -> Result<(), Error> {
    if outputs.is_empty() {
        return Err(Error::Inner(
            "Π_RLC fold_digest consistency requires at least one Π_CCS output".into(),
        ));
    }
    for output in outputs {
        for lane in 0..output.fold_digest_fields.len() {
            builder.enforce_eq(
                &Lc::from_var(dec_wires.parent.fold_digest_fields[lane]),
                &Lc::from_var(output.fold_digest_fields[lane]),
            );
        }
    }
    Ok(())
}

/// Convert DEC's flat-K representation `[k0.c0, k0.c1, k1.c0, k1.c1, …]`
/// back into a `Vec<KVar>` view, no fresh wire allocation. DEC stores
/// `y_ring`/`y_zcol` as `Vec<Var>` of length `d_pad * K_LIMBS` where
/// `K_LIMBS = 2`; SplitNc consumers want `Vec<KVar>` of length `d_pad`.
fn kvars_from_flat_dec(flat: &[Var]) -> Result<Vec<KVar>, Error> {
    if flat.len() % 2 != 0 {
        return Err(Error::Inner(format!(
            "DEC flat K-vector has odd limb count {}",
            flat.len()
        )));
    }
    Ok(flat
        .chunks_exact(2)
        .map(|c| KVar { c0: c[0], c1: c[1] })
        .collect())
}

/// Construct a [`RlcPaddedKVectorWires`] from already-allocated wires
/// (no fresh allocation). Used by both `y_ring[j]` and `y_zcol` folds.
fn padded_k_vector_wires_from_existing(
    rhos: &[[Var; D]],
    inputs: &[Vec<KVar>],
    combined: &[KVar],
    d_pad: usize,
) -> Result<RlcPaddedKVectorWires, Error> {
    if combined.len() != d_pad {
        return Err(Error::Inner(format!(
            "padded K-vector: combined.len ({}) != d_pad ({})",
            combined.len(),
            d_pad
        )));
    }
    if rhos.len() != inputs.len() {
        return Err(Error::Inner(format!(
            "padded K-vector: ρ count {} != input count {}",
            rhos.len(),
            inputs.len()
        )));
    }
    let mut pair_wires = Vec::with_capacity(inputs.len());
    for (idx, (rho, y)) in rhos.iter().zip(inputs.iter()).enumerate() {
        if y.len() != d_pad {
            return Err(Error::Inner(format!(
                "padded K-vector: inputs[{idx}].len ({}) != d_pad ({})",
                y.len(),
                d_pad
            )));
        }
        let y_c0: Vec<Var> = y.iter().map(|v| v.c0).collect();
        let y_c1: Vec<Var> = y.iter().map(|v| v.c1).collect();
        pair_wires.push(RlcPaddedKVectorPairWires {
            rho_coeffs: *rho,
            y_c0,
            y_c1,
        });
    }
    let combined_c0: Vec<Var> = combined.iter().map(|v| v.c0).collect();
    let combined_c1: Vec<Var> = combined.iter().map(|v| v.c1).collect();
    Ok(RlcPaddedKVectorWires {
        inputs: pair_wires,
        combined_c0,
        combined_c1,
        d_pad,
    })
}

fn enforce_kvar_vec_eq(builder: &mut R1csBuilder, a: &[KVar], b: &[KVar]) -> Result<(), Error> {
    if a.len() != b.len() {
        return Err(Error::Inner(format!(
            "KVar vector length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    for (x, y) in a.iter().zip(b.iter()) {
        builder.enforce_eq(&Lc::from_var(x.c0), &Lc::from_var(y.c0));
        builder.enforce_eq(&Lc::from_var(x.c1), &Lc::from_var(y.c1));
    }
    Ok(())
}

// Suppress unused-import warnings if F/K become unused in future edits.
#[allow(dead_code)]
fn _silence_unused_f_k(_f: F, _k: K) {}
