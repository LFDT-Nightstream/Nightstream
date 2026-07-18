//! SplitNcV1 — full Π_CCS.V composition.
//!
//! Wires every sub-gadget added in sub-steps A-I into one verifier
//! function that mirrors
//! `optimized_verify_with_cache_and_public_instance_digest_impl` plus the
//! `tr.digest32()` catch-up in `crate::engine::optimized::verify_pi_ccs`.
//!
//! ## Composition order
//!
//! ```text
//! 1. Allocate fresh/running/output wires (once)
//! 2. Recompute per-fresh CCS claim digests from those wires
//! 3. Strictly verify the running children against their Pi_DEC parent,
//!    hash the parent cache, and independently hash the exact child CE core
//! 4. Compute pi_ccs_instance_digest from authoritative digests
//! 5. Absorb header bundle + instance digest (raw [11, …]/[12, …])
//! 6. Absorb the exact-child ME handle (raw [4]/[5,count]/[13, handle])
//! 7. Sample α, β_a, β_r, γ, β_m
//! 8. FE claimed_initial
//! 9. FE sumcheck driver → r_prime, alpha_prime, fe_final
//! 10. NC sumcheck driver → s_col_prime, alpha_prime_nc, nc_final
//! 11. Bind outputs to fresh/running (wire-to-wire, not wire-to-const),
//!     bind output.r/s_col to r_prime/s_col_prime, and canonicalize
//!     y_ring's padded lanes
//! 12. FE terminal identity, pin to fe_final
//! 13. NC terminal identity, pin to nc_final
//! 14. header_digest catch-up squeeze + bind every output.fold_digest
//!     to the caught-up transcript digest
//! 15. Recompute and verify the wire-format `outputs_digest` that NIFS.V
//!     carries into Π_RLC
//! ```
//!
//! ## Soundness rules
//!
//! - The public-instance digest absorbed in step 5 is **recomputed** from
//!   authoritative claim wires in steps 2-4 — it is never accepted from the
//!   prover as a witness wire.
//! - Output→input binding (step 11) uses **wire-to-wire** equality. The
//!   native values are only consulted by [`R1csBuilder::alloc`] to seed the
//!   witness; constraints reference the wires, so the same `verifier.rs`
//!   can be reused inside F' where the inputs are *also* witness wires
//!   rather than test-time constants.
//!
//! Owns: ordered Pi_CCS verifier orchestration and its derived wire surface.
//!
//! Does not own: leaf digest, transcript, FE, or NC arithmetic.
//!
//! Emits constraints: yes, by composing the child phases below.
//!
//! Authority boundary: fresh claims and the exact ordered running CE cores are
//! authoritative; the checked parent is a cache. The optimized `y_zcol`
//! sidecar still has an open delayed-NC source-binding obligation.
//!
//! | Child phase | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | allocation | Fixed-shape claim materialization | yes | `verifier` | concrete refinement open |
//! | running authority | Exact child-core binding, Pi_DEC consistency, and canonical views | yes | `verifier`, `pi_dec_circuit` | authority bridge open |
//! | running `y_zcol` elision | Do not pretend an unproved sidecar equation is authority | no | `alloc_ce_wires_without_y_zcol` | delayed-NC refinement open |
//! | claim hashes | Fresh claims, exact child handle, and parent-cache binding | yes | `digests` | digest bridge open |
//! | transcript | Instance, handle, and challenge schedule | yes | `transcript` | transcript bridge open |
//! | FE | Claimed initial, SumCheck, terminal identity | yes | `fe` | FE bridge open |
//! | NC | SumCheck and terminal identity | yes | `nc` | NC bridge open |
//! | outputs | Input continuity, header, and Pi_RLC message | yes | `verifier`, `digests` | output bridge open |

use neo_ccs::{CcsStructure, SparsePoly};
use neo_math::ring::D;
use neo_math::{KExtensions, F, K};
use p3_field::PrimeCharacteristicRing;

use super::stage;
use super::{
    absorb_engine_header_bundle_and_instance_digest, absorb_engine_header_bundle_wires_and_instance_digest,
    absorb_engine_me_inputs_accumulator_handle, alloc_constant_var, enforce_accumulator_ce_claim_digest,
    enforce_accumulator_claims_digest, enforce_ccs_claim_digest, enforce_fe_claimed_initial,
    enforce_fe_sumcheck_driver, enforce_fe_terminal_identity, enforce_header_digest_catch_up_wires,
    enforce_nc_sumcheck_driver, enforce_nc_terminal_identity, enforce_pi_ccs_instance_digest_parent_authority,
    enforce_pi_ccs_outputs_digest, header_digest_bytes_to_fields, sample_engine_beta_m, sample_engine_challenges,
    AccumulatorCeClaimDigestInputs, Error, FeClaimedInitialInputs, FeTerminalInputs, NcTerminalInputs,
    PiCcsOutputMessageDigestInputs, PiCcsOutputsPreimage,
};
use crate::engine::r1cs_circuit::boolean;
use crate::engine::r1cs_circuit::builder::{Lc, Var};
use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::transcript::TranscriptGadget;
use crate::engine::r1cs_circuit::R1csBuilder;
use crate::paper::digest::AccumulatorHandle;
use crate::paper::params::Params;
use crate::paper::reductions::pi_ccs_output_message::Profile as PiCcsOutputMessageProfile;
use crate::paper::reductions::pi_dec_circuit::{
    enforce_dec_v_strict, enforce_split_nc_d_pad_shape, CeClaimWires as DecCeClaimWires, DecInputWires,
};
use crate::paper::relations::product_commitment_circuit::{alloc_adv, enforce_adv_equality, AdvCommitmentWires};
use crate::paper::relations::{validate_adv_shape, CcsClaim, CeClaim};

/// Matrix-independent CCS header consumed by the in-circuit verifier.
///
/// The verifier evaluates the relation polynomial over claimed matrix
/// evaluations; it never reads a matrix coefficient. Owning only this header
/// makes that boundary explicit and lets preprocessing discover recursive
/// dimensions without allocating candidate matrices.
#[derive(Clone)]
pub struct SplitNcVerifierRelation {
    n: usize,
    m: usize,
    polynomial: SparsePoly<F>,
}

impl SplitNcVerifierRelation {
    pub fn from_structure(structure: &CcsStructure<F>) -> Self {
        Self::from_parts(structure.n, structure.m, structure.f.clone())
    }

    pub(crate) fn from_parts(n: usize, m: usize, polynomial: SparsePoly<F>) -> Self {
        assert!(n > 0, "SplitNc verifier relation requires at least one row");
        assert!(m > 0, "SplitNc verifier relation requires at least one column");
        assert!(
            polynomial.arity() > 0,
            "SplitNc verifier relation requires a nonempty polynomial"
        );
        Self { n, m, polynomial }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn t(&self) -> usize {
        self.polynomial.arity()
    }

    pub fn max_degree(&self) -> u32 {
        self.polynomial.max_degree()
    }

    pub fn polynomial(&self) -> &SparsePoly<F> {
        &self.polynomial
    }
}

impl From<&CcsStructure<F>> for SplitNcVerifierRelation {
    fn from(structure: &CcsStructure<F>) -> Self {
        Self::from_structure(structure)
    }
}

/// Static configuration for one SplitNc Π_CCS.V invocation.
///
/// Everything here is baked in at gadget-emit time:
/// - `params` / `structure` come from the F'-side public structure.
/// - `header_bundle` is the four-lane Poseidon2 digest of `(params, structure,
///   dims, mat_digest)`, computed natively once and embedded as a constant.
/// - `ell_d`, `ell_n`, `ell_m`, `d_sc` come from `Dims` (also a function of
///   `(params, structure)`).
pub struct SplitNcPiCcsVConfig<'a> {
    pub params: &'a Params,
    pub structure: SplitNcVerifierRelation,
    pub header_bundle: [F; 4],
    pub ell_d: usize,
    pub ell_n: usize,
    pub ell_m: usize,
    pub d_sc: usize,
}

/// Witness/protocol messages from a real native `pi_ccs::prove` proof.
///
/// - `fresh` / `running`: the public claim arrays both sides already hold.
/// - `outputs`: the K+k output CE claims the prover sends on the wire.
/// - `sumcheck_rounds_fe` / `sumcheck_rounds_nc`: per-round K-coeff lists.
/// - `header_digest`: native `tr.digest32()` captured after the engine's
///   verify path; the catch-up squeeze pins to it.
pub struct SplitNcPiCcsVMessages<'a> {
    pub fresh: &'a [CcsClaim],
    pub running: &'a [CeClaim],
    pub running_parent_authority: Option<&'a CeClaim>,
    pub outputs: &'a [CeClaim],
    /// Redundant wire-format digest checked by native `pi_ccs::verify`.
    /// The circuit recomputes it from `outputs`; this field is never treated
    /// as authority.
    pub outputs_digest: [F; 4],
    /// Optional prover copy of the FE claimed initial sum. Native verification
    /// accepts absence and, when present, requires equality to the value
    /// derived from the public inputs.
    pub sc_initial_sum: Option<K>,
    pub sumcheck_rounds_fe: &'a [Vec<K>],
    pub sumcheck_rounds_nc: &'a [Vec<K>],
    pub header_digest: &'a [u8],
}

/// Per-output wire bundle that downstream Π_RLC.V (and any consumer that
/// needs to fold output claims) consumes after a successful SplitNc verify.
///
/// Each output is one CE claim, and Π_RLC combines them into a parent CE.
/// Every field here is a witness wire (or row of wires) already constrained
/// by the SplitNc verifier's identity and transcript checks.
#[derive(Clone)]
pub struct SplitNcPiCcsOutputWires {
    pub c_d: usize,
    pub c_d_var: Var,
    pub c_kappa: usize,
    pub c_kappa_var: Var,
    pub c_data: Vec<Var>,
    pub adv: Option<AdvCommitmentWires>,
    pub x: Vec<Var>,
    pub x_rows: usize,
    pub x_rows_var: Var,
    pub x_cols: usize,
    pub x_cols_var: Var,
    pub m_in: usize,
    pub m_in_var: Var,
    pub r: Vec<KVar>,
    pub s_col: Vec<KVar>,
    pub y_ring: Vec<Vec<KVar>>,
    /// SuperNeo scalar/constant-term view of `y_ring`.
    ///
    /// The SplitNc verifier constrains this as `ct[j] == y_ring[j][0]`;
    /// downstream continuity gates then carry the denormalized field
    /// wire-to-wire without treating it as independent authority.
    pub ct: Vec<KVar>,
    pub y_zcol: Vec<KVar>,
    /// `fold_digest` field of the CE claim, projected to four base-field
    /// lanes (matches `digest32_as_fields`). Carried by the bundle so
    /// downstream consumers (decider CE-continuity gate, audit tooling)
    /// can re-bind without re-walking the original claim's witness.
    pub fold_digest_fields: [Var; 4],
}

/// Derived wires the SplitNc Π_CCS.V verifier exposes to the downstream
/// NIFS.V / Π_RLC composition once verification passes.
///
/// `fresh_x` and `running_c_data` are surfaced separately because F' R1CS
/// (above NIFS.V) needs them to enforce the HyperNova recursive link
/// (`u_i.public == enc_inst(hash(prior_state))`) and the accumulator
/// digest binding (`acc_digest_in == digest(running)`). They're already
/// constrained by the Π_CCS.V verifier; this just exposes the wires
/// without forcing F' to re-walk the witness layout.
pub struct SplitNcPiCcsVDerived {
    pub r_prime: Vec<KVar>,
    pub s_col_prime: Vec<KVar>,
    pub outputs: Vec<SplitNcPiCcsOutputWires>,
    /// Π_CCS output digest recomputed from the constrained output wires and
    /// checked against the redundant proof field. NIFS.V appends these same
    /// wires before sampling Π_RLC challenges.
    pub output_claims_digest: [Var; 4],
    /// Exact pre-SIS field-to-column ownership used to compute
    /// [`Self::output_claims_digest`].
    pub output_message_preimage: PiCcsOutputsPreimage,
    /// `fresh_x[i]` = the `m_in` public-input `F`-wires of `fresh[i]`.
    pub fresh_x: Vec<Vec<Var>>,
    /// Product-commitment coordinates of those same fresh claims. Each
    /// entry shares allocation and transcript binding with `fresh_x[i]`.
    pub fresh_adv: Vec<Option<AdvCommitmentWires>>,
    /// `running_c_data[i]` = the `D * kappa` commitment-data wires of `running[i]`.
    pub running_c_data: Vec<Vec<Var>>,
    /// Per-running-claim CE core wires used by the exact ordered accumulator
    /// binding. `y_zcol` remains absent until its source relation is proved.
    pub running: Vec<SplitNcPiCcsOutputWires>,
    /// Π_RLC parent whose Π_DEC children form `running`. It is a separately
    /// checked transcript input, not the Construction-2 accumulator handle.
    pub running_parent_authority: Option<SplitNcPiCcsOutputWires>,
    /// Four-lane Poseidon2 handle of the exact ordered running child vector.
    /// The checked parent is only a recomposition cache.
    pub running_acc_digest: [Var; 4],
}

// ── Wire-allocation structs ───────────────────────────────────────────────

/// Wires for one fresh CCS claim, allocated once and reused across the
/// digest gadget and any wire-to-wire output binding.
struct CcsClaimWires {
    c_d: usize,
    c_d_var: Var,
    c_kappa: usize,
    c_kappa_var: Var,
    c_data: Vec<Var>,
    adv: Option<AdvCommitmentWires>,
    x: Vec<Var>,
    m_in: usize,
    m_in_var: Var,
}

/// Wires for one CE claim (running input or output), allocated once.
///
/// `x` is the X matrix in row-major order: `x[r * x_cols + c] = X[(r, c)]`.
/// The SuperNeo-packed view used by the ME-projection digest is built on
/// demand via [`Self::x_packed`] without reallocating any wires.
struct CeClaimWires {
    c_d: usize,
    c_d_var: Var,
    c_kappa: usize,
    c_kappa_var: Var,
    c_data: Vec<Var>,
    adv: Option<AdvCommitmentWires>,
    x: Vec<Var>,
    x_rows: usize,
    x_rows_var: Var,
    x_cols: usize,
    x_cols_var: Var,
    r: Vec<KVar>,
    s_col: Vec<KVar>,
    y_ring: Vec<Vec<KVar>>,
    ct: Vec<KVar>,
    y_zcol: Vec<KVar>,
    m_in: usize,
    m_in_var: Var,
    fold_digest_fields: [Var; 4],
}

impl CeClaimWires {
    /// SuperNeo-packed view of X: `x_packed[c] = X[(c % D, c / D)]` for
    /// `c ∈ 0..m_in`. Returns `Var` copies (indices) into `self.x`, no fresh
    /// witness wires.
    fn x_packed(&self) -> Vec<Var> {
        (0..self.m_in)
            .map(|c| self.x[(c % D) * self.x_cols + (c / D)])
            .collect()
    }
}

fn accumulator_digest_inputs(claim: &CeClaimWires) -> AccumulatorCeClaimDigestInputs<'_> {
    AccumulatorCeClaimDigestInputs {
        c_d: claim.c_d,
        c_kappa: claim.c_kappa,
        c_data: &claim.c_data,
        x_rows: claim.x_rows,
        x_cols: claim.x_cols,
        x_flat_row_major: &claim.x,
        r: &claim.r,
        s_col: &claim.s_col,
        y_ring: &claim.y_ring,
        ct: &claim.ct,
        m_in: claim.m_in,
        fold_digest_fields: claim.fold_digest_fields,
        adv: claim.adv.as_ref(),
    }
}

// ── Public entry ──────────────────────────────────────────────────────────

/// Compose the full SplitNcV1 Π_CCS.V verifier on top of `transcript`.
pub fn enforce_split_nc_pi_ccs_v(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    cfg: &SplitNcPiCcsVConfig<'_>,
    msg: &SplitNcPiCcsVMessages<'_>,
) -> Result<SplitNcPiCcsVDerived, Error> {
    enforce_split_nc_pi_ccs_v_inner(builder, transcript, cfg, msg, None)
}

/// Folded-F' entrypoint. The header is verifier-key advice, so its values
/// do not become constants in a relation that ultimately verifies itself.
pub fn enforce_split_nc_pi_ccs_v_with_header_bundle_wires(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    cfg: &SplitNcPiCcsVConfig<'_>,
    msg: &SplitNcPiCcsVMessages<'_>,
    header_bundle: [Var; 4],
) -> Result<SplitNcPiCcsVDerived, Error> {
    enforce_split_nc_pi_ccs_v_inner(builder, transcript, cfg, msg, Some(header_bundle))
}

fn enforce_split_nc_pi_ccs_v_inner(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    cfg: &SplitNcPiCcsVConfig<'_>,
    msg: &SplitNcPiCcsVMessages<'_>,
    header_bundle: Option<[Var; 4]>,
) -> Result<SplitNcPiCcsVDerived, Error> {
    let k_mcs = msg.fresh.len();
    let k_me = msg.running.len();
    let k_total = k_mcs + k_me;
    let t = cfg.structure.t();
    let d_pad = 1usize << cfg.ell_d;

    if k_mcs == 0 {
        return Err(Error::Shape(
            "SplitNc Π_CCS.V requires at least one fresh CCS claim".into(),
        ));
    }
    if k_mcs > cfg.params.max_fresh_count() {
        return Err(Error::Shape(format!(
            "fresh length {k_mcs} exceeds params.max_fresh_count()={}",
            cfg.params.max_fresh_count()
        )));
    }
    if msg.outputs.len() != k_total {
        return Err(Error::Shape(format!(
            "outputs.len={} expected fresh+running={k_total}",
            msg.outputs.len()
        )));
    }
    // Native parity: `pi_ccs::validate_verifier_shape` rejects a non-empty
    // running whose length is not `params.k_rho()`. The structural shape
    // is fixed at gadget-emit time, but mirroring the guard here keeps
    // the in-circuit verifier from silently accepting a malformed
    // running batch in a future caller that wires a different `msg`
    // shape (cheap, no constraints — pure native validation).
    if k_me != 0 && (k_me as u32) != cfg.params.k_rho() {
        return Err(Error::Shape(format!(
            "running length {k_me} does not match params.k_rho()={}",
            cfg.params.k_rho()
        )));
    }

    // ── 0. Validate fresh / running / output shapes up front ─────────────
    // Mirrors native shape checks (kappa, m_in, X dims, r/s_col/y_ring/y_zcol
    // lengths) so the verifier never reaches indexing paths with malformed
    // inputs.
    for (i, f) in msg.fresh.iter().enumerate() {
        validate_fresh_shape(cfg, i, f)?;
    }
    for (i, r) in msg.running.iter().enumerate() {
        validate_ce_shape_without_y_zcol(cfg, &format!("running[{i}]"), r)?;
    }
    match (k_me, msg.running_parent_authority) {
        (0, None) => {}
        (0, Some(_)) => {
            return Err(Error::Shape(
                "running parent authority present while running is empty".into(),
            ))
        }
        (_, Some(parent)) => validate_running_parent_authority_shape(cfg, parent)?,
        (_, None) => {
            return Err(Error::Shape(
                "non-empty running accumulator missing Pi_RLC parent authority".into(),
            ))
        }
    }
    for (i, o) in msg.outputs.iter().enumerate() {
        // Outputs may have different X cols than running (fresh outputs use
        // m_in columns); the per-idx output X shape is also checked in
        // `bind_outputs_to_inputs`. The structural CE invariants — kappa,
        // r/s_col length, y_ring outer/inner length, y_zcol length — apply
        // uniformly.
        validate_output_ce_shape(cfg, &format!("outputs[{i}]"), o)?;
    }

    // ── 1. Allocate fresh / running / output wires once ──────────────────
    let allocation_start = builder.rows();
    builder.begin_encoding_stage(stage::ROOT);
    builder.begin_encoding_stage(stage::ALLOCATE_AND_NORMALIZE);
    builder.begin_encoding_stage(stage::ALLOCATE_FRESH);
    let fresh_wires: Vec<CcsClaimWires> = msg
        .fresh
        .iter()
        .map(|f| alloc_fresh_wires(builder, f))
        .collect();
    builder.begin_encoding_stage(stage::ALLOCATE_RUNNING);
    let running_wires: Vec<CeClaimWires> = msg
        .running
        .iter()
        .map(|r| alloc_ce_wires_without_y_zcol(builder, r))
        .collect::<Result<_, _>>()?;
    builder.begin_encoding_stage(stage::ALLOCATE_RUNNING_PARENT);
    let running_parent_authority_wires = msg
        .running_parent_authority
        .map(|parent| alloc_ce_wires_with_canonical_y_zcol(builder, parent, d_pad))
        .transpose()?;
    builder.begin_encoding_stage(stage::ALLOCATE_OUTPUTS);
    let output_wires: Vec<CeClaimWires> = msg
        .outputs
        .iter()
        .map(|o| alloc_ce_wires(builder, o))
        .collect::<Result<_, _>>()?;
    builder.record_row_family(stage::ROW_ALLOCATION, allocation_start);

    let authority_start = builder.rows();
    builder.begin_encoding_stage(stage::RUNNING_AUTHORITY);
    builder.begin_encoding_stage(stage::RUNNING_AUTHORITY_PARENT_DEC);
    // The compact accumulator handle below is valid only after the running
    // children have been checked as a strict Pi_DEC reduction of their
    // parent. Keep that precondition in this verifier, beside the handle,
    // so standalone Pi_CCS composition cannot expose unconstrained children.
    enforce_running_parent_authority_consistency(
        builder,
        cfg,
        &running_wires,
        running_parent_authority_wires.as_ref(),
    )?;

    // Output `ct` is a denormalized scalar/constant-term view of `y_ring`.
    // Running `ct` and padding are already covered by strict Pi_DEC above.
    for (idx, ow) in output_wires.iter().enumerate() {
        builder.begin_encoding_stage(stage::RUNNING_AUTHORITY_OUTPUT_CT);
        enforce_ct_from_y_ring(builder, &format!("outputs[{idx}]"), ow)?;
        builder.begin_encoding_stage(stage::RUNNING_AUTHORITY_OUTPUT_Y_RING_PADDING);
        enforce_y_ring_padding_zero(builder, ow);
        builder.begin_encoding_stage(stage::RUNNING_AUTHORITY_OUTPUT_Y_ZCOL_PADDING);
        enforce_y_zcol_padding_zero(builder, ow);
    }
    builder.record_row_family(stage::ROW_AUTHORITY, authority_start);

    // ── 2. Fresh CCS digests (from allocated wires) ──────────────────────
    let fresh_digests_start = builder.rows();
    builder.begin_encoding_stage(stage::FRESH_CLAIM_HASHES);
    builder.begin_encoding_stage(stage::FRESH_CLAIM_HASHES_DIGEST);
    let mut fresh_digests: Vec<[Var; 4]> = Vec::with_capacity(k_mcs);
    for fw in &fresh_wires {
        fresh_digests.push(enforce_ccs_claim_digest(
            builder,
            fw.c_d,
            fw.c_kappa,
            &fw.c_data,
            &fw.x,
            fw.m_in,
            fw.adv.as_ref(),
        ));
    }
    builder.record_row_family(stage::ROW_FRESH_DIGESTS, fresh_digests_start);

    // ── 3. Running parent digest + shared-r check ────────────────────────
    let running_authority_start = builder.rows();
    //
    // Bind the checked Π_RLC parent cache independently. This digest enters
    // the instance header; the exact child vector enters through the separate
    // accumulator handle below. Neither binding substitutes for the other.
    builder.begin_encoding_stage(stage::RUNNING_PARENT_HASH);
    builder.begin_encoding_stage(stage::RUNNING_PARENT_HASH_SHARED_R);
    for (idx, rw) in running_wires.iter().enumerate() {
        // Shared-r check (mirrors `shared_me_input_r` in the engine): all
        // running ME inputs must carry the same evaluation point.
        if idx > 0 {
            let first = &running_wires[0].r;
            if rw.r.len() != first.len() {
                return Err(Error::Shape("running ME inputs must share evaluation point r".into()));
            }
            for (a, b) in rw.r.iter().zip(first.iter()) {
                enforce_kvar_eq(builder, *a, *b);
            }
        }
    }
    builder.begin_encoding_stage(stage::RUNNING_PARENT_HASH_DIGEST);
    let running_parent_digest = running_parent_authority_wires
        .as_ref()
        .map(|parent| enforce_accumulator_ce_claim_digest(builder, &accumulator_digest_inputs(parent)))
        .transpose()?;
    builder.record_row_family(stage::ROW_RUNNING_AUTHORITY, running_authority_start);

    // ── 4-5. Instance digest + header/instance absorbs ───────────────────
    let transcript_start = builder.rows();
    builder.begin_encoding_stage(stage::INSTANCE_HASH_AND_ABSORB);
    builder.begin_encoding_stage(stage::INSTANCE_HASH);
    let instance_digest =
        enforce_pi_ccs_instance_digest_parent_authority(builder, &fresh_digests, k_me, running_parent_digest);
    builder.begin_encoding_stage(stage::INSTANCE_HEADER_ABSORB);
    match header_bundle {
        Some(header_bundle) => {
            absorb_engine_header_bundle_wires_and_instance_digest(builder, transcript, header_bundle, instance_digest)
        }
        None => {
            absorb_engine_header_bundle_and_instance_digest(builder, transcript, cfg.header_bundle, instance_digest)
        }
    }

    // ── 6. ME-input absorb (exact ordered accumulator handle) ────────────
    //
    // Strict Π_DEC recomposition is not injective in the child vector. Hash
    // every paper-level child core in order; the parent digest above remains
    // only an independently checked cache used by the instance transcript.
    builder.begin_encoding_stage(stage::RUNNING_HANDLE_HASH_AND_ABSORB);
    builder.begin_encoding_stage(stage::RUNNING_HANDLE_CHILD_DIGESTS);
    let child_digests = running_wires
        .iter()
        .map(|child| enforce_accumulator_ce_claim_digest(builder, &accumulator_digest_inputs(child)))
        .collect::<Result<Vec<_>, _>>()?;
    builder.begin_encoding_stage(stage::RUNNING_HANDLE_AGGREGATE);
    let running_acc_digest = if child_digests.is_empty() {
        let empty = AccumulatorHandle::empty().digest_fields();
        std::array::from_fn(|lane| alloc_constant_var(builder, empty[lane]))
    } else {
        enforce_accumulator_claims_digest(builder, &child_digests)
    };
    builder.begin_encoding_stage(stage::RUNNING_HANDLE_ABSORB);
    absorb_engine_me_inputs_accumulator_handle(builder, transcript, k_me, running_acc_digest);

    // ── 7. Sample engine challenges + β_m ────────────────────────────────
    builder.begin_encoding_stage(stage::ENGINE_CHALLENGES);
    builder.begin_encoding_stage(stage::ENGINE_CHALLENGES_MAIN);
    let ch = sample_engine_challenges(builder, transcript, cfg.ell_d, cfg.ell_n);
    builder.begin_encoding_stage(stage::ENGINE_CHALLENGES_BETA_M);
    let beta_m = sample_engine_beta_m(builder, transcript, cfg.ell_m);
    builder.record_row_family(stage::ROW_TRANSCRIPT, transcript_start);

    // ── 8. FE claimed_initial ────────────────────────────────────────────
    let fe_initial_start = builder.rows();
    builder.begin_encoding_stage(stage::FE_CLAIM_AND_SUMCHECK);
    builder.begin_encoding_stage(stage::FE_CLAIMED_INITIAL);
    let running_y_ring_view: Vec<Vec<Vec<KVar>>> = running_wires.iter().map(|rw| rw.y_ring.clone()).collect();
    let claimed_initial = enforce_fe_claimed_initial(
        builder,
        &FeClaimedInitialInputs {
            k_mcs,
            t,
            ell_d: cfg.ell_d,
            gamma: ch.gamma,
            alpha: &ch.alpha,
            running_y_ring: &running_y_ring_view,
        },
    )?;
    builder.record_row_family(stage::ROW_FE_INITIAL, fe_initial_start);
    let fe_optional_claim_start = builder.rows();
    builder.begin_encoding_stage(stage::FE_OPTIONAL_CLAIM);
    enforce_optional_k_equality(builder, msg.sc_initial_sum, claimed_initial);
    builder.record_row_family(stage::ROW_FE_OPTIONAL_CLAIM, fe_optional_claim_start);

    // ── 9. FE sumcheck driver ────────────────────────────────────────────
    let fe_sumcheck_start = builder.rows();
    builder.begin_encoding_stage(stage::FE_ROUNDS);
    let fe_rounds: Vec<Vec<KVar>> = msg
        .sumcheck_rounds_fe
        .iter()
        .map(|r| alloc_k_vec(builder, r))
        .collect();
    builder.begin_encoding_stage(stage::FE_SUMCHECK_DRIVER);
    let fe = enforce_fe_sumcheck_driver(
        builder,
        transcript,
        cfg.ell_n,
        cfg.ell_d,
        cfg.d_sc,
        claimed_initial,
        &fe_rounds,
    )?;
    builder.record_row_family(stage::ROW_FE_SUMCHECK, fe_sumcheck_start);

    // ── 10. NC sumcheck driver ───────────────────────────────────────────
    let nc_sumcheck_start = builder.rows();
    builder.begin_encoding_stage(stage::NC_SUMCHECK);
    builder.begin_encoding_stage(stage::NC_SUMCHECK_ROUNDS);
    let nc_rounds: Vec<Vec<KVar>> = msg
        .sumcheck_rounds_nc
        .iter()
        .map(|r| alloc_k_vec(builder, r))
        .collect();
    builder.begin_encoding_stage(stage::NC_SUMCHECK_DRIVER);
    let nc = enforce_nc_sumcheck_driver(builder, transcript, cfg.ell_m, cfg.ell_d, cfg.d_sc, &nc_rounds)?;
    builder.record_row_family(stage::ROW_NC_SUMCHECK, nc_sumcheck_start);

    // ── 11. Bind outputs to inputs (wire-to-wire) ────────────────────────
    let output_binding_start = builder.rows();
    builder.begin_encoding_stage(stage::OUTPUT_BINDING_AND_TERMINAL_CHECKS);
    builder.begin_encoding_stage(stage::OUTPUT_BINDING);
    bind_outputs_to_inputs(
        builder,
        &fresh_wires,
        &running_wires,
        &output_wires,
        &fe.r_prime,
        &nc.s_col_prime,
        d_pad,
    )?;
    builder.record_row_family(stage::ROW_OUTPUT_BINDING, output_binding_start);

    let output_y_ring_view: Vec<Vec<Vec<KVar>>> = output_wires.iter().map(|ow| ow.y_ring.clone()).collect();
    let output_y_zcol_view: Vec<Vec<KVar>> = output_wires.iter().map(|ow| ow.y_zcol.clone()).collect();

    // ── 12. FE terminal identity ─────────────────────────────────────────
    let fe_terminal_start = builder.rows();
    builder.begin_encoding_stage(stage::FE_TERMINAL_IDENTITY);
    let me_input_r: Option<&[KVar]> = running_wires.first().map(|w| w.r.as_slice());
    let rhs_fe = enforce_fe_terminal_identity(
        builder,
        &FeTerminalInputs {
            poly: cfg.structure.polynomial(),
            t,
            k_mcs,
            gamma: ch.gamma,
            alpha: &ch.alpha,
            beta_a: &ch.beta_a,
            beta_r: &ch.beta_r,
            r_prime: &fe.r_prime,
            alpha_prime: &fe.alpha_prime,
            me_input_r,
            output_y_ring: &output_y_ring_view,
        },
    )?;
    builder.begin_encoding_stage(stage::FE_TERMINAL_FINAL_SUM);
    enforce_kvar_eq(builder, fe.final_sum, rhs_fe);
    builder.record_row_family(stage::ROW_FE_TERMINAL, fe_terminal_start);

    // ── 13. NC terminal identity ─────────────────────────────────────────
    let nc_terminal_start = builder.rows();
    builder.begin_encoding_stage(stage::NC_TERMINAL_IDENTITY);
    let rhs_nc = enforce_nc_terminal_identity(
        builder,
        &NcTerminalInputs {
            b: cfg.params.b(),
            gamma: ch.gamma,
            beta_a: &ch.beta_a,
            beta_m: &beta_m,
            s_col_prime: &nc.s_col_prime,
            alpha_prime: &nc.alpha_prime,
            output_y_zcol: &output_y_zcol_view,
        },
    )?;
    builder.begin_encoding_stage(stage::NC_TERMINAL_FINAL_SUM);
    enforce_kvar_eq(builder, nc.final_sum, rhs_nc);
    builder.record_row_family(stage::ROW_NC_TERMINAL, nc_terminal_start);

    // ── 14. Header digest catch-up squeeze ───────────────────────────────
    let catchup_start = builder.rows();
    builder.begin_encoding_stage(stage::HEADER_CATCH_UP);
    builder.begin_encoding_stage(stage::HEADER_FIELDS);
    let header_fields = header_digest_bytes_to_fields(msg.header_digest)?;
    let header_wires = header_fields.map(|value| builder.alloc(value));
    builder.begin_encoding_stage(stage::HEADER_TRANSCRIPT);
    enforce_header_digest_catch_up_wires(builder, transcript, header_wires);
    builder.begin_encoding_stage(stage::HEADER_OUTPUT_BINDING);
    enforce_output_fold_digest_matches_header(builder, &output_wires, header_wires);
    builder.record_row_family(stage::ROW_CATCH_UP, catchup_start);

    // ── 15. Recompute the wire-format Π_CCS output digest ────────────────
    //
    // Native `pi_ccs::verify` rejects a stale `Proof::outputs_digest` before
    // handing the outputs to Π_RLC. Keep the digest computation inside the
    // Π_CCS verifier and surface the resulting wires so NIFS.V can append the
    // exact same value without hashing the output surface a second time.
    let output_message_hashes_start = builder.rows();
    builder.begin_encoding_stage(stage::OUTPUT_MESSAGE_HASHES);
    builder.begin_encoding_stage(stage::OUTPUT_MESSAGE_DIGEST);
    let output_digest_inputs: Vec<_> = output_wires
        .iter()
        .map(|output| PiCcsOutputMessageDigestInputs {
            y_ring: &output.y_ring,
            y_zcol: &output.y_zcol,
        })
        .collect();
    let output_digest_wires = enforce_pi_ccs_outputs_digest(
        builder,
        PiCcsOutputMessageProfile::new(k_total, t),
        &output_digest_inputs,
    )?;
    let output_claims_digest = output_digest_wires.digest;
    builder.begin_encoding_stage(stage::OUTPUT_MESSAGE_CLAIM);
    let claimed_output_digest: [Var; 4] = std::array::from_fn(|lane| builder.alloc(msg.outputs_digest[lane]));
    builder.begin_encoding_stage(stage::OUTPUT_MESSAGE_BINDING);
    for lane in 0..4 {
        enforce_var_eq(builder, output_claims_digest[lane], claimed_output_digest[lane]);
    }
    builder.record_row_family(stage::OUTPUT_MESSAGE_HASHES, output_message_hashes_start);

    // Surface the full output wire bundle so downstream Π_RLC.V / NIFS.V
    // composition can fold c.data, X, r, s_col, y_ring, ct, y_zcol without
    // re-allocating any wires.
    let outputs: Vec<SplitNcPiCcsOutputWires> = output_wires
        .into_iter()
        .map(|ow| SplitNcPiCcsOutputWires {
            c_d: ow.c_d,
            c_d_var: ow.c_d_var,
            c_kappa: ow.c_kappa,
            c_kappa_var: ow.c_kappa_var,
            c_data: ow.c_data,
            adv: ow.adv,
            x: ow.x,
            x_rows: ow.x_rows,
            x_rows_var: ow.x_rows_var,
            x_cols: ow.x_cols,
            x_cols_var: ow.x_cols_var,
            m_in: ow.m_in,
            m_in_var: ow.m_in_var,
            r: ow.r,
            s_col: ow.s_col,
            y_ring: ow.y_ring,
            ct: ow.ct,
            y_zcol: ow.y_zcol,
            fold_digest_fields: ow.fold_digest_fields,
        })
        .collect();

    // Snapshot fresh.x and running.c_data wires so F'-side composition can
    // consume them without re-walking the witness layout.
    let fresh_x: Vec<Vec<Var>> = fresh_wires.iter().map(|fw| fw.x.clone()).collect();
    let fresh_adv = fresh_wires.iter().map(|fw| fw.adv.clone()).collect();
    let running_c_data: Vec<Vec<Var>> = running_wires.iter().map(|rw| rw.c_data.clone()).collect();
    // Surface the full per-running-claim wire bundle so the decider's
    // CE-continuity gate can pin `prev.children == next.running` field
    // by field. Built from the same allocated wires the SplitNc verifier
    // already constrained via shape checks + digest absorbs.
    let running: Vec<SplitNcPiCcsOutputWires> = running_wires
        .into_iter()
        .map(|rw| SplitNcPiCcsOutputWires {
            c_d: rw.c_d,
            c_d_var: rw.c_d_var,
            c_kappa: rw.c_kappa,
            c_kappa_var: rw.c_kappa_var,
            c_data: rw.c_data,
            adv: rw.adv,
            x: rw.x,
            x_rows: rw.x_rows,
            x_rows_var: rw.x_rows_var,
            x_cols: rw.x_cols,
            x_cols_var: rw.x_cols_var,
            m_in: rw.m_in,
            m_in_var: rw.m_in_var,
            r: rw.r,
            s_col: rw.s_col,
            y_ring: rw.y_ring,
            ct: rw.ct,
            y_zcol: rw.y_zcol,
            fold_digest_fields: rw.fold_digest_fields,
        })
        .collect();
    let running_parent_authority = running_parent_authority_wires.map(|rw| SplitNcPiCcsOutputWires {
        c_d: rw.c_d,
        c_d_var: rw.c_d_var,
        c_kappa: rw.c_kappa,
        c_kappa_var: rw.c_kappa_var,
        c_data: rw.c_data,
        adv: rw.adv,
        x: rw.x,
        x_rows: rw.x_rows,
        x_rows_var: rw.x_rows_var,
        x_cols: rw.x_cols,
        x_cols_var: rw.x_cols_var,
        m_in: rw.m_in,
        m_in_var: rw.m_in_var,
        r: rw.r,
        s_col: rw.s_col,
        y_ring: rw.y_ring,
        ct: rw.ct,
        y_zcol: rw.y_zcol,
        fold_digest_fields: rw.fold_digest_fields,
    });

    Ok(SplitNcPiCcsVDerived {
        r_prime: fe.r_prime,
        s_col_prime: nc.s_col_prime,
        outputs,
        output_claims_digest,
        output_message_preimage: output_digest_wires.preimage,
        fresh_x,
        fresh_adv,
        running_c_data,
        running,
        running_parent_authority,
        running_acc_digest,
    })
}

// ── Private helpers ───────────────────────────────────────────────────────

fn enforce_running_parent_authority_consistency(
    builder: &mut R1csBuilder,
    cfg: &SplitNcPiCcsVConfig<'_>,
    children: &[CeClaimWires],
    parent: Option<&CeClaimWires>,
) -> Result<(), Error> {
    match (children.is_empty(), parent) {
        (true, None) => return Ok(()),
        (true, Some(_)) => {
            return Err(Error::Shape(
                "empty running accumulator carried a parent authority".into(),
            ))
        }
        (false, None) => {
            return Err(Error::Shape(
                "non-empty running accumulator missing parent authority".into(),
            ))
        }
        (false, Some(_)) => {}
    }

    let wires = DecInputWires {
        parent: as_dec_claim_wires(parent.expect("non-empty branch checked above")),
        children: children.iter().map(as_dec_claim_wires).collect(),
    };
    enforce_split_nc_d_pad_shape(&wires, cfg.structure.t(), 1usize << cfg.ell_d)
        .map_err(|error| Error::Shape(format!("running parent Pi_DEC shape: {error}")))?;
    enforce_dec_v_strict(builder, cfg.params, &wires)
        .map_err(|error| Error::Shape(format!("running parent Pi_DEC: {error}")))
}

fn as_dec_claim_wires(claim: &CeClaimWires) -> DecCeClaimWires {
    DecCeClaimWires {
        c_data: claim.c_data.clone(),
        c_d: claim.c_d,
        c_d_var: claim.c_d_var,
        c_kappa: claim.c_kappa,
        c_kappa_var: claim.c_kappa_var,
        adv: claim.adv.clone(),
        x: claim.x.clone(),
        x_rows: claim.x_rows,
        x_rows_var: claim.x_rows_var,
        x_cols: claim.x_cols,
        x_cols_var: claim.x_cols_var,
        m_in: claim.m_in,
        m_in_var: claim.m_in_var,
        aux_openings_len: 0,
        c_step_coords_len: 0,
        u_offset: 0,
        u_len: 0,
        y_ring: claim
            .y_ring
            .iter()
            .map(|row| row.iter().flat_map(|value| [value.c0, value.c1]).collect())
            .collect(),
        y_ring_lanes: claim.y_ring.first().map_or(0, Vec::len),
        ct: claim.ct.clone(),
        r: claim.r.clone(),
        s_col: claim.s_col.clone(),
        y_zcol: claim
            .y_zcol
            .iter()
            .flat_map(|value| [value.c0, value.c1])
            .collect(),
        y_zcol_lanes: claim.y_zcol.len(),
        fold_digest_fields: claim.fold_digest_fields,
    }
}

/// Native-mirror shape check for one fresh CCS claim. Catches kappa /
/// commitment-data length / m_in / public-input-length mismatches before
/// the verifier reaches indexing-heavy gadgets.
fn validate_fresh_shape(cfg: &SplitNcPiCcsVConfig<'_>, idx: usize, f: &CcsClaim) -> Result<(), Error> {
    let kappa = cfg.params.kappa() as usize;
    if f.m_in > cfg.structure.m() {
        return Err(Error::Shape(format!(
            "fresh[{idx}].m_in ({}) > structure.m ({})",
            f.m_in,
            cfg.structure.m()
        )));
    }
    if f.x.len() != f.m_in {
        return Err(Error::Shape(format!(
            "fresh[{idx}].x.len ({}) != m_in ({})",
            f.x.len(),
            f.m_in
        )));
    }
    if f.c.d != D {
        return Err(Error::Shape(format!("fresh[{idx}].c.d ({}) != D ({})", f.c.d, D)));
    }
    if f.c.kappa != kappa {
        return Err(Error::Shape(format!(
            "fresh[{idx}].c.kappa ({}) != params.kappa ({})",
            f.c.kappa, kappa
        )));
    }
    if f.c.data.len() != D * kappa {
        return Err(Error::Shape(format!(
            "fresh[{idx}].c.data.len ({}) != D*kappa ({})",
            f.c.data.len(),
            D * kappa
        )));
    }
    validate_adv_shape(f.adv.as_ref(), D, kappa, &format!("fresh[{idx}]")).map_err(Error::Shape)?;
    Ok(())
}

/// The checked Π_RLC parent retains a fixed-width `y_zcol` view for the
/// current verifier, but that view is not yet semantic accumulator authority.
fn validate_running_parent_authority_shape(cfg: &SplitNcPiCcsVConfig<'_>, ce: &CeClaim) -> Result<(), Error> {
    validate_ce_shape_without_y_zcol(cfg, "running_parent_authority", ce)
}

fn validate_ce_shape_without_y_zcol(cfg: &SplitNcPiCcsVConfig<'_>, label: &str, ce: &CeClaim) -> Result<(), Error> {
    let kappa = cfg.params.kappa() as usize;
    let d_pad = 1usize << cfg.ell_d;

    if ce.c.d != D {
        return Err(Error::Shape(format!("{label}.c.d ({}) != D ({D})", ce.c.d)));
    }
    if ce.c.kappa != kappa {
        return Err(Error::Shape(format!(
            "{label}.c.kappa ({}) != params.kappa ({kappa})",
            ce.c.kappa
        )));
    }
    if ce.c.data.len() != D * kappa {
        return Err(Error::Shape(format!(
            "{label}.c.data.len ({}) != D*kappa ({})",
            ce.c.data.len(),
            D * kappa
        )));
    }
    validate_adv_shape(ce.adv.as_ref(), D, kappa, label).map_err(Error::Shape)?;
    if ce.m_in > cfg.structure.m() {
        return Err(Error::Shape(format!(
            "{label}.m_in ({}) > structure.m ({})",
            ce.m_in,
            cfg.structure.m()
        )));
    }
    if ce.X.rows() != D || ce.X.cols() != ce.m_in {
        return Err(Error::Shape(format!(
            "{label}.X shape ({}×{}) != ({D}×{})",
            ce.X.rows(),
            ce.X.cols(),
            ce.m_in
        )));
    }
    validate_ce_common_shape(cfg, label, ce, d_pad)
}

/// Output CE invariants that don't depend on whether the output is a fresh
/// or running slot — kappa / X.rows / r / s_col / y_ring / y_zcol shapes.
/// The X.cols check is in `bind_outputs_to_inputs` where it can compare
/// against either fresh.m_in or running.X.cols.
fn validate_output_ce_shape(cfg: &SplitNcPiCcsVConfig<'_>, label: &str, ce: &CeClaim) -> Result<(), Error> {
    let kappa = cfg.params.kappa() as usize;
    let d_pad = 1usize << cfg.ell_d;

    if ce.c.d != D {
        return Err(Error::Shape(format!("{label}.c.d ({}) != D ({D})", ce.c.d)));
    }
    if ce.c.kappa != kappa {
        return Err(Error::Shape(format!(
            "{label}.c.kappa ({}) != params.kappa ({kappa})",
            ce.c.kappa
        )));
    }
    if ce.c.data.len() != D * kappa {
        return Err(Error::Shape(format!(
            "{label}.c.data.len ({}) != D*kappa ({})",
            ce.c.data.len(),
            D * kappa
        )));
    }
    validate_adv_shape(ce.adv.as_ref(), D, kappa, label).map_err(Error::Shape)?;
    if ce.m_in > cfg.structure.m() {
        return Err(Error::Shape(format!(
            "{label}.m_in ({}) > structure.m ({})",
            ce.m_in,
            cfg.structure.m()
        )));
    }
    if ce.X.rows() != D {
        return Err(Error::Shape(format!("{label}.X.rows ({}) != D ({D})", ce.X.rows())));
    }
    validate_ce_common_shape(cfg, label, ce, d_pad)?;
    validate_y_zcol_shape(label, ce, d_pad)
}

fn validate_ce_common_shape(
    cfg: &SplitNcPiCcsVConfig<'_>,
    label: &str,
    ce: &CeClaim,
    d_pad: usize,
) -> Result<(), Error> {
    if ce.r.len() != cfg.ell_n {
        return Err(Error::Shape(format!(
            "{label}.r.len ({}) != ell_n ({})",
            ce.r.len(),
            cfg.ell_n
        )));
    }
    if ce.s_col.len() != cfg.ell_m {
        return Err(Error::Shape(format!(
            "{label}.s_col.len ({}) != ell_m ({})",
            ce.s_col.len(),
            cfg.ell_m
        )));
    }
    if ce.y_ring.len() != cfg.structure.t() {
        return Err(Error::Shape(format!(
            "{label}.y_ring.len ({}) != structure.t ({})",
            ce.y_ring.len(),
            cfg.structure.t()
        )));
    }
    if ce.ct.len() != cfg.structure.t() {
        return Err(Error::Shape(format!(
            "{label}.ct.len ({}) != structure.t ({})",
            ce.ct.len(),
            cfg.structure.t()
        )));
    }
    for (j, row) in ce.y_ring.iter().enumerate() {
        if row.len() != d_pad {
            return Err(Error::Shape(format!(
                "{label}.y_ring[{j}].len ({}) != d_pad ({})",
                row.len(),
                d_pad
            )));
        }
    }
    validate_ce_sidecars(label, ce)
}

fn validate_y_zcol_shape(label: &str, ce: &CeClaim, d_pad: usize) -> Result<(), Error> {
    if ce.y_zcol.len() != d_pad {
        return Err(Error::Shape(format!(
            "{label}.y_zcol.len ({}) != d_pad ({})",
            ce.y_zcol.len(),
            d_pad
        )));
    }
    Ok(())
}

fn validate_ce_sidecars(label: &str, ce: &CeClaim) -> Result<(), Error> {
    if !ce.aux_openings.is_empty() {
        return Err(Error::Shape(format!(
            "{label}.aux_openings.len ({}) != 0 for clean SplitNc circuit",
            ce.aux_openings.len()
        )));
    }
    if !ce.c_step_coords.is_empty() || ce.u_offset != 0 || ce.u_len != 0 {
        return Err(Error::Shape(format!(
            "{label} carries unsupported Pattern-A fields (c_step_coords.len={}, u_offset={}, u_len={})",
            ce.c_step_coords.len(),
            ce.u_offset,
            ce.u_len
        )));
    }
    Ok(())
}

fn enforce_kvar_eq(builder: &mut R1csBuilder, a: KVar, b: KVar) {
    builder.enforce_eq(&Lc::from_var(a.c0), &Lc::from_var(b.c0));
    builder.enforce_eq(&Lc::from_var(a.c1), &Lc::from_var(b.c1));
}

fn enforce_var_eq(builder: &mut R1csBuilder, a: Var, b: Var) {
    builder.enforce_eq(&Lc::from_var(a), &Lc::from_var(b));
}

fn enforce_var_usize_eq(builder: &mut R1csBuilder, var: Var, expected: usize) {
    builder.enforce_eq(&Lc::from_var(var), &Lc::from_const(F::from_u64(expected as u64)));
}

fn alloc_usize(builder: &mut R1csBuilder, value: usize) -> Var {
    let v = builder.alloc(F::from_u64(value as u64));
    builder.enforce_eq(&Lc::from_var(v), &Lc::from_const(F::from_u64(value as u64)));
    v
}

fn enforce_output_fold_digest_matches_header(
    builder: &mut R1csBuilder,
    output_wires: &[CeClaimWires],
    header_wires: [Var; 4],
) {
    for output in output_wires {
        for (output, expected) in output.fold_digest_fields.iter().zip(header_wires) {
            builder.enforce_eq(&Lc::from_var(*output), &Lc::from_var(expected));
        }
    }
}

/// Mirror the native verifier's optional FE-initial-sum check without making
/// the R1CS relation depend on whether the optional field is present.
///
/// `present = 1` enforces `claimed == expected`; `present = 0` accepts absence
/// and requires the otherwise-unused value wires to use a unique zero padding.
fn enforce_optional_k_equality(builder: &mut R1csBuilder, claimed: Option<K>, expected: KVar) {
    let present_value = if claimed.is_some() { F::ONE } else { F::ZERO };
    let present = builder.alloc(present_value);
    boolean::enforce_bit(builder, present);

    let claimed = alloc_k(builder, claimed.unwrap_or(K::ZERO));
    let mut diff_c0 = Lc::from_var(claimed.c0);
    diff_c0.add_term(expected.c0, -F::ONE);
    let mut diff_c1 = Lc::from_var(claimed.c1);
    diff_c1.add_term(expected.c1, -F::ONE);
    builder.enforce(&Lc::from_var(present), &diff_c0, &Lc::zero());
    builder.enforce(&Lc::from_var(present), &diff_c1, &Lc::zero());

    let mut absent = Lc::from_const(F::ONE);
    absent.add_term(present, -F::ONE);
    builder.enforce(&absent, &Lc::from_var(claimed.c0), &Lc::zero());
    builder.enforce(&absent, &Lc::from_var(claimed.c1), &Lc::zero());
}

fn enforce_ct_from_y_ring(builder: &mut R1csBuilder, label: &str, claim: &CeClaimWires) -> Result<(), Error> {
    if claim.ct.len() != claim.y_ring.len() {
        return Err(Error::Shape(format!(
            "{label}.ct.len ({}) != y_ring.len ({})",
            claim.ct.len(),
            claim.y_ring.len()
        )));
    }
    for (j, (ct, row)) in claim.ct.iter().zip(claim.y_ring.iter()).enumerate() {
        let y0 = row.first().copied().ok_or_else(|| {
            Error::Shape(format!(
                "{label}.y_ring[{j}] has no lane-0 constant term for ct binding"
            ))
        })?;
        enforce_kvar_eq(builder, *ct, y0);
    }
    Ok(())
}

fn enforce_y_ring_padding_zero(builder: &mut R1csBuilder, claim: &CeClaimWires) {
    for row in &claim.y_ring {
        for lane in row.iter().skip(D) {
            builder.enforce_eq(&Lc::from_var(lane.c0), &Lc::zero());
            builder.enforce_eq(&Lc::from_var(lane.c1), &Lc::zero());
        }
    }
}

fn enforce_y_zcol_padding_zero(builder: &mut R1csBuilder, claim: &CeClaimWires) {
    for lane in claim.y_zcol.iter().skip(D) {
        builder.enforce_eq(&Lc::from_var(lane.c0), &Lc::zero());
        builder.enforce_eq(&Lc::from_var(lane.c1), &Lc::zero());
    }
}

fn alloc_k(builder: &mut R1csBuilder, v: K) -> KVar {
    let [c0, c1] = v.as_coeffs();
    KVar::alloc(builder, c0, c1)
}

fn alloc_k_vec(builder: &mut R1csBuilder, xs: &[K]) -> Vec<KVar> {
    xs.iter().copied().map(|x| alloc_k(builder, x)).collect()
}

fn alloc_k_rows(builder: &mut R1csBuilder, rows: &[Vec<K>]) -> Vec<Vec<KVar>> {
    rows.iter().map(|r| alloc_k_vec(builder, r)).collect()
}

/// Decode a 32-byte digest (e.g. `fold_digest`) into 4 F lanes and allocate
/// each as a witness wire. The verifier doesn't pin these — the constraints
/// come from the digest gadget that absorbs them.
fn digest32_witness_fields(builder: &mut R1csBuilder, bytes: &[u8]) -> Result<[Var; 4], Error> {
    let fields = header_digest_bytes_to_fields(bytes)?;
    Ok(std::array::from_fn(|i| builder.alloc(fields[i])))
}

fn alloc_fresh_wires(builder: &mut R1csBuilder, fresh: &CcsClaim) -> CcsClaimWires {
    CcsClaimWires {
        c_d: fresh.c.d,
        c_d_var: alloc_usize(builder, fresh.c.d),
        c_kappa: fresh.c.kappa,
        c_kappa_var: alloc_usize(builder, fresh.c.kappa),
        c_data: builder.alloc_vec(&fresh.c.data),
        adv: alloc_adv(builder, fresh.adv.as_ref()),
        x: builder.alloc_vec(&fresh.x),
        m_in: fresh.m_in,
        m_in_var: alloc_usize(builder, fresh.m_in),
    }
}

/// Allocate wires for one CE claim. `x` is stored as a flat row-major
/// `Vec<Var>` of length `x_rows * x_cols`; the SuperNeo-packed view is
/// derived on demand via [`CeClaimWires::x_packed`].
fn alloc_ce_wires(builder: &mut R1csBuilder, ce: &CeClaim) -> Result<CeClaimWires, Error> {
    alloc_ce_wires_from_y_zcol(builder, ce, &ce.y_zcol)
}

/// Allocate the paper-level CE core consumed by strict Π_DEC and the exact
/// Construction-2 child handle. The optimized `y_zcol` sidecar is omitted
/// until a verifier-owned source relation is proved for it.
fn alloc_ce_wires_without_y_zcol(builder: &mut R1csBuilder, ce: &CeClaim) -> Result<CeClaimWires, Error> {
    alloc_ce_wires_from_y_zcol(builder, ce, &[])
}

/// Allocate the current fixed-width parent `y_zcol` view. Native Π_DEC.V does
/// not treat it as accumulator authority; canonicalizing the shape only keeps
/// the verifier relation fixed while the delayed-NC bridge remains open.
fn alloc_ce_wires_with_canonical_y_zcol(
    builder: &mut R1csBuilder,
    ce: &CeClaim,
    lanes: usize,
) -> Result<CeClaimWires, Error> {
    let canonical: Vec<K> = (0..lanes)
        .map(|lane| ce.y_zcol.get(lane).copied().unwrap_or(K::ZERO))
        .collect();
    alloc_ce_wires_from_y_zcol(builder, ce, &canonical)
}

fn alloc_ce_wires_from_y_zcol(builder: &mut R1csBuilder, ce: &CeClaim, y_zcol: &[K]) -> Result<CeClaimWires, Error> {
    let mut x: Vec<Var> = Vec::with_capacity(ce.X.rows() * ce.X.cols());
    let active_cols = crate::paper::relations::superneo_public_x_cols(ce.m_in);
    let inactive_nonzero = (0..ce.X.rows()).any(|r| (active_cols..ce.X.cols()).any(|c| ce.X[(r, c)] != F::ZERO));
    let inactive_zero = builder.alloc(if inactive_nonzero { F::ONE } else { F::ZERO });
    builder.enforce_eq(&Lc::from_var(inactive_zero), &Lc::zero());
    for r in 0..ce.X.rows() {
        for c in 0..ce.X.cols() {
            x.push(if c < active_cols {
                builder.alloc(ce.X[(r, c)])
            } else {
                inactive_zero
            });
        }
    }
    Ok(CeClaimWires {
        c_d: ce.c.d,
        c_d_var: alloc_usize(builder, ce.c.d),
        c_kappa: ce.c.kappa,
        c_kappa_var: alloc_usize(builder, ce.c.kappa),
        c_data: builder.alloc_vec(&ce.c.data),
        adv: alloc_adv(builder, ce.adv.as_ref()),
        x,
        x_rows: ce.X.rows(),
        x_rows_var: alloc_usize(builder, ce.X.rows()),
        x_cols: ce.X.cols(),
        x_cols_var: alloc_usize(builder, ce.X.cols()),
        r: alloc_k_vec(builder, &ce.r),
        s_col: alloc_k_vec(builder, &ce.s_col),
        y_ring: alloc_k_rows(builder, &ce.y_ring),
        ct: alloc_k_vec(builder, &ce.ct),
        y_zcol: alloc_k_vec(builder, y_zcol),
        m_in: ce.m_in,
        m_in_var: alloc_usize(builder, ce.m_in),
        fold_digest_fields: digest32_witness_fields(builder, &ce.fold_digest)?,
    })
}

fn enforce_unique_zero_wires(builder: &mut R1csBuilder, wires: impl Iterator<Item = Var>) {
    let mut constrained = std::collections::HashSet::new();
    for wire in wires {
        if constrained.insert(wire.col()) {
            builder.enforce_eq(&Lc::from_var(wire), &Lc::zero());
        }
    }
}

/// Mirror of native `validate_me_outputs_against_inputs`, but wire-to-wire
/// (no `wire == constant`). Constraints emitted per output:
/// - `output.r == r_prime`
/// - `output.s_col == s_col_prime`
/// - `output.y_zcol.len == d_pad` (shape, no silent truncation)
/// - Fresh outputs (idx < k_mcs):
///     - `output.c_data == fresh.c_data` (lane-wise)
///     - `output.X[(c % D, c / D)] == fresh.x[c]` for `c ∈ 0..m_in`
/// - Running outputs (idx ≥ k_mcs):
///     - `output.c_data == running.c_data`
///     - `output.X[r, c] == running.X[r, c]` (full row-major)
fn bind_outputs_to_inputs(
    builder: &mut R1csBuilder,
    fresh_wires: &[CcsClaimWires],
    running_wires: &[CeClaimWires],
    output_wires: &[CeClaimWires],
    r_prime: &[KVar],
    s_col_prime: &[KVar],
    d_pad: usize,
) -> Result<(), Error> {
    let k_mcs = fresh_wires.len();

    for (idx, ow) in output_wires.iter().enumerate() {
        // r / s_col bind to FE/NC challenge halves.
        if ow.r.len() != r_prime.len() {
            return Err(Error::Shape(format!(
                "output[{idx}].r.len ({}) != r_prime.len ({})",
                ow.r.len(),
                r_prime.len()
            )));
        }
        if ow.s_col.len() != s_col_prime.len() {
            return Err(Error::Shape(format!(
                "output[{idx}].s_col.len ({}) != s_col_prime.len ({})",
                ow.s_col.len(),
                s_col_prime.len()
            )));
        }
        for (i, v) in ow.r.iter().enumerate() {
            enforce_kvar_eq(builder, *v, r_prime[i]);
        }
        for (i, v) in ow.s_col.iter().enumerate() {
            enforce_kvar_eq(builder, *v, s_col_prime[i]);
        }

        // y_zcol shape (native asserts y_zcol.len == d_pad in `validate_me_outputs_against_inputs`).
        if ow.y_zcol.len() != d_pad {
            return Err(Error::Shape(format!(
                "output[{idx}].y_zcol.len ({}) != d_pad ({})",
                ow.y_zcol.len(),
                d_pad
            )));
        }
        let active_x_cols = crate::paper::relations::superneo_public_x_cols(ow.m_in);
        if active_x_cols > ow.x_cols {
            return Err(Error::Shape(format!(
                "output[{idx}].active_x_cols ({active_x_cols}) > X.cols ({})",
                ow.x_cols
            )));
        }
        enforce_unique_zero_wires(
            builder,
            (0..ow.x_rows).flat_map(|row| (active_x_cols..ow.x_cols).map(move |col| ow.x[row * ow.x_cols + col])),
        );

        if idx < k_mcs {
            let fw = &fresh_wires[idx];

            // m_in must match native `validate_me_outputs_against_inputs`
            // (`out.m_in == fresh.m_in`). m_in is structural — checked here
            // at gadget-emit time rather than as a wire constraint.
            if ow.m_in != fw.m_in {
                return Err(Error::Shape(format!(
                    "fresh output[{idx}].m_in ({}) != fresh.m_in ({})",
                    ow.m_in, fw.m_in
                )));
            }
            enforce_var_eq(builder, ow.c_d_var, fw.c_d_var);
            enforce_var_eq(builder, ow.c_kappa_var, fw.c_kappa_var);
            enforce_var_usize_eq(builder, ow.x_rows_var, D);
            enforce_var_eq(builder, ow.x_cols_var, fw.m_in_var);
            enforce_var_eq(builder, ow.m_in_var, fw.m_in_var);

            // commitment data length + lane-wise equality.
            if ow.c_data.len() != fw.c_data.len() {
                return Err(Error::Shape(format!(
                    "fresh output[{idx}].c_data.len ({}) != fresh.c_data.len ({})",
                    ow.c_data.len(),
                    fw.c_data.len()
                )));
            }
            for (a, b) in ow.c_data.iter().zip(fw.c_data.iter()) {
                enforce_var_eq(builder, *a, *b);
            }
            enforce_adv_equality(
                builder,
                ow.adv.as_ref(),
                fw.adv.as_ref(),
                &format!("fresh output[{idx}]"),
            )
            .map_err(Error::Shape)?;

            // X shape: must be D × m_in for SuperNeo packing.
            if ow.x_rows != D || ow.x_cols != fw.m_in {
                return Err(Error::Shape(format!(
                    "fresh output[{idx}].X shape ({}×{}) != expected ({D}×{})",
                    ow.x_rows, ow.x_cols, fw.m_in
                )));
            }

            // Public X lanes inherit from fresh.x[c] for c ∈ 0..m_in.
            // Other lanes are owned by L_in(z) and may be non-zero after
            // ring-linear folding — don't constrain them.
            for c in 0..fw.m_in {
                let row = c % D;
                let col = c / D;
                let out_wire = ow.x[row * ow.x_cols + col];
                let fresh_wire = fw.x[c];
                enforce_var_eq(builder, out_wire, fresh_wire);
            }
        } else {
            let rw = &running_wires[idx - k_mcs];

            // m_in must match native `validate_me_outputs_against_inputs`
            // (`out.m_in == running.m_in`). Structural, gadget-emit time.
            if ow.m_in != rw.m_in {
                return Err(Error::Shape(format!(
                    "running output[{idx}].m_in ({}) != running.m_in ({})",
                    ow.m_in, rw.m_in
                )));
            }
            enforce_var_eq(builder, ow.c_d_var, rw.c_d_var);
            enforce_var_eq(builder, ow.c_kappa_var, rw.c_kappa_var);
            enforce_var_eq(builder, ow.x_rows_var, rw.x_rows_var);
            enforce_var_eq(builder, ow.x_cols_var, rw.x_cols_var);
            enforce_var_eq(builder, ow.m_in_var, rw.m_in_var);

            if ow.c_data.len() != rw.c_data.len() {
                return Err(Error::Shape(format!(
                    "running output[{idx}].c_data.len ({}) != running.c_data.len ({})",
                    ow.c_data.len(),
                    rw.c_data.len()
                )));
            }
            for (a, b) in ow.c_data.iter().zip(rw.c_data.iter()) {
                enforce_var_eq(builder, *a, *b);
            }
            enforce_adv_equality(
                builder,
                ow.adv.as_ref(),
                rw.adv.as_ref(),
                &format!("running output[{idx}]"),
            )
            .map_err(Error::Shape)?;

            if ow.x_rows != rw.x_rows || ow.x_cols != rw.x_cols {
                return Err(Error::Shape(format!(
                    "running output[{idx}].X shape ({}×{}) != running.X shape ({}×{})",
                    ow.x_rows, ow.x_cols, rw.x_rows, rw.x_cols
                )));
            }
            // Strict Pi_DEC above proves every active running-X coordinate is
            // in {-1, 0, 1}. These output coordinates are constrained equal
            // to those same wires, so the Road A compiler may keep their
            // centered representation instead of expanding them to 64 bits.
            for row in 0..ow.x_rows {
                for col in 0..active_x_cols {
                    builder.record_centered_unit(ow.x[row * ow.x_cols + col]);
                }
            }
            // Full X matrix is inherited from the running input (every lane).
            for (a, b) in ow.x.iter().zip(rw.x.iter()) {
                enforce_var_eq(builder, *a, *b);
            }
        }
    }

    Ok(())
}
