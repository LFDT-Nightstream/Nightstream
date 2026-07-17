//! Materialization of one encoded `F'` step from a fixed image plan.
//!
//! Owns: filling the source image, constructing the matching CCS structure, and
//! returning the instance plus satisfying assignment.
//!
//! Does not own: plan semantics, application-specific witness validation, or
//! NIFS proof authority.
//!
//! Emits constraints: yes, indirectly through [`build_f_prime_structure`]; this
//! file owns orchestration rather than row formulas.
//!
//! Authority boundary: encoder output is prover data until the returned
//! structure and instance are verified; successful local assembly is not proof
//! authority.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Image fill | [`encode_f_prime_step`] | no | Caller-supplied typed inputs |
//! | CCS structure | [`build_f_prime_structure`] | yes | Fixed image plan |
//! | Instance and witness | [`EncodedFPrimeStep`] | no | Verified CCS relation |

use std::sync::Arc;

use neo_ajtai::AjtaiSModule;
use neo_math::F;

use crate::engine::ccs_native::poseidon2_transcript::SpongeTraceImage;
use crate::frontends::f_prime::image::{
    FPrimeImage, FPrimeImageLayout, KMulView, NifsCcsClaimView, NifsCeClaimView, StateIn, StateOut,
};
use crate::frontends::f_prime::recursive_plan::{build_recursive_step_image_config, RecursiveStepImagePlan};
use crate::frontends::f_prime::structure::{build_f_prime_structure, FPrimeStructure};
use crate::paper::f_prime::poseidon_trace::PoseidonTraceImage;
use crate::paper::f_prime::ring_action_trace::RingActionTraceImage;
use crate::paper::params::Params;
use crate::paper::relations::{CcsInstance, RelationError};

/// One NIFS-payload input. The encoder picks the right
/// `fill_nifs_{ccs,ce}_claim_at` method per variant.
#[derive(Clone, Debug)]
pub enum NifsPayloadInput {
    Ccs(NifsCcsClaimView),
    Ce(NifsCeClaimView),
}

/// All data the encoder needs for one Fibonacci recursive step.
///
/// Everything here is "real prover output": no test-local stand-ins
/// remain except for explicit `Constant`s in the plan's preimage
/// sources (domain tags and legacy non-unified `child_count`).
pub struct FPrimeStepInput {
    /// The image config plan. Drives layout sizing and which Poseidon
    /// transition enforcements / digest bindings are emitted.
    pub plan: RecursiveStepImagePlan,
    /// Raw bits for the boundary region. Must have length
    /// `plan.boundary_bits`. Caller writes the public-x_out lanes,
    /// prior-x_out, counters, pc, etc. into this buffer.
    pub boundary_bits: Vec<F>,
    /// State-in digests (vk_fs, structure, z_0, z_i_in,
    /// acc_digest_in, public_trace_in).
    pub state_in: StateIn,
    /// State-out digests + counters (must agree with the traces).
    pub state_out: StateOut,
    /// Chunk digest for this step.
    pub chunk_digest: [F; 4],
    /// Fibonacci app-private carry bits.
    pub app_private_carries: Vec<F>,
    /// `is_base` lane value: `true` marks this step as the base case
    /// (no prior fold), `false` as a recursive step. Under the unified
    /// plan the structure's selector reads this bit to pick between the
    /// base accumulator trace's digest and the recursive accumulator
    /// trace's digest for `state_out.new_acc_digest`. Under the legacy
    /// plan this lane is reserved but algebraically unconstrained; pass
    /// `false` (zero).
    pub is_base: bool,
    /// NIFS payloads (CcsClaim / CeClaim views), in fill order.
    ///
    /// Legacy non-unified plans fill this from `plan.nifs_payload_shapes`.
    /// Unified plans use delayed accumulator-handle binding and do not
    /// reserve source-image payload lanes, so this is empty even though
    /// the plan still keeps CE-shape metadata for compiler validation.
    pub nifs_payloads: Vec<NifsPayloadInput>,
    /// K-mul Karatsuba views, one per K-mul invocation.
    pub kmul_views: Vec<KMulView>,
    /// Ring-action pair traces, one per pair.
    pub ring_action_pairs: Vec<RingActionTraceImage>,
    /// One-shot Poseidon traces, indexed by the same `one_shot_index`
    /// the plan's enforcements / bindings reference.
    pub one_shot_traces: Vec<PoseidonTraceImage>,
    /// F' sponge transcript trace, if the step's plan asks for one.
    pub sponge_trace: Option<SpongeTraceImage>,
}

/// Encoder output: image + structure + satisfying witness.
///
/// `structure` is shared via [`Arc`] so frontends can cache one
/// `FPrimeStructure` per preprocessing and reuse it across every
/// encoded step in a chain instead of rebuilding (~1.5M sparse
/// constraint rows for SHA-256-sized R1CS shapes).
#[derive(Debug)]
pub struct EncodedFPrimeStep {
    pub image: FPrimeImage,
    pub structure: Arc<FPrimeStructure>,
    pub witness: Vec<F>,
}

/// Build one `enc(F'_i)` instance from a real recursive step.
///
/// Strict on input shape: every region count must match the plan/layout
/// before any region is filled. A silently-unfilled region would leave
/// reserved bits at zero and could pass downstream checks for the wrong
/// reason. Panics with a row index if the resulting witness fails to
/// satisfy the structure — that's a prover/encoder bug, not a soundness
/// gate. External fault-injection tests should tamper the returned
/// witness after this function returns.
pub fn encode_f_prime_step(input: FPrimeStepInput) -> EncodedFPrimeStep {
    let config = build_recursive_step_image_config(&input.plan);
    let structure = Arc::new(build_f_prime_structure(FPrimeImageLayout::new(config)));
    encode_f_prime_step_with_structure(input, structure)
}

/// Encode one `enc(F'_i)` step using a verifier-owned cached structure.
///
/// This is the fixed-`pc` path HyperNova Construction 2 wants: all steps
/// in one chain share the same F' structure, so callers that already
/// have the cached structure should not rebuild or re-digest it per
/// step. The config assertion keeps the cached structure tied to the
/// input plan before any witness bits are filled.
pub fn encode_f_prime_step_with_structure(
    input: FPrimeStepInput,
    structure: Arc<FPrimeStructure>,
) -> EncodedFPrimeStep {
    let config = build_recursive_step_image_config(&input.plan);
    assert_eq!(
        structure.layout.config, config,
        "cached F' structure config must match input plan"
    );

    // ── Strict input-shape gate ──────────────────────────────────────
    assert_eq!(
        input.boundary_bits.len(),
        input.plan.boundary_bits,
        "boundary bits must match plan.boundary_bits"
    );
    assert_eq!(
        input.app_private_carries.len(),
        input.plan.limbs.saturating_sub(1),
        "app-private carries must have length plan.limbs - 1"
    );
    assert_eq!(
        input.nifs_payloads.len(),
        config.nifs_payload_shapes.len(),
        "NIFS payload count must match source-image config"
    );
    assert_eq!(
        input.kmul_views.len(),
        input.plan.kmul_count,
        "K-mul view count must match plan.kmul_count"
    );
    assert_eq!(
        input.ring_action_pairs.len(),
        input.plan.ring_action_pair_count,
        "ring-action pair count must match plan.ring_action_pair_count"
    );
    assert_eq!(
        input.one_shot_traces.len(),
        config.poseidon_one_shot_preimage_lens.len(),
        "one-shot Poseidon trace count must match the plan's one-shot config"
    );
    assert_eq!(
        input.sponge_trace.is_some(),
        config.sponge_transcript_permutes > 0,
        "sponge_trace presence must match sponge_transcript_permutes > 0"
    );

    let layout = structure.layout.clone();
    let mut image = FPrimeImage::new(layout.clone());

    image.fill_boundary(&input.boundary_bits);
    image.fill_state_in(&input.state_in);
    image.fill_state_out(&input.state_out);
    image.fill_chunk_digest(input.chunk_digest);
    image.fill_app_private(&input.app_private_carries);
    image.fill_is_base(input.is_base);

    // NIFS payloads in fill order. Each fill returns the next region-
    // relative offset; the running offset must cover the entire
    // `nifs_payloads` region by the time we're done.
    let mut nifs_offset = 0usize;
    for payload in &input.nifs_payloads {
        nifs_offset = match payload {
            NifsPayloadInput::Ccs(view) => image.fill_nifs_ccs_claim_at(nifs_offset, view),
            NifsPayloadInput::Ce(view) => image.fill_nifs_ce_claim_at(nifs_offset, view),
        };
    }
    assert_eq!(
        nifs_offset, image.layout.nifs_payloads.bits,
        "NIFS payload fills must cover exactly the planned region"
    );

    image.fill_all_kmul(&input.kmul_views);

    for (idx, pair) in input.ring_action_pairs.iter().enumerate() {
        image.splice_ring_action_pair(idx, pair);
    }
    for (idx, trace) in input.one_shot_traces.iter().enumerate() {
        image.splice_one_shot_poseidon(idx, trace);
    }
    if let Some(trace) = &input.sponge_trace {
        image.splice_sponge_transcript(trace);
    }

    let witness = structure.extend_witness_from_image(&image);

    assert!(
        structure.is_satisfied(&witness),
        "encoded F' step must satisfy its structure; first failing row: {:?}",
        structure.first_unsatisfied_row(&witness)
    );

    EncodedFPrimeStep {
        image,
        structure,
        witness,
    }
}

impl EncodedFPrimeStep {
    /// Canonical CCS public-input length for this encoded F' instance:
    /// the constant slot `z[0] = 1` plus the boundary public bits
    /// (`enc_inst(x_out)` body). Everything past this index is private
    /// witness `w`.
    pub fn public_input_len(&self) -> usize {
        1 + self.image.layout.boundary.bits
    }

    /// Convert this encoded F' step into one foldable [`CcsInstance`].
    ///
    /// The witness is strict low-norm: exactly `image.values`, with
    /// `z[0] = 1` and low-norm image coordinates. This is the only
    /// boundary that should call [`CcsInstance::from_low_norm_assignment`]
    /// for F' steps — downstream code (NIFS, lifecycle) folds
    /// the returned instance directly.
    pub fn to_ccs_instance(
        &self,
        params: &Params,
        log: &AjtaiSModule,
        m_in: usize,
    ) -> Result<CcsInstance, RelationError> {
        debug_assert_eq!(self.witness, self.image.values);
        debug_assert_eq!(self.structure.ccs.m, self.witness.len());

        CcsInstance::from_low_norm_assignment(params, log, &self.structure.ccs, &self.witness, m_in)
    }

    /// Convert using the canonical [`public_input_len`](Self::public_input_len)
    /// split. Lifecycle callers should prefer this entry; tests that need
    /// to exercise a different `m_in` (e.g. degenerate `m_in = 1` shapes)
    /// can use [`to_ccs_instance`](Self::to_ccs_instance) directly.
    pub fn to_public_ccs_instance(&self, params: &Params, log: &AjtaiSModule) -> Result<CcsInstance, RelationError> {
        self.to_ccs_instance(params, log, self.public_input_len())
    }
}
