//! Image filling for one R1CS-backed encoded `F'` step.
//!
//! Owns: assignment bit encoding, R1CS-specific image fill, and satisfaction
//! against the cached [`FPrimeStructure`].
//!
//! Does not own: structure construction, source-R1CS validation, chain state, or
//! folding proof verification.
//!
//! Emits constraints: no. It reuses a verifier-owned cached structure.
//!
//! Authority boundary: assignment bits and trace images are prover data; the
//! cached structure and its verified satisfaction determine acceptance.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Assignment bits | [`assignment_to_bits`] | no | Canonical field assignment |
//! | Image fill | [`encode_r1cs_f_prime_step`] | no | Typed encoder inputs |
//! | Relation check | cached [`FPrimeStructure`] | no | Verifier preprocessing |

use std::sync::Arc;

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use crate::engine::ccs_native::poseidon2_transcript::SpongeTraceImage;
use crate::frontends::f_prime::encoder::{EncodedFPrimeStep, NifsPayloadInput};
use crate::frontends::f_prime::image::{FPrimeImage, KMulView, StateIn, StateOut};
use crate::frontends::f_prime::recursive_plan::{build_recursive_step_image_config, RecursiveStepImagePlan};
use crate::frontends::f_prime::structure::FPrimeStructure;
use crate::paper::f_prime::poseidon_trace::PoseidonTraceImage;
use crate::paper::f_prime::ring_action_trace::RingActionTraceImage;

/// One R1CS encoder input. Same shape as the Fibonacci encoder input
/// except `assignment_bits` carries the encoded R1CS variable
/// assignment instead of Fibonacci's app-private carries.
pub struct R1csEncoderInput {
    pub plan: RecursiveStepImagePlan,
    pub boundary_bits: Vec<F>,
    pub state_in: StateIn,
    pub state_out: StateOut,
    pub chunk_digest: [F; 4],
    /// Encoded R1CS variable assignment `z`. Legacy plans use 64
    /// little-endian bits per variable. Typed plans use the verifier-owned
    /// bounded width for each variable.
    pub assignment_bits: Vec<F>,
    pub is_base: bool,
    pub nifs_payloads: Vec<NifsPayloadInput>,
    pub kmul_views: Vec<KMulView>,
    pub ring_action_pairs: Vec<RingActionTraceImage>,
    pub one_shot_traces: Vec<PoseidonTraceImage>,
    pub sponge_trace: Option<SpongeTraceImage>,
}

/// Decompose an R1CS assignment `z` into the bit-string the encoder
/// expects: 64 little-endian bits per variable, concatenated in
/// variable order.
pub fn assignment_to_bits(assignment: &[F]) -> Vec<F> {
    let mut bits = Vec::with_capacity(assignment.len() * POSEIDON2_GOLDILOCKS_BITS);
    for v in assignment {
        let value = v.as_canonical_u64();
        for i in 0..POSEIDON2_GOLDILOCKS_BITS {
            bits.push(if ((value >> i) & 1) == 1 { F::ONE } else { F::ZERO });
        }
    }
    bits
}

/// Encode one `enc(F')` step that includes the in-circuit R1CS rows.
///
/// `structure` is the verifier-owned cached R1CS-F' structure (see
/// [`crate::frontends::r1cs_f_prime::R1csFPrimePreprocessing::structure`]).
/// The encoder reuses it as-is — it does **not** rebuild the structure
/// per step — and the returned [`EncodedFPrimeStep`] shares the same
/// [`Arc`] so downstream consumers (`build_instance`, lifecycle folds)
/// can fast-path the digest check via pointer equality.
pub fn encode_r1cs_f_prime_step(input: R1csEncoderInput, structure: Arc<FPrimeStructure>) -> EncodedFPrimeStep {
    let config = build_recursive_step_image_config(&input.plan);
    let layout = structure.layout.clone();

    // ── Strict input-shape gate ──────────────────────────────────────
    assert_eq!(
        input.boundary_bits.len(),
        input.plan.boundary_bits,
        "boundary bits must match plan.boundary_bits"
    );
    assert_eq!(
        input.assignment_bits.len(),
        layout.app_private.bits,
        "assignment_bits must match cached structure's app_private region"
    );
    assert_eq!(
        input.plan.limbs,
        layout.app_private.bits + 1,
        "plan.limbs must equal app_private.bits + 1"
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

    let mut image = FPrimeImage::new(layout);

    image.fill_boundary(&input.boundary_bits);
    image.fill_state_in(&input.state_in);
    image.fill_state_out(&input.state_out);
    image.fill_chunk_digest(input.chunk_digest);
    image.fill_app_private(&input.assignment_bits);
    image.fill_is_base(input.is_base);

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

    #[cfg(feature = "perf-timers")]
    let t_witness = std::time::Instant::now();
    let witness = structure.extend_witness_from_image(&image);
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[r1cs-encode] extend witness               {:>7.2}s",
        t_witness.elapsed().as_secs_f64()
    );

    #[cfg(feature = "perf-timers")]
    let t_satisfied = std::time::Instant::now();
    assert!(
        structure.is_satisfied(&witness),
        "encoded R1CS F' step must satisfy its structure; first failing row: {:?}",
        structure.first_unsatisfied_row(&witness)
    );
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[r1cs-encode] structure.is_satisfied       {:>7.2}s",
        t_satisfied.elapsed().as_secs_f64()
    );

    EncodedFPrimeStep {
        image,
        structure,
        witness,
    }
}
