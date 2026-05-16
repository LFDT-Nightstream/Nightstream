//! R1CS F' step encoder.
//!
//! Mirrors [`crate::frontends::f_prime_shell::encoder::encode_f_prime_step`]
//! but builds an R1CS-aware [`FPrimeStructure`] for the final
//! satisfaction check, so the resulting encoded step also enforces every
//! R1CS constraint.
//!
//! The image region is the same; only the appended structure rows differ.
//! `app_private` is repurposed as the app-assignment region: callers pass
//! the bit-decomposed R1CS variable assignment `z = [x | w]` in fill
//! order via [`R1csEncoderInput::assignment_bits`]. The encoder writes
//! those bits into `image.app_private` verbatim.

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use crate::engine::ccs_native::poseidon2_transcript::SpongeTraceImage;
use crate::frontends::direct_ccs::R1cs;
use crate::frontends::f_prime_shell::encoder::{EncodedFPrimeStep, NifsPayloadInput};
use crate::frontends::f_prime_shell::image::{FPrimeImage, FPrimeImageLayout, KMulView, StateIn, StateOut};
use crate::frontends::f_prime_shell::recursive_plan::{build_recursive_step_image_config, RecursiveStepImagePlan};
use crate::frontends::r1cs_f_prime::structure::{build_r1cs_f_prime_structure, R1csRowAnchors};
use crate::paper::f_prime::poseidon_trace::PoseidonTraceImage;
use crate::paper::f_prime::ring_action_trace::RingActionTraceImage;

/// One R1CS encoder input. Same shape as the Fibonacci encoder input
/// except `assignment_bits` carries the bit-decomposed R1CS variable
/// assignment instead of Fibonacci's app-private carries.
pub struct R1csEncoderInput {
    pub plan: RecursiveStepImagePlan,
    pub boundary_bits: Vec<F>,
    pub state_in: StateIn,
    pub state_out: StateOut,
    pub chunk_digest: [F; 4],
    /// Bit-decomposed R1CS variable assignment `z` of length
    /// `r1cs.m() * 64`. Variable `z_j`'s 64 bits occupy
    /// `assignment_bits[j*64 .. (j+1)*64]` in little-endian order.
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
/// Returns the encoded step plus the R1CS row-anchor metadata so the
/// compiler can plumb it through to tests/verifiers without rebuilding
/// the structure.
pub fn encode_r1cs_f_prime_step(input: R1csEncoderInput, r1cs: &R1cs) -> (EncodedFPrimeStep, R1csRowAnchors) {
    let config = build_recursive_step_image_config(&input.plan);

    // ── Strict input-shape gate ──────────────────────────────────────
    assert_eq!(
        input.boundary_bits.len(),
        input.plan.boundary_bits,
        "boundary bits must match plan.boundary_bits"
    );
    assert_eq!(
        input.assignment_bits.len(),
        r1cs.m() * POSEIDON2_GOLDILOCKS_BITS,
        "assignment_bits must have length r1cs.m() * 64"
    );
    assert_eq!(
        input.plan.limbs,
        r1cs.m() * POSEIDON2_GOLDILOCKS_BITS + 1,
        "plan.limbs must equal r1cs.m() * 64 + 1 (limbs - 1 = app_private bit count)"
    );
    assert_eq!(
        input.nifs_payloads.len(),
        input.plan.nifs_payload_shapes.len(),
        "NIFS payload count must match plan.nifs_payload_shapes"
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

    let layout = FPrimeImageLayout::new(config);
    let mut image = FPrimeImage::new(layout.clone());

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

    let (structure, anchors) = build_r1cs_f_prime_structure(layout, r1cs);
    let witness = structure.extend_witness_from_image(&image);

    assert!(
        structure.is_satisfied(&witness),
        "encoded R1CS F' step must satisfy its structure; first failing row: {:?}",
        structure.first_unsatisfied_row(&witness)
    );

    (
        EncodedFPrimeStep {
            image,
            structure,
            witness,
        },
        anchors,
    )
}
