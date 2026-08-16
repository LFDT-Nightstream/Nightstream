//! Authenticated semantic envelope for one phased F-prime continuation.
//!
//! Owns the Poseidon2 compression of one phase-local state digest and the exact
//! delayed Nebula payload bits. The lifecycle relation recomputes this envelope
//! and binds it to its semantic-state lane. A selected phase must still derive
//! and link both source slices; this digest is not independent authority.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::boolean::enforce_bit;
use crate::engine::r1cs_circuit::{enforce_poseidon2_hash, Lc, R1csBuilder, Var};
use crate::paper::construction2::StackShape;
use crate::paper::digest::pack_bytes_as_fields;
use crate::paper::f_prime::digest_circuit::{alloc_const_tag, alloc_constant};
use crate::paper::f_prime::nebula_lane_circuit::delayed_nebula_public_suffix_len;
use crate::paper::f_prime::poseidon_trace::encode_poseidon_trace;

const PHASE_SEMANTIC_DOMAIN: &[u8] = b"neo.fold.clean/nebula/f-prime/streaming-phase-semantic/v1";

/// The frozen streaming profile uses the stack-less Nebula step layout.
pub const STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS: usize = delayed_nebula_public_suffix_len(StackShape::NONE);

#[doc(hidden)]
pub const STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY: &str = "fprime.streaming.phase.before.local_state_digest";
#[doc(hidden)]
pub const STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY: &str =
    "fprime.streaming.phase.before.delayed_payload.raw_bits";
#[doc(hidden)]
pub const STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY: &str = "fprime.streaming.phase.after.local_state_digest";
#[doc(hidden)]
pub const STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY: &str = "fprime.streaming.phase.after.delayed_payload.raw_bits";

#[doc(hidden)]
pub const STREAMING_CARRY_PHASE_ENVELOPE_FAMILY: &str = "fprime.streaming.phase.carry.semantic_envelope";

/// Semantic digests and exact private source slices for a carry phase.
pub struct StreamingCarryPhaseSemanticEnvelope {
    pub before_semantic_digest: [Var; 4],
    pub after_semantic_digest: [Var; 4],
    pub before_local_state_source_digest: [Var; 4],
    pub after_local_state_source_digest: [Var; 4],
    pub before_local_state_digest: [Var; 4],
    pub after_local_state_digest: [Var; 4],
    pub delayed_payload_bits: Vec<Var>,
}

/// Native mirror of [`enforce_streaming_phase_semantic_digest`].
#[doc(hidden)]
pub fn streaming_phase_semantic_digest(local_state_digest: [F; 4], delayed_payload_bits: &[F]) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(PHASE_SEMANTIC_DOMAIN);
    preimage.push(F::from_u64(delayed_payload_bits.len() as u64));
    preimage.extend(local_state_digest);
    preimage.extend_from_slice(delayed_payload_bits);
    encode_poseidon_trace(&preimage).digest_native
}

/// Recompute the carried semantic digest from the exact phase-local digest and
/// delayed payload source. `check_bits` is false only when another row family
/// already proves bitness for this same wire slice.
#[doc(hidden)]
pub fn enforce_streaming_phase_semantic_digest(
    builder: &mut R1csBuilder,
    local_state_digest: [Var; 4],
    delayed_payload_bits: &[Var],
    check_bits: bool,
) -> [Var; 4] {
    if check_bits {
        for &bit in delayed_payload_bits {
            enforce_bit(builder, bit);
        }
    }
    let mut preimage = alloc_const_tag(builder, PHASE_SEMANTIC_DOMAIN);
    preimage.push(alloc_constant(builder, F::from_u64(delayed_payload_bits.len() as u64)));
    preimage.extend(local_state_digest);
    preimage.extend_from_slice(delayed_payload_bits);
    enforce_poseidon2_hash(builder, &preimage)
}

/// Bind one phase-local transition to the frozen delayed Nebula payload.
///
/// A carry phase cannot produce a new delayed payload. The before and after
/// source families therefore name the same exact wire slice. The scheduled
/// common-to-phase links bind both lifecycle boundaries to that slice.
#[doc(hidden)]
pub fn enforce_streaming_carry_phase_semantic_envelope(
    builder: &mut R1csBuilder,
    before_local_state_digest: [Var; 4],
    after_local_state_digest: [Var; 4],
) -> StreamingCarryPhaseSemanticEnvelope {
    let row_start = builder.rows();

    let before_local_start = builder.cols();
    let before_local_source = before_local_state_digest.map(|source| builder.alloc(builder.witness()[source.col()]));
    for (&source, &alias) in before_local_state_digest.iter().zip(&before_local_source) {
        builder.enforce_eq(&Lc::from_var(alias), &Lc::from_var(source));
    }
    builder.record_column_family(STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY, before_local_start);

    let payload_start = builder.cols();
    let delayed_payload_bits = builder.alloc_vec(&vec![F::ZERO; STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS]);
    for &bit in &delayed_payload_bits {
        enforce_bit(builder, bit);
    }
    builder.record_column_family(STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY, payload_start);
    builder.record_column_family(STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY, payload_start);

    let after_local_start = builder.cols();
    let after_local_source = after_local_state_digest.map(|source| builder.alloc(builder.witness()[source.col()]));
    for (&source, &alias) in after_local_state_digest.iter().zip(&after_local_source) {
        builder.enforce_eq(&Lc::from_var(alias), &Lc::from_var(source));
    }
    builder.record_column_family(STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY, after_local_start);

    let before_semantic_digest =
        enforce_streaming_phase_semantic_digest(builder, before_local_source, &delayed_payload_bits, false);
    let after_semantic_digest =
        enforce_streaming_phase_semantic_digest(builder, after_local_source, &delayed_payload_bits, false);
    builder.record_row_family(STREAMING_CARRY_PHASE_ENVELOPE_FAMILY, row_start);

    StreamingCarryPhaseSemanticEnvelope {
        before_semantic_digest,
        after_semantic_digest,
        before_local_state_source_digest: before_local_state_digest,
        after_local_state_source_digest: after_local_state_digest,
        before_local_state_digest: before_local_source,
        after_local_state_digest: after_local_source,
        delayed_payload_bits,
    }
}
