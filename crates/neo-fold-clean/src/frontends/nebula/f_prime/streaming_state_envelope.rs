//! Public full-state envelope for one phased Nebula F-prime circuit.
//!
//! Owns the stateful, Nebula-present `x_out` recomputation and its canonical
//! 256-bit public encoding. It does not own input authority, phase-state
//! semantics, lifecycle transitions, or public-column placement.

use crate::engine::r1cs_circuit::poseidon2::DIGEST_LEN;
use crate::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use crate::engine::r1cs_circuit::{R1csBuilder, Var};
use crate::paper::digest::StateXOutDigestMode;
use crate::paper::f_prime::digest_circuit::{
    enforce_state_x_out_digest_with_nebula_circuit_wires, StateXOutDigestInputs,
};
use crate::paper::f_prime::public_input_link::F_PRIME_ENC_INST_BITS;

pub(crate) struct StreamingStateXOutWires {
    pub digest: [Var; DIGEST_LEN],
    pub preimage: Vec<Var>,
    pub public_bits: [Var; F_PRIME_ENC_INST_BITS],
}

/// Recompute the complete stateful `x_out` with a present Nebula lane, then
/// encode its four Goldilocks lanes as 256 little-endian public bits.
///
/// The caller must bind every input wire to its phase or lifecycle authority.
/// In particular, `inputs.semantic_acc` is the independently recomputed local
/// phase-state digest. This function never treats that digest as `x_out`.
pub fn enforce_streaming_state_x_out_bits(
    builder: &mut R1csBuilder,
    inputs: &StateXOutDigestInputs,
    nebula_lane_digest: [Var; DIGEST_LEN],
) -> [Var; F_PRIME_ENC_INST_BITS] {
    enforce_streaming_state_x_out(builder, inputs, nebula_lane_digest).public_bits
}

pub(crate) fn enforce_streaming_state_x_out(
    builder: &mut R1csBuilder,
    inputs: &StateXOutDigestInputs,
    nebula_lane_digest: [Var; DIGEST_LEN],
) -> StreamingStateXOutWires {
    assert_eq!(
        inputs.mode,
        StateXOutDigestMode::Stateful,
        "Nebula streaming phases require stateful x_out",
    );
    let wires = enforce_state_x_out_digest_with_nebula_circuit_wires(builder, inputs, nebula_lane_digest);
    let public_bits = wires
        .digest
        .into_iter()
        .flat_map(|lane| decompose_var_to_u64_bits(builder, lane))
        .collect::<Vec<_>>()
        .try_into()
        .expect("four state_x_out lanes contain 256 bits");
    StreamingStateXOutWires {
        digest: wires.digest,
        preimage: wires.preimage,
        public_bits,
    }
}

const _: () = assert!(F_PRIME_ENC_INST_BITS == DIGEST_LEN * 64);
