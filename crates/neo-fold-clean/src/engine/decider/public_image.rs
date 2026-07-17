//! Public-image authority and state-digest constraints for the audit decider.
//!
//! Owns: verifier-owned public pins, canonical digest decoding, and the final
//! `state_x_out` recomputation. Does not own lifecycle orchestration or NIFS.
//! Emits constraints: yes. Authority boundary: public values and preprocessing
//! constants are pinned to circuit-derived wires; no carried digest is trusted.
//!
//! | Obligation | Mathematical check | Constraint owner |
//! |---|---|---|
//! | Preprocessing anchors | public constants equal verifier-owned constants | `enforce_public_preprocessing_anchors` |
//! | Terminal public image | all ten public coordinates equal derived wires | `pin_public_image` |
//! | Canonical words | each 64-bit digest limb is below the field modulus | `canonical_digest32_fields` |
//! | Terminal state handle | `x_out = H(state_out, final_accumulator)` | `pin_public_image` |

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{enforce_digest_eq, REQUIRED_PUBLIC_IMAGE_PINS};
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::lifecycle::Preprocessing;
use crate::paper::construction2::{SemanticStateMode, State};
use crate::paper::decider::PublicImage;
use crate::paper::digest::{
    digest32_as_fields, initial_boundary_digest, state_x_out_digest_with_mode, StateXOutDigestMode,
};
use crate::paper::f_prime::digest_circuit::{enforce_state_x_out_digest_circuit, StateXOutDigestInputs};
use crate::paper::f_prime::nebula_lane_circuit::delayed_nebula_public_suffix_len;
use crate::paper::f_prime::r1cs::{FPrimePublicInputLayout, FPrimeStepOutput};

/// Pin every terminal field of `statement.public` to chain-derived wires.
/// The initial semantic-state digest is pinned with the base-state seed.
pub(super) fn pin_public_image(
    builder: &mut R1csBuilder,
    public: &PublicImage,
    prep: &Preprocessing,
    last: &FPrimeStepOutput,
    final_acc_digest: &[Var; 4],
) -> usize {
    let so = &last.state_out;
    enforce_public_preprocessing_anchors(builder, prep, public);
    pin_digest32(builder, &so.vk_fs_digest, public.vk_fs_digest);
    pin_u64(builder, so.chunk_count, public.chunk_count);
    pin_u64(builder, so.step_count, public.step_count);
    pin_digest32(builder, &so.z_0, public.z_0);
    pin_digest32(builder, &so.z_i, public.z_i);
    pin_u64(builder, so.pc, public.pc);
    pin_digest32(builder, &so.semantic_state_digest, public.semantic_state_digest);
    if matches!(prep.semantic_state_mode(), SemanticStateMode::Stateless) {
        // The stateless x_out preimage omits duplicate semantic lanes, so this
        // equality is load-bearing for public-image binding.
        enforce_digest_eq(builder, &so.semantic_state_digest, &so.acc_digest);
    }
    pin_digest32(builder, final_acc_digest, public.acc_digest);
    pin_digest32(builder, &so.public_trace, public.public_trace);

    let terminal_x_out_inputs = StateXOutDigestInputs {
        mode: state_x_out_digest_mode(prep),
        vk_fs_digest: so.vk_fs_digest,
        pi_ccs_header_bundle: so.pi_ccs_header_bundle,
        structure_digest: so.pi_ccs_header_bundle,
        chunk_count: so.chunk_count,
        step_count: so.step_count,
        initial_boundary: so.z_0,
        current_boundary: so.z_i,
        pc: so.pc,
        semantic_acc: so.semantic_state_digest,
        construction2_acc: *final_acc_digest,
        public_trace: so.public_trace,
    };
    let terminal_x_out = enforce_state_x_out_digest_circuit(builder, &terminal_x_out_inputs);
    pin_digest32(builder, &terminal_x_out, public.x_out.digest_bytes);

    let header_bundle = prep.pi_ccs_header_bundle();
    for k in 0..4 {
        builder.enforce_eq(
            &Lc::from_var(so.pi_ccs_header_bundle[k]),
            &Lc::from_const(header_bundle[k]),
        );
    }
    REQUIRED_PUBLIC_IMAGE_PINS
}

fn enforce_public_preprocessing_anchors(builder: &mut R1csBuilder, prep: &Preprocessing, public: &PublicImage) {
    let structure_lanes = *prep.structure_digest();
    let expected_z_0 = initial_boundary_digest(&structure_lanes, prep.public_input_len);

    enforce_digest32_const_eq(builder, public.vk_fs_digest, prep.vk.digest());
    enforce_digest32_const_eq(builder, public.z_0, expected_z_0);
    enforce_digest32_const_eq(
        builder,
        public.initial_semantic_state_digest,
        prep.initial_semantic_state_digest(),
    );
}

fn enforce_digest32_const_eq(builder: &mut R1csBuilder, actual: [u8; 32], expected: [u8; 32]) {
    let Some(actual) = canonical_digest32_fields(actual) else {
        enforce_unsat(builder);
        return;
    };
    let Some(expected) = canonical_digest32_fields(expected) else {
        enforce_unsat(builder);
        return;
    };
    for k in 0..4 {
        builder.enforce_eq(&Lc::from_const(actual[k]), &Lc::from_const(expected[k]));
    }
}

pub(super) fn pin_digest32(builder: &mut R1csBuilder, wires: &[Var; 4], expected: [u8; 32]) {
    let Some(expected_lanes) = canonical_digest32_fields(expected) else {
        enforce_unsat(builder);
        return;
    };
    for k in 0..4 {
        builder.enforce_eq(&Lc::from_var(wires[k]), &Lc::from_const(expected_lanes[k]));
    }
}

pub(super) fn pin_u64(builder: &mut R1csBuilder, wire: Var, expected: u64) {
    if expected >= F::ORDER_U64 {
        enforce_unsat(builder);
        return;
    }
    builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(F::from_u64(expected)));
}

fn canonical_digest32_fields(bytes: [u8; 32]) -> Option<[F; 4]> {
    let mut fields = [F::ZERO; 4];
    for (lane, out) in fields.iter_mut().enumerate() {
        let start = lane * 8;
        let value = u64::from_le_bytes(
            bytes[start..start + 8]
                .try_into()
                .expect("8-byte digest limb"),
        );
        if value >= F::ORDER_U64 {
            return None;
        }
        *out = F::from_u64(value);
    }
    Some(fields)
}

fn enforce_unsat(builder: &mut R1csBuilder) {
    builder.enforce_eq(&Lc::zero(), &Lc::from_const(F::ONE));
}

pub(super) fn state_x_out_lanes(prep: &Preprocessing, state: &State) -> [F; 4] {
    digest32_as_fields(state_x_out_digest_with_mode(
        state_x_out_digest_mode(prep),
        prep.vk.digest(),
        prep.pi_ccs_header_bundle(),
        prep.structure_digest(),
        state.chunk_count,
        state.step_count,
        state.z_0,
        state.z_i,
        state.pc,
        state.semantic_state_digest,
        state.acc_digest,
        state.public_trace,
        state.nebula.as_ref().map(|lane| lane.digest()),
    ))
}

pub(super) fn state_x_out_digest_mode(prep: &Preprocessing) -> StateXOutDigestMode {
    match prep.semantic_state_mode() {
        SemanticStateMode::Stateless => StateXOutDigestMode::Stateless,
        SemanticStateMode::Stateful => StateXOutDigestMode::Stateful,
    }
}

pub(super) fn f_prime_public_input_layout(prep: &Preprocessing) -> FPrimePublicInputLayout {
    match prep.nebula() {
        None => FPrimePublicInputLayout::plain(),
        Some(config) => FPrimePublicInputLayout::with_suffix(delayed_nebula_public_suffix_len(config.stacks)),
    }
}
