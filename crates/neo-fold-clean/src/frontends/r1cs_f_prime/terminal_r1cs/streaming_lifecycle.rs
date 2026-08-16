//! Streaming lifecycle rows appended to the direct terminal relation.
//!
//! Owns the exact terminal-arm selection, source-field reconstruction,
//! 32-field XOut authority, phase semantic digest, delayed Nebula finalizer,
//! program binding, and final closed-lane predicate.
//!
//! Does not own the eight SuperNeo terminal families, the final selective CCS
//! rows, source-profile construction, verifier-native statement derivation,
//! or Spartan.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::frontends::nebula::f_prime::{
    enforce_streaming_phase_semantic_digest, NebulaFPrimeStreamingTerminalFieldBinding,
    NebulaFPrimeStreamingTerminalProfile, STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS,
};
use crate::paper::construction2::NebulaConfig;
use crate::paper::digest::{F_PRIME_STATE_X_OUT_DOMAIN, NEBULA_ADV_PRESENT_MARKER};
use crate::paper::f_prime::digest_circuit::alloc_constant;
use crate::paper::f_prime::nebula_lane_circuit::{
    decode_delayed_nebula_public_suffix_circuit, enforce_delayed_nebula_claim_data_circuit,
    enforce_nebula_lane_digest_selected_circuit, enforce_nebula_program_binding_digest_circuit,
    enforce_nebula_terminal_closed_circuit, NebulaLaneWires, NebulaOpenContextWires,
};
use crate::paper::relations::product_commitment_circuit::AdvCommitmentDataWires;

/// Reviewed lifecycle-family vocabulary added after the complete SuperNeo
/// terminal relation.
pub const STREAMING_TERMINAL_R1CS_FAMILY_NAMES: [&str; 8] = [
    "terminal.streaming.source_binding",
    "terminal.streaming.profile_selection",
    "terminal.streaming.x_out_context",
    "terminal.streaming.phase_semantic",
    "terminal.streaming.nebula_state_digest",
    "terminal.streaming.nebula_program_binding",
    "terminal.streaming.nebula_finalizer",
    "terminal.streaming.nebula_closed",
];

/// Verifier-derived public values used by the terminal lifecycle rows.
///
/// The caller allocates these as public columns. The verifier derives them
/// from preprocessing, the terminal claim, and the checked running instance.
#[derive(Clone, Copy)]
pub struct StreamingTerminalPublicWires {
    pub vk_fs_digest: [Var; 4],
    pub pi_ccs_header: [Var; 4],
    pub current_boundary: [Var; 4],
    pub accumulator_digest: [Var; 4],
}

/// Decoded state retained for exact tests and later terminal integration.
pub struct StreamingTerminalLifecycleOutput {
    pub post_phase_lane: NebulaLaneWires,
    pub final_lane: NebulaLaneWires,
    pub delayed_payload: Vec<Var>,
}

#[derive(Debug, Error)]
pub enum StreamingTerminalLifecycleError {
    #[error("streaming terminal lifecycle profile width mismatch for {what}: expected {expected}, got {got}")]
    Width {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("streaming terminal lifecycle decoder for {what} uses final column {column} outside width {width}")]
    DecoderColumn {
        what: &'static str,
        column: usize,
        width: usize,
    },
    #[error("streaming terminal delayed Nebula input: {0}")]
    DelayedInput(String),
    #[error("streaming terminal delayed Nebula finalizer: {0}")]
    Finalizer(String),
}

/// Append the complete streaming-specific terminal lifecycle relation.
///
/// `final_witness` is the exact final selective-CCS assignment, including its
/// public prefix. `fresh_adv` is the public trailing-claim commitment opened
/// from three slices of this same assignment by the terminal commitment rows.
pub fn enforce_streaming_terminal_lifecycle(
    builder: &mut R1csBuilder,
    profile: &NebulaFPrimeStreamingTerminalProfile,
    final_witness: &[Var],
    fresh_adv: &AdvCommitmentDataWires,
    config: &NebulaConfig,
    public: StreamingTerminalPublicWires,
) -> Result<StreamingTerminalLifecycleOutput, StreamingTerminalLifecycleError> {
    require_width(
        "final selective assignment",
        profile.final_columns(),
        final_witness.len(),
    )?;
    require_width(
        "accepted work items",
        STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS,
        profile.accepted_work_items(),
    )?;

    let family_start = builder.rows();
    let x_out = decode_fixed::<32>(builder, profile.after_x_out().fields(), final_witness, "XOut preimage")?;
    let lane_fields = decode_fixed::<50>(
        builder,
        profile.after_nebula_lane().fields(),
        final_witness,
        "post-phase Nebula lane",
    )?;
    let local_state = decode_fixed::<4>(
        builder,
        profile.after_local_state_digest().fields(),
        final_witness,
        "phase-local state",
    )?;
    let delayed_payload = decode_fields(
        builder,
        profile.after_delayed_payload().fields(),
        final_witness,
        "delayed Nebula payload",
    )?;
    builder.record_row_family(STREAMING_TERMINAL_R1CS_FAMILY_NAMES[0], family_start);

    let family_start = builder.rows();
    for column in [
        profile.schedule_selector_column(),
        profile.lifecycle_selector_column(),
        profile.phase_selector_column(),
    ] {
        let selector = *final_witness
            .get(column)
            .ok_or(StreamingTerminalLifecycleError::DecoderColumn {
                what: "terminal selector",
                column,
                width: final_witness.len(),
            })?;
        builder.enforce_eq(&Lc::from_var(selector), &Lc::from_const(F::ONE));
    }
    builder.record_row_family(STREAMING_TERMINAL_R1CS_FAMILY_NAMES[1], family_start);

    let family_start = builder.rows();
    bind_const(builder, x_out[0], F::from_u64(F_PRIME_STATE_X_OUT_DOMAIN));
    bind_array(builder, &x_out[1..5], &public.vk_fs_digest);
    bind_array(builder, &x_out[5..9], &public.pi_ccs_header);
    bind_const(
        builder,
        x_out[9],
        F::from_u64(STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS as u64),
    );
    bind_const(builder, x_out[10], F::ZERO);
    bind_const(
        builder,
        x_out[11],
        F::from_u64(STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS as u64),
    );
    bind_const(builder, x_out[12], F::ZERO);
    bind_const(builder, x_out[13], F::ONE);
    bind_const(builder, x_out[14], F::ZERO);
    bind_array(builder, &x_out[15..19], &public.current_boundary);
    bind_array(builder, &x_out[23..27], &public.accumulator_digest);
    bind_const(builder, x_out[27], F::from_u64(NEBULA_ADV_PRESENT_MARKER));
    builder.record_row_family(STREAMING_TERMINAL_R1CS_FAMILY_NAMES[2], family_start);

    let family_start = builder.rows();
    let semantic_digest = enforce_streaming_phase_semantic_digest(builder, local_state, &delayed_payload, false);
    bind_array(builder, &x_out[19..23], &semantic_digest);
    builder.record_row_family(STREAMING_TERMINAL_R1CS_FAMILY_NAMES[3], family_start);

    let post_phase_lane = lane_from_fields(lane_fields);
    let family_start = builder.rows();
    let lane_digest = enforce_nebula_lane_digest_selected_circuit(builder, &post_phase_lane);
    bind_array(builder, &x_out[28..32], &lane_digest);
    builder.record_row_family(STREAMING_TERMINAL_R1CS_FAMILY_NAMES[4], family_start);

    let family_start = builder.rows();
    let initial_semantic = config
        .initial_semantic_state_digest
        .map(|value| alloc_constant(builder, value));
    let plan_digest = config
        .plan_digest
        .map(|value| alloc_constant(builder, value));
    let d_init = config.d_init.map(|value| alloc_constant(builder, value));
    let program_binding = enforce_nebula_program_binding_digest_circuit(builder, initial_semantic, plan_digest, d_init);
    bind_array(builder, &post_phase_lane.program_binding_digest, &program_binding);
    builder.record_row_family(STREAMING_TERMINAL_R1CS_FAMILY_NAMES[5], family_start);

    let family_start = builder.rows();
    let delayed = decode_delayed_nebula_public_suffix_circuit(builder, &delayed_payload, config.stacks)
        .map_err(|error| StreamingTerminalLifecycleError::DelayedInput(error.to_string()))?;
    let context = NebulaOpenContextWires {
        vk_fs: public.vk_fs_digest,
        z_i: public.current_boundary,
        acc_digest: public.accumulator_digest,
    };
    let transition = enforce_delayed_nebula_claim_data_circuit(
        builder,
        &post_phase_lane,
        &delayed,
        fresh_adv,
        &context,
        config.steps_per_segment,
        config.seg_max,
    )
    .map_err(StreamingTerminalLifecycleError::Finalizer)?;
    builder.enforce_eq(&Lc::from_var(transition.closed), &Lc::from_const(F::ONE));
    builder.record_row_family(STREAMING_TERMINAL_R1CS_FAMILY_NAMES[6], family_start);

    let family_start = builder.rows();
    enforce_nebula_terminal_closed_circuit(builder, &transition.lane);
    builder.record_row_family(STREAMING_TERMINAL_R1CS_FAMILY_NAMES[7], family_start);

    Ok(StreamingTerminalLifecycleOutput {
        post_phase_lane,
        final_lane: transition.lane,
        delayed_payload,
    })
}

fn require_width(what: &'static str, expected: usize, got: usize) -> Result<(), StreamingTerminalLifecycleError> {
    if expected == got {
        Ok(())
    } else {
        Err(StreamingTerminalLifecycleError::Width { what, expected, got })
    }
}

fn decode_fixed<const N: usize>(
    builder: &mut R1csBuilder,
    fields: &[NebulaFPrimeStreamingTerminalFieldBinding],
    final_witness: &[Var],
    what: &'static str,
) -> Result<[Var; N], StreamingTerminalLifecycleError> {
    require_width(what, N, fields.len())?;
    decode_fields(builder, fields, final_witness, what)?
        .try_into()
        .map_err(|fields: Vec<Var>| StreamingTerminalLifecycleError::Width {
            what,
            expected: N,
            got: fields.len(),
        })
}

fn decode_fields(
    builder: &mut R1csBuilder,
    fields: &[NebulaFPrimeStreamingTerminalFieldBinding],
    final_witness: &[Var],
    what: &'static str,
) -> Result<Vec<Var>, StreamingTerminalLifecycleError> {
    fields
        .iter()
        .map(|field| {
            let mut decoding = Lc::zero();
            for term in field.decoder_terms() {
                let source =
                    *final_witness
                        .get(term.final_column())
                        .ok_or(StreamingTerminalLifecycleError::DecoderColumn {
                            what,
                            column: term.final_column(),
                            width: final_witness.len(),
                        })?;
                decoding.add_term(source, term.coefficient());
            }
            let decoded = builder.alloc(builder.eval(&decoding));
            builder.enforce_eq(&Lc::from_var(decoded), &decoding);
            Ok(decoded)
        })
        .collect()
}

fn lane_from_fields(fields: [Var; 50]) -> NebulaLaneWires {
    let k = |start: usize| KVar::new(fields[start], fields[start + 1]);
    NebulaLaneWires {
        program_binding_digest: fields[0..4]
            .try_into()
            .expect("four program-binding fields"),
        open: fields[4],
        seg_idx: fields[5],
        idx: fields[6],
        ts: fields[7],
        gamma: [k(8), k(10)],
        h: [k(12), k(14), k(16), k(18)],
        sp: [fields[20], fields[21]],
        d_pre: [
            fields[22..26]
                .try_into()
                .expect("four ops pre-chain fields"),
            fields[26..30].try_into().expect("four IS pre-chain fields"),
            fields[30..34].try_into().expect("four FS pre-chain fields"),
        ],
        d_seen: [
            fields[34..38]
                .try_into()
                .expect("four ops seen-chain fields"),
            fields[38..42]
                .try_into()
                .expect("four IS seen-chain fields"),
            fields[42..46]
                .try_into()
                .expect("four FS seen-chain fields"),
        ],
        d_mem: fields[46..50]
            .try_into()
            .expect("four memory-digest fields"),
    }
}

fn bind_const(builder: &mut R1csBuilder, actual: Var, expected: F) {
    builder.enforce_eq(&Lc::from_var(actual), &Lc::from_const(expected));
}

fn bind_array(builder: &mut R1csBuilder, actual: &[Var], expected: &[Var]) {
    debug_assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        builder.enforce_eq(&Lc::from_var(actual), &Lc::from_var(expected));
    }
}

const _: () = assert!(STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS <= u32::MAX as usize);
