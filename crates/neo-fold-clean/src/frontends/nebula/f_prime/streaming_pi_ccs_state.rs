//! Shared fixed-width continuation carried by PiCCS start, round, and finish.
//!
//! Owns one canonical 67-field order and its Poseidon2 state digest. It does
//! not own transcript messages, SumCheck arithmetic, terminal checks, phase
//! selection, or lifecycle authority.

use neo_math::F;

use crate::engine::r1cs_circuit::{KVar, R1csBuilder, TranscriptGadget, Var};

pub(super) const PI_CCS_SPONGE_WIDTH: usize = 8;
pub(super) const PI_CCS_POINT_COUNT: usize = 26;
pub(super) const PI_CCS_CONTEXT_DIGEST_FIELDS: usize = 4;
pub(super) const PI_CCS_LOCAL_STATE_FIELDS: usize =
    PI_CCS_SPONGE_WIDTH + 2 + 2 * PI_CCS_POINT_COUNT + 1 + PI_CCS_CONTEXT_DIGEST_FIELDS;

const STATE_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/nebula/f-prime/streaming-pi-ccs-round-state/v1";
const STATE_DIGEST_FIELDS_LABEL: &[u8] = b"state";

pub(super) type PiCcsPair = [F; 2];

#[derive(Clone, Copy)]
pub(super) struct StreamingPiCcsStateValue {
    pub transcript: [F; PI_CCS_SPONGE_WIDTH],
    pub current: PiCcsPair,
    pub reverse_point: [PiCcsPair; PI_CCS_POINT_COUNT],
    pub round_cursor: F,
    pub context_digest: [F; PI_CCS_CONTEXT_DIGEST_FIELDS],
}

#[derive(Clone, Copy)]
pub(super) struct StreamingPiCcsStateVars {
    pub transcript: [Var; PI_CCS_SPONGE_WIDTH],
    pub current: KVar,
    pub reverse_point: [KVar; PI_CCS_POINT_COUNT],
    pub round_cursor: Var,
    pub context_digest: [Var; PI_CCS_CONTEXT_DIGEST_FIELDS],
}

pub(super) fn alloc_streaming_pi_ccs_state(
    builder: &mut R1csBuilder,
    value: StreamingPiCcsStateValue,
) -> StreamingPiCcsStateVars {
    StreamingPiCcsStateVars {
        transcript: value.transcript.map(|value| builder.alloc(value)),
        current: KVar::alloc(builder, value.current[0], value.current[1]),
        reverse_point: value
            .reverse_point
            .map(|value| KVar::alloc(builder, value[0], value[1])),
        round_cursor: builder.alloc(value.round_cursor),
        context_digest: value.context_digest.map(|value| builder.alloc(value)),
    }
}

fn state_fields(state: StreamingPiCcsStateVars) -> Vec<Var> {
    let mut fields = Vec::with_capacity(PI_CCS_LOCAL_STATE_FIELDS);
    fields.extend(state.transcript);
    fields.push(state.current.c0);
    fields.push(state.current.c1);
    for point in state.reverse_point {
        fields.push(point.c0);
        fields.push(point.c1);
    }
    fields.push(state.round_cursor);
    fields.extend(state.context_digest);
    debug_assert_eq!(fields.len(), PI_CCS_LOCAL_STATE_FIELDS);
    fields
}

pub(super) fn digest_streaming_pi_ccs_state(builder: &mut R1csBuilder, state: StreamingPiCcsStateVars) -> [Var; 4] {
    let fields = state_fields(state);
    let mut transcript = TranscriptGadget::new(builder, STATE_DIGEST_DOMAIN);
    transcript.append_fields(builder, STATE_DIGEST_FIELDS_LABEL, &fields);
    transcript.digest_fields(builder)
}

const _: () = assert!(PI_CCS_LOCAL_STATE_FIELDS == 67);
const _: () = assert!(PI_CCS_POINT_COUNT == 26);
