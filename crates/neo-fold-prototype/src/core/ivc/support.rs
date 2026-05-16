//! Owns small mechanical helpers for native SuperNeo IVC flows.

use std::time::Instant;

use neo_reductions::error::PiCcsError;

use crate::proof::Carry;

pub(super) fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

pub(super) fn carry_matches(left: &Carry, right: &Carry) -> bool {
    left.claims == right.claims && left.witnesses == right.witnesses
}

pub(super) fn ivc_protocol_error(message: impl Into<String>) -> PiCcsError {
    PiCcsError::ProtocolError(message.into())
}
