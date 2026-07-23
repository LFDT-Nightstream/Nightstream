//! Compact Lean literal encoding for the runtime execution exporter.

use neo_fold_clean::frontends::r1cs_f_prime::ivc::{R1csIvcGeneratedKSlot, R1csIvcPublicWriteSource};
use neo_math::{KExtensions, F, K};
use p3_field::PrimeField64;

pub(super) fn lean_k(value: K) -> String {
    let [c0, c1] = value.as_coeffs();
    format!("{{ c0 := {}, c1 := {} }}", c0.as_canonical_u64(), c1.as_canonical_u64())
}

pub(super) fn lean_k_list(values: &[K]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .copied()
            .map(lean_k)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub(super) fn lean_nat_values(values: &[F]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| value.as_canonical_u64().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub(super) fn lean_nat_pairs(values: &[[usize; 2]]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| format!("({}, {})", value[0], value[1]))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub(super) fn source_tag(source: R1csIvcPublicWriteSource) -> usize {
    match source {
        R1csIvcPublicWriteSource::ConstantOne => 0,
        R1csIvcPublicWriteSource::BuilderColumn => 1,
        R1csIvcPublicWriteSource::FixedZero => 2,
    }
}

pub(super) fn lean_option_nat(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| format!("some {value}"))
}

pub(super) fn slot_tag(slot: R1csIvcGeneratedKSlot) -> (usize, usize, usize) {
    match slot {
        R1csIvcGeneratedKSlot::Gamma => (0, 0, 0),
        R1csIvcGeneratedKSlot::BetaLane(index) => (1, index, 0),
        R1csIvcGeneratedKSlot::BetaBlock(index) => (2, index, 0),
        R1csIvcGeneratedKSlot::ProducerBeta => (3, 0, 0),
        R1csIvcGeneratedKSlot::BatchWeight => (4, 0, 0),
        R1csIvcGeneratedKSlot::PendingOldBlock(index) => (5, index, 0),
        R1csIvcGeneratedKSlot::PendingParentYZcol(index) => (6, index, 0),
        R1csIvcGeneratedKSlot::OutputYZcol { source, lane } => (7, source, lane),
        R1csIvcGeneratedKSlot::BlockPoint(index) => (8, index, 0),
        R1csIvcGeneratedKSlot::LanePoint(index) => (9, index, 0),
        R1csIvcGeneratedKSlot::ClaimedInitial => (10, 0, 0),
        R1csIvcGeneratedKSlot::FinalSum => (11, 0, 0),
        R1csIvcGeneratedKSlot::TerminalRhs => (12, 0, 0),
        R1csIvcGeneratedKSlot::RoundCoefficient { round, coefficient } => (13, round, coefficient),
        R1csIvcGeneratedKSlot::RoundChallenge(round) => (14, round, 0),
        R1csIvcGeneratedKSlot::RoundClaimIn(round) => (15, round, 0),
        R1csIvcGeneratedKSlot::RoundClaimOut(round) => (16, round, 0),
    }
}
