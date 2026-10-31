//! Sum-check rounds against challenges that are already drawn, on plain values. A round
//! message is `h(0), h(2), ..., h(degree)`: the verifier derives `h(1) = claim - h(0)`, and the
//! next claim is `h` at the round's challenge.
//!
//! `formal/compression-lean/CompressionLeanRust/Rounds.lean` proves `round_claim` equal to the
//! round polynomial of `Lib.roundValues` at the challenge, and `replay_claim` equal to the claim
//! where `Game.sumcheck` ends.

use crate::algebra::interpolate;
use crate::field::{ext_reduce, ext_sub, Ext};
use crate::native::Native;

/// `h(r)` for the round whose message is the first `degree` values of `message`.
pub fn round_claim(degree: usize, message: &[Ext], claim: Ext, r: Ext) -> Ext {
    let mut values = Vec::new();
    values.push(message[0]);
    values.push(ext_sub(claim, message[0]));
    let mut j = 1;
    while j < degree {
        values.push(message[j]);
        j = j.wrapping_add(1);
    }
    interpolate(&mut Native, &values, r)
}

/// The claim after `rounds` from `claim`, with round `i` bound at `point[i]`. With no round it
/// is `claim` with canonical words, as every other result is.
pub fn replay_claim(degree: usize, rounds: &[Vec<Ext>], claim: Ext, point: &[Ext]) -> Ext {
    let mut claim = ext_reduce(claim);
    let mut i = 0;
    while i < rounds.len() {
        claim = round_claim(degree, &rounds[i], claim, point[i]);
        i = i.wrapping_add(1);
    }
    claim
}
