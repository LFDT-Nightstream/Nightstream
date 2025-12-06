//! Test helpers for neo-math tests
//! This module contains constants and functions that are only needed for testing.

use neo_math::{Fq, Rq};
use p3_field::{PrimeField64, TwoAdicField};
use p3_goldilocks::Goldilocks;

/// Goldilocks modulus for test assertions
#[allow(dead_code)]
pub const GOLDILOCKS_MODULUS: u128 = 18446744069414584321u128;

/// Two-adicity of F_q^* (Goldilocks has 2^32 | q-1).
#[allow(dead_code)]
pub const TWO_ADICITY: usize = <Fq as TwoAdicField>::TWO_ADICITY;

/// A fixed quadratic non-residue for F_q; verified in tests via Euler's criterion.
/// We use 7, which is known to be a quadratic non-residue modulo the Goldilocks prime.
#[allow(dead_code)]
pub fn nonresidue() -> Fq {
    // Goldilocks's internal representation is a u64, wrapped in struct Goldilocks(u64)
    // We construct it by using raw value 7
    unsafe { std::mem::transmute::<u64, Goldilocks>(7) }
}

/// Provide a two-adic generator (2^bits-th root of unity) for NTT tests.
#[inline]
#[allow(dead_code)]
pub fn two_adic_generator(bits: usize) -> Fq {
    <Fq as TwoAdicField>::two_adic_generator(bits)
}

/// Infinity norm for ring elements - used in tests
/// This is a copy of the inf_norm implementation from ring.rs for testing purposes
#[allow(dead_code)]
pub fn inf_norm(a: &Rq) -> u128 {
    let p: u128 = GOLDILOCKS_MODULUS;
    let half = (p - 1) / 2;
    let mut m = 0u128;
    for &c in a.0.iter() {
        let x = <Fq as PrimeField64>::as_canonical_u64(&c) as u128;
        let centered = if x <= half { x } else { p - x };
        if centered > m {
            m = centered;
        }
    }
    m
}
