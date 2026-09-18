//! Nebula public-coin fingerprint over `K` — spec §4.3.
//!
//! Owns: the packed tuple map `packed(t, g) = t + 2^TS_BITS · g`, the
//! fingerprint `f_γ(t, g, v) = γ2 − (packed + γ1 · v)`, and running
//! products. Used by the native prover (product columns), the trace oracle,
//! and tests; the `S_mem` circuit re-expresses the same equations as rows.
//!
//! Does not own: γ derivation (F′ transcript, spec §6.2) or the soundness
//! bound (security-note Lemma 3, which this construction instantiates).
//!
//! Packing is overflow-free by plan validation ([`super::layout::NebulaParams::new`]
//! enforces `TS_BITS + bits(R + M) ≤ 62`), and injective on `(t, g)` by the
//! same range checks the lanes carry as bitness.

use neo_math::field::KExtensions;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

use crate::frontends::nebula::layout::{H_FS, H_IS, H_RS, H_WS, TS_BITS};

/// The per-segment challenges `(γ1, γ2)`, sampled after every lane
/// commitment of the segment is absorbed (spec §6.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gammas {
    /// Value-mixing challenge.
    pub gamma1: K,
    /// Offset challenge.
    pub gamma2: K,
}

/// One multiset element: `(timestamp, global cell index, value)`.
/// `g = addr + seg · R` per spec §3.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemTuple {
    /// Timestamp (`rt` for RS entries, write timestamp for WS, cell
    /// timestamp for IS/FS).
    pub t: u64,
    /// Global cell index.
    pub g: u64,
    /// Cell value.
    pub v: u32,
}

/// `packed(t, g) = t + 2^TS_BITS · g` as a base-field element.
///
/// Debug-asserts the ranges the lanes enforce as bitness; release builds
/// rely on plan validation for the overflow bound.
pub fn packed(t: u64, g: u64) -> F {
    debug_assert!(t >> TS_BITS == 0, "timestamp exceeds TS_BITS");
    debug_assert!((g as u128) << TS_BITS < 1 << 62, "global index exceeds packing bound");
    F::from_u64(t + (g << TS_BITS))
}

/// `f_γ(e) = γ2 − (packed(e) + γ1 · v(e))` over `K`.
pub fn fingerprint(gammas: &Gammas, e: &MemTuple) -> K {
    gammas.gamma2 - (K::from(packed(e.t, e.g)) + gammas.gamma1.scale_base(F::from_u64(e.v as u64)))
}

/// Product of fingerprints over a multiset (`1_K` for the empty multiset).
pub fn product<'a>(gammas: &Gammas, tuples: impl IntoIterator<Item = &'a MemTuple>) -> K {
    tuples
        .into_iter()
        .fold(K::ONE, |acc, e| acc * fingerprint(gammas, e))
}

/// The Nebula balance check `h_is · h_ws == h_rs · h_fs` (spec §1) on
/// products ordered per [`H_RS`](crate::frontends::nebula::layout::H_RS).
pub fn balanced(h: &[K; 4]) -> bool {
    h[H_IS] * h[H_WS] == h[H_RS] * h[H_FS]
}
