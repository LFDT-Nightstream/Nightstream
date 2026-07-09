//! Bit-backed layout + encoder + decoder for the projection-checked
//! ring action (Road A / encoding.md candidate E; security-note
//! Lemma 5) inside the `enc(F')` image.
//!
//! Owns: the committed-width contract of the three projection region
//! kinds and their native fill —
//!
//! - one **pair** `(ρ_i, c_i-component)`: operand lanes plus the two
//!   β-evaluation partial-product runs and the `ρ_i(β)·c_i(β)`
//!   Karatsuba slot;
//! - one **identity** (per output ring component, per client): the
//!   claimed output lanes, the division quotient `q`, both their
//!   β-evaluations, and the `q(β)·Φ(β)` Karatsuba slot;
//! - the per-step **shared** region: β and its power ladder through
//!   `β^D` (one Karatsuba slot per rung).
//!
//! Does not own: constraint emission. Phase A ships the layout, the
//! native fill, and round-trip parity (the same discipline
//! `ring_action_trace` used in Phase 1.2/1.3); the semantic CCS rows —
//! evaluation sums, the Karatsuba relations, the final identity — are
//! the tracked next phase (`ivc_invariants.rs::
//! projection_shell_semantic_rows_must_be_enforced`). Committed-bit
//! range enforcement is the protocol's NC check, as everywhere in the
//! image.
//!
//! Every committed value here is one canonical-u64 lane (64 bits); K
//! values are two lanes `(c0, c1)`. Karatsuba slots follow
//! `engine::r1cs_circuit::field_ext::KMulIntermediates`:
//! `p = a0·b0`, `q = a1·b1`, `r = (a0+a1)·(b0+b1)`, and the output is
//! `(p + W·q, r − p − q)` with `W` the extension's binomial constant.

use neo_math::field::KExtensions;
use neo_math::ring::D;
use neo_math::{F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::engine::r1cs_circuit::ring_action::{projection_quotient, PROJECTION_QUOTIENT_LEN};

/// Lanes for one β-evaluation of a length-`n` coefficient vector: the
/// `j = 1..n` partial products (two lanes each — the `j = 0` term is
/// linear) plus the bound sum (two lanes).
const fn eval_lanes(n: usize) -> usize {
    2 * (n - 1) + 2
}

/// One Karatsuba K-mult slot: `p, q, r` intermediates + `(c0, c1)` out.
const KMUL_SLOT_LANES: usize = 5;

/// Lanes of one projection **pair**: `ρ` (D) + `c` (D) + two
/// evaluations + the pair-term Karatsuba slot.
pub const PROJECTION_PAIR_LANES: usize = 2 * D + 2 * eval_lanes(D) + KMUL_SLOT_LANES;
/// Lanes of one projection **identity**: `out` (D) + `q` (D − 1) +
/// their evaluations + the `q·Φ(β)` Karatsuba slot.
pub const PROJECTION_IDENTITY_LANES: usize =
    D + PROJECTION_QUOTIENT_LEN + eval_lanes(D) + eval_lanes(PROJECTION_QUOTIENT_LEN) + KMUL_SLOT_LANES;
/// Lanes of the per-step **shared** region: β (2) + the `β^0..β^D`
/// ladder values (2 each) + one Karatsuba triple per rung `1..=D`.
pub const PROJECTION_SHARED_LANES: usize = 2 + 2 * (D + 1) + 3 * D;

/// Bits per lane (canonical u64).
pub const LANE_BITS: usize = 64;

/// Region widths in bits.
pub const PROJECTION_PAIR_BITS: usize = PROJECTION_PAIR_LANES * LANE_BITS;
/// See [`PROJECTION_PAIR_BITS`].
pub const PROJECTION_IDENTITY_BITS: usize = PROJECTION_IDENTITY_LANES * LANE_BITS;
/// See [`PROJECTION_PAIR_BITS`].
pub const PROJECTION_SHARED_BITS: usize = PROJECTION_SHARED_LANES * LANE_BITS;

/// The extension's binomial constant `W` (`u² = W`), read off
/// `neo_math::K` so this encoder can never disagree with native K
/// arithmetic (same derivation as `S_mem`'s circuit builder).
fn binomial_w() -> F {
    let u = K::from_coeffs([F::ZERO, F::ONE]);
    let coeffs = (u * u).as_coeffs();
    assert!(coeffs[1] == F::ZERO, "K must be a binomial extension");
    coeffs[0]
}

fn k_limbs(v: K) -> [F; 2] {
    let (c0, c1) = v.to_limbs_u64();
    [F::from_u64(c0), F::from_u64(c1)]
}

/// Push one Karatsuba slot for `a · b` in K; returns the product.
fn push_kmul(lanes: &mut Vec<F>, a: K, b: K) -> K {
    let [a0, a1] = k_limbs(a);
    let [b0, b1] = k_limbs(b);
    let p = a0 * b0;
    let q = a1 * b1;
    let r = (a0 + a1) * (b0 + b1);
    let out = a * b;
    let [o0, o1] = k_limbs(out);
    debug_assert_eq!(o0, p + binomial_w() * q, "Karatsuba c0 identity");
    debug_assert_eq!(o1, r - p - q, "Karatsuba c1 identity");
    lanes.extend([p, q, r, o0, o1]);
    out
}

/// Push one β-evaluation run for `coeffs`: partial products for
/// `j = 1..n`, then the bound sum. Returns the evaluation.
fn push_eval(lanes: &mut Vec<F>, coeffs: &[F], powers: &[K]) -> K {
    let mut sum = K::from(coeffs[0]);
    for (j, &coeff) in coeffs.iter().enumerate().skip(1) {
        let term = powers[j].scale_base(coeff);
        let [t0, t1] = k_limbs(term);
        lanes.extend([t0, t1]);
        sum += term;
    }
    let [s0, s1] = k_limbs(sum);
    lanes.extend([s0, s1]);
    sum
}

/// Native fill of the per-step shared region; returns the power list
/// `β^0..β^D` for the pair/identity encoders.
pub fn encode_projection_shared(beta: K) -> (Vec<F>, Vec<K>) {
    let mut lanes = Vec::with_capacity(PROJECTION_SHARED_LANES);
    lanes.extend(k_limbs(beta));
    let mut powers = Vec::with_capacity(D + 1);
    powers.push(K::ONE);
    lanes.extend(k_limbs(K::ONE));
    for k in 1..=D {
        let next = push_kmul(&mut lanes, powers[k - 1], beta);
        // Slot order per rung: Karatsuba triple + product lanes; the
        // product lanes double as the ladder value.
        powers.push(next);
    }
    debug_assert_eq!(lanes.len(), PROJECTION_SHARED_LANES);
    (lanes, powers)
}

/// Native fill of one pair region; returns `ρ(β)·c(β)`.
pub fn encode_projection_pair(rho: &[F; D], c: &[F; D], powers: &[K]) -> (Vec<F>, K) {
    let mut lanes = Vec::with_capacity(PROJECTION_PAIR_LANES);
    lanes.extend_from_slice(rho);
    lanes.extend_from_slice(c);
    let rho_eval = push_eval(&mut lanes, rho, powers);
    let c_eval = push_eval(&mut lanes, c, powers);
    let term = push_kmul(&mut lanes, rho_eval, c_eval);
    debug_assert_eq!(lanes.len(), PROJECTION_PAIR_LANES);
    (lanes, term)
}

/// Native fill of one identity region for the batched inputs it
/// consumes; returns the identity residual `Σ terms − out(β) − q(β)·Φ(β)`
/// (zero for an honest fill — the future semantic rows enforce it).
pub fn encode_projection_identity(pairs: &[([F; D], [F; D])], powers: &[K], pair_terms: &[K]) -> (Vec<F>, K) {
    let (out, quotient) = projection_quotient(pairs);
    let mut lanes = Vec::with_capacity(PROJECTION_IDENTITY_LANES);
    lanes.extend_from_slice(&out);
    lanes.extend_from_slice(&quotient);
    let out_eval = push_eval(&mut lanes, &out, powers);
    let q_eval = push_eval(&mut lanes, &quotient, powers);
    // Φ(β) = β^D + β^{27} + 1 — a linear form over the shared ladder.
    let phi_beta = powers[D] + powers[27] + K::ONE;
    let q_phi = push_kmul(&mut lanes, q_eval, phi_beta);
    debug_assert_eq!(lanes.len(), PROJECTION_IDENTITY_LANES);
    let residual = pair_terms.iter().copied().sum::<K>() - out_eval - q_phi;
    (lanes, residual)
}

/// Decode a lane vector back to canonical u64 field values (identity
/// codec at lane granularity — bits are the NC check's concern). Used
/// by parity tests.
pub fn decode_lanes(lanes: &[F]) -> Vec<u64> {
    lanes.iter().map(|f| f.as_canonical_u64()).collect()
}
