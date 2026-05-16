//! Conditional-select (multiplexer) gadgets.
//!
//! Used by F' R1CS for the base-case branch:
//!   - if `i == 0`: `U_{i+1} = u_⊥`
//!   - else:        `U_{i+1} = NIFS.V(...)`
//!
//! The selector bit `s ∈ {0, 1}` is the "is-base-case" indicator. Output:
//!   `mux(s, a, b) = s · a + (1 - s) · b = b + s · (a - b)`
//!
//! Cost per scalar mux: 1 mult constraint (`s · (a - b) = out - b`) plus
//! 1 linear equality if we allocate `out` as a separate witness.
//!
//! `enforce_bit(s)` should be called separately if not already enforced.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};

/// `out = mux(s, a, b)`. If `s = 1`: returns `a`. If `s = 0`: returns `b`.
///
/// Caller is responsible for enforcing `s ∈ {0, 1}` separately (via
/// `boolean::enforce_bit`).
pub fn enforce_mux_var(builder: &mut R1csBuilder, s: Var, a: Var, b: Var) -> Var {
    let s_val = builder.witness()[s.col()];
    let a_val = builder.witness()[a.col()];
    let b_val = builder.witness()[b.col()];
    let out_val = s_val * a_val + (F::ONE - s_val) * b_val;
    let out = builder.alloc(out_val);

    // s · (a - b) = out - b
    let a_minus_b = Lc::from_var(a).add_scaled(&Lc::from_var(b), -F::ONE);
    let s_lc = Lc::from_var(s);
    let out_minus_b = Lc::from_var(out).add_scaled(&Lc::from_var(b), -F::ONE);
    builder.enforce(&s_lc, &a_minus_b, &out_minus_b);

    out
}

/// Vector mux: `out[i] = mux(s, a[i], b[i])` for all i.
pub fn enforce_mux_vec(builder: &mut R1csBuilder, s: Var, a: &[Var], b: &[Var]) -> Vec<Var> {
    assert_eq!(a.len(), b.len(), "mux_vec: a and b length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(&av, &bv)| enforce_mux_var(builder, s, av, bv))
        .collect()
}
