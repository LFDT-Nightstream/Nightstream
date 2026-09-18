//! Projection-checked batched ring action — encoding.md candidate E,
//! prototype stage.
//!
//! Parity: the projection identity accepts exactly the `out` that the
//! trusted `D²`-product gadget (`enforce_ring_mul`) computes, for the
//! same operands. Rejection: tampering any operand, the output, or the
//! quotient breaks a row. Cost: the head-to-head committed-wire count
//! against the `D²` gadget, printed and pinned.
//!
//! β is allocated directly here; in any real integration it MUST be a
//! transcript challenge sampled after all operands and the quotient are
//! committed (commit-then-challenge — the gadget only enforces algebra).

use neo_fold_clean::engine::r1cs_circuit::field_ext::KVar;
use neo_fold_clean::engine::r1cs_circuit::ring_action::{
    enforce_beta_ladder, enforce_eval_at_beta, enforce_ring_action_projection_batch, enforce_ring_mul,
    projection_quotient, PROJECTION_QUOTIENT_LEN,
};
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, Var};
use neo_math::ring::D;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

/// Deterministic SplitMix64 — reproducible operands, zero deps.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// A challenge-set-shaped ρ: coefficients in `[−2, 2]` (SuperNeo
    /// Appendix B.2's strong sampling set).
    fn rho(&mut self) -> [F; D] {
        std::array::from_fn(|_| {
            let v = (self.next() % 5) as i64 - 2;
            if v >= 0 {
                F::from_u64(v as u64)
            } else {
                F::ZERO - F::from_u64((-v) as u64)
            }
        })
    }

    /// A commitment-shaped c: full-range coefficients.
    fn c(&mut self) -> [F; D] {
        std::array::from_fn(|_| F::from_u64(self.next()))
    }
}

fn alloc_coeffs(b: &mut R1csBuilder, coeffs: &[F; D]) -> [Var; D] {
    coeffs.map(|v| b.alloc(v))
}

const PAIRS: usize = 3;

/// Build the full projection circuit for `PAIRS` honest pairs; returns
/// the builder plus the wires the rejection sweep tampers.
struct Fixture {
    b: R1csBuilder,
    rho0: [Var; D],
    out: [Var; D],
    quotient: [Var; PROJECTION_QUOTIENT_LEN],
    out_native: [F; D],
    q_native: [F; PROJECTION_QUOTIENT_LEN],
}

fn build_fixture(seed: u64) -> Fixture {
    let mut rng = Rng(seed);
    let native: Vec<([F; D], [F; D])> = (0..PAIRS).map(|_| (rng.rho(), rng.c())).collect();
    let (out_native, q_native) = projection_quotient(&native);

    let mut b = R1csBuilder::new();
    let beta = KVar::alloc(&mut b, F::from_u64(rng.next()), F::from_u64(rng.next()));
    let powers = enforce_beta_ladder(&mut b, beta, D);
    let wires: Vec<([Var; D], [Var; D])> = native
        .iter()
        .map(|(rho, c)| (alloc_coeffs(&mut b, rho), alloc_coeffs(&mut b, c)))
        .collect();
    let out = alloc_coeffs(&mut b, &out_native);
    let quotient = q_native.map(|v| b.alloc(v));
    let pair_refs: Vec<(&[Var; D], &[Var; D])> = wires.iter().map(|(r, c)| (r, c)).collect();
    enforce_ring_action_projection_batch(&mut b, &powers, &pair_refs, &out, &quotient);
    Fixture {
        rho0: wires[0].0,
        b,
        out,
        quotient,
        out_native,
        q_native,
    }
}

/// The projection identity accepts exactly the batched ring-action
/// result the trusted `D²` gadget computes.
#[test]
fn projection_matches_the_d_squared_gadget() {
    let mut rng = Rng(7);
    let native: Vec<([F; D], [F; D])> = (0..PAIRS).map(|_| (rng.rho(), rng.c())).collect();
    let (out_native, _) = projection_quotient(&native);

    // Reference: Σ_i enforce_ring_mul(ρ_i, c_i) with the D²-product
    // gadget, summed natively from its output wires.
    let mut b = R1csBuilder::new();
    let mut expected = [F::ZERO; D];
    for (rho, c) in &native {
        let rho_w = alloc_coeffs(&mut b, rho);
        let c_w = alloc_coeffs(&mut b, c);
        let out_w = enforce_ring_mul(&mut b, &rho_w, &c_w);
        for (acc, w) in expected.iter_mut().zip(out_w.iter()) {
            *acc += b.witness()[w.col()];
        }
    }
    assert!(b.is_satisfied(), "reference gadget satisfied");
    assert_eq!(out_native, expected, "projection_quotient ≡ D²-gadget batched output");

    // And the projection circuit accepts that same result.
    let f = build_fixture(7);
    assert_eq!(f.out_native, expected);
    assert!(f.b.is_satisfied(), "projection circuit accepts the honest batch");
}

/// Every operand of the identity is load-bearing: tampering the output,
/// the quotient, or an input coefficient breaks a row (restore
/// re-satisfies, so each rejection is attributable).
#[test]
fn projection_rejects_tampered_operands() {
    let mut f = build_fixture(21);
    assert!(f.b.is_satisfied(), "honest baseline");

    // (a) Forged output coefficient — the attack the check exists for:
    // accepting a wrong folded commitment.
    let col = f.out[3].col();
    let honest = f.out_native[3];
    f.b.tamper_witness(col, honest + F::ONE);
    assert!(!f.b.is_satisfied(), "a forged out coefficient must be rejected");
    f.b.tamper_witness(col, honest);
    assert!(f.b.is_satisfied(), "restore");

    // (b) Forged quotient — q is a witness; a lying q must not be able
    // to absorb a wrong out (the identity pins both together).
    let col = f.quotient[10].col();
    let honest = f.q_native[10];
    f.b.tamper_witness(col, honest + F::ONE);
    assert!(!f.b.is_satisfied(), "a forged quotient must be rejected");
    f.b.tamper_witness(col, honest);
    assert!(f.b.is_satisfied(), "restore");

    // (c) Tampered input ρ coefficient — the evaluation wires go stale.
    let col = f.rho0[0].col();
    let honest = f.b.witness()[col];
    f.b.tamper_witness(col, honest + F::ONE);
    assert!(!f.b.is_satisfied(), "a tampered input must be rejected");
}

/// Head-to-head committed-wire cost vs the `D²` gadget, same operands.
/// Prints the table encoding.md's candidate E cites; pins the ≥ 5×
/// per-pair improvement so a regression reopens the design.
#[test]
fn projection_cost_beats_d_squared_materialization() {
    let mut rng = Rng(42);
    let native: Vec<([F; D], [F; D])> = (0..PAIRS).map(|_| (rng.rho(), rng.c())).collect();

    // D² gadget: pairs only (its outputs are LC-derived, no out alloc).
    let mut b_ref = R1csBuilder::new();
    for (rho, c) in &native {
        let rho_w = alloc_coeffs(&mut b_ref, rho);
        let c_w = alloc_coeffs(&mut b_ref, c);
        let _ = enforce_ring_mul(&mut b_ref, &rho_w, &c_w);
    }
    let (ref_cols, ref_rows) = (b_ref.cols(), b_ref.rows());

    let f = build_fixture(42);
    let (proj_cols, proj_rows) = (f.b.cols(), f.b.rows());

    // Marginal per pair: subtract a 0-pair projection circuit (ladder +
    // out/q evaluations are per-step shared/edge costs).
    let shared = {
        let mut b = R1csBuilder::new();
        let beta = KVar::alloc(&mut b, F::from_u64(3), F::from_u64(5));
        let powers = enforce_beta_ladder(&mut b, beta, D);
        let out = alloc_coeffs(&mut b, &[F::ZERO; D]);
        let q: [Var; PROJECTION_QUOTIENT_LEN] = std::array::from_fn(|_| b.alloc(F::ZERO));
        let _ = enforce_eval_at_beta(&mut b, out.as_slice(), &powers);
        let _ = enforce_eval_at_beta(&mut b, q.as_slice(), &powers);
        b.cols()
    };
    let per_pair_proj = (proj_cols - shared) / PAIRS;
    let per_pair_ref = ref_cols / PAIRS;

    println!("== projection vs D²-materialization ({PAIRS} pairs, full-field cols) ==");
    println!("D² gadget       cols {ref_cols:>7}  rows {ref_rows:>7}  (~{per_pair_ref}/pair)");
    println!(
        "projection      cols {proj_cols:>7}  rows {proj_rows:>7}  (~{per_pair_proj}/pair marginal + {shared} shared)"
    );
    println!(
        "per-pair ratio  {:.1}×   (bit-backed implication: ~{}k vs ~{}k committed bits/pair)",
        per_pair_ref as f64 / per_pair_proj as f64,
        per_pair_proj * 64 / 1000,
        per_pair_ref * 64 / 1000,
    );

    assert!(
        per_pair_proj * 5 <= per_pair_ref,
        "projection must stay ≥ 5× cheaper per pair than D² materialization \
         ({per_pair_proj} vs {per_pair_ref} cols)"
    );
}
