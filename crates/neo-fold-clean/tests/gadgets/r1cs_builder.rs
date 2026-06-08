//! Gadget primitive tests: boolean, field_ext (K = F_{q^2}), and small composites.
//!
//! These tests pin the gadget output to native arithmetic. If any of these
//! fail, the higher-level Π_RLC.V and Π_CCS.V circuits will produce wrong
//! results — every other gadget depends on these.

use neo_fold_clean::engine::r1cs_circuit::boolean::{enforce_bit, enforce_low_norm};
use neo_fold_clean::engine::r1cs_circuit::field_ext::{enforce_k_mul, KLc, KVar};
use neo_fold_clean::engine::r1cs_circuit::ring_action::alloc_and_enforce_ring_mul;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use neo_math::ring::{cf, Rq, D};
use neo_math::{KExtensions, F, K};
use p3_field::PrimeCharacteristicRing;

// ── builder diagnostics ─────────────────────────────────────────────────

#[test]
fn unconstrained_columns_reports_allocated_wires_with_no_rows() {
    let mut b = R1csBuilder::new();
    let bound = b.alloc(F::from_u64(7));
    let unbound = b.alloc(F::from_u64(11));

    b.enforce_eq(&Lc::from_var(bound), &Lc::from_const(F::from_u64(7)));

    assert_eq!(
        b.unconstrained_columns(),
        vec![unbound.col()],
        "audit helper must report allocated columns that never appear in A/B/C rows"
    );
}

// ── boolean ───────────────────────────────────────────────────────────────

#[test]
fn enforce_bit_accepts_0_and_1() {
    for v in [F::ZERO, F::ONE] {
        let mut b = R1csBuilder::new();
        let var = b.alloc(v);
        enforce_bit(&mut b, var);
        assert!(b.is_satisfied(), "bit constraint rejected v = {:?}", v);
    }
}

#[test]
fn enforce_bit_rejects_2() {
    let mut b = R1csBuilder::new();
    let var = b.alloc(F::from_u64(2));
    enforce_bit(&mut b, var);
    assert!(!b.is_satisfied(), "bit constraint accepted v = 2");
}

#[test]
fn enforce_low_norm_b3_accepts_0_1_2() {
    for v in [F::ZERO, F::ONE, F::from_u64(2)] {
        let mut b = R1csBuilder::new();
        let var = b.alloc(v);
        enforce_low_norm(&mut b, var, 3);
        assert!(b.is_satisfied(), "b=3 range check rejected v = {:?}", v);
    }
}

#[test]
fn enforce_low_norm_b3_rejects_3() {
    let mut b = R1csBuilder::new();
    let var = b.alloc(F::from_u64(3));
    enforce_low_norm(&mut b, var, 3);
    assert!(!b.is_satisfied(), "b=3 range check accepted v = 3");
}

// ── field_ext (K-mul) ─────────────────────────────────────────────────────

fn k_to_limbs(k: K) -> (F, F) {
    let [c0, c1] = k.as_coeffs();
    (c0, c1)
}

fn alloc_klc_from_k(b: &mut R1csBuilder, value: K) -> KVar {
    let (c0, c1) = k_to_limbs(value);
    KVar::alloc(b, c0, c1)
}

#[test]
fn k_mul_matches_native_for_simple_inputs() {
    let cases: [(K, K); 5] = [
        (K::ZERO, K::ONE),
        (K::ONE, K::ONE),
        (
            K::from_coeffs([F::from_u64(3), F::from_u64(5)]),
            K::from_coeffs([F::from_u64(7), F::from_u64(11)]),
        ),
        (
            K::from_coeffs([-F::ONE, F::from_u64(2)]),
            K::from_coeffs([F::from_u64(4), -F::ONE]),
        ),
        (
            K::from_coeffs([F::from_u64(123456789), F::from_u64(987654321)]),
            K::from_coeffs([F::from_u64(42), F::from_u64(99)]),
        ),
    ];

    for (a_k, b_k) in cases {
        let expected = a_k * b_k;
        let (exp_c0, exp_c1) = k_to_limbs(expected);

        let mut b = R1csBuilder::new();
        let a_var = alloc_klc_from_k(&mut b, a_k);
        let b_var = alloc_klc_from_k(&mut b, b_k);
        let out = enforce_k_mul(&mut b, &KLc::from_var(a_var), &KLc::from_var(b_var));

        assert!(
            b.is_satisfied(),
            "K-mul circuit unsatisfied for a={:?}, b={:?} (first bad row: {:?})",
            a_k,
            b_k,
            b.first_unsatisfied_row()
        );
        assert_eq!(
            b.witness()[out.c0.col()],
            exp_c0,
            "K-mul c0 mismatch for a={:?}, b={:?}",
            a_k,
            b_k
        );
        assert_eq!(
            b.witness()[out.c1.col()],
            exp_c1,
            "K-mul c1 mismatch for a={:?}, b={:?}",
            a_k,
            b_k
        );
    }
}

#[test]
fn k_mul_rejects_tampered_output() {
    let a_k = K::from_coeffs([F::from_u64(7), F::from_u64(11)]);
    let b_k = K::from_coeffs([F::from_u64(13), F::from_u64(17)]);

    let mut b = R1csBuilder::new();
    let a_var = alloc_klc_from_k(&mut b, a_k);
    let b_var = alloc_klc_from_k(&mut b, b_k);
    let out = enforce_k_mul(&mut b, &KLc::from_var(a_var), &KLc::from_var(b_var));

    assert!(b.is_satisfied(), "baseline must be satisfied before tampering");

    let tampered = b.witness()[out.c0.col()] + F::ONE;
    b.tamper_witness(out.c0.col(), tampered);

    assert!(!b.is_satisfied(), "K-mul circuit accepted a tampered output c0");
}

// ── compose: linear combination over K vector (the pattern Π_CCS.V uses) ──

/// Sanity test for the constraint pattern used by Π_RLC.V and Π_CCS.V:
/// `target = Σ_i scalar_i · k_value_i` where `scalar_i ∈ F` and
/// `k_value_i ∈ K`. The combination is linear, so it doesn't need K-mul —
/// just lane-wise scaled sums on (c0, c1).
#[test]
fn k_linear_combination_matches_native() {
    let scalars = [F::from_u64(1), F::from_u64(2), F::from_u64(4)];
    let inputs = [
        K::from_coeffs([F::from_u64(11), F::from_u64(22)]),
        K::from_coeffs([F::from_u64(33), F::from_u64(44)]),
        K::from_coeffs([F::from_u64(55), F::from_u64(66)]),
    ];

    let expected: K = inputs
        .iter()
        .zip(scalars.iter())
        .map(|(k, &s)| k.scale_base(s))
        .fold(K::ZERO, |acc, x| acc + x);
    let (exp_c0, exp_c1) = k_to_limbs(expected);

    let mut b = R1csBuilder::new();
    let input_vars: Vec<KVar> = inputs
        .iter()
        .map(|&k| alloc_klc_from_k(&mut b, k))
        .collect();
    let target = alloc_klc_from_k(&mut b, expected);

    // Build the LC and constrain target == LC.
    let mut combo_c0 = Lc::zero();
    let mut combo_c1 = Lc::zero();
    for (v, &s) in input_vars.iter().zip(scalars.iter()) {
        combo_c0.add_term(v.c0, s);
        combo_c1.add_term(v.c1, s);
    }
    b.enforce_eq(&Lc::from_var(target.c0), &combo_c0);
    b.enforce_eq(&Lc::from_var(target.c1), &combo_c1);

    assert!(b.is_satisfied(), "K-linear-combination unsatisfied");
    assert_eq!(b.witness()[target.c0.col()], exp_c0);
    assert_eq!(b.witness()[target.c1.col()], exp_c1);
}

// ── ring_action (Π_RLC.V's core gadget) ───────────────────────────────────

fn rq_from_seed(seed: u64) -> Rq {
    // Deterministic small ρ-like value for tests. We don't need it to be from
    // the strong sampling set; we just need ring_mul to match Rq::mul.
    let mut coeffs = [F::ZERO; D];
    let mut s = seed;
    for slot in coeffs.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *slot = F::from_u64(s & 0xFF);
    }
    Rq(coeffs)
}

#[test]
fn ring_mul_circuit_matches_native_rq_mul() {
    let cases = [
        (rq_from_seed(1), rq_from_seed(2)),
        (rq_from_seed(3), rq_from_seed(4)),
        (Rq::one(), rq_from_seed(5)),
        (rq_from_seed(7), Rq::one()),
        (Rq::zero(), rq_from_seed(11)),
    ];

    for (rho_rq, c_rq) in cases {
        let expected = rho_rq.mul(&c_rq);
        let exp_coeffs = cf(expected);
        let rho_vals = cf(rho_rq);
        let c_vals = cf(c_rq);

        let mut b = R1csBuilder::new();
        let out = alloc_and_enforce_ring_mul(&mut b, &rho_vals, &c_vals);

        assert!(
            b.is_satisfied(),
            "ring_mul unsatisfied for ρ-seed pair (first bad row: {:?})",
            b.first_unsatisfied_row()
        );
        for (m, &exp) in exp_coeffs.iter().enumerate() {
            let got = b.witness()[out[m].col()];
            assert_eq!(
                got, exp,
                "ring_mul coefficient {} mismatch: gadget={:?}, native={:?}",
                m, got, exp
            );
        }
    }
}

#[test]
fn ring_mul_toom3_circuit_matches_native_with_lower_row_count() {
    use neo_fold_clean::engine::r1cs_circuit::ring_action::alloc_and_enforce_ring_mul_toom3;

    let rho = rq_from_seed(31);
    let c = rq_from_seed(37);
    let expected = cf(rho.mul(&c));
    let mut b = R1csBuilder::new();
    let out = alloc_and_enforce_ring_mul_toom3(&mut b, &cf(rho), &cf(c));

    assert!(
        b.is_satisfied(),
        "toom3 ring_mul unsatisfied (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    for (idx, &want) in expected.iter().enumerate() {
        assert_eq!(b.witness()[out[idx].col()], want, "toom3 coeff {idx}");
    }
    assert_eq!(
        b.rows(),
        5 * 18 * 18 + D,
        "3-way ring_mul should use five 18x18 products plus D output rows"
    );
}

#[test]
fn ring_mul_circuit_rejects_tampered_output() {
    let rho = rq_from_seed(13);
    let c = rq_from_seed(17);
    let mut b = R1csBuilder::new();
    let out = alloc_and_enforce_ring_mul(&mut b, &cf(rho), &cf(c));

    assert!(b.is_satisfied(), "baseline must be satisfied");

    let target_col = out[0].col();
    let tampered = b.witness()[target_col] + F::ONE;
    b.tamper_witness(target_col, tampered);

    assert!(!b.is_satisfied(), "ring_mul accepted a tampered output coefficient");
}

#[test]
fn ring_mul_circuit_rejects_tampered_rho_coefficient() {
    let rho = rq_from_seed(19);
    let c = rq_from_seed(23);
    let mut b = R1csBuilder::new();
    // We need to grab the ρ wires; use enforce_ring_mul directly so we can
    // tamper a specific input column.
    use neo_fold_clean::engine::r1cs_circuit::ring_action::enforce_ring_mul;
    let rho_vals = cf(rho);
    let c_vals = cf(c);
    let mut rho_vars = [Var::ONE; D];
    for (slot, &v) in rho_vars.iter_mut().zip(rho_vals.iter()) {
        *slot = b.alloc(v);
    }
    let mut c_vars = [Var::ONE; D];
    for (slot, &v) in c_vars.iter_mut().zip(c_vals.iter()) {
        *slot = b.alloc(v);
    }
    let _out = enforce_ring_mul(&mut b, &rho_vars, &c_vars);

    assert!(b.is_satisfied(), "baseline must be satisfied");

    let rho_col = rho_vars[0].col();
    let tampered = b.witness()[rho_col] + F::ONE;
    b.tamper_witness(rho_col, tampered);

    assert!(!b.is_satisfied(), "ring_mul accepted a tampered ρ coefficient");
}

// Silence "unused" warnings on imports that may be used only in some tests.
fn _unused(_: Var) {}
