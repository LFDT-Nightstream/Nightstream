//! Sumcheck verifier primitive tests: γ-powers, Horner eval, single round,
//! eq product. Tests pin gadget output to native K arithmetic.

use neo_fold_clean::engine::r1cs_circuit::field_ext::{KLc, KVar};
use neo_fold_clean::engine::r1cs_circuit::{
    enforce_eq_k, enforce_gamma_indexed_sum, enforce_k_mul, enforce_norm_check_b2, enforce_r1cs_f_term,
    enforce_sumcheck_round, enforce_sumcheck_walk, gamma_powers, horner_eval_k, R1csBuilder,
};
use neo_math::{KExtensions, F, K};
use p3_field::{Field, PrimeCharacteristicRing};

fn k(a: u64, b: u64) -> K {
    K::from_coeffs([F::from_u64(a), F::from_u64(b)])
}

fn alloc_kvar(b: &mut R1csBuilder, v: K) -> KVar {
    let [c0, c1] = v.as_coeffs();
    KVar::alloc(b, c0, c1)
}

fn kvar_value(b: &R1csBuilder, v: KVar) -> K {
    let c0 = b.witness()[v.c0.col()];
    let c1 = b.witness()[v.c1.col()];
    K::from_coeffs([c0, c1])
}

// ── γ-power table ─────────────────────────────────────────────────────────

#[test]
fn gamma_powers_matches_native() {
    let gamma = k(5, 3);
    let mut b = R1csBuilder::new();
    let gamma_var = alloc_kvar(&mut b, gamma);
    let table = gamma_powers(&mut b, gamma_var, 6);

    assert!(b.is_satisfied(), "γ-power table circuit unsatisfied");

    let mut expected = K::ONE;
    for (i, var) in table.iter().enumerate() {
        let got = kvar_value(&b, *var);
        assert_eq!(got, expected, "γ^{i} mismatch: gadget={:?}, native={:?}", got, expected);
        expected *= gamma;
    }
}

#[test]
fn gamma_powers_rejects_tampered_first_power() {
    let gamma = k(7, 2);
    let mut b = R1csBuilder::new();
    let gamma_var = alloc_kvar(&mut b, gamma);
    let table = gamma_powers(&mut b, gamma_var, 4);

    assert!(b.is_satisfied(), "baseline must be satisfied");

    let target = table[1].c0.col();
    let tampered = b.witness()[target] + F::ONE;
    b.tamper_witness(target, tampered);

    assert!(!b.is_satisfied(), "γ-power table accepted a tampered γ^1.c0");
}

// ── Horner evaluation ─────────────────────────────────────────────────────

#[test]
fn horner_eval_matches_native_polynomial_evaluation() {
    let coeffs_k = vec![k(1, 0), k(2, 1), k(3, 4), k(5, 0), k(0, 7)]; // degree 4
    let r = k(11, 13);

    // Native: Σ coeffs[i] * r^i.
    let mut expected = K::ZERO;
    let mut r_pow = K::ONE;
    for c in &coeffs_k {
        expected += *c * r_pow;
        r_pow *= r;
    }

    let mut b = R1csBuilder::new();
    let coeff_vars: Vec<KVar> = coeffs_k
        .iter()
        .copied()
        .map(|c| alloc_kvar(&mut b, c))
        .collect();
    let r_var = alloc_kvar(&mut b, r);
    let out = horner_eval_k(&mut b, &coeff_vars, r_var);

    assert!(b.is_satisfied(), "Horner circuit unsatisfied");
    assert_eq!(kvar_value(&b, out), expected, "Horner eval mismatch");
}

#[test]
fn horner_eval_rejects_tampered_output() {
    let coeffs_k = vec![k(1, 0), k(2, 1), k(3, 4)];
    let r = k(7, 11);
    let mut b = R1csBuilder::new();
    let coeff_vars: Vec<KVar> = coeffs_k
        .iter()
        .copied()
        .map(|c| alloc_kvar(&mut b, c))
        .collect();
    let r_var = alloc_kvar(&mut b, r);
    let out = horner_eval_k(&mut b, &coeff_vars, r_var);

    assert!(b.is_satisfied(), "baseline must be satisfied");

    let target = out.c0.col();
    let tampered = b.witness()[target] + F::ONE;
    b.tamper_witness(target, tampered);

    assert!(!b.is_satisfied(), "Horner accepted a tampered output");
}

// ── single sumcheck round ─────────────────────────────────────────────────

/// Build a degree-`d` polynomial with random K coefficients, compute its
/// `g(0) + g(1)` claim, then verify the gadget accepts and returns g(r).
#[test]
fn sumcheck_round_accepts_honest_polynomial_and_returns_g_at_r() {
    let coeffs = vec![k(2, 0), k(3, 1), k(5, 2), k(0, 4), k(11, 13)]; // degree 4
    let r = k(17, 19);
    let claim_in_k: K = {
        // g(0) + g(1) = coeffs[0] + Σ coeffs[i] = 2·coeffs[0] + Σ_{i≥1} coeffs[i].
        let mut acc = coeffs[0];
        for c in &coeffs {
            acc += *c;
        }
        acc
    };

    // Native g(r):
    let mut g_at_r = K::ZERO;
    let mut r_pow = K::ONE;
    for c in &coeffs {
        g_at_r += *c * r_pow;
        r_pow *= r;
    }

    let mut b = R1csBuilder::new();
    let coeff_vars: Vec<KVar> = coeffs
        .iter()
        .copied()
        .map(|c| alloc_kvar(&mut b, c))
        .collect();
    let r_var = alloc_kvar(&mut b, r);
    let claim_in_var = alloc_kvar(&mut b, claim_in_k);

    let claim_out = enforce_sumcheck_round(&mut b, &coeff_vars, r_var, claim_in_var);

    assert!(
        b.is_satisfied(),
        "honest sumcheck round must accept (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    assert_eq!(kvar_value(&b, claim_out), g_at_r, "returned claim_out must equal g(r)");
}

#[test]
fn sumcheck_round_rejects_inconsistent_claim() {
    let coeffs = vec![k(1, 0), k(2, 3), k(5, 7)]; // degree 2
    let r = k(13, 17);
    // Wrong claim: subtract 1 to make it inconsistent.
    let wrong_claim = {
        let mut acc = coeffs[0];
        for c in &coeffs {
            acc += *c;
        }
        acc - K::ONE
    };

    let mut b = R1csBuilder::new();
    let coeff_vars: Vec<KVar> = coeffs
        .iter()
        .copied()
        .map(|c| alloc_kvar(&mut b, c))
        .collect();
    let r_var = alloc_kvar(&mut b, r);
    let claim_in_var = alloc_kvar(&mut b, wrong_claim);

    let _ = enforce_sumcheck_round(&mut b, &coeff_vars, r_var, claim_in_var);

    assert!(!b.is_satisfied(), "sumcheck round accepted an inconsistent claim");
}

#[test]
fn sumcheck_round_rejects_tampered_coefficient() {
    let coeffs = vec![k(1, 0), k(2, 3), k(5, 7), k(11, 0)];
    let r = k(13, 17);
    let claim_in_k = {
        let mut acc = coeffs[0];
        for c in &coeffs {
            acc += *c;
        }
        acc
    };

    let mut b = R1csBuilder::new();
    let coeff_vars: Vec<KVar> = coeffs
        .iter()
        .copied()
        .map(|c| alloc_kvar(&mut b, c))
        .collect();
    let r_var = alloc_kvar(&mut b, r);
    let claim_in_var = alloc_kvar(&mut b, claim_in_k);

    let _ = enforce_sumcheck_round(&mut b, &coeff_vars, r_var, claim_in_var);

    assert!(b.is_satisfied(), "baseline must be satisfied");

    let target = coeff_vars[2].c0.col();
    let tampered = b.witness()[target] + F::ONE;
    b.tamper_witness(target, tampered);

    assert!(
        !b.is_satisfied(),
        "sumcheck round accepted a tampered round-polynomial coefficient"
    );
}

// ── eq product ────────────────────────────────────────────────────────────

#[test]
fn eq_k_matches_native_for_paper_formula() {
    let a = [k(2, 3), k(5, 7), k(11, 13), k(0, 1)];
    let b = [k(17, 19), k(0, 4), k(8, 0), k(1, 1)];

    // Native: Π (a_i · b_i + (1 - a_i)(1 - b_i)).
    let mut expected = K::ONE;
    for (av, bv) in a.iter().zip(b.iter()) {
        let term = *av * *bv + (K::ONE - *av) * (K::ONE - *bv);
        expected *= term;
    }

    let mut bd = R1csBuilder::new();
    let a_vars: Vec<KVar> = a.iter().copied().map(|v| alloc_kvar(&mut bd, v)).collect();
    let b_vars: Vec<KVar> = b.iter().copied().map(|v| alloc_kvar(&mut bd, v)).collect();
    let out = enforce_eq_k(&mut bd, &a_vars, &b_vars);

    assert!(bd.is_satisfied(), "eq_k circuit unsatisfied");
    assert_eq!(kvar_value(&bd, out), expected, "eq_k mismatch vs native");
}

#[test]
fn eq_k_evaluates_to_one_when_a_equals_b_on_hypercube() {
    // eq(x, x) = 1 for x ∈ {0,1}^ell  (paper §4 Preliminaries).
    let zeros_ones = [k(0, 0), k(1, 0), k(0, 0), k(1, 0)];
    let mut bd = R1csBuilder::new();
    let av: Vec<KVar> = zeros_ones
        .iter()
        .copied()
        .map(|v| alloc_kvar(&mut bd, v))
        .collect();
    let bv: Vec<KVar> = zeros_ones
        .iter()
        .copied()
        .map(|v| alloc_kvar(&mut bd, v))
        .collect();
    let out = enforce_eq_k(&mut bd, &av, &bv);

    assert!(bd.is_satisfied(), "eq_k(x,x) circuit unsatisfied");
    assert_eq!(kvar_value(&bd, out), K::ONE, "eq_k(x,x) on hypercube must equal 1");
}

#[test]
fn eq_k_evaluates_to_zero_when_disagreeing_on_hypercube() {
    // eq(x, y) = 0 if x and y are distinct hypercube points.
    let x = [k(0, 0), k(1, 0)];
    let y = [k(1, 0), k(0, 0)];
    let mut bd = R1csBuilder::new();
    let av: Vec<KVar> = x.iter().copied().map(|v| alloc_kvar(&mut bd, v)).collect();
    let bv: Vec<KVar> = y.iter().copied().map(|v| alloc_kvar(&mut bd, v)).collect();
    let out = enforce_eq_k(&mut bd, &av, &bv);

    assert!(bd.is_satisfied(), "eq_k(x,y) circuit unsatisfied");
    assert_eq!(
        kvar_value(&bd, out),
        K::ZERO,
        "eq_k(x,y) on distinct hypercube points must equal 0"
    );
}

#[test]
fn eq_k_rejects_tampered_a() {
    let a = [k(2, 1), k(3, 4)];
    let b = [k(5, 0), k(7, 11)];
    let mut bd = R1csBuilder::new();
    let av: Vec<KVar> = a.iter().copied().map(|v| alloc_kvar(&mut bd, v)).collect();
    let bv: Vec<KVar> = b.iter().copied().map(|v| alloc_kvar(&mut bd, v)).collect();
    let _ = enforce_eq_k(&mut bd, &av, &bv);

    assert!(bd.is_satisfied(), "baseline must be satisfied");

    let target = av[0].c0.col();
    let tampered = bd.witness()[target] + F::ONE;
    bd.tamper_witness(target, tampered);

    assert!(!bd.is_satisfied(), "eq_k accepted a tampered a[0].c0");
}

// ── multi-round sumcheck walk ────────────────────────────────────────────

#[test]
fn sumcheck_walk_threads_claim_through_multiple_rounds() {
    // Three rounds. For each round q, the prover picks a degree-2 polynomial
    // g_q(X) = a + b·X + c·X². The chain is:
    //   claim_0 = T
    //   claim_{q+1} = g_q(r_q)
    //   each round must satisfy g_q(0) + g_q(1) = claim_q.
    let rs = [k(7, 0), k(11, 13), k(0, 5)];

    let g0 = [k(3, 1), k(2, 0), k(1, 0)]; // g0(0)+g0(1) = 3 + (3+2+1) = 9 (with K::ZERO imag carryovers)
    let claim0 = {
        let mut s = g0[0];
        for c in &g0 {
            s += *c;
        }
        s
    };
    let claim1 = {
        let mut acc = K::ZERO;
        let mut r_pow = K::ONE;
        for c in &g0 {
            acc += *c * r_pow;
            r_pow *= rs[0];
        }
        acc
    };

    let g1 = [k(4, 0), k(2, 7), k(0, 1)];
    // g1(0)+g1(1) must equal claim1 — choose g1[0] freely, then adjust to enforce.
    // For test simplicity, just set claim1_expected to whatever g1(0)+g1(1) is, and chain forward.
    let claim1_via_g1 = {
        let mut s = g1[0];
        for c in &g1 {
            s += *c;
        }
        s
    };
    let _ = claim1; // unused; we use claim1_via_g1 so the chain stays consistent
    let claim2_via_g1 = {
        let mut acc = K::ZERO;
        let mut r_pow = K::ONE;
        for c in &g1 {
            acc += *c * r_pow;
            r_pow *= rs[1];
        }
        acc
    };

    let g2 = [k(0, 0), k(0, 0), k(0, 0)]; // zero polynomial: g2(0)+g2(1) = 0; g2(r) = 0
                                          // claim2 must equal 0 for honesty. Let's set claim_initial so the chain works:
                                          //   initial_claim = claim0_via_g0 such that the g_0 step works,
                                          //   the chain proceeds through g_1 needing claim_in = g_0(r_0),
                                          //   then through g_2 needing claim_in = g_1(r_1) = 0.
                                          //
                                          // We'll construct a chain where g_0 and g_1 are honest, g_2 is zero, and
                                          // we check that the chain accepts when the final claim chain is satisfied.

    // Build a self-consistent test by constructing claim0 and g_0 freely,
    // then computing claim1 from g_0(r_0), then constructing g_1 to satisfy
    // g_1(0)+g_1(1) = claim1, then computing claim2 from g_1(r_1), then
    // constructing g_2 to satisfy g_2(0)+g_2(1) = claim2.
    let rounds = build_consistent_sumcheck_chain(&rs, claim0);

    let mut b = R1csBuilder::new();
    let initial_var = alloc_kvar(&mut b, claim0);
    let r_vars: Vec<KVar> = rs.iter().copied().map(|v| alloc_kvar(&mut b, v)).collect();
    let round_vars: Vec<Vec<KVar>> = rounds
        .iter()
        .map(|coeffs| {
            coeffs
                .iter()
                .copied()
                .map(|v| alloc_kvar(&mut b, v))
                .collect()
        })
        .collect();

    let final_v = enforce_sumcheck_walk(&mut b, &round_vars, &r_vars, initial_var);

    assert!(
        b.is_satisfied(),
        "sumcheck walk must accept a consistent chain (first bad row: {:?})",
        b.first_unsatisfied_row()
    );

    // Final v should equal g_{last}(r_{last}).
    let expected_final = {
        let last = &rounds[rounds.len() - 1];
        let r = rs[rs.len() - 1];
        let mut acc = K::ZERO;
        let mut r_pow = K::ONE;
        for c in last {
            acc += *c * r_pow;
            r_pow *= r;
        }
        acc
    };
    assert_eq!(kvar_value(&b, final_v), expected_final, "final v mismatch");

    let _ = (g0, g1, g2, claim1_via_g1, claim2_via_g1); // silence
}

/// Construct a self-consistent sumcheck chain: pick arbitrary g_0, then for
/// each subsequent round pick coefficients[1..] freely and set coefficients[0]
/// so that `g(0) + g(1) == previous claim`.
fn build_consistent_sumcheck_chain(rs: &[K], initial_claim: K) -> Vec<Vec<K>> {
    let mut chain = Vec::with_capacity(rs.len());
    let mut claim = initial_claim;
    let mut seed: u64 = 0xCAFE_BABE;
    for r in rs {
        // Pick coeffs[1], coeffs[2] freely; solve for coeffs[0].
        // g(0) + g(1) = coeffs[0] + (coeffs[0] + coeffs[1] + coeffs[2]) = 2·coeffs[0] + coeffs[1] + coeffs[2].
        // → coeffs[0] = (claim - coeffs[1] - coeffs[2]) / 2.
        let c1 = next_k(&mut seed);
        let c2 = next_k(&mut seed);
        let two_inv = K::from_u64(2).inverse();
        let c0 = (claim - c1 - c2) * two_inv;
        let coeffs = vec![c0, c1, c2];

        // Update claim to g(r).
        let mut next_claim = K::ZERO;
        let mut r_pow = K::ONE;
        for c in &coeffs {
            next_claim += *c * r_pow;
            r_pow *= *r;
        }
        chain.push(coeffs);
        claim = next_claim;
    }
    chain
}

fn next_k(seed: &mut u64) -> K {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let c0 = F::from_u64(*seed & 0xFFFF);
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let c1 = F::from_u64(*seed & 0xFFFF);
    K::from_coeffs([c0, c1])
}

// ── γ-indexed sum (T / E_inner pattern) ──────────────────────────────────

#[test]
fn gamma_indexed_sum_matches_native_t_pattern() {
    // Synthetic: k=2 carried claims × t=3 matrices × d=4 lanes = 24 terms.
    // Indices I(h,j,ℓ) = h + k·(j-1) + k·t·(ℓ-1).
    let k_arity = 2;
    let t = 3;
    let d = 4;
    let gamma = k(7, 11);
    let max_idx = k_arity * t * d - 1;

    // Build native γ-power table.
    let mut gamma_pows = vec![K::ONE; max_idx + 1];
    for i in 1..=max_idx {
        gamma_pows[i] = gamma_pows[i - 1] * gamma;
    }

    // Build the (indices, k_values) pair using the paper's I(h,j,ℓ) layout.
    let mut indices = Vec::new();
    let mut k_values_k = Vec::new();
    let mut seed: u64 = 0x1234;
    for h in 0..k_arity {
        for j_minus_one in 0..t {
            for l_minus_one in 0..d {
                let idx = h + k_arity * j_minus_one + k_arity * t * l_minus_one;
                indices.push(idx);
                k_values_k.push(next_k(&mut seed));
            }
        }
    }

    let expected: K = indices
        .iter()
        .zip(k_values_k.iter())
        .map(|(&idx, &v)| gamma_pows[idx] * v)
        .fold(K::ZERO, |acc, x| acc + x);

    let mut b = R1csBuilder::new();
    let gamma_var = alloc_kvar(&mut b, gamma);
    let table = gamma_powers(&mut b, gamma_var, max_idx + 1);
    let k_vars: Vec<KVar> = k_values_k
        .iter()
        .copied()
        .map(|v| alloc_kvar(&mut b, v))
        .collect();
    let out = enforce_gamma_indexed_sum(&mut b, &table, &indices, &k_vars);

    assert!(b.is_satisfied(), "gamma_indexed_sum unsatisfied");
    assert_eq!(kvar_value(&b, out), expected, "gamma_indexed_sum mismatch vs native");
}

// ── F-term and N-term ────────────────────────────────────────────────────

#[test]
fn r1cs_f_term_matches_x_times_y_minus_z() {
    let y0 = k(3, 2);
    let y1 = k(5, 7);
    let y2 = k(11, 4);
    let expected = y0 * y1 - y2;

    let mut b = R1csBuilder::new();
    let y0v = alloc_kvar(&mut b, y0);
    let y1v = alloc_kvar(&mut b, y1);
    let y2v = alloc_kvar(&mut b, y2);
    let out = enforce_r1cs_f_term(&mut b, y0v, y1v, y2v);

    assert!(b.is_satisfied(), "f-term unsatisfied");
    assert_eq!(kvar_value(&b, out), expected, "f-term value mismatch");
}

#[test]
fn norm_check_b2_zero_at_minus_one_zero_one() {
    for z_val in [k(0, 0), k(1, 0), -K::ONE] {
        let mut b = R1csBuilder::new();
        let zv = alloc_kvar(&mut b, z_val);
        let out = enforce_norm_check_b2(&mut b, zv);
        assert!(b.is_satisfied(), "norm-check unsatisfied for z={:?}", z_val);
        assert_eq!(kvar_value(&b, out), K::ZERO, "norm-check must vanish at z={:?}", z_val);
    }
}

#[test]
fn norm_check_b2_nonzero_off_centered_range() {
    // For z ∈ K outside {-1, 0, 1}, norm_i = (z+1)·z·(z-1) ≠ 0.
    for z_val in [k(2, 0), k(0, 3), k(7, 11)] {
        let mut b = R1csBuilder::new();
        let zv = alloc_kvar(&mut b, z_val);
        let out = enforce_norm_check_b2(&mut b, zv);
        assert!(b.is_satisfied(), "norm-check unsatisfied for z={:?}", z_val);
        let got = kvar_value(&b, out);
        let expected = (z_val + K::ONE) * z_val * (z_val - K::ONE);
        assert_eq!(got, expected, "norm-check value mismatch at z={:?}", z_val);
        assert_ne!(got, K::ZERO, "norm-check should be nonzero outside centered range");
    }
}

// ── engine-parity sumcheck walk (verify_sumcheck_rounds_poseidon_v3) ─────

#[test]
fn engine_sumcheck_walk_matches_native_verify_sumcheck_rounds_poseidon_v3() {
    use neo_fold_clean::engine::r1cs_circuit::{enforce_sumcheck_rounds_engine, TranscriptGadget};
    use neo_reductions::sumcheck::verify_sumcheck_rounds_poseidon_v3;
    use neo_transcript::{Poseidon2Transcript, Transcript};

    const APP: &[u8] = b"neo.test.sumcheck.engine/v1";

    // Build a self-consistent 4-round sumcheck where each round polynomial
    // satisfies g_q(0) + g_q(1) == claim_q. We CHOOSE the prover's challenges
    // by driving the native transcript — the in-circuit transcript will
    // squeeze the same values because we use the same APP label and absorb
    // the same coefficients in the same order.
    //
    // First, generate the chain by running a *mock* prover where each
    // round's challenges come from the native transcript itself.
    let initial_claim = k(123, 45);

    let mut native_tr = Poseidon2Transcript::new(APP);
    let mut native_rounds: Vec<Vec<K>> = Vec::new();
    let mut native_running = initial_claim;
    let mut seed: u64 = 0xC0FFEE_DEAD_BEEF;
    let mut native_challenges: Vec<K> = Vec::new();
    for _ in 0..4 {
        // Build a self-consistent round: pick c1, c2 freely; solve for c0
        // such that 2·c0 + c1 + c2 == native_running.
        let c1 = next_k(&mut seed);
        let c2 = next_k(&mut seed);
        let two_inv = K::from_u64(2).inverse();
        let c0 = (native_running - c1 - c2) * two_inv;
        let coeffs = vec![c0, c1, c2];

        // Native sponge: append packed coefficients, squeeze 2 fields, form K.
        let packed: Vec<F> = coeffs.iter().flat_map(|c| c.as_coeffs()).collect();
        native_tr.append_fields_raw(&packed);
        let pair = native_tr.challenge_fields_raw(2);
        let challenge = neo_math::from_complex(pair[0], pair[1]);
        native_challenges.push(challenge);

        // Update running_sum = horner_eval(coeffs, challenge).
        let mut new_running = K::ZERO;
        let mut r_pow = K::ONE;
        for c in &coeffs {
            new_running += *c * r_pow;
            r_pow *= challenge;
        }
        native_running = new_running;
        native_rounds.push(coeffs);
    }

    // Sanity: run `verify_sumcheck_rounds_poseidon_v3` against the chain.
    // It should accept and return the same challenges and running sum.
    let mut verify_tr = Poseidon2Transcript::new(APP);
    let (verify_challenges, verify_running, ok) =
        verify_sumcheck_rounds_poseidon_v3(&mut verify_tr, /*degree_bound=*/ 2, initial_claim, &native_rounds);
    assert!(ok, "native verify_sumcheck_rounds_poseidon_v3 rejected an honest chain");
    assert_eq!(verify_challenges, native_challenges, "native verifier challenges");
    assert_eq!(verify_running, native_running, "native verifier final running sum");

    // In-circuit: drive `enforce_sumcheck_rounds_engine` with the same chain.
    let mut b = R1csBuilder::new();
    let initial_var = alloc_kvar(&mut b, initial_claim);
    let round_vars: Vec<Vec<KVar>> = native_rounds
        .iter()
        .map(|coeffs| {
            coeffs
                .iter()
                .copied()
                .map(|v| alloc_kvar(&mut b, v))
                .collect()
        })
        .collect();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let (challenge_vars, final_running) = enforce_sumcheck_rounds_engine(&mut b, &mut tr, initial_var, &round_vars);

    assert!(
        b.is_satisfied(),
        "engine-parity sumcheck walk unsatisfied (first bad row: {:?})",
        b.first_unsatisfied_row()
    );

    // Each challenge wire's witness value must equal the native challenge.
    assert_eq!(challenge_vars.len(), native_challenges.len());
    for (i, var) in challenge_vars.iter().enumerate() {
        assert_eq!(
            kvar_value(&b, *var),
            native_challenges[i],
            "engine sumcheck challenge {i} mismatch"
        );
    }
    // Final running sum wire must equal the native one.
    assert_eq!(
        kvar_value(&b, final_running),
        native_running,
        "engine sumcheck final running sum mismatch"
    );
}

#[test]
fn engine_sumcheck_walk_rejects_tampered_round_coefficient() {
    // Same setup as the happy-path test but tamper one round coefficient
    // after the gadget runs — the per-round `g(0)+g(1)==claim` check must
    // fail.
    use neo_fold_clean::engine::r1cs_circuit::{enforce_sumcheck_rounds_engine, TranscriptGadget};
    use neo_transcript::{Poseidon2Transcript, Transcript};

    const APP: &[u8] = b"neo.test.sumcheck.engine/tamper/v1";

    let initial_claim = k(7, 11);
    let mut native_tr = Poseidon2Transcript::new(APP);
    let mut native_rounds: Vec<Vec<K>> = Vec::new();
    let mut running = initial_claim;
    let mut seed: u64 = 0xDEAD;
    for _ in 0..3 {
        let c1 = next_k(&mut seed);
        let c2 = next_k(&mut seed);
        let two_inv = K::from_u64(2).inverse();
        let c0 = (running - c1 - c2) * two_inv;
        let coeffs = vec![c0, c1, c2];
        let packed: Vec<F> = coeffs.iter().flat_map(|c| c.as_coeffs()).collect();
        native_tr.append_fields_raw(&packed);
        let pair = native_tr.challenge_fields_raw(2);
        let challenge = neo_math::from_complex(pair[0], pair[1]);
        let mut nr = K::ZERO;
        let mut r_pow = K::ONE;
        for c in &coeffs {
            nr += *c * r_pow;
            r_pow *= challenge;
        }
        running = nr;
        native_rounds.push(coeffs);
    }

    let mut b = R1csBuilder::new();
    let initial_var = alloc_kvar(&mut b, initial_claim);
    let round_vars: Vec<Vec<KVar>> = native_rounds
        .iter()
        .map(|coeffs| {
            coeffs
                .iter()
                .copied()
                .map(|v| alloc_kvar(&mut b, v))
                .collect()
        })
        .collect();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let _ = enforce_sumcheck_rounds_engine(&mut b, &mut tr, initial_var, &round_vars);

    assert!(b.is_satisfied(), "baseline must be satisfied");

    // Tamper round 1's coefficient c1 — this breaks the g(0)+g(1)==claim
    // identity at that round.
    let target = round_vars[1][1].c0.col();
    let tampered = b.witness()[target] + F::ONE;
    b.tamper_witness(target, tampered);

    assert!(
        !b.is_satisfied(),
        "engine sumcheck walk accepted a tampered round coefficient"
    );
}

// ── χ_α multilinear extension (engine common::chi_alpha mirror) ──────────

/// Native `χ_α(ρ) = Π_bit (bit_is_one(ρ, bit) ? α[bit] : 1 - α[bit])`.
/// Mirrors `optimized_engine/common.rs:chi_alpha` and the inline copies in
/// `rhs_terminal_identity_fe_with_k_mcs`, `rhs_terminal_identity_nc`, and
/// `claimed_initial_sum_from_inputs_with_k_mcs`.
fn native_chi_alpha(alpha: &[K]) -> Vec<K> {
    let d_sz = 1usize << alpha.len();
    let mut out = vec![K::ZERO; d_sz];
    for rho in 0..d_sz {
        let mut w = K::ONE;
        for (bit, &a) in alpha.iter().enumerate() {
            let is_one = ((rho >> bit) & 1) == 1;
            w *= if is_one { a } else { K::ONE - a };
        }
        out[rho] = w;
    }
    out
}

#[test]
fn chi_alpha_empty_returns_singleton_one() {
    use neo_fold_clean::engine::r1cs_circuit::enforce_chi_alpha;

    let mut b = R1csBuilder::new();
    let chi = enforce_chi_alpha(&mut b, &[]);
    assert_eq!(chi.len(), 1);
    assert_eq!(kvar_value(&b, chi[0]), K::ONE);
    assert!(b.is_satisfied());
}

#[test]
fn chi_alpha_matches_native_for_small_ell_d() {
    use neo_fold_clean::engine::r1cs_circuit::enforce_chi_alpha;

    for ell_d in 1usize..=6 {
        let alpha: Vec<K> = (0..ell_d)
            .map(|i| k(i as u64 + 3, (i as u64) * 7 + 5))
            .collect();
        let native = native_chi_alpha(&alpha);

        let mut b = R1csBuilder::new();
        let alpha_vars: Vec<KVar> = alpha
            .iter()
            .copied()
            .map(|v| alloc_kvar(&mut b, v))
            .collect();
        let chi_vars = enforce_chi_alpha(&mut b, &alpha_vars);

        assert!(
            b.is_satisfied(),
            "χ_α circuit unsatisfied at ell_d={ell_d} (first bad row: {:?})",
            b.first_unsatisfied_row()
        );
        assert_eq!(chi_vars.len(), native.len(), "χ_α length at ell_d={ell_d}");
        for (rho, var) in chi_vars.iter().enumerate() {
            assert_eq!(
                kvar_value(&b, *var),
                native[rho],
                "χ_α[{rho}] mismatch at ell_d={ell_d}"
            );
        }
    }
}

#[test]
fn chi_alpha_evaluates_to_indicator_on_hypercube_alpha() {
    use neo_fold_clean::engine::r1cs_circuit::enforce_chi_alpha;

    // When α ∈ {0,1}^ell_d, χ_α is the indicator: χ_α[ρ] = 1 iff ρ == α-mask, else 0.
    let alpha_bits = [1u64, 0, 1, 1];
    let alpha: Vec<K> = alpha_bits.iter().map(|&b| K::from_u64(b)).collect();
    let mut alpha_mask = 0usize;
    for (i, &b) in alpha_bits.iter().enumerate() {
        if b == 1 {
            alpha_mask |= 1 << i;
        }
    }

    let mut bd = R1csBuilder::new();
    let alpha_vars: Vec<KVar> = alpha
        .iter()
        .copied()
        .map(|v| alloc_kvar(&mut bd, v))
        .collect();
    let chi = enforce_chi_alpha(&mut bd, &alpha_vars);

    assert!(bd.is_satisfied());
    for (rho, var) in chi.iter().enumerate() {
        let expected = if rho == alpha_mask { K::ONE } else { K::ZERO };
        assert_eq!(
            kvar_value(&bd, *var),
            expected,
            "χ_α at hypercube α: ρ={rho} mask={alpha_mask}"
        );
    }
}

#[test]
fn chi_alpha_rejects_tampered_alpha_bit() {
    use neo_fold_clean::engine::r1cs_circuit::enforce_chi_alpha;

    let alpha = [k(2, 1), k(3, 4), k(0, 7)];
    let mut bd = R1csBuilder::new();
    let alpha_vars: Vec<KVar> = alpha
        .iter()
        .copied()
        .map(|v| alloc_kvar(&mut bd, v))
        .collect();
    let _ = enforce_chi_alpha(&mut bd, &alpha_vars);

    assert!(bd.is_satisfied(), "baseline must be satisfied");

    let target = alpha_vars[1].c0.col();
    let tampered = bd.witness()[target] + F::ONE;
    bd.tamper_witness(target, tampered);
    assert!(!bd.is_satisfied(), "χ_α accepted a tampered α[1].c0");
}

// Silence "unused" import warning when not every helper is referenced.
fn _unused(_: KLc) {}
fn _unused_k_mul(_: KVar) {
    let _ = enforce_k_mul;
}
