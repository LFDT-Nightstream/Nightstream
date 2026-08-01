//! SplitNcV1 — FE channel parity tests.
//!
//! Splits out of `pi_ccs_split_nc.rs` (which had grown past 1 200 lines, near
//! the 1 500 file cap). Houses native-helper parity tests for:
//!
//! - `enforce_fe_claimed_initial` ↔ `claimed_initial_sum_from_inputs_with_k_mcs`
//! - `enforce_fe_sumcheck_driver`  ↔ FE channel block in
//!   `optimized_verify_with_cache_and_public_instance_digest_impl`
//! - `enforce_fe_terminal_identity` ↔ `rhs_terminal_identity_fe_with_k_mcs`
//!
//! "Native parity" here means the tests call the actual native helper, build
//! the same witness on both sides, and assert in-circuit equality to the
//! native value — *not* a hand-reimplementation of the formula. The hand-
//! reimplementation tests stay in `pi_ccs_split_nc.rs` as formula sanity.

#![allow(non_snake_case)]

use neo_ajtai::Commitment;
use neo_ccs::{CcsMatrix, CcsStructure, CeClaim as NeoCeClaim, Mat, SparsePoly, Term};
use neo_fold_clean::engine::r1cs_circuit::field_ext::KVar;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder, TranscriptGadget};
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::{
    enforce_fe_claimed_initial, enforce_fe_sumcheck_driver, enforce_fe_terminal_identity, FeClaimedInitialInputs,
    FeTerminalInputs,
};
use neo_math::ring::D;
use neo_math::{from_complex, KExtensions, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::optimized_engine::legacy_split_nc::{
    claimed_initial_sum_from_inputs_with_k_mcs, rhs_terminal_identity_fe_with_k_mcs,
};
use neo_reductions::engines::optimized_engine::Challenges;
use neo_reductions::sumcheck::{verify_sumcheck_rounds_poseidon_v3, SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{Field, PrimeCharacteristicRing};

const APP: &[u8] = b"neo.test.pi_ccs.split_nc.fe/v1";

type CeClaim = NeoCeClaim<Commitment, F, K>;

// ── Test fixtures ─────────────────────────────────────────────────────────

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f(&mut self) -> F {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        F::from_u64(self.0 & 0xFFFF)
    }
    fn next_k(&mut self) -> K {
        let c0 = self.next_f();
        let c1 = self.next_f();
        K::from_coeffs([c0, c1])
    }
    fn next_k_vec(&mut self, n: usize) -> Vec<K> {
        (0..n).map(|_| self.next_k()).collect()
    }
}

/// Build a minimal CCS structure with `t` 1×1 identity-ish matrices and the
/// R1CS polynomial `f(X, Y, Z) = X·Y - Z` when `t == 3`, else a degenerate
/// polynomial. The actual matrices/polynomial aren't read by `claimed_initial`
/// (only `s.t()` is consulted), but `rhs_terminal_identity_fe_with_k_mcs`
/// uses `s.f.eval_in_ext::<K>(...)` so the polynomial does matter there.
fn build_test_ccs_structure(t: usize) -> CcsStructure<F> {
    let mat = CcsMatrix::Identity { n: 1 };
    let matrices: Vec<CcsMatrix<F>> = (0..t).map(|_| mat.clone()).collect();
    let poly = if t >= 3 {
        // f(X, Y, Z) = X·Y - Z, with trailing zero-exps for variables beyond
        // the first three to match arity == t.
        let mut e_xy = vec![0u32; t];
        e_xy[0] = 1;
        e_xy[1] = 1;
        let mut e_z = vec![0u32; t];
        e_z[2] = 1;
        SparsePoly::new(
            t,
            vec![
                Term {
                    coeff: F::ONE,
                    exps: e_xy,
                },
                Term {
                    coeff: -F::ONE,
                    exps: e_z,
                },
            ],
        )
    } else {
        SparsePoly::new(t, vec![])
    };
    CcsStructure::new_sparse(matrices, poly).expect("CcsStructure::new_sparse")
}

fn build_test_ce_claim(rng: &mut Rng, m_in: usize, t: usize, d_sz: usize, kappa: usize, r_len: usize) -> CeClaim {
    let c_data: Vec<F> = (0..(D * kappa)).map(|_| rng.next_f()).collect();
    let c = Commitment {
        d: D,
        kappa,
        data: c_data,
    };
    let mut x = Mat::zero(D, m_in.max(1), F::ZERO);
    for col in 0..m_in.max(1) {
        for row in 0..D {
            x.set(row, col, rng.next_f());
        }
    }
    let r = rng.next_k_vec(r_len);
    let y_ring: Vec<Vec<K>> = (0..t).map(|_| rng.next_k_vec(d_sz)).collect();
    CeClaim {
        adv: None,
        c,
        X: x,
        r,
        s_col: Vec::new(),
        y_ring,
        ct: Vec::new(),
        aux_openings: Vec::new(),
        y_zcol: Vec::new(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}

fn alloc_witness_k(b: &mut R1csBuilder, v: K) -> KVar {
    let [c0, c1] = v.as_coeffs();
    KVar::alloc(b, c0, c1)
}

fn alloc_y_ring_wires(b: &mut R1csBuilder, y_ring: &[Vec<K>]) -> Vec<Vec<KVar>> {
    y_ring
        .iter()
        .map(|row| row.iter().copied().map(|v| alloc_witness_k(b, v)).collect())
        .collect()
}

fn k_value(b: &R1csBuilder, v: KVar) -> K {
    let c0 = b.witness()[v.c0.col()];
    let c1 = b.witness()[v.c1.col()];
    K::from_coeffs([c0, c1])
}

fn pin_k(b: &mut R1csBuilder, var: KVar, native: K) {
    let [c0, c1] = native.as_coeffs();
    b.enforce_eq(&Lc::from_var(var.c0), &Lc::from_const(c0));
    b.enforce_eq(&Lc::from_var(var.c1), &Lc::from_const(c1));
}

// ── F.1: enforce_fe_claimed_initial vs native helper ─────────────────────

#[test]
fn fe_claimed_initial_matches_native_helper() {
    // Real native call: `claimed_initial_sum_from_inputs_with_k_mcs(s, ch, k_mcs, me_inputs)`.
    let ell_d = 3usize;
    let t = 3usize;
    let me_len = 2usize;
    let k_mcs = 1usize;
    let d_sz = 1usize << ell_d;

    let mut rng = Rng::new(0xFE_C1A1);
    let s = build_test_ccs_structure(t);

    let gamma = rng.next_k();
    let alpha = rng.next_k_vec(ell_d);
    let ch = Challenges {
        alpha: alpha.clone(),
        beta_a: rng.next_k_vec(ell_d), // unused by claimed_initial
        beta_r: rng.next_k_vec(ell_d), // unused
        beta_m: rng.next_k_vec(ell_d), // unused
        gamma,
    };

    let me_inputs: Vec<CeClaim> = (0..me_len)
        .map(|_| build_test_ce_claim(&mut rng, 2, t, d_sz, 1, 2))
        .collect();

    let native = claimed_initial_sum_from_inputs_with_k_mcs(&s, &ch, k_mcs, &me_inputs);

    let mut b = R1csBuilder::new();
    let gamma_var = alloc_witness_k(&mut b, gamma);
    let alpha_vars: Vec<KVar> = alpha
        .iter()
        .copied()
        .map(|a| alloc_witness_k(&mut b, a))
        .collect();
    let y_vars: Vec<Vec<Vec<KVar>>> = me_inputs
        .iter()
        .map(|me| alloc_y_ring_wires(&mut b, &me.y_ring))
        .collect();
    let inputs = FeClaimedInitialInputs {
        k_mcs,
        t,
        ell_d,
        gamma: gamma_var,
        alpha: &alpha_vars,
        running_y_ring: &y_vars,
    };
    let result = enforce_fe_claimed_initial(&mut b, &inputs).expect("FE claimed_initial");

    assert_eq!(
        k_value(&b, result),
        native,
        "FE claimed_initial mismatch vs native helper"
    );
    pin_k(&mut b, result, native);
    assert!(
        b.is_satisfied(),
        "circuit unsatisfied (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
}

// ── F.2: enforce_fe_sumcheck_driver vs native verify_sumcheck_rounds_poseidon_v3 ──

/// Build a self-consistent sumcheck chain: pick coeffs[1..] freely, solve
/// coeffs[0] so `2·coeffs[0] + Σ_{i≥1} coeffs[i] == claim_in`. Returns the
/// per-round polynomials and the final running sum.
fn build_consistent_chain(rng: &mut Rng, initial: K, num_rounds: usize, d_sc: usize) -> Vec<Vec<K>> {
    let two_inv = K::from_u64(2).inverse();
    let mut chain = Vec::with_capacity(num_rounds);
    let mut running = initial;

    // We need to drive a native transcript to derive the actual challenges so
    // the chain we build is consistent with the transcript-derived challenges.
    let mut tr = Poseidon2Transcript::new(APP);
    // Mimic the FE absorb prefix the driver applies before sumcheck rounds.
    tr.append_fields_raw(&[F::from_u64(
        neo_reductions::engines::utils::PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG,
    )]);
    tr.append_fields_raw(&[F::from_u64(
        neo_reductions::engines::utils::PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
    )]);
    tr.append_fields_raw(&initial.as_coeffs());
    tr.append_fields_raw(&[F::from_u64(SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);

    for _ in 0..num_rounds {
        let mut coeffs = vec![K::ZERO; d_sc + 1];
        let mut sum_rest = K::ZERO;
        for cv in coeffs.iter_mut().skip(1) {
            *cv = rng.next_k();
            sum_rest += *cv;
        }
        coeffs[0] = (running - sum_rest) * two_inv;

        // Absorb the coefficients and sample the challenge in the same order
        // `verify_sumcheck_rounds_poseidon_v3` does.
        let packed: Vec<F> = coeffs.iter().flat_map(|c| c.as_coeffs()).collect();
        tr.append_fields_raw(&packed);
        let pair = tr.challenge_fields_raw(2);
        let challenge = from_complex(pair[0], pair[1]);

        // running := g(challenge) via Horner.
        let mut next = K::ZERO;
        let mut r_pow = K::ONE;
        for c in &coeffs {
            next += *c * r_pow;
            r_pow *= challenge;
        }
        running = next;
        chain.push(coeffs);
    }
    chain
}

#[test]
fn fe_sumcheck_driver_matches_native_verify_sumcheck_v3() {
    let ell_n = 3usize;
    let ell_d = 2usize;
    let d_sc = 4usize; // degree bound — exercises non-trivial round-polynomial widths.

    let mut rng = Rng::new(0xFE_5C_0001);
    let claimed_initial = rng.next_k();
    let rounds = build_consistent_chain(&mut rng, claimed_initial, ell_n + ell_d, d_sc);

    // Native side: re-run `verify_sumcheck_rounds_poseidon_v3` with the same
    // absorb prefix to derive expected challenges + final running sum.
    let mut native_tr = Poseidon2Transcript::new(APP);
    native_tr.append_fields_raw(&[F::from_u64(
        neo_reductions::engines::utils::PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG,
    )]);
    native_tr.append_fields_raw(&[F::from_u64(
        neo_reductions::engines::utils::PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
    )]);
    native_tr.append_fields_raw(&claimed_initial.as_coeffs());
    native_tr.append_fields_raw(&[F::from_u64(SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);
    let (native_challenges, native_final, ok) =
        verify_sumcheck_rounds_poseidon_v3(&mut native_tr, d_sc, claimed_initial, &rounds);
    assert!(ok, "native verify_sumcheck_rounds_poseidon_v3 must accept honest chain");
    assert_eq!(native_challenges.len(), ell_n + ell_d);
    let (native_r_prime, native_alpha_prime) = native_challenges.split_at(ell_n);

    // Circuit side.
    let mut b = R1csBuilder::new();
    let claimed_var = alloc_witness_k(&mut b, claimed_initial);
    let round_vars: Vec<Vec<KVar>> = rounds
        .iter()
        .map(|coeffs| {
            coeffs
                .iter()
                .copied()
                .map(|v| alloc_witness_k(&mut b, v))
                .collect()
        })
        .collect();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let result = enforce_fe_sumcheck_driver(&mut b, &mut tr, ell_n, ell_d, d_sc, claimed_var, &round_vars)
        .expect("FE sumcheck driver");

    assert_eq!(result.r_prime.len(), ell_n);
    assert_eq!(result.alpha_prime.len(), ell_d);
    for (i, var) in result.r_prime.iter().enumerate() {
        assert_eq!(k_value(&b, *var), native_r_prime[i], "r_prime[{i}]");
        pin_k(&mut b, *var, native_r_prime[i]);
    }
    for (i, var) in result.alpha_prime.iter().enumerate() {
        assert_eq!(k_value(&b, *var), native_alpha_prime[i], "alpha_prime[{i}]");
        pin_k(&mut b, *var, native_alpha_prime[i]);
    }
    assert_eq!(k_value(&b, result.final_sum), native_final, "final_sum");
    pin_k(&mut b, result.final_sum, native_final);
    assert!(
        b.is_satisfied(),
        "FE driver circuit unsatisfied (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
}

#[test]
fn fe_sumcheck_driver_rejects_tampered_round_coefficient() {
    let ell_n = 2usize;
    let ell_d = 2usize;
    let d_sc = 3usize;

    let mut rng = Rng::new(0xFE_5C_0002);
    let claimed_initial = rng.next_k();
    let rounds = build_consistent_chain(&mut rng, claimed_initial, ell_n + ell_d, d_sc);

    let mut b = R1csBuilder::new();
    let claimed_var = alloc_witness_k(&mut b, claimed_initial);
    let round_vars: Vec<Vec<KVar>> = rounds
        .iter()
        .map(|coeffs| {
            coeffs
                .iter()
                .copied()
                .map(|v| alloc_witness_k(&mut b, v))
                .collect()
        })
        .collect();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let _ = enforce_fe_sumcheck_driver(&mut b, &mut tr, ell_n, ell_d, d_sc, claimed_var, &round_vars).expect("driver");

    assert!(b.is_satisfied(), "baseline");

    // Tamper round 1 coeff 2 — breaks the per-round g(0)+g(1)==claim identity.
    let target = round_vars[1][2].c0.col();
    let tampered = b.witness()[target] + F::ONE;
    b.tamper_witness(target, tampered);
    assert!(!b.is_satisfied(), "tampered FE round must be rejected");
}

// ── F.3: enforce_fe_terminal_identity vs native rhs_terminal_identity_fe_with_k_mcs ──

#[test]
fn fe_terminal_identity_matches_native_rhs() {
    // Real native call: `rhs_terminal_identity_fe_with_k_mcs(s, &params, &ch,
    //   &r_prime, &alpha_prime, &out_me, k_mcs, me_inputs_r_opt)`.
    let ell_d = 3usize;
    let ell_n = 2usize;
    let t = 3usize;
    let k_mcs = 1usize;
    let me_len = 2usize;
    let k_total = k_mcs + me_len;
    let d_sz = 1usize << ell_d;

    let mut rng = Rng::new(0xFE_FED0_0001);
    let s = build_test_ccs_structure(t);
    let params = NeoParams::goldilocks_paper_b2();

    let gamma = rng.next_k();
    let alpha = rng.next_k_vec(ell_d);
    let beta_a = rng.next_k_vec(ell_d);
    let beta_r = rng.next_k_vec(ell_n);
    let beta_m = rng.next_k_vec(ell_d); // unused by FE terminal
    let r_prime = rng.next_k_vec(ell_n);
    let alpha_prime = rng.next_k_vec(ell_d);

    let ch = Challenges {
        alpha: alpha.clone(),
        beta_a: beta_a.clone(),
        beta_r: beta_r.clone(),
        beta_m,
        gamma,
    };

    // out_me: k_total CeClaim outputs. The native function reads y_ring[j].first()
    // (for the F' term) and y_ring[j][rho] for rho ∈ [0, 2^ell_d). Build with d_sz lanes.
    let out_me: Vec<CeClaim> = (0..k_total)
        .map(|_| build_test_ce_claim(&mut rng, 2, t, d_sz, 1, ell_n))
        .collect();

    // me_input_r: the shared evaluation point r every running ME input must carry.
    // For the test we just pick a fresh r of length ell_n.
    let me_input_r = rng.next_k_vec(ell_n);

    let native = rhs_terminal_identity_fe_with_k_mcs(
        &s,
        &params,
        &ch,
        &r_prime,
        &alpha_prime,
        &out_me,
        k_mcs,
        Some(&me_input_r),
    );

    // Circuit side.
    let mut b = R1csBuilder::new();
    let gamma_var = alloc_witness_k(&mut b, gamma);
    let alpha_vars: Vec<KVar> = alpha
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut b, v))
        .collect();
    let beta_a_vars: Vec<KVar> = beta_a
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut b, v))
        .collect();
    let beta_r_vars: Vec<KVar> = beta_r
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut b, v))
        .collect();
    let r_prime_vars: Vec<KVar> = r_prime
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut b, v))
        .collect();
    let alpha_prime_vars: Vec<KVar> = alpha_prime
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut b, v))
        .collect();
    let me_input_r_vars: Vec<KVar> = me_input_r
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut b, v))
        .collect();

    let output_y_ring: Vec<Vec<Vec<KVar>>> = out_me
        .iter()
        .map(|c| alloc_y_ring_wires(&mut b, &c.y_ring))
        .collect();

    let result = enforce_fe_terminal_identity(
        &mut b,
        &FeTerminalInputs {
            poly: &s.f,
            t,
            k_mcs,
            gamma: gamma_var,
            alpha: &alpha_vars,
            beta_a: &beta_a_vars,
            beta_r: &beta_r_vars,
            r_prime: &r_prime_vars,
            alpha_prime: &alpha_prime_vars,
            me_input_r: Some(&me_input_r_vars),
            output_y_ring: &output_y_ring,
        },
    )
    .expect("FE terminal");

    assert_eq!(k_value(&b, result), native, "FE terminal mismatch vs native rhs");
    pin_k(&mut b, result, native);
    assert!(
        b.is_satisfied(),
        "FE terminal circuit unsatisfied (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
}

#[test]
fn fe_terminal_identity_rejects_tampered_output_y_lane() {
    let ell_d = 3usize;
    let ell_n = 2usize;
    let t = 3usize;
    let k_mcs = 1usize;
    let me_len = 2usize;
    let k_total = k_mcs + me_len;
    let d_sz = 1usize << ell_d;

    let mut rng = Rng::new(0xFE_FED0_0002);
    let s = build_test_ccs_structure(t);
    let params = NeoParams::goldilocks_paper_b2();

    let gamma = rng.next_k();
    let alpha = rng.next_k_vec(ell_d);
    let beta_a = rng.next_k_vec(ell_d);
    let beta_r = rng.next_k_vec(ell_n);
    let r_prime = rng.next_k_vec(ell_n);
    let alpha_prime = rng.next_k_vec(ell_d);
    let me_input_r = rng.next_k_vec(ell_n);
    let ch = Challenges {
        alpha: alpha.clone(),
        beta_a: beta_a.clone(),
        beta_r: beta_r.clone(),
        beta_m: rng.next_k_vec(ell_d),
        gamma,
    };
    let out_me: Vec<CeClaim> = (0..k_total)
        .map(|_| build_test_ce_claim(&mut rng, 2, t, d_sz, 1, ell_n))
        .collect();
    let native = rhs_terminal_identity_fe_with_k_mcs(
        &s,
        &params,
        &ch,
        &r_prime,
        &alpha_prime,
        &out_me,
        k_mcs,
        Some(&me_input_r),
    );

    let mut b = R1csBuilder::new();
    let gamma_var = alloc_witness_k(&mut b, gamma);
    let alpha_vars: Vec<KVar> = alpha
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut b, v))
        .collect();
    let beta_a_vars: Vec<KVar> = beta_a
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut b, v))
        .collect();
    let beta_r_vars: Vec<KVar> = beta_r
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut b, v))
        .collect();
    let r_prime_vars: Vec<KVar> = r_prime
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut b, v))
        .collect();
    let alpha_prime_vars: Vec<KVar> = alpha_prime
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut b, v))
        .collect();
    let me_input_r_vars: Vec<KVar> = me_input_r
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut b, v))
        .collect();
    let output_y_ring: Vec<Vec<Vec<KVar>>> = out_me
        .iter()
        .map(|c| alloc_y_ring_wires(&mut b, &c.y_ring))
        .collect();

    let result = enforce_fe_terminal_identity(
        &mut b,
        &FeTerminalInputs {
            poly: &s.f,
            t,
            k_mcs,
            gamma: gamma_var,
            alpha: &alpha_vars,
            beta_a: &beta_a_vars,
            beta_r: &beta_r_vars,
            r_prime: &r_prime_vars,
            alpha_prime: &alpha_prime_vars,
            me_input_r: Some(&me_input_r_vars),
            output_y_ring: &output_y_ring,
        },
    )
    .expect("FE terminal");
    pin_k(&mut b, result, native);
    assert!(b.is_satisfied(), "baseline");

    // Tamper an output y_ring lane in the running region (idx = k_mcs, j = 1, rho = 2).
    let target = output_y_ring[k_mcs][1][2].c0.col();
    let tampered = b.witness()[target] + F::ONE;
    b.tamper_witness(target, tampered);
    assert!(!b.is_satisfied(), "tampered output y must break FE terminal pin");
}
