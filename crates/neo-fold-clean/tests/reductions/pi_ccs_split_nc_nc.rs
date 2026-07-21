//! SplitNcV1 — NC channel parity tests.
//!
//! Houses native-helper parity tests for:
//!
//! - `enforce_nc_range_product`     ↔ `range_product(val, b)` in
//!   `neo_reductions::engines::optimized_engine::terminal_identities`
//!   (private — reconstructed in tests via its trivial 3-line formula).
//! - `enforce_nc_sumcheck_driver`   ↔ NC channel block (sub-step H).
//! - `enforce_nc_terminal_identity` ↔ `rhs_terminal_identity_nc`        (sub-step H).
//!
//! Kept separate from `pi_ccs_split_nc.rs` and `pi_ccs_split_nc_fe.rs` per
//! the 1500-LoC discipline and the file-split guidance.

#![allow(non_snake_case)]

use neo_ajtai::Commitment;
use neo_ccs::{CeClaim as NeoCeClaim, Mat};
use neo_fold_clean::engine::r1cs_circuit::field_ext::KVar;
use neo_fold_clean::engine::r1cs_circuit::sumcheck::enforce_norm_check_b2;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder, TranscriptGadget};
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::{
    enforce_block_lane_claimed_initial, enforce_block_lane_terminal_identity, enforce_nc_range_product,
    enforce_nc_sumcheck_driver, enforce_nc_terminal_identity, BlockLaneTerminalInputs, NcTerminalInputs,
};
use neo_math::ring::D;
use neo_math::{from_complex, KExtensions, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::optimized_engine::{rhs_terminal_identity_nc, Challenges};
use neo_reductions::engines::utils::{PI_CCS_SUMCHECK_INITIAL_RAW_TAG, PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG};
use neo_reductions::sumcheck::{verify_sumcheck_rounds_poseidon_v3, SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{Field, PrimeCharacteristicRing};

const APP: &[u8] = b"neo.test.pi_ccs.split_nc.nc/v1";

type CeClaim = NeoCeClaim<Commitment, F, K>;

fn k(c0: u64, c1: u64) -> K {
    K::from_coeffs([F::from_u64(c0), F::from_u64(c1)])
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

fn alloc_witness_k(b: &mut R1csBuilder, v: K) -> KVar {
    let [c0, c1] = v.as_coeffs();
    KVar::alloc(b, c0, c1)
}

/// Local mirror of the private native `range_product`:
/// `∏_{t = -(b-1)..=(b-1)} (val - t)`.
fn native_range_product(val: K, b: u32) -> K {
    let lo = -((b as i64) - 1);
    let hi = (b as i64) - 1;
    let mut prod = K::ONE;
    for t in lo..=hi {
        prod *= val - K::from(F::from_i64(t));
    }
    prod
}

fn native_mle(values: &[K], point: &[K]) -> K {
    values
        .iter()
        .take(1usize << point.len())
        .enumerate()
        .fold(K::ZERO, |sum, (index, value)| {
            let weight = point
                .iter()
                .enumerate()
                .fold(K::ONE, |weight, (bit, coordinate)| {
                    weight
                        * if ((index >> bit) & 1) == 1 {
                            *coordinate
                        } else {
                            K::ONE - *coordinate
                        }
                });
            sum + *value * weight
        })
}

fn native_eq(left: &[K], right: &[K]) -> K {
    left.iter().zip(right).fold(K::ONE, |value, (left, right)| {
        value * (*left * *right + (K::ONE - *left) * (K::ONE - *right))
    })
}

fn native_beta_power_selector(beta: K, point: &[K]) -> K {
    let mut power = beta;
    point.iter().fold(K::ONE, |selector, coordinate| {
        let factor = (K::ONE - *coordinate) + *coordinate * power;
        power *= power;
        selector * factor
    })
}

// ── G.1: enforce_nc_range_product vs hand-reconstructed native formula ──

#[test]
fn nc_range_product_matches_native_for_typical_b() {
    // Exercise b ∈ {1, 2, 3, 5}. For each b, evaluate at a non-trivial K
    // point and assert in-circuit equality to the native formula.
    let val = K::from_coeffs([F::from_u64(123), F::from_u64(456)]);

    for &b in &[1u32, 2, 3, 5] {
        let expected = native_range_product(val, b);

        let mut bd = R1csBuilder::new();
        let v = alloc_witness_k(&mut bd, val);
        let out = enforce_nc_range_product(&mut bd, v, b).expect("range_product");

        assert_eq!(k_value(&bd, out), expected, "b={b}: NC range_product mismatch");
        pin_k(&mut bd, out, expected);
        assert!(
            bd.is_satisfied(),
            "b={b}: circuit unsatisfied (first bad row: {:?})",
            bd.first_unsatisfied_row()
        );
    }
}

#[test]
fn nc_range_product_b2_matches_existing_norm_check_b2_gadget() {
    // The existing `enforce_norm_check_b2` already implements `(z+1)·z·(z-1)`.
    // The new `enforce_nc_range_product` with b=2 must produce the same wire
    // value for any z.
    let zs = [
        K::from_coeffs([F::from_u64(7), F::from_u64(0)]),
        K::from_coeffs([F::from_u64(0), F::from_u64(11)]),
        K::from_coeffs([F::from_u64(2), F::from_u64(3)]),
        K::ZERO,
        K::ONE,
        -K::ONE,
    ];

    for &z in &zs {
        let mut bd = R1csBuilder::new();
        let z_var = alloc_witness_k(&mut bd, z);

        let nc = enforce_nc_range_product(&mut bd, z_var, 2).expect("range_product");
        let norm_b2 = enforce_norm_check_b2(&mut bd, z_var);

        // Both must equal the same K value.
        assert_eq!(k_value(&bd, nc), k_value(&bd, norm_b2), "b=2 parity at z={:?}", z);
        // And both must equal native (z+1)·z·(z-1).
        let expected = (z + K::ONE) * z * (z - K::ONE);
        assert_eq!(k_value(&bd, nc), expected, "b=2 native parity at z={:?}", z);
        assert!(bd.is_satisfied(), "b=2 circuit unsatisfied");
    }
}

#[test]
fn nc_range_product_b2_vanishes_at_minus_one_zero_one() {
    // Soundness witness: the centered low-norm condition for b=2 is
    // `z ∈ {-1, 0, 1}`. At those points range_product must equal K::ZERO.
    let zeros = [K::ZERO, K::ONE, -K::ONE];

    for &z in &zeros {
        let mut bd = R1csBuilder::new();
        let z_var = alloc_witness_k(&mut bd, z);
        let out = enforce_nc_range_product(&mut bd, z_var, 2).expect("range_product");

        assert_eq!(k_value(&bd, out), K::ZERO, "b=2 should vanish at z={:?}", z);
        pin_k(&mut bd, out, K::ZERO);
        assert!(bd.is_satisfied(), "circuit unsatisfied at z={:?}", z);
    }
}

#[test]
fn nc_range_product_b2_rejects_coordinate_two() {
    // Regression for the first coordinate outside the production b=2
    // alphabet: 2 is a valid Goldilocks field element, but it is not a valid
    // centered low-norm assignment coordinate.
    let mut bd = R1csBuilder::new();
    let coordinate = alloc_witness_k(&mut bd, K::from(F::from_u64(2)));
    let residual = enforce_nc_range_product(&mut bd, coordinate, 2).expect("range_product");

    assert_eq!(k_value(&bd, residual), K::from(F::from_u64(6)));
    assert!(
        bd.is_satisfied(),
        "the exact nonzero residual must satisfy its defining rows"
    );
    pin_k(&mut bd, residual, K::ZERO);
    assert!(!bd.is_satisfied(), "the NC zero claim must reject encoded coordinate 2");
}

#[test]
fn nc_range_product_b3_vanishes_at_centered_integers() {
    // For b=3, range_product vanishes at z ∈ {-2, -1, 0, 1, 2}.
    let zeros: Vec<K> = (-2i64..=2).map(|t| K::from(F::from_i64(t))).collect();

    for z in zeros {
        let mut bd = R1csBuilder::new();
        let z_var = alloc_witness_k(&mut bd, z);
        let out = enforce_nc_range_product(&mut bd, z_var, 3).expect("range_product");

        assert_eq!(k_value(&bd, out), K::ZERO, "b=3 should vanish at z={:?}", z);
        pin_k(&mut bd, out, K::ZERO);
        assert!(bd.is_satisfied(), "circuit unsatisfied at z={:?}", z);
    }
}

#[test]
fn nc_range_product_b3_nonzero_outside_centered_range() {
    // For b=3 and z outside {-2..2}, the product is non-zero.
    let nonzeros = [
        K::from_u64(3),
        K::from_u64(100),
        K::from_coeffs([F::from_u64(5), F::from_u64(7)]),
    ];

    for z in nonzeros {
        let mut bd = R1csBuilder::new();
        let z_var = alloc_witness_k(&mut bd, z);
        let out = enforce_nc_range_product(&mut bd, z_var, 3).expect("range_product");

        let expected = native_range_product(z, 3);
        assert_ne!(expected, K::ZERO, "test setup: native must be non-zero at z={:?}", z);
        assert_eq!(k_value(&bd, out), expected, "b=3 native mismatch at z={:?}", z);
        pin_k(&mut bd, out, expected);
        assert!(bd.is_satisfied(), "b=3 circuit unsatisfied");
    }
}

#[test]
fn nc_range_product_rejects_b_zero() {
    // b = 0 is structurally meaningless (empty product == K::ONE silently).
    // The gadget must surface this as `Err(Shape)`, not return K::ONE.
    let mut bd = R1csBuilder::new();
    let v = alloc_witness_k(&mut bd, K::from_u64(42));
    assert!(enforce_nc_range_product(&mut bd, v, 0).is_err(), "b=0 must be rejected");
}

// ── H.1: NC sumcheck driver vs native verify_sumcheck_rounds_poseidon_v3 ──

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

/// Build a self-consistent NC sumcheck chain (initial claim = K::ZERO).
/// Mirror of the FE helper in `pi_ccs_split_nc_fe.rs` but with the NC
/// transcript prefix `[8], [9], [0, 0], [10]`.
fn build_consistent_nc_chain(rng: &mut Rng, num_rounds: usize, d_sc: usize) -> Vec<Vec<K>> {
    let two_inv = K::from_u64(2).inverse();
    let mut chain = Vec::with_capacity(num_rounds);
    let mut running = K::ZERO;

    let mut tr = Poseidon2Transcript::new(APP);
    tr.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)]);
    tr.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    tr.append_fields_raw(&K::ZERO.as_coeffs());
    tr.append_fields_raw(&[F::from_u64(SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);

    for _ in 0..num_rounds {
        let mut coeffs = vec![K::ZERO; d_sc + 1];
        let mut sum_rest = K::ZERO;
        for cv in coeffs.iter_mut().skip(1) {
            *cv = rng.next_k();
            sum_rest += *cv;
        }
        coeffs[0] = (running - sum_rest) * two_inv;

        let packed: Vec<F> = coeffs.iter().flat_map(|c| c.as_coeffs()).collect();
        tr.append_fields_raw(&packed);
        let pair = tr.challenge_fields_raw(2);
        let challenge = from_complex(pair[0], pair[1]);

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

fn alloc_round_vars(b: &mut R1csBuilder, rounds: &[Vec<K>]) -> Vec<Vec<KVar>> {
    rounds
        .iter()
        .map(|coeffs| {
            coeffs
                .iter()
                .copied()
                .map(|v| alloc_witness_k(b, v))
                .collect()
        })
        .collect()
}

#[test]
fn nc_sumcheck_driver_matches_native_verify_sumcheck_v3() {
    let ell_m = 3usize;
    let ell_d = 2usize;
    let d_sc = 4usize;

    let mut rng = Rng::new(0xC0_5C_0001);
    let rounds = build_consistent_nc_chain(&mut rng, ell_m + ell_d, d_sc);

    // Native: re-run verify_sumcheck_rounds_poseidon_v3 with the same prefix.
    let mut native_tr = Poseidon2Transcript::new(APP);
    native_tr.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)]);
    native_tr.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    native_tr.append_fields_raw(&K::ZERO.as_coeffs());
    native_tr.append_fields_raw(&[F::from_u64(SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);
    let (native_challenges, native_final, ok) =
        verify_sumcheck_rounds_poseidon_v3(&mut native_tr, d_sc, K::ZERO, &rounds);
    assert!(ok, "native NC verifier must accept honest chain");
    assert_eq!(native_challenges.len(), ell_m + ell_d);
    let (native_s_col_prime, native_alpha_prime) = native_challenges.split_at(ell_m);

    // Circuit side.
    let mut b = R1csBuilder::new();
    let round_vars = alloc_round_vars(&mut b, &rounds);
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let result =
        enforce_nc_sumcheck_driver(&mut b, &mut tr, ell_m, ell_d, d_sc, &round_vars).expect("NC sumcheck driver");

    assert_eq!(result.s_col_prime.len(), ell_m);
    assert_eq!(result.alpha_prime.len(), ell_d);
    for (i, var) in result.s_col_prime.iter().enumerate() {
        assert_eq!(k_value(&b, *var), native_s_col_prime[i], "s_col_prime[{i}]");
        pin_k(&mut b, *var, native_s_col_prime[i]);
    }
    for (i, var) in result.alpha_prime.iter().enumerate() {
        assert_eq!(k_value(&b, *var), native_alpha_prime[i], "alpha_prime[{i}]");
        pin_k(&mut b, *var, native_alpha_prime[i]);
    }
    assert_eq!(k_value(&b, result.final_sum), native_final, "final_sum");
    pin_k(&mut b, result.final_sum, native_final);
    assert!(
        b.is_satisfied(),
        "NC driver circuit unsatisfied (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
}

#[test]
fn nc_sumcheck_driver_rejects_tampered_round_coefficient() {
    let ell_m = 2usize;
    let ell_d = 2usize;
    let d_sc = 3usize;
    let mut rng = Rng::new(0xC0_5C_0002);
    let rounds = build_consistent_nc_chain(&mut rng, ell_m + ell_d, d_sc);

    let mut b = R1csBuilder::new();
    let round_vars = alloc_round_vars(&mut b, &rounds);
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let _ = enforce_nc_sumcheck_driver(&mut b, &mut tr, ell_m, ell_d, d_sc, &round_vars).expect("driver");
    assert!(b.is_satisfied(), "baseline");

    let target = round_vars[1][2].c0.col();
    let tampered = b.witness()[target] + F::ONE;
    b.tamper_witness(target, tampered);
    assert!(!b.is_satisfied(), "tampered NC round must be rejected");
}

// ── H.2: NC terminal identity vs native rhs_terminal_identity_nc ──────────

fn build_test_ce_claim_with_zcol(seed: u64, t: usize, d_sz: usize, ell_m: usize) -> CeClaim {
    let mut s = seed;
    let mut next_f = || -> F {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        F::from_u64(s & 0xFFFF)
    };

    let c_data: Vec<F> = (0..D).map(|_| next_f()).collect();
    let c = Commitment {
        d: D,
        kappa: 1,
        data: c_data,
    };
    let x = Mat::zero(D, 1, F::ZERO);
    let r: Vec<K> = (0..2)
        .map(|_| K::from_coeffs([next_f(), next_f()]))
        .collect();
    let s_col: Vec<K> = (0..ell_m)
        .map(|_| K::from_coeffs([next_f(), next_f()]))
        .collect();
    let y_ring: Vec<Vec<K>> = (0..t)
        .map(|_| {
            (0..d_sz)
                .map(|_| K::from_coeffs([next_f(), next_f()]))
                .collect()
        })
        .collect();
    let y_zcol: Vec<K> = (0..d_sz)
        .map(|_| K::from_coeffs([next_f(), next_f()]))
        .collect();

    CeClaim {
        adv: None,
        c,
        X: x,
        r,
        s_col,
        y_ring,
        ct: Vec::new(),
        aux_openings: Vec::new(),
        y_zcol,
        m_in: 1,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}

fn alloc_y_zcol_wires(b: &mut R1csBuilder, y_zcol: &[K]) -> Vec<KVar> {
    y_zcol
        .iter()
        .copied()
        .map(|v| alloc_witness_k(b, v))
        .collect()
}

#[test]
fn nc_terminal_identity_matches_native_rhs() {
    let ell_d = 3usize;
    let ell_m = 2usize;
    let t = 3usize;
    let k_total = 3usize;
    let d_sz = 1usize << ell_d;

    let mut rng = Rng::new(0xC0_FE_D0_0001);
    let params = NeoParams::goldilocks_paper_b2();
    let b_norm = params.b;

    let gamma = rng.next_k();
    let alpha = rng.next_k_vec(ell_d);
    let beta_a = rng.next_k_vec(ell_d);
    let beta_r = rng.next_k_vec(ell_d);
    let beta_m = rng.next_k_vec(ell_m);
    let alpha_prime = rng.next_k_vec(ell_d);
    let s_col_prime = rng.next_k_vec(ell_m);
    let ch = Challenges {
        alpha,
        beta_a: beta_a.clone(),
        beta_r,
        beta_m: beta_m.clone(),
        gamma,
    };

    let out_me: Vec<CeClaim> = (0..k_total)
        .map(|i| build_test_ce_claim_with_zcol(0xCE_0000 + i as u64, t, d_sz, ell_m))
        .collect();

    let native = rhs_terminal_identity_nc(&params, &ch, &s_col_prime, &alpha_prime, &out_me);

    // Circuit.
    let mut bd = R1csBuilder::new();
    let gamma_var = alloc_witness_k(&mut bd, gamma);
    let beta_a_vars: Vec<KVar> = beta_a
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut bd, v))
        .collect();
    let beta_m_vars: Vec<KVar> = beta_m
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut bd, v))
        .collect();
    let s_col_prime_vars: Vec<KVar> = s_col_prime
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut bd, v))
        .collect();
    let alpha_prime_vars: Vec<KVar> = alpha_prime
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut bd, v))
        .collect();
    let output_y_zcol: Vec<Vec<KVar>> = out_me
        .iter()
        .map(|c| alloc_y_zcol_wires(&mut bd, &c.y_zcol))
        .collect();

    let result = enforce_nc_terminal_identity(
        &mut bd,
        &NcTerminalInputs {
            b: b_norm,
            gamma: gamma_var,
            beta_a: &beta_a_vars,
            beta_m: &beta_m_vars,
            s_col_prime: &s_col_prime_vars,
            alpha_prime: &alpha_prime_vars,
            output_y_zcol: &output_y_zcol,
        },
    )
    .expect("NC terminal");

    assert_eq!(k_value(&bd, result), native, "NC terminal mismatch vs native rhs");
    pin_k(&mut bd, result, native);
    assert!(
        bd.is_satisfied(),
        "NC terminal circuit unsatisfied (first bad row: {:?})",
        bd.first_unsatisfied_row()
    );
}

#[test]
fn nc_terminal_identity_rejects_tampered_y_zcol_lane() {
    let ell_d = 3usize;
    let ell_m = 2usize;
    let t = 3usize;
    let k_total = 3usize;
    let d_sz = 1usize << ell_d;

    let mut rng = Rng::new(0xC0_FE_D0_0002);
    let params = NeoParams::goldilocks_paper_b2();
    let b_norm = params.b;

    let gamma = rng.next_k();
    let alpha = rng.next_k_vec(ell_d);
    let beta_a = rng.next_k_vec(ell_d);
    let beta_r = rng.next_k_vec(ell_d);
    let beta_m = rng.next_k_vec(ell_m);
    let alpha_prime = rng.next_k_vec(ell_d);
    let s_col_prime = rng.next_k_vec(ell_m);
    let ch = Challenges {
        alpha,
        beta_a: beta_a.clone(),
        beta_r,
        beta_m: beta_m.clone(),
        gamma,
    };
    let out_me: Vec<CeClaim> = (0..k_total)
        .map(|i| build_test_ce_claim_with_zcol(0xCE_1000 + i as u64, t, d_sz, ell_m))
        .collect();
    let native = rhs_terminal_identity_nc(&params, &ch, &s_col_prime, &alpha_prime, &out_me);

    let mut bd = R1csBuilder::new();
    let gamma_var = alloc_witness_k(&mut bd, gamma);
    let beta_a_vars: Vec<KVar> = beta_a
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut bd, v))
        .collect();
    let beta_m_vars: Vec<KVar> = beta_m
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut bd, v))
        .collect();
    let s_col_prime_vars: Vec<KVar> = s_col_prime
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut bd, v))
        .collect();
    let alpha_prime_vars: Vec<KVar> = alpha_prime
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut bd, v))
        .collect();
    let output_y_zcol: Vec<Vec<KVar>> = out_me
        .iter()
        .map(|c| alloc_y_zcol_wires(&mut bd, &c.y_zcol))
        .collect();

    let result = enforce_nc_terminal_identity(
        &mut bd,
        &NcTerminalInputs {
            b: b_norm,
            gamma: gamma_var,
            beta_a: &beta_a_vars,
            beta_m: &beta_m_vars,
            s_col_prime: &s_col_prime_vars,
            alpha_prime: &alpha_prime_vars,
            output_y_zcol: &output_y_zcol,
        },
    )
    .expect("NC terminal");
    pin_k(&mut bd, result, native);
    assert!(bd.is_satisfied(), "baseline");

    let target = output_y_zcol[1][2].c0.col();
    let tampered = bd.witness()[target] + F::ONE;
    bd.tamper_witness(target, tampered);
    assert!(!bd.is_satisfied(), "tampered y_zcol must break NC terminal pin");
}

#[test]
fn nc_range_product_rejects_tampered_val() {
    // Tampering the input `val` wire after the gadget pins its output must
    // break the chain of K-mults that produces the product.
    let val = K::from_coeffs([F::from_u64(42), F::from_u64(13)]);
    let expected = native_range_product(val, 3);

    let mut bd = R1csBuilder::new();
    let v = alloc_witness_k(&mut bd, val);
    let out = enforce_nc_range_product(&mut bd, v, 3).expect("range_product");
    pin_k(&mut bd, out, expected);
    assert!(bd.is_satisfied(), "baseline");

    let target = v.c0.col();
    let tampered = bd.witness()[target] + F::ONE;
    bd.tamper_witness(target, tampered);
    assert!(!bd.is_satisfied(), "tampered val must break range_product pin");
}

#[test]
fn block_lane_initial_and_terminal_match_the_fixed_delayed_polynomial() {
    let gamma = K::from_coeffs([F::from_u64(3), F::from_u64(1)]);
    let producer_beta = K::from_coeffs([F::from_u64(5), F::from_u64(2)]);
    let batch_weight = K::from_coeffs([F::from_u64(7), F::from_u64(4)]);
    let block_point = [
        K::from_coeffs([F::from_u64(11), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(13), F::from_u64(2)]),
    ];
    let beta_block = [
        K::from_coeffs([F::from_u64(17), F::from_u64(3)]),
        K::from_coeffs([F::from_u64(19), F::from_u64(4)]),
    ];
    let lane_point = [
        K::from_coeffs([F::from_u64(23), F::from_u64(5)]),
        K::from_coeffs([F::from_u64(29), F::from_u64(6)]),
    ];
    let beta_lane = [
        K::from_coeffs([F::from_u64(31), F::from_u64(7)]),
        K::from_coeffs([F::from_u64(37), F::from_u64(8)]),
    ];
    let old_block = [
        K::from_coeffs([F::from_u64(41), F::from_u64(9)]),
        K::from_coeffs([F::from_u64(43), F::from_u64(10)]),
    ];
    let parent = [
        K::from_coeffs([F::from_u64(47), F::from_u64(11)]),
        K::from_coeffs([F::from_u64(53), F::from_u64(12)]),
        K::from_coeffs([F::from_u64(59), F::from_u64(13)]),
        K::from_coeffs([F::from_u64(61), F::from_u64(14)]),
    ];
    let outputs = [
        [k(1, 0), k(0, 1), k(1, 0), k(0, 1)],
        [k(2, 1), k(3, 1), k(5, 2), k(7, 3)],
        [k(11, 4), k(13, 5), k(17, 6), k(19, 7)],
    ];

    let expected_initial = batch_weight
        * parent
            .iter()
            .rev()
            .fold(K::ZERO, |value, coefficient| value * producer_beta + *coefficient);
    let values = outputs.map(|output| native_mle(&output, &lane_point));
    let mut gamma_power = K::ONE;
    let ordinary_sum = values.iter().fold(K::ZERO, |sum, value| {
        let next = sum + gamma_power * (*value * *value * *value - *value);
        gamma_power *= gamma;
        next
    });
    let ordinary = native_eq(&block_point, &beta_block) * native_eq(&lane_point, &beta_lane) * ordinary_sum;
    let running = values[1] + K::from(F::from_u64(2)) * values[2];
    let delayed = batch_weight
        * native_eq(&block_point, &old_block)
        * native_beta_power_selector(producer_beta, &lane_point)
        * running;
    let expected_terminal = ordinary + delayed;

    let mut builder = R1csBuilder::new();
    let gamma_var = alloc_witness_k(&mut builder, gamma);
    let producer_beta_var = alloc_witness_k(&mut builder, producer_beta);
    let batch_weight_var = alloc_witness_k(&mut builder, batch_weight);
    let block_vars = block_point.map(|value| alloc_witness_k(&mut builder, value));
    let beta_block_vars = beta_block.map(|value| alloc_witness_k(&mut builder, value));
    let lane_vars = lane_point.map(|value| alloc_witness_k(&mut builder, value));
    let beta_lane_vars = beta_lane.map(|value| alloc_witness_k(&mut builder, value));
    let old_block_vars = old_block.map(|value| alloc_witness_k(&mut builder, value));
    let parent_vars = parent.map(|value| alloc_witness_k(&mut builder, value));
    let output_vars = outputs
        .iter()
        .map(|output| {
            output
                .iter()
                .copied()
                .map(|value| alloc_witness_k(&mut builder, value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let initial =
        enforce_block_lane_claimed_initial(&mut builder, Some(&parent_vars), producer_beta_var, batch_weight_var);
    let terminal = enforce_block_lane_terminal_identity(
        &mut builder,
        &BlockLaneTerminalInputs {
            fresh_count: 1,
            gamma: gamma_var,
            beta_lane: &beta_lane_vars,
            beta_block: &beta_block_vars,
            producer_beta: producer_beta_var,
            batch_weight: batch_weight_var,
            block_point: &block_vars,
            lane_point: &lane_vars,
            output_y_zcol: &output_vars,
            pending_old_block: Some(&old_block_vars),
        },
    )
    .expect("block/lane terminal");

    assert_eq!(k_value(&builder, initial), expected_initial);
    assert_eq!(k_value(&builder, terminal), expected_terminal);
    pin_k(&mut builder, initial, expected_initial);
    pin_k(&mut builder, terminal, expected_terminal);
    assert!(
        builder.is_satisfied(),
        "fixed delayed polynomial circuit disagrees at row {:?}",
        builder.first_unsatisfied_row()
    );

    let target = output_vars[2][3].c0.col();
    builder.tamper_witness(target, builder.witness()[target] + F::ONE);
    assert!(
        !builder.is_satisfied(),
        "raw running-output mutation must break the pinned terminal identity"
    );
}

#[test]
fn zero_batch_weight_removes_only_the_delayed_summand() {
    let outputs = vec![vec![k(0, 0), k(1, 0)], vec![k(1, 0), k(0, 0)]];
    let point = [k(3, 1)];
    let beta = [k(5, 2)];

    let mut builder = R1csBuilder::new();
    let gamma = alloc_witness_k(&mut builder, k(7, 3));
    let producer_beta = alloc_witness_k(&mut builder, k(11, 4));
    let batch_weight = alloc_witness_k(&mut builder, K::ZERO);
    let point_vars = point.map(|value| alloc_witness_k(&mut builder, value));
    let beta_vars = beta.map(|value| alloc_witness_k(&mut builder, value));
    let old_vars = [alloc_witness_k(&mut builder, k(13, 5))];
    let output_vars = outputs
        .iter()
        .map(|values| {
            values
                .iter()
                .copied()
                .map(|value| alloc_witness_k(&mut builder, value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let result = enforce_block_lane_terminal_identity(
        &mut builder,
        &BlockLaneTerminalInputs {
            fresh_count: 1,
            gamma,
            beta_lane: &beta_vars,
            beta_block: &beta_vars,
            producer_beta,
            batch_weight,
            block_point: &point_vars,
            lane_point: &point_vars,
            output_y_zcol: &output_vars,
            pending_old_block: Some(&old_vars),
        },
    )
    .expect("zero-weight block/lane terminal");

    let values = outputs
        .iter()
        .map(|values| native_mle(values, &point))
        .collect::<Vec<_>>();
    let ordinary_sum =
        (values[0] * values[0] * values[0] - values[0]) + k(7, 3) * (values[1] * values[1] * values[1] - values[1]);
    let expected = native_eq(&point, &beta) * native_eq(&point, &beta) * ordinary_sum;
    assert_eq!(k_value(&builder, result), expected);
    pin_k(&mut builder, result, expected);
    assert!(
        builder.is_satisfied(),
        "batchWeight=0 must leave the ordinary NC identity intact"
    );
}
