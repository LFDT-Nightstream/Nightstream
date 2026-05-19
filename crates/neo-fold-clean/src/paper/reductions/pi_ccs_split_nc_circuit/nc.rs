//! SplitNcV1 — NC channel (range_product, sumcheck driver, terminal identity).
//!
//! Mirrors:
//! - The `range_product` factor (centered low-norm polynomial) baked into
//!   `rhs_terminal_identity_nc` in
//!   `neo_reductions::engines::optimized_engine::terminal_identities`.
//! - The NC block in
//!   `optimized_verify_with_cache_and_public_instance_digest_impl` (raw
//!   absorbs `[8]`, `[9]`, `K::ZERO.as_coeffs()`, `[10]`, then
//!   `verify_sumcheck_rounds_poseidon_v3`).
//! - `rhs_terminal_identity_nc` itself.
//!
//! Sub-step status: G (range_product) landed; H (sumcheck driver + terminal
//! identity) pending.

use neo_math::F;
use neo_reductions::engines::utils::{PI_CCS_SUMCHECK_INITIAL_RAW_TAG, PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG};
use neo_reductions::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG;
use p3_field::PrimeCharacteristicRing;

use super::Error;
use crate::engine::r1cs_circuit::builder::Lc;
use crate::engine::r1cs_circuit::field_ext::{alloc_klc, enforce_k_mul, KLc, KVar};
use crate::engine::r1cs_circuit::sumcheck::{
    enforce_chi_alpha, enforce_eq_k, enforce_sumcheck_rounds_engine, gamma_powers,
};
use crate::engine::r1cs_circuit::transcript::TranscriptGadget;
use crate::engine::r1cs_circuit::R1csBuilder;

/// In-circuit mirror of the private `range_product(val, b)` factor used by
/// `rhs_terminal_identity_nc` in `neo_reductions::engines::optimized_engine`.
///
/// Returns the K-value `∏_{t = -(b-1)..=(b-1)} (val - t)`. Vanishes iff
/// `val` is an integer in the centered range `{-(b-1), …, -1, 0, 1, …, b-1}`,
/// which is the low-norm condition the NC channel enforces.
///
/// Costs `2b - 2` K-multiplications (one per factor after the first). The
/// `b = 2` specialization matches the existing
/// [`crate::engine::r1cs_circuit::sumcheck::enforce_norm_check_b2`] gadget:
/// `(val + 1)·val·(val - 1)`.
///
/// `b == 0` is rejected (the empty product would silently return `K::ONE`,
/// which the soundness contract would not catch).
pub fn enforce_nc_range_product(builder: &mut R1csBuilder, val: KVar, b: u32) -> Result<KVar, Error> {
    if b == 0 {
        return Err(Error::Shape(
            "enforce_nc_range_product: norm bound b must be >= 1".into(),
        ));
    }

    // Build factor (val - t) as a `KLc` (no fresh allocation; constants live
    // in the Lc itself), then fold into the running product via one
    // `enforce_k_mul` per successive factor.
    let mut acc: Option<KVar> = None;
    let b_minus_1 = (b as i64) - 1;
    for t in -b_minus_1..=b_minus_1 {
        // factor = val - t. In F, `-t` is the additive negation of `F::from_i64(t)`.
        let neg_t = -F::from_i64(t);
        let factor_lc = KLc {
            c0: {
                let mut lc = Lc::from_var(val.c0);
                lc.constant = neg_t;
                lc
            },
            c1: Lc::from_var(val.c1),
        };

        acc = Some(match acc {
            None => alloc_klc(builder, &factor_lc),
            Some(prev) => enforce_k_mul(builder, &KLc::from_var(prev), &factor_lc),
        });
    }

    Ok(acc.expect("range_product: at least one factor for b >= 1"))
}

// ── NC sumcheck driver ────────────────────────────────────────────────────

/// Result of running the NC channel sumcheck.
///
/// Native `optimized_verify_with_cache_and_public_instance_digest_impl`
/// splits the NC challenge vector as `r_all_nc = s_col_prime ‖ alpha_prime`
/// with `len(s_col_prime) = ell_m`, `len(alpha_prime) = ell_d`. We expose
/// both halves plus the final running sum so the NC terminal-identity check
/// can consume them directly.
#[derive(Clone, Debug)]
pub struct NcSumcheckResult {
    pub s_col_prime: Vec<KVar>,
    pub alpha_prime: Vec<KVar>,
    pub final_sum: KVar,
}

/// Mirror of the NC channel inside
/// `optimized_verify_with_cache_and_public_instance_digest_impl`. Native flow:
///
/// ```text
/// tr.append_fields_raw([PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG])     // 8
/// tr.append_fields_raw([PI_CCS_SUMCHECK_INITIAL_RAW_TAG])       // 9
/// tr.append_fields_raw(K::ZERO.as_coeffs())                     // [0, 0]
/// tr.append_fields_raw([SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG]) // 10
/// verify_sumcheck_rounds_poseidon_v3(tr, d_sc, K::ZERO, &rounds_nc)
/// ```
///
/// The NC channel's initial claim is always `K::ZERO` (matches the native
/// `claimed_nc = K::ZERO`). We allocate a zero `KVar` and absorb its two
/// lanes via `append_fields_raw_vars`, matching the native two-field absorb
/// of `K::ZERO.as_coeffs()`.
///
/// Strict round-count check matches the native verifier:
/// `split Π_CCS: sumcheck_rounds_nc.len()=… expected ell_m + ell_d`.
pub fn enforce_nc_sumcheck_driver(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    ell_m: usize,
    ell_d: usize,
    d_sc: usize,
    rounds: &[Vec<KVar>],
) -> Result<NcSumcheckResult, Error> {
    let want_rounds = ell_m
        .checked_add(ell_d)
        .ok_or_else(|| Error::Shape("NC sumcheck round count overflow".into()))?;
    if rounds.len() != want_rounds {
        return Err(Error::Shape(format!(
            "NC sumcheck rounds.len ({}) must equal ell_m + ell_d ({})",
            rounds.len(),
            want_rounds
        )));
    }
    // Mirror native `verify_sumcheck_rounds_poseidon_v3` degree-bound reject.
    for (i, round) in rounds.iter().enumerate() {
        if round.len() > d_sc + 1 {
            return Err(Error::Shape(format!(
                "NC sumcheck round {i} degree too high: coeffs={}, max={}",
                round.len(),
                d_sc + 1
            )));
        }
    }

    // Native claimed_nc = K::ZERO. Allocate a zero KVar so its lanes can be
    // absorbed via `append_fields_raw_vars`.
    let claimed_nc = alloc_klc(builder, &KLc::from_base_const(F::ZERO));

    transcript.append_fields_raw_const(builder, &[F::from_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)]);
    transcript.append_fields_raw_const(builder, &[F::from_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    transcript.append_fields_raw_vars(builder, &[claimed_nc.c0, claimed_nc.c1]);
    transcript.append_fields_raw_const(builder, &[F::from_u64(SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);

    let (challenges, final_sum) = enforce_sumcheck_rounds_engine(builder, transcript, claimed_nc, rounds);

    debug_assert_eq!(challenges.len(), want_rounds);
    let s_col_prime = challenges[..ell_m].to_vec();
    let alpha_prime = challenges[ell_m..].to_vec();

    Ok(NcSumcheckResult {
        s_col_prime,
        alpha_prime,
        final_sum,
    })
}

// ── NC terminal identity ──────────────────────────────────────────────────

/// Witness wires for the NC channel's terminal identity.
///
/// Mirrors the arguments of `rhs_terminal_identity_nc` in
/// `neo_reductions::engines::optimized_engine::terminal_identities`:
/// - `b`: norm bound (drives `range_product`'s degree).
/// - `gamma`, `beta_a`, `beta_m`: from [`super::EngineChallenges`].
/// - `s_col_prime`, `alpha_prime`: split from the NC sumcheck challenge
///   vector (length `ell_m` and `ell_d`).
/// - `output_y_zcol[i][rho]`: the ρ-th coefficient of the NC output column
///   `y_{zcol,i}`. Outer length is `k_total = k_mcs + me_inputs.len()`,
///   inner is `>= 2^ell_d` (consumed prefix is `2^ell_d`).
pub struct NcTerminalInputs<'a> {
    pub b: u32,
    pub gamma: KVar,
    pub beta_a: &'a [KVar],
    pub beta_m: &'a [KVar],
    pub s_col_prime: &'a [KVar],
    pub alpha_prime: &'a [KVar],
    pub output_y_zcol: &'a [Vec<KVar>],
}

/// Mirror of `rhs_terminal_identity_nc` in
/// `neo_reductions::engines::optimized_engine::terminal_identities`.
/// Computes the NC channel's right-hand-side claim:
///
/// ```text
/// rhs_nc = eq(α', β_a) · eq(s'_col, β_m)
///        · Σ_i γ^{i+1} · range_product(⟨y_{zcol,i}, χ_{α'}⟩, b)
/// ```
///
/// where `i` is 0-indexed and `γ^{i+1}` matches native's `g = γ` initial
/// (i.e. γ^1 for i=0, γ^2 for i=1, …).
pub fn enforce_nc_terminal_identity(builder: &mut R1csBuilder, inputs: &NcTerminalInputs<'_>) -> Result<KVar, Error> {
    let k_total = inputs.output_y_zcol.len();
    if k_total == 0 {
        return Err(Error::Shape("NC terminal: need at least one output (k_total=0)".into()));
    }
    if inputs.alpha_prime.len() != inputs.beta_a.len() {
        return Err(Error::Shape(format!(
            "NC terminal: alpha_prime.len ({}) != beta_a.len ({})",
            inputs.alpha_prime.len(),
            inputs.beta_a.len()
        )));
    }
    if inputs.s_col_prime.len() != inputs.beta_m.len() {
        return Err(Error::Shape(format!(
            "NC terminal: s_col_prime.len ({}) != beta_m.len ({})",
            inputs.s_col_prime.len(),
            inputs.beta_m.len()
        )));
    }
    if inputs.b == 0 {
        return Err(Error::Shape("NC terminal: b must be >= 1".into()));
    }

    // eq(α', β_a) · eq(s'_col, β_m).
    let eq_alpha_beta = enforce_eq_k(builder, inputs.alpha_prime, inputs.beta_a);
    let eq_s_beta = enforce_eq_k(builder, inputs.s_col_prime, inputs.beta_m);
    let eq_apsp_beta = enforce_k_mul(builder, &KLc::from_var(eq_alpha_beta), &KLc::from_var(eq_s_beta));

    // χ_{α'} over the Ajtai-padded ring dimension.
    let chi_alpha_prime = enforce_chi_alpha(builder, inputs.alpha_prime);
    let d_sz = chi_alpha_prime.len();

    // γ-powers up to γ^{k_total}. Native uses `g = γ; g *= γ` per iteration,
    // so output i is weighted by γ^{i+1}.
    let powers = gamma_powers(builder, inputs.gamma, k_total + 1);

    // Σ_i γ^{i+1} · range_product(⟨y_zcol_i, χ_{α'}⟩, b).
    let mut nc_sum_lc = KLc::zero();
    for (i, y_zcol) in inputs.output_y_zcol.iter().enumerate() {
        if y_zcol.len() < d_sz {
            return Err(Error::Shape(format!(
                "NC terminal: output_y_zcol[{}].len ({}) < 2^ell_d ({})",
                i,
                y_zcol.len(),
                d_sz
            )));
        }

        // y_eval = ⟨y_zcol_i, χ_{α'}⟩.
        let mut y_eval_lc = KLc::zero();
        for rho in 0..d_sz {
            let term = enforce_k_mul(
                builder,
                &KLc::from_var(y_zcol[rho]),
                &KLc::from_var(chi_alpha_prime[rho]),
            );
            y_eval_lc = KLc {
                c0: y_eval_lc.c0.add_scaled(&Lc::from_var(term.c0), F::ONE),
                c1: y_eval_lc.c1.add_scaled(&Lc::from_var(term.c1), F::ONE),
            };
        }
        let y_eval = alloc_klc(builder, &y_eval_lc);

        // N_i = range_product(y_eval, b).
        let n_i = enforce_nc_range_product(builder, y_eval, inputs.b)?;

        // weighted = γ^{i+1} · N_i.
        let weighted = enforce_k_mul(builder, &KLc::from_var(powers[i + 1]), &KLc::from_var(n_i));
        nc_sum_lc = KLc {
            c0: nc_sum_lc.c0.add_scaled(&Lc::from_var(weighted.c0), F::ONE),
            c1: nc_sum_lc.c1.add_scaled(&Lc::from_var(weighted.c1), F::ONE),
        };
    }
    let nc_prime_sum = alloc_klc(builder, &nc_sum_lc);

    // rhs_nc = eq_apsp_beta · nc_prime_sum.
    Ok(enforce_k_mul(
        builder,
        &KLc::from_var(eq_apsp_beta),
        &KLc::from_var(nc_prime_sum),
    ))
}
