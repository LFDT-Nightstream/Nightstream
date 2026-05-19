//! SplitNcV1 — FE channel (claimed initial, sumcheck driver, terminal identity).
//!
//! Mirrors:
//! - `claimed_initial_sum_from_inputs_with_k_mcs` in
//!   `neo_reductions::engines::paper_exact_engine` (computes T).
//! - FE block inside `optimized_verify_with_cache_and_public_instance_digest_impl`
//!   (raw absorbs `[7]`, `[9]`, `claimed_initial.as_coeffs()`, `[10]`, then
//!   `verify_sumcheck_rounds_poseidon_v3`).
//! - `rhs_terminal_identity_fe_with_k_mcs` in
//!   `neo_reductions::engines::optimized_engine::terminal_identities`.
//!
//! All three are exercised by native-helper parity tests in
//! `tests/reductions/pi_ccs_split_nc_fe.rs`.

use neo_ccs::SparsePoly;
use neo_math::F;
use neo_reductions::engines::utils::{PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG, PI_CCS_SUMCHECK_INITIAL_RAW_TAG};
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

// ── FE claimed-initial sum ────────────────────────────────────────────────

/// Witness wires for the FE channel's claimed-initial sum.
///
/// Mirrors the arguments of
/// `neo_reductions::engines::paper_exact_engine::claimed_initial_sum_from_inputs_with_k_mcs`,
/// in `(s, ch, k_mcs, me_inputs)` order:
/// - `t`: number of CCS matrices (= `s.t()`).
/// - `ell_d`: log2 of the Ajtai-padded ring dimension (drives `χ_α` size).
/// - `gamma`, `alpha`: from [`super::EngineChallenges`].
/// - `k_mcs`: number of fresh CCS instances.
/// - `running_y_ring[idx][j][ρ]`: the ρ-th coefficient of the j-th matrix
///   output `y_{me_inputs[idx], j}`. Outer length is `me_inputs.len()`,
///   middle length is `t`, inner length must be `>= 2^ell_d` (native asserts
///   `yj.len() >= d_sz`; we enforce strict equality on the consumed prefix).
pub struct FeClaimedInitialInputs<'a> {
    pub k_mcs: usize,
    pub t: usize,
    pub ell_d: usize,
    pub gamma: KVar,
    pub alpha: &'a [KVar],
    pub running_y_ring: &'a [Vec<Vec<KVar>>],
}

/// Mirror of `claimed_initial_sum_from_inputs_with_k_mcs` in
/// `neo_reductions::engines::paper_exact_engine`. Computes the FE channel's
/// public claimed sum `T`:
///
/// ```text
/// T = γ^{k_total} · Σ_{j=0..t} Σ_{idx=0..|me|} γ^{k_mcs + idx + j·k_total}
///     · ⟨ y_{me_inputs[idx], j}, χ_α ⟩
/// ```
///
/// where `k_total = k_mcs + me_inputs.len()`. Edge case `k_total < 2`
/// returns the K-zero wire — matching native's early return.
pub fn enforce_fe_claimed_initial(builder: &mut R1csBuilder, inputs: &FeClaimedInitialInputs) -> Result<KVar, Error> {
    if inputs.alpha.len() != inputs.ell_d {
        return Err(Error::Shape(format!(
            "enforce_fe_claimed_initial: alpha.len ({}) must equal ell_d ({})",
            inputs.alpha.len(),
            inputs.ell_d
        )));
    }
    let d_sz = 1usize << inputs.ell_d;
    for (idx, row_outer) in inputs.running_y_ring.iter().enumerate() {
        if row_outer.len() != inputs.t {
            return Err(Error::Shape(format!(
                "enforce_fe_claimed_initial: running_y_ring[{}].len ({}) must equal t ({})",
                idx,
                row_outer.len(),
                inputs.t
            )));
        }
        for (j, row) in row_outer.iter().enumerate() {
            if row.len() < d_sz {
                return Err(Error::Shape(format!(
                    "enforce_fe_claimed_initial: running_y_ring[{}][{}].len ({}) must be >= 2^ell_d ({})",
                    idx,
                    j,
                    row.len(),
                    d_sz
                )));
            }
        }
    }

    let k_total = inputs.k_mcs + inputs.running_y_ring.len();

    // Native early return when there's no Eval block.
    if k_total < 2 {
        return Ok(alloc_klc(builder, &KLc::from_base_const(F::ZERO)));
    }

    // χ_α table over the Ajtai-padded ring dimension.
    let chi_alpha = enforce_chi_alpha(builder, inputs.alpha);
    debug_assert_eq!(chi_alpha.len(), d_sz);

    // γ-power table covering `γ^0 .. γ^{t·k_total}`. We need:
    //  - γ^{k_total}             (outer scale)
    //  - γ^{k_mcs + idx + j·k_total}  for idx ∈ [0, |me|), j ∈ [0, t)
    // Maximum exponent across the inner sum is `t·k_total - 1`; we extend by
    // one to also have γ^{t·k_total} for symmetry, then pick `γ^{k_total}`
    // out of the table by index.
    let powers = gamma_powers(builder, inputs.gamma, inputs.t * k_total + 1);
    let gamma_to_k = powers[k_total];

    // Inner sum: Σ_j Σ_idx γ^{...} · ⟨y, χ_α⟩.
    //
    // We build the running accumulator as a `KLc` and `alloc_klc` once at
    // the end of each (idx, j) term. The per-term mul is via `enforce_k_mul`.
    let mut inner_lc = KLc::zero();
    for j in 0..inputs.t {
        for (idx, row_outer) in inputs.running_y_ring.iter().enumerate() {
            let y_row = &row_outer[j];

            // ⟨y, χ_α⟩ over the first d_sz lanes. Build the dot product as
            // a chain of K-mul allocations; the sum is linear.
            let mut dot_lc = KLc::zero();
            for rho in 0..d_sz {
                let prod = enforce_k_mul(builder, &KLc::from_var(y_row[rho]), &KLc::from_var(chi_alpha[rho]));
                dot_lc = KLc {
                    c0: dot_lc.c0.add_scaled(&Lc::from_var(prod.c0), F::ONE),
                    c1: dot_lc.c1.add_scaled(&Lc::from_var(prod.c1), F::ONE),
                };
            }
            let y_eval = alloc_klc(builder, &dot_lc);

            // weight = γ^{k_mcs + idx + j·k_total}.
            let weight_idx = inputs.k_mcs + idx + j * k_total;
            let weight = powers[weight_idx];

            // term = weight · y_eval.
            let term = enforce_k_mul(builder, &KLc::from_var(weight), &KLc::from_var(y_eval));
            inner_lc = KLc {
                c0: inner_lc.c0.add_scaled(&Lc::from_var(term.c0), F::ONE),
                c1: inner_lc.c1.add_scaled(&Lc::from_var(term.c1), F::ONE),
            };
        }
    }
    let inner = alloc_klc(builder, &inner_lc);

    // T = γ^{k_total} · inner.
    Ok(enforce_k_mul(
        builder,
        &KLc::from_var(gamma_to_k),
        &KLc::from_var(inner),
    ))
}

// ── FE sumcheck driver ────────────────────────────────────────────────────

/// Result of running the FE channel sumcheck.
///
/// Native `optimized_verify_with_cache_and_public_instance_digest_impl`
/// splits the FE challenge vector as `r_all = r_prime ‖ alpha_prime` with
/// `len(r_prime) = ell_n`, `len(alpha_prime) = ell_d`. We expose both
/// halves plus the final running sum so the FE terminal-identity check can
/// consume them directly.
#[derive(Clone, Debug)]
pub struct FeSumcheckResult {
    pub r_prime: Vec<KVar>,
    pub alpha_prime: Vec<KVar>,
    pub final_sum: KVar,
}

/// Mirror of the FE channel inside
/// `optimized_verify_with_cache_and_public_instance_digest_impl`. Native flow:
///
/// ```text
/// tr.append_fields_raw([PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG])     // 7
/// tr.append_fields_raw([PI_CCS_SUMCHECK_INITIAL_RAW_TAG])       // 9
/// tr.append_fields_raw(claimed_initial.as_coeffs())             // [c0, c1]
/// tr.append_fields_raw([SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG]) // 10
/// verify_sumcheck_rounds_poseidon_v3(tr, d_sc, claimed_initial, &rounds)
/// ```
///
/// We mirror the four raw absorbs, then call [`enforce_sumcheck_rounds_engine`]
/// to drive the per-round absorb/squeeze sequence and assemble the FE
/// challenge vector. Strict round-count check matches the native verifier
/// (`split Π_CCS: sumcheck_rounds.len()=… expected ell_n + ell_d`).
pub fn enforce_fe_sumcheck_driver(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    ell_n: usize,
    ell_d: usize,
    d_sc: usize,
    claimed_initial: KVar,
    rounds: &[Vec<KVar>],
) -> Result<FeSumcheckResult, Error> {
    let want_rounds = ell_n
        .checked_add(ell_d)
        .ok_or_else(|| Error::Shape("FE sumcheck round count overflow".into()))?;
    if rounds.len() != want_rounds {
        return Err(Error::Shape(format!(
            "FE sumcheck rounds.len ({}) must equal ell_n + ell_d ({})",
            rounds.len(),
            want_rounds
        )));
    }
    // Mirror native `verify_sumcheck_rounds_poseidon_v3` degree-bound reject.
    for (i, round) in rounds.iter().enumerate() {
        if round.len() > d_sc + 1 {
            return Err(Error::Shape(format!(
                "FE sumcheck round {i} degree too high: coeffs={}, max={}",
                round.len(),
                d_sc + 1
            )));
        }
    }

    transcript.append_fields_raw_const(builder, &[F::from_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)]);
    transcript.append_fields_raw_const(builder, &[F::from_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    transcript.append_fields_raw_vars(builder, &[claimed_initial.c0, claimed_initial.c1]);
    transcript.append_fields_raw_const(builder, &[F::from_u64(SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);

    let (challenges, final_sum) = enforce_sumcheck_rounds_engine(builder, transcript, claimed_initial, rounds);

    debug_assert_eq!(challenges.len(), want_rounds);
    let r_prime = challenges[..ell_n].to_vec();
    let alpha_prime = challenges[ell_n..].to_vec();

    Ok(FeSumcheckResult {
        r_prime,
        alpha_prime,
        final_sum,
    })
}

// ── Sparse polynomial eval gadget ─────────────────────────────────────────

/// Evaluate `Σ_term term.coeff · Π_j x[j]^{term.exps[j]}` in-circuit, where
/// `x[j]` are K-valued wires and `term.coeff` is a base-field constant.
/// Byte-for-byte mirror of `SparsePoly::eval_in_ext::<K>` in `neo-ccs`.
///
/// Skips variables with exponent 0 (native shortcut). For non-zero `pow`,
/// builds `x^pow` via `pow - 1` K-mults, then folds into the running `m`
/// via one K-mul per non-zero variable. Final accumulation across terms
/// is linear (no extra K-mults).
///
/// This is what `F'` in the FE terminal identity instantiates per-instance:
/// `f_i = poly.eval_in_ext(ct(y'_{i, 0..t-1}))`.
fn enforce_sparse_poly_eval_ext(builder: &mut R1csBuilder, poly: &SparsePoly<F>, xs: &[KVar]) -> Result<KVar, Error> {
    if xs.len() != poly.arity() {
        return Err(Error::Shape(format!(
            "sparse poly arity mismatch: xs.len ({}) != arity ({})",
            xs.len(),
            poly.arity()
        )));
    }

    let mut acc = KLc::zero();
    for term in poly.terms() {
        if term.exps.len() != poly.arity() {
            return Err(Error::Shape(format!(
                "sparse poly term exps.len ({}) != arity ({})",
                term.exps.len(),
                poly.arity()
            )));
        }

        // Start with the coefficient as a base-field constant lifted into K.
        let mut m = alloc_klc(builder, &KLc::from_base_const(term.coeff));

        for (xi, &pow) in xs.iter().zip(term.exps.iter()) {
            if pow == 0 {
                continue;
            }
            // p = xi^pow via pow-1 K-mults.
            let mut p = *xi;
            for _ in 1..pow {
                p = enforce_k_mul(builder, &KLc::from_var(p), &KLc::from_var(*xi));
            }
            // m *= p
            m = enforce_k_mul(builder, &KLc::from_var(m), &KLc::from_var(p));
        }

        acc = KLc {
            c0: acc.c0.add_scaled(&Lc::from_var(m.c0), F::ONE),
            c1: acc.c1.add_scaled(&Lc::from_var(m.c1), F::ONE),
        };
    }

    Ok(alloc_klc(builder, &acc))
}

// ── FE terminal identity ──────────────────────────────────────────────────

/// Witness wires for the FE channel's terminal identity.
///
/// Mirrors the arguments of `rhs_terminal_identity_fe_with_k_mcs` in
/// `neo_reductions::engines::optimized_engine::terminal_identities`:
/// - `poly`: the CCS structure's polynomial `s.f` (used for `F'`).
/// - `t`: number of CCS matrices.
/// - `k_mcs`: number of fresh CCS instances.
/// - `gamma`, `alpha`, `beta_a`, `beta_r`: from [`super::EngineChallenges`].
/// - `r_prime`, `alpha_prime`: split from the FE sumcheck challenge vector
///   (`r_prime` length `ell_n`, `alpha_prime` length `ell_d`).
/// - `me_input_r`: the shared evaluation point `r` carried by every running
///   ME input (per `shared_me_input_r`). Required iff `k_total > k_mcs`.
/// - `output_y_ring[i][j][rho]`: the ρ-th coefficient of the output
///   `y'_{i, j}`. Outer length is `k_total = k_mcs + me_inputs.len()`,
///   middle is `t`, inner is `>= 2^ell_d` (consumed prefix is `2^ell_d`).
pub struct FeTerminalInputs<'a> {
    pub poly: &'a SparsePoly<F>,
    pub t: usize,
    pub k_mcs: usize,
    pub gamma: KVar,
    pub alpha: &'a [KVar],
    pub beta_a: &'a [KVar],
    pub beta_r: &'a [KVar],
    pub r_prime: &'a [KVar],
    pub alpha_prime: &'a [KVar],
    pub me_input_r: Option<&'a [KVar]>,
    pub output_y_ring: &'a [Vec<Vec<KVar>>],
}

/// Mirror of `rhs_terminal_identity_fe_with_k_mcs` in
/// `neo_reductions::engines::optimized_engine::terminal_identities`.
/// Computes the FE channel's right-hand-side claim:
///
/// ```text
/// rhs_fe = eq(α', β_a) · eq(r', β_r) · F'
///        + eq(α', α) · eq(r', r_in) · γ^{k_total} · eval_sum
///
/// F'        = Σ_{i ∈ [0, k_mcs)} γ^i · s.f.eval_in_ext(ct(y'_{i, 0..t-1}))
/// eval_sum  = Σ_{j, i ∈ [k_mcs, k_total)} γ^{i + j·k_total} · ⟨y'_{i,j}, χ_{α'}⟩
/// ```
///
/// where the second term is `K::ZERO` when `k_total <= k_mcs` (no running
/// ME inputs in this fold).
pub fn enforce_fe_terminal_identity(builder: &mut R1csBuilder, inputs: &FeTerminalInputs<'_>) -> Result<KVar, Error> {
    let k_total = inputs.output_y_ring.len();
    if k_total == 0 {
        return Err(Error::Shape("FE terminal: need at least one output (k_total=0)".into()));
    }
    if inputs.k_mcs == 0 || inputs.k_mcs > k_total {
        return Err(Error::Shape(format!(
            "FE terminal: invalid k_mcs={} for k_total={}",
            inputs.k_mcs, k_total
        )));
    }
    if inputs.alpha_prime.len() != inputs.beta_a.len() {
        return Err(Error::Shape(format!(
            "FE terminal: alpha_prime.len ({}) != beta_a.len ({})",
            inputs.alpha_prime.len(),
            inputs.beta_a.len()
        )));
    }
    if inputs.alpha_prime.len() != inputs.alpha.len() {
        return Err(Error::Shape(format!(
            "FE terminal: alpha_prime.len ({}) != alpha.len ({})",
            inputs.alpha_prime.len(),
            inputs.alpha.len()
        )));
    }
    if inputs.r_prime.len() != inputs.beta_r.len() {
        return Err(Error::Shape(format!(
            "FE terminal: r_prime.len ({}) != beta_r.len ({})",
            inputs.r_prime.len(),
            inputs.beta_r.len()
        )));
    }
    if inputs.t != inputs.poly.arity() {
        return Err(Error::Shape(format!(
            "FE terminal: t ({}) != poly.arity ({})",
            inputs.t,
            inputs.poly.arity()
        )));
    }
    for (i, row_outer) in inputs.output_y_ring.iter().enumerate() {
        if row_outer.len() != inputs.t {
            return Err(Error::Shape(format!(
                "FE terminal: output_y_ring[{}].len ({}) != t ({})",
                i,
                row_outer.len(),
                inputs.t
            )));
        }
    }

    // eq(α', β_a) · eq(r', β_r).
    let eq_alpha_beta = enforce_eq_k(builder, inputs.alpha_prime, inputs.beta_a);
    let eq_r_beta = enforce_eq_k(builder, inputs.r_prime, inputs.beta_r);
    let eq_beta = enforce_k_mul(builder, &KLc::from_var(eq_alpha_beta), &KLc::from_var(eq_r_beta));

    // eq(α', α) · eq(r', r_in) when there are running ME inputs; otherwise 0.
    let eq_ar = if k_total > inputs.k_mcs {
        let r = inputs
            .me_input_r
            .ok_or_else(|| Error::Shape("FE terminal: missing me_input_r when k_total > k_mcs".into()))?;
        if r.len() != inputs.r_prime.len() {
            return Err(Error::Shape(format!(
                "FE terminal: me_input_r.len ({}) != r_prime.len ({})",
                r.len(),
                inputs.r_prime.len()
            )));
        }
        let eq_alpha = enforce_eq_k(builder, inputs.alpha_prime, inputs.alpha);
        let eq_r = enforce_eq_k(builder, inputs.r_prime, r);
        enforce_k_mul(builder, &KLc::from_var(eq_alpha), &KLc::from_var(eq_r))
    } else {
        alloc_klc(builder, &KLc::from_base_const(F::ZERO))
    };

    // γ-powers up to γ^{t·k_total}. Needed for:
    //  - γ^i, i ∈ [0, k_mcs) (F' weights)
    //  - γ^{i + j·k_total}, i ∈ [k_mcs, k_total), j ∈ [0, t) (eval_sum weights)
    //  - γ^{k_total} (outer scale on eval_sum)
    let powers = gamma_powers(builder, inputs.gamma, inputs.t * k_total + 1);
    let gamma_to_k = powers[k_total];

    // F' = Σ_{i ∈ [0, k_mcs)} γ^i · poly.eval_in_ext(ct(y'_{i, 0..t-1})).
    //
    // ct(y'_{i, j}) = output_y_ring[i][j][0] (the "constant term" — first lane).
    let mut f_lc = KLc::zero();
    for i in 0..inputs.k_mcs {
        let mut m_vals: Vec<KVar> = Vec::with_capacity(inputs.t);
        for j in 0..inputs.t {
            let row = &inputs.output_y_ring[i][j];
            let first = *row.first().ok_or_else(|| {
                Error::Shape(format!(
                    "FE terminal: output_y_ring[{}][{}] empty — needs constant term",
                    i, j
                ))
            })?;
            m_vals.push(first);
        }
        let f_i = enforce_sparse_poly_eval_ext(builder, inputs.poly, &m_vals)?;
        let weighted = enforce_k_mul(builder, &KLc::from_var(powers[i]), &KLc::from_var(f_i));
        f_lc = KLc {
            c0: f_lc.c0.add_scaled(&Lc::from_var(weighted.c0), F::ONE),
            c1: f_lc.c1.add_scaled(&Lc::from_var(weighted.c1), F::ONE),
        };
    }
    let f_prime = alloc_klc(builder, &f_lc);

    // eval_sum = Σ_{j, i ∈ [k_mcs, k_total)} γ^{i + j·k_total} · ⟨y'_{i,j}, χ_{α'}⟩.
    // Skipped entirely when k_total <= k_mcs.
    let chi_alpha_prime = enforce_chi_alpha(builder, inputs.alpha_prime);
    let d_sz = chi_alpha_prime.len();

    let mut eval_lc = KLc::zero();
    if k_total > inputs.k_mcs {
        for j in 0..inputs.t {
            for i_abs in inputs.k_mcs..k_total {
                let y = &inputs.output_y_ring[i_abs][j];
                if y.len() < d_sz {
                    return Err(Error::Shape(format!(
                        "FE terminal: output_y_ring[{}][{}].len ({}) < 2^ell_d ({})",
                        i_abs,
                        j,
                        y.len(),
                        d_sz
                    )));
                }

                let mut y_eval_lc = KLc::zero();
                for rho in 0..d_sz {
                    let term = enforce_k_mul(builder, &KLc::from_var(y[rho]), &KLc::from_var(chi_alpha_prime[rho]));
                    y_eval_lc = KLc {
                        c0: y_eval_lc.c0.add_scaled(&Lc::from_var(term.c0), F::ONE),
                        c1: y_eval_lc.c1.add_scaled(&Lc::from_var(term.c1), F::ONE),
                    };
                }
                let y_eval = alloc_klc(builder, &y_eval_lc);

                let weight_idx = i_abs + j * k_total;
                let weighted = enforce_k_mul(builder, &KLc::from_var(powers[weight_idx]), &KLc::from_var(y_eval));
                eval_lc = KLc {
                    c0: eval_lc.c0.add_scaled(&Lc::from_var(weighted.c0), F::ONE),
                    c1: eval_lc.c1.add_scaled(&Lc::from_var(weighted.c1), F::ONE),
                };
            }
        }
    }
    let eval_sum = alloc_klc(builder, &eval_lc);

    // γ^{k_total} · eval_sum.
    let gamma_eval = enforce_k_mul(builder, &KLc::from_var(gamma_to_k), &KLc::from_var(eval_sum));

    // rhs_fe = eq_beta · F' + eq_ar · γ^{k_total} · eval_sum.
    let left = enforce_k_mul(builder, &KLc::from_var(eq_beta), &KLc::from_var(f_prime));
    let right = enforce_k_mul(builder, &KLc::from_var(eq_ar), &KLc::from_var(gamma_eval));
    let rhs = KLc {
        c0: Lc::from_var(left.c0).add_scaled(&Lc::from_var(right.c0), F::ONE),
        c1: Lc::from_var(left.c1).add_scaled(&Lc::from_var(right.c1), F::ONE),
    };
    Ok(alloc_klc(builder, &rhs))
}
