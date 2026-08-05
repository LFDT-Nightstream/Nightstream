//! Sumcheck verifier primitives for in-circuit Π_CCS.V (paper §7.3).
//!
//! Owns:
//! - [`gamma_powers`] — table of `[γ^0, γ^1, …, γ^{count-1}]` over 𝕂.
//! - [`horner_eval_k`] — evaluate a univariate K-polynomial at a K-point.
//! - [`enforce_sumcheck_round`] — one round's verifier check:
//!   `g_q(0) + g_q(1) == claim_q` plus returning `claim_{q+1} = g_q(r'_q)`.
//! - [`enforce_sumcheck_rounds_engine`] — engine-parity full walk that
//!   drives the transcript the same way `verify_sumcheck_rounds_poseidon_v3`
//!   does (raw absorbs and raw squeezes, no labels).
//! - [`enforce_eq_k`] — `eq(a, b) = Π_i (2 a_i b_i + 1 - a_i - b_i)` over 𝕂.
//!
//! Mechanical. No paper-level claims live here; the paper math is in
//! `paper/reductions/pi_ccs_circuit/` (the one-joint Π_CCS.V
//! verifier), which composes these primitives.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, SumcheckRoundAudit, Var};
use crate::engine::r1cs_circuit::field_ext::{alloc_klc, enforce_k_mul, KLc, KVar};
use crate::engine::r1cs_circuit::transcript::TranscriptGadget;

/// `[γ^0, γ^1, …, γ^{count-1}]` as K-vars. Uses `count - 1` K-mults.
///
/// `γ^0` is the constant `1` (allocated and equality-constrained).
pub fn gamma_powers(builder: &mut R1csBuilder, gamma: KVar, count: usize) -> Vec<KVar> {
    assert!(count > 0, "gamma_powers: count must be > 0");
    let mut out = Vec::with_capacity(count);
    out.push(alloc_klc(builder, &KLc::from_base_const(F::ONE)));
    for i in 1..count {
        let prev = out[i - 1];
        let next = enforce_k_mul(builder, &KLc::from_var(prev), &KLc::from_var(gamma));
        out.push(next);
    }
    out
}

/// Evaluate `coeffs[0] + coeffs[1]·r + … + coeffs[d]·r^d` via Horner.
///
/// `d = coeffs.len() - 1`. Uses `d` K-mults; returns the evaluation as a
/// fresh `KVar`.
pub fn horner_eval_k(builder: &mut R1csBuilder, coeffs: &[KVar], r: KVar) -> KVar {
    assert!(!coeffs.is_empty(), "horner_eval_k: empty coefficient list");
    let mut acc = coeffs[coeffs.len() - 1];
    for i in (0..coeffs.len() - 1).rev() {
        let mul = enforce_k_mul(builder, &KLc::from_var(acc), &KLc::from_var(r));
        let sum = KLc {
            c0: Lc::from_var(mul.c0).add_scaled(&Lc::from_var(coeffs[i].c0), F::ONE),
            c1: Lc::from_var(mul.c1).add_scaled(&Lc::from_var(coeffs[i].c1), F::ONE),
        };
        acc = alloc_klc(builder, &sum);
    }
    acc
}

/// Enforce one sumcheck round's verifier check.
///
/// Given the prover's round polynomial `g_q` (its `d_sc + 1` coefficients),
/// the verifier-sampled point `r'_q`, and the running claim `claim_q`, this
/// gadget:
/// 1. Enforces `g_q(0) + g_q(1) == claim_q` (one K-equality = two F-equalities).
/// 2. Allocates and returns `claim_{q+1} = g_q(r'_q)` via Horner.
///
/// `g_q(0) = coeffs[0]` and `g_q(1) = Σ_i coeffs[i]`, so the check is the
/// linear constraint `2·coeffs[0] + Σ_{i≥1} coeffs[i] == claim_q`.
pub fn enforce_sumcheck_round(builder: &mut R1csBuilder, coeffs: &[KVar], r_q: KVar, claim_in: KVar) -> KVar {
    assert!(!coeffs.is_empty(), "enforce_sumcheck_round: empty coefficient list");
    let row_start = builder.rows();
    let first_allocated_column = builder.cols();
    let two = F::from_u64(2);

    let mut sum_c0 = Lc::zero();
    let mut sum_c1 = Lc::zero();
    sum_c0.add_term(coeffs[0].c0, two);
    sum_c1.add_term(coeffs[0].c1, two);
    for c in &coeffs[1..] {
        sum_c0.add_term(c.c0, F::ONE);
        sum_c1.add_term(c.c1, F::ONE);
    }
    builder.enforce_eq(&Lc::from_var(claim_in.c0), &sum_c0);
    builder.enforce_eq(&Lc::from_var(claim_in.c1), &sum_c1);

    let claim_out = horner_eval_k(builder, coeffs, r_q);
    builder.record_sumcheck_round(SumcheckRoundAudit {
        row_start,
        row_end: builder.rows(),
        first_allocated_column,
        allocated_cols: (first_allocated_column..builder.cols()).collect(),
        coefficient_cols: coeffs
            .iter()
            .map(|coefficient| [coefficient.c0.col(), coefficient.c1.col()])
            .collect(),
        challenge_cols: [r_q.c0.col(), r_q.c1.col()],
        claim_in_cols: [claim_in.c0.col(), claim_in.c1.col()],
        claim_out_cols: [claim_out.c0.col(), claim_out.c1.col()],
    });
    claim_out
}

/// Walk a complete sumcheck verifier, threading the running claim through
/// `log m` rounds.
///
/// On exit, returns `v = g_{log m - 1}(r'_{log m - 1})`, the value Π_CCS.V's
/// terminal identity check compares against
/// `eq(r', α)·(F + γ^K·N) + γ^{2K+k}·E`.
pub fn enforce_sumcheck_walk(
    builder: &mut R1csBuilder,
    rounds: &[Vec<KVar>],
    challenges: &[KVar],
    initial_claim: KVar,
) -> KVar {
    assert_eq!(
        rounds.len(),
        challenges.len(),
        "sumcheck_walk: round/challenge length mismatch"
    );
    let mut claim = initial_claim;
    for (round_coeffs, r_q) in rounds.iter().zip(challenges.iter()) {
        claim = enforce_sumcheck_round(builder, round_coeffs, *r_q, claim);
    }
    claim
}

/// Engine-parity sumcheck walk: drives a [`TranscriptGadget`] in the exact
/// pattern of `neo_reductions::sumcheck::verify_sumcheck_rounds_poseidon_v3`.
///
/// For each round the gadget:
/// 1. Absorbs the round polynomial's `(d_sc + 1)` K-coefficients as raw
///    `(c0, c1)` lanes via [`TranscriptGadget::append_fields_raw_vars`].
/// 2. Squeezes a 2-lane raw challenge via
///    [`TranscriptGadget::challenge_fields_raw`] and packs it as a `KVar`.
/// 3. Enforces `g(0) + g(1) == running_sum` and advances
///    `running_sum := g(challenge)` via [`enforce_sumcheck_round`].
///
/// Returns `(challenges, final_running_sum)` — the same `(r_all, running_sum)`
/// the native verifier returns.
pub fn enforce_sumcheck_rounds_engine(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    initial_sum: KVar,
    rounds: &[Vec<KVar>],
) -> (Vec<KVar>, KVar) {
    let mut running_sum = initial_sum;
    let mut challenges = Vec::with_capacity(rounds.len());
    let mut packed: Vec<Var> = Vec::new();
    for round_coeffs in rounds {
        packed.clear();
        packed.reserve(round_coeffs.len() * 2);
        for k in round_coeffs {
            packed.push(k.c0);
            packed.push(k.c1);
        }
        transcript.append_fields_raw_vars(builder, &packed);
        let lanes = transcript.challenge_fields_raw(builder, 2);
        let challenge = KVar::new(lanes[0], lanes[1]);
        running_sum = enforce_sumcheck_round(builder, round_coeffs, challenge, running_sum);
        challenges.push(challenge);
    }
    (challenges, running_sum)
}

/// Compute `Σ_n γ_table[idx(n)] · k_value[n]` where `idx(n)` is the
/// caller-supplied γ-exponent for the `n`-th term.
///
/// Used by Π_CCS.V for both the initial sumcheck claim `T` and the terminal
/// `E_inner`: both have the same paper shape
/// `Σ_{h,j,ℓ} γ^{I(h,j,ℓ)} · cf(y)_ℓ`.
///
/// Per term: 1 K-mult to compute `γ^I · k_value`. Returns a `KVar`
/// constrained to the running sum.
pub fn enforce_gamma_indexed_sum(
    builder: &mut R1csBuilder,
    gamma_table: &[KVar],
    indices: &[usize],
    k_values: &[KVar],
) -> KVar {
    assert_eq!(
        indices.len(),
        k_values.len(),
        "gamma_indexed_sum: indices/values length mismatch"
    );
    assert!(!indices.is_empty(), "gamma_indexed_sum: empty input");
    let mut acc_lc = KLc::zero();
    for (idx, &v) in indices.iter().zip(k_values.iter()) {
        let gamma_pow = gamma_table[*idx];
        let term = enforce_k_mul(builder, &KLc::from_var(gamma_pow), &KLc::from_var(v));
        acc_lc = KLc {
            c0: acc_lc.c0.add_scaled(&Lc::from_var(term.c0), F::ONE),
            c1: acc_lc.c1.add_scaled(&Lc::from_var(term.c1), F::ONE),
        };
    }
    alloc_klc(builder, &acc_lc)
}

/// Per-instance R1CS `f` evaluation: `f_i = y0 · y1 - y2` for `f(X,Y,Z) = X·Y - Z`.
///
/// Inputs are the constant terms `ct(y'_{i,j})` (j ∈ [3]) of the prover's
/// output ring evaluations. Used to compute Π_CCS.V's terminal `F`.
pub fn enforce_r1cs_f_term(builder: &mut R1csBuilder, y0: KVar, y1: KVar, y2: KVar) -> KVar {
    let prod = enforce_k_mul(builder, &KLc::from_var(y0), &KLc::from_var(y1));
    let f_lc = KLc {
        c0: Lc::from_var(prod.c0).add_scaled(&Lc::from_var(y2.c0), -F::ONE),
        c1: Lc::from_var(prod.c1).add_scaled(&Lc::from_var(y2.c1), -F::ONE),
    };
    alloc_klc(builder, &f_lc)
}

/// Per-instance norm-check polynomial for `b = 2`:
///   `norm_i = (z + 1) · z · (z - 1)`.
///
/// `z = ct(y'_{i,1})`. Vanishes iff `z ∈ {-1, 0, 1}`, the centered low-norm
/// condition for `b = 2`.
pub fn enforce_norm_check_b2(builder: &mut R1csBuilder, z: KVar) -> KVar {
    let z_plus_one = KLc {
        c0: {
            let mut lc = Lc::from_var(z.c0);
            lc.constant = F::ONE;
            lc
        },
        c1: Lc::from_var(z.c1),
    };
    let step1 = enforce_k_mul(builder, &z_plus_one, &KLc::from_var(z));
    let z_minus_one = KLc {
        c0: {
            let mut lc = Lc::from_var(z.c0);
            lc.constant = -F::ONE;
            lc
        },
        c1: Lc::from_var(z.c1),
    };
    enforce_k_mul(builder, &KLc::from_var(step1), &z_minus_one)
}

/// Tabulate `χ_α(ρ)` for every `ρ ∈ {0,1}^ell_d` as a vector of `2^ell_d`
/// K-elements, where `ell_d = alpha.len()`.
///
/// `χ_α(ρ) = Π_{bit=0..ell_d} (bit_is_one(ρ, bit) ? α[bit] : 1 - α[bit])`.
///
/// Bit ordering matches `optimized_engine/common.rs:eq_points`/`χ_α`: bit 0 of
/// `ρ` is the LSB and pairs with `α[0]`. Built incrementally — each iteration
/// `k` doubles the table by multiplying old entries by `1 - α[k]` for the
/// new-bit-zero half and by `α[k]` for the new-bit-one half. Total K-mults:
/// `Σ_{k=1..ell_d} 2^k = 2^{ell_d+1} - 2`.
///
/// Empty `alpha` returns `[K::ONE]` (the empty-product convention).
pub fn enforce_chi_alpha(builder: &mut R1csBuilder, alpha: &[KVar]) -> Vec<KVar> {
    let mut chi = vec![alloc_klc(builder, &KLc::from_base_const(F::ONE))];
    for a in alpha {
        let one_minus_a = KLc {
            c0: {
                let mut lc = Lc::zero();
                lc.add_term(a.c0, -F::ONE);
                lc.constant = F::ONE;
                lc
            },
            c1: {
                let mut lc = Lc::zero();
                lc.add_term(a.c1, -F::ONE);
                lc
            },
        };
        let mut next = Vec::with_capacity(chi.len() * 2);
        // First half: bit_k = 0 — multiply by (1 - α[k]).
        for c in &chi {
            let term0 = enforce_k_mul(builder, &KLc::from_var(*c), &one_minus_a);
            next.push(term0);
        }
        // Second half: bit_k = 1 — multiply by α[k].
        for c in &chi {
            let term1 = enforce_k_mul(builder, &KLc::from_var(*c), &KLc::from_var(*a));
            next.push(term1);
        }
        chi = next;
    }
    chi
}

/// `eq(a, b) = Π_i (a_i · b_i + (1 - a_i)(1 - b_i)) = Π_i (2 a_i b_i + 1 - a_i - b_i)`.
///
/// Per element: 1 K-mult for `a_i · b_i`, 1 K-mult to fold into the accumulator
/// (except the first element which is just the term itself). Total: `2 ell - 1`
/// K-mults for length-`ell` vectors.
pub fn enforce_eq_k(builder: &mut R1csBuilder, a: &[KVar], b: &[KVar]) -> KVar {
    assert_eq!(a.len(), b.len(), "eq_k: a and b length mismatch");
    assert!(!a.is_empty(), "eq_k: empty input");

    let mut acc: Option<KVar> = None;
    for (av, bv) in a.iter().zip(b.iter()) {
        // ab = a · b
        let ab = enforce_k_mul(builder, &KLc::from_var(*av), &KLc::from_var(*bv));
        // term = 2·ab + 1 - a - b
        let term_lc = KLc {
            c0: {
                let mut lc = Lc::zero();
                lc.add_term(ab.c0, F::from_u64(2));
                lc.add_term(av.c0, -F::ONE);
                lc.add_term(bv.c0, -F::ONE);
                lc.constant = F::ONE;
                lc
            },
            c1: {
                let mut lc = Lc::zero();
                lc.add_term(ab.c1, F::from_u64(2));
                lc.add_term(av.c1, -F::ONE);
                lc.add_term(bv.c1, -F::ONE);
                lc
            },
        };
        let term_var = alloc_klc(builder, &term_lc);
        acc = Some(match acc {
            None => term_var,
            Some(prev) => enforce_k_mul(builder, &KLc::from_var(prev), &KLc::from_var(term_var)),
        });
    }
    acc.expect("eq_k: input was empty")
}

#[cfg(test)]
mod tests {
    use super::*;
    use neo_math::{from_complex, KExtensions, K};
    use neo_reductions::sumcheck::verify_sumcheck_rounds_poseidon_v3;
    use neo_transcript::{Poseidon2Transcript, Transcript as _};
    use p3_field::Field;

    const APP: &[u8] = b"neo.fold.clean.unit.sumcheck/v1";

    fn k(a: u64, b: u64) -> K {
        K::from_coeffs([F::from_u64(a), F::from_u64(b)])
    }

    fn alloc_k(builder: &mut R1csBuilder, value: K) -> KVar {
        let [c0, c1] = value.as_coeffs();
        KVar::alloc(builder, c0, c1)
    }

    fn read_k(builder: &R1csBuilder, value: KVar) -> K {
        K::from_coeffs([builder.witness()[value.c0.col()], builder.witness()[value.c1.col()]])
    }

    fn eval_poly(coeffs: &[K], r: K) -> K {
        let mut acc = K::ZERO;
        let mut pow = K::ONE;
        for coeff in coeffs {
            acc += *coeff * pow;
            pow *= r;
        }
        acc
    }

    #[test]
    fn horner_eval_matches_native_polynomial_value() {
        let coeffs = [k(3, 4), k(5, 6), k(7, 8), k(9, 10)];
        let r = k(11, 12);
        let expected = eval_poly(&coeffs, r);

        let mut builder = R1csBuilder::new();
        let coeff_vars = coeffs
            .iter()
            .copied()
            .map(|v| alloc_k(&mut builder, v))
            .collect::<Vec<_>>();
        let r_var = alloc_k(&mut builder, r);
        let out = horner_eval_k(&mut builder, &coeff_vars, r_var);

        assert_eq!(read_k(&builder, out), expected);
        assert!(
            builder.is_satisfied(),
            "Horner gadget unsatisfied (first bad row: {:?})",
            builder.first_unsatisfied_row()
        );
    }

    #[test]
    fn one_sumcheck_round_enforces_claim_and_returns_next_claim() {
        let claim = k(100, 7);
        let c1 = k(3, 1);
        let c2 = k(5, 2);
        let c0 = (claim - c1 - c2) * K::from_u64(2).inverse();
        let coeffs = [c0, c1, c2];
        let r = k(13, 21);
        let expected_next = eval_poly(&coeffs, r);

        let mut builder = R1csBuilder::new();
        let coeff_vars = coeffs
            .iter()
            .copied()
            .map(|v| alloc_k(&mut builder, v))
            .collect::<Vec<_>>();
        let r_var = alloc_k(&mut builder, r);
        let claim_var = alloc_k(&mut builder, claim);
        let next = enforce_sumcheck_round(&mut builder, &coeff_vars, r_var, claim_var);

        assert_eq!(read_k(&builder, next), expected_next);
        assert!(builder.is_satisfied(), "honest round should satisfy");

        builder.tamper_witness(coeff_vars[1].c0.col(), F::from_u64(999));
        assert!(
            !builder.is_satisfied(),
            "tampered round coefficient must break g(0)+g(1)=claim"
        );
    }

    #[test]
    fn engine_sumcheck_walk_matches_native_v3_transcript() {
        let initial_claim = k(55, 89);
        let mut native_tr = Poseidon2Transcript::new(APP);
        let mut rounds = Vec::<Vec<K>>::new();
        let mut challenges = Vec::<K>::new();
        let mut running = initial_claim;

        for i in 0..3u64 {
            let c1 = k(10 + i, 20 + i);
            let c2 = k(30 + i, 40 + i);
            let c0 = (running - c1 - c2) * K::from_u64(2).inverse();
            let coeffs = vec![c0, c1, c2];

            let packed = coeffs
                .iter()
                .flat_map(|c| c.as_coeffs())
                .collect::<Vec<_>>();
            native_tr.append_fields_raw(&packed);
            let pair = native_tr.challenge_fields_raw(2);
            let challenge = from_complex(pair[0], pair[1]);
            running = eval_poly(&coeffs, challenge);
            challenges.push(challenge);
            rounds.push(coeffs);
        }

        let mut verify_tr = Poseidon2Transcript::new(APP);
        let (native_challenges, native_final, ok) =
            verify_sumcheck_rounds_poseidon_v3(&mut verify_tr, 2, initial_claim, &rounds);
        assert!(ok, "native v3 verifier rejected self-consistent rounds");
        assert_eq!(native_challenges, challenges);
        assert_eq!(native_final, running);

        let mut builder = R1csBuilder::new();
        let initial_var = alloc_k(&mut builder, initial_claim);
        let round_vars = rounds
            .iter()
            .map(|round| {
                round
                    .iter()
                    .copied()
                    .map(|v| alloc_k(&mut builder, v))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut gadget_tr = TranscriptGadget::new(&mut builder, APP);
        let (challenge_vars, final_var) =
            enforce_sumcheck_rounds_engine(&mut builder, &mut gadget_tr, initial_var, &round_vars);

        for (wire, expected) in challenge_vars.into_iter().zip(native_challenges) {
            assert_eq!(read_k(&builder, wire), expected);
        }
        assert_eq!(read_k(&builder, final_var), native_final);
        assert!(
            builder.is_satisfied(),
            "engine sumcheck walk unsatisfied (first bad row: {:?})",
            builder.first_unsatisfied_row()
        );
    }
}
