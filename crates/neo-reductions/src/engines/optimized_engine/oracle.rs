//! Optimized RoundOracle for Q(X) evaluation in Π_CCS.
//!
//! This oracle uses factored algebra, precomputed terms, and cached sparse formats
//! to efficiently evaluate the Q polynomial during sumcheck rounds. Mathematically
//! equivalent to paper-exact but ~10x faster.
//!
//! Variable order (rounds): first the `ell_n` row bits, then the `ell_d` Ajtai bits.

#![allow(non_snake_case)]

mod nc;
mod optimized;

pub use nc::NcOracle;
pub use optimized::OptimizedOracle;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{Fq, KExtensions, D, K};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;
use std::sync::Arc;

use crate::sumcheck::RoundOracle;

use super::backend::{FeEvalTable, FeMcsRowTables, FeSumcheckBackend};
use super::common::Challenges;
use super::digit_table::{build_nc_digit_table_compact, NcDigitMasks, NcDigitTable};
use super::row_poly::{
    accumulate_factored_groups_times_affine, accumulate_factored_groups_times_affine_base, accumulate_fast_term,
    accumulate_fast_term_base, factor_common_linear_terms, CompiledPolyGroup, CompiledPolyTerm, CompiledPolyTermKind,
    RowTable,
};
pub use super::sparse::SparseCache;
pub use crate::superneo_eval::SuperneoRingLinearForm;
use crate::superneo_eval::{SuperneoEvalCache, SuperneoZBlocks};

/// Read-only view of `OptimizedOracle`'s row-phase state for accelerator
/// backends. Field meanings match the private `RowStreamState`; see
/// `OptimizedOracle::row_phase_snapshot`.
#[derive(Clone, Copy)]
pub struct RowTableSnapshot<'a> {
    pub real: &'a [Fq],
    pub imag: Option<&'a [Fq]>,
}

pub struct RowPhaseSnapshot<'a> {
    pub cur_len: usize,
    pub active_len: usize,
    /// Row-domain equality point whose χ table is `eq_beta_r_tbl`.
    pub beta_r: &'a [K],
    /// Optional carried-input row point whose χ table is `eq_r_inputs_tbl`.
    pub r_inputs: Option<&'a [K]>,
    /// Empty when the backend advertises challenge-native equality expansion.
    pub eq_beta_r_tbl: &'a [K],
    /// Present but empty when the optional equality table is backend-owned.
    pub eq_r_inputs_tbl: Option<&'a [K]>,
    pub eval_tbl: Option<&'a [K]>,
    /// True when the carried eval table is owned by the accelerator backend
    /// and intentionally absent from the CPU oracle.
    pub deferred_eval_tbl: bool,
    pub gamma_to_k: K,
    pub gamma_pow_mcs: &'a [K],
    pub zero_mcs: &'a [bool],
    /// True for non-zero MCS slots whose row tables are owned by the
    /// accelerator backend and intentionally absent from the CPU oracle.
    pub deferred_mcs: &'a [bool],
    pub f_at_zero: K,
    /// Canonical sumcheck degree bound used for transcript/proof encoding.
    /// The row-phase polynomial may have a smaller active degree; proofs
    /// still serialize coefficients at this width so transcripts remain
    /// byte-identical.
    pub sumcheck_degree_bound: usize,
    pub row_phase_deg_max: usize,
    pub f_var_count: usize,
    /// Per MCS slot, per f-variable: the row-domain table (empty for zero MCS).
    pub f_var_tables_by_mcs: Vec<Vec<RowTableSnapshot<'a>>>,
    /// Compiled f terms as `(coeff, [(var_pos, exponent)])`.
    pub f_terms: Vec<(K, Vec<(usize, u32)>)>,
}

/// Read-only view of `NcOracle`'s column-phase state for accelerator
/// backends; see `NcOracle::col_phase_snapshot`.
pub struct NcColSnapshot<'a> {
    pub cur_len: usize,
    /// Column-domain equality point whose χ table is `eq_beta_m_tbl`.
    pub beta_m: &'a [K],
    pub eq_beta_m_tbl: &'a [K],
    /// Per witness: `weights[rho] = γ^{i+1} · χ_{β_a}(rho)`.
    pub weights: &'a [[K; D]],
    pub digit_tables: Vec<NcDigitTableView<'a>>,
}

/// Borrowed view of one `NcDigitTable` representation.
pub enum NcDigitTableView<'a> {
    Zero {
        len: usize,
    },
    Lane0(&'a [K]),
    Strided {
        width: usize,
        values: &'a [K],
    },
    Dense(&'a [[K; D]]),
    /// The host deferred the build; only the length is known. A backend
    /// must source the values itself (resident planes) or decline.
    Deferred {
        len: usize,
    },
}

/// Row-phase streaming state (over the row/time hypercube).
///
/// This replaces the old `evals_row_phase` strategy of enumerating row tails and repeatedly
/// running `precompute_for_r`. Instead, we materialize row-domain tables once and fold them
/// in-place as row challenges arrive.
struct RowStreamState {
    /// Current table length = 2^(remaining row bits).
    cur_len: usize,
    /// Current support length inside the padded row-domain table.
    ///
    /// When `f(0, ..., 0) == 0`, rows beyond `s.n` are provably zero across
    /// the MCS and Eval terms. Folding preserves that with `ceil(len / 2)`, so
    /// the row-phase sumcheck only needs to scan this prefix. If compact
    /// all-zero MCS slots contribute `f(0, ..., 0)`, padded rows may be nonzero
    /// and this stays equal to `cur_len`.
    active_len: usize,

    /// χ_{β_r}(row) table over the padded row domain (len = cur_len).
    eq_beta_r_tbl: Vec<K>,
    /// Optional χ_{r_inputs}(row) table (len = cur_len) for Eval gating.
    eq_r_inputs_tbl: Option<Vec<K>>,

    /// γ^{i-1} weights for the MCS slots (i is 1-based).
    gamma_pow_mcs: Vec<K>,

    /// Per-MCS tables for the variables used by the CCS polynomial `f`.
    /// Each entry is a row-domain table of `m_j(row) = (M_j · z_i)[row]` at boolean row points.
    f_var_tables_by_mcs: Vec<Vec<RowTable>>,
    /// True when the corresponding MCS witness is all zero and its row tables are omitted.
    zero_mcs: Vec<bool>,
    /// True when the corresponding MCS row tables are intentionally held by
    /// the FE backend instead of this CPU oracle.
    deferred_mcs: Vec<bool>,
    /// Number of row-table variables used by `f`.
    f_var_count: usize,
    /// Compiled sparse polynomial terms for `f` using `f_var_tables_by_mcs[i]` indices.
    f_terms: Vec<CompiledPolyTerm>,
    /// Disjoint common-linear-factor groups used by the base-field row hot path.
    f_factored_groups: Vec<CompiledPolyGroup>,
    f_factored_terms: usize,
    /// Exact value of `f(0, ..., 0)` for compact all-zero MCS slots.
    f_at_zero: K,
    /// Maximum univariate degree needed for row-phase sumcheck coefficients.
    row_phase_deg_max: usize,

    /// Combined Eval block table over rows (already summed over α' and (i,j) coefficients).
    /// When present, Eval contribution is: `eq_r_inputs(r') * gamma_to_k * eval_tbl(r')`.
    eval_tbl: Option<Vec<K>>,
    /// The carried eval table is backend-owned and intentionally absent.
    deferred_eval_tbl: bool,
    gamma_to_k: K,

    b: u32,
    /// True if all streamed tables are still in the base-field embedding (imag=0).
    ///
    /// When this holds and evaluation points are also base-field, we can evaluate the hot
    /// row-phase logic entirely in `Fq` for a large speedup.
    all_base: bool,
    /// Whether row-phase tables were built through SuperNeo cached rows.
    use_superneo_rows: bool,
}

impl RowStreamState {
    fn build<Ff>(
        s: &CcsStructure<Ff>,
        b: u32,
        ch: &Challenges,
        ell_d: usize,
        ell_n: usize,
        mcs_witnesses: &[CcsWitness<Ff>],
        me_witnesses: &[Mat<Ff>],
        r_inputs: Option<&[K]>,
        _sparse: &SparseCache<Ff>,
        superneo_cache: &SuperneoEvalCache,
        witness_z_blocks: &[SuperneoZBlocks],
        mut fe_backend: Option<&mut (dyn FeSumcheckBackend + '_)>,
    ) -> Self
    where
        Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
        K: From<Ff>,
    {
        let n_pad = 1usize << ell_n;
        let n_eff = s.n;
        let t_mats = s.t();

        #[cfg(feature = "perf-timers")]
        let t_total = std::time::Instant::now();

        #[cfg(feature = "perf-timers")]
        let t_chi = std::time::Instant::now();
        let defer_row_equality_tables = fe_backend
            .as_ref()
            .is_some_and(|backend| backend.defers_row_equality_tables());
        let eq_beta_r_tbl = maybe_chi_tail_weights(&ch.beta_r, defer_row_equality_tables);

        let eval_inputs_present = r_inputs.is_some();
        let mut eq_r_inputs_tbl = None;
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "RowStreamState::build: 1. chi tables                {:.2?} @{}",
            t_chi.elapsed(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );

        let all_base = ch.gamma.imag() == Fq::ZERO
            && ch.alpha.iter().all(|x| x.imag() == Fq::ZERO)
            && ch.beta_a.iter().all(|x| x.imag() == Fq::ZERO)
            && ch.beta_r.iter().all(|x| x.imag() == Fq::ZERO)
            && r_inputs
                .map(|r| r.iter().all(|x| x.imag() == Fq::ZERO))
                .unwrap_or(true);

        #[cfg(feature = "perf-timers")]
        let t_f_compile = std::time::Instant::now();
        // Compile CCS polynomial f to avoid scanning t variables per evaluation.
        if s.f.arity() != t_mats {
            panic!(
                "CCS polynomial arity mismatch: f.arity()={}, but s.t()={}",
                s.f.arity(),
                t_mats
            );
        }
        let mut used_vars = vec![false; t_mats];
        for term in s.f.terms() {
            if term.exps.len() != t_mats {
                panic!(
                    "CCS polynomial exponent vector length mismatch: got {}, expected {}",
                    term.exps.len(),
                    t_mats
                );
            }
            for (j, &exp) in term.exps.iter().enumerate() {
                if exp != 0 {
                    used_vars[j] = true;
                }
            }
        }
        let f_var_indices: Vec<usize> = used_vars
            .iter()
            .enumerate()
            .filter_map(|(j, &u)| u.then_some(j))
            .collect();

        let mut pos_by_j = vec![usize::MAX; t_mats];
        for (pos, &j) in f_var_indices.iter().enumerate() {
            pos_by_j[j] = pos;
        }

        let f_terms: Vec<CompiledPolyTerm> =
            s.f.terms()
                .iter()
                .map(|term| {
                    let mut vars = Vec::new();
                    for (j, &exp) in term.exps.iter().enumerate() {
                        if exp != 0 {
                            let pos = pos_by_j[j];
                            debug_assert_ne!(pos, usize::MAX, "missing f var mapping");
                            vars.push((pos, exp));
                        }
                    }
                    let kind = CompiledPolyTermKind::from_vars(&vars);
                    CompiledPolyTerm {
                        coeff: K::from(term.coeff),
                        vars,
                        kind,
                    }
                })
                .collect();
        let f_at_zero = f_terms
            .iter()
            .filter(|term| term.vars.is_empty())
            .fold(K::ZERO, |acc, term| acc + term.coeff);
        let (f_factored_groups, f_factored_terms) = factor_common_linear_terms(&f_terms);
        let f_max_term_deg = f_terms
            .iter()
            .map(|term| {
                term.vars
                    .iter()
                    .map(|&(_, exp)| exp as usize)
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0);
        // eq_beta_r(X) adds one degree; Eval block is quadratic.
        let row_phase_deg_max = core::cmp::max(2, f_max_term_deg + 1);
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "RowStreamState::build: 2. f compile / f_var_indices {:.2?} (used_vars={}, terms={}, factored={})",
            t_f_compile.elapsed(),
            f_var_indices.len(),
            f_terms.len(),
            f_factored_terms,
        );

        let k_mcs = mcs_witnesses.len();

        let k_total = k_mcs + me_witnesses.len();
        debug_assert_eq!(k_mcs + me_witnesses.len(), k_total);
        debug_assert_eq!(
            witness_z_blocks.len(),
            k_total,
            "RowStreamState::build: witness block cache length mismatch"
        );

        // Sanity: challenge vectors for Ajtai rounds must match ell_d.
        if ch.beta_a.len() != ell_d || ch.alpha.len() != ell_d {
            panic!(
                "Challenge length mismatch: alpha.len()={}, beta_a.len()={}, ell_d={ell_d}",
                ch.alpha.len(),
                ch.beta_a.len()
            );
        }
        let mut gamma_pow_mcs = vec![K::ONE; k_mcs];
        for i in 1..k_mcs {
            gamma_pow_mcs[i] = gamma_pow_mcs[i - 1] * ch.gamma;
        }

        // Optimized oracle now uses one canonical SuperNeo row-lifted path.
        let use_superneo_rows = true;

        #[cfg(feature = "perf-timers")]
        let t_f_var_tables = std::time::Instant::now();
        // f-var tables: m_j(row) = (M_j * z_i)[row] for each used variable and each MCS slot.
        let mut f_var_tables_by_mcs: Vec<Vec<RowTable>> = Vec::with_capacity(k_mcs);
        let mut zero_mcs = Vec::with_capacity(k_mcs);
        let mut deferred_mcs = Vec::with_capacity(k_mcs);
        for mcs_idx in 0..k_mcs {
            let z_blocks = &witness_z_blocks[mcs_idx];
            if z_blocks.all_zero() {
                zero_mcs.push(true);
                deferred_mcs.push(false);
                f_var_tables_by_mcs.push(Vec::new());
                continue;
            }
            zero_mcs.push(false);
            if let Some(tables) = fe_backend
                .as_mut()
                .and_then(|b| b.mcs_row_tables(superneo_cache, mcs_idx, &f_var_indices, z_blocks, n_eff, n_pad))
            {
                match tables {
                    FeMcsRowTables::Host(tables) => {
                        deferred_mcs.push(false);
                        f_var_tables_by_mcs.push(tables.into_iter().map(RowTable::from_extension).collect());
                    }
                    FeMcsRowTables::Deferred => {
                        deferred_mcs.push(true);
                        f_var_tables_by_mcs.push(Vec::new());
                    }
                }
                continue;
            }
            deferred_mcs.push(false);
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            let f_tables_i: Vec<RowTable> = f_var_indices
                .par_iter()
                .map(|&j| {
                    let mut out = vec![Fq::ZERO; n_pad];
                    let mat_cache = superneo_cache
                        .matrix(j)
                        .unwrap_or_else(|| panic!("superneo cache missing matrix j={j}"));
                    mat_cache.fill_row_dots_base_with_blocks(&mut out[..n_eff], z_blocks);
                    RowTable::from_base(out)
                })
                .collect();
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            let f_tables_i: Vec<RowTable> = f_var_indices
                .iter()
                .map(|&j| {
                    let mut out = vec![Fq::ZERO; n_pad];
                    let mat_cache = superneo_cache
                        .matrix(j)
                        .unwrap_or_else(|| panic!("superneo cache missing matrix j={j}"));
                    mat_cache.fill_row_dots_base_with_blocks(&mut out[..n_eff], z_blocks);
                    RowTable::from_base(out)
                })
                .collect();
            f_var_tables_by_mcs.push(f_tables_i);
        }
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "RowStreamState::build: 4. f_var_tables_by_mcs       {:.2?} (k_mcs={k_mcs}, vars={}, n_eff={n_eff})",
            t_f_var_tables.elapsed(),
            f_var_indices.len()
        );

        // Eval table (optional): only when both (a) there are carried witnesses, and (b) r_inputs exist.
        let mut gamma_to_k = K::ONE;
        for _ in 0..k_total {
            gamma_to_k *= ch.gamma;
        }

        let mut deferred_eval_tbl = false;
        let eval_tbl = if k_total > k_mcs && eval_inputs_present {
            let mut gamma_pow_i = vec![K::ONE; k_total];
            for i in 1..k_total {
                gamma_pow_i[i] = gamma_pow_i[i - 1] * ch.gamma;
            }
            let carried_coeffs: Vec<K> = gamma_pow_i[k_mcs..k_total].to_vec();

            let r_inputs = r_inputs.expect("r_inputs checked above");
            let mut w_alpha = [K::ZERO; D];
            for (rho, slot) in w_alpha.iter_mut().enumerate() {
                *slot = eq_points_bool_mask(rho, &ch.alpha);
            }
            let mut gamma_k_pow_j = vec![K::ONE; t_mats];
            for j in 1..t_mats {
                gamma_k_pow_j[j] = gamma_k_pow_j[j - 1] * gamma_to_k;
            }

            // Backend-resident path first: the backend owns the carried
            // combination (the running witnesses' host z blocks were never
            // built when it serves this). Field sums are exact, so values
            // are identical to the host combination in any order.
            #[cfg(feature = "perf-timers")]
            let t_eval = std::time::Instant::now();
            let backend_tbl = fe_backend.as_mut().and_then(|b| {
                b.carried_eval_table(
                    superneo_cache,
                    &carried_coeffs,
                    k_mcs,
                    &w_alpha,
                    &gamma_k_pow_j,
                    n_eff,
                    n_pad,
                )
            });
            if let Some(eval_tbl) = backend_tbl {
                eq_r_inputs_tbl = Some(maybe_chi_tail_weights(r_inputs, defer_row_equality_tables));
                #[cfg(feature = "perf-timers")]
                eprintln!(
                    "RowStreamState::build: 6. eval_tbl (backend carried) {:.2?} (carried={}, t_mats={t_mats}, n_eff={n_eff})",
                    t_eval.elapsed(),
                    k_total - k_mcs
                );
                match eval_tbl {
                    FeEvalTable::Host(eval_tbl) => Some(eval_tbl),
                    FeEvalTable::Deferred => {
                        deferred_eval_tbl = true;
                        None
                    }
                }
            } else {
                let carried_z_blocks =
                    SuperneoZBlocks::linear_combination_real(&witness_z_blocks[k_mcs..k_total], &carried_coeffs);

                if carried_z_blocks.all_zero() {
                    #[cfg(feature = "perf-timers")]
                    eprintln!("RowStreamState::build: 6. eval_tbl skipped       (carried linear combination is zero)");
                    None
                } else {
                    eq_r_inputs_tbl = Some(maybe_chi_tail_weights(r_inputs, defer_row_equality_tables));
                    let eval_tbl = fe_backend
                        .as_mut()
                        .and_then(|b| {
                            b.eval_weighted_row_table(
                                superneo_cache,
                                &carried_z_blocks,
                                &w_alpha,
                                &gamma_k_pow_j,
                                n_eff,
                                n_pad,
                            )
                        })
                        .unwrap_or_else(|| {
                            superneo_cache.eval_weighted_row_table(
                                &carried_z_blocks,
                                &w_alpha,
                                &gamma_k_pow_j,
                                n_eff,
                                n_pad,
                            )
                        });
                    #[cfg(feature = "perf-timers")]
                    eprintln!(
                    "RowStreamState::build: 6. eval_tbl loop             {:.2?} (carried={}, t_mats={t_mats}, n_eff={n_eff})",
                    t_eval.elapsed(),
                    k_total - k_mcs
                );

                    Some(eval_tbl)
                }
            }
        } else {
            #[cfg(feature = "perf-timers")]
            eprintln!(
                "RowStreamState::build: 5+6. eval_tbl skipped       (k_total={k_total}, k_mcs={k_mcs}, r_inputs={})",
                eval_inputs_present
            );
            None
        };

        #[cfg(feature = "perf-timers")]
        eprintln!(
            "RowStreamState::build: TOTAL                       {:.2?}",
            t_total.elapsed()
        );

        let active_len = if f_at_zero == K::ZERO { n_eff.max(1) } else { n_pad };

        Self {
            cur_len: n_pad,
            active_len,
            eq_beta_r_tbl,
            eq_r_inputs_tbl,
            gamma_pow_mcs,
            f_var_tables_by_mcs,
            zero_mcs,
            deferred_mcs,
            f_var_count: f_var_indices.len(),
            f_terms,
            f_factored_groups,
            f_factored_terms,
            f_at_zero,
            row_phase_deg_max,
            eval_tbl,
            deferred_eval_tbl,
            gamma_to_k,
            b,
            all_base,
            use_superneo_rows,
        }
    }

    #[inline]
    fn fold_table_inplace(table: &mut Vec<K>, r: K) {
        debug_assert!(table.len() >= 2 && table.len() % 2 == 0);
        let half = table.len() / 2;
        for i in 0..half {
            let lo = table[2 * i];
            let hi = table[2 * i + 1];
            table[i] = lo + (hi - lo) * r;
        }
        table.truncate(half);
    }

    #[inline]
    fn fold_table_inplace_base(table: &mut Vec<K>, r: Fq) {
        debug_assert!(table.len() >= 2 && table.len() % 2 == 0);
        let half = table.len() / 2;
        for i in 0..half {
            let lo = table[2 * i].real();
            let hi = table[2 * i + 1].real();
            table[i] = K::from(lo + (hi - lo) * r);
        }
        table.truncate(half);
    }

    fn fold_inplace(&mut self, r: K) {
        if self.all_base && r.imag() == Fq::ZERO {
            let r0 = r.real();
            Self::fold_table_inplace_base(&mut self.eq_beta_r_tbl, r0);
            if let Some(tbl) = self.eq_r_inputs_tbl.as_mut() {
                Self::fold_table_inplace_base(tbl, r0);
            }
            for per_mcs in self.f_var_tables_by_mcs.iter_mut() {
                for tbl in per_mcs.iter_mut() {
                    tbl.fold_inplace(K::from(r0));
                }
            }
            if let Some(tbl) = self.eval_tbl.as_mut() {
                Self::fold_table_inplace_base(tbl, r0);
            }
        } else {
            self.all_base = false;
            Self::fold_table_inplace(&mut self.eq_beta_r_tbl, r);
            if let Some(tbl) = self.eq_r_inputs_tbl.as_mut() {
                Self::fold_table_inplace(tbl, r);
            }
            for per_mcs in self.f_var_tables_by_mcs.iter_mut() {
                for tbl in per_mcs.iter_mut() {
                    tbl.fold_inplace(r);
                }
            }
            if let Some(tbl) = self.eval_tbl.as_mut() {
                Self::fold_table_inplace(tbl, r);
            }
        }
        self.active_len = self.active_len.div_ceil(2).max(1);
        self.cur_len /= 2;
    }

    fn release_finalized_tables(&mut self) {
        debug_assert_eq!(self.cur_len, 1, "row tables may be released only after row folding");
        self.eq_beta_r_tbl = Vec::new();
        self.eq_r_inputs_tbl = None;
        self.f_var_tables_by_mcs.clear();
        self.f_var_tables_by_mcs.shrink_to_fit();
        self.eval_tbl = None;
    }

    fn materialize_deferred_equality_tables(&mut self, beta_r: &[K], r_inputs: Option<&[K]>) {
        if self.eq_beta_r_tbl.is_empty() {
            self.eq_beta_r_tbl = chi_tail_weights(beta_r);
        }
        if self.eq_r_inputs_tbl.as_ref().is_some_and(Vec::is_empty) {
            self.eq_r_inputs_tbl = r_inputs.map(chi_tail_weights);
        }
    }

    #[inline]
    fn poly_mul_affine_inplace_base(poly: &mut [Fq], a: Fq, b: Fq, current_deg: usize) {
        // Coeffs are low→high. Output truncates to input length:
        // new[0] = a*old[0]; new[d] = a*old[d] + b*old[d-1] (d>=1).
        let mut prev = Fq::ZERO;
        for coeff in poly.iter_mut().take(current_deg + 2) {
            let old = *coeff;
            *coeff = a * old + b * prev;
            prev = old;
        }
    }

    #[inline]
    fn poly_eval_base(coeffs: &[Fq], x: Fq) -> Fq {
        if coeffs.is_empty() {
            return Fq::ZERO;
        }
        let mut result = coeffs[coeffs.len() - 1];
        for &c in coeffs.iter().rev().skip(1) {
            result = result * x + c;
        }
        result
    }

    #[inline]
    fn accumulate_weighted_f_poly_base(&self, idx: usize, deg_max: usize, inner: &mut [Fq], term_poly: &mut [Fq]) {
        inner.fill(Fq::ZERO);

        for (mcs_idx, per_mcs_tables) in self.f_var_tables_by_mcs.iter().enumerate() {
            let g = self
                .gamma_pow_mcs
                .get(mcs_idx)
                .copied()
                .unwrap_or(K::ONE)
                .real();
            if g == Fq::ZERO {
                continue;
            }
            if self.zero_mcs[mcs_idx] {
                inner[0] += self.f_at_zero.real() * g;
                continue;
            }

            for term in &self.f_terms {
                let coeff = term.coeff.real() * g;
                if accumulate_fast_term_base(&term.kind, per_mcs_tables, idx, deg_max, inner, coeff) {
                    continue;
                }
                term_poly.fill(Fq::ZERO);
                term_poly[0] = coeff;
                let mut current_deg = 0usize;
                for &(var_pos, exp) in &term.vars {
                    let tbl = &per_mcs_tables[var_pos];
                    let a = tbl.real(idx);
                    let b = tbl.real(idx + 1) - a;
                    for _ in 0..exp {
                        Self::poly_mul_affine_inplace_base(term_poly, a, b, current_deg);
                        current_deg += 1;
                    }
                }
                for i in 0..=core::cmp::min(current_deg, deg_max) {
                    inner[i] += term_poly[i];
                }
            }
        }
    }

    #[inline]
    fn accumulate_weighted_f_times_affine_base(
        &self,
        idx: usize,
        deg_max: usize,
        outer_a: Fq,
        outer_b: Fq,
        out: &mut [Fq],
        inner: &mut [Fq],
        scratch: &mut [Fq],
    ) {
        if self.f_factored_terms == self.f_terms.len() && !self.f_factored_groups.is_empty() {
            for (mcs_idx, per_mcs_tables) in self.f_var_tables_by_mcs.iter().enumerate() {
                if self.zero_mcs[mcs_idx] {
                    continue;
                }
                let scale = self
                    .gamma_pow_mcs
                    .get(mcs_idx)
                    .copied()
                    .unwrap_or(K::ONE)
                    .real();
                if scale != Fq::ZERO {
                    accumulate_factored_groups_times_affine_base(
                        &self.f_factored_groups,
                        per_mcs_tables,
                        idx,
                        deg_max,
                        outer_a,
                        outer_b,
                        scale,
                        out,
                        scratch,
                    );
                }
            }
            return;
        }

        self.accumulate_weighted_f_poly_base(idx, deg_max, inner, scratch);
        out[0] += outer_a * inner[0];
        for degree in 1..=deg_max {
            out[degree] += outer_a * inner[degree] + outer_b * inner[degree - 1];
        }
    }

    #[inline]
    fn accumulate_weighted_f_poly(&self, idx: usize, deg_max: usize, inner: &mut [K], term_poly: &mut [K]) {
        inner.fill(K::ZERO);

        for (mcs_idx, per_mcs_tables) in self.f_var_tables_by_mcs.iter().enumerate() {
            let g = self.gamma_pow_mcs.get(mcs_idx).copied().unwrap_or(K::ONE);
            if g == K::ZERO {
                continue;
            }
            if self.zero_mcs[mcs_idx] {
                inner[0] += self.f_at_zero * g;
                continue;
            }

            for term in &self.f_terms {
                let coeff = term.coeff * g;
                if accumulate_fast_term(&term.kind, per_mcs_tables, idx, deg_max, inner, coeff) {
                    continue;
                }
                term_poly.fill(K::ZERO);
                term_poly[0] = coeff;
                let mut current_deg = 0usize;
                for &(var_pos, exp) in &term.vars {
                    let tbl = &per_mcs_tables[var_pos];
                    let a = tbl.get(idx);
                    let b = tbl.get(idx + 1) - a;
                    for _ in 0..exp {
                        Self::poly_mul_affine_inplace(term_poly, a, b, current_deg);
                        current_deg += 1;
                    }
                }
                for i in 0..=core::cmp::min(current_deg, deg_max) {
                    inner[i] += term_poly[i];
                }
            }
        }
    }

    #[inline]
    fn accumulate_weighted_f_times_affine(
        &self,
        idx: usize,
        deg_max: usize,
        outer_a: K,
        outer_b: K,
        out: &mut [K],
        inner: &mut [K],
        scratch: &mut [K],
    ) {
        if self.f_factored_terms == self.f_terms.len() && !self.f_factored_groups.is_empty() {
            for (mcs_idx, per_mcs_tables) in self.f_var_tables_by_mcs.iter().enumerate() {
                if self.zero_mcs[mcs_idx] {
                    continue;
                }
                let scale = self.gamma_pow_mcs.get(mcs_idx).copied().unwrap_or(K::ONE);
                if scale != K::ZERO {
                    accumulate_factored_groups_times_affine(
                        &self.f_factored_groups,
                        per_mcs_tables,
                        idx,
                        deg_max,
                        outer_a,
                        outer_b,
                        scale,
                        out,
                        scratch,
                    );
                }
            }
            return;
        }

        self.accumulate_weighted_f_poly(idx, deg_max, inner, scratch);
        out[0] += outer_a * inner[0];
        for degree in 1..=deg_max {
            out[degree] += outer_a * inner[degree] + outer_b * inner[degree - 1];
        }
    }

    fn evals_row_phase_b2_base(&self, tail_len: usize, xs: &[K]) -> Vec<K> {
        let deg_max = self.row_phase_deg_max;

        const PAR_THRESHOLD: usize = 1 << 14;
        let coeffs_seq = |tail_len: usize| -> Vec<Fq> {
            let mut coeffs = vec![Fq::ZERO; deg_max + 1];
            let mut inner = vec![Fq::ZERO; deg_max + 1];
            let mut term_poly = vec![Fq::ZERO; deg_max + 1];

            for t in 0..tail_len {
                let idx = 2 * t;
                let e0 = self.eq_beta_r_tbl[idx].real();
                let e1 = self.eq_beta_r_tbl[idx + 1].real() - e0;

                self.accumulate_weighted_f_times_affine_base(
                    idx,
                    deg_max,
                    e0,
                    e1,
                    &mut coeffs,
                    &mut inner,
                    &mut term_poly,
                );

                if let (Some(eq_tbl), Some(eval_tbl)) = (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref()) {
                    let r0 = eq_tbl[idx].real();
                    let r1 = eq_tbl[idx + 1].real() - r0;
                    let v0 = eval_tbl[idx].real();
                    let v1 = eval_tbl[idx + 1].real() - v0;

                    let g = self.gamma_to_k.real();
                    coeffs[0] += g * (r0 * v0);
                    coeffs[1] += g * (r0 * v1 + r1 * v0);
                    coeffs[2] += g * (r1 * v1);
                }
            }

            coeffs
        };

        let coeffs = if tail_len >= PAR_THRESHOLD {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                (0..tail_len)
                    .into_par_iter()
                    .fold(
                        || {
                            (
                                vec![Fq::ZERO; deg_max + 1],
                                vec![Fq::ZERO; deg_max + 1],
                                vec![Fq::ZERO; deg_max + 1],
                            )
                        },
                        |(mut coeffs, mut inner, mut term_poly), t| {
                            let idx = 2 * t;
                            // eq_beta_r(X) = e0 + e1·X
                            let e0 = self.eq_beta_r_tbl[idx].real();
                            let e1 = self.eq_beta_r_tbl[idx + 1].real() - e0;

                            self.accumulate_weighted_f_times_affine_base(
                                idx,
                                deg_max,
                                e0,
                                e1,
                                &mut coeffs,
                                &mut inner,
                                &mut term_poly,
                            );

                            // Eval: eq_r_inputs(X) * gamma_to_k * eval_tbl(X) (quadratic).
                            if let (Some(eq_tbl), Some(eval_tbl)) =
                                (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref())
                            {
                                let r0 = eq_tbl[idx].real();
                                let r1 = eq_tbl[idx + 1].real() - r0;
                                let v0 = eval_tbl[idx].real();
                                let v1 = eval_tbl[idx + 1].real() - v0;

                                let g = self.gamma_to_k.real();
                                coeffs[0] += g * (r0 * v0);
                                coeffs[1] += g * (r0 * v1 + r1 * v0);
                                coeffs[2] += g * (r1 * v1);
                            }

                            (coeffs, inner, term_poly)
                        },
                    )
                    .map(|(coeffs, _, _)| coeffs)
                    .reduce(
                        || vec![Fq::ZERO; deg_max + 1],
                        |mut a, b| {
                            for i in 0..=deg_max {
                                a[i] += b[i];
                            }
                            a
                        },
                    )
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                coeffs_seq(tail_len)
            }
        } else {
            coeffs_seq(tail_len)
        };

        xs.iter()
            .map(|&x| K::from(Self::poly_eval_base(&coeffs, x.real())))
            .collect()
    }

    fn evals_row_phase_b3_base(&self, tail_len: usize, xs: &[K]) -> Vec<K> {
        let deg_max = self.row_phase_deg_max;

        const PAR_THRESHOLD: usize = 1 << 14;
        let coeffs_seq = |tail_len: usize| -> Vec<Fq> {
            let mut coeffs = vec![Fq::ZERO; deg_max + 1];
            let mut inner = vec![Fq::ZERO; deg_max + 1];
            let mut term_poly = vec![Fq::ZERO; deg_max + 1];

            for t in 0..tail_len {
                let idx = 2 * t;
                let e0 = self.eq_beta_r_tbl[idx].real();
                let e1 = self.eq_beta_r_tbl[idx + 1].real() - e0;

                self.accumulate_weighted_f_times_affine_base(
                    idx,
                    deg_max,
                    e0,
                    e1,
                    &mut coeffs,
                    &mut inner,
                    &mut term_poly,
                );

                if let (Some(eq_tbl), Some(eval_tbl)) = (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref()) {
                    let r0 = eq_tbl[idx].real();
                    let r1 = eq_tbl[idx + 1].real() - r0;
                    let v0 = eval_tbl[idx].real();
                    let v1 = eval_tbl[idx + 1].real() - v0;

                    let g = self.gamma_to_k.real();
                    coeffs[0] += g * (r0 * v0);
                    coeffs[1] += g * (r0 * v1 + r1 * v0);
                    coeffs[2] += g * (r1 * v1);
                }
            }

            coeffs
        };

        let coeffs = if tail_len >= PAR_THRESHOLD {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                (0..tail_len)
                    .into_par_iter()
                    .fold(
                        || {
                            (
                                vec![Fq::ZERO; deg_max + 1],
                                vec![Fq::ZERO; deg_max + 1],
                                vec![Fq::ZERO; deg_max + 1],
                            )
                        },
                        |(mut coeffs, mut inner, mut term_poly), t| {
                            let idx = 2 * t;
                            // eq_beta_r(X) = e0 + e1·X
                            let e0 = self.eq_beta_r_tbl[idx].real();
                            let e1 = self.eq_beta_r_tbl[idx + 1].real() - e0;

                            self.accumulate_weighted_f_times_affine_base(
                                idx,
                                deg_max,
                                e0,
                                e1,
                                &mut coeffs,
                                &mut inner,
                                &mut term_poly,
                            );

                            // Eval: eq_r_inputs(X) * gamma_to_k * eval_tbl(X) (quadratic).
                            if let (Some(eq_tbl), Some(eval_tbl)) =
                                (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref())
                            {
                                let r0 = eq_tbl[idx].real();
                                let r1 = eq_tbl[idx + 1].real() - r0;
                                let v0 = eval_tbl[idx].real();
                                let v1 = eval_tbl[idx + 1].real() - v0;

                                let g = self.gamma_to_k.real();
                                coeffs[0] += g * (r0 * v0);
                                coeffs[1] += g * (r0 * v1 + r1 * v0);
                                coeffs[2] += g * (r1 * v1);
                            }

                            (coeffs, inner, term_poly)
                        },
                    )
                    .map(|(coeffs, _, _)| coeffs)
                    .reduce(
                        || vec![Fq::ZERO; deg_max + 1],
                        |mut a, b| {
                            for i in 0..=deg_max {
                                a[i] += b[i];
                            }
                            a
                        },
                    )
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                coeffs_seq(tail_len)
            }
        } else {
            coeffs_seq(tail_len)
        };

        xs.iter()
            .map(|&x| K::from(Self::poly_eval_base(&coeffs, x.real())))
            .collect()
    }

    /// Multiply a polynomial by an affine `(a + b·x)` in-place.
    ///
    /// Coefficients are in low→high order. Output is truncated to the input length.
    #[inline]
    fn poly_mul_affine_inplace(poly: &mut [K], a: K, b: K, current_deg: usize) {
        let mut prev = K::ZERO;
        for coeff in poly.iter_mut().take(current_deg + 2) {
            let old = *coeff;
            *coeff = a * old + b * prev;
            prev = old;
        }
    }

    fn evals_row_phase_impl<Ff>(&self, xs: &[K], allow_base: bool) -> Vec<K>
    where
        Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
        K: From<Ff>,
    {
        debug_assert!(self.cur_len >= 2 && self.cur_len % 2 == 0);
        let tail_len = self.active_len.div_ceil(2);
        debug_assert!(tail_len <= self.cur_len / 2);
        let xs_are_base = xs.iter().all(|&x| x.imag() == Fq::ZERO);
        let xs_all_base = allow_base && self.all_base && xs_are_base;

        // Fast path for b=2: build the univariate coefficients once per round,
        // then evaluate cheaply at all requested points.
        if self.b == 2 {
            if xs_all_base {
                return self.evals_row_phase_b2_base(tail_len, xs);
            }

            let deg_max = self.row_phase_deg_max;

            // Sequential per-`t` step, factored out so seq and par paths share one body.
            let step = |coeffs: &mut [K], inner: &mut [K], term_poly: &mut [K], t: usize| {
                let e0 = self.eq_beta_r_tbl[2 * t];
                let e1 = self.eq_beta_r_tbl[2 * t + 1] - e0;

                self.accumulate_weighted_f_times_affine(2 * t, deg_max, e0, e1, coeffs, inner, term_poly);

                // Eval: eq_r_inputs(X) * gamma_to_k * eval_tbl(X) (quadratic).
                if let (Some(eq_tbl), Some(eval_tbl)) = (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref()) {
                    let r0 = eq_tbl[2 * t];
                    let r1 = eq_tbl[2 * t + 1] - r0;
                    let v0 = eval_tbl[2 * t];
                    let v1 = eval_tbl[2 * t + 1] - v0;

                    let g = self.gamma_to_k;
                    coeffs[0] += g * (r0 * v0);
                    if deg_max >= 1 {
                        coeffs[1] += g * (r0 * v1 + r1 * v0);
                    }
                    if deg_max >= 2 {
                        coeffs[2] += g * (r1 * v1);
                    }
                }
            };

            const PAR_THRESHOLD: usize = 1 << 14;
            let coeffs = if tail_len >= PAR_THRESHOLD {
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                {
                    (0..tail_len)
                        .into_par_iter()
                        .fold(
                            || {
                                (
                                    vec![K::ZERO; deg_max + 1],
                                    vec![K::ZERO; deg_max + 1],
                                    vec![K::ZERO; deg_max + 1],
                                )
                            },
                            |(mut coeffs, mut inner, mut term_poly), t| {
                                step(&mut coeffs, &mut inner, &mut term_poly, t);
                                (coeffs, inner, term_poly)
                            },
                        )
                        .map(|(coeffs, _, _)| coeffs)
                        .reduce(
                            || vec![K::ZERO; deg_max + 1],
                            |mut a, b| {
                                for i in 0..=deg_max {
                                    a[i] += b[i];
                                }
                                a
                            },
                        )
                }
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                {
                    let mut coeffs = vec![K::ZERO; deg_max + 1];
                    let mut inner = vec![K::ZERO; deg_max + 1];
                    let mut term_poly = vec![K::ZERO; deg_max + 1];
                    for t in 0..tail_len {
                        step(&mut coeffs, &mut inner, &mut term_poly, t);
                    }
                    coeffs
                }
            } else {
                let mut coeffs = vec![K::ZERO; deg_max + 1];
                let mut inner = vec![K::ZERO; deg_max + 1];
                let mut term_poly = vec![K::ZERO; deg_max + 1];
                for t in 0..tail_len {
                    step(&mut coeffs, &mut inner, &mut term_poly, t);
                }
                coeffs
            };

            return if xs_are_base {
                xs.iter()
                    .map(|&x| crate::sumcheck::poly_eval_k_base(&coeffs, x.real()))
                    .collect()
            } else {
                xs.iter()
                    .map(|&x| crate::sumcheck::poly_eval_k(&coeffs, x))
                    .collect()
            };
        }

        // Fast path for b=3: range polynomial is N(y) = y(y^2-1)(y^2-4) = y^5 - 5y^3 + 4y.
        // As in the b=2 case, we build the univariate coefficients once per round and then
        // evaluate at all requested points.
        if self.b == 3 {
            if xs_all_base {
                return self.evals_row_phase_b3_base(tail_len, xs);
            }

            let deg_max = self.row_phase_deg_max;

            let coeffs = {
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                {
                    (0..tail_len)
                        .into_par_iter()
                        .fold(
                            || {
                                (
                                    vec![K::ZERO; deg_max + 1],
                                    vec![K::ZERO; deg_max + 1],
                                    vec![K::ZERO; deg_max + 1],
                                )
                            },
                            |(mut coeffs, mut inner, mut term_poly), t| {
                                // eq_beta_r(X) = e0 + e1·X
                                let e0 = self.eq_beta_r_tbl[2 * t];
                                let e1 = self.eq_beta_r_tbl[2 * t + 1] - e0;

                                self.accumulate_weighted_f_times_affine(
                                    2 * t,
                                    deg_max,
                                    e0,
                                    e1,
                                    &mut coeffs,
                                    &mut inner,
                                    &mut term_poly,
                                );

                                // Eval: eq_r_inputs(X) * gamma_to_k * eval_tbl(X) (quadratic).
                                if let (Some(eq_tbl), Some(eval_tbl)) =
                                    (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref())
                                {
                                    let r0 = eq_tbl[2 * t];
                                    let r1 = eq_tbl[2 * t + 1] - r0;
                                    let v0 = eval_tbl[2 * t];
                                    let v1 = eval_tbl[2 * t + 1] - v0;

                                    let g = self.gamma_to_k;
                                    coeffs[0] += g * (r0 * v0);
                                    coeffs[1] += g * (r0 * v1 + r1 * v0);
                                    coeffs[2] += g * (r1 * v1);
                                }

                                (coeffs, inner, term_poly)
                            },
                        )
                        .map(|(coeffs, _, _)| coeffs)
                        .reduce(
                            || vec![K::ZERO; deg_max + 1],
                            |mut a, b| {
                                for i in 0..=deg_max {
                                    a[i] += b[i];
                                }
                                a
                            },
                        )
                }
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                {
                    let mut coeffs = vec![K::ZERO; deg_max + 1];
                    let mut inner = vec![K::ZERO; deg_max + 1];
                    let mut term_poly = vec![K::ZERO; deg_max + 1];

                    for t in 0..tail_len {
                        // eq_beta_r(X) = e0 + e1·X
                        let e0 = self.eq_beta_r_tbl[2 * t];
                        let e1 = self.eq_beta_r_tbl[2 * t + 1] - e0;

                        self.accumulate_weighted_f_times_affine(
                            2 * t,
                            deg_max,
                            e0,
                            e1,
                            &mut coeffs,
                            &mut inner,
                            &mut term_poly,
                        );

                        // Eval: eq_r_inputs(X) * gamma_to_k * eval_tbl(X) (quadratic).
                        if let (Some(eq_tbl), Some(eval_tbl)) = (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref())
                        {
                            let r0 = eq_tbl[2 * t];
                            let r1 = eq_tbl[2 * t + 1] - r0;
                            let v0 = eval_tbl[2 * t];
                            let v1 = eval_tbl[2 * t + 1] - v0;

                            let g = self.gamma_to_k;
                            coeffs[0] += g * (r0 * v0);
                            coeffs[1] += g * (r0 * v1 + r1 * v0);
                            coeffs[2] += g * (r1 * v1);
                        }
                    }

                    coeffs
                }
            };

            return if xs_are_base {
                xs.iter()
                    .map(|&x| crate::sumcheck::poly_eval_k_base(&coeffs, x.real()))
                    .collect()
            } else {
                xs.iter()
                    .map(|&x| crate::sumcheck::poly_eval_k(&coeffs, x))
                    .collect()
            };
        }

        // Generic fallback: evaluate directly at each x (slower, but supports any b / K>1).
        let f_arity = self.f_var_count;

        // `xs` is typically very small (sumcheck evaluation points), so Rayon overhead dominates here.
        xs.iter()
            .map(|&x| {
                let one_minus = K::ONE - x;
                let mut var_vals = vec![K::ZERO; f_arity];
                let mut sum_x = K::ZERO;

                for t in 0..tail_len {
                    let eq_beta_r = one_minus * self.eq_beta_r_tbl[2 * t] + x * self.eq_beta_r_tbl[2 * t + 1];

                    // f_prime = Σ_{i=1..k_mcs} γ^{i-1} · f_i(m_vals_i).
                    let mut f_prime = K::ZERO;

                    for (mcs_idx, per_mcs_tables) in self.f_var_tables_by_mcs.iter().enumerate() {
                        let g_i = self.gamma_pow_mcs.get(mcs_idx).copied().unwrap_or(K::ONE);
                        if self.zero_mcs[mcs_idx] {
                            f_prime += g_i * self.f_at_zero;
                            continue;
                        }

                        // f variables at (prefix, x, tail) for this MCS slot
                        for (pos, tbl) in per_mcs_tables.iter().enumerate() {
                            var_vals[pos] = one_minus * tbl.get(2 * t) + x * tbl.get(2 * t + 1);
                        }

                        let mut f_i = K::ZERO;
                        for term in &self.f_terms {
                            let mut acc = term.coeff;
                            for &(var_pos, exp) in &term.vars {
                                let xi = var_vals[var_pos];
                                let mut p = xi;
                                for _ in 1..exp {
                                    p *= xi;
                                }
                                acc *= p;
                            }
                            f_i += acc;
                        }

                        f_prime += g_i * f_i;
                    }

                    let mut out = eq_beta_r * f_prime;

                    // Eval: eq_r_inputs(r') * gamma_to_k * eval_tbl(r')
                    if let (Some(eq_tbl), Some(eval_tbl)) = (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref()) {
                        let eq_r_inputs = one_minus * eq_tbl[2 * t] + x * eq_tbl[2 * t + 1];
                        if eq_r_inputs != K::ZERO {
                            let e = one_minus * eval_tbl[2 * t] + x * eval_tbl[2 * t + 1];
                            out += eq_r_inputs * (self.gamma_to_k * e);
                        }
                    }

                    sum_x += out;
                }

                sum_x
            })
            .collect()
    }

    #[inline]
    fn evals_row_phase<Ff>(&self, xs: &[K]) -> Vec<K>
    where
        Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
        K: From<Ff>,
    {
        self.evals_row_phase_impl::<Ff>(xs, true)
    }

    #[inline]
    fn evals_row_phase_force_generic<Ff>(&self, xs: &[K]) -> Vec<K>
    where
        Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
        K: From<Ff>,
    {
        self.evals_row_phase_impl::<Ff>(xs, false)
    }
}

/// Symmetric range polynomial: ∏_{t=-(b-1)}^{b-1} (y - t) = y · ∏_{t=1}^{b-1} (y² - t²)
/// using cached `t²` values for `t=1..(b-1)`.
#[inline]
fn range_product_cached(y: K, range_t_sq: &[K]) -> K {
    if range_t_sq.is_empty() {
        return y;
    }
    let y2 = y * y;
    let mut prod = y;
    for &tt2 in range_t_sq {
        prod *= y2 - tt2;
    }
    prod
}

#[inline]
fn eq_lin(a: K, b: K) -> K {
    (K::ONE - a) * (K::ONE - b) + a * b
}

/// Fold one Ajtai bit into-place for a digits table (size D).
#[inline]
fn fold_bit_inplace(digits: &mut [K; D], bit: usize, a: K) {
    let stride = 1usize << bit;
    let step = stride << 1;
    let n = D;
    let mut base = 0usize;
    while base < n {
        let mut off = 0usize;
        while off < stride {
            let i0 = base + off;
            if i0 >= n {
                break;
            }
            let i1 = i0 + stride;
            let lo = digits[i0];
            let hi = if i1 < n { digits[i1] } else { K::ZERO };
            digits[i0] = lo + (hi - lo) * a;
            off += 1;
        }
        base += step;
    }
}

/// Compute `c0 + c1·x`, where that affine polynomial is the tail-weighted
/// dot after folding the current Ajtai bit into `digits_pref`.
#[inline]
fn ajtai_tail_weighted_dot_affine_prefolded(
    digits_pref: &[K; D],
    bit: usize,
    head_stride: usize,
    w_tail: &[K],
) -> (K, K) {
    let stride = 1usize << bit;
    let mut c0 = K::ZERO;
    let mut c1 = K::ZERO;
    for (t, &w) in w_tail.iter().enumerate() {
        let idx = t * head_stride;
        if idx < D {
            let lo = digits_pref[idx];
            let hi_idx = idx + stride;
            let hi = if hi_idx < D { digits_pref[hi_idx] } else { K::ZERO };
            c0 += w * lo;
            c1 += w * (hi - lo);
        }
    }
    (c0, c1)
}

/// Fold the current Ajtai bit into `digits_pref` (which already has the prefix folded),
/// then compute the tail-weighted sum of the range polynomial N(·) over the MLE heads.
#[inline]
fn ajtai_tail_weighted_range_prefolded(
    digits_pref: &[K; D],
    x: K,
    bit: usize,
    head_stride: usize,
    w_tail: &[K],
    range_t_sq: &[K],
) -> K {
    let mut tmp = *digits_pref;
    fold_bit_inplace(&mut tmp, bit, x);
    let mut acc = K::ZERO;
    for (t, &w) in w_tail.iter().enumerate() {
        let idx = t * head_stride;
        if idx < D {
            acc += w * range_product_cached(tmp[idx], range_t_sq);
        }
    }
    acc
}

#[inline]
fn chi_tail_weights(bits: &[K]) -> Vec<K> {
    let t = bits.len();
    let len = 1usize << t;
    let mut w = vec![K::ZERO; len];
    w[0] = K::ONE;
    for (i, &b) in bits.iter().enumerate() {
        let step = 1usize << i;
        let one_minus = K::ONE - b;
        for mask in 0..step {
            let v = w[mask];
            w[mask] = v * one_minus;
            w[mask + step] = v * b;
        }
    }
    w
}

fn maybe_chi_tail_weights(bits: &[K], deferred: bool) -> Vec<K> {
    (!deferred)
        .then(|| chi_tail_weights(bits))
        .unwrap_or_default()
}

/// Precomputation for a fixed r' (row assignment) - eliminates redundant v_j recomputation
struct RPrecomp {
    /// Y_eval[i][j][ρ] = (Z_i · v_j)[ρ] for Eval terms  
    y_eval: Vec<Vec<[K; D]>>,
    /// F' = f(z_1 · v_j) - independent of α'
    f_prime: K,
    /// eq(r', β_r) - independent of α'
    eq_beta_r: K,
    /// eq(r', r_inputs) if present - independent of α'
    eq_r_inputs: K,
}

#[inline]
fn materialize_y_ring_from_precomputed_digits(y_by_mat: &[[K; D]], d_pad: usize) -> (Vec<Vec<K>>, Vec<K>) {
    let mut y_ring = Vec::with_capacity(y_by_mat.len());
    let mut ct = Vec::with_capacity(y_by_mat.len());
    for digits in y_by_mat {
        let mut row = vec![K::ZERO; d_pad];
        row[..D].copy_from_slice(digits);
        ct.push(digits[0]);
        y_ring.push(row);
    }
    (y_ring, ct)
}

/// Helper: compute eq for a boolean mask against a field vector
#[inline]
fn eq_points_bool_mask(mask: usize, points: &[K]) -> K {
    let mut prod = K::ONE;
    for (bit_idx, &p) in points.iter().enumerate() {
        let is_one = ((mask >> bit_idx) & 1) == 1;
        prod *= if is_one { p } else { K::ONE - p };
    }
    prod
}
