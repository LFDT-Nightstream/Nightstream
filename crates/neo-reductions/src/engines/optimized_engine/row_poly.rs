use neo_math::{Fq, K};
use p3_field::PrimeCharacteristicRing;

#[derive(Clone, Debug)]
pub(super) struct CompiledPolyTerm {
    pub coeff: K,
    /// (var_pos, exponent), where `var_pos` indexes the inner
    /// `Vec<Vec<K>>` of each row-stream MCS entry.
    pub vars: Vec<(usize, u32)>,
    pub kind: CompiledPolyTermKind,
}

#[derive(Clone, Debug)]
pub(super) enum CompiledPolyTermKind {
    Constant,
    Linear { var: usize },
    Power { var: usize, exp: u32 },
    Product2 { left: usize, right: usize },
    Generic,
}

impl CompiledPolyTermKind {
    pub(super) fn from_vars(vars: &[(usize, u32)]) -> Self {
        match vars {
            [] => Self::Constant,
            [(var, 1)] => Self::Linear { var: *var },
            [(var, exp @ 2..=8)] => Self::Power { var: *var, exp: *exp },
            [(left, 1), (right, 1)] => Self::Product2 {
                left: *left,
                right: *right,
            },
            _ => Self::Generic,
        }
    }
}

pub(super) fn accumulate_fast_term_base(
    kind: &CompiledPolyTermKind,
    per_mcs_tables: &[Vec<K>],
    idx: usize,
    deg_max: usize,
    inner: &mut [Fq],
    coeff: Fq,
) -> bool {
    match *kind {
        CompiledPolyTermKind::Constant => {
            inner[0] += coeff;
            true
        }
        CompiledPolyTermKind::Linear { var } => {
            let tbl = &per_mcs_tables[var];
            let a = tbl[idx].real();
            let b = tbl[idx + 1].real() - a;
            if a == Fq::ZERO && b == Fq::ZERO {
                return true;
            }
            inner[0] += coeff * a;
            if deg_max >= 1 {
                inner[1] += coeff * b;
            }
            true
        }
        CompiledPolyTermKind::Power { var, exp } => {
            let tbl = &per_mcs_tables[var];
            let a = tbl[idx].real();
            let b = tbl[idx + 1].real() - a;
            if a == Fq::ZERO && b == Fq::ZERO {
                return true;
            }
            accumulate_affine_power_base(inner, coeff, a, b, exp as usize, deg_max);
            true
        }
        CompiledPolyTermKind::Product2 { left, right } => {
            let left_tbl = &per_mcs_tables[left];
            let right_tbl = &per_mcs_tables[right];
            let a0 = left_tbl[idx].real();
            let b0 = left_tbl[idx + 1].real() - a0;
            let a1 = right_tbl[idx].real();
            let b1 = right_tbl[idx + 1].real() - a1;
            if (a0 == Fq::ZERO && b0 == Fq::ZERO) || (a1 == Fq::ZERO && b1 == Fq::ZERO) {
                return true;
            }
            inner[0] += coeff * (a0 * a1);
            if deg_max >= 1 {
                inner[1] += coeff * (a0 * b1 + b0 * a1);
            }
            if deg_max >= 2 {
                inner[2] += coeff * (b0 * b1);
            }
            true
        }
        CompiledPolyTermKind::Generic => false,
    }
}

pub(super) fn accumulate_fast_term(
    kind: &CompiledPolyTermKind,
    per_mcs_tables: &[Vec<K>],
    idx: usize,
    deg_max: usize,
    inner: &mut [K],
    coeff: K,
) -> bool {
    match *kind {
        CompiledPolyTermKind::Constant => {
            inner[0] += coeff;
            true
        }
        CompiledPolyTermKind::Linear { var } => {
            let tbl = &per_mcs_tables[var];
            let a = tbl[idx];
            let b = tbl[idx + 1] - a;
            if a == K::ZERO && b == K::ZERO {
                return true;
            }
            inner[0] += coeff * a;
            if deg_max >= 1 {
                inner[1] += coeff * b;
            }
            true
        }
        CompiledPolyTermKind::Power { var, exp } => {
            let tbl = &per_mcs_tables[var];
            let a = tbl[idx];
            let b = tbl[idx + 1] - a;
            if a == K::ZERO && b == K::ZERO {
                return true;
            }
            accumulate_affine_power(inner, coeff, a, b, exp as usize, deg_max);
            true
        }
        CompiledPolyTermKind::Product2 { left, right } => {
            let left_tbl = &per_mcs_tables[left];
            let right_tbl = &per_mcs_tables[right];
            let a0 = left_tbl[idx];
            let b0 = left_tbl[idx + 1] - a0;
            let a1 = right_tbl[idx];
            let b1 = right_tbl[idx + 1] - a1;
            if (a0 == K::ZERO && b0 == K::ZERO) || (a1 == K::ZERO && b1 == K::ZERO) {
                return true;
            }
            inner[0] += coeff * (a0 * a1);
            if deg_max >= 1 {
                inner[1] += coeff * (a0 * b1 + b0 * a1);
            }
            if deg_max >= 2 {
                inner[2] += coeff * (b0 * b1);
            }
            true
        }
        CompiledPolyTermKind::Generic => false,
    }
}

#[inline]
fn accumulate_affine_power_base(inner: &mut [Fq], coeff: Fq, a: Fq, b: Fq, exp: usize, deg_max: usize) {
    debug_assert!(exp <= 8);
    let mut a_pows = [Fq::ONE; 9];
    let mut b_pows = [Fq::ONE; 9];
    for i in 1..=exp {
        a_pows[i] = a_pows[i - 1] * a;
        b_pows[i] = b_pows[i - 1] * b;
    }
    for i in 0..=core::cmp::min(exp, deg_max) {
        let binom = Fq::from_u64(binom_u64(exp, i));
        inner[i] += coeff * binom * a_pows[exp - i] * b_pows[i];
    }
}

#[inline]
fn accumulate_affine_power(inner: &mut [K], coeff: K, a: K, b: K, exp: usize, deg_max: usize) {
    debug_assert!(exp <= 8);
    let mut a_pows = [K::ONE; 9];
    let mut b_pows = [K::ONE; 9];
    for i in 1..=exp {
        a_pows[i] = a_pows[i - 1] * a;
        b_pows[i] = b_pows[i - 1] * b;
    }
    for i in 0..=core::cmp::min(exp, deg_max) {
        let binom = K::from(Fq::from_u64(binom_u64(exp, i)));
        inner[i] += coeff * binom * a_pows[exp - i] * b_pows[i];
    }
}

#[inline]
fn binom_u64(n: usize, k: usize) -> u64 {
    let k = core::cmp::min(k, n - k);
    let mut out = 1u64;
    for i in 0..k {
        out = (out * (n - i) as u64) / (i as u64 + 1);
    }
    out
}
