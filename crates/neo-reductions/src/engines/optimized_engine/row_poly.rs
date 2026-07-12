use neo_math::{Fq, KExtensions, K};
use p3_field::PrimeCharacteristicRing;

pub(super) struct RowTable {
    real: Vec<Fq>,
    imag: Option<Vec<Fq>>,
}

impl RowTable {
    pub(super) fn from_base(real: Vec<Fq>) -> Self {
        Self { real, imag: None }
    }

    #[inline]
    pub(super) fn get(&self, index: usize) -> K {
        K::from_coeffs([
            self.real[index],
            self.imag.as_ref().map_or(Fq::ZERO, |imag| imag[index]),
        ])
    }

    #[inline]
    pub(super) fn real(&self, index: usize) -> Fq {
        self.real[index]
    }

    pub(super) fn fold_inplace(&mut self, challenge: K) {
        debug_assert!(self.real.len() >= 2 && self.real.len().is_multiple_of(2));
        let half = self.real.len() / 2;
        if let Some(imag) = self.imag.as_mut() {
            for index in 0..half {
                let low = K::from_coeffs([self.real[2 * index], imag[2 * index]]);
                let high = K::from_coeffs([self.real[2 * index + 1], imag[2 * index + 1]]);
                let folded = low + (high - low) * challenge;
                [self.real[index], imag[index]] = folded.as_coeffs();
            }
            self.real.truncate(half);
            imag.truncate(half);
            return;
        }

        let [challenge_real, challenge_imag] = challenge.as_coeffs();
        if challenge_imag == Fq::ZERO {
            for index in 0..half {
                let low = self.real[2 * index];
                let high = self.real[2 * index + 1];
                self.real[index] = low + (high - low) * challenge_real;
            }
            self.real.truncate(half);
            return;
        }

        let mut folded_imag = vec![Fq::ZERO; half];
        for index in 0..half {
            let low = self.real[2 * index];
            let delta = self.real[2 * index + 1] - low;
            self.real[index] = low + delta * challenge_real;
            folded_imag[index] = delta * challenge_imag;
        }
        self.real.truncate(half);
        self.imag = Some(folded_imag);
    }
}

#[derive(Clone, Debug)]
pub(super) struct CompiledPolyTerm {
    pub coeff: K,
    /// (var_pos, exponent), where `var_pos` indexes the inner
    /// `RowTable` list of each row-stream MCS entry.
    pub vars: Vec<(usize, u32)>,
    pub kind: CompiledPolyTermKind,
}

#[derive(Clone, Debug)]
pub(super) struct CompiledPolyGroup {
    pub selector: usize,
    pub terms: Vec<CompiledPolyTerm>,
}

#[derive(Clone, Debug)]
pub(super) enum CompiledPolyTermKind {
    Constant,
    Linear {
        var: usize,
    },
    Power {
        var: usize,
        exp: u32,
    },
    Product2 {
        left: usize,
        right: usize,
    },
    ScaledPower {
        linear: usize,
        powered: usize,
        exp: u32,
    },
    Product3 {
        first: usize,
        second: usize,
        third: usize,
    },
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
            [(linear, 1), (powered, exp @ 2..=8)] | [(powered, exp @ 2..=8), (linear, 1)] => Self::ScaledPower {
                linear: *linear,
                powered: *powered,
                exp: *exp,
            },
            [(first, 1), (second, 1), (third, 1)] => Self::Product3 {
                first: *first,
                second: *second,
                third: *third,
            },
            _ => Self::Generic,
        }
    }

    fn is_fast(&self) -> bool {
        !matches!(self, Self::Generic)
    }
}

pub(super) fn factor_common_linear_terms(terms: &[CompiledPolyTerm]) -> (Vec<CompiledPolyGroup>, usize) {
    let max_var = terms
        .iter()
        .flat_map(|term| term.vars.iter().map(|&(var, _)| var))
        .max()
        .map_or(0, |var| var + 1);
    let mut counts = vec![0usize; max_var];
    for term in terms {
        for &(var, exponent) in &term.vars {
            if exponent == 1 {
                counts[var] += 1;
            }
        }
    }
    let mut candidates = (0..max_var).collect::<Vec<_>>();
    candidates.sort_unstable_by_key(|&var| (core::cmp::Reverse(counts[var]), var));

    let mut assigned = vec![false; terms.len()];
    let mut groups = Vec::new();
    let mut covered = 0usize;
    for selector in candidates {
        let indices = terms
            .iter()
            .enumerate()
            .filter_map(|(index, term)| {
                (!assigned[index]
                    && term
                        .vars
                        .iter()
                        .any(|&(var, exponent)| var == selector && exponent == 1))
                .then_some(index)
            })
            .collect::<Vec<_>>();
        if indices.len() < 2 {
            continue;
        }

        let reduced = indices
            .iter()
            .map(|&index| {
                let term = &terms[index];
                let vars = term
                    .vars
                    .iter()
                    .copied()
                    .filter(|&(var, exponent)| var != selector || exponent != 1)
                    .collect::<Vec<_>>();
                let kind = CompiledPolyTermKind::from_vars(&vars);
                CompiledPolyTerm {
                    coeff: term.coeff,
                    vars,
                    kind,
                }
            })
            .collect::<Vec<_>>();
        if reduced.iter().any(|term| !term.kind.is_fast()) {
            continue;
        }

        for index in indices {
            assigned[index] = true;
            covered += 1;
        }
        groups.push(CompiledPolyGroup {
            selector,
            terms: reduced,
        });
    }
    (groups, covered)
}

pub(super) fn accumulate_factored_groups_times_affine_base(
    groups: &[CompiledPolyGroup],
    per_mcs_tables: &[RowTable],
    idx: usize,
    deg_max: usize,
    outer_a: Fq,
    outer_b: Fq,
    scale: Fq,
    out: &mut [Fq],
    scratch: &mut [Fq],
) {
    for group in groups {
        let selector = &per_mcs_tables[group.selector];
        let selector_a = selector.real(idx);
        let selector_b = selector.real(idx + 1) - selector_a;
        if selector_a == Fq::ZERO && selector_b == Fq::ZERO {
            continue;
        }

        scratch.fill(Fq::ZERO);
        for term in &group.terms {
            let handled = accumulate_fast_term_base(
                &term.kind,
                per_mcs_tables,
                idx,
                deg_max,
                scratch,
                term.coeff.real() * scale,
            );
            debug_assert!(handled, "factored polynomial group retained a generic term");
        }

        let q0 = outer_a * selector_a;
        let q1 = outer_a * selector_b + outer_b * selector_a;
        let q2 = outer_b * selector_b;
        for degree in 0..=deg_max {
            out[degree] += q0 * scratch[degree];
            if degree >= 1 {
                out[degree] += q1 * scratch[degree - 1];
            }
            if degree >= 2 {
                out[degree] += q2 * scratch[degree - 2];
            }
        }
    }
}

pub(super) fn accumulate_factored_groups_times_affine(
    groups: &[CompiledPolyGroup],
    per_mcs_tables: &[RowTable],
    idx: usize,
    deg_max: usize,
    outer_a: K,
    outer_b: K,
    scale: K,
    out: &mut [K],
    scratch: &mut [K],
) {
    for group in groups {
        let selector = &per_mcs_tables[group.selector];
        let selector_a = selector.get(idx);
        let selector_b = selector.get(idx + 1) - selector_a;
        if selector_a == K::ZERO && selector_b == K::ZERO {
            continue;
        }

        scratch.fill(K::ZERO);
        for term in &group.terms {
            let handled = accumulate_fast_term(&term.kind, per_mcs_tables, idx, deg_max, scratch, term.coeff * scale);
            debug_assert!(handled, "factored polynomial group retained a generic term");
        }

        let q0 = outer_a * selector_a;
        let q1 = outer_a * selector_b + outer_b * selector_a;
        let q2 = outer_b * selector_b;
        for degree in 0..=deg_max {
            out[degree] += q0 * scratch[degree];
            if degree >= 1 {
                out[degree] += q1 * scratch[degree - 1];
            }
            if degree >= 2 {
                out[degree] += q2 * scratch[degree - 2];
            }
        }
    }
}

pub(super) fn accumulate_fast_term_base(
    kind: &CompiledPolyTermKind,
    per_mcs_tables: &[RowTable],
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
            let a = tbl.real(idx);
            let b = tbl.real(idx + 1) - a;
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
            let a = tbl.real(idx);
            let b = tbl.real(idx + 1) - a;
            if a == Fq::ZERO && b == Fq::ZERO {
                return true;
            }
            accumulate_affine_power_base(inner, coeff, a, b, exp as usize, deg_max);
            true
        }
        CompiledPolyTermKind::Product2 { left, right } => {
            let left_tbl = &per_mcs_tables[left];
            let right_tbl = &per_mcs_tables[right];
            let a0 = left_tbl.real(idx);
            let b0 = left_tbl.real(idx + 1) - a0;
            let a1 = right_tbl.real(idx);
            let b1 = right_tbl.real(idx + 1) - a1;
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
        CompiledPolyTermKind::ScaledPower { linear, powered, exp } => {
            let linear_tbl = &per_mcs_tables[linear];
            let powered_tbl = &per_mcs_tables[powered];
            let linear_a = linear_tbl.real(idx);
            let linear_b = linear_tbl.real(idx + 1) - linear_a;
            let powered_a = powered_tbl.real(idx);
            let powered_b = powered_tbl.real(idx + 1) - powered_a;
            if (linear_a == Fq::ZERO && linear_b == Fq::ZERO) || (powered_a == Fq::ZERO && powered_b == Fq::ZERO) {
                return true;
            }
            accumulate_affine_power_times_affine_base(
                inner,
                coeff,
                powered_a,
                powered_b,
                exp as usize,
                linear_a,
                linear_b,
                deg_max,
            );
            true
        }
        CompiledPolyTermKind::Product3 { first, second, third } => {
            let first_tbl = &per_mcs_tables[first];
            let second_tbl = &per_mcs_tables[second];
            let third_tbl = &per_mcs_tables[third];
            let a0 = first_tbl.real(idx);
            let b0 = first_tbl.real(idx + 1) - a0;
            let a1 = second_tbl.real(idx);
            let b1 = second_tbl.real(idx + 1) - a1;
            let a2 = third_tbl.real(idx);
            let b2 = third_tbl.real(idx + 1) - a2;
            accumulate_affine_product3_base(inner, coeff, [a0, a1, a2], [b0, b1, b2], deg_max);
            true
        }
        CompiledPolyTermKind::Generic => false,
    }
}

pub(super) fn accumulate_fast_term(
    kind: &CompiledPolyTermKind,
    per_mcs_tables: &[RowTable],
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
            let a = tbl.get(idx);
            let b = tbl.get(idx + 1) - a;
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
            let a = tbl.get(idx);
            let b = tbl.get(idx + 1) - a;
            if a == K::ZERO && b == K::ZERO {
                return true;
            }
            accumulate_affine_power(inner, coeff, a, b, exp as usize, deg_max);
            true
        }
        CompiledPolyTermKind::Product2 { left, right } => {
            let left_tbl = &per_mcs_tables[left];
            let right_tbl = &per_mcs_tables[right];
            let a0 = left_tbl.get(idx);
            let b0 = left_tbl.get(idx + 1) - a0;
            let a1 = right_tbl.get(idx);
            let b1 = right_tbl.get(idx + 1) - a1;
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
        CompiledPolyTermKind::ScaledPower { linear, powered, exp } => {
            let linear_tbl = &per_mcs_tables[linear];
            let powered_tbl = &per_mcs_tables[powered];
            let linear_a = linear_tbl.get(idx);
            let linear_b = linear_tbl.get(idx + 1) - linear_a;
            let powered_a = powered_tbl.get(idx);
            let powered_b = powered_tbl.get(idx + 1) - powered_a;
            if (linear_a == K::ZERO && linear_b == K::ZERO) || (powered_a == K::ZERO && powered_b == K::ZERO) {
                return true;
            }
            accumulate_affine_power_times_affine(
                inner,
                coeff,
                powered_a,
                powered_b,
                exp as usize,
                linear_a,
                linear_b,
                deg_max,
            );
            true
        }
        CompiledPolyTermKind::Product3 { first, second, third } => {
            let first_tbl = &per_mcs_tables[first];
            let second_tbl = &per_mcs_tables[second];
            let third_tbl = &per_mcs_tables[third];
            let a0 = first_tbl.get(idx);
            let b0 = first_tbl.get(idx + 1) - a0;
            let a1 = second_tbl.get(idx);
            let b1 = second_tbl.get(idx + 1) - a1;
            let a2 = third_tbl.get(idx);
            let b2 = third_tbl.get(idx + 1) - a2;
            accumulate_affine_product3(inner, coeff, [a0, a1, a2], [b0, b1, b2], deg_max);
            true
        }
        CompiledPolyTermKind::Generic => false,
    }
}

#[inline]
fn accumulate_affine_power_times_affine_base(
    inner: &mut [Fq],
    coeff: Fq,
    a: Fq,
    b: Fq,
    exp: usize,
    linear_a: Fq,
    linear_b: Fq,
    deg_max: usize,
) {
    debug_assert!(exp <= 8);
    let mut a_pows = [Fq::ONE; 9];
    let mut b_pows = [Fq::ONE; 9];
    for i in 1..=exp {
        a_pows[i] = a_pows[i - 1] * a;
        b_pows[i] = b_pows[i - 1] * b;
    }
    let scaled_a = coeff * linear_a;
    let scaled_b = coeff * linear_b;
    for i in 0..=exp {
        let power_coeff = Fq::from_u64(binom_u64(exp, i)) * a_pows[exp - i] * b_pows[i];
        if i <= deg_max {
            inner[i] += scaled_a * power_coeff;
        }
        if i < deg_max {
            inner[i + 1] += scaled_b * power_coeff;
        }
    }
}

#[inline]
fn accumulate_affine_power_times_affine(
    inner: &mut [K],
    coeff: K,
    a: K,
    b: K,
    exp: usize,
    linear_a: K,
    linear_b: K,
    deg_max: usize,
) {
    debug_assert!(exp <= 8);
    let mut a_pows = [K::ONE; 9];
    let mut b_pows = [K::ONE; 9];
    for i in 1..=exp {
        a_pows[i] = a_pows[i - 1] * a;
        b_pows[i] = b_pows[i - 1] * b;
    }
    let scaled_a = coeff * linear_a;
    let scaled_b = coeff * linear_b;
    for i in 0..=exp {
        let power_coeff = K::from(Fq::from_u64(binom_u64(exp, i))) * a_pows[exp - i] * b_pows[i];
        if i <= deg_max {
            inner[i] += scaled_a * power_coeff;
        }
        if i < deg_max {
            inner[i + 1] += scaled_b * power_coeff;
        }
    }
}

#[inline]
fn accumulate_affine_product3_base(inner: &mut [Fq], coeff: Fq, a: [Fq; 3], b: [Fq; 3], deg_max: usize) {
    if a.iter()
        .zip(b)
        .any(|(&a_i, b_i)| a_i == Fq::ZERO && b_i == Fq::ZERO)
    {
        return;
    }
    inner[0] += coeff * (a[0] * a[1] * a[2]);
    if deg_max >= 1 {
        inner[1] += coeff * (b[0] * a[1] * a[2] + a[0] * b[1] * a[2] + a[0] * a[1] * b[2]);
    }
    if deg_max >= 2 {
        inner[2] += coeff * (b[0] * b[1] * a[2] + b[0] * a[1] * b[2] + a[0] * b[1] * b[2]);
    }
    if deg_max >= 3 {
        inner[3] += coeff * (b[0] * b[1] * b[2]);
    }
}

#[inline]
fn accumulate_affine_product3(inner: &mut [K], coeff: K, a: [K; 3], b: [K; 3], deg_max: usize) {
    if a.iter()
        .zip(b)
        .any(|(&a_i, b_i)| a_i == K::ZERO && b_i == K::ZERO)
    {
        return;
    }
    inner[0] += coeff * (a[0] * a[1] * a[2]);
    if deg_max >= 1 {
        inner[1] += coeff * (b[0] * a[1] * a[2] + a[0] * b[1] * a[2] + a[0] * a[1] * b[2]);
    }
    if deg_max >= 2 {
        inner[2] += coeff * (b[0] * b[1] * a[2] + b[0] * a[1] * b[2] + a[0] * b[1] * b[2]);
    }
    if deg_max >= 3 {
        inner[3] += coeff * (b[0] * b[1] * b[2]);
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
