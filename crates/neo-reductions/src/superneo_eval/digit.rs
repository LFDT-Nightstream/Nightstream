use neo_math::{Rq, D, F};
use p3_field::PrimeCharacteristicRing;

#[inline]
pub(super) fn mul_by_digit_block(form: &Rq, digits: &Rq) -> Rq {
    let mut out = Rq::zero();
    accumulate_by_digit_block(&mut out.0, form, digits);
    out
}

#[inline]
pub(super) fn accumulate_by_digit_block(out: &mut [F; D], form: &Rq, digits: &Rq) {
    let neg_one = F::ZERO - F::ONE;
    for (idx, &digit) in digits.0.iter().enumerate() {
        if digit == F::ZERO {
            continue;
        }
        if digit == F::ONE {
            add_monomial_in_place(out, form, idx);
        } else if digit == neg_one {
            sub_monomial_in_place(out, form, idx);
        } else {
            rollback_digit_prefix(out, form, digits, idx, neg_one);
            accumulate_product(out, &form.mul(digits));
            return;
        }
    }
}

#[inline]
pub(super) fn accumulate_pair_by_digit_block(
    out_a: &mut [F; D],
    out_b: &mut [F; D],
    form_a: &Rq,
    form_b: &Rq,
    digits: &Rq,
) {
    let neg_one = F::ZERO - F::ONE;
    for (idx, &digit) in digits.0.iter().enumerate() {
        if digit == F::ZERO {
            continue;
        }
        if digit == F::ONE {
            apply_monomial_pair_in_place::<false>(out_a, out_b, form_a, form_b, idx);
        } else if digit == neg_one {
            apply_monomial_pair_in_place::<true>(out_a, out_b, form_a, form_b, idx);
        } else {
            rollback_digit_pair_prefix(out_a, out_b, form_a, form_b, digits, idx, neg_one);
            accumulate_product(out_a, &form_a.mul(digits));
            accumulate_product(out_b, &form_b.mul(digits));
            return;
        }
    }
}

#[inline]
fn accumulate_product(out: &mut [F; D], product: &Rq) {
    for i in 0..D {
        out[i] += product.0[i];
    }
}

#[inline]
fn rollback_digit_prefix(out: &mut [F; D], form: &Rq, digits: &Rq, until: usize, neg_one: F) {
    for (idx, &digit) in digits.0.iter().take(until).enumerate() {
        if digit == F::ONE {
            sub_monomial_in_place(out, form, idx);
        } else if digit == neg_one {
            add_monomial_in_place(out, form, idx);
        }
    }
}

#[inline]
fn rollback_digit_pair_prefix(
    out_a: &mut [F; D],
    out_b: &mut [F; D],
    form_a: &Rq,
    form_b: &Rq,
    digits: &Rq,
    until: usize,
    neg_one: F,
) {
    for (idx, &digit) in digits.0.iter().take(until).enumerate() {
        if digit == F::ONE {
            apply_monomial_pair_in_place::<true>(out_a, out_b, form_a, form_b, idx);
        } else if digit == neg_one {
            apply_monomial_pair_in_place::<false>(out_a, out_b, form_a, form_b, idx);
        }
    }
}

#[inline]
fn add_monomial_in_place(out: &mut [F; D], form: &Rq, j: usize) {
    apply_monomial_in_place::<false>(out, form, j);
}

#[inline]
fn sub_monomial_in_place(out: &mut [F; D], form: &Rq, j: usize) {
    apply_monomial_in_place::<true>(out, form, j);
}

#[inline]
fn add_or_sub<const SUB: bool>(out: &mut F, value: F) {
    if SUB {
        *out -= value;
    } else {
        *out += value;
    }
}

#[inline]
fn add_or_sub_opposite<const SUB: bool>(out: &mut F, value: F) {
    if SUB {
        *out += value;
    } else {
        *out -= value;
    }
}

#[inline]
fn apply_monomial_in_place<const SUB: bool>(out: &mut [F; D], form: &Rq, j: usize) {
    debug_assert!(j < D, "digit block monomial index must be < D");
    if j == 0 {
        for i in 0..D {
            add_or_sub::<SUB>(&mut out[i], form.0[i]);
        }
        return;
    }

    let first_reduced = D - j;
    let first_wrap = (D + D / 2).saturating_sub(j).min(D);

    for i in 0..first_reduced {
        add_or_sub::<SUB>(&mut out[i + j], form.0[i]);
    }

    for i in first_reduced..first_wrap {
        let coeff = form.0[i];
        let reduced = i + j - D;
        add_or_sub_opposite::<SUB>(&mut out[reduced], coeff);
        add_or_sub_opposite::<SUB>(&mut out[reduced + D / 2], coeff);
    }

    // Since Phi_81(X) = X^54 + X^27 + 1, X^81 = 1 in this ring.
    // The generic two-step reduction for degrees 81..106 cancels the
    // intermediate X^(deg-54) terms, leaving a single wrapped monomial.
    for i in first_wrap..D {
        add_or_sub::<SUB>(&mut out[i + j - D - D / 2], form.0[i]);
    }
}

#[inline]
fn apply_monomial_pair_in_place<const SUB: bool>(
    out_a: &mut [F; D],
    out_b: &mut [F; D],
    form_a: &Rq,
    form_b: &Rq,
    j: usize,
) {
    debug_assert!(j < D, "digit block monomial index must be < D");
    if j == 0 {
        for i in 0..D {
            add_or_sub::<SUB>(&mut out_a[i], form_a.0[i]);
            add_or_sub::<SUB>(&mut out_b[i], form_b.0[i]);
        }
        return;
    }

    let first_reduced = D - j;
    let first_wrap = (D + D / 2).saturating_sub(j).min(D);

    for i in 0..first_reduced {
        let idx = i + j;
        add_or_sub::<SUB>(&mut out_a[idx], form_a.0[i]);
        add_or_sub::<SUB>(&mut out_b[idx], form_b.0[i]);
    }

    for i in first_reduced..first_wrap {
        let reduced = i + j - D;
        let coeff_a = form_a.0[i];
        let coeff_b = form_b.0[i];
        add_or_sub_opposite::<SUB>(&mut out_a[reduced], coeff_a);
        add_or_sub_opposite::<SUB>(&mut out_a[reduced + D / 2], coeff_a);
        add_or_sub_opposite::<SUB>(&mut out_b[reduced], coeff_b);
        add_or_sub_opposite::<SUB>(&mut out_b[reduced + D / 2], coeff_b);
    }

    for i in first_wrap..D {
        let idx = i + j - D - D / 2;
        add_or_sub::<SUB>(&mut out_a[idx], form_a.0[i]);
        add_or_sub::<SUB>(&mut out_b[idx], form_b.0[i]);
    }
}
