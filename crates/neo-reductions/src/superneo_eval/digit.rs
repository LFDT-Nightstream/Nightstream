use neo_math::{
    signed_sums::{SignedLimbs, SmallDigits},
    Rq, D, F,
};

#[inline]
pub(super) fn mul_by_digit_block(form: &Rq, digits: &Rq) -> Rq {
    let mut out = Rq::zero();
    accumulate_by_digit_block(&mut out.0, form, digits);
    out
}

#[inline]
pub(super) fn mul_by_signed_unit_masks(form: &Rq, positive: u64, negative: u64) -> Rq {
    let mut out = Rq::zero();
    accumulate_by_signed_unit_masks(&mut out.0, form, positive, negative);
    out
}

#[inline]
pub(super) fn accumulate_by_digit_block(out: &mut [F; D], form: &Rq, digits: &Rq) {
    let Some(small) = SmallDigits::new(&digits.0) else {
        return accumulate_product(out, &form.mul(digits).0);
    };
    match small.signed_unit_masks() {
        Some((positive, negative)) => accumulate_by_signed_unit_masks(out, form, positive, negative),
        None => accumulate_product(out, &small.mul(&SignedLimbs::new(&form.0))),
    }
}

#[inline]
pub(super) fn accumulate_by_signed_unit_masks(out: &mut [F; D], form: &Rq, mut positive: u64, mut negative: u64) {
    debug_assert_eq!(positive & negative, 0, "signed-unit masks overlap");
    while positive != 0 {
        let index = positive.trailing_zeros() as usize;
        add_monomial_in_place(out, form, index);
        positive &= positive - 1;
    }
    while negative != 0 {
        let index = negative.trailing_zeros() as usize;
        sub_monomial_in_place(out, form, index);
        negative &= negative - 1;
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
    let Some(small) = SmallDigits::new(&digits.0) else {
        accumulate_product(out_a, &form_a.mul(digits).0);
        accumulate_product(out_b, &form_b.mul(digits).0);
        return;
    };
    if let Some((positive, negative)) = small.signed_unit_masks() {
        accumulate_pair_by_signed_unit_masks(out_a, out_b, form_a, form_b, positive, negative);
    } else {
        accumulate_product(out_a, &small.mul(&SignedLimbs::new(&form_a.0)));
        accumulate_product(out_b, &small.mul(&SignedLimbs::new(&form_b.0)));
    }
}

#[inline]
pub(super) fn accumulate_pair_by_signed_unit_masks(
    out_a: &mut [F; D],
    out_b: &mut [F; D],
    form_a: &Rq,
    form_b: &Rq,
    mut positive: u64,
    mut negative: u64,
) {
    debug_assert_eq!(positive & negative, 0, "signed-unit masks overlap");
    while positive != 0 {
        let index = positive.trailing_zeros() as usize;
        apply_monomial_pair_in_place::<false>(out_a, out_b, form_a, form_b, index);
        positive &= positive - 1;
    }
    while negative != 0 {
        let index = negative.trailing_zeros() as usize;
        apply_monomial_pair_in_place::<true>(out_a, out_b, form_a, form_b, index);
        negative &= negative - 1;
    }
}

#[inline]
fn accumulate_product(out: &mut [F; D], product: &[F; D]) {
    for i in 0..D {
        out[i] += product[i];
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

    // X^D = −1.
    let first_reduced = D - j;
    for i in 0..first_reduced {
        add_or_sub::<SUB>(&mut out[i + j], form.0[i]);
    }
    for i in first_reduced..D {
        add_or_sub_opposite::<SUB>(&mut out[i + j - D], form.0[i]);
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
    for i in 0..first_reduced {
        let idx = i + j;
        add_or_sub::<SUB>(&mut out_a[idx], form_a.0[i]);
        add_or_sub::<SUB>(&mut out_b[idx], form_b.0[i]);
    }
    for i in first_reduced..D {
        let idx = i + j - D;
        add_or_sub_opposite::<SUB>(&mut out_a[idx], form_a.0[i]);
        add_or_sub_opposite::<SUB>(&mut out_b[idx], form_b.0[i]);
    }
}
