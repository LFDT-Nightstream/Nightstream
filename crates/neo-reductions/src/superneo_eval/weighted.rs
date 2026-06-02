use neo_math::{superneo_bar_block, KExtensions, Rq, D, F, K};
use p3_field::PrimeCharacteristicRing;

use super::coeff_dot;

#[inline]
fn weighted_projection_basis_forms(weights: &Rq) -> [Rq; D] {
    let mut forms = [Rq([F::ZERO; D]); D];
    for (local, slot) in forms.iter_mut().enumerate() {
        let mut e = [F::ZERO; D];
        e[local] = F::ONE;
        let bar = Rq(superneo_bar_block(e));
        *slot = weighted_projection_form(&bar, weights);
    }
    forms
}

#[inline]
pub(super) fn weighted_projection_basis_forms_from_k(weights: &[K; D]) -> ([Rq; D], [Rq; D]) {
    let mut weight_re = [F::ZERO; D];
    let mut weight_im = [F::ZERO; D];
    for (i, weight) in weights.iter().enumerate() {
        let [re, im] = weight.as_coeffs();
        weight_re[i] = re;
        weight_im[i] = im;
    }
    (
        weighted_projection_basis_forms(&Rq(weight_re)),
        weighted_projection_basis_forms(&Rq(weight_im)),
    )
}

#[inline]
pub(super) fn weighted_projection_form_from_orig(orig: &Rq, basis_forms: &[Rq; D]) -> Rq {
    let neg_one = F::ZERO - F::ONE;
    let mut first = None;
    let mut multiple = false;
    for (local, &coeff) in orig.0.iter().enumerate() {
        if coeff == F::ZERO {
            continue;
        }
        if first.is_none() {
            first = Some((local, coeff));
        } else {
            multiple = true;
            break;
        }
    }

    match (first, multiple) {
        (None, _) => return Rq([F::ZERO; D]),
        (Some((local, coeff)), false) => {
            let mut out = basis_forms[local];
            if coeff == F::ONE {
                return out;
            }
            if coeff == neg_one {
                for slot in &mut out.0 {
                    *slot = F::ZERO - *slot;
                }
            } else {
                for slot in &mut out.0 {
                    *slot *= coeff;
                }
            }
            return out;
        }
        _ => {}
    }

    let mut out = [F::ZERO; D];
    for (local, &coeff) in orig.0.iter().enumerate() {
        if coeff == F::ZERO {
            continue;
        }
        let form = &basis_forms[local].0;
        if coeff == F::ONE {
            for i in 0..D {
                out[i] += form[i];
            }
        } else if coeff == neg_one {
            for i in 0..D {
                out[i] -= form[i];
            }
        } else {
            for i in 0..D {
                out[i] += coeff * form[i];
            }
        }
    }
    Rq(out)
}

#[inline]
fn weighted_projection_form(lhs: &Rq, weights: &Rq) -> Rq {
    let mut out = [F::ZERO; D];
    for (basis, out_cell) in out.iter_mut().enumerate() {
        *out_cell = coeff_dot_mul_by_monomial(lhs, weights, basis);
    }
    Rq(out)
}

#[inline]
fn coeff_dot_mul_by_monomial(lhs: &Rq, weights: &Rq, basis: usize) -> F {
    if basis == 0 {
        return coeff_dot(weights, lhs);
    }

    let mut acc = F::ZERO;
    for i in 0..D {
        let coeff = lhs.0[i];
        if coeff == F::ZERO {
            continue;
        }

        let new_deg = i + basis;
        if new_deg < D {
            acc += weights.0[new_deg] * coeff;
        } else if new_deg < D + D / 2 {
            let reduced_deg = new_deg - D;
            acc -= weights.0[reduced_deg] * coeff;
            acc -= weights.0[reduced_deg + D / 2] * coeff;
        } else {
            let deg1 = new_deg - D / 2;
            let deg2 = new_deg - D;
            if deg2 < D {
                acc -= weights.0[deg2] * coeff;
            }
            if deg1 >= D {
                let deg1_red = deg1 - D;
                if deg1_red < D {
                    acc += weights.0[deg1_red] * coeff;
                    if deg1_red + D / 2 < D {
                        acc += weights.0[deg1_red + D / 2] * coeff;
                    }
                }
            } else {
                acc -= weights.0[deg1] * coeff;
            }
        }
    }
    acc
}
