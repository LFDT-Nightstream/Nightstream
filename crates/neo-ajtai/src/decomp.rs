use crate::error::{AjtaiError, AjtaiResult};
use crate::util::to_balanced_i64;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;

/// Decomposition style: `Balanced` digits in [-(b-1)..(b-1)] or `NonNegative` digits in [0..b-1].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DecompStyle {
    Balanced,
    NonNegative,
}

/// decomp_b: vector z ∈ F_q^m → Z ∈ F_q^{d×m} with ||Z||_∞ < b (Def. 11).
#[allow(non_snake_case)]
pub fn decomp_b(z: &[Fq], b: u32, d: usize, style: DecompStyle) -> Vec<Fq> {
    let m = z.len();
    if b == 2 {
        let mut Z = Vec::with_capacity(d * m);
        match style {
            DecompStyle::NonNegative => {
                for &zij in z {
                    let mut a = to_balanced_i64(zij);
                    for _ in 0..d {
                        // Euclidean division by 2: r ∈ {0,1} is just the parity bit.
                        let digit = (a & 1) as u64;
                        Z.push(Fq::from_u64(digit));
                        // For divisor 2, `div_euclid(2)` matches arithmetic shift-right.
                        a >>= 1;
                    }
                }
            }
            DecompStyle::Balanced => {
                let one = Fq::ONE;
                let neg_one = Fq::ZERO - one;
                for &zij in z {
                    let mut a = to_balanced_i64(zij);
                    for _ in 0..d {
                        // Balanced digits in {-1,0,1} for b=2.
                        let digit = if (a & 1) == 0 {
                            0i64
                        } else if a >= 0 {
                            1i64
                        } else {
                            -1i64
                        };
                        Z.push(match digit {
                            -1 => neg_one,
                            0 => Fq::ZERO,
                            1 => one,
                            _ => unreachable!("b=2 digit must be -1/0/1"),
                        });
                        // (a - digit) is even, so an arithmetic shift matches `/ 2`.
                        a = (a - digit) >> 1;
                    }
                }
            }
        }
        return Z;
    }
    // Column-major by input convention: columns are cf digits of each entry.
    // Use `with_capacity` + `push` to avoid an upfront zero-fill of a ~d*m buffer.
    let mut Z = Vec::with_capacity(d * m);
    let b_i64 = b as i64;
    match style {
        DecompStyle::NonNegative => {
            for &zij in z {
                let mut a = to_balanced_i64(zij);
                for _ in 0..d {
                    // Constant-time: always compute digit even if a == 0 to prevent timing side-channel
                    let r = a.rem_euclid(b_i64);
                    let q = a.div_euclid(b_i64);
                    let digit = r as i32;
                    Z.push(if digit >= 0 {
                        Fq::from_u64(digit as u64)
                    } else {
                        Fq::ZERO - Fq::from_u64((-digit) as u64)
                    });
                    a = q; // if a was 0 this just propagates zeros
                }
                // remaining digits already zero
            }
        }
        DecompStyle::Balanced => {
            // Balanced in [-(b-1)..(b-1)]; choose residue with smallest absolute value.
            let half = b_i64 / 2;
            for &zij in z {
                let mut a = to_balanced_i64(zij);
                for _ in 0..d {
                    // Constant-time: always compute digit even if a == 0 to prevent timing side-channel
                    let mut r = a % b_i64;
                    if r > half {
                        r -= b_i64;
                    }
                    if r < -half {
                        r += b_i64;
                    }
                    let digit = r as i32;
                    Z.push(if digit >= 0 {
                        Fq::from_u64(digit as u64)
                    } else {
                        Fq::ZERO - Fq::from_u64((-digit) as u64)
                    });
                    a = (a - r) / b_i64; // if a was 0 this just propagates zeros
                }
                // remaining digits already zero
            }
        }
    }
    Z
}

/// decomp_b_row_major: vector z ∈ F_q^m → Z ∈ F_q^{d×m} with ||Z||_∞ < b (Def. 11),
/// returned in **row-major** order.
///
/// This is equivalent to `decomp_b` followed by a transpose into `neo_ccs::matrix::Mat` layout,
/// but avoids the extra allocation and transpose pass.
#[allow(non_snake_case)]
pub fn decomp_b_row_major(z: &[Fq], b: u32, d: usize, style: DecompStyle) -> Vec<Fq> {
    let m = z.len();
    if b == 2 {
        // Row-major output: write contiguously by iterating rows outermost.
        let mut a_vals: Vec<i64> = z.iter().copied().map(to_balanced_i64).collect();
        let mut Z = Vec::with_capacity(d * m);
        match style {
            DecompStyle::NonNegative => {
                for _row in 0..d {
                    for a in a_vals.iter_mut() {
                        let digit = (*a & 1) as u64;
                        Z.push(Fq::from_u64(digit));
                        *a >>= 1;
                    }
                }
            }
            DecompStyle::Balanced => {
                let one = Fq::ONE;
                let neg_one = Fq::ZERO - one;
                for _row in 0..d {
                    for a in a_vals.iter_mut() {
                        let a0 = *a;
                        let digit = if (a0 & 1) == 0 {
                            0i64
                        } else if a0 >= 0 {
                            1i64
                        } else {
                            -1i64
                        };
                        Z.push(match digit {
                            -1 => neg_one,
                            0 => Fq::ZERO,
                            1 => one,
                            _ => unreachable!("b=2 digit must be -1/0/1"),
                        });
                        *a = (a0 - digit) >> 1;
                    }
                }
            }
        }
        return Z;
    }
    if b == 3 {
        // Fast path: constant divisor enables LLVM to optimize division/mod by 3.
        let mut a_vals: Vec<i64> = z.iter().copied().map(to_balanced_i64).collect();
        let mut Z = Vec::with_capacity(d * m);
        match style {
            DecompStyle::NonNegative => {
                let two = Fq::from_u64(2);
                for _row in 0..d {
                    for a in a_vals.iter_mut() {
                        let r = a.rem_euclid(3);
                        Z.push(match r {
                            0 => Fq::ZERO,
                            1 => Fq::ONE,
                            2 => two,
                            _ => unreachable!("rem_euclid(3) must be in 0..=2"),
                        });
                        *a = a.div_euclid(3);
                    }
                }
            }
            DecompStyle::Balanced => {
                let one = Fq::ONE;
                let neg_one = Fq::ZERO - one;
                for _row in 0..d {
                    for a in a_vals.iter_mut() {
                        let mut r = *a % 3;
                        if r > 1 {
                            r -= 3;
                        }
                        if r < -1 {
                            r += 3;
                        }
                        Z.push(match r {
                            -1 => neg_one,
                            0 => Fq::ZERO,
                            1 => one,
                            _ => unreachable!("balanced mod 3 digit must be -1/0/1"),
                        });
                        *a = (*a - r) / 3;
                    }
                }
            }
        }
        return Z;
    }
    let b_i64 = b as i64;
    // Row-major output: write contiguously by iterating rows outermost. This avoids the
    // strided stores of the column-outer decomposition and also avoids zero-filling a
    // ~d*m buffer up front.
    let mut a_vals: Vec<i64> = z.iter().copied().map(to_balanced_i64).collect();
    let mut Z = Vec::with_capacity(d * m);
    match style {
        DecompStyle::NonNegative => {
            for _row in 0..d {
                for a in a_vals.iter_mut() {
                    // Constant-time: always compute digit even if a == 0 to prevent timing side-channel
                    let r = a.rem_euclid(b_i64);
                    let q = a.div_euclid(b_i64);
                    let digit = r as i32;
                    Z.push(if digit >= 0 {
                        Fq::from_u64(digit as u64)
                    } else {
                        Fq::ZERO - Fq::from_u64((-digit) as u64)
                    });
                    *a = q; // if a was 0 this just propagates zeros
                }
            }
        }
        DecompStyle::Balanced => {
            // Balanced in [-(b-1)..(b-1)]; choose residue with smallest absolute value.
            let half = b_i64 / 2;
            for _row in 0..d {
                for a in a_vals.iter_mut() {
                    // Constant-time: always compute digit even if a == 0 to prevent timing side-channel
                    let mut r = *a % b_i64;
                    if r > half {
                        r -= b_i64;
                    }
                    if r < -half {
                        r += b_i64;
                    }
                    let digit = r as i32;
                    Z.push(if digit >= 0 {
                        Fq::from_u64(digit as u64)
                    } else {
                        Fq::ZERO - Fq::from_u64((-digit) as u64)
                    });
                    *a = (*a - r) / b_i64; // if a was 0 this just propagates zeros
                }
            }
        }
    }
    Z
}

/// split_b: matrix Z (d×m) with ||Z||∞ < b^k → (Z1..Zk), each ||Zi||∞ < b and Σ b^{i-1} Zi = Z (Def. 11).
#[allow(non_snake_case)]
pub fn split_b(Z: &[Fq], b: u32, d: usize, m: usize, k: usize, style: DecompStyle) -> Vec<Vec<Fq>> {
    let mut out = vec![vec![Fq::ZERO; d * m]; k];
    let b_i64 = b as i64;
    match style {
        DecompStyle::NonNegative => {
            for col in 0..m {
                for row in 0..d {
                    let idx = col * d + row;
                    let mut a = to_balanced_i64(Z[idx]);
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..k {
                        // Constant-time: always compute digit even if a == 0 to prevent timing side-channel
                        let r = a.rem_euclid(b_i64);
                        let q = a.div_euclid(b_i64);
                        let digit = r as i32;
                        out[i][idx] = if digit >= 0 {
                            Fq::from_u64(digit as u64)
                        } else {
                            Fq::ZERO - Fq::from_u64((-digit) as u64)
                        };
                        a = q; // if a was 0 this just propagates zeros
                    }
                }
            }
        }
        DecompStyle::Balanced => {
            let half = b_i64 / 2;
            for col in 0..m {
                for row in 0..d {
                    let idx = col * d + row;
                    let mut a = to_balanced_i64(Z[idx]);
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..k {
                        // Constant-time: always compute digit even if a == 0 to prevent timing side-channel
                        let mut r = a % b_i64;
                        if r > half {
                            r -= b_i64;
                        }
                        if r < -half {
                            r += b_i64;
                        }
                        let digit = r as i32;
                        out[i][idx] = if digit >= 0 {
                            Fq::from_u64(digit as u64)
                        } else {
                            Fq::ZERO - Fq::from_u64((-digit) as u64)
                        };
                        a = (a - r) / b_i64; // if a was 0 this just propagates zeros
                    }
                }
            }
        }
    }
    out
}

/// Range assertion (MUST): checks max ∞-norm of coefficients is < b on Z ∈ F_q^{d×m}.
#[allow(non_snake_case)]
pub fn assert_range_b(Z: &[Fq], b: u32) -> AjtaiResult<()> {
    let b_i = b as i128;
    for &x in Z {
        let v = crate::util::to_balanced_i128(x);
        if v.abs() >= b_i {
            return Err(AjtaiError::RangeViolation { value: v, bound: b });
        }
    }
    Ok(())
}
