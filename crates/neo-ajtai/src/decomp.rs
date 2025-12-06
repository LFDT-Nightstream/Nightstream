use crate::error::{AjtaiError, AjtaiResult};
use crate::util::to_balanced_i128;
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
    let mut Z = vec![Fq::ZERO; d * m]; // col-major by input convention: columns are cf digits of each entry
    for (j, &zij) in z.iter().enumerate() {
        let mut a = to_balanced_i128(zij);
        for i in 0..d {
            // Constant-time: always compute digit even if a == 0 to prevent timing side-channel
            let (digit, new_a) = match style {
                DecompStyle::NonNegative => {
                    let b_i128 = b as i128;
                    let r = ((a % b_i128) + b_i128) % b_i128;
                    (r as i32, (a - r) / b_i128)
                }
                DecompStyle::Balanced => {
                    // balanced in [-(b-1)..(b-1)]; choose residue with smallest absolute value
                    let mut r = a % (b as i128);
                    if r > (b as i128) / 2 {
                        r -= b as i128;
                    }
                    if r < -((b as i128) / 2) {
                        r += b as i128;
                    }
                    (r as i32, (a - r) / b as i128)
                }
            };
            Z[j * d + i] = if digit >= 0 {
                Fq::from_u64(digit as u64)
            } else {
                Fq::ZERO - Fq::from_u64((-digit) as u64)
            };
            a = new_a; // if a was 0 this just propagates zeros
        }
        // remaining digits already zero
    }
    Z
}

/// split_b: matrix Z (d×m) with ||Z||∞ < b^k → (Z1..Zk), each ||Zi||∞ < b and Σ b^{i-1} Zi = Z (Def. 11).
#[allow(non_snake_case)]
pub fn split_b(Z: &[Fq], b: u32, d: usize, m: usize, k: usize, style: DecompStyle) -> Vec<Vec<Fq>> {
    let mut out = vec![vec![Fq::ZERO; d * m]; k];
    for col in 0..m {
        for row in 0..d {
            let idx = col * d + row;
            let mut a = to_balanced_i128(Z[idx]);
            #[allow(clippy::needless_range_loop)]
            for i in 0..k {
                // Constant-time: always compute digit even if a == 0 to prevent timing side-channel
                let (digit, new_a) = match style {
                    DecompStyle::NonNegative => {
                        let b_i128 = b as i128;
                        let r = ((a % b_i128) + b_i128) % b_i128;
                        (r as i32, (a - r) / b_i128)
                    }
                    DecompStyle::Balanced => {
                        let mut r = a % (b as i128);
                        if r > (b as i128) / 2 {
                            r -= b as i128;
                        }
                        if r < -((b as i128) / 2) {
                            r += b as i128;
                        }
                        (r as i32, (a - r) / b as i128)
                    }
                };
                out[i][idx] = if digit >= 0 {
                    Fq::from_u64(digit as u64)
                } else {
                    Fq::ZERO - Fq::from_u64((-digit) as u64)
                };
                a = new_a; // if a was 0 this just propagates zeros
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
