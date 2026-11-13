//! The Nightstream parameter set and its checks.
//!
//! Enforces:
//!  1) (k+1)·T·(b−1) < B where B=b^k  [Π_RLC bound]
//!  2) extension policy v1 for field-space soundness factors; and
//!  3) the production padded-row Π_CCS statistical census, which combines
//!     the joint SumCheck, the paper mixing polynomial, and the corrected
//!     coordinate-fork loss.
//!
//! Symbols match the paper: q, η, d=φ(η), κ (kappa), m, b, k, B, T, s.
//!
//! References: SuperNeo v1.2 Section 4 (Ajtai, strong set), Section 7.4 and
//! Definition 22 (Π_RLC bound), and `decisions/field-2p64-59-ring-x64.md`.

use core::fmt;
use num_bigint::BigUint;
use thiserror::Error;

pub mod poseidon2;

/// A validated parameter set. The fields are private, so no code can change
/// `b`, `k_rho` or `B` after `new` checks them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_snake_case)]
pub struct NeoParams {
    q: u64,
    eta: u32,
    d: u32,
    kappa: u32,
    m: u64,
    b: u32,
    k_rho: u32,
    B: u64,
    T: u32,
    s: u32,
    lambda: u32,
}

/// The only profile, "Nightstream P64, k_rho = 16": `q = 2^64 − 59` and the
/// ring `F_q[X]/(X^64 + 1)` (`decisions/field-2p64-59-ring-x64.md`).
pub mod nightstream_p64_k16 {
    pub const NAME: &str = "Nightstream P64, k_rho = 16";
    pub const Q: u64 = 0u64.wrapping_sub(59);
    pub const ETA: usize = 128;
    pub const D: usize = 64;
    pub const KAPPA: u32 = 18;
    pub const M: u64 = 1u64 << 30;
    pub const B_BASE: u32 = 2;
    pub const K_RHO: u32 = 16;
    pub const B: u64 = 1u64 << K_RHO;
    /// SuperNeo v1.2 Theorem 5 for a power-of-two cyclotomic ring: `T = 2·φ(η)`.
    pub const T: u32 = 128;
    pub const EXTENSION_DEGREE: u32 = 2;
    pub const LAMBDA: u32 = 125;
    /// SuperNeo v1.2 Theorem 4: `floor(q^(1/2)/√2)`. A nonzero ring element
    /// with infinity norm at most this value is invertible.
    pub const B_INV_FLOOR: u64 = 3_037_000_499;
    /// `floor(log2(5^64))`. The challenge set has `5^64` elements, more than a `u128` holds.
    pub const CHALLENGE_SET_LOG2_FLOOR: u32 = 148;
    /// The coefficients of `X^64 + 1` below degree `D`.
    pub static PHI_COEFFS: [i32; D] = {
        let mut coeffs = [0i32; D];
        coeffs[0] = 1;
        coeffs
    };

    pub static CHALLENGE_ALPHABET: [i8; 5] = [-2, -1, 0, 1, 2];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtensionSummary {
    pub s_min: u32,
    pub s_supported: u32,
    pub slack_bits: i32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaddedRowSecuritySummary {
    pub s_supported: u32,
    pub cube_variables: u32,
    pub verifier_degree: u32,
    pub sumcheck_factor: u128,
    pub mixing_factor: u128,
    pub field_factor: u128,
    pub fork_factor: u128,
    pub challenge_set_cardinality: BigUint,
    pub security_bits: u32,
    pub slack_bits: i32,
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum ParamsError {
    #[error("invalid parameter: {0}")]
    Invalid(&'static str),
    #[error("guard violated: (k+1)·T·(b−1) < B fails")]
    GuardInequality,
    #[error("unsupported extension degree; required s={required}, supported s=2")]
    UnsupportedExtension { required: u32 },
    #[error("arithmetic overflow while computing {0}")]
    ArithmeticOverflow(&'static str),
    #[error(
        "padded-row statistical security target is {required} bits, but the combined field and coordinate-fork census provides only {available} bits"
    )]
    InsufficientStatisticalSecurity { required: u32, available: u32 },
}

impl NeoParams {
    #[allow(non_snake_case)]
    pub fn new(
        q: u64,
        eta: u32,
        d: u32,
        kappa: u32,
        m: u64,
        b: u32,
        k_rho: u32,
        T: u32,
        s: u32,
        lambda: u32,
    ) -> Result<Self, ParamsError> {
        if q == 0 {
            return Err(ParamsError::Invalid("q must be nonzero"));
        }
        if eta == 0 {
            return Err(ParamsError::Invalid("eta must be > 0"));
        }
        if d == 0 {
            return Err(ParamsError::Invalid("d must be > 0"));
        }
        if kappa == 0 {
            return Err(ParamsError::Invalid("kappa must be > 0"));
        }
        if m == 0 {
            return Err(ParamsError::Invalid("m must be > 0"));
        }
        if b != 2 {
            return Err(ParamsError::Invalid("Nightstream fixes the decomposition base b = 2"));
        }
        if k_rho == 0 {
            return Err(ParamsError::Invalid("k_rho must be > 0"));
        }
        if T == 0 {
            return Err(ParamsError::Invalid("T must be > 0"));
        }
        if s != 2 {
            return Err(ParamsError::UnsupportedExtension { required: s });
        }
        if lambda == 0 {
            return Err(ParamsError::Invalid("lambda must be > 0"));
        }

        let B = pow_u64_checked(b as u64, k_rho)?;
        if (B as u128) * 2 >= q as u128 {
            return Err(ParamsError::Invalid("2*B must be strictly smaller than q"));
        }
        let lhs = (k_rho as u128 + 1) * (T as u128) * ((b as u128).saturating_sub(1));
        if lhs >= (B as u128) {
            return Err(ParamsError::GuardInequality);
        }

        Ok(Self {
            q,
            eta,
            d,
            kappa,
            m,
            b,
            k_rho,
            B,
            T,
            s,
            lambda,
        })
    }

    pub fn q(&self) -> u64 {
        self.q
    }

    pub fn eta(&self) -> u32 {
        self.eta
    }

    pub fn d(&self) -> u32 {
        self.d
    }

    pub fn kappa(&self) -> u32 {
        self.kappa
    }

    pub fn m(&self) -> u64 {
        self.m
    }

    pub fn b(&self) -> u32 {
        self.b
    }

    pub fn k_rho(&self) -> u32 {
        self.k_rho
    }

    /// The norm bound `B = b^k_rho`.
    #[allow(non_snake_case)]
    pub fn B(&self) -> u64 {
        self.B
    }

    /// The strong-set expansion factor.
    #[allow(non_snake_case)]
    pub fn T(&self) -> u32 {
        self.T
    }

    /// The extension degree of `K` over the base field.
    pub fn s(&self) -> u32 {
        self.s
    }

    pub fn lambda(&self) -> u32 {
        self.lambda
    }

    /// The same parameters with a different positive statistical target.
    pub fn with_lambda(self, lambda: u32) -> Result<Self, ParamsError> {
        if lambda == 0 {
            return Err(ParamsError::Invalid("lambda must be > 0"));
        }
        Ok(Self { lambda, ..self })
    }

    pub fn nightstream_p64_k16() -> Self {
        Self::new(
            nightstream_p64_k16::Q,
            nightstream_p64_k16::ETA as u32,
            nightstream_p64_k16::D as u32,
            nightstream_p64_k16::KAPPA,
            nightstream_p64_k16::M,
            nightstream_p64_k16::B_BASE,
            nightstream_p64_k16::K_RHO,
            nightstream_p64_k16::T,
            nightstream_p64_k16::EXTENSION_DEGREE,
            nightstream_p64_k16::LAMBDA,
        )
        .expect("Nightstream P64 k_rho=16 parameters")
    }

    #[inline]
    fn bitlen_u128(x: u128) -> u32 {
        if x == 0 {
            0
        } else {
            128 - x.leading_zeros()
        }
    }

    fn s2_feasible_factor(&self, soundness_factor: u128) -> Option<bool> {
        let q2 = (self.q as u128).checked_mul(self.q as u128)?;
        let pow2 = 1u128.checked_shl(self.lambda)?;
        let rhs = pow2.checked_mul(soundness_factor)?;
        Some(q2 >= rhs)
    }

    pub fn s_min(&self, ell: u32, d_sc: u32) -> u32 {
        let factor = (ell as u128) * (d_sc as u128);
        self.s_min_factor(factor)
    }

    pub fn s_min_factor(&self, soundness_factor: u128) -> u32 {
        if let Some(pow2) = 1u128.checked_shl(self.lambda) {
            if let Some(rhs) = pow2.checked_mul(soundness_factor) {
                if (self.q as u128) >= rhs {
                    return 1;
                }
            }
        }

        match self.s2_feasible_factor(soundness_factor) {
            Some(true) => 2,
            Some(false) => 3,
            None => 3,
        }
    }

    pub fn extension_check(&self, ell: u32, d_sc: u32) -> Result<ExtensionSummary, ParamsError> {
        let factor = (ell as u128)
            .checked_mul(d_sc as u128)
            .ok_or(ParamsError::UnsupportedExtension { required: 3 })?;
        self.extension_check_factor(factor)
    }

    pub fn extension_check_factor(&self, soundness_factor: u128) -> Result<ExtensionSummary, ParamsError> {
        let s_min = self.s_min_factor(soundness_factor);
        if s_min > 2 {
            return Err(ParamsError::UnsupportedExtension { required: s_min });
        }

        let q = self.q as u128;
        let q2 = q * q;

        let rhs = 1u128
            .checked_shl(self.lambda)
            .and_then(|p| p.checked_mul(soundness_factor))
            .ok_or(ParamsError::UnsupportedExtension { required: 3 })?;

        let mut slack_bits = Self::bitlen_u128(q2) as i32 - Self::bitlen_u128(rhs) as i32;
        if let Some(shifted) = rhs.checked_shl(slack_bits as u32) {
            if q2 < shifted {
                slack_bits -= 1;
            }
        }

        Ok(ExtensionSummary {
            s_min,
            s_supported: 2,
            slack_bits,
        })
    }

    pub fn padded_row_security_check_for_shape(
        &self,
        row_count: usize,
        column_count: usize,
        matrix_count: usize,
        poly_degree: u32,
        challenge_alphabet_size: u32,
    ) -> Result<PaddedRowSecuritySummary, ParamsError> {
        let summary = self.padded_row_security_summary_for_shape(
            row_count,
            column_count,
            matrix_count,
            poly_degree,
            challenge_alphabet_size,
        )?;
        if summary.slack_bits < 0 {
            return Err(ParamsError::InsufficientStatisticalSecurity {
                required: self.lambda,
                available: summary.security_bits,
            });
        }
        Ok(summary)
    }

    pub fn padded_row_security_summary_for_shape(
        &self,
        row_count: usize,
        column_count: usize,
        matrix_count: usize,
        poly_degree: u32,
        challenge_alphabet_size: u32,
    ) -> Result<PaddedRowSecuritySummary, ParamsError> {
        let PaddedRowFieldComponents {
            cube_variables,
            verifier_degree,
            sumcheck_factor,
            mixing_factor,
            field_factor,
        } = self.padded_row_field_components_for_shape(row_count, column_count, matrix_count, poly_degree)?;
        let fresh_count = self.max_fresh_count_from_rlc_guard()?;
        let fork_factor = (fresh_count as u128)
            .checked_add(self.k_rho as u128)
            .and_then(|value| value.checked_add(1))
            .ok_or(ParamsError::ArithmeticOverflow("coordinate-fork numerator"))?;
        if challenge_alphabet_size == 0 {
            return Err(ParamsError::Invalid("challenge set must be nonempty"));
        }

        let field_size = BigUint::from(self.q).pow(self.s);
        let challenge_set_cardinality = BigUint::from(challenge_alphabet_size).pow(self.d);
        let total_numerator =
            BigUint::from(field_factor) * &challenge_set_cardinality + BigUint::from(fork_factor) * &field_size;
        let total_denominator = &field_size * &challenge_set_cardinality;
        let security_bits = floor_log2_ratio(&total_denominator, &total_numerator);
        let slack_bits = signed_difference(security_bits, self.lambda);

        Ok(PaddedRowSecuritySummary {
            s_supported: self.s,
            cube_variables,
            verifier_degree,
            sumcheck_factor,
            mixing_factor,
            field_factor,
            fork_factor,
            challenge_set_cardinality,
            security_bits,
            slack_bits,
        })
    }

    pub fn pi_ccs_padded_row_field_factor_for_shape(
        &self,
        row_count: usize,
        column_count: usize,
        matrix_count: usize,
        poly_degree: u32,
    ) -> Result<u128, ParamsError> {
        Ok(self
            .padded_row_field_components_for_shape(row_count, column_count, matrix_count, poly_degree)?
            .field_factor)
    }

    /// Maximum fresh CCS inputs K allowed by Definition 22's RLC guard:
    /// `(K + k)·T·(b - 1) < B`.
    pub fn max_fresh_count_from_rlc_guard(&self) -> Result<u32, ParamsError> {
        let denom = (self.T as u128)
            .checked_mul((self.b as u128).saturating_sub(1))
            .ok_or(ParamsError::GuardInequality)?;
        if denom == 0 {
            return Err(ParamsError::GuardInequality);
        }
        let max_total = ((self.B as u128).saturating_sub(1)) / denom;
        if max_total <= self.k_rho as u128 {
            return Err(ParamsError::GuardInequality);
        }
        (max_total - self.k_rho as u128)
            .try_into()
            .map_err(|_| ParamsError::GuardInequality)
    }

    fn padded_row_field_components_for_shape(
        &self,
        row_count: usize,
        column_count: usize,
        matrix_count: usize,
        poly_degree: u32,
    ) -> Result<PaddedRowFieldComponents, ParamsError> {
        if matrix_count == 0 {
            return Err(ParamsError::Invalid("matrix_count must be > 0"));
        }
        if row_count == 0 || column_count == 0 {
            return Err(ParamsError::Invalid("row_count and column_count must be > 0"));
        }
        let ring_degree = self.d as usize;
        if ring_degree == 0 {
            return Err(ParamsError::Invalid("ring degree must be > 0"));
        }
        let row_bound =
            usize::try_from(self.m).map_err(|_| ParamsError::ArithmeticOverflow("parameter row-domain bound"))?;
        if row_count > row_bound {
            return Err(ParamsError::Invalid("row_count exceeds parameter row-domain bound"));
        }
        let assignment_bound = row_bound / ring_degree * ring_degree;
        if column_count > assignment_bound {
            return Err(ParamsError::Invalid(
                "column_count exceeds parameter packed-assignment bound",
            ));
        }
        let carrier_width = column_count
            .checked_add(ring_degree - 1)
            .and_then(|value| value.checked_div(ring_degree))
            .and_then(|blocks| blocks.checked_mul(ring_degree))
            .ok_or(ParamsError::ArithmeticOverflow("padded assignment width"))?;
        let cube_variables =
            padded_cube_variables(row_count.max(carrier_width), "joint padded-row domain must be nonempty")?;
        let ccs_degree = poly_degree
            .checked_add(1)
            .ok_or(ParamsError::ArithmeticOverflow("CCS SumCheck degree"))?;
        let norm_degree = self
            .b
            .checked_mul(2)
            .ok_or(ParamsError::ArithmeticOverflow("norm SumCheck degree"))?;
        let verifier_degree = ccs_degree.max(norm_degree).max(2);
        let numerator = pi_ccs_padded_row_field_numerator(
            cube_variables,
            verifier_degree,
            self.max_fresh_count_from_rlc_guard()? as u128,
            self.k_rho as u128,
            self.d as u128,
            matrix_count as u128,
        )?;
        let field_factor = numerator
            .sumcheck
            .checked_add(numerator.mixing)
            .ok_or(ParamsError::ArithmeticOverflow("padded-row field numerator"))?;

        Ok(PaddedRowFieldComponents {
            cube_variables,
            verifier_degree,
            sumcheck_factor: numerator.sumcheck,
            mixing_factor: numerator.mixing,
            field_factor,
        })
    }
}

struct PaddedRowFieldComponents {
    cube_variables: u32,
    verifier_degree: u32,
    sumcheck_factor: u128,
    mixing_factor: u128,
    field_factor: u128,
}

pub struct PaddedRowFieldNumerator {
    pub sumcheck: u128,
    pub mixing: u128,
}

pub fn pi_ccs_padded_row_field_numerator(
    cube_variables: u32,
    verifier_degree: u32,
    fresh_count: u128,
    running_count: u128,
    ring_degree: u128,
    matrix_count: u128,
) -> Result<PaddedRowFieldNumerator, ParamsError> {
    let sumcheck_factor = (cube_variables as u128)
        .checked_mul(verifier_degree as u128)
        .ok_or(ParamsError::ArithmeticOverflow("joint SumCheck numerator"))?;
    let source_count = fresh_count
        .checked_add(running_count)
        .ok_or(ParamsError::ArithmeticOverflow("paper source count"))?;
    let carried_offset = fresh_count
        .checked_add(source_count)
        .ok_or(ParamsError::ArithmeticOverflow("carried-evaluation offset"))?;
    let joint_matrix_count = matrix_count
        .checked_add(1)
        .ok_or(ParamsError::ArithmeticOverflow("identity-first matrix count"))?;
    let carried_count = running_count
        .checked_mul(joint_matrix_count)
        .and_then(|v| v.checked_mul(ring_degree))
        .ok_or(ParamsError::ArithmeticOverflow("carried-coordinate count"))?;
    let joint_coefficient_count = carried_offset
        .checked_add(carried_count)
        .ok_or(ParamsError::ArithmeticOverflow("joint coefficient count"))?;
    let mixing_factor = joint_coefficient_count
        .checked_sub(1)
        .and_then(|degree| degree.checked_add(cube_variables as u128))
        .ok_or(ParamsError::ArithmeticOverflow("gamma and alpha mixing degree"))?;
    Ok(PaddedRowFieldNumerator {
        sumcheck: sumcheck_factor,
        mixing: mixing_factor,
    })
}

fn padded_cube_variables(size: usize, zero_error: &'static str) -> Result<u32, ParamsError> {
    if size == 0 {
        return Err(ParamsError::Invalid(zero_error));
    }
    let padded = size
        .max(2)
        .checked_next_power_of_two()
        .ok_or(ParamsError::ArithmeticOverflow("padded rectangular dimension"))?;
    Ok(padded.trailing_zeros())
}

fn floor_log2_ratio(denominator: &BigUint, numerator: &BigUint) -> u32 {
    if numerator == &BigUint::from(0u8) {
        return u32::MAX;
    }
    if denominator < numerator {
        return 0;
    }

    let bit_gap = denominator.bits().saturating_sub(numerator.bits());
    let mut bits = bit_gap.min(u32::MAX as u64) as u32;
    if (numerator << bits) > *denominator {
        bits = bits.saturating_sub(1);
    } else if bits < u32::MAX && (numerator << (bits + 1)) <= *denominator {
        bits += 1;
    }
    bits
}

fn signed_difference(left: u32, right: u32) -> i32 {
    let difference = left as i64 - right as i64;
    difference.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

fn pow_u64_checked(base: u64, mut exp: u32) -> Result<u64, ParamsError> {
    let mut acc: u128 = 1;
    let mut b: u128 = base as u128;
    while exp > 0 {
        if (exp & 1) == 1 {
            acc = acc
                .checked_mul(b)
                .ok_or(ParamsError::Invalid("B overflow"))?;
        }
        exp >>= 1;
        if exp > 0 {
            b = b.checked_mul(b).ok_or(ParamsError::Invalid("B overflow"))?;
        }
    }
    acc.try_into()
        .map_err(|_| ParamsError::Invalid("B overflow"))
}

impl fmt::Display for NeoParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NeoParams{{ q=0x{:016X}, η={}, d={}, κ={}, m={}, b={}, k_rho={}, B={}, T={}, s={}, λ={} }}",
            self.q, self.eta, self.d, self.kappa, self.m, self.b, self.k_rho, self.B, self.T, self.s, self.lambda
        )
    }
}
