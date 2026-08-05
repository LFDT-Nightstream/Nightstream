//! Typed parameter sets for Neo (Nguyen–Setty 2025).
//!
//! Exposes field/cyclotomic/commitment/folding parameters and enforces:
//!  1) (k+1)·T·(b−1) < B where B=b^k  [Π_RLC bound]
//!  2) extension policy v1 for field-space soundness factors; and
//!  3) the production rectangular Π_CCS statistical census, which combines
//!     both SumChecks, the rectangular mixing polynomial, and the corrected
//!     coordinate-fork loss.
//!
//! Symbols match the paper: q, η, d=φ(η), κ (kappa), m, b, k, B, T, s.
//!
//! References: Sec. 3–4 (Ajtai, strong set, Π_RLC bound); Appendix B.2 (Goldilocks preset).
//!
//! NOTE: The per-instance (ℓ, d_sc) used in the sum-check live in neo-fold.
//!       Use `extension_check()` *there* with the preset's q, s, λ.
//!
//! ## Cryptographic Primitives
//!
//! This crate also provides the canonical Poseidon2 configuration used throughout Neo.
//! All hash operations (transcripts, digests) MUST use this single source of truth.

use core::fmt;
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Production Poseidon2 over Goldilocks (single source of truth)
pub mod poseidon2_goldilocks;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[allow(non_snake_case)] // Allow mathematical notation from paper (B, T, etc.)
pub struct NeoParams {
    /// Base field modulus q (e.g., Goldilocks q = 2^64 − 2^32 + 1).
    pub q: u64,
    /// Cyclotomic index η (e.g., 81). φ(η) = d is the ring/coefficient dimension.
    pub eta: u32,
    /// d = φ(η) (e.g., 54 when η=81). Also the dimension of S⊂F_q^{d×d}.
    pub d: u32,
    /// MSIS module rank κ used in Ajtai Setup(M ∈ R_q^{κ×m}).
    pub kappa: u32,
    /// Number of columns (message length) m committed with Ajtai.
    pub m: u64,
    /// Decomposition base b (usually 2).
    pub b: u32,
    /// Folding exponent k so that B = b^k.
    /// Related to decomposition exponent where B = b^{k_rho}.
    pub k_rho: u32,
    /// Upper ℓ∞ bound used by Ajtai binding *after* RLC: B = b^{k_rho}.
    pub B: u64,
    /// Expansion factor of the strong challenge set C ⊂ S (empirical/spec bound).
    pub T: u32,
    /// Extension degree s used by sum-check over K=F_{q^s} (v1 supports s=2 only).
    pub s: u32,
    /// Statistical target λ bound into the protocol header.
    ///
    /// The Appendix B.2 reference value is not, by itself, an end-to-end
    /// security claim. Executable rectangular profiles select this value from
    /// the combined field and coordinate-fork census for their concrete shape.
    pub lambda: u32,
}

/// Single source of truth for the SuperNeo Appendix B.2 Goldilocks profile.
pub mod goldilocks_paper_b2 {
    pub const Q: u64 = 0xFFFF_FFFF_0000_0001;
    pub const ETA: usize = 81;
    pub const D: usize = 54;
    pub const PHI_MID_DEGREE: usize = 27;
    pub const KAPPA: u32 = 18;
    pub const M: u64 = 1u64 << 30;
    pub const B_BASE: u32 = 2;
    pub const K_RHO: u32 = 14;
    pub const B: u64 = 1u64 << K_RHO;
    pub const T: u32 = 216;
    pub const EXTENSION_DEGREE: u32 = 2;
    pub const LAMBDA: u32 = 125;
    pub const MAX_FRESH_K: u32 = 61;
    pub const B_INV_FLOOR: u64 = 2_500_000_000;
    pub const CHALLENGE_SET_CARDINALITY: u128 = 55_511_151_231_257_827_021_181_583_404_541_015_625;

    pub static PHI_COEFFS: [i32; D] = {
        let mut coeffs = [0i32; D];
        coeffs[0] = 1;
        coeffs[PHI_MID_DEGREE] = 1;
        coeffs
    };

    pub static CHALLENGE_ALPHABET: [i8; 5] = [-2, -1, 0, 1, 2];
}

/// Summary returned by the extension policy check for a concrete soundness factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtensionSummary {
    pub s_min: u32,
    pub s_supported: u32,
    /// slack_bits = s_supported·log2(q) − (λ + log2(soundness_factor))
    pub slack_bits: i32,
}

/// Exact statistical census for the one-joint padded-row protocol.
///
/// The field and coordinate-fork terms have different denominators. The
/// `security_bits` value is therefore computed from their exact sum, not from
/// the minimum of two separately rounded bit estimates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaddedRowSecuritySummary {
    pub s_supported: u32,
    pub cube_variables: u32,
    pub verifier_degree: u32,
    pub sumcheck_factor: u128,
    pub mixing_factor: u128,
    pub field_factor: u128,
    pub fork_factor: u128,
    pub challenge_set_cardinality: u128,
    /// Floor of `-log2(field_factor / q^s + fork_factor / |C|)`.
    pub security_bits: u32,
    /// `security_bits - lambda` for the selected parameter set.
    pub slack_bits: i32,
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum ParamsError {
    #[error("invalid parameter: {0}")]
    Invalid(&'static str),
    #[error("guard violated: (k+1)·T·(b−1) < B fails")]
    GuardInequality,
    #[error("unsupported extension degree; required s={required}, supported s=2")]
    /// This includes cases where 2^λ * (ℓ·d_sc) overflows u128, implying s_min ≥ 3.
    UnsupportedExtension { required: u32 },
    #[error("arithmetic overflow while computing {0}")]
    ArithmeticOverflow(&'static str),
    #[error(
        "rectangular statistical security target is {required} bits, but the combined field and coordinate-fork census provides only {available} bits"
    )]
    InsufficientStatisticalSecurity { required: u32, available: u32 },
}

impl NeoParams {
    /// Construct and validate a parameter set; computes B=b^{k_rho} and enforces the RLC guard.
    #[allow(non_snake_case)] // Allow mathematical notation from paper
    #[allow(clippy::too_many_arguments)] // All parameters needed for comprehensive validation
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
        if b < 2 {
            return Err(ParamsError::Invalid("b must be >= 2"));
        }
        if k_rho == 0 {
            return Err(ParamsError::Invalid("k_rho must be > 0"));
        }
        if T == 0 {
            return Err(ParamsError::Invalid("T must be > 0"));
        }
        if s != 2 {
            return Err(ParamsError::UnsupportedExtension { required: s });
        } // v1 policy
        if lambda == 0 {
            return Err(ParamsError::Invalid("lambda must be > 0"));
        }

        let B = pow_u64_checked(b as u64, k_rho)?;
        if (B as u128) * 2 >= q as u128 {
            return Err(ParamsError::Invalid("2*B must be strictly smaller than q"));
        }
        // Enforce (k_rho+1)·T·(b-1) < B   [Π_RLC bound]
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

    /// Goldilocks, Appendix B.2: η=81, d=54, κ=18, m=2^30, b=2, k_rho=14, B=2^14, T=216, s=2.
    /// With Goldilocks q = 2^64 - 2^32 + 1, log₂(q) ≈ 63.999999999966 < 64, so q² < 2^128.
    /// We set λ=125 to stay within the paper's |C|≈2^125 challenge-set size.
    /// Guard: (k_rho+1)T(b−1)=15·216·1=3240 < 16384 ✓
    #[allow(non_snake_case)] // Allow mathematical notation from paper
    pub fn goldilocks_paper_b2() -> Self {
        // new() computes/validates B and guard; unwrap() is safe for a known-good preset.
        Self::new(
            goldilocks_paper_b2::Q,
            goldilocks_paper_b2::ETA as u32,
            goldilocks_paper_b2::D as u32,
            goldilocks_paper_b2::KAPPA,
            goldilocks_paper_b2::M,
            goldilocks_paper_b2::B_BASE,
            goldilocks_paper_b2::K_RHO,
            goldilocks_paper_b2::T,
            goldilocks_paper_b2::EXTENSION_DEGREE,
            goldilocks_paper_b2::LAMBDA,
        )
        .unwrap()
    }

    pub fn is_goldilocks_paper_b2(&self) -> bool {
        self.has_goldilocks_paper_b2_core() && self.lambda == goldilocks_paper_b2::LAMBDA
    }

    pub fn has_goldilocks_paper_b2_core(&self) -> bool {
        self.q == goldilocks_paper_b2::Q
            && self.eta == goldilocks_paper_b2::ETA as u32
            && self.d == goldilocks_paper_b2::D as u32
            && self.kappa == goldilocks_paper_b2::KAPPA
            && self.m == goldilocks_paper_b2::M
            && self.b == goldilocks_paper_b2::B_BASE
            && self.k_rho == goldilocks_paper_b2::K_RHO
            && self.B == goldilocks_paper_b2::B
            && self.T == goldilocks_paper_b2::T
            && self.s == goldilocks_paper_b2::EXTENSION_DEGREE
    }

    /// Auto-pick parameters for a square R1CS-derived CCS test shape.
    ///
    /// Production callers that know distinct row and column counts must use
    /// [`Self::goldilocks_auto_rectangular_r1cs_ccs_with`]. This convenience
    /// method treats `shape_size` as both dimensions.
    pub fn goldilocks_auto_r1cs_ccs(n_rows: usize) -> Result<Self, ParamsError> {
        Self::goldilocks_auto_r1cs_ccs_with(n_rows, 100, 2)
    }

    /// Square-shape convenience with explicit security policy values.
    pub fn goldilocks_auto_r1cs_ccs_with(
        shape_size: usize,
        min_lambda: u32,
        safety_margin: u32,
    ) -> Result<Self, ParamsError> {
        Self::goldilocks_auto_rectangular_r1cs_ccs_with(shape_size, shape_size, min_lambda, safety_margin)
    }

    /// Square-shape convenience for a general CCS relation.
    pub fn goldilocks_auto_ccs_with(
        shape_size: usize,
        matrix_count: usize,
        poly_degree: u32,
        min_lambda: u32,
        safety_margin: u32,
    ) -> Result<Self, ParamsError> {
        Self::goldilocks_auto_rectangular_ccs_with(
            shape_size,
            shape_size,
            matrix_count,
            poly_degree,
            min_lambda,
            safety_margin,
        )
    }

    /// Auto-pick Appendix B.2 parameters for the implemented rectangular
    /// R1CS specialization (`t = 3`, `u = 2`).
    pub fn goldilocks_auto_rectangular_r1cs_ccs_with(
        row_count: usize,
        column_count: usize,
        min_lambda: u32,
        safety_margin: u32,
    ) -> Result<Self, ParamsError> {
        Self::goldilocks_auto_rectangular_ccs_with(row_count, column_count, 3, 2, min_lambda, safety_margin)
    }

    /// Auto-pick Appendix B.2 core parameters for the one-joint padded-row
    /// protocol on a rectangular CCS shape.
    ///
    /// The selected `lambda` charges the exact sum of:
    ///
    /// - the one joint SumCheck degree budget;
    /// - the paper alpha/gamma mixing degree; and
    /// - Appendix D.5's conservative coordinate-fork loss over `5^54`.
    pub fn goldilocks_auto_rectangular_ccs_with(
        row_count: usize,
        column_count: usize,
        matrix_count: usize,
        poly_degree: u32,
        min_lambda: u32,
        safety_margin: u32,
    ) -> Result<Self, ParamsError> {
        if matrix_count == 0 {
            return Err(ParamsError::Invalid("matrix_count must be > 0"));
        }
        if min_lambda == 0 {
            return Err(ParamsError::Invalid("min_lambda must be > 0"));
        }

        let mut p = Self::goldilocks_paper_b2();

        // Search λ downward while the implemented extension remains s=2.
        let mut lam = p.lambda.max(min_lambda);
        while lam >= min_lambda {
            p.lambda = lam;
            let summary = p.padded_row_security_summary_for_shape(
                row_count,
                column_count,
                matrix_count,
                poly_degree,
                goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
            )?;
            if summary.slack_bits >= 0 && summary.slack_bits as u32 >= safety_margin {
                return Ok(p);
            }
            lam = lam.saturating_sub(1);
        }

        let required = min_lambda.saturating_add(safety_margin);
        let mut probe = p;
        probe.lambda = min_lambda;
        let summary = probe.padded_row_security_summary_for_shape(
            row_count,
            column_count,
            matrix_count,
            poly_degree,
            goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
        )?;
        Err(ParamsError::InsufficientStatisticalSecurity {
            required,
            available: summary.security_bits,
        })
    }

    #[inline]
    fn bitlen_u128(x: u128) -> u32 {
        if x == 0 {
            0
        } else {
            128 - x.leading_zeros()
        }
    }

    /// Exact check for s=2: q^2 ≥ 2^λ · soundness_factor.
    /// Returns None if overflow prevents the check.
    fn s2_feasible_factor(&self, soundness_factor: u128) -> Option<bool> {
        let q2 = (self.q as u128).checked_mul(self.q as u128)?; // q^2 fits for 64-bit q
        let pow2 = 1u128.checked_shl(self.lambda)?; // None if λ ≥ 128
        let rhs = pow2.checked_mul(soundness_factor)?; // None if overflow
        Some(q2 >= rhs)
    }

    /// Compute the minimal extension degree using EXACT integer comparisons for s ∈ {1, 2}.
    /// This eliminates boundary-case optimism from bit-length ceiling approximations.
    /// Critical for soundness: bit-length methods can accept cases that actually need s=3!
    pub fn s_min(&self, ell: u32, d_sc: u32) -> u32 {
        let factor = (ell as u128) * (d_sc as u128);
        self.s_min_factor(factor)
    }

    /// Compute the minimal extension degree for a precomputed soundness factor.
    pub fn s_min_factor(&self, soundness_factor: u128) -> u32 {
        // Check s=1 exactly: q ≥ 2^λ · (ℓ·d_sc)
        if let Some(pow2) = 1u128.checked_shl(self.lambda) {
            if let Some(rhs) = pow2.checked_mul(soundness_factor) {
                if (self.q as u128) >= rhs {
                    return 1;
                }
            }
        }

        // Check s=2 exactly: q^2 ≥ 2^λ · soundness_factor.
        match self.s2_feasible_factor(soundness_factor) {
            Some(true) => 2,  // s=2 is sufficient
            Some(false) => 3, // s=2 insufficient, need s≥3
            None => 3,        // overflow on RHS ⇒ requires s ≥ 3
        }
    }

    /// Extension policy v1: support s=2 only. If s_min>2, return UnsupportedExtension{required=s_min}.
    /// When s_min=2, compute exact slack_bits by comparing q^2 against 2^λ·(ℓ·d_sc) directly.
    pub fn extension_check(&self, ell: u32, d_sc: u32) -> Result<ExtensionSummary, ParamsError> {
        let factor = (ell as u128)
            .checked_mul(d_sc as u128)
            .ok_or(ParamsError::UnsupportedExtension { required: 3 })?;
        self.extension_check_factor(factor)
    }

    /// Extension policy v1 for a precomputed soundness factor.
    pub fn extension_check_factor(&self, soundness_factor: u128) -> Result<ExtensionSummary, ParamsError> {
        let s_min = self.s_min_factor(soundness_factor);
        if s_min > 2 {
            return Err(ParamsError::UnsupportedExtension { required: s_min });
        }

        // Exact slack for s=2: compute floor(log₂(q²/(2^λ·factor))) without floating point.
        let q = self.q as u128;
        let q2 = q * q; // q^2 cannot overflow u128 for 64-bit q

        let rhs = 1u128
            .checked_shl(self.lambda)
            .and_then(|p| p.checked_mul(soundness_factor))
            .ok_or(ParamsError::UnsupportedExtension { required: 3 })?;

        let slack_bits = if q2 < rhs {
            // This case should not happen if s_min=2, but handle gracefully
            -1
        } else {
            // Compute floor(log₂(q²/rhs)) using bit lengths
            let mut slack = Self::bitlen_u128(q2) as i32 - Self::bitlen_u128(rhs) as i32;
            // Adjust if the division has no fractional part
            if let Some(shifted) = rhs.checked_shl(slack as u32) {
                if q2 < shifted {
                    slack -= 1;
                }
            }
            slack
        };

        Ok(ExtensionSummary {
            s_min,
            s_supported: 2,
            slack_bits,
        })
    }

    /// Validate the combined statistical error of the one-joint padded-row
    /// protocol for the maximum source counts allowed by this profile.
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

    /// Compute the exact production census without first requiring it to meet
    /// this parameter set's selected `lambda`.
    pub fn padded_row_security_summary_for_shape(
        &self,
        row_count: usize,
        column_count: usize,
        matrix_count: usize,
        poly_degree: u32,
        challenge_alphabet_size: u32,
    ) -> Result<PaddedRowSecuritySummary, ParamsError> {
        let (cube_variables, verifier_degree, sumcheck_factor, mixing_factor, field_factor) =
            self.padded_row_field_components_for_shape(row_count, column_count, matrix_count, poly_degree)?;
        let fresh_count = self.max_fresh_count_from_rlc_guard()?;
        let fork_factor = (fresh_count as u128)
            .checked_add(self.k_rho as u128)
            .and_then(|value| value.checked_add(1))
            .ok_or(ParamsError::ArithmeticOverflow("coordinate-fork numerator"))?;
        let challenge_set_cardinality =
            pow_u128_checked(challenge_alphabet_size as u128, self.d, "challenge-set cardinality")?;
        if challenge_set_cardinality == 0 {
            return Err(ParamsError::Invalid("challenge set must be nonempty"));
        }

        let field_size = BigUint::from(self.q).pow(self.s);
        let challenge_size = BigUint::from(challenge_set_cardinality);
        let total_numerator = BigUint::from(field_factor) * &challenge_size + BigUint::from(fork_factor) * &field_size;
        let total_denominator = &field_size * &challenge_size;
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

    /// Exact field-space numerator for the one-joint padded-row protocol.
    /// The corresponding error is `factor / q^s`.
    pub fn pi_ccs_padded_row_field_factor_for_shape(
        &self,
        row_count: usize,
        column_count: usize,
        matrix_count: usize,
        poly_degree: u32,
    ) -> Result<u128, ParamsError> {
        let (_, _, _, _, field_factor) =
            self.padded_row_field_components_for_shape(row_count, column_count, matrix_count, poly_degree)?;
        Ok(field_factor)
    }

    /// Maximum fresh CCS inputs K allowed by Definition 14's RLC guard:
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

    /// Return the one-joint padded-row field-space components.
    #[allow(clippy::type_complexity)]
    fn padded_row_field_components_for_shape(
        &self,
        row_count: usize,
        column_count: usize,
        matrix_count: usize,
        poly_degree: u32,
    ) -> Result<(u32, u32, u128, u128, u128), ParamsError> {
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
        let sumcheck_factor = (cube_variables as u128)
            .checked_mul(verifier_degree as u128)
            .ok_or(ParamsError::ArithmeticOverflow("joint SumCheck numerator"))?;

        let fresh_count = self.max_fresh_count_from_rlc_guard()? as u128;
        let running_count = self.k_rho as u128;
        let source_count = fresh_count
            .checked_add(running_count)
            .ok_or(ParamsError::ArithmeticOverflow("paper source count"))?;
        let carried_offset = fresh_count
            .checked_add(source_count)
            .ok_or(ParamsError::ArithmeticOverflow("carried-evaluation offset"))?;
        let joint_matrix_count = (matrix_count as u128)
            .checked_add(1)
            .ok_or(ParamsError::ArithmeticOverflow("identity-first matrix count"))?;
        let carried_count = running_count
            .checked_mul(joint_matrix_count)
            .and_then(|v| v.checked_mul(self.d as u128))
            .ok_or(ParamsError::ArithmeticOverflow("carried-coordinate count"))?;
        let alpha_dependent_degree = (cube_variables as u128)
            .checked_add(
                carried_offset
                    .checked_sub(1)
                    .ok_or(ParamsError::ArithmeticOverflow("paper mixing offset"))?,
            )
            .ok_or(ParamsError::ArithmeticOverflow("alpha-dependent mixing degree"))?;
        let joint_coefficient_count = carried_offset
            .checked_add(carried_count)
            .ok_or(ParamsError::ArithmeticOverflow("joint coefficient count"))?;
        let carried_mixing_degree = joint_coefficient_count
            .checked_sub(1)
            .ok_or(ParamsError::ArithmeticOverflow("carried mixing degree"))?;
        let mixing_factor = alpha_dependent_degree.max(carried_mixing_degree);
        let field_factor = sumcheck_factor
            .checked_add(mixing_factor)
            .ok_or(ParamsError::ArithmeticOverflow("padded-row field numerator"))?;

        Ok((
            cube_variables,
            verifier_degree,
            sumcheck_factor,
            mixing_factor,
            field_factor,
        ))
    }
}

// ---------- small helpers ----------

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

fn pow_u128_checked(base: u128, mut exp: u32, label: &'static str) -> Result<u128, ParamsError> {
    let mut result = 1u128;
    let mut power = base;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result
                .checked_mul(power)
                .ok_or(ParamsError::ArithmeticOverflow(label))?;
        }
        exp >>= 1;
        if exp > 0 {
            power = power
                .checked_mul(power)
                .ok_or(ParamsError::ArithmeticOverflow(label))?;
        }
    }
    Ok(result)
}

/// Largest `bits` such that `2^bits * numerator <= denominator`.
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

// Tests live in `crates/neo-params/tests/` (no in-file test modules).
