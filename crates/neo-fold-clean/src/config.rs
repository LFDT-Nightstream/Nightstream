//! Audited parameter profile used by `neo-fold-clean`.
//!
//! Owns: choosing the default protocol parameters for this crate.
//! Does not own: deriving or validating those constants; that lives in
//! `neo-params`.

use crate::paper::params::Params;

/// Human-readable identifier for the strict production profile.
pub const PRODUCTION_PROFILE: &str = "superneo-appendix-b2-goldilocks-b2";
/// Identifier for R1CS tests/frontends that use the production core with a
/// shape-specific effective lambda, matching `neo-fold-prototype`.
pub const R1CS_PROFILE: &str = "superneo-appendix-b2-goldilocks-b2-r1cs-effective-lambda";

/// Base field modulus q = 2^64 - 2^32 + 1.
pub const Q: u64 = neo_params::goldilocks_paper_b2::Q;
/// Cyclotomic index eta.
pub const ETA: usize = neo_params::goldilocks_paper_b2::ETA;
/// Ring degree d = phi(eta).
pub const D: usize = neo_params::goldilocks_paper_b2::D;
/// Ajtai module rank kappa.
pub const KAPPA: u32 = neo_params::goldilocks_paper_b2::KAPPA;
/// Ajtai message length m.
pub const M: u64 = neo_params::goldilocks_paper_b2::M;
/// Small norm base b.
pub const B_BASE: u32 = neo_params::goldilocks_paper_b2::B_BASE;
/// Decomposition/RLC exponent: B = b^k_rho.
pub const K_RHO: u32 = neo_params::goldilocks_paper_b2::K_RHO;
/// Large norm bound B = b^k_rho.
pub const BIG_B: u64 = neo_params::goldilocks_paper_b2::B;
/// Strong sampling set expansion factor T.
pub const T: u32 = neo_params::goldilocks_paper_b2::T;
/// Extension degree s for the current optimized engine policy.
pub const EXTENSION_DEGREE: u32 = neo_params::goldilocks_paper_b2::EXTENSION_DEGREE;
/// Target security parameter lambda.
pub const LAMBDA: u32 = neo_params::goldilocks_paper_b2::LAMBDA;
/// Minimum full SuperNeo D.4 effective lambda accepted by the executable
/// `s = 2` R1CS profile.
///
/// With Appendix B.2's `s = 2`, the D.4 Schwartz-Zippel term dominates for
/// R1CS-derived CCS shapes. A 120-bit floor requires `s = 3`; the current
/// engine supports `s = 2`, so this profile rejects any shape below 100 bits
/// under the conservative max-`K` Appendix B.2 RLC guard.
pub const MIN_EFFECTIVE_LAMBDA: u32 = 100;
/// Extra slack required by the extension-field policy.
pub const EXTENSION_SAFETY_MARGIN_BITS: u32 = 2;

/// Return the production SuperNeo Appendix B.2 Goldilocks parameters.
pub fn production_params() -> Params {
    Params::production()
}

/// Return the Appendix B.2 core parameters for an R1CS-derived CCS shape.
///
/// This keeps q, eta, d, kappa, m, b, k_rho, B, T, and s fixed to the
/// production profile. The effective lambda may be lower than 125 when
/// the concrete CCS shape needs more room under the current `s = 2`
/// optimized-engine extension policy. That is the same split used in
/// `neo-fold-prototype`'s R1CS/IVC paths, but with a hard floor at
/// [`MIN_EFFECTIVE_LAMBDA`].
///
/// The size input is `max(ccs.n, ccs.m)`, not just the constraint count:
/// FE rounds are row-driven, while NC/witness checks can be width-driven.
pub fn r1cs_params(ccs_rows: usize, ccs_vars: usize) -> Result<Params, neo_params::ParamsError> {
    Params::for_r1cs_shape_with(
        ccs_rows.max(ccs_vars),
        MIN_EFFECTIVE_LAMBDA,
        EXTENSION_SAFETY_MARGIN_BITS,
    )
}
