//! Audited parameter profile used by `neo-fold-clean`.
//!
//! Owns: choosing the default protocol parameters for this crate.
//! Does not own: deriving or validating those constants; that lives in
//! `neo-params`.

use crate::paper::params::Params;

/// Human-readable identifier for the strict production profile.
pub const PRODUCTION_PROFILE: &str = "superneo-appendix-b2-goldilocks-b2";
/// Identifier for R1CS tests/frontends that use the production core with a
/// shape-specific effective lambda.
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
/// Appendix B.2's reference lambda value.
///
/// This value is not the executable profile's combined statistical security.
/// Shape-specific constructors select the header-bound value below it.
pub const LAMBDA: u32 = neo_params::goldilocks_paper_b2::LAMBDA;
/// Minimum combined statistical security accepted by the executable
/// rectangular `s = 2` profile.
///
/// The census adds both SumCheck errors, the rectangular mixing error, and
/// Appendix D.5's conservative coordinate-fork error. This is a per-protocol
/// invocation floor; it is not a lifetime or computational-hardness claim.
pub const MIN_EFFECTIVE_LAMBDA: u32 = 100;
/// Extra slack required by the extension-field policy.
pub const EXTENSION_SAFETY_MARGIN_BITS: u32 = 2;
/// Declared end-to-end target for a maximum-geometry Nebula chain under the
/// conservative, pre-review projection census.
pub const NEBULA_END_TO_END_SECURITY_BITS: u32 = 64;
/// `log2(q_H)` bound on adversarial random-oracle queries used by the
/// Fiat-Shamir reduction for the maximum-chain security statement.
pub const NEBULA_MAX_FS_QUERY_BITS: u32 = 16;

/// Return the unmodified SuperNeo Appendix B.2 reference parameters.
///
/// This constructor is for table and serialization comparisons. Executable
/// callers must use [`r1cs_params`] or [`ccs_params`] so the concrete shape
/// receives the combined rectangular security check.
pub fn production_params() -> Params {
    Params::production()
}

/// Return the Appendix B.2 core parameters for an R1CS-derived CCS shape.
///
/// This keeps q, eta, d, kappa, m, b, k_rho, B, T, and s fixed to the
/// production profile. The effective lambda may be lower than 125 when
/// the concrete CCS shape needs more room under the current `s = 2`
/// optimized-engine extension policy, with a hard floor at
/// [`MIN_EFFECTIVE_LAMBDA`].
///
/// FE rounds use `ccs_rows`; NC rounds use `ccs_vars`. Keep both dimensions
/// separate so a rectangular relation is not charged as a square one.
pub fn r1cs_params(ccs_rows: usize, ccs_vars: usize) -> Result<Params, neo_params::ParamsError> {
    Params::for_r1cs_shape_with(ccs_rows, ccs_vars, MIN_EFFECTIVE_LAMBDA, EXTENSION_SAFETY_MARGIN_BITS)
}

/// Return Appendix B.2 core params for a concrete CCS shape.
///
/// This charges the rectangular field/fork census using the actual row and
/// column dimensions, matrix count `t`, and polynomial degree `u`. R1CS callers should keep using
/// [`r1cs_params`], which is the `(t=3, u=2)` specialization.
pub fn ccs_params(
    ccs_rows: usize,
    ccs_vars: usize,
    matrix_count: usize,
    poly_degree: u32,
) -> Result<Params, neo_params::ParamsError> {
    Params::for_ccs_shape_with(
        ccs_rows,
        ccs_vars,
        matrix_count,
        poly_degree,
        MIN_EFFECTIVE_LAMBDA,
        EXTENSION_SAFETY_MARGIN_BITS,
    )
}
