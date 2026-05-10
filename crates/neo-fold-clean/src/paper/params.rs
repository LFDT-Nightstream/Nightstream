//! Definition 14: Global Reduction Parameters.
//!
//! Owns: the typed bundle of parameters every reduction reads. Wraps
//! `neo-params::NeoParams` so an auditor sees paper symbols.
//!
//! Does not own: parameter selection (lives in `neo-params`), or the strong
//! sampling set itself (lives in `paper/sampling.rs`).
//!
//! ## Naming note
//!
//! `NeoParams` already uses paper-near names internally. The thin wrapper
//! exists so the paper layer reads `pp.b()`, `pp.k_rho()`, etc., without
//! anyone having to remember which fields are public on `NeoParams`.

use neo_params::NeoParams;

/// Definition 14 globals — opaque handle around a validated `NeoParams`.
///
/// Construction goes through named constructors. `production` is the
/// SuperNeo Appendix B.2 Goldilocks preset. `for_r1cs_shape` keeps the
/// same Appendix B.2 core parameters and derives the effective λ needed
/// by the current `s = 2` optimized engine for a concrete R1CS shape.
#[derive(Clone, Debug)]
pub struct Params {
    inner: NeoParams,
}

impl Params {
    /// Production SuperNeo Goldilocks preset — Appendix B.2 (`b = 2` row).
    ///
    /// This is the audited reference preset: λ = 125, b = 2, k_rho = 14,
    /// B = 2^14, T = 216, s = 2, kappa = 18.
    pub fn production() -> Self {
        Self {
            inner: NeoParams::goldilocks_paper_b2(),
        }
    }

    /// Backwards-readable alias for auditors comparing against
    /// `neo_params::NeoParams::goldilocks_paper_b2`.
    pub fn goldilocks_paper_b2() -> Self {
        Self::production()
    }

    /// Appendix B.2 Goldilocks core with shape-specific effective λ for
    /// an R1CS-derived CCS of `n_rows` constraints.
    ///
    /// Mirrors the production path in `neo-fold-next`: q, eta, d, kappa,
    /// m, b, k_rho, B, T, and s remain the Appendix B.2 values; λ is
    /// lowered only when the concrete sumcheck shape cannot satisfy the
    /// current `s = 2` extension policy at λ = 125.
    pub fn for_r1cs_shape(n_rows: usize) -> Result<Self, neo_params::ParamsError> {
        Self::for_r1cs_shape_with(n_rows, 96, 2)
    }

    /// Same as [`Params::for_r1cs_shape`], with an explicit minimum
    /// effective λ and extension-policy safety margin.
    pub fn for_r1cs_shape_with(
        n_rows: usize,
        min_lambda: u32,
        safety_margin: u32,
    ) -> Result<Self, neo_params::ParamsError> {
        Ok(Self {
            inner: NeoParams::goldilocks_auto_r1cs_ccs_with(n_rows, min_lambda, safety_margin)?,
        })
    }

    /// Norm bound `b` (Definition 1).
    pub fn b(&self) -> u32 {
        self.inner.b
    }

    /// Decomposition/RLC exponent `k_rho`; B = b^k_rho.
    pub fn k_rho(&self) -> u32 {
        self.inner.k_rho
    }

    /// Big norm B = b^k_rho (Definition 1). Pre-computed in `NeoParams.B`.
    pub fn big_b(&self) -> u64 {
        self.inner.B
    }

    /// Theorem 9 expansion factor T. Pre-computed in `NeoParams.T`.
    #[allow(non_snake_case)]
    pub fn T(&self) -> u32 {
        self.inner.T
    }

    /// Ajtai commitment width κ (Definition 18).
    pub fn kappa(&self) -> u32 {
        self.inner.kappa
    }

    /// Base field modulus q.
    pub fn q(&self) -> u64 {
        self.inner.q
    }

    /// Cyclotomic index eta.
    pub fn eta(&self) -> u32 {
        self.inner.eta
    }

    /// Ring degree d = phi(eta).
    pub fn d(&self) -> u32 {
        self.inner.d
    }

    /// Constraint count m.
    pub fn m(&self) -> u64 {
        self.inner.m
    }

    /// Extension degree s.
    pub fn extension_degree(&self) -> u32 {
        self.inner.s
    }

    /// Target security parameter λ.
    pub fn lambda(&self) -> u32 {
        self.inner.lambda
    }

    /// True exactly for the SuperNeo Appendix B.2 Goldilocks production preset.
    pub fn is_production(&self) -> bool {
        self.inner.is_goldilocks_paper_b2()
    }

    /// True when all Appendix B.2 core parameters match, ignoring the
    /// shape-specific effective λ.
    pub fn has_production_core(&self) -> bool {
        self.inner.has_goldilocks_paper_b2_core()
    }

    /// Borrow the underlying `NeoParams` for `engine::*` calls.
    ///
    /// **Auditor**: the only legitimate use of this is wiring the engine
    /// implementation. Paper-layer logic must use the named accessors above.
    pub(crate) fn inner(&self) -> &NeoParams {
        &self.inner
    }

    /// Strong-sampling-set ring metadata for the audited preset
    /// (Definition 17 + Theorem 9). Forwarded for engine calls that need it.
    pub(crate) fn ring(&self) -> neo_reductions::common::RotRing {
        neo_reductions::common::RotRing::goldilocks()
    }
}
