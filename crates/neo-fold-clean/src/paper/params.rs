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
    /// Unmodified SuperNeo Goldilocks reference preset — Appendix B.2
    /// (`b = 2` row).
    ///
    /// This is the audited table value: λ = 125, b = 2, k_rho = 14,
    /// B = 2^14, T = 216, s = 2, kappa = 18. Its λ is not a combined
    /// executable security claim. Use a shape-specific constructor before
    /// proving or verification.
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
    /// an R1CS-derived CCS with the supplied row and column dimensions.
    ///
    /// Production policy: q, eta, d, kappa,
    /// m, b, k_rho, B, T, and s remain the Appendix B.2 values; λ is
    /// lowered only when the concrete sumcheck shape cannot satisfy the
    /// current `s = 2` extension policy at λ = 125.
    pub fn for_r1cs_shape(rows: usize, columns: usize) -> Result<Self, neo_params::ParamsError> {
        Self::for_r1cs_shape_with(rows, columns, 96, 2)
    }

    /// Same as [`Params::for_r1cs_shape`], with an explicit minimum
    /// effective λ and extension-policy safety margin.
    pub fn for_r1cs_shape_with(
        rows: usize,
        columns: usize,
        min_lambda: u32,
        safety_margin: u32,
    ) -> Result<Self, neo_params::ParamsError> {
        Ok(Self {
            inner: NeoParams::goldilocks_auto_rectangular_r1cs_ccs_with(rows, columns, min_lambda, safety_margin)?,
        })
    }

    /// Appendix B.2 Goldilocks core with shape-specific effective λ for a
    /// concrete CCS shape. `matrix_count` is SuperNeo's `t`; `poly_degree` is
    /// SuperNeo's `u`.
    pub fn for_ccs_shape_with(
        rows: usize,
        columns: usize,
        matrix_count: usize,
        poly_degree: u32,
        min_lambda: u32,
        safety_margin: u32,
    ) -> Result<Self, neo_params::ParamsError> {
        Ok(Self {
            inner: NeoParams::goldilocks_auto_rectangular_ccs_with(
                rows,
                columns,
                matrix_count,
                poly_degree,
                min_lambda,
                safety_margin,
            )?,
        })
    }

    /// Same as [`Params::for_ccs_shape_with`], with the default effective-λ
    /// floor and safety margin used by [`Params::for_r1cs_shape`].
    pub fn for_ccs_shape(
        rows: usize,
        columns: usize,
        matrix_count: usize,
        poly_degree: u32,
    ) -> Result<Self, neo_params::ParamsError> {
        Self::for_ccs_shape_with(rows, columns, matrix_count, poly_degree, 96, 2)
    }

    /// Test/probe escape hatch for wrapping a caller-built [`NeoParams`].
    ///
    /// This can lower cryptographic security parameters. Do not use in
    /// production proving or verification paths. Normal callers should
    /// use [`Params::production`] or [`Params::for_r1cs_shape`].
    pub fn test_only_from_neo_params(inner: NeoParams) -> Self {
        Self { inner }
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

    /// Maximum fresh CCS instances (`K`) allowed in one lifecycle step under
    /// this parameter profile.
    ///
    /// Definition 14 requires `(K + k)T(b - 1) < B`; the running side has
    /// `k = k_rho` after Π_DEC. We use the same cap even for transient
    /// empty-running steps so every proof stays inside the verifier's
    /// advertised parameter profile.
    pub fn max_fresh_count(&self) -> usize {
        let denom = (self.T() as u128) * (self.b().saturating_sub(1) as u128);
        if denom == 0 {
            return 0;
        }
        let max_total = (self.big_b() as u128).saturating_sub(1) / denom;
        max_total
            .saturating_sub(self.k_rho() as u128)
            .min(usize::MAX as u128) as usize
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

    /// Validate these selected parameters against the actual rectangular CCS
    /// shape and the combined field/fork statistical census.
    pub fn validate_ccs_shape(
        &self,
        rows: usize,
        columns: usize,
        matrix_count: usize,
        poly_degree: u32,
    ) -> Result<neo_params::PaddedRowSecuritySummary, neo_params::ParamsError> {
        self.inner.padded_row_security_check_for_shape(
            rows,
            columns,
            matrix_count,
            poly_degree,
            neo_params::goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
        )
    }

    /// Exact field-space numerator for the one-joint padded-row shape.
    pub fn ccs_padded_row_field_factor(
        &self,
        rows: usize,
        columns: usize,
        matrix_count: usize,
        poly_degree: u32,
    ) -> Result<u128, neo_params::ParamsError> {
        self.inner
            .pi_ccs_padded_row_field_factor_for_shape(rows, columns, matrix_count, poly_degree)
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

    /// Borrow the underlying `NeoParams` for `engine::*` calls and external
    /// prover backends that drive `neo_reductions` helpers directly.
    ///
    /// **Auditor**: the only legitimate uses are wiring the engine
    /// implementation and accelerator backends replicating it. Paper-layer
    /// logic must use the named accessors above.
    pub fn inner(&self) -> &NeoParams {
        &self.inner
    }

    /// Strong-sampling-set ring metadata for the audited preset
    /// (Definition 17 + Theorem 9). Forwarded for engine calls that need it.
    pub(crate) fn ring(&self) -> neo_reductions::common::RotRing {
        neo_reductions::common::RotRing::goldilocks()
    }
}
