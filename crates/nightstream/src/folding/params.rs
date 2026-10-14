//! Selected Nightstream Goldilocks k_rho = 16 parameters.
use neo_params::NeoParams;
#[derive(Clone, Debug)]
pub(crate) struct Params {
    inner: NeoParams,
    /// The caller's statistical minimum, in bits.
    minimum_security_bits: u32,
    /// `-log2` of one fold's exact census error (not floored).
    fold_error_bits: f64,
}
impl Params {
    pub fn production() -> Self {
        Self {
            inner: NeoParams::nightstream_goldilocks_k16(),
            minimum_security_bits: 0,
            fold_error_bits: f64::INFINITY,
        }
    }
    pub fn for_ccs_shape(
        rows: usize,
        columns: usize,
        matrix_count: usize,
        poly_degree: u32,
        minimum_security_bits: u32,
    ) -> Result<Self, neo_params::ParamsError> {
        if minimum_security_bits == 0 {
            return Err(neo_params::ParamsError::Invalid(
                "minimum statistical security must be positive",
            ));
        }
        let mut inner = NeoParams::nightstream_goldilocks_k16();
        let summary = inner.padded_row_security_summary_for_shape(
            rows,
            columns,
            matrix_count,
            poly_degree,
            neo_params::goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
        )?;
        inner.lambda = inner.lambda.min(summary.security_bits);
        if inner.lambda < minimum_security_bits {
            return Err(neo_params::ParamsError::InsufficientStatisticalSecurity {
                required: minimum_security_bits,
                available: inner.lambda,
            });
        }
        // field_factor / q^s + fork_factor / |C|, the error `security_bits` floors.
        let field = (summary.field_factor as f64).log2() - f64::from(inner.s) * (inner.q as f64).log2();
        let fork = (summary.fork_factor as f64).log2() - (summary.challenge_set_cardinality as f64).log2();
        let fold_error_bits = -(field.exp2() + fork.exp2()).log2();
        Ok(Self {
            inner,
            minimum_security_bits,
            fold_error_bits,
        })
    }
    /// `-log2` of the error the compression argument may add after a final
    /// fold, so that the total stays within the caller's minimum.
    pub fn compression_security_bits(&self) -> Option<f64> {
        let remaining = (-f64::from(self.minimum_security_bits)).exp2() - (-self.fold_error_bits).exp2();
        (remaining > 0.0).then(|| -remaining.log2())
    }
    pub fn b(&self) -> u32 {
        self.inner.b
    }
    pub fn k_rho(&self) -> u32 {
        self.inner.k_rho
    }
    pub fn big_b(&self) -> u64 {
        self.inner.B
    }
    #[allow(non_snake_case)]
    pub fn T(&self) -> u32 {
        self.inner.T
    }
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
    pub fn kappa(&self) -> u32 {
        self.inner.kappa
    }
    pub fn inner(&self) -> &NeoParams {
        &self.inner
    }
    pub(crate) fn ring(&self) -> neo_reductions::common::RotRing {
        neo_reductions::common::RotRing::goldilocks()
    }
}
