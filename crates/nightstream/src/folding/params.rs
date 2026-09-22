//! Selected Nightstream Goldilocks k_rho = 16 parameters.
use neo_params::NeoParams;
#[derive(Clone, Debug)]
pub(crate) struct Params {
    inner: NeoParams,
}
impl Params {
    pub fn production() -> Self {
        Self {
            inner: NeoParams::nightstream_goldilocks_k16(),
        }
    }
    pub fn for_ccs_shape(
        rows: usize,
        columns: usize,
        matrix_count: usize,
        poly_degree: u32,
    ) -> Result<Self, neo_params::ParamsError> {
        let mut inner = NeoParams::nightstream_goldilocks_k16();
        let summary = inner.padded_row_security_summary_for_shape(
            rows,
            columns,
            matrix_count,
            poly_degree,
            neo_params::goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
        )?;
        inner.lambda = inner.lambda.min(summary.security_bits);
        if inner.lambda == 0 {
            return Err(neo_params::ParamsError::InsufficientStatisticalSecurity {
                required: 1,
                available: 0,
            });
        }
        Ok(Self { inner })
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
