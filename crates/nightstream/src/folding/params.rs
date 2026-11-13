//! The parameters of the profile Nightstream P64, k_rho = 16.
use neo_params::NeoParams;
#[derive(Clone, Debug)]
pub(crate) struct Params {
    inner: NeoParams,
}
impl Params {
    pub(crate) fn production() -> Self {
        Self {
            inner: NeoParams::nightstream_p64_k16(),
        }
    }
    pub(crate) fn for_ccs_shape(
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
        let inner = NeoParams::nightstream_p64_k16();
        let alphabet = neo_params::nightstream_p64_k16::CHALLENGE_ALPHABET.len() as u32;
        let summary =
            inner.padded_row_security_summary_for_shape(rows, columns, matrix_count, poly_degree, alphabet)?;
        // Lean `RefreshKnowledge.statisticalError 0`, the larger fold error,
        // since either fold can run under either relation: the Π_CCS test
        // error at each of the 2ℓ + 1 + sources transcript challenges, and
        // the Π_RLC sample error `1/|C| + 2^-173` sources² times
        // (`PiRlcSampler.distance_lt`).
        let sources = f64::from(1 + inner.k_rho());
        let challenges = 2.0 * f64::from(summary.cube_variables) + 1.0 + sources;
        let test = (challenges * summary.field_factor as f64).log2() - f64::from(inner.s()) * (inner.q() as f64).log2();
        let challenge_set = f64::from(alphabet).powi(inner.d() as i32);
        let sample = sources * sources * (1.0 / challenge_set + (-173.0f64).exp2());
        let fold_error_bits = -(test.exp2() + sample).log2();
        let available = inner
            .lambda()
            .min(summary.security_bits)
            .min(fold_error_bits.floor() as u32);
        if available < minimum_security_bits {
            return Err(neo_params::ParamsError::InsufficientStatisticalSecurity {
                required: minimum_security_bits,
                available,
            });
        }
        let inner = inner.with_lambda(available)?;
        Ok(Self { inner })
    }
    pub(crate) fn b(&self) -> u32 {
        self.inner.b()
    }
    pub(crate) fn k_rho(&self) -> u32 {
        self.inner.k_rho()
    }
    pub(crate) fn big_b(&self) -> u64 {
        self.inner.B()
    }
    pub(crate) fn kappa(&self) -> u32 {
        self.inner.kappa()
    }
    pub(crate) fn inner(&self) -> &NeoParams {
        &self.inner
    }
    pub(crate) fn ring(&self) -> neo_reductions::common::RotRing {
        neo_reductions::common::RotRing::profile()
    }
}
