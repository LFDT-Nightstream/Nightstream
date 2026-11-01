//! Selected Nightstream Goldilocks k_rho = 16 parameters.
use neo_params::NeoParams;
#[derive(Clone, Debug)]
pub(crate) struct Params {
    inner: NeoParams,
    /// The caller's statistical minimum, in bits.
    minimum_security_bits: u32,
    /// `-log2` of a conservative one-fold zero-query statistical error (not floored).
    fold_error_bits: f64,
}
impl Params {
    pub(crate) fn production() -> Self {
        Self {
            inner: NeoParams::nightstream_goldilocks_k16(),
            minimum_security_bits: 0,
            fold_error_bits: f64::INFINITY,
        }
    }
    pub(crate) fn for_ccs_shape(
        rows: usize,
        columns: usize,
        matrix_count: usize,
        poly_degree: u32,
        minimum_security_bits: u32,
    ) -> Result<Self, crate::Error> {
        if minimum_security_bits == 0 {
            return Err(neo_params::ParamsError::Invalid("minimum statistical security must be positive").into());
        }
        let params = Self::of_shape(rows, columns, matrix_count, poly_degree)?;
        let available = params.security_bits();
        if available < minimum_security_bits {
            return Err(crate::Error::InsufficientSecurity {
                required: minimum_security_bits,
                available,
            });
        }
        Ok(Self {
            minimum_security_bits,
            ..params
        })
    }
    /// The parameters of a CCS shape at the largest minimum it accepts.
    pub(crate) fn of_shape(
        rows: usize,
        columns: usize,
        matrix_count: usize,
        poly_degree: u32,
    ) -> Result<Self, neo_params::ParamsError> {
        let inner = NeoParams::nightstream_goldilocks_k16();
        let summary = inner.padded_row_security_summary_for_shape(
            rows,
            columns,
            matrix_count,
            poly_degree,
            neo_params::goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
        )?;
        // Upper bound for Lean `RandomOracleKnowledge.statisticalError 0`:
        // the generic field numerator at each of the 2ℓ + 1 + sources
        // transcript challenges
        // (`RandomOracleUniqueness.challenges_length`), and the Π_RLC sample
        // error `1/|C| + 2^-132` sources² times (`PiRlcSampler.distance_lt`).
        let sources = nightstream_fprime::PI_CCS_V1_2_SOURCE_COUNT as f64;
        let challenges = 2.0 * f64::from(summary.cube_variables) + 1.0 + sources;
        let test = (challenges * summary.field_factor as f64).log2() - f64::from(inner.s()) * (inner.q() as f64).log2();
        let sample = sources * sources * (1.0 / summary.challenge_set_cardinality as f64 + (-132.0f64).exp2());
        let fold_error_bits = -(test.exp2() + sample).log2();
        let available = inner
            .lambda()
            .min(summary.security_bits)
            .min(fold_error_bits.floor() as u32);
        Ok(Self {
            inner: inner.with_lambda(available)?,
            minimum_security_bits: available,
            fold_error_bits,
        })
    }
    /// The largest caller minimum these parameters accept.
    pub(crate) fn security_bits(&self) -> u32 {
        self.inner.lambda()
    }
    /// `-log2` of the error the compression argument may add after a final
    /// fold, so that the total stays within the caller's minimum.
    pub(crate) fn compression_security_bits(&self) -> Option<f64> {
        let remaining = (-f64::from(self.minimum_security_bits)).exp2() - (-self.fold_error_bits).exp2();
        (remaining > 0.0).then(|| -remaining.log2())
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
        neo_reductions::common::RotRing::goldilocks()
    }
}
