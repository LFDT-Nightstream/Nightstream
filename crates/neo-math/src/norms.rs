use thiserror::Error;

/// Errors for normative requirement violations and math failures.
#[derive(Debug, Error)]
#[allow(dead_code)]
pub(crate) enum NeoMathError {
    #[error("MUST violated: {0}")]
    Must(&'static str),
    #[error("SHOULD violated (strict mode): {0}")]
    Should(&'static str),
    #[error("dimension mismatch")]
    Dim,
}

/// Behavior for MUST/SHOULD checks at runtime.
#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)]
pub(crate) struct Norms {
    /// If true, **SHOULD** violations become errors; otherwise they are no-ops.
    pub strict_should: bool,
}

#[allow(dead_code)]
impl Norms {
    #[inline]
    pub fn must(self, ok: bool, msg: &'static str) -> Result<(), NeoMathError> {
        if !ok {
            return Err(NeoMathError::Must(msg));
        }
        Ok(())
    }
    #[inline]
    pub fn should(self, ok: bool, msg: &'static str) -> Result<(), NeoMathError> {
        if !ok && self.strict_should {
            return Err(NeoMathError::Should(msg));
        }
        Ok(())
    }
}
