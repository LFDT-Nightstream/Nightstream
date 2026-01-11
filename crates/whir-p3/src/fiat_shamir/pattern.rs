use p3_field::Field;

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Pattern {
    Sample = 0,
    Observe = 1,
    Hint = 2,
}

impl Pattern {
    #[must_use]
    pub fn as_field_element<F: Field>(self) -> F {
        F::from_u8(self as u8)
    }
}

/// Labels for items that are sampled.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Sample {
    InitialCombinationRandomness = 0,
    FoldingRandomnessSkip = 1,
    FoldingRandomness = 2,
    CombinationRandomness = 3,
    StirQueries = 4,
    FinalQueries = 5,
    PowQueries = 6,
    OodQuery = 7,
    Mock = 255,
}

impl Sample {
    #[must_use]
    pub fn as_field_element<F: Field>(self) -> F {
        F::from_u8(self as u8)
    }
}

/// Labels for items that are observed.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Observe {
    MerkleDigest = 0,
    OodAnswers = 1,
    SumcheckPoly = 2,
    SumcheckPolySkip = 3,
    StirAnswers = 4,
    FinalCoeffs = 5,
    PowNonce = 6,
    Mock = 255,
}

impl Observe {
    #[must_use]
    pub fn as_field_element<F: Field>(self) -> F {
        F::from_u8(self as u8)
    }
}

/// Labels for items that are hints.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Hint {
    StirQueries = 0,
    StirAnswers = 1,
    MerkleProof = 2,
    DeferredWeightEvaluations = 3,
    Mock = 255,
}

impl Hint {
    #[must_use]
    pub fn as_field_element<F: Field>(self) -> F {
        F::from_u8(self as u8)
    }
}
