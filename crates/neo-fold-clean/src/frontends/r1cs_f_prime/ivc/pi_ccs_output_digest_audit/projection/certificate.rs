//! Read-only certificates retained after exact PiRLC projection replay.
//!
//! Owns: lossless copies of polynomial-evaluation and K-product trace data
//! needed by assurance exporters after the builder traces are no longer in
//! scope. Does not interpret or replace the source R1CS rows.
//!
//! Emits constraints: no.
//!
//! | Certificate | Retained evidence |
//! |---|---|
//! | polynomial evaluation | rows, allocation, coefficients, powers, outputs |
//! | K product | rows, allocation, retained outputs, exact product-sum LCs |

use std::ops::Range;

use neo_math::F;

use crate::engine::r1cs_circuit::builder::{Lc, PolynomialEvaluationTrace, ProductSumBatchTrace};

/// One exact linear combination as recorded by the source builder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiRlcYZcolLinearCombinationAudit {
    terms: Vec<(usize, F)>,
    constant: F,
}

impl PiRlcYZcolLinearCombinationAudit {
    pub fn terms(&self) -> &[(usize, F)] {
        &self.terms
    }

    pub fn constant(&self) -> F {
        self.constant
    }
}

/// One scaled product in an exact product-sum identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiRlcYZcolProductFactorAudit {
    left: PiRlcYZcolLinearCombinationAudit,
    right: PiRlcYZcolLinearCombinationAudit,
    coefficient: F,
}

impl PiRlcYZcolProductFactorAudit {
    pub fn left(&self) -> &PiRlcYZcolLinearCombinationAudit {
        &self.left
    }

    pub fn right(&self) -> &PiRlcYZcolLinearCombinationAudit {
        &self.right
    }

    pub fn coefficient(&self) -> F {
        self.coefficient
    }
}

/// One exact `result = sum(left_i * right_i * coefficient_i)` identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiRlcYZcolProductIdentityAudit {
    factors: Vec<PiRlcYZcolProductFactorAudit>,
    result: PiRlcYZcolLinearCombinationAudit,
}

impl PiRlcYZcolProductIdentityAudit {
    pub fn factors(&self) -> &[PiRlcYZcolProductFactorAudit] {
        &self.factors
    }

    pub fn result(&self) -> &PiRlcYZcolLinearCombinationAudit {
        &self.result
    }
}

/// Exact retained trace for one five-row extension-field product.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiRlcYZcolKMulAudit {
    rows: Range<usize>,
    allocated_columns: Vec<usize>,
    retained_columns: Vec<usize>,
    intermediate_columns: [usize; 3],
    output_columns: [usize; 2],
    identities: Vec<PiRlcYZcolProductIdentityAudit>,
}

impl PiRlcYZcolKMulAudit {
    pub fn rows(&self) -> Range<usize> {
        self.rows.clone()
    }

    pub fn allocated_columns(&self) -> &[usize] {
        &self.allocated_columns
    }

    pub fn retained_columns(&self) -> &[usize] {
        &self.retained_columns
    }

    pub fn intermediate_columns(&self) -> [usize; 3] {
        self.intermediate_columns
    }

    pub fn output_columns(&self) -> [usize; 2] {
        self.output_columns
    }

    pub fn identities(&self) -> &[PiRlcYZcolProductIdentityAudit] {
        &self.identities
    }
}

/// Exact retained trace for one extension-field polynomial evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiRlcYZcolPolynomialEvaluationAudit {
    rows: Range<usize>,
    allocated_columns: Vec<usize>,
    coefficient_columns: Vec<usize>,
    power_columns: Vec<[usize; 2]>,
    output_columns: [usize; 2],
}

impl PiRlcYZcolPolynomialEvaluationAudit {
    pub fn rows(&self) -> Range<usize> {
        self.rows.clone()
    }

    pub fn allocated_columns(&self) -> &[usize] {
        &self.allocated_columns
    }

    pub fn coefficient_columns(&self) -> &[usize] {
        &self.coefficient_columns
    }

    pub fn power_columns(&self) -> &[[usize; 2]] {
        &self.power_columns
    }

    pub fn output_columns(&self) -> [usize; 2] {
        self.output_columns
    }
}

pub(super) fn linear_combination(lc: &Lc) -> PiRlcYZcolLinearCombinationAudit {
    PiRlcYZcolLinearCombinationAudit {
        terms: lc.terms.clone(),
        constant: lc.constant,
    }
}

pub(super) fn polynomial(trace: &PolynomialEvaluationTrace) -> PiRlcYZcolPolynomialEvaluationAudit {
    PiRlcYZcolPolynomialEvaluationAudit {
        rows: trace.row_start..trace.row_end,
        allocated_columns: trace.allocated_columns.clone(),
        coefficient_columns: trace.coefficient_cols.clone(),
        power_columns: trace.power_cols.clone(),
        output_columns: trace.output_cols,
    }
}

pub(super) fn k_mul(trace: &ProductSumBatchTrace) -> PiRlcYZcolKMulAudit {
    let [p, q, r, out0, out1] = trace.allocated_columns.as_slice() else {
        unreachable!("K-product certificate is retained only after exact five-column replay")
    };
    PiRlcYZcolKMulAudit {
        rows: trace.row_start..trace.row_end,
        allocated_columns: trace.allocated_columns.clone(),
        retained_columns: trace.retained_columns.clone(),
        intermediate_columns: [*p, *q, *r],
        output_columns: [*out0, *out1],
        identities: trace
            .identities
            .iter()
            .map(|identity| PiRlcYZcolProductIdentityAudit {
                factors: identity
                    .factors
                    .iter()
                    .map(|factor| PiRlcYZcolProductFactorAudit {
                        left: linear_combination(&factor.left),
                        right: linear_combination(&factor.right),
                        coefficient: factor.coefficient,
                    })
                    .collect(),
                result: linear_combination(&identity.result),
            })
            .collect(),
    }
}
