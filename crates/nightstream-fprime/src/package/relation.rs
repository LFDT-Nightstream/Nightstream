use serde::{Deserialize, Serialize};

use super::{
    canonical_field, word_to_usize, Layout, PackageError, MAX_JOINT_DOMAIN, PI_CCS_V1_1_MATRIX_COUNT,
    PI_CCS_V1_1_ROUND_COUNT,
};

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawCcsRelation(u64, u64, u64, Vec<u64>, u64, Vec<RawPolynomialTerm>);

#[derive(Debug, Deserialize, Serialize)]
struct RawPolynomialTerm(u64, Vec<u64>);

/// Physical matrix selected by one Lean-owned logical CCS slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CcsMatrixSource {
    Bit,
    GeneralSelector,
    A,
    B,
    C,
    SboxInput,
    CenteredUnit,
    EvalSelector,
    Class0,
    Class1,
    Class2,
    Class3,
    Class4,
    Zero,
}

/// One sparse term of the Lean-owned CCS constraint polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackagePolynomialTerm {
    coefficient: u64,
    exponents: Vec<usize>,
}

impl PackagePolynomialTerm {
    pub fn coefficient(&self) -> u64 {
        self.coefficient
    }

    pub fn exponents(&self) -> &[usize] {
        &self.exponents
    }
}

/// Exact logical CCS relation decoded from the identity-bound package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageCcsRelation {
    row_count: usize,
    column_count: usize,
    cube_variables: usize,
    matrix_sources: Vec<CcsMatrixSource>,
    degree_bound: usize,
    terms: Vec<PackagePolynomialTerm>,
}

impl PackageCcsRelation {
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn column_count(&self) -> usize {
        self.column_count
    }

    pub fn cube_variables(&self) -> usize {
        self.cube_variables
    }

    pub fn matrix_sources(&self) -> &[CcsMatrixSource] {
        &self.matrix_sources
    }

    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    pub fn terms(&self) -> &[PackagePolynomialTerm] {
        &self.terms
    }
}

pub(super) fn validate(raw: RawCcsRelation, layout: &Layout, schema: u64) -> Result<PackageCcsRelation, PackageError> {
    let RawCcsRelation(rows, columns, cube_variables, matrix_sources, degree_bound, terms) = raw;
    let row_count = word_to_usize(rows, "CCS relation row count")?;
    let column_count = word_to_usize(columns, "CCS relation column count")?;
    let cube_variables = word_to_usize(cube_variables, "CCS relation cube variables")?;
    let degree_bound = word_to_usize(degree_bound, "CCS relation degree bound")?;

    match schema {
        7 => {
            if row_count != layout.row_count {
                return Err(PackageError::Invalid("CCS relation row count"));
            }
            if column_count != layout.total_column_count {
                return Err(PackageError::Invalid("CCS relation column count"));
            }
        }
        8 => {
            let carrier_width = column_count
                .checked_add(53)
                .map(|rounded| rounded / 54)
                .and_then(|blocks| blocks.checked_mul(54))
                .ok_or(PackageError::Invalid("CCS relation carrier width"))?;
            if row_count > MAX_JOINT_DOMAIN || carrier_width > MAX_JOINT_DOMAIN {
                return Err(PackageError::Invalid("CCS relation 2^28 domain"));
            }
        }
        _ => return Err(PackageError::Invalid("schema version")),
    }
    if cube_variables != PI_CCS_V1_1_ROUND_COUNT {
        return Err(PackageError::Invalid("CCS relation cube variables"));
    }
    if matrix_sources.len() != PI_CCS_V1_1_MATRIX_COUNT {
        return Err(PackageError::Invalid("CCS relation matrix count"));
    }
    if degree_bound == 0 || terms.is_empty() {
        return Err(PackageError::Invalid("CCS relation polynomial shape"));
    }

    let matrix_sources = matrix_sources
        .into_iter()
        .map(|source| match source {
            0 => Ok(CcsMatrixSource::Bit),
            1 => Ok(CcsMatrixSource::GeneralSelector),
            2 => Ok(CcsMatrixSource::A),
            3 => Ok(CcsMatrixSource::B),
            4 => Ok(CcsMatrixSource::C),
            5 => Ok(CcsMatrixSource::SboxInput),
            6 => Ok(CcsMatrixSource::CenteredUnit),
            7 => Ok(CcsMatrixSource::EvalSelector),
            8 => Ok(CcsMatrixSource::Class0),
            9 => Ok(CcsMatrixSource::Class1),
            10 => Ok(CcsMatrixSource::Class2),
            11 => Ok(CcsMatrixSource::Class3),
            12 => Ok(CcsMatrixSource::Class4),
            13 => Ok(CcsMatrixSource::Zero),
            _ => Err(PackageError::Invalid("CCS relation matrix source")),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let terms = terms
        .into_iter()
        .map(|RawPolynomialTerm(coefficient, exponents)| {
            canonical_field(coefficient, "CCS polynomial coefficient")?;
            if exponents.len() != PI_CCS_V1_1_MATRIX_COUNT {
                return Err(PackageError::Invalid("CCS polynomial exponent count"));
            }
            let exponents = exponents
                .into_iter()
                .map(|exponent| word_to_usize(exponent, "CCS polynomial exponent"))
                .collect::<Result<Vec<_>, _>>()?;
            let total_degree = exponents.iter().try_fold(0usize, |total, exponent| {
                total
                    .checked_add(*exponent)
                    .ok_or(PackageError::Invalid("CCS polynomial degree overflow"))
            })?;
            if total_degree > degree_bound {
                return Err(PackageError::Invalid("CCS polynomial degree bound"));
            }
            Ok(PackagePolynomialTerm { coefficient, exponents })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PackageCcsRelation {
        row_count,
        column_count,
        cube_variables,
        matrix_sources,
        degree_bound,
        terms,
    })
}
