//! Scalar polynomial-combination certificates for simple R1CS redundancy.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{Problem, Row, Selection, Source, GOLDILOCKS_MODULUS};

pub const SCALAR_CERTIFICATE_SCHEMA: &str = "nightstream/r1cs-scalar-redundancy-certificate/v1";

type Monomial = Vec<usize>;
type Polynomial = BTreeMap<Monomial, u64>;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct ScalarSupport {
    pub source_index: usize,
    pub coefficient: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct ScalarRowCertificate {
    pub candidate_source_index: usize,
    pub support: Vec<ScalarSupport>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct ScalarCertificate {
    pub schema: String,
    pub source: Source,
    pub selection: Selection,
    pub rows: Vec<ScalarRowCertificate>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CertificateError(String);

impl CertificateError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for CertificateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for CertificateError {}

/// Derive one scalar-span certificate for every selected candidate row.
///
/// `None` means that this deliberately small certificate grammar cannot prove
/// the implication. It does not mean that the candidate is necessary.
pub fn derive_scalar_certificate(
    problem: &Problem,
    selection: &Selection,
) -> Result<Option<ScalarCertificate>, CertificateError> {
    let partition = problem
        .partition(selection)
        .map_err(|error| CertificateError::new(error.to_string()))?;
    let retained = partition
        .retained
        .iter()
        .map(|&(problem_index, row)| polynomial(row).map(|polynomial| (problem_index, polynomial)))
        .collect::<Result<Vec<_>, _>>()?;
    let basis = build_basis(&retained)?;
    let mut rows = Vec::with_capacity(partition.removed.len());
    for &(_, candidate) in &partition.removed {
        let Some(combination) = solve(&basis, polynomial(candidate)?) else {
            return Ok(None);
        };
        let support = combination
            .into_iter()
            .filter(|(_, coefficient)| *coefficient != 0)
            .map(|(problem_index, coefficient)| ScalarSupport {
                source_index: problem.rows[problem_index].source_index,
                coefficient: coefficient.to_string(),
            })
            .collect();
        rows.push(ScalarRowCertificate {
            candidate_source_index: candidate.source_index,
            support,
        });
    }
    let certificate = ScalarCertificate {
        schema: SCALAR_CERTIFICATE_SCHEMA.to_owned(),
        source: problem.source.clone(),
        selection: selection.clone(),
        rows,
    };
    validate_scalar_certificate(problem, &certificate)?;
    Ok(Some(certificate))
}

/// Recompute all artifact references and polynomial equalities.
pub fn validate_scalar_certificate(problem: &Problem, certificate: &ScalarCertificate) -> Result<(), CertificateError> {
    problem
        .validate()
        .map_err(|error| CertificateError::new(error.to_string()))?;
    if certificate.schema != SCALAR_CERTIFICATE_SCHEMA {
        return Err(CertificateError::new(format!(
            "unsupported scalar certificate schema {:?}",
            certificate.schema
        )));
    }
    if certificate.source != problem.source {
        return Err(CertificateError::new(
            "certificate source does not equal the problem source",
        ));
    }
    let partition = problem
        .partition(&certificate.selection)
        .map_err(|error| CertificateError::new(error.to_string()))?;
    let expected_candidates = partition
        .removed
        .iter()
        .map(|(_, row)| row.source_index)
        .collect::<Vec<_>>();
    let actual_candidates = certificate
        .rows
        .iter()
        .map(|row| row.candidate_source_index)
        .collect::<Vec<_>>();
    if actual_candidates != expected_candidates {
        return Err(CertificateError::new(
            "certificate candidates do not exactly match the selection",
        ));
    }

    let retained_by_source = partition
        .retained
        .iter()
        .map(|(_, row)| (row.source_index, *row))
        .collect::<BTreeMap<_, _>>();
    let candidates_by_source = partition
        .removed
        .iter()
        .map(|(_, row)| (row.source_index, *row))
        .collect::<BTreeMap<_, _>>();
    for row_certificate in &certificate.rows {
        let candidate = candidates_by_source
            .get(&row_certificate.candidate_source_index)
            .ok_or_else(|| CertificateError::new("certificate references an unknown candidate row"))?;
        let mut prior_support = None;
        let mut combination = Polynomial::new();
        for support in &row_certificate.support {
            if prior_support.is_some_and(|prior| support.source_index <= prior) {
                return Err(CertificateError::new(
                    "certificate support rows must be strictly ordered and unique",
                ));
            }
            prior_support = Some(support.source_index);
            let support_row = retained_by_source
                .get(&support.source_index)
                .ok_or_else(|| {
                    CertificateError::new(format!(
                        "certificate support row {} is not retained",
                        support.source_index
                    ))
                })?;
            let coefficient = parse_nonzero_coefficient(&support.coefficient)?;
            add_scaled(&mut combination, &polynomial(support_row)?, coefficient, false);
        }
        if combination != polynomial(candidate)? {
            return Err(CertificateError::new(format!(
                "scalar polynomial identity failed for source row {}",
                row_certificate.candidate_source_index
            )));
        }
    }
    Ok(())
}

#[derive(Clone)]
struct BasisEntry {
    polynomial: Polynomial,
    combination: BTreeMap<usize, u64>,
}

fn build_basis(retained: &[(usize, Polynomial)]) -> Result<BTreeMap<Monomial, BasisEntry>, CertificateError> {
    let mut basis = BTreeMap::<Monomial, BasisEntry>::new();
    for (problem_index, polynomial) in retained {
        let mut work = polynomial.clone();
        let mut combination = BTreeMap::from([(*problem_index, 1)]);
        reduce_representation(&mut work, &mut combination, &basis);
        let Some((pivot, pivot_coefficient)) = work
            .first_key_value()
            .map(|(monomial, coefficient)| (monomial.clone(), *coefficient))
        else {
            continue;
        };
        let inverse = inverse(pivot_coefficient)?;
        scale(&mut work, inverse);
        scale(&mut combination, inverse);
        basis.insert(
            pivot,
            BasisEntry {
                polynomial: work,
                combination,
            },
        );
    }
    Ok(basis)
}

fn reduce_representation(
    polynomial: &mut Polynomial,
    combination: &mut BTreeMap<usize, u64>,
    basis: &BTreeMap<Monomial, BasisEntry>,
) {
    while let Some((monomial, coefficient, entry)) = polynomial.iter().find_map(|(monomial, coefficient)| {
        basis
            .get(monomial)
            .map(|entry| (monomial.clone(), *coefficient, entry))
    }) {
        debug_assert_eq!(entry.polynomial.get(&monomial), Some(&1));
        add_scaled(polynomial, &entry.polynomial, coefficient, true);
        add_scaled(combination, &entry.combination, coefficient, true);
    }
}

fn solve(basis: &BTreeMap<Monomial, BasisEntry>, mut target: Polynomial) -> Option<BTreeMap<usize, u64>> {
    let mut combination = BTreeMap::new();
    while let Some((monomial, coefficient, entry)) = target.iter().find_map(|(monomial, coefficient)| {
        basis
            .get(monomial)
            .map(|entry| (monomial.clone(), *coefficient, entry))
    }) {
        debug_assert_eq!(entry.polynomial.get(&monomial), Some(&1));
        add_scaled(&mut target, &entry.polynomial, coefficient, true);
        add_scaled(&mut combination, &entry.combination, coefficient, false);
    }
    target.is_empty().then_some(combination)
}

fn polynomial(row: &Row) -> Result<Polynomial, CertificateError> {
    let mut polynomial = Polynomial::new();
    for left in &row.a {
        for right in &row.b {
            let mut monomial = vec![left.column, right.column];
            monomial.sort_unstable();
            let coefficient = multiply(
                parse_coefficient(&left.coefficient)?,
                parse_coefficient(&right.coefficient)?,
            );
            add_coefficient(&mut polynomial, monomial, coefficient, false);
        }
    }
    for output in &row.c {
        add_coefficient(
            &mut polynomial,
            vec![output.column],
            parse_coefficient(&output.coefficient)?,
            true,
        );
    }
    Ok(polynomial)
}

fn parse_nonzero_coefficient(value: &str) -> Result<u64, CertificateError> {
    let coefficient = parse_coefficient(value)?;
    if coefficient == 0 {
        return Err(CertificateError::new("certificate coefficient must be nonzero"));
    }
    Ok(coefficient)
}

fn parse_coefficient(value: &str) -> Result<u64, CertificateError> {
    let modulus = modulus();
    let coefficient = value
        .parse::<u64>()
        .map_err(|_| CertificateError::new(format!("invalid field coefficient {value:?}")))?;
    if coefficient >= modulus || value != coefficient.to_string() {
        return Err(CertificateError::new(format!(
            "noncanonical field coefficient {value:?}"
        )));
    }
    Ok(coefficient)
}

fn add_scaled<K: Ord + Clone>(
    destination: &mut BTreeMap<K, u64>,
    source: &BTreeMap<K, u64>,
    factor: u64,
    subtract: bool,
) {
    for (key, coefficient) in source {
        let scaled = multiply(factor, *coefficient);
        let prior = destination.get(key).copied().unwrap_or(0);
        let next = if subtract {
            subtract_field(prior, scaled)
        } else {
            add(prior, scaled)
        };
        if next == 0 {
            destination.remove(key);
        } else {
            destination.insert(key.clone(), next);
        }
    }
}

fn add_coefficient(polynomial: &mut Polynomial, monomial: Monomial, coefficient: u64, subtract: bool) {
    let prior = polynomial.get(&monomial).copied().unwrap_or(0);
    let next = if subtract {
        subtract_field(prior, coefficient)
    } else {
        add(prior, coefficient)
    };
    if next == 0 {
        polynomial.remove(&monomial);
    } else {
        polynomial.insert(monomial, next);
    }
}

fn scale<K: Ord + Clone>(values: &mut BTreeMap<K, u64>, factor: u64) {
    for coefficient in values.values_mut() {
        *coefficient = multiply(*coefficient, factor);
    }
    values.retain(|_, coefficient| *coefficient != 0);
}

fn modulus() -> u64 {
    GOLDILOCKS_MODULUS
        .parse::<u64>()
        .expect("fixed Goldilocks modulus fits in u64")
}

fn add(left: u64, right: u64) -> u64 {
    ((u128::from(left) + u128::from(right)) % u128::from(modulus())) as u64
}

fn subtract_field(left: u64, right: u64) -> u64 {
    if left >= right {
        left - right
    } else {
        modulus() - (right - left)
    }
}

fn multiply(left: u64, right: u64) -> u64 {
    (u128::from(left) * u128::from(right) % u128::from(modulus())) as u64
}

fn inverse(value: u64) -> Result<u64, CertificateError> {
    if value == 0 {
        return Err(CertificateError::new("cannot invert zero"));
    }
    let mut exponent = modulus() - 2;
    let mut base = value;
    let mut result = 1u64;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = multiply(result, base);
        }
        base = multiply(base, base);
        exponent >>= 1;
    }
    if multiply(value, result) != 1 {
        return Err(CertificateError::new(
            "field inverse check failed for the Goldilocks modulus",
        ));
    }
    Ok(result)
}
