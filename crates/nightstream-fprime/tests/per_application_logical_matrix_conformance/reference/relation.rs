//! Independent decoder for the Lean-authored final CCS polynomial.

use serde::Deserialize;

use super::{array, exact_array, field, word, Field, Result, MATRIX_COUNT};

const EXPECTED_TERM_COUNT: usize = 74;
const EXPECTED_DEGREE_BOUND: usize = 9;

#[derive(Clone, Debug)]
struct Term {
    coefficient: Field,
    exponents: [usize; MATRIX_COUNT],
}

#[derive(Clone, Debug)]
pub struct Relation {
    terms: Vec<Term>,
}

#[derive(Deserialize)]
struct RawSealed(
    u64,
    RawPackage,
    serde::de::IgnoredAny,
    serde::de::IgnoredAny,
    serde::de::IgnoredAny,
    serde::de::IgnoredAny,
    u64,
);

#[derive(Deserialize)]
struct RawPackage(
    u64,
    serde::de::IgnoredAny,
    serde::de::IgnoredAny,
    serde::de::IgnoredAny,
    serde_json::Value,
    serde::de::IgnoredAny,
    serde::de::IgnoredAny,
    serde::de::IgnoredAny,
    serde::de::IgnoredAny,
    serde::de::IgnoredAny,
    serde::de::IgnoredAny,
    serde::de::IgnoredAny,
    serde::de::IgnoredAny,
    serde::de::IgnoredAny,
);

impl Relation {
    pub fn decode(sealed_bytes: &[u8]) -> Result<Self> {
        if sealed_bytes.last() != Some(&b'\n') {
            return Err("sealed package is not newline terminated".into());
        }
        let RawSealed(
            outer_schema,
            RawPackage(inner_schema, _, _, _, raw, _, _, _, _, _, _, _, _, _),
            _,
            _,
            _,
            _,
            logical_public,
        ) = serde_json::from_slice(sealed_bytes).map_err(|error| format!("independent relation decode: {error}"))?;
        if outer_schema != 6 || inner_schema != 8 || logical_public != 270 {
            return Err("unexpected sealed relation envelope".into());
        }
        let fields = exact_array(&raw, 6, "CCS relation")?;
        if word(&fields[0], "CCS row count")? != 6_377_559
            || word(&fields[1], "CCS column count")? != 264_627_433
            || word(&fields[2], "CCS cube variables")? != 28
            || array(&fields[3], "CCS matrix sources")?
                .iter()
                .map(|value| word(value, "CCS matrix source"))
                .collect::<Result<Vec<_>>>()?
                != (0..MATRIX_COUNT).collect::<Vec<_>>()
            || word(&fields[4], "CCS degree bound")? != EXPECTED_DEGREE_BOUND
        {
            return Err("unexpected final CCS relation shape".into());
        }
        let raw_terms = array(&fields[5], "CCS polynomial terms")?;
        if raw_terms.len() != EXPECTED_TERM_COUNT {
            return Err(format!(
                "CCS polynomial has {} terms, expected {EXPECTED_TERM_COUNT}",
                raw_terms.len()
            ));
        }
        let terms = raw_terms
            .iter()
            .map(|value| {
                let fields = exact_array(value, 2, "CCS polynomial term")?;
                let coefficient = field(&fields[0], "CCS polynomial coefficient")?;
                let exponent_values = exact_array(&fields[1], MATRIX_COUNT, "CCS polynomial exponents")?;
                let exponents: [usize; MATRIX_COUNT] = exponent_values
                    .iter()
                    .map(|value| word(value, "CCS polynomial exponent"))
                    .collect::<Result<Vec<_>>>()?
                    .try_into()
                    .map_err(|_| "CCS polynomial exponent count".to_string())?;
                let degree = exponents.iter().try_fold(0usize, |sum, exponent| {
                    sum.checked_add(*exponent)
                        .ok_or_else(|| "CCS polynomial degree overflow".to_string())
                })?;
                if degree >= EXPECTED_DEGREE_BOUND {
                    return Err("CCS polynomial term exceeds the degree bound".into());
                }
                Ok(Term { coefficient, exponents })
            })
            .collect::<Result<Vec<_>>>()?;
        let exact_bound = terms
            .iter()
            .map(|term| term.exponents.iter().sum::<usize>())
            .max()
            .and_then(|degree| degree.checked_add(1));
        if exact_bound != Some(EXPECTED_DEGREE_BOUND) {
            return Err("CCS polynomial does not attain its strict degree bound".into());
        }
        Ok(Self { terms })
    }

    pub fn evaluate(&self, matrix_values: &[Field; MATRIX_COUNT]) -> Field {
        self.terms.iter().fold(Field::ZERO, |sum, term| {
            let product = matrix_values
                .iter()
                .zip(term.exponents)
                .fold(term.coefficient, |product, (value, exponent)| {
                    product * pow(*value, exponent)
                });
            sum + product
        })
    }

    pub fn term_count(&self) -> usize {
        self.terms.len()
    }
}

fn pow(value: Field, exponent: usize) -> Field {
    (0..exponent).fold(Field::ONE, |product, _| product * value)
}
