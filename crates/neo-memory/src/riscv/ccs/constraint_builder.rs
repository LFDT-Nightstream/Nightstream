use neo_ccs::matrix::Mat;
use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::relations::CcsStructure;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

#[derive(Clone, Debug)]
pub(super) struct Constraint<Ff: PrimeCharacteristicRing + Copy> {
    pub condition_col: usize,
    pub negate_condition: bool,
    pub additional_condition_cols: Vec<usize>,
    pub b_terms: Vec<(usize, Ff)>,
    pub c_terms: Vec<(usize, Ff)>,
}

impl<Ff: PrimeCharacteristicRing + Copy> Constraint<Ff> {
    pub fn eq_const(condition_col: usize, const_one_col: usize, left: usize, c: u64) -> Self {
        Self {
            condition_col,
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(left, Ff::ONE), (const_one_col, -Ff::from_u64(c))],
            c_terms: Vec::new(),
        }
    }

    pub fn zero(condition_col: usize, col: usize) -> Self {
        Self {
            condition_col,
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(col, Ff::ONE)],
            c_terms: Vec::new(),
        }
    }

    pub fn terms(condition_col: usize, negate_condition: bool, b_terms: Vec<(usize, Ff)>) -> Self {
        Self {
            condition_col,
            negate_condition,
            additional_condition_cols: Vec::new(),
            b_terms,
            c_terms: Vec::new(),
        }
    }

    pub fn terms_or(condition_cols: &[usize], negate_condition: bool, b_terms: Vec<(usize, Ff)>) -> Self {
        assert!(!condition_cols.is_empty(), "need at least one condition column");
        Self {
            condition_col: condition_cols[0],
            negate_condition,
            additional_condition_cols: condition_cols[1..].to_vec(),
            b_terms,
            c_terms: Vec::new(),
        }
    }

    pub fn mul(left: usize, right: usize, out: usize) -> Self {
        Self {
            condition_col: left,
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(right, Ff::ONE)],
            c_terms: vec![(out, Ff::ONE)],
        }
    }
}

pub(super) fn build_identity_first_r1cs_ccs(
    constraints: &[Constraint<F>],
    m: usize,
    const_one_col: usize,
) -> Result<CcsStructure<F>, String> {
    let n = m;
    if constraints.len() > n {
        return Err(format!(
            "RV32 B1 CCS: too many constraints ({}) for square CCS with m=n={}",
            constraints.len(),
            n
        ));
    }

    let mut a_data = vec![F::ZERO; n * m];
    let mut b_data = vec![F::ZERO; n * m];
    let mut c_data = vec![F::ZERO; n * m];

    for (row, c) in constraints.iter().enumerate() {
        if c.negate_condition {
            a_data[row * m + const_one_col] = F::ONE;
            a_data[row * m + c.condition_col] += -F::ONE;
            for &col in &c.additional_condition_cols {
                a_data[row * m + col] += -F::ONE;
            }
        } else {
            a_data[row * m + c.condition_col] += F::ONE;
            for &col in &c.additional_condition_cols {
                a_data[row * m + col] += F::ONE;
            }
        }

        for &(col, coeff) in &c.b_terms {
            b_data[row * m + col] += coeff;
        }
        for &(col, coeff) in &c.c_terms {
            c_data[row * m + col] += coeff;
        }
    }

    let i_n = Mat::identity(n);
    let a = Mat::from_row_major(n, m, a_data);
    let b = Mat::from_row_major(n, m, b_data);
    let c = Mat::from_row_major(n, m, c_data);

    let f = SparsePoly::new(
        4,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![0, 1, 1, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 0, 1],
            },
        ],
    );

    CcsStructure::new(vec![i_n, a, b, c], f).map_err(|e| format!("RV32 B1 CCS: invalid structure: {e:?}"))
}

