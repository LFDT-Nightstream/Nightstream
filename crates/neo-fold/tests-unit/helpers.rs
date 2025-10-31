use neo_ccs::{Mat, McsInstance, McsWitness, r1cs_to_ccs, SModuleHomomorphism};
use neo_ajtai::Commitment;
use neo_params::NeoParams;
use neo_math::{F, D};
use p3_field::PrimeCharacteristicRing;

pub struct DummyS;

impl SModuleHomomorphism<F, Commitment> for DummyS {
    fn commit(&self, z: &Mat<F>) -> Commitment {
        Commitment::zeros(z.rows(), 4)
    }
    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows();
        let cols = m_in.min(z.cols());
        let mut result = Mat::zero(rows, cols, F::ZERO);
        for r in 0..rows {
            for c in 0..cols {
                result[(r, c)] = z[(r, c)];
            }
        }
        result
    }
}

pub fn create_test_ccs() -> neo_ccs::CcsStructure<F> {
    // n=4, m=5, same as smoke tests
    let rows = 4usize;
    let cols = 5usize;
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];
    a[0 * cols + 4] = F::ONE;    // out
    a[0 * cols + 1] = -F::ONE;   // -x1
    a[0 * cols + 2] = -F::ONE;   // -x2
    for row in 0..rows { b[row * cols + 0] = F::ONE; } // const on B
    r1cs_to_ccs(
        Mat::from_row_major(rows, cols, a),
        Mat::from_row_major(rows, cols, b),
        Mat::from_row_major(rows, cols, c),
    )
}

pub fn mk_mcs(params: &NeoParams, z_full: Vec<F>, m_in: usize, l: &DummyS)
  -> (McsInstance<Commitment, F>, McsWitness<F>)
{
    let d = D;
    let m = z_full.len();
    let z_digits = neo_ajtai::decomp_b(&z_full, params.b, d, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for col in 0..m {
        for row in 0..d {
            row_major[row * m + col] = z_digits[col * d + row];
        }
    }
    let z_mat = Mat::from_row_major(d, m, row_major);
    let c = l.commit(&z_mat);
    (
        McsInstance { c: c.clone(), x: z_full[..m_in].to_vec(), m_in },
        McsWitness { w: z_full[m_in..].to_vec(), Z: z_mat },
    )
}
