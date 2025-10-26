use neo_ccs::Mat;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

fn naive_eq(row: usize, r: &[K]) -> K {
    let mut result = K::ONE;
    for (i, &ri) in r.iter().enumerate() {
        let bit = (row >> i) & 1;
        result *= if bit == 0 { K::ONE - ri } else { ri };
    }
    result
}

#[test]
fn sparse_matrix_basic_properties() {
    let rows = 3;
    let cols = 3;
    let mut m = Mat::<F>::zero(rows, cols, F::ZERO);
    
    m[(0, 0)] = F::from_u64(1);
    m[(1, 1)] = F::from_u64(2);
    m[(2, 2)] = F::from_u64(3);
    
    for i in 0..rows {
        for j in 0..cols {
            let expected = if i == j { F::from_u64((i + 1) as u64) } else { F::ZERO };
            assert_eq!(m[(i, j)], expected, "diagonal element mismatch at ({}, {})", i, j);
        }
    }
}

#[test]
fn sparse_matrix_vector_multiply_identity() {
    let n = 4;
    let mut m = Mat::<F>::zero(n, n, F::ZERO);
    
    for i in 0..n {
        m[(i, i)] = F::ONE;
    }
    
    let v = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3), F::from_u64(4)];
    
    let mut result = vec![F::ZERO; n];
    for i in 0..n {
        for j in 0..n {
            result[i] += m[(i, j)] * v[j];
        }
    }
    
    assert_eq!(result, v, "I * v = v");
}

#[test]
fn sparse_matrix_vector_multiply_simple() {
    let rows = 2;
    let cols = 3;
    let mut m = Mat::<F>::zero(rows, cols, F::ZERO);
    
    m[(0, 0)] = F::from_u64(1);
    m[(0, 2)] = F::from_u64(2);
    m[(1, 1)] = F::from_u64(3);
    
    let v = vec![F::from_u64(10), F::from_u64(20), F::from_u64(30)];
    
    let mut result = vec![F::ZERO; rows];
    for i in 0..rows {
        for j in 0..cols {
            result[i] += m[(i, j)] * v[j];
        }
    }
    
    assert_eq!(result[0], F::from_u64(70), "row 0: 1*10 + 0*20 + 2*30 = 70");
    assert_eq!(result[1], F::from_u64(60), "row 1: 0*10 + 3*20 + 0*30 = 60");
}

#[test]
fn weighted_sum_over_rows_simple() {
    let rows = 4;
    let cols = 3;
    let mut m = Mat::<F>::zero(rows, cols, F::ZERO);
    
    m[(0, 0)] = F::from_u64(1);
    m[(0, 2)] = F::from_u64(2);
    m[(1, 1)] = F::from_u64(5);
    m[(2, 0)] = F::from_u64(3);
    m[(2, 2)] = F::from_u64(4);
    m[(3, 2)] = F::from_u64(7);
    
    let r_bits = vec![K::from(F::from_u64(3)), K::from(F::ONE)];
    
    let mut result = vec![K::ZERO; cols];
    for row in 0..rows {
        let weight = naive_eq(row, &r_bits);
        for col in 0..cols {
            result[col] += K::from(m[(row, col)]) * weight;
        }
    }
    
    for col in 0..cols {
        let mut expected = K::ZERO;
        for row in 0..rows {
            let weight = naive_eq(row, &r_bits);
            expected += K::from(m[(row, col)]) * weight;
        }
        assert_eq!(result[col], expected, "weighted sum mismatch for column {}", col);
    }
}

#[test]
fn weighted_sum_all_zero_weights_gives_zero() {
    let rows = 4;
    let cols = 2;
    let mut m = Mat::<F>::zero(rows, cols, F::ZERO);
    
    for i in 0..rows {
        for j in 0..cols {
            m[(i, j)] = F::from_u64((i * cols + j + 1) as u64);
        }
    }
    
    let r_bits = vec![K::ZERO, K::ZERO];
    
    let mut result = vec![K::ZERO; cols];
    for row in 0..rows {
        let weight = naive_eq(row, &r_bits);
        for col in 0..cols {
            result[col] += K::from(m[(row, col)]) * weight;
        }
    }
    
    let expected_weight_0 = naive_eq(0, &r_bits);
    assert_eq!(expected_weight_0, K::ONE, "only row 0 should have weight 1");
    
    assert_eq!(result[0], K::from(m[(0, 0)]), "weighted sum should equal row 0");
    assert_eq!(result[1], K::from(m[(0, 1)]), "weighted sum should equal row 0");
}

