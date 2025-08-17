use neo_decomp::decomp_b;
use neo_fields::F;
use neo_poly::Polynomial;
use neo_ring::{RingElement};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

#[test]
fn toy_neo_commit_and_fold() {
    // Toy parameters
    let b: u64 = 2;
    let d: usize = 2;
    let m: usize = 1; // scalar vector

    // Public matrix as ring element: 5 + 2X
    let m_poly = Polynomial::new(vec![F::from_u64(5), F::from_u64(2)]);
    let m_ring = RingElement::new(m_poly, d);

    // Witness vectors z1 = 3, z2 = 1
    let z1_vec = vec![F::from_u64(3)];
    let z2_vec = vec![F::from_u64(1)];

    // Decompose
    let z1_mat: RowMajorMatrix<F> = decomp_b(&z1_vec, b, d);
    let z2_mat: RowMajorMatrix<F> = decomp_b(&z2_vec, b, d);

    // Embed to ring via coefficients
    let z1_coeffs: Vec<F> = (0..d).map(|row| z1_mat.get(row, 0).unwrap()).collect();
    let z1_ring = RingElement::from_coeffs(z1_coeffs, d);
    let z2_coeffs: Vec<F> = (0..d).map(|row| z2_mat.get(row, 0).unwrap()).collect();
    let z2_ring = RingElement::from_coeffs(z2_coeffs, d);

    // Commitments
    let c1 = z1_ring.commit(&m_ring);
    let c2 = z2_ring.commit(&m_ring);

    // Reconstruction check
    let recon_z1 = (0..d).rev().fold(F::ZERO, |acc, i| {
        acc * F::from_u64(b) + z1_mat.get(i, 0).unwrap()
    });
    assert_eq!(recon_z1, F::from_u64(3));

    // Folding with rho = 1
    let rho = F::ONE;
    let z_new_data: Vec<F> = (0..d)
        .flat_map(|row| vec![z1_mat.get(row, 0).unwrap() + rho * z2_mat.get(row, 0).unwrap()])
        .collect();
    let z_new_mat = RowMajorMatrix::new(z_new_data, m);
    let z_new_coeffs: Vec<F> = (0..d).map(|row| z_new_mat.get(row, 0).unwrap()).collect();
    let z_new_ring = RingElement::from_coeffs(z_new_coeffs, d);

    // Homomorphic commitment: c1 + c2 (since rho = 1)
    let c_new_hom: Vec<F> = c1.iter().zip(c2.iter()).map(|(&a, &b)| a + b).collect();

    // Direct commitment on folded witness
    let c_new_direct = z_new_ring.commit(&m_ring);

    // Verify
    let recon_new = (0..d).rev().fold(F::ZERO, |acc, i| {
        acc * F::from_u64(b) + z_new_mat.get(i, 0).unwrap()
    });
    assert_eq!(recon_new, F::from_u64(4));
    assert_eq!(c_new_hom, c_new_direct);
}
