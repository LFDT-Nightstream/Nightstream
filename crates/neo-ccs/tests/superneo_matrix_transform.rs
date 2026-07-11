use neo_ccs::{sparse::CcsMatrix, tensor_point, CcsStructure, Mat, SparsePoly};
use neo_math::{cf, cf_inv, ct, Rq, D, F, K};
use p3_field::PrimeCharacteristicRing;

fn deterministic_vec(len: usize, seed: u64) -> Vec<F> {
    let mut x = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push(F::from_u64(x));
    }
    out
}

fn field_mul_rows(matrix: &CcsMatrix<F>, x: &[F], nrows: usize) -> Vec<F> {
    let mut out = vec![F::ZERO; nrows];
    matrix.add_mul_into(x, &mut out, nrows);
    out
}

fn ring_row_eval(matrix: &CcsMatrix<F>, row: usize, z_ring: &[Rq], ncols: usize) -> Rq {
    let n_ring_cols = ncols / D;
    let mut row_blocks = vec![[F::ZERO; D]; n_ring_cols];

    match matrix {
        CcsMatrix::Identity { .. } => {
            panic!("identity sentinel should not appear after superneo transform")
        }
        CcsMatrix::Csc(m) => {
            for c in 0..m.ncols {
                let s = m.col_ptr[c];
                let e = m.col_ptr[c + 1];
                let block = c / D;
                let local = c % D;
                for k in s..e {
                    if m.row_idx[k] == row {
                        row_blocks[block][local] += m.vals[k];
                    }
                }
            }
        }
        CcsMatrix::CscWithSeededPhi81 { csc, blocks } => {
            for c in 0..csc.ncols {
                let block = c / D;
                let local = c % D;
                for k in csc.col_ptr[c]..csc.col_ptr[c + 1] {
                    if csc.row_idx[k] == row {
                        row_blocks[block][local] += csc.vals[k];
                    }
                }
                for seeded in blocks {
                    row_blocks[block][local] += seeded.entry::<F>(row, c);
                }
            }
        }
    }

    let mut acc = Rq::zero();
    for (blk, coeffs) in row_blocks.iter().enumerate() {
        let mr = Rq(*coeffs);
        acc = acc + mr.mul(&z_ring[blk]);
    }
    acc
}

fn ring_ct_row_eval(matrix: &CcsMatrix<F>, row: usize, z_ring: &[Rq], ncols: usize) -> F {
    ct(&ring_row_eval(matrix, row, z_ring, ncols))
}

fn to_ring_blocks(z: &[F]) -> Vec<Rq> {
    assert!(z.len().is_multiple_of(D));
    let mut out = Vec::with_capacity(z.len() / D);
    for chunk in z.chunks_exact(D) {
        let mut coeffs = [F::ZERO; D];
        coeffs.copy_from_slice(chunk);
        out.push(Rq(coeffs));
    }
    out
}

fn reduce_k_mod_phi_81(coeffs: &mut [K; 2 * D - 1]) {
    for i in (D..(2 * D - 1)).rev() {
        let t = coeffs[i];
        coeffs[i] = K::ZERO;
        coeffs[i - D] -= t;
        let idx_27 = i - 27;
        if idx_27 < D {
            coeffs[idx_27] -= t;
        } else {
            coeffs[idx_27 - D] += t;
            coeffs[idx_27 - 27] += t;
        }
    }
}

fn ring_mul_f_by_k(lhs: &Rq, rhs: &[K; D]) -> [K; D] {
    let mut tmp = [K::ZERO; 2 * D - 1];
    for (i, &lhs_coeff) in lhs.0.iter().enumerate() {
        for (j, &rhs_coeff) in rhs.iter().enumerate() {
            tmp[i + j] += K::from(lhs_coeff) * rhs_coeff;
        }
    }
    reduce_k_mod_phi_81(&mut tmp);
    let mut out = [K::ZERO; D];
    out.copy_from_slice(&tmp[..D]);
    out
}

fn ring_eval_at_r(matrix: &CcsMatrix<F>, z_ring: &[Rq], nrows: usize, ncols: usize, r: &[K]) -> [K; D] {
    let weights = tensor_point(r);
    let mut out = [K::ZERO; D];
    for (row, &weight) in weights.iter().enumerate().take(nrows) {
        let row_eval = ring_row_eval(matrix, row, z_ring, ncols);
        for (rho, &coeff) in row_eval.0.iter().enumerate() {
            out[rho] += K::from(coeff) * weight;
        }
    }
    out
}

fn add_ring_k_assign(lhs: &mut [K; D], rhs: &[K; D]) {
    for (left, right) in lhs.iter_mut().zip(rhs.iter()) {
        *left += *right;
    }
}

fn ring_challenge_linear_combination(challenges: &[Rq], witnesses: &[Vec<Rq>]) -> Vec<Rq> {
    assert_eq!(challenges.len(), witnesses.len());
    let block_count = witnesses.first().expect("witnesses").len();
    let mut out = vec![Rq::zero(); block_count];
    for (challenge, witness) in challenges.iter().zip(witnesses.iter()) {
        assert_eq!(witness.len(), block_count);
        for (acc, block) in out.iter_mut().zip(witness.iter()) {
            *acc = *acc + challenge.mul(block);
        }
    }
    out
}

#[test]
fn superneo_coefficient_embedding_roundtrip_packs_d_entries_per_ring_slot() {
    let z = deterministic_vec(2 * D, 0x1234_5678_9abc_def0);

    for (block_idx, chunk) in z.chunks_exact(D).enumerate() {
        let mut block = [F::ZERO; D];
        block.copy_from_slice(chunk);
        let ring = cf_inv(block);

        assert_eq!(cf(ring), block, "coefficient roundtrip mismatch at block {block_idx}");
        assert_eq!(
            ring.field_coeffs(),
            block.to_vec(),
            "ring slot must expose exactly the packed D field entries at block {block_idx}"
        );
    }
}

#[test]
fn superneo_transform_identity_matrix_recovers_z_via_ct() {
    let m = 2 * D;
    let n = m;
    let s = CcsStructure::new(vec![Mat::identity(n)], SparsePoly::new(1, vec![])).expect("valid CCS");
    let s_bar = s.transform_matrices_superneo().expect("superneo transform");

    let z = deterministic_vec(m, 0x1111_2222_3333_4444);
    let z_ring = to_ring_blocks(&z);

    let matrix_bar = &s_bar.matrices[0];
    for (r, zr) in z.iter().enumerate().take(n) {
        let got = ring_ct_row_eval(matrix_bar, r, &z_ring, m);
        assert_eq!(*zr, got, "identity row mismatch at row={r}");
    }
}

#[test]
fn superneo_evaluation_homomorphism_holds_for_ring_challenge_linear_combinations() {
    let n = 4usize;
    let m = 2 * D;
    let mut mat = Mat::zero(n, m, F::ZERO);
    for r in 0..n {
        for c in (r + 1..m).step_by(19) {
            mat[(r, c)] = F::from_u64((7 * r as u64 + 11) * (3 * c as u64 + 13));
        }
    }

    let s = CcsStructure::new(vec![mat], SparsePoly::new(1, vec![])).expect("valid CCS");
    let s_bar = s.transform_matrices_superneo().expect("superneo transform");
    let matrix_bar = &s_bar.matrices[0];
    let z1 = to_ring_blocks(&deterministic_vec(m, 0xaaaa_bbbb_cccc_dddd));
    let z2 = to_ring_blocks(&deterministic_vec(m, 0xdddd_cccc_bbbb_aaaa));
    let rho1 = to_ring_blocks(&deterministic_vec(D, 0x0102_0304_0506_0708))
        .into_iter()
        .next()
        .expect("rho1 block");
    let rho2 = to_ring_blocks(&deterministic_vec(D, 0x1112_1314_1516_1718))
        .into_iter()
        .next()
        .expect("rho2 block");
    let r = [K::from(F::from_u64(5)), K::from(F::from_u64(9))];

    let y1 = ring_eval_at_r(matrix_bar, &z1, n, m, &r);
    let y2 = ring_eval_at_r(matrix_bar, &z2, n, m, &r);
    let combined_z = ring_challenge_linear_combination(&[rho1, rho2], &[z1, z2]);
    let actual = ring_eval_at_r(matrix_bar, &combined_z, n, m, &r);
    let mut expected = ring_mul_f_by_k(&rho1, &y1);
    add_ring_k_assign(&mut expected, &ring_mul_f_by_k(&rho2, &y2));

    assert_eq!(
        actual, expected,
        "SuperNeo evaluation must commute with verifier-chosen ring-linear combinations"
    );
}

#[test]
fn superneo_transform_general_matrix_matches_field_matrix_vector_product() {
    let n = 4usize;
    let m = 2 * D;
    let mut mat = Mat::zero(n, m, F::ZERO);
    for r in 0..n {
        for c in (r..m).step_by(17) {
            // Keep matrix sparse but nontrivial across both D-blocks.
            mat[(r, c)] = F::from_u64((r as u64 + 3) * (c as u64 + 5));
        }
    }

    let s = CcsStructure::new(vec![mat], SparsePoly::new(1, vec![])).expect("valid CCS");
    let s_bar = s.transform_matrices_superneo().expect("superneo transform");

    let z = deterministic_vec(m, 0x9999_aaaa_bbbb_cccc);
    let z_ring = to_ring_blocks(&z);

    let y_field = field_mul_rows(&s.matrices[0], &z, n);
    let matrix_bar = &s_bar.matrices[0];
    for (r, y) in y_field.iter().enumerate().take(n) {
        let got = ring_ct_row_eval(matrix_bar, r, &z_ring, m);
        assert_eq!(*y, got, "row mismatch at row={r}");
    }
}
