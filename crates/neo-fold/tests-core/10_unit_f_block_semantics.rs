//! Test F-block semantics: initial sum must be eq-weighted rowwise f(M_j·z),
//! NOT f applied to eq-weighted row averages.
//!
//! This test verifies the paper-correct computation:
//! F_block = Σ_row χ_{β_r}(row) · f( (M_0·z)[row], (M_1·z)[row], ..., (M_t·z)[row] )
//!
//! and ensures we don't regress to the incorrect form:
//! F_wrong = f( Σ_row χ_{β_r}(row)·(M_0·z)[row], ..., Σ_row χ_{β_r}(row)·(M_t·z)[row] )

use neo_fold::pi_ccs::{context, precompute, transcript};
use neo_fold::pi_ccs::eq_weights::{HalfTableEq, RowWeight};
use neo_fold::pi_ccs::sparse_matrix::to_csr;
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_ccs::{
    CcsStructure, Mat, McsInstance, McsWitness, SparsePoly, Term, SModuleHomomorphism,
};
use neo_params::NeoParams;
use neo_ajtai::Commitment;
use neo_math::{F, K, D};
use p3_field::PrimeCharacteristicRing;

/// Dummy S-module with a trivial commitment used in other tests too.
struct DummyS;

impl SModuleHomomorphism<F, Commitment> for DummyS {
    fn commit(&self, z: &Mat<F>) -> Commitment {
        let d = z.rows();
        Commitment::zeros(d, 4)
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

#[test]
#[allow(non_snake_case)]
fn f_block_initial_sum_matches_rowwise_sum() {
    // Non-linear f(a,b,c) = a*b + c to ensure non-commutation.
    let n: usize = 4;
    let m: usize = 4;
    let t: usize = 3;

    // Three CCS matrices M0, M1, M2 with varied per-row coefficients.
    let mut m0 = Mat::zero(n, m, F::ZERO);
    m0[(0, 1)] = F::ONE;             // row0:  a
    m0[(1, 1)] = F::from_u64(2);     // row1: 2a
    m0[(3, 1)] = -F::ONE;            // row3: -a

    let mut m1 = Mat::zero(n, m, F::ZERO);
    m1[(0, 2)] = F::ONE;             // row0:  b
    m1[(1, 2)] = -F::ONE;            // row1: -b
    m1[(2, 2)] = F::from_u64(3);     // row2: 3b
    m1[(3, 2)] = F::ONE;             // row3:  b

    let mut m2 = Mat::zero(n, m, F::ZERO);
    m2[(0, 3)] = F::ONE;             // row0:  c
    m2[(2, 3)] = F::ONE;             // row2:  c
    m2[(3, 3)] = -F::from_u64(2);    // row3: -2c

    let f = SparsePoly::new(
        t,
        vec![
            Term { coeff: F::ONE, exps: vec![1, 1, 0] }, // a*b
            Term { coeff: F::ONE, exps: vec![0, 0, 1] }, // + c
        ],
    );

    let s = CcsStructure {
        n,
        m,
        matrices: vec![m0, m1, m2],
        f,
    };

    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let d = D;

    // Witness z = [1 (const), a, b, c]
    let a = F::from_u64(2);
    let b = F::from_u64(3);
    let c = F::from_u64(5);
    let z_full = vec![F::ONE, a, b, c];
    let m_in = 1;

    // Z = Decomp_b(z) in row-major (d × m)
    let z_digits = neo_ajtai::decomp_b(&z_full, params.b, d, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for col in 0..m {
        for row in 0..d {
            row_major[row * m + col] = z_digits[col * d + row];
        }
    }
    let Z = Mat::from_row_major(d, m, row_major);

    // MCS instance & witness (DummyS makes c = L(Z) trivial)
    let l = DummyS;
    let cmt = l.commit(&Z);
    let mcs_inst = McsInstance {
        c: cmt,
        x: vec![z_full[0]],
        m_in,
    };
    let mcs_wit = McsWitness {
        w: z_full[m_in..].to_vec(),
        Z,
    };

    // Transcript & challenges (deterministic)
    let dims = context::build_dims_and_policy(&params, &s).expect("dims");
    let mut tr = Poseidon2Transcript::new(b"test/pi-ccs/f-block");

    transcript::bind_header_and_instances(
        &mut tr,
        &params,
        &s,
        &[mcs_inst.clone()],
        dims.ell,
        dims.d_sc,
        0,
    )
    .unwrap();

    transcript::bind_me_inputs(&mut tr, &[]).unwrap();

    let ch = transcript::sample_challenges(&mut tr, dims.ell_d, dims.ell).unwrap();

    // Prepare instances and CSR matrices
    let mats_csr = s
        .matrices
        .iter()
        .map(|m| to_csr::<F>(m, s.n, s.m))
        .collect::<Vec<_>>();

    let witnesses = vec![mcs_wit];

    let insts =
        precompute::prepare_instances(&s, &params, &[mcs_inst], &witnesses, &mats_csr, &l)
            .unwrap();

    // Compute beta block (code under test).
    let beta_blk = precompute::precompute_beta_block(
        &s,
        &params,
        &insts,
        &witnesses,
        &[],
        &ch,
        dims.ell_d,
        dims.ell_n,
    )
    .unwrap();

    // Expected: Σ_row χ_{β_r}(row) * f( (M_0 z)[row], (M_1 z)[row], (M_2 z)[row] )
    let w_beta_r = HalfTableEq::new(&ch.beta_r);

    let mz0 = &insts[0].mz[0];
    let mz1 = &insts[0].mz[1];
    let mz2 = &insts[0].mz[2];

    let mut expected = K::ZERO;
    for row in 0..s.n {
        let args = vec![K::from(mz0[row]), K::from(mz1[row]), K::from(mz2[row])];
        let f_row = s.f.eval_in_ext::<K>(&args);
        expected += w_beta_r.w(row) * f_row;
    }

    assert_eq!(
        beta_blk.f_at_beta_r, expected,
        "F-block initial sum must equal the eq-weighted rowwise f."
    );

    // Also demonstrate that non-linear f does NOT commute with averaging.
    let mut wrong_args = vec![K::ZERO; s.t()];
    for j in 0..s.t() {
        let mut acc = K::ZERO;
        for row in 0..s.n {
            acc += K::from(insts[0].mz[j][row]) * w_beta_r.w(row);
        }
        wrong_args[j] = acc;
    }
    let wrong = s.f.eval_in_ext::<K>(&wrong_args);

    assert_ne!(
        expected, wrong,
        "Non-linear f should not equal f of eq-weighted row averages"
    );
}

