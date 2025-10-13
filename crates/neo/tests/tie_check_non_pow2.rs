#![cfg(feature = "testing")]
use neo::tie_check_with_r_public;
use neo::F;
use p3_field::PrimeCharacteristicRing;
use neo_ccs::{SparsePoly, Term};

#[test]
fn tie_check_with_r_non_pow2_n() {
    // Non-power-of-two n
    let n = 3usize; // rows in M
    let m = 2usize; // cols
    let d = neo_math::D; // Ajtai dimension

    // Single-matrix CCS (t=1)
    let m0 = neo_ccs::Mat::from_row_major(
        n,
        m,
        vec![
            F::ONE, F::ZERO,
            F::ZERO, F::ONE,
            F::ONE, F::ONE,
        ],
    );
    let f = SparsePoly::new(1, vec![Term { coeff: F::ONE, exps: vec![1] }]);
    let s = neo_ccs::CcsStructure::new(vec![m0], f).expect("build CCS");
    assert_eq!(s.n, n);
    assert_eq!(s.m, m);
    assert_eq!(s.t(), 1);

    // r with ell=2 (LSB-first)
    let r: Vec<neo_math::K> = vec![
        neo_math::K::from(F::from_u64(2)),
        neo_math::K::from(F::from_u64(5)),
    ];

    // Build Z with two nonzero rows
    let mut z_data = vec![F::ZERO; d * m];
    z_data[0 * m + 0] = F::ONE;
    z_data[1 * m + 1] = F::ONE;
    let z_mat = neo_ccs::Mat::from_row_major(d, m, z_data.clone());
    let wit_parent = neo_ccs::MeWitness { Z: z_mat };

    // Compute v = M^T * chi_r and y_pred = Z * v
    // Reuse crate-local helper signature via public wrapper
    // Build chi manually
    let ell = r.len();
    let mut chi = vec![neo_math::K::ZERO; n];
    for i in 0..n {
        let mut w = neo_math::K::ONE;
        let mut ii = i;
        for k in 0..ell {
            let rk = r[k];
            let bit_is_one = (ii & 1) == 1;
            let term = if bit_is_one { rk } else { neo_math::K::ONE - rk };
            w *= term;
            ii >>= 1;
        }
        chi[i] = w;
    }
    let m0_ref = &s.matrices[0];
    let mut v = vec![neo_math::K::ZERO; m];
    for i in 0..n {
        let wi = chi[i];
        for col in 0..m {
            let mij = m0_ref[(i, col)];
            if mij != F::ZERO {
                v[col] += neo_math::K::from(mij) * wi;
            }
        }
    }
    let mut y_pred = vec![neo_math::K::ZERO; d];
    for row in 0..d {
        let mut acc = neo_math::K::ZERO;
        for col in 0..m {
            let z_rc = z_data[row * m + col];
            if z_rc != F::ZERO { acc += neo_math::K::from(z_rc) * v[col]; }
        }
        y_pred[row] = acc;
    }

    // me_parent: set y as computed, X as Z[:, :m_in]
    let m_in = 1usize;
    let mut x_slice = vec![F::ZERO; d * m_in];
    for row in 0..d { x_slice[row * m_in + 0] = z_data[row * m + 0]; }
    let x_mat = neo_ccs::Mat::from_row_major(d, m_in, x_slice);
    let me_parent = neo_ccs::MeInstance::<neo_ajtai::Commitment, F, neo_math::K> {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: neo_ajtai::Commitment::zeros(neo_math::D, 1),
        X: x_mat,
        r: r.clone(),
        y: vec![y_pred],
        y_scalars: vec![neo_math::K::ZERO],
        m_in,
        fold_digest: [0u8; 32],
    };

    tie_check_with_r_public(&s, &me_parent, &wit_parent, &r)
        .expect("tie_with_r should pass for non-pow2 n");
}
