use neo_ajtai::Commitment;
use neo_ccs::{Mat, MeInstance};
use neo_fold::bridge_digests::{compute_accumulator_digest_v2, compute_obligations_digest_v1};
use neo_math::{F as NeoF, K as NeoK, KExtensions};
use p3_field::PrimeCharacteristicRing;

fn sample_me_instance(seed: u64) -> MeInstance<Commitment, NeoF, NeoK> {
    let d = neo_math::D;
    let kappa = 16usize;
    let m_in = 2usize;

    let mut c_data = Vec::with_capacity(d * kappa);
    for i in 0..(d * kappa) {
        c_data.push(NeoF::from_u64(seed.wrapping_add(i as u64)));
    }
    let c = Commitment { d, kappa, data: c_data };

    let mut x_data = Vec::with_capacity(d * m_in);
    for i in 0..(d * m_in) {
        x_data.push(NeoF::from_u64(seed.wrapping_add(10_000 + i as u64)));
    }
    let x = Mat::from_row_major(d, m_in, x_data);

    let r = vec![
        NeoK::from_coeffs([NeoF::from_u64(seed + 1), NeoF::from_u64(seed + 2)]),
        NeoK::from_coeffs([NeoF::from_u64(seed + 3), NeoF::from_u64(seed + 4)]),
        NeoK::from_coeffs([NeoF::from_u64(seed + 5), NeoF::from_u64(seed + 6)]),
    ];

    let t = 2usize;
    let mut y = Vec::with_capacity(t);
    for j in 0..t {
        let mut yj = Vec::with_capacity(d);
        for rho in 0..d {
            let a = seed.wrapping_add(20_000 + (j * 1_000 + rho) as u64);
            let b = seed.wrapping_add(30_000 + (j * 1_000 + rho) as u64);
            yj.push(NeoK::from_coeffs([NeoF::from_u64(a), NeoF::from_u64(b)]));
        }
        y.push(yj);
    }

    let y_scalars = vec![NeoK::ZERO; t];

    MeInstance {
        c,
        X: x,
        r,
        y,
        y_scalars,
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}

#[test]
fn accumulator_digest_v2_test_vector() {
    let acc = vec![sample_me_instance(123)];
    let got = compute_accumulator_digest_v2(2, acc.as_slice());

    // NOTE: This is a regression test vector. If this changes, bump the domain/version.
    let expected = [
        207, 238, 198, 127, 210, 249, 231, 207, 209, 224, 247, 132, 154, 19, 202, 223, 213, 64,
        72, 172, 145, 238, 120, 25, 64, 163, 193, 132, 119, 172, 180, 84,
    ];
    assert_eq!(got, expected);
}

#[test]
fn accumulator_digest_v2_ignores_y_scalars() {
    let mut a = sample_me_instance(123);
    let mut b = sample_me_instance(123);

    a.y_scalars = vec![NeoK::from_coeffs([NeoF::from_u64(1), NeoF::from_u64(2)]); a.y.len()];
    b.y_scalars = vec![NeoK::from_coeffs([NeoF::from_u64(999), NeoF::from_u64(1_000)]); b.y.len()];

    let da = compute_accumulator_digest_v2(2, std::slice::from_ref(&a));
    let db = compute_accumulator_digest_v2(2, std::slice::from_ref(&b));
    assert_eq!(da, db);
}

#[test]
fn obligations_digest_v1_test_vector() {
    let acc_main = [1u8; 32];
    let acc_val = [2u8; 32];
    let pp_id = [3u8; 32];
    let got = compute_obligations_digest_v1(acc_main, acc_val, pp_id);

    // NOTE: This is a regression test vector. If this changes, bump the domain/version.
    let expected = [
        173, 61, 162, 243, 57, 132, 202, 34, 177, 95, 106, 103, 27, 196, 88, 185, 8, 138, 113,
        158, 147, 17, 10, 228, 114, 173, 127, 102, 204, 101, 129, 193,
    ];
    assert_eq!(got, expected);
}
