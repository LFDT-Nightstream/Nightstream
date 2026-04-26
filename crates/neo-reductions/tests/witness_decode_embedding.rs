use neo_ccs::Mat;
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::common::decode_z_from_witness_mat;
use neo_reductions::error::PiCcsError;
use p3_field::PrimeCharacteristicRing;

#[test]
fn decode_witness_z_rejects_neo_digit_layout_compat_mode() {
    let params = NeoParams::goldilocks_127();
    let m = 6usize;
    let mut z_mat = Mat::zero(D, m, F::ZERO);

    // z[c] = z0 + b*z1 + b^2*z2 for b=2 with small signed digits.
    let one = F::ONE;
    let neg_one = F::ZERO - one;
    let d0 = [one, F::ZERO, one, neg_one, one, F::ZERO];
    let d1 = [F::ZERO, one, neg_one, one, F::ZERO, one];
    let d2 = [one, one, F::ZERO, F::ZERO, neg_one, one];
    for c in 0..m {
        z_mat[(0, c)] = d0[c];
        z_mat[(1, c)] = d1[c];
        z_mat[(2, c)] = d2[c];
    }

    let err = decode_z_from_witness_mat(&params, &z_mat, m)
        .expect_err("Neo digit layout must not decode after SuperNeo-only cleanup");
    assert!(matches!(err, PiCcsError::InvalidInput(_)));
}

#[test]
fn decode_witness_z_superneo_packed_layout_flattens_coeff_blocks() {
    let params = NeoParams::goldilocks_127();
    let blocks = 2usize;
    let expected_m = blocks * D;
    let mut packed = Mat::zero(D, blocks, F::ZERO);

    for blk in 0..blocks {
        for rho in 0..D {
            let v = (blk as u64 + 1) * 1_000 + rho as u64;
            packed[(rho, blk)] = F::from_u64(v);
        }
    }

    let got = decode_z_from_witness_mat(&params, &packed, expected_m).expect("superneo packed decode");
    for blk in 0..blocks {
        for rho in 0..D {
            let idx = blk * D + rho;
            let want = K::from(F::from_u64((blk as u64 + 1) * 1_000 + rho as u64));
            assert_eq!(got[idx], want, "packed flatten mismatch at idx={idx}");
        }
    }
}

#[test]
fn decode_witness_z_rejects_unknown_shape() {
    let params = NeoParams::goldilocks_127();
    let bad = Mat::zero(D, 5, F::ZERO);
    let err = decode_z_from_witness_mat(&params, &bad, 37).expect_err("shape must be rejected");
    assert!(matches!(err, PiCcsError::InvalidInput(_)));
}
