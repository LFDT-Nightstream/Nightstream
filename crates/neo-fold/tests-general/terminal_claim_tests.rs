use neo_ccs::{CcsStructure, Mat, SparsePoly, Term, MeInstance};
use neo_math::{F, K};
use neo_fold::pi_ccs::pi_ccs_compute_terminal_claim_r1cs_or_ccs;
use neo_ajtai::Commitment as Cmt;
use p3_field::PrimeCharacteristicRing;

fn r1cs_shape_ccs() -> CcsStructure<F> {
    // Build a CCS with t=3 and f(y)=y0*y1 - y2
    let m0 = Mat::from_row_major(1, 1, vec![F::ONE]);
    let m1 = Mat::from_row_major(1, 1, vec![F::ONE]);
    let m2 = Mat::from_row_major(1, 1, vec![F::ONE]);
    let f = SparsePoly::new(3, vec![
        Term { coeff: F::ONE, exps: vec![1,1,0] },
        Term { coeff: -F::ONE, exps: vec![0,0,1] },
    ]);
    CcsStructure::new(vec![m0, m1, m2], f).unwrap()
}

#[test]
fn r1cs_terminal_claim_matches_expected() {
    let s = r1cs_shape_ccs();

    // Two instances with simple y_scalars triples
    let me0 = MeInstance::<Cmt, F, K> {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: Cmt::zeros(neo_math::D, 1),
        X: Mat::from_row_major(1,1,vec![F::ZERO]),
        r: vec![],
        y: vec![vec![K::ZERO; neo_math::D]; 3],
        y_scalars: vec![K::from(F::from_u64(2)), K::from(F::from_u64(3)), K::from(F::from_u64(6))],
        m_in: 0,
        fold_digest: [0u8; 32],
    };
    let me1 = MeInstance::<Cmt, F, K> {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: Cmt::zeros(neo_math::D, 1),
        X: Mat::from_row_major(1,1,vec![F::ZERO]),
        r: vec![],
        y: vec![vec![K::ZERO; neo_math::D]; 3],
        y_scalars: vec![K::from(F::from_u64(5)), K::from(F::from_u64(4)), K::from(F::from_u64(20))],
        m_in: 0,
        fold_digest: [0u8; 32],
    };
    let out_me = vec![me0, me1];

    let alphas = vec![K::from(F::from_u64(7)), K::from(F::from_u64(9))];
    let wr = K::from(F::from_u64(11));

    // Expected = wr * sum_i alpha_i * (a*b - c)
    let expected = wr * (
        alphas[0] * (out_me[0].y_scalars[0] * out_me[0].y_scalars[1] - out_me[0].y_scalars[2]) +
        alphas[1] * (out_me[1].y_scalars[0] * out_me[1].y_scalars[1] - out_me[1].y_scalars[2])
    );

    let got = pi_ccs_compute_terminal_claim_r1cs_or_ccs(&s, wr, &alphas, &out_me);
    assert_eq!(got, expected, "R1CS terminal claim mismatch");
}

#[test]
fn generic_ccs_terminal_claim_matches_expected() {
    // t=1, f(y)=y0; generic CCS path ignores wr
    let m0 = Mat::from_row_major(1, 1, vec![F::ONE]);
    let f = SparsePoly::new(1, vec![Term { coeff: F::ONE, exps: vec![1] }]);
    let s = CcsStructure::new(vec![m0], f).unwrap();

    let me0 = MeInstance::<Cmt, F, K> {
        c_step_coords: vec![], u_offset: 0, u_len: 0, c: Cmt::zeros(neo_math::D, 1),
        X: Mat::from_row_major(1,1,vec![F::ZERO]), r: vec![], y: vec![vec![K::ZERO; neo_math::D]],
        y_scalars: vec![K::from(F::from_u64(3))], m_in: 0, fold_digest: [0u8; 32]
    };
    let me1 = MeInstance::<Cmt, F, K> {
        c_step_coords: vec![], u_offset: 0, u_len: 0, c: Cmt::zeros(neo_math::D, 1),
        X: Mat::from_row_major(1,1,vec![F::ZERO]), r: vec![], y: vec![vec![K::ZERO; neo_math::D]],
        y_scalars: vec![K::from(F::from_u64(10))], m_in: 0, fold_digest: [0u8; 32]
    };
    let out_me = vec![me0, me1];

    let alphas = vec![K::from(F::from_u64(2)), K::from(F::from_u64(5))];
    let wr = K::from(F::from_u64(12345)); // should be ignored for generic CCS

    // Expected = sum_i alpha_i * f(y_i) = sum_i alpha_i * y0
    let expected = alphas[0] * out_me[0].y_scalars[0] + alphas[1] * out_me[1].y_scalars[0];
    let got = pi_ccs_compute_terminal_claim_r1cs_or_ccs(&s, wr, &alphas, &out_me);
    assert_eq!(got, expected, "Generic CCS terminal claim mismatch");
}
