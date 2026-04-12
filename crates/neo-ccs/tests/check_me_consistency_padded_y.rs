#![allow(non_snake_case)]

use neo_ccs::{
    poly::SparsePoly, poly::Term, relations::check_ce_consistency, traits::SModuleHomomorphism, utils::tensor_point,
    CcsStructure, CeClaim, CeWitness, Mat,
};
use neo_math::ring::D;
use neo_math::K;
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;

struct TestL;

impl SModuleHomomorphism<Fq, Vec<Fq>> for TestL {
    fn commit(&self, z: &Mat<Fq>) -> Vec<Fq> {
        z.as_slice().to_vec()
    }

    fn project_x(&self, z: &Mat<Fq>, min: usize) -> Mat<Fq> {
        let (d, m) = (z.rows(), z.cols());
        assert!(min <= m);
        let mut out = Mat::zero(d, min, Fq::ZERO);
        for c in 0..min {
            for r in 0..d {
                out[(r, c)] = z[(r, c)];
            }
        }
        out
    }
}

#[test]
fn me_consistency_accepts_padded_y_rows() {
    let params = NeoParams::goldilocks_127();

    // CCS: n=4 (power of two), SuperNeo-compatible m=D, t=1, f(y)=y0 (linear)
    let n = 4usize;
    let m = D;
    let mut m0 = Mat::zero(n, m, Fq::ZERO);
    for i in 0..n {
        m0[(i, i)] = Fq::ONE;
    }
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: Fq::ONE,
            exps: vec![1],
        }],
    );
    let s = CcsStructure::new(vec![m0.clone()], f).unwrap();

    // Construct packed SuperNeo witness Z ∈ F^{D×(m/D)} = F^{D×1}.
    let d = D;
    let mut Z = Mat::zero(d, m / D, Fq::ZERO);
    Z[(0, 0)] = Fq::from_u64(2);

    let L = TestL;

    let m_in = 1usize;
    let c = L.commit(&Z);
    let X = L.project_x(&Z, m_in);

    // Choose r ∈ K^ell with ell=log2(n)=2.
    let r = vec![K::from(Fq::from_u64(3)), K::from(Fq::from_u64(5))]; // arbitrary
    let rb = tensor_point::<K>(&r);
    let mut v_k_m = vec![K::ZERO; s.m];
    s.matrices[0].add_mul_transpose_into(&rb, &mut v_k_m, s.n);
    let mut y0 = vec![K::ZERO; d];
    for rho in 0..d {
        let mut acc = K::ZERO;
        for c in 0..m {
            if c % D == rho {
                acc += K::from(Z[(rho, c / D)]) * v_k_m[c];
            }
        }
        y0[rho] = acc;
    }

    // Pad y to 2^{ell_d} (typically 64 for D=54).
    let mut y0_padded = y0.clone();
    let d_pad = D.next_power_of_two();
    y0_padded.resize(d_pad, K::ZERO);
    let ct0 = y0_padded[0];

    let inst = CeClaim::<_, Fq, K> {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c,
        X,
        r,
        s_col: vec![],
        y_ring: vec![y0_padded.clone()],
        ct: vec![ct0],
        aux_openings: vec![],
        y_zcol: vec![],
        m_in,
        fold_digest: [0u8; 32],
    };
    let wit = CeWitness::<Fq> { Z: Z.clone() };

    assert!(check_ce_consistency(&params, &s, &L, &inst, &wit).is_ok());

    // Non-zero padding must be rejected.
    let mut inst_bad = inst.clone();
    inst_bad.y_ring[0][D] += K::ONE;
    assert!(check_ce_consistency(&params, &s, &L, &inst_bad, &wit).is_err());
}
