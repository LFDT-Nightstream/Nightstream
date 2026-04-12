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
fn me_consistency_superneo_packed_enforces_constant_term_ct() {
    let params = NeoParams::goldilocks_127();

    // CCS: n=1, m=D (SuperNeo-compatible), t=1, linear f(y)=y0.
    let n = 1usize;
    let m = D;
    let m0 = Mat::from_row_major(n, m, vec![Fq::ONE; m]);
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: Fq::ONE,
            exps: vec![1],
        }],
    );
    let s = CcsStructure::new(vec![m0], f).unwrap();

    // Packed witness layout for m=D is D×1.
    let mut Z = Mat::zero(D, 1, Fq::ZERO);
    for rho in 0..D {
        Z[(rho, 0)] = Fq::from_u64((rho as u64) + 1);
    }

    let L = TestL;
    let m_in = 1usize;
    let c = L.commit(&Z);
    let X = {
        let mut out = Mat::zero(D, m_in, Fq::ZERO);
        out[(0, 0)] = Z[(0, 0)];
        out
    };

    // n=1 => ell_n=0 => r is empty and rb=[1].
    let r: Vec<K> = vec![];

    let rb = tensor_point::<K>(&r);
    let mut v_k_m = vec![K::ZERO; s.m];
    s.matrices[0].add_mul_transpose_into(&rb, &mut v_k_m, s.n);
    let mut y0 = vec![K::ZERO; D];
    for rho in 0..D {
        let mut acc = K::ZERO;
        for c in 0..m {
            if c % D == rho {
                acc += K::from(Z[(rho, c / D)]) * v_k_m[c];
            }
        }
        y0[rho] = acc;
    }

    let inst = CeClaim::<_, Fq, K> {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c,
        X,
        r,
        s_col: vec![],
        y_ring: vec![y0.clone()],
        ct: vec![y0[0]],
        aux_openings: vec![],
        y_zcol: vec![],
        m_in,
        fold_digest: [0u8; 32],
    };
    let wit = CeWitness::<Fq> { Z: Z.clone() };

    assert!(check_ce_consistency(&params, &s, &L, &inst, &wit).is_ok());

    // Tamper ct to Neo-style recomposition value; packed layout must reject it.
    let mut ct_neo = K::ZERO;
    let mut pow = K::ONE;
    let b_k = K::from(Fq::from_u64(params.b as u64));
    for rho in 0..D {
        ct_neo += y0[rho] * pow;
        pow *= b_k;
    }
    assert_ne!(ct_neo, y0[0], "test requires non-trivial ct difference");

    let mut inst_bad = inst.clone();
    inst_bad.ct[0] = ct_neo;
    assert!(check_ce_consistency(&params, &s, &L, &inst_bad, &wit).is_err());
}
