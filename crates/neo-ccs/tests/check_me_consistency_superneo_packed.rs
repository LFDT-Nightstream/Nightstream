#![allow(non_snake_case)]

mod support;

use neo_ccs::{
    poly::SparsePoly, poly::Term, relations::check_ce_consistency, traits::SModuleHomomorphism, CcsStructure, CeClaim,
    CeWitness, Mat,
};
use neo_math::ring::D;
use neo_math::K;
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;
use support::superneo_y_ring;

struct TestL;

impl SModuleHomomorphism<Fq, Vec<Fq>> for TestL {
    fn commit(&self, z: &Mat<Fq>) -> Vec<Fq> {
        z.as_slice().to_vec()
    }
}

fn superneo_project_x(z: &Mat<Fq>, m_in: usize) -> Mat<Fq> {
    let mut out = Mat::zero(D, m_in, Fq::ZERO);
    let active_cols = core::cmp::min(m_in, z.cols());
    for col in 0..active_cols {
        for rho in 0..D {
            out[(rho, col)] = z[(rho, col)];
        }
    }
    out
}

#[test]
fn me_consistency_superneo_packed_enforces_constant_term_ct() {
    let params = NeoParams::goldilocks_paper_b2();

    // CCS: n=1, m=D, t=1, linear f(y)=y0.
    let n = 1usize;
    let m = D;
    let m0 = Mat::from_row_major(n, m, (0..m).map(|c| Fq::from_u64((c as u64) + 1)).collect());
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
    let X = superneo_project_x(&Z, m_in);

    // n=1 => ell_n=0 => r is empty and rb=[1].
    let r: Vec<K> = vec![];

    let y0 = superneo_y_ring(&s, &Z, &r).remove(0);

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

    // Tamper ct to a non-constant-term recomposition; SuperNeo must reject it.
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

    // A pre-SuperNeo lane-style y_ring skips the matrix transform and must not verify.
    let mut lane_style_y = vec![K::ZERO; D];
    for rho in 0..D {
        lane_style_y[rho] = K::from(Z[(rho, 0)]) * K::from(Fq::from_u64((rho as u64) + 1));
    }
    assert_ne!(
        lane_style_y, y0,
        "test requires lane-style evaluation to differ from SuperNeo ring form"
    );
    let mut inst_bad_y = inst.clone();
    inst_bad_y.y_ring[0] = lane_style_y;
    inst_bad_y.ct[0] = inst_bad_y.y_ring[0][0];
    assert!(check_ce_consistency(&params, &s, &L, &inst_bad_y, &wit).is_err());
}
