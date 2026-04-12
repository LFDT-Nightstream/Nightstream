// crates/neo-ccs/tests/red_team_me.rs
#![allow(non_snake_case)] // Allow uppercase math variables like Z, X, L

use neo_ajtai::{commit as ajtai_commit, setup as ajtai_setup, PP};
use neo_ccs::{
    poly::SparsePoly, poly::Term, relations::check_ce_consistency, traits::SModuleHomomorphism, utils::tensor_point,
    CcsStructure, CeClaim, CeWitness, Mat,
};
use neo_math::ring::D;
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;
use rand::SeedableRng;

struct AjtaiL {
    pp: PP<neo_math::ring::Rq>,
}

impl SModuleHomomorphism<Fq, neo_ajtai::Commitment> for AjtaiL {
    fn commit(&self, z: &Mat<Fq>) -> neo_ajtai::Commitment {
        assert_eq!(z.rows(), D);
        let (d, m) = (z.rows(), z.cols());
        let mut col_major = vec![Fq::ZERO; d * m];
        for c in 0..m {
            for r in 0..d {
                col_major[c * d + r] = z[(r, c)];
            }
        }
        ajtai_commit(&self.pp, &col_major)
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
fn me_consistency_rejects_tamper() {
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
    for r in 0..d {
        Z[(r, 0)] = Fq::from_u64(((r % 5) + 1) as u64);
    }

    // Ajtai map
    let pp = ajtai_setup(&mut rand::rngs::StdRng::from_seed([11u8; 32]), d, 8, Z.cols());
    let L = AjtaiL {
        pp: pp.expect("Setup should succeed"),
    };

    // Instance: c, X (first m_in columns), r, y
    let m_in = 1usize;
    let c = L.commit(&Z);
    // SuperNeo packed projection for the first field column: only row 0 contributes.
    let mut X = Mat::zero(d, m_in, Fq::ZERO);
    X[(0, 0)] = Z[(0, 0)];

    // Choose r ∈ K^ell with ell=log2(n)=2
    use neo_math::K;
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
    // Pad y to Ajtai digit length (2^ell_d) expected by CE checks.
    let mut y0_padded = y0.clone();
    y0_padded.resize(D.next_power_of_two(), K::ZERO);
    let ct0 = y0_padded[0];

    let inst = CeClaim::<_, Fq, K> {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: c.clone(),
        X: X.clone(),
        r: r.clone(),
        s_col: vec![],
        y_ring: vec![y0_padded.clone()],
        ct: vec![ct0],
        aux_openings: vec![],
        y_zcol: vec![],
        m_in,
        fold_digest: [0u8; 32],
    };
    let wit = CeWitness::<Fq> { Z: Z.clone() };

    // Baseline must succeed
    assert!(check_ce_consistency(&params, &s, &L, &inst, &wit).is_ok());

    // Tamper y → fail
    let mut inst_bad = inst.clone();
    inst_bad.y_ring[0][0] += K::ONE;
    assert!(
        check_ce_consistency(&params, &s, &L, &inst_bad, &wit).is_err(),
        "tampered y must be rejected"
    );

    // Tamper ct (constant term) -> fail
    let mut inst_bad_ct = inst.clone();
    inst_bad_ct.ct[0] += K::ONE;
    assert!(
        check_ce_consistency(&params, &s, &L, &inst_bad_ct, &wit).is_err(),
        "tampered ct must be rejected"
    );

    // Tamper X → fail
    let mut X_bad = X.clone();
    X_bad[(0, 0)] += Fq::ONE;
    let inst_bad2 = CeClaim {
        X: X_bad,
        ..inst.clone()
    };
    assert!(
        check_ce_consistency(&params, &s, &L, &inst_bad2, &wit).is_err(),
        "tampered X must be rejected"
    );

    // Tamper c → fail
    let mut c_bad = c.clone();
    c_bad.data[0] += Fq::ONE;
    let inst_bad3 = CeClaim { c: c_bad, ..inst };
    assert!(
        check_ce_consistency(&params, &s, &L, &inst_bad3, &wit).is_err(),
        "tampered Ajtai commitment must be rejected"
    );
}
