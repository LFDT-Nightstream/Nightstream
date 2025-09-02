// crates/neo-ccs/tests/red_team_me.rs
#![allow(non_snake_case)] // Allow uppercase math variables like Z, X, L

use neo_ccs::{
    traits::SModuleHomomorphism, Mat, CcsStructure, MeInstance, MeWitness,
    poly::SparsePoly, poly::Term, relations::{check_me_consistency}
};
use p3_goldilocks::Goldilocks as Fq;
use p3_field::PrimeCharacteristicRing;
use neo_ajtai::{PP, setup as ajtai_setup, commit as ajtai_commit};
use neo_math::ring::D;
use rand::SeedableRng;

struct AjtaiL { pp: PP<neo_math::ring::Rq> }

impl SModuleHomomorphism<Fq, neo_ajtai::Commitment> for AjtaiL {
    fn commit(&self, z: &Mat<Fq>) -> neo_ajtai::Commitment {
        assert_eq!(z.rows(), D);
        let (d, m) = (z.rows(), z.cols());
        let mut col_major = vec![Fq::ZERO; d*m];
        for c in 0..m { for r in 0..d { col_major[c*d + r] = z[(r, c)]; } }
        ajtai_commit(&self.pp, &col_major)
    }
    fn project_x(&self, z: &Mat<Fq>, min: usize) -> Mat<Fq> {
        let (d, m) = (z.rows(), z.cols());
        assert!(min <= m);
        let mut out = Mat::zero(d, min, Fq::ZERO);
        for c in 0..min { for r in 0..d { out[(r,c)] = z[(r,c)]; } }
        out
    }
}

#[test]
fn me_consistency_rejects_tamper() {
    // CCS: n=4 (power of two), m=3, t=1, f(y)=y0 (linear)
    let n = 4usize; let m = 3usize;
    let m0 = Mat::from_row_major(n, m, vec![
        Fq::ONE, Fq::ZERO, Fq::ZERO,
        Fq::ZERO, Fq::ONE, Fq::ZERO,
        Fq::ZERO, Fq::ZERO, Fq::ONE,
        Fq::ONE, Fq::ONE, Fq::ONE,
    ]);
    let f = SparsePoly::new(1, vec![Term{coeff: Fq::ONE, exps: vec![1]}]);
    let s = CcsStructure::new(vec![m0.clone()], f).unwrap();

    // Construct Z (d×m) with small entries
    let d = D;
    let mut Z = Mat::zero(d, m, Fq::ZERO);
    for c in 0..m { for r in 0..d { Z[(r,c)] = Fq::from_u64(((r + c) % 5) as u64); } }

    // Ajtai map
    let pp = ajtai_setup(&mut rand::rngs::StdRng::from_seed([11u8;32]), d, 8, m);
    let L = AjtaiL { pp: pp.expect("Setup should succeed") };

    // Instance: c, X (first m_in columns), r, y
    let m_in = 1usize;
    let c = L.commit(&Z);
    let X = L.project_x(&Z, m_in);

    // Choose r ∈ K^ell with ell=log2(n)=2
    type K = neo_math::ExtF;
    let r = vec![K::from(Fq::from_u64(3)), K::from(Fq::from_u64(5))]; // arbitrary
    let rb = neo_ccs::utils::tensor_point::<K>(&r);

    // v = M^T rb (in K^m)
    let v_k = {
        let mut v = vec![K::ZERO; m];
        for row in 0..n {
            let rb_r = rb[row];
            let row_slice = m0.row(row).to_vec();
            for cidx in 0..m {
                v[cidx] += K::from(row_slice[cidx]) * rb_r;
            }
        }
        v
    };

    // y = Z * v (in K^d)
    let y0 = neo_ccs::utils::mat_vec_mul_fk::<Fq, K>(Z.as_slice(), d, m, &v_k);

    let inst = MeInstance::<_, Fq, K> { c: c.clone(), X: X.clone(), r: r.clone(), y: vec![y0.clone()], m_in, fold_digest: [0u8; 32] };
    let wit  = MeWitness::<Fq> { Z: Z.clone() };

    // Baseline must succeed
    assert!(check_me_consistency(&s, &L, &inst, &wit).is_ok());

    // Tamper y → fail
    let mut inst_bad = inst.clone();
    inst_bad.y[0][0] += K::ONE;
    assert!(check_me_consistency(&s, &L, &inst_bad, &wit).is_err(), "tampered y must be rejected");

    // Tamper X → fail
    let mut X_bad = X.clone();
    X_bad[(0,0)] += Fq::ONE;
    let inst_bad2 = MeInstance { X: X_bad, ..inst.clone() };
    assert!(check_me_consistency(&s, &L, &inst_bad2, &wit).is_err(), "tampered X must be rejected");

    // Tamper c → fail
    let mut c_bad = c.clone();
    c_bad.data[0] += Fq::ONE;
    let inst_bad3 = MeInstance { c: c_bad, ..inst };
    assert!(check_me_consistency(&s, &L, &inst_bad3, &wit).is_err(), "tampered Ajtai commitment must be rejected");
}
