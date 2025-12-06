use neo_ajtai::{commit as ajtai_commit, set_global_pp, setup, AjtaiSModule};
use neo_ccs::{traits::SModuleHomomorphism, Mat};
use neo_math::F as Fq;
use p3_field::PrimeCharacteristicRing as _;
use std::sync::Arc;

#[test]
fn ajtai_smodule_commit_matches_direct_commit() {
    let mut rng = rand::rng();
    let d = neo_math::ring::D;
    let kappa = 4;
    let m = 3;
    let pp = setup(&mut rng, d, kappa, m).unwrap();
    set_global_pp(pp.clone()).unwrap();
    let l = AjtaiSModule::new(Arc::new(pp.clone()));

    // Random Z as a d×m row-major matrix, then re-commit both ways
    let mut z = Mat::zero(d, m, Fq::ZERO);
    for r in 0..d {
        for c in 0..m {
            z[(r, c)] = Fq::from_u64((r as u64) * 17 + (c as u64) * 13);
        }
    }

    let mut col_major = vec![Fq::ZERO; d * m];
    for c in 0..m {
        for r in 0..d {
            col_major[c * d + r] = z[(r, c)];
        }
    }

    let c1 = ajtai_commit(&pp, &col_major);
    let c2 = l.commit(&z);
    assert_eq!(c1, c2);
}
