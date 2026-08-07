use neo_ajtai::{commit as ajtai_commit, set_global_pp, set_global_pp_seeded, setup, AjtaiSModule};
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

#[test]
fn ajtai_smodule_materializes_owned_pp() {
    let mut rng = rand::rng();
    let d = neo_math::ring::D;
    let kappa = 4;
    let m = 5;
    let pp = setup(&mut rng, d, kappa, m).unwrap();
    let l = AjtaiSModule::new(Arc::new(pp.clone()));

    let materialized = l.materialize_pp().unwrap();
    assert_eq!(materialized.d, pp.d);
    assert_eq!(materialized.m, pp.m);
    assert_eq!(materialized.kappa, pp.kappa);
    assert_eq!(materialized.m_rows, pp.m_rows);
}

#[test]
fn ajtai_smodule_materializes_seeded_global_pp() {
    let d = neo_math::ring::D;
    let kappa = 3;
    let m = 7;
    let seed = [41u8; 32];
    set_global_pp_seeded(d, kappa, m, seed).unwrap();
    let l = AjtaiSModule::from_global_for_dims(d, m).unwrap();

    let materialized = l.materialize_pp().unwrap();
    assert_eq!(materialized.d, d);
    assert_eq!(materialized.m, m);
    assert_eq!(materialized.kappa, kappa);
}

#[test]
fn equal_shaped_seeded_modules_keep_independent_setup_identity() {
    let d = neo_math::ring::D;
    let kappa = 3;
    let m = 2;
    let seed_a = [51u8; 32];
    let seed_b = [52u8; 32];
    let module_a = AjtaiSModule::from_seeded(seed_a, d, kappa, m).expect("seed A module");
    let module_b = AjtaiSModule::from_seeded(seed_b, d, kappa, m).expect("seed B module");
    let module_a_again = AjtaiSModule::from_seeded(seed_a, d, kappa, m).expect("second seed A module");

    let mut z = neo_ccs::Mat::zero(d, m, Fq::ZERO);
    z[(0, 0)] = Fq::ONE;
    z[(1, 1)] = Fq::ONE;

    assert_eq!(module_a.seeded_params(), Some((kappa, seed_a)));
    assert_eq!(module_b.seeded_params(), Some((kappa, seed_b)));
    assert_ne!(module_a.commit(&z), module_b.commit(&z));
    assert_eq!(module_a.commit(&z), module_a_again.commit(&z));
}
