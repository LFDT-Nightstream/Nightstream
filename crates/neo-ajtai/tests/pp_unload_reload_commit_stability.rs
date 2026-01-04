#![allow(non_snake_case)]

use neo_ajtai::{
    get_global_pp_for_dims, set_global_pp_seeded, unload_global_pp_for_dims, AjtaiSModule,
};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

#[test]
fn seeded_pp_commit_is_stable_across_load_and_unload() {
    let d = D;
    let m = 23usize;
    let kappa = 2usize;
    let seed = [9u8; 32];

    set_global_pp_seeded(d, kappa, m, seed).expect("set_global_pp_seeded");
    let l = AjtaiSModule::from_global_for_dims(d, m).expect("from_global_for_dims");

    let data: Vec<F> = (0..(d * m))
        .map(|i| F::from_u64((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5_A5A5_A5A5_A5A5))
        .collect();
    let Z = Mat::from_row_major(d, m, data);

    // Seeded (no PP materialized yet).
    let c_seeded = l.commit(&Z);

    // Force materialization.
    let pp = get_global_pp_for_dims(d, m).expect("get_global_pp_for_dims");
    assert_eq!(pp.kappa, kappa);
    let c_loaded = l.commit(&Z);
    assert_eq!(c_seeded, c_loaded, "loaded PP must match seeded commit");

    // Unload and re-commit via the seeded streaming path again.
    let unloaded = unload_global_pp_for_dims(d, m).expect("unload_global_pp_for_dims");
    assert!(unloaded, "expected PP to be loaded before unload");
    let c_seeded_again = l.commit(&Z);
    assert_eq!(c_seeded, c_seeded_again, "commit must be stable across unload/reload");
}
