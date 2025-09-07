use neo_ajtai::{setup, set_global_pp, get_global_pp_for_dims, decomp_b, commit, DecompStyle};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use rand::SeedableRng;

#[test]
fn ajtai_pp_registry_handles_multiple_m() {
    let d = neo_math::D;
    let b = 2;

    // m = 3 case  
    let w3: Vec<F> = vec![F::ONE, F::ZERO, F::ONE];
    let mut rng = rand::rngs::StdRng::from_seed([7u8; 32]);
    let pp3 = setup(&mut rng, d, 16, w3.len()).expect("setup 3");
    set_global_pp(pp3).expect("register 3");

    let z3 = decomp_b(&w3, b, d, DecompStyle::Balanced);
    assert_eq!(z3.len(), d * 3, "z3 has wrong length");
    let pp3_ref = get_global_pp_for_dims(d, 3).expect("get (d,3)");
    let _c3 = commit(&pp3_ref, &z3); // must succeed

    // m = 4 case in the SAME process
    let w4: Vec<F> = vec![
        F::ONE,
        F::ZERO,
        F::ONE,
        F::ONE,
    ];
    // new PP for (d,4)
    let pp4 = setup(&mut rng, d, 16, w4.len()).expect("setup 4");
    set_global_pp(pp4).expect("register 4");

    let z4 = decomp_b(&w4, b, d, DecompStyle::Balanced);
    assert_eq!(z4.len(), d * 4, "z4 has wrong length");
    let pp4_ref = get_global_pp_for_dims(d, 4).expect("get (d,4)");
    let _c4 = commit(&pp4_ref, &z4); // would have failed before; now OK

    println!("✅ Registry handles multiple (d,m) parameters correctly");
}
