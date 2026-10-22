use neo_ajtai::{commit, get_global_pp_for_dims, set_global_pp, setup};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use rand::SeedableRng;

#[test]
fn ajtai_pp_registry_handles_multiple_m() {
    let d = neo_math::D;

    let w3: Vec<F> = vec![F::ONE, F::ZERO, F::ONE];
    let mut rng = rand::rngs::StdRng::from_seed([7u8; 32]);
    let pp3 = setup(&mut rng, d, 16, w3.len()).expect("setup 3");
    set_global_pp(pp3).expect("register 3");

    let z3 = vec![F::ONE; d * w3.len()];
    let pp3_ref = get_global_pp_for_dims(d, 3).expect("get (d,3)");
    let _c3 = commit(&pp3_ref, &z3);

    let w4: Vec<F> = vec![F::ONE, F::ZERO, F::ONE, F::ONE];
    let pp4 = setup(&mut rng, d, 16, w4.len()).expect("setup 4");
    set_global_pp(pp4).expect("register 4");

    let z4 = vec![F::ONE; d * w4.len()];
    let pp4_ref = get_global_pp_for_dims(d, 4).expect("get (d,4)");
    let _c4 = commit(&pp4_ref, &z4);

    println!("✅ Registry handles multiple (d,m) parameters correctly");
}
