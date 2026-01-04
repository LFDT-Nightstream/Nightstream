use neo_ajtai::{set_global_pp, set_global_pp_seeded, unload_global_pp_for_dims, setup};
use rand::SeedableRng;

#[test]
fn seeded_registry_rejects_mismatched_seed_on_loaded_pp() {
    let d = neo_math::D;
    let kappa = 2;
    let m = 8;

    let mut rng = rand::rngs::StdRng::from_seed([9u8; 32]);
    let pp = setup(&mut rng, d, kappa, m).expect("setup");
    set_global_pp(pp).expect("set_global_pp");

    let err = set_global_pp_seeded(d, kappa, m, [7u8; 32]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("already loaded") || msg.contains("disallowed") || msg.contains("cannot register"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn unseeded_registry_refuses_to_unload() {
    let d = neo_math::D;
    let kappa = 2;
    let m = 8;

    let mut rng = rand::rngs::StdRng::from_seed([1u8; 32]);
    let pp = setup(&mut rng, d, kappa, m).expect("setup");
    set_global_pp(pp).expect("set_global_pp");

    let err = unload_global_pp_for_dims(d, m).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("not seeded") || msg.contains("refusing to unload"),
        "unexpected error message: {msg}"
    );
}

