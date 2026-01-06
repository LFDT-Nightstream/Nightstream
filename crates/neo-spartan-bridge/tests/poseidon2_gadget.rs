#![allow(non_snake_case)]

use bellpepper_core::num::AllocatedNum;
use bellpepper_core::test_cs::TestConstraintSystem;
use bellpepper_core::ConstraintSystem;
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use p3_symmetric::Permutation;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use neo_spartan_bridge::gadgets::poseidon2::{permute_w8, WIDTH};
use neo_spartan_bridge::CircuitF;

fn to_circuit(x: Goldilocks) -> CircuitF {
    CircuitF::from(x.as_canonical_u64())
}

fn native_permute(mut st: [Goldilocks; WIDTH]) -> [Goldilocks; WIDTH] {
    let perm = neo_ccs::crypto::poseidon2_goldilocks::permutation();
    perm.permute_mut(&mut st);
    st
}

fn constrain_equal<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    a: &AllocatedNum<CircuitF>,
    b: &AllocatedNum<CircuitF>,
    label: &str,
) {
    cs.enforce(
        || format!("{label}_eq"),
        |lc| lc + a.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + b.get_variable(),
    );
}

#[test]
fn poseidon2_permute_w8_matches_native_for_random_state() {
    let mut rng = ChaCha8Rng::seed_from_u64(12345);
    let st_in: [Goldilocks; WIDTH] = core::array::from_fn(|_| rng.random());
    let st_out = native_permute(st_in);

    let mut cs = TestConstraintSystem::<CircuitF>::new();

    let mut st_vars: [AllocatedNum<CircuitF>; WIDTH] = core::array::from_fn(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("in_{i}")), || Ok(to_circuit(st_in[i])))
            .expect("alloc input")
    });

    permute_w8(&mut cs, &mut st_vars).expect("permute_w8");

    for i in 0..WIDTH {
        let expected =
            AllocatedNum::alloc(cs.namespace(|| format!("expected_{i}")), || Ok(to_circuit(st_out[i]))).unwrap();
        constrain_equal(&mut cs, &st_vars[i], &expected, &format!("out_{i}"));
    }

    assert!(cs.is_satisfied(), "poseidon2 gadget constraints must satisfy");
}
