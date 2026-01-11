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
use neo_spartan_bridge::gadgets::sponge::Poseidon2Sponge;
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
        AllocatedNum::alloc(cs.namespace(|| format!("in_{i}")), || Ok(to_circuit(st_in[i]))).expect("alloc input")
    });

    permute_w8(&mut cs, &mut st_vars).expect("permute_w8");

    for i in 0..WIDTH {
        let expected =
            AllocatedNum::alloc(cs.namespace(|| format!("expected_{i}")), || Ok(to_circuit(st_out[i]))).unwrap();
        constrain_equal(&mut cs, &st_vars[i], &expected, &format!("out_{i}"));
    }

    assert!(cs.is_satisfied(), "poseidon2 gadget constraints must satisfy");
}

#[test]
fn poseidon2_sponge_digest32_matches_compute_obligations_digest_v2() {
    let mut rng = ChaCha8Rng::seed_from_u64(999);
    let acc_final_main_digest: [u8; 32] = rng.random();
    let acc_final_val_digest: [u8; 32] = rng.random();
    let pp_id_digest: [u8; 32] = rng.random();

    let expected = neo_fold::bridge_digests::compute_obligations_digest_v2(
        acc_final_main_digest,
        acc_final_val_digest,
        pp_id_digest,
    );

    let mut cs = TestConstraintSystem::<CircuitF>::new();
    let mut sponge = Poseidon2Sponge::new(&mut cs, "obligations_digest_v2").expect("Poseidon2Sponge::new");

    for (i, &b) in b"neo/spartan-bridge/obligations_digest/v2".iter().enumerate() {
        let x =
            AllocatedNum::alloc(cs.namespace(|| format!("dst_byte_{i}")), || Ok(CircuitF::from(b as u64))).unwrap();
        sponge.absorb(&mut cs, x).unwrap();
    }

    let mut absorb_digest_u32 = |label: &str, d: &[u8; 32]| {
        for (i, chunk) in d.chunks_exact(4).enumerate() {
            let mut limb = [0u8; 4];
            limb.copy_from_slice(chunk);
            let u = u32::from_le_bytes(limb) as u64;
            let x = AllocatedNum::alloc(cs.namespace(|| format!("{label}_u32_{i}")), || Ok(CircuitF::from(u))).unwrap();
            sponge.absorb(&mut cs, x).unwrap();
        }
    };

    absorb_digest_u32("acc_main", &acc_final_main_digest);
    absorb_digest_u32("acc_val", &acc_final_val_digest);
    absorb_digest_u32("pp_id", &pp_id_digest);

    let out = sponge.digest32(&mut cs, "digest32").unwrap();

    for i in 0..4 {
        let mut limb = [0u8; 8];
        limb.copy_from_slice(&expected[i * 8..(i + 1) * 8]);
        let u = u64::from_le_bytes(limb);

        let expected_i =
            AllocatedNum::alloc(cs.namespace(|| format!("expected_{i}")), || Ok(CircuitF::from(u))).unwrap();
        constrain_equal(&mut cs, &out[i], &expected_i, &format!("digest32_{i}"));
    }

    assert!(
        cs.is_satisfied(),
        "Poseidon2 sponge digest32 must match compute_obligations_digest_v2"
    );
}
