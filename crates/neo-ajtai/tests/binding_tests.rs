#![allow(non_snake_case)]
use neo_ajtai::*;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn random_Z(rng: &mut ChaCha20Rng, d: usize, m: usize, b: u32) -> Vec<Fq> {
    (0..d * m)
        .map(|_| {
            let x: u8 = rng.random::<u8>() % (2 * b as u8); // small digit in [0..2b-1]
            Fq::from_u64((x as u64).min((b - 1) as u64))
        })
        .collect()
}

#[test]
fn binding_harness_no_collision() {
    let d = 54usize;
    let m = 64usize;
    let kappa = 16usize;
    let b = 2u32;
    let mut rng = ChaCha20Rng::seed_from_u64(99);
    let pp = setup(&mut rng, d, kappa, m).expect("Setup should succeed");

    // small trial budget; collisions would imply MSIS break or luck
    for _ in 0..64 {
        let Z1 = random_Z(&mut rng, d, m, b);
        let mut Z2 = random_Z(&mut rng, d, m, b);
        if Z1 == Z2 {
            Z2[0] += Fq::ONE;
        } // ensure Z1≠Z2
        let c1 = commit(&pp, &Z1);
        let c2 = commit(&pp, &Z2);
        assert_ne!(c1, c2, "binding harness found collision (extremely unlikely)");
    }
}
