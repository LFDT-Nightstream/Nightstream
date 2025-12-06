#![allow(non_snake_case, unused_variables)]
use neo_ajtai::*;
use neo_math::ring::{cf_inv, Rq as RqEl};
use neo_math::s_action::SAction;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn sample_Z(d: usize, m: usize, b: u32) -> Vec<Fq> {
    // random digits in {0,1} (b=2) to exercise pay-per-bit path
    let mut rng = ChaCha20Rng::seed_from_u64(7);
    (0..d * m)
        .map(|_| if rng.next_u64() & 1 == 1 { Fq::ONE } else { Fq::ZERO })
        .collect()
}

#[test]
fn s_module_homomorphism() {
    let d = 54usize;
    let m = 64usize;
    let kappa = 8usize;
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let pp = setup(&mut rng, d, kappa, m).expect("Setup should succeed");
    let Z1 = sample_Z(d, m, 2);
    let Z2 = sample_Z(d, m, 2);
    let c1 = commit(&pp, &Z1);
    let c2 = commit(&pp, &Z2);

    // Sample ρ1, ρ2 as rotations of random small ring elements
    let small = |val: i32| -> RqEl {
        let coeffs: [Fq; neo_math::ring::D] = (0..neo_math::ring::D)
            .map(|i| {
                if i == 0 {
                    Fq::from_u64(val.rem_euclid(3) as u64)
                } else {
                    Fq::ZERO
                }
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        cf_inv(coeffs)
    };
    let rho1 = small(1);
    let rho2 = small(2);

    // ρ1 L(Z1)+ ρ2 L(Z2)
    let left = {
        let t1 = s_mul(&rho1, &c1);
        let t2 = s_mul(&rho2, &c2);
        let mut sum = t1.clone();
        sum.add_inplace(&t2);
        sum
    };

    // L(ρ1 Z1 + ρ2 Z2)
    let right = {
        // Apply ρ·Z in coefficient space (matrix * matrix by columns)
        let mut comb = vec![Fq::ZERO; d * m];
        let mut apply = |rho: &RqEl, Z: &Vec<Fq>| {
            let s_action = SAction::from_ring(*rho);
            for col in 0..m {
                let src: [Fq; neo_math::ring::D] = Z[col * d..(col + 1) * d].try_into().unwrap();
                let dst_slice = &mut comb[col * d..(col + 1) * d];
                let result = s_action.apply_vec(&src);
                for (d, &r) in dst_slice.iter_mut().zip(&result) {
                    *d += r;
                }
            }
        };
        apply(&rho1, &Z1);
        apply(&rho2, &Z2);
        commit(&pp, &comb)
    };

    assert_eq!(left, right, "S-homomorphism failed");
}
