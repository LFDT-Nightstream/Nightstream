//! The reduction of a ring equation to an inner product (`neo_compress::ring`)
//! against schoolbook arithmetic in `K[X]/(X^64 + 1)`.

mod support;

use neo_compress::ring::{conjugate, Point};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use support::{random_extension, ring_product};

fn random_element(rng: &mut ChaCha8Rng) -> [K; D] {
    std::array::from_fn(|_| random_extension(rng))
}

/// `Σ_j P_j · z_j = Y` gives `Σ_j Σ_h z_j[h] · lanes(P_j)[h] = Y(δ)` at each
/// `δ`, and another `Y` does not.
#[test]
fn a_ring_equation_is_an_inner_product_at_each_point() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for blocks in [1, 2, 7] {
        let coefficients: Vec<[K; D]> = (0..blocks).map(|_| random_element(&mut rng)).collect();
        let witness: Vec<[F; D]> = (0..blocks)
            .map(|_| std::array::from_fn(|_| F::from_u64(rng.next_u64() % 5) - F::TWO))
            .collect();
        let mut product = [K::ZERO; D];
        for (coefficient, block) in coefficients.iter().zip(&witness) {
            let block = block.map(K::from);
            for (sum, term) in product.iter_mut().zip(ring_product(coefficient, &block)) {
                *sum += term;
            }
        }
        for _ in 0..4 {
            let point = Point::new(random_extension(&mut rng));
            let inner: K = coefficients
                .iter()
                .zip(&witness)
                .flat_map(|(coefficient, block)| {
                    point
                        .lanes(coefficient)
                        .into_iter()
                        .zip(block)
                        .map(|(weight, &cell)| weight * cell)
                })
                .sum();
            assert_eq!(inner, point.value(&product), "{blocks} blocks");
            let mut other = product;
            other[rng.next_u32() as usize % D] += K::ONE;
            assert_ne!(inner, point.value(&other), "{blocks} blocks, changed product");
        }
    }
}

/// `lanes(P)[h]` is the value at `δ` of `X^h · P`.
#[test]
fn a_lane_weight_is_the_value_of_the_shifted_element() {
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let element = random_element(&mut rng);
    for delta in [K::ZERO, K::ONE, random_extension(&mut rng)] {
        let point = Point::new(delta);
        let lanes = point.lanes(&element);
        for (lane, weight) in lanes.into_iter().enumerate() {
            let mut monomial = [K::ZERO; D];
            monomial[lane] = K::ONE;
            assert_eq!(weight, point.value(&ring_product(&monomial, &element)), "lane {lane}");
        }
    }
}

/// `conjugate(a) · z` has the constant term `Σ_h a[h] · z[h]`.
#[test]
fn the_conjugate_product_has_the_inner_product_as_its_constant_term() {
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let (left, right) = (random_element(&mut rng), random_element(&mut rng));
    let inner: K = left
        .iter()
        .zip(&right)
        .map(|(&left, &right)| left * right)
        .sum();
    assert_eq!(ring_product(&conjugate(&left), &right)[0], inner);
    assert_eq!(conjugate(&left), support::conjugate(&left));
}
