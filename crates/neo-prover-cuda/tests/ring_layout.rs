//! Round-trip and equivalence checks for the flat ring-column layout (no GPU).

use neo_math::{D, F};
use neo_prover_cuda::kernels::goldilocks::GOLDILOCKS_MODULUS;
use neo_prover_cuda::ring_layout::{assignment_to_mat, assignment_to_words, mat_from_words, mat_to_words};
use p3_field::PrimeCharacteristicRing;
use rand::{Rng, SeedableRng};

#[test]
fn words_and_mat_views_of_an_assignment_agree() {
    let cols = 37;
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let z: Vec<F> = (0..cols * D - 11)
        .map(|_| F::from_u64(rng.random::<u64>() % GOLDILOCKS_MODULUS))
        .collect();

    let mat = assignment_to_mat(&z, cols);
    let words = assignment_to_words(&z, cols);
    assert_eq!(
        mat_to_words(&mat),
        words,
        "flat layout must equal the padded assignment"
    );
    assert_eq!(mat_from_words(&words, cols), mat, "words → mat must invert mat → words");
}
