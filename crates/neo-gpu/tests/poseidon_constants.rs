use neo_ccs::crypto::poseidon2_goldilocks as p2;
use p3_field::PrimeCharacteristicRing;
use p3_field::PrimeField64;
use p3_symmetric::Permutation;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
#[ignore = "helper for generating Mojo Poseidon2 constants and vectors"]
fn dump_poseidon2_width8_constants_and_vectors() {
    let mut rng = ChaCha8Rng::from_seed(p2::SEED);
    let rounds = p3_poseidon2::poseidon2_round_numbers_128::<p3_goldilocks::Goldilocks>(p2::WIDTH, 7)
        .expect("width-8 round numbers");
    println!("rounds_f={} rounds_p={}", rounds.0, rounds.1);

    let external = p3_poseidon2::ExternalLayerConstants::<p3_goldilocks::Goldilocks, { p2::WIDTH }>::new_from_rng(
        rounds.0, &mut rng,
    );
    println!(
        "initial={:#?}\nterminal={:#?}",
        external.get_initial_constants(),
        external.get_terminal_constants()
    );

    let internal: Vec<_> = (0..rounds.1)
        .map(|_| rng.random::<p3_goldilocks::Goldilocks>().as_canonical_u64())
        .collect();
    println!("internal={internal:#x?}");

    for state in [
        [0u64; p2::WIDTH],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [3, 5, 7, 11, 13, 17, 19, 23],
    ] {
        let out = p2::permutation()
            .permute(state.map(p3_goldilocks::Goldilocks::from_u64))
            .map(|x| x.as_canonical_u64());
        println!("state={state:?}\nout={out:?}");
    }
}
