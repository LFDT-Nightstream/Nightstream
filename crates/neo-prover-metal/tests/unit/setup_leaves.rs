use super::*;
use neo_ajtai::nightstream_fprime_setup::{coefficient_block, PRODUCTION_SEED};
use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64, TwoAdicField};

const VARIABLES: usize = 10;

/// Key row `row` on the cube `x = 64·b + lane`, as neo-spartan's `oracle`.
fn key_table(row: u32, blocks: usize) -> Vec<u64> {
    let mut table = vec![0; 1 << VARIABLES];
    for block in 0..blocks {
        for (lane, &value) in coefficient_block(&PRODUCTION_SEED, row, block as u64)
            .iter()
            .enumerate()
        {
            table[LANES * block + lane] = value;
        }
    }
    table
}

/// The monomial coefficients of `table`'s multilinear extension.
fn mobius(table: &[u64]) -> Vec<F> {
    let mut c: Vec<F> = table.iter().map(|&word| F::from_u64(word)).collect();
    for t in 0..VARIABLES {
        for x in 0..c.len() {
            if x >> t & 1 == 1 {
                c[x] = c[x] - c[x ^ (1 << t)];
            }
        }
    }
    c
}

/// The codeword at `position` from its definition: `Σ_x c_x (g·ω^position)^x`.
fn codeword_at(c: &[F], position: usize) -> u64 {
    let point = F::GENERATOR * F::two_adic_generator(VARIABLES + 1).exp_u64(position as u64);
    let mut power = F::ONE;
    let mut sum = F::ZERO;
    for &coefficient in c {
        sum += coefficient * power;
        power *= point;
    }
    sum.as_canonical_u64()
}

#[test]
fn setup_leaves_match_the_codeword_definition() {
    let session = MetalSession::new().unwrap();
    let table: Vec<u64> = (0..1u64 << VARIABLES)
        .map(|x| F::from_u64(x.wrapping_mul(0x9e37_79b9_7f4a_7c15)).as_canonical_u64())
        .collect();
    let generator = F::two_adic_generator(VARIABLES + 1).as_canonical_u64();
    let shift = F::GENERATOR.as_canonical_u64();
    let quarter = 1 << (VARIABLES - 2);
    let leaves = [0u32, 1, 77, quarter as u32 - 1];
    // A partial last block, a table, and a full cube of blocks.
    let oracles = [
        SetupSource::KeyRow { row: 3, blocks: 13 },
        SetupSource::Table(&table),
        SetupSource::KeyRow { row: 21, blocks: 16 },
    ];
    let tables = [key_table(3, 13), table.clone(), key_table(21, 16)];
    let values = session
        .setup_leaves(&oracles, VARIABLES, &leaves, generator, shift)
        .unwrap();
    assert_eq!(values.len(), oracles.len());
    for (oracle, table) in values.iter().zip(&tables) {
        let c = mobius(table);
        for (got, &leaf) in oracle.iter().zip(&leaves) {
            let expected: [u64; LEAF] = std::array::from_fn(|v| codeword_at(&c, leaf as usize + v * quarter));
            assert_eq!(*got, expected, "leaf {leaf}");
        }
    }

    let refused = |oracles: &[SetupSource<'_>], leaves: &[u32]| {
        matches!(
            session.setup_leaves(oracles, VARIABLES, leaves, generator, shift),
            Err(MetalError::Shape(_))
        )
    };
    assert!(refused(&oracles, &[quarter as u32]), "a leaf past the codeword");
    assert!(
        refused(&[SetupSource::KeyRow { row: 3, blocks: 17 }], &leaves),
        "a key row past the cube"
    );
    assert!(
        refused(&[SetupSource::KeyRow { row: 22, blocks: 1 }], &leaves),
        "a row past the key"
    );
    assert!(refused(&[SetupSource::Table(&table[1..])], &leaves), "a short table");
}
