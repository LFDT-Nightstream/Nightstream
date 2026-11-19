use p3_field::PrimeField64;

use super::*;

fn canonical_key_word(index: u64) -> u64 {
    let q = Fq::ORDER_U64;
    match index % 11 {
        0 => 0,
        1 => 1,
        2 => q / 2,
        3 => q / 2 + 1,
        4 => q - 1,
        5 => q - 2,
        _ => index.wrapping_mul(0x9e3779b97f4a7c15) % q,
    }
}

fn scalar(blocks: &[SignedBlock], key: Option<&ExpandedKey>) -> Commitment {
    let mut expected = Commitment::zeros(PRODUCTION_VERIFIER_ROWS as usize);
    for row in 0..PRODUCTION_VERIFIER_ROWS as u32 {
        expected
            .col_mut(row as usize)
            .copy_from_slice(&signed_sums(row, blocks, key).reduce());
    }
    expected
}

fn bytes(commitment: &Commitment) -> Vec<u8> {
    commitment
        .data()
        .iter()
        .flat_map(|word| word.as_canonical_u64().to_le_bytes())
        .collect()
}

#[test]
fn crt_signed_unit_tiles_preserve_commitment_bytes_and_fallbacks() {
    let columns = 2 * TASK_COLUMNS;
    let key = ExpandedKey::of_pairs(columns, |row, pair| {
        std::array::from_fn(|within| {
            std::array::from_fn(|lane| {
                canonical_key_word((u64::from(row) * columns + 2 * pair + within as u64) * D as u64 + lane as u64)
            })
        })
    });
    let signed = |index, bits: u64| {
        let positive = bits.rotate_left(index as u32) & 0xd8f1_8331_73b4_2ec9;
        SignedBlock {
            index,
            positive,
            negative: bits.rotate_left(index as u32) & !positive,
        }
    };
    let mut blocks: Vec<_> = (0..columns)
        .filter(|index| index % TASK_COLUMNS >= TASK_COLUMNS / 2)
        .map(|index| signed(index, u64::MAX ^ (index << 32)))
        .collect();
    blocks.push(signed(columns + 7, (1 << 63) | 1));
    for tile in 0..2 {
        let start = tile as u64 * TASK_COLUMNS;
        let local = column_slice(&blocks, &(start..start + TASK_COLUMNS));
        assert!(crt::use_tile(local));
        let mut transformed = crt::zero();
        key.crt_tile(tile)
            .unwrap()
            .add_to(start, local, &mut transformed);
        let expected = scalar(local, Some(&key));
        assert_eq!(transformed.into_iter().flatten().collect::<Vec<_>>(), expected.data());
    }

    let expected = scalar(&blocks, Some(&key));
    let got = commit_signed_blocks_with_key(&blocks, Some(&key));
    assert_eq!(bytes(&got), bytes(&expected));
    assert!(key.crt_tile((columns / TASK_COLUMNS) as usize).is_none());

    let sparse = [signed(0, 1 << 63), signed(TASK_COLUMNS + 7, 1), signed(columns + 15, 3)];
    assert!(!crt::use_tile(&sparse));
    assert_eq!(
        bytes(&commit_signed_blocks_with_key(&sparse, Some(&key))),
        bytes(&scalar(&sparse, Some(&key)))
    );
    assert_eq!(
        bytes(&commit_signed_blocks_with_key(&sparse, None)),
        bytes(&scalar(&sparse, None))
    );
}

#[test]
fn repeated_crt_batches_match_scalar_on_two_mib_workers() {
    let columns = 2 * TASK_COLUMNS;
    let key = ExpandedKey::of_pairs(columns, |row, pair| {
        std::array::from_fn(|within| {
            std::array::from_fn(|lane| {
                canonical_key_word((u64::from(row) * columns + 2 * pair + within as u64) * D as u64 + lane as u64)
            })
        })
    });
    let blocks: Vec<_> = (0..columns)
        .filter(|index| index % TASK_COLUMNS < 520)
        .map(|index| {
            let positive = 0x9fb1_38d5_a331_320e_u64.rotate_left(index as u32);
            SignedBlock {
                index,
                positive,
                negative: !positive,
            }
        })
        .collect();
    let scalar = scalar(&blocks, Some(&key));
    let expected = bytes(&scalar);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .stack_size(2 * 1024 * 1024)
        .build()
        .unwrap();
    pool.install(|| {
        for _ in 0..4 {
            (0..3).into_par_iter().for_each(|_| {
                let actual = commit_signed_blocks_with_key(&blocks, Some(&key));
                assert_eq!(bytes(&actual), expected);
            });
        }

        let tiles: Vec<_> = (0..2)
            .map(|index| {
                let start = index as u64 * TASK_COLUMNS;
                let slice = column_slice(&blocks, &(start..start + TASK_COLUMNS));
                assert!(crt::use_tile(slice));
                (key.crt_tile(index).unwrap(), start, slice)
            })
            .collect();
        let repetitions = 64;
        let jobs: Vec<_> = (0..repetitions)
            .flat_map(|_| tiles.iter().copied())
            .collect();
        let scale = Fq::from_usize(repetitions);
        let expected: Vec<_> = scalar.data().iter().map(|&word| word * scale).collect();
        (0..3).into_par_iter().for_each(|_| {
            let actual = crt::sum_tiles(&jobs);
            assert_eq!(actual.into_iter().flatten().collect::<Vec<_>>(), expected);
        });
    });
}
