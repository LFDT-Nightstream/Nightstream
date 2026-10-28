#![cfg(feature = "metal")]

use neo_ccs::crypto::poseidon2_goldilocks as p2;
use neo_math::{D, F};
use neo_prover_metal::MetalSession;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand::{Rng, SeedableRng};

const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;

#[test]
fn metal_merkle_trees_match_cpu_sponge() {
    let session = MetalSession::new().expect("Metal session");
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x004d_4554_414c);
    // Merkle trees: leaves by the overwrite sponge (partial last blocks,
    // non-canonical words), nodes by the same sponge over both children.
    let sponge = |words: &[u64]| -> [u64; 4] {
        let mut state = [F::ZERO; p2::WIDTH];
        for block in words.chunks(p2::RATE) {
            for (lane, &word) in state.iter_mut().zip(block) {
                *lane = F::from_u64(word);
            }
            state = p2::permute_state(state);
        }
        std::array::from_fn(|lane| state[lane].as_canonical_u64())
    };
    for (rows, width) in [
        (256, 1),
        (256, 5),
        (256, 12),
        (256, 13),
        (512, 16),
        (128, 24),
        (256, 48),
        (1, 7),
    ] {
        let fields = (0..rows * width)
            .map(|_| rng.random::<u64>())
            .collect::<Vec<_>>();
        let layers = session
            .poseidon2_merkle_layers(rows, width, |words| words.copy_from_slice(&fields))
            .expect("Poseidon2 Metal Merkle tree");
        let mut expected: Vec<[u64; 4]> = fields.chunks_exact(width).map(sponge).collect();
        let leaves = session
            .poseidon2_merkle_leaves(&fields, width)
            .expect("Poseidon2 Metal Merkle leaves");
        assert_eq!(leaves, expected, "leaves of rows {rows} width {width}");
        for layer in &layers {
            assert_eq!(layer, &expected, "rows {rows} width {width}");
            expected = expected
                .chunks_exact(2)
                .map(|pair| sponge(&[pair[0], pair[1]].concat()))
                .collect();
        }
        assert_eq!(layers.last().map(Vec::len), Some(1), "rows {rows} width {width}");
    }
    // Leaves of rows on whole pages are read in place: only the width word
    // is uploaded. Rows one row off the page are copied.
    let page = std::alloc::Layout::from_size_align(1 << 16, 1 << 14).unwrap();
    let pointer = unsafe { std::alloc::alloc(page) }.cast::<u64>();
    assert!(!pointer.is_null());
    let words = unsafe { std::slice::from_raw_parts_mut(pointer, page.size() / 8) };
    words
        .iter_mut()
        .for_each(|word| *word = rng.random::<u64>());
    for (start, len, in_place) in [(0, words.len(), true), (16, words.len() / 2, false)] {
        let rows = &words[start..start + len];
        let uploaded = session.activity().uploaded_bytes;
        let leaves = session
            .poseidon2_merkle_leaves(rows, 16)
            .expect("Poseidon2 Metal Merkle leaves");
        let uploaded = session.activity().uploaded_bytes - uploaded;
        assert_eq!(leaves, rows.chunks_exact(16).map(sponge).collect::<Vec<_>>());
        assert_eq!(uploaded == 8, in_place, "uploaded {uploaded} bytes");
    }
    unsafe { std::alloc::dealloc(pointer.cast(), page) };
}

#[test]
fn metal_key_weights_match_cpu_key_rows() {
    use neo_ajtai::nightstream_fprime_setup::{coefficient_block, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS};
    let prover = neo_prover_metal::MetalRowProver::new().expect("Metal prover");
    let mut rng = rand::rngs::StdRng::seed_from_u64(91);
    let rows = PRODUCTION_VERIFIER_ROWS as usize;
    let columns = 5;
    let powers: Vec<[u64; 3]> = (0..rows)
        .map(|_| std::array::from_fn(|_| rng.random::<u64>() % GOLDILOCKS_MODULUS))
        .collect();
    let actual = prover
        .production_key_weights(&powers, columns)
        .expect("Metal key weights");
    assert_eq!(actual.len(), columns * D);
    for column in 0..columns {
        let keys: Vec<[u64; D]> = (0..rows)
            .map(|row| coefficient_block(&PRODUCTION_SEED, row as u32, column as u64))
            .collect();
        for lane in 0..D {
            let expected: [u64; 3] = std::array::from_fn(|c| {
                (0..rows)
                    .map(|row| F::from_u64(powers[row][c]) * F::from_u64(keys[row][lane]))
                    .sum::<F>()
                    .as_canonical_u64()
            });
            assert_eq!(actual[column * D + lane], expected, "column {column} lane {lane}");
        }
    }
}

/// The radix-2 DFT of `values` at the powers of `omega`, recursively.
fn fft(values: &[F], omega: F) -> Vec<F> {
    if values.len() == 1 {
        return values.to_vec();
    }
    let half = values.len() / 2;
    let split = |parity: usize| {
        values
            .iter()
            .skip(parity)
            .step_by(2)
            .copied()
            .collect::<Vec<_>>()
    };
    let (even, odd) = (fft(&split(0), omega * omega), fft(&split(1), omega * omega));
    let mut out = vec![F::ZERO; values.len()];
    let mut twiddle = F::ONE;
    for k in 0..half {
        out[k] = even[k] + twiddle * odd[k];
        out[k + half] = even[k] - twiddle * odd[k];
        twiddle *= omega;
    }
    out
}

#[test]
fn metal_dft_matches_cpu_fft() {
    use p3_field::TwoAdicField;
    let session = MetalSession::new().expect("Metal session");
    let mut rng = rand::rngs::StdRng::seed_from_u64(17);
    // One pass, two passes and three passes; narrow and wide rows.
    for (log_rows, width) in [(1, 1), (3, 5), (8, 16), (9, 48), (10, 32), (17, 16)] {
        let rows = 1usize << log_rows;
        let values: Vec<u64> = (0..rows * width).map(|_| rng.random::<u64>()).collect();
        let omega = F::two_adic_generator(log_rows);
        let mut actual = values.clone();
        session
            .dft_columns(&mut actual, width, omega.as_canonical_u64())
            .expect("Metal DFT");
        for column in 0..width {
            let input: Vec<F> = (0..rows)
                .map(|r| F::from_u64(values[r * width + column]))
                .collect();
            for (row, expected) in fft(&input, omega).into_iter().enumerate() {
                assert_eq!(
                    actual[row * width + column],
                    expected.as_canonical_u64(),
                    "2^{log_rows} rows, width {width}, row {row}, column {column}"
                );
            }
        }
    }
    // Past 2^24 rows a pass of one stage would launch 2^32 threads; spot
    // check rows of a 2^25-row DFT by direct evaluation.
    let log_rows = 25;
    let values: Vec<u64> = (0..1u64 << log_rows).map(|_| rng.random::<u64>()).collect();
    let omega = F::two_adic_generator(log_rows);
    let mut actual = values.clone();
    session
        .dft_columns(&mut actual, 1, omega.as_canonical_u64())
        .expect("Metal DFT");
    for row in [0usize, 1, 3, 1 << 24, (1 << 25) - 1] {
        let point = omega.exp_u64(row as u64);
        let mut power = F::ONE;
        let mut expected = F::ZERO;
        for &value in &values {
            expected += F::from_u64(value) * power;
            power *= point;
        }
        assert_eq!(actual[row], expected.as_canonical_u64(), "2^25 rows, row {row}");
    }
}

type E3 = [F; 3];

fn e3_words(value: E3) -> [u64; 3] {
    value.map(|c| c.as_canonical_u64())
}

fn e3_add(a: E3, b: E3) -> E3 {
    std::array::from_fn(|c| a[c] + b[c])
}

fn e3_sub(a: E3, b: E3) -> E3 {
    std::array::from_fn(|c| a[c] - b[c])
}

/// `F[X] / (X^3 - X - 1)`.
fn e3_mul(a: E3, b: E3) -> E3 {
    let mut product = [F::ZERO; 5];
    for i in 0..3 {
        for j in 0..3 {
            product[i + j] += a[i] * b[j];
        }
    }
    // X^3 = X + 1, X^4 = X^2 + X.
    [
        product[0] + product[3],
        product[1] + product[3] + product[4],
        product[2] + product[4],
    ]
}

fn e3_scale(a: E3, x: u64) -> E3 {
    a.map(|c| c * F::from_u64(x))
}

/// One GKR layer on the CPU, as the prover computes it: the round messages,
/// then each tree's children.
fn gkr_layer_reference(
    tables: &mut [Vec<Vec<E3>>],
    factors: &[usize],
    point: &[E3],
    weights: &[E3],
    mu: E3,
    degree: usize,
    challenges: &[E3],
) -> (Vec<Vec<E3>>, Vec<Vec<E3>>) {
    let mut eq = vec![[F::ONE, F::ZERO, F::ZERO]];
    for &c in point {
        let high: Vec<E3> = eq.iter().map(|&v| e3_mul(v, c)).collect();
        let low: Vec<E3> = eq.iter().zip(&high).map(|(&v, &h)| e3_sub(v, h)).collect();
        eq = [low, high].concat();
    }
    let nodes: Vec<u64> = std::iter::once(0).chain(2..=degree as u64).collect();
    let line = |table: &[E3], at: usize, x: u64| e3_add(table[at], e3_scale(e3_sub(table[at + 1], table[at]), x));
    let mut messages = Vec::new();
    for &r in challenges {
        let message = nodes
            .iter()
            .map(|&x| {
                let mut sum = [F::ZERO; 3];
                for m in 0..eq.len() / 2 {
                    let mut total = [F::ZERO; 3];
                    for ((parts, &f), &weight) in tables.iter().zip(factors).zip(weights) {
                        let half = parts[0].len() / 2;
                        let child = |part: usize, b: usize| line(&parts[part], b * half + 2 * m, x);
                        let p =
                            |b: usize| (0..f).fold([F::ONE, F::ZERO, F::ZERO], |acc, part| e3_mul(acc, child(part, b)));
                        let (q0, q1) = (child(f, 0), child(f, 1));
                        let term = e3_add(e3_add(e3_mul(p(0), q1), e3_mul(p(1), q0)), e3_mul(e3_mul(mu, q0), q1));
                        total = e3_add(total, e3_mul(weight, term));
                    }
                    sum = e3_add(sum, e3_mul(line(&eq, 2 * m, x), total));
                }
                sum
            })
            .collect();
        messages.push(message);
        let fold = |table: &mut Vec<E3>| {
            *table = table
                .chunks_exact(2)
                .map(|pair| e3_add(pair[0], e3_mul(r, e3_sub(pair[1], pair[0]))))
                .collect();
        };
        fold(&mut eq);
        for parts in tables.iter_mut() {
            parts.iter_mut().for_each(fold);
        }
    }
    let children = tables
        .iter()
        .map(|parts| parts.iter().flat_map(|part| [part[0], part[1]]).collect())
        .collect();
    (messages, children)
}

#[test]
fn metal_gkr_layers_match_cpu() {
    let session = MetalSession::new().expect("Metal session");
    let mut rng = rand::rngs::StdRng::seed_from_u64(29);
    let mut random = || -> E3 { std::array::from_fn(|_| F::from_u64(rng.random::<u64>())) };
    // (depth, factors): trees ending at different layers, with zero to two
    // factors. Layers from 14 up have tables of whole pages.
    let shapes = [(16usize, 0usize), (16, 2), (15, 1), (14, 0)];
    let mut levels: Vec<Vec<Vec<Vec<E3>>>> = shapes
        .iter()
        .map(|&(depth, factors)| {
            let leaves: Vec<Vec<E3>> = (0..=factors)
                .map(|_| (0..1 << depth).map(|_| random()).collect())
                .collect();
            let mut tree = vec![leaves];
            while tree.last().unwrap()[0].len() > 1 {
                let child = tree.last().unwrap();
                let f = child.len() - 1;
                let half = child[0].len() / 2;
                let numerator =
                    |y: usize| (0..f).fold([F::ONE, F::ZERO, F::ZERO], |acc, part| e3_mul(acc, child[part][y]));
                let (p, q) = (0..half)
                    .map(|y| {
                        let (q0, q1) = (child[f][y], child[f][y + half]);
                        (
                            e3_add(e3_mul(numerator(y), q1), e3_mul(numerator(y + half), q0)),
                            e3_mul(q0, q1),
                        )
                    })
                    .unzip();
                tree.push(vec![p, q]);
            }
            tree.reverse();
            tree
        })
        .collect();
    let words = |values: &[E3]| {
        values
            .iter()
            .flat_map(|&v| e3_words(v))
            .collect::<Vec<u64>>()
    };
    for k in 14..=16 {
        let active: Vec<usize> = (0..shapes.len()).filter(|&t| shapes[t].0 >= k).collect();
        let factors: Vec<usize> = active
            .iter()
            .map(|&t| if shapes[t].0 == k { shapes[t].1 } else { 1 })
            .collect();
        let degree = factors.iter().map(|f| f + 2).max().unwrap().max(3);
        let point: Vec<E3> = (1..k).map(|_| random()).collect();
        let (mu, weights) = (random(), active.iter().map(|_| random()).collect::<Vec<_>>());
        let challenges: Vec<E3> = (1..k).map(|_| random()).collect();
        let mut tables: Vec<Vec<Vec<E3>>> = active
            .iter()
            .map(|&t| std::mem::take(&mut levels[t][k]))
            .collect();
        let mut host: Vec<Vec<Vec<u64>>> = tables
            .iter()
            .map(|parts| parts.iter().map(|part| words(part)).collect())
            .collect();
        let mut eq = vec![[F::ONE, F::ZERO, F::ZERO]];
        for &c in &point {
            let high: Vec<E3> = eq.iter().map(|&v| e3_mul(v, c)).collect();
            let low: Vec<E3> = eq.iter().zip(&high).map(|(&v, &h)| e3_sub(v, h)).collect();
            eq = [low, high].concat();
        }
        let mut eq_words = words(&eq);
        let (messages, children) =
            gkr_layer_reference(&mut tables, &factors, &point, &weights, mu, degree, &challenges);
        {
            let trees: Vec<Vec<&mut [u64]>> = host
                .iter_mut()
                .map(|parts| parts.iter_mut().map(Vec::as_mut_slice).collect())
                .collect();
            let mut layer = session
                .gkr_layer(trees, &mut eq_words)
                .expect("the device takes a layer of whole pages");
            let weight_words: Vec<[u64; 3]> = weights.iter().map(|&w| e3_words(w)).collect();
            for (round, (message, &r)) in messages.iter().zip(&challenges).enumerate() {
                let actual = session
                    .gkr_message(&layer, &weight_words, e3_words(mu), degree)
                    .expect("Metal GKR message");
                let expected: Vec<[u64; 3]> = message.iter().map(|&v| e3_words(v)).collect();
                assert_eq!(actual, expected, "layer {k} round {round}");
                session
                    .gkr_fold(&mut layer, e3_words(r))
                    .expect("Metal GKR fold");
            }
        }
        // The folded host tables hold logical entry j at element j·2^(k-1).
        let stride = 1 << (k - 1);
        for (parts, expected) in host.iter().zip(&children) {
            let actual: Vec<[u64; 3]> = parts
                .iter()
                .flat_map(|part| [0, stride].map(|at| std::array::from_fn(|c| part[3 * at + c])))
                .collect();
            let expected: Vec<[u64; 3]> = expected.iter().map(|&v| e3_words(v)).collect();
            assert_eq!(actual, expected, "layer {k} children");
        }
    }
}
