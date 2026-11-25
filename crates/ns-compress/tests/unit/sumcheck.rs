//! SPEC.md Algorithm 9, steps 8 and 10.

use neo_math::{F, K};
use neo_reductions::sumcheck::poly_eval_k;
use neo_transcript::Poseidon2Transcript;
use p3_field::{Field, PrimeCharacteristicRing};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::{
    lookup::{prove, Slot},
    prove_product,
};
use crate::layout::{Layout, Slice};
use crate::support::{equality, random_extension};
use crate::transcript::{absorb_extension, challenge};
use crate::weights::Challenges;

fn direct([z, g, u, e, s]: [K; 5], ch: &Challenges) -> K {
    let factors = ch.taus.map(|t| K::from(t) - z);
    let denominator = factors.into_iter().product::<K>();
    let numerator: K = ch
        .psi
        .iter()
        .enumerate()
        .map(|(k, weight)| {
            *weight
                * factors
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != k)
                    .map(|(_, factor)| *factor)
                    .product::<K>()
        })
        .sum();
    z * g + u * (e * denominator + s * numerator)
}

fn check_first_rounds(slices: &[(usize, usize)], rounds: usize, zeros: bool) {
    let mut rng = ChaCha8Rng::seed_from_u64(1907);
    let width = 1 << rounds;
    let span = slices
        .iter()
        .map(|&(offset, vars)| offset + (1 << vars))
        .max()
        .unwrap();
    let span_vars = span.next_power_of_two().trailing_zeros() as usize;
    let layout = Layout {
        cells: Vec::new(),
        key_vars: 20,
        witness_vars: rounds,
        witness_len: width,
        table_witness_vars: span_vars,
        table_witness_len: span,
        inverse_vars: 16,
    };
    let mut random = || random_extension(&mut rng);
    let ch = Challenges {
        taus: [19, 19, 113, 37].map(F::from_u64),
        omega: random(),
        delta: random(),
        mu: random(),
        gamma: random(),
        psi: std::array::from_fn(|_| random()),
        xi: slices.iter().map(|_| random()).collect(),
        lambda: slices.iter().map(|_| random()).collect(),
        beta: (0..span_vars).map(|_| random()).collect(),
    };
    let mut tables: [Vec<K>; 5] = std::array::from_fn(|_| vec![K::ZERO; width]);
    let weights: Vec<K> = (0..width).map(|_| random()).collect();
    let mut witness: Vec<F> = (0..width).map(|_| F::from_u64(rng.next_u64())).collect();
    let mut slots = Vec::new();
    for (i, &(offset, vars)) in slices.iter().enumerate() {
        let slice = Slice { offset, vars };
        let codes: Vec<u16> = (0..slice.len())
            .map(|_| if zeros { 3 } else { (rng.next_u64() % 7) as u16 })
            .collect();
        let inverses: Vec<_> = (0..7)
            .map(|c| {
                ch.taus
                    .iter()
                    .map(|t| *t - (F::from_u64(c) - F::from_u64(3)))
                    .product::<F>()
                    .inverse()
            })
            .collect();
        for (index, &code) in slice.range().zip(&codes) {
            witness[index] = F::from_u16(code) - F::from_u64(3);
            tables[2][index] = K::from(inverses[usize::from(code)]);
            tables[3][index] = ch.xi[i] * equality(&ch.beta[..vars], index - offset);
            tables[4][index] = ch.lambda[i];
        }
        slots.push(Slot::new(
            slice,
            3,
            &codes,
            &inverses,
            [ch.xi[i], ch.lambda[i]],
            &witness[slice.range()],
            &ch.beta,
        ));
    }
    tables[0] = witness.iter().copied().map(K::from).collect();
    tables[1] = weights.clone();
    let sum = (0..width)
        .map(|i| direct(std::array::from_fn(|j| tables[j][i]), &ch))
        .sum();
    let (messages, rho, w, u) = prove(
        &mut Poseidon2Transcript::new_v1_2(),
        witness,
        weights,
        slots,
        &ch,
        &layout,
        sum,
        None,
        None::<fn(&[K]) -> Result<K, crate::Error>>,
    )
    .unwrap();
    for (bit, (&coordinate, message)) in rho.iter().zip(messages).enumerate() {
        if bit == span_vars {
            assert_eq!(u, tables[2][0]);
        }
        for t in (0..7).map(K::from_u64) {
            let expected = (0..tables[0].len() / 2)
                .map(|i| {
                    direct(
                        std::array::from_fn(|j| {
                            let a = tables[j][2 * i];
                            a + t * (tables[j][2 * i + 1] - a)
                        }),
                        &ch,
                    )
                })
                .sum();
            assert_eq!(poly_eval_k(&message, t), expected, "round {bit}");
        }
        for column in &mut tables {
            *column = column
                .chunks_exact(2)
                .map(|pair| pair[0] + coordinate * (pair[1] - pair[0]))
                .collect();
        }
    }
    assert_eq!(w, tables[0][0]);
    if span_vars == rounds {
        assert_eq!(u, tables[2][0]);
    }
}

#[test]
fn shared_lookup_rounds_match_the_full_polynomial() {
    check_first_rounds(&[(0, 2)], 4, true);
    check_first_rounds(&[(0, 4), (24, 2)], 5, false);
    check_first_rounds(&[(0, 7), (128, 6)], 9, false);
    check_first_rounds(&[(0, 15)], 16, false);
}

fn check_product(values: Vec<K>, lanes: Vec<K>, blocks: Vec<K>, rounds: usize) {
    let weights: Vec<K> = blocks
        .iter()
        .flat_map(|&block| lanes.iter().map(move |&lane| lane * block))
        .collect();
    let cells = 1 << rounds;
    let pad = |table: &[K]| -> Vec<K> {
        (0..cells)
            .map(|cell| table.get(cell).copied().unwrap_or(K::ZERO))
            .collect()
    };
    let mut tables = [pad(&values), pad(&weights)];
    let sum: K = (0..cells)
        .map(|cell| tables[0][cell] * tables[1][cell])
        .sum();
    let mut transcript = Poseidon2Transcript::new_v1_2();
    let (mut claim, mut messages, mut point) = (sum, Vec::new(), Vec::new());
    for _ in 0..rounds {
        let at = |x: u64| -> K {
            let x = K::from_u64(x);
            (0..tables[0].len() / 2)
                .map(|pair| {
                    let [value, weight] = [&tables[0], &tables[1]].map(|table| {
                        let (low, high) = (table[2 * pair], table[2 * pair + 1]);
                        low + x * (high - low)
                    });
                    value * weight
                })
                .sum()
        };
        let values = [at(0), at(1), at(2)];
        assert_eq!(values[0] + values[1], claim);
        let quadratic = (values[2] - values[1].double() + values[0]).halve();
        let message = [values[0], values[1] - values[0] - quadratic, quadratic];
        absorb_extension(&mut transcript, &message);
        let challenge = challenge(&mut transcript);
        claim = poly_eval_k(&message, challenge);
        for table in &mut tables {
            *table = table
                .chunks(2)
                .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
                .collect();
        }
        messages.push(message);
        point.push(challenge);
    }
    assert_eq!(claim, tables[0][0] * tables[1][0]);
    let proved = prove_product(
        &mut Poseidon2Transcript::new_v1_2(),
        values,
        &lanes,
        &blocks,
        sum,
        rounds,
        |_, _| {},
    );
    assert!(
        proved == (messages, point, tables[0][0]),
        "the key rounds differ from their definition"
    );
}

#[test]
fn the_key_sum_check_is_its_definition() {
    let mut rng = ChaCha8Rng::seed_from_u64(10);
    let mut random = |cells: usize| -> Vec<K> { (0..cells).map(|_| random_extension(&mut rng)).collect() };
    check_product(random(128), random(64), random(2), 7);
    check_product(random(256), random(64), random(3), 9);
    check_product(random(36), random(4), random(9), 6);
    check_product(random(37), random(1), random(21), 6);
    check_product(random(1 << 15), random(64), random(1 << 9), 15);
}
