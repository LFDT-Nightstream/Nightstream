//! The first sum-check (SPEC.md Algorithm 9, step 8) against its definition
//! and against the sum-check over full tables of `W`, `G`, `E`, `S`, `A` and
//! `B`, on small layouts, and the key sum-check (step 10) against its
//! definition.

use neo_math::{F, K};
use neo_reductions::sumcheck::poly_eval_k;
use neo_transcript::Poseidon2Transcript;
use p3_field::{Field, PrimeCharacteristicRing};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::{prove, prove_product, Lookup, Round, Stage, Start, Tables};
use crate::layout::Slice;
use crate::lookup::OUTSIDE;
use crate::support::{equality, random_extension};
use crate::transcript::{absorb_extension, challenge};

struct Table {
    slice: Slice,
    xi: K,
    lambda: K,
    inverse: Vec<K>,
    scaled: Vec<K>,
}

struct Case {
    witness: Vec<F>,
    weights: Vec<K>,
    codes: Vec<u16>,
    beta: Vec<K>,
    tables: Vec<Table>,
    rounds: usize,
}

type Run = (Vec<Round>, Vec<K>, K);

impl Case {
    fn random(seed: u64, slices: &[(usize, usize, u64)], cells: usize, rounds: usize) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let table_cells = slices
            .iter()
            .map(|&(offset, vars, _)| offset + (1 << vars))
            .max()
            .unwrap_or(0);
        let mut codes = vec![OUTSIDE; table_cells];
        let tables = slices
            .iter()
            .map(|&(offset, vars, bound)| {
                let values = 2 * bound as usize + 1;
                for code in &mut codes[offset..offset + (1 << vars)] {
                    *code = (rng.next_u64() % values as u64) as u16;
                }
                Table {
                    slice: Slice { offset, vars },
                    xi: random_extension(&mut rng),
                    lambda: random_extension(&mut rng),
                    inverse: (0..values).map(|_| random_extension(&mut rng)).collect(),
                    scaled: (0..values).map(|_| random_extension(&mut rng)).collect(),
                }
            })
            .collect();
        let vars = slices.iter().map(|&(_, vars, _)| vars).max().unwrap_or(0);
        Self {
            witness: (0..cells).map(|_| F::new(rng.next_u64())).collect(),
            weights: (0..cells).map(|_| random_extension(&mut rng)).collect(),
            codes,
            beta: (0..vars).map(|_| random_extension(&mut rng)).collect(),
            tables,
            rounds,
        }
    }

    fn start(&self) -> Start<'_> {
        Start {
            witness: self.witness.clone(),
            weights: self.weights.clone(),
            codes: &self.codes,
            beta: &self.beta,
            lookups: self
                .tables
                .iter()
                .map(|table| Lookup {
                    slice: table.slice,
                    xi: table.xi,
                    lambda: table.lambda,
                    inverse: table.inverse.clone(),
                    scaled: table.scaled.clone(),
                })
                .collect(),
        }
    }

    fn tables(&self, cells: usize, lookup_cells: usize) -> [Vec<K>; 6] {
        let mut tables: [Vec<K>; 6] = std::array::from_fn(|_| vec![K::ZERO; lookup_cells]);
        tables[0] = (0..cells)
            .map(|cell| {
                self.witness
                    .get(cell)
                    .map_or(K::ZERO, |&value| K::from(value))
            })
            .collect();
        tables[1] = (0..cells)
            .map(|cell| self.weights.get(cell).copied().unwrap_or(K::ZERO))
            .collect();
        for table in &self.tables {
            let at = |values: &[K], code: u16| values.get(usize::from(code)).copied().unwrap_or(K::ZERO);
            for cell in table.slice.range() {
                let beta = &self.beta[..table.slice.vars];
                tables[2][cell] = table.xi * equality(beta, cell - table.slice.offset);
                tables[3][cell] = table.lambda;
                tables[4][cell] = at(&table.inverse, self.codes[cell]);
                tables[5][cell] = at(&table.scaled, self.codes[cell]);
            }
        }
        tables
    }
}

fn summand(values: [K; 6]) -> K {
    let [witness, weights, check, sum, inverse, scaled] = values;
    witness * (weights - check * inverse) + check * scaled + sum * inverse
}

fn by_definition(case: &Case) -> (K, Run, K) {
    let cells = 1 << case.rounds;
    let mut tables = case.tables(cells, cells);
    let at = |tables: &[Vec<K>; 6], cell: usize| summand(std::array::from_fn(|table| tables[table][cell]));
    let sum: K = (0..cells).map(|cell| at(&tables, cell)).sum();
    let mut transcript = Poseidon2Transcript::new_v1_2();
    let (mut claim, mut messages, mut point) = (sum, Vec::new(), Vec::new());
    let inverse = |value: u64| F::from_u64(value).inverse();
    for _ in 0..case.rounds {
        let values: [K; 4] = std::array::from_fn(|x| {
            let x = K::from_u64(x as u64);
            (0..tables[0].len() / 2)
                .map(|pair| {
                    summand(std::array::from_fn(|table| {
                        let (low, high) = (tables[table][2 * pair], tables[table][2 * pair + 1]);
                        low + x * (high - low)
                    }))
                })
                .sum()
        });
        assert_eq!(values[0] + values[1], claim);
        let first = values[1] - values[0];
        let second = values[2] - values[1].double() + values[0];
        let third = values[3] - values[2] * F::from_u64(3) + values[1] * F::from_u64(3) - values[0];
        let message = [
            values[0],
            first - second.halve() + third * inverse(3),
            (second - third).halve(),
            third * inverse(6),
        ];
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
    assert_eq!(claim, at(&tables, 0), "the last claim is the summand at the point");
    (sum, (messages, point, tables[0][0]), claim)
}

fn by_full_tables(case: &Case, mut claim: K) -> Run {
    let [witness, weights, check, sum, inverse, scaled_inverse] = case.tables(case.witness.len(), case.codes.len());
    let tables = Tables {
        witness,
        weights,
        check,
        sum,
        inverse,
        scaled_inverse,
    };
    let mut sums = tables.round();
    let mut stage = Stage::Tables(tables);
    let mut transcript = Poseidon2Transcript::new_v1_2();
    let (mut messages, mut point) = (Vec::new(), Vec::new());
    for _ in 0..case.rounds {
        let [zero, two, cubic] = sums;
        let message = super::coefficients(zero, claim - zero, two, cubic);
        absorb_extension(&mut transcript, &message);
        let challenge = challenge(&mut transcript);
        claim = poly_eval_k(&message, challenge);
        (stage, sums) = stage.bind(challenge);
        messages.push(message);
        point.push(challenge);
    }
    (messages, point, stage.witness())
}

fn last_claim(run: &Run) -> K {
    let (messages, point, _) = run;
    poly_eval_k(messages.last().expect("a round"), *point.last().expect("a round"))
}

fn check(case: &Case) {
    let (sum, defined, claim) = by_definition(case);
    let proved = prove(&mut Poseidon2Transcript::new_v1_2(), case.start(), sum, case.rounds);
    assert!(proved == defined, "the rounds differ from their definition");
    assert!(
        by_full_tables(case, sum) == defined,
        "the rounds over full tables differ from their definition"
    );
    assert_eq!(last_claim(&proved), claim);
}

#[test]
fn one_slice_and_cells_without_a_table() {
    check(&Case::random(1, &[(0, 6, 3)], 200, 8));
}

#[test]
fn slices_of_different_sizes() {
    check(&Case::random(2, &[(0, 8, 5), (256, 7, 2), (384, 6, 9)], 701, 11));
}

#[test]
fn a_slice_of_one_variable() {
    check(&Case::random(3, &[(0, 1, 1)], 5, 3));
    check(&Case::random(4, &[(0, 2, 0), (4, 1, 2)], 7, 4));
}

#[test]
fn a_witness_of_one_variable() {
    check(&Case::random(5, &[(0, 1, 1)], 2, 1));
}

#[test]
fn a_zero_witness() {
    let mut case = Case::random(6, &[(0, 7, 4), (128, 6, 4)], 300, 9);
    case.witness.fill(F::ZERO);
    case.codes.fill(4);
    check(&case);
}

#[test]
fn codes_at_their_largest_value() {
    let mut case = Case::random(7, &[(0, 6, 32767), (64, 6, 32767)], 130, 8);
    case.codes.fill(65534);
    case.codes[5] = OUTSIDE;
    case.codes[70] = OUTSIDE;
    check(&case);
}

#[test]
fn a_cell_between_slices_without_a_table() {
    check(&Case::random(8, &[(0, 6, 3), (128, 6, 3)], 260, 9));
}

#[test]
fn rounds_of_many_tasks() {
    check(&Case::random(9, &[(0, 16, 7), (65536, 15, 7)], 131075, 18));
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
