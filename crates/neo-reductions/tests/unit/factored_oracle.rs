use super::*;
use crate::superneo_eval::{FactoredEntries, FactoredRows, FactoredSlots, MatrixRowSink, MatrixShape, MatrixWindow};
use neo_ccs::poly::{SparsePoly, Term};
use neo_math::KExtensions;
use rand_chacha::{
    rand_core::{RngCore, SeedableRng},
    ChaCha8Rng,
};
use std::{
    ops::Range,
    sync::atomic::{AtomicUsize, Ordering},
};

const ROWS: usize = 37;
const COLUMNS: usize = 3 * D;
const MATRICES: usize = 4;
const VARIABLES: usize = 8;

struct Relation {
    offsets: Vec<usize>,
    slots: Vec<u32>,
    coefficients: Vec<F>,
    starts: Vec<u32>,
    widths: Vec<u8>,
    visits: AtomicUsize,
}

impl Relation {
    fn new(rng: &mut ChaCha8Rng) -> Self {
        let inside_block = (10, 41);
        let across_block_1 = (51, 41);
        let across_block_2 = (92, 41);
        let unread = (0, 0);
        let mut spans: Vec<(u32, u8)> = (0..10).map(|column| (column, 1)).collect();
        spans.extend([inside_block, across_block_1, across_block_2, unread]);
        spans.extend((133..COLUMNS as u32).map(|column| (column, 1)));
        let (starts, widths): (Vec<_>, Vec<_>) = spans.into_iter().unzip();
        let readable: Vec<u32> = (0..widths.len() as u32)
            .filter(|&slot| widths[slot as usize] != 0)
            .collect();
        let mut offsets = vec![0];
        let mut slots = Vec::new();
        let mut coefficients = Vec::new();
        for _ in 0..ROWS * MATRICES {
            let mut row: Vec<u32> = Vec::new();
            while row.len() < (rng.next_u32() % 6) as usize {
                let slot = readable[rng.next_u32() as usize % readable.len()];
                if !row.contains(&slot) {
                    row.push(slot);
                }
            }
            for slot in row {
                slots.push(slot);
                coefficients.push(match rng.next_u32() % 3 {
                    0 => F::ONE,
                    1 => -F::ONE,
                    _ => F::from_u64(rng.next_u64()),
                });
            }
            offsets.push(slots.len());
        }
        Self {
            offsets,
            slots,
            coefficients,
            starts,
            widths,
            visits: AtomicUsize::new(0),
        }
    }

    fn shape() -> MatrixShape {
        MatrixShape {
            rows: ROWS,
            columns: COLUMNS,
            matrices: MATRICES,
        }
    }

    fn entries(&self) -> FactoredEntries<'_> {
        FactoredEntries {
            ports: neo_ccs::sparse::FactoredPorts::Offsets(&self.offsets),
            slots: &self.slots,
            coefficients: &self.coefficients,
        }
    }

    fn spans(&self) -> FactoredSlots<'_> {
        FactoredSlots {
            starts: &self.starts,
            widths: &self.widths,
            ratio: F::from_u64(3),
        }
    }

    fn rows(&self) -> FactoredRows<'_> {
        FactoredRows::new(Self::shape(), self.entries(), self.spans()).unwrap()
    }
}

struct Factored<'a>(&'a Relation);

struct Runs<'a>(&'a Relation);

impl MatrixRows for Factored<'_> {
    fn shape(&self) -> MatrixShape {
        self.0.rows().shape()
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        self.0.visits.fetch_add(1, Ordering::Relaxed);
        self.0.rows().visit_rows(rows, sink)
    }

    fn factored(&self) -> Result<Option<FactoredRows<'_>>, PiCcsError> {
        Ok(Some(self.0.rows()))
    }
}

impl MatrixRows for Runs<'_> {
    fn shape(&self) -> MatrixShape {
        self.0.rows().shape()
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        self.0.visits.fetch_add(1, Ordering::Relaxed);
        self.0.rows().visit_rows(rows, sink)
    }
}

fn coin(rng: &mut ChaCha8Rng) -> K {
    K::from_coeffs([F::from_u64(rng.next_u64()), F::from_u64(rng.next_u64())])
}

fn signed_unit(rng: &mut ChaCha8Rng) -> Mat<F> {
    let blocks = COLUMNS / D;
    let positive: Vec<u64> = (0..blocks)
        .map(|_| rng.next_u64() & (u64::MAX >> (u64::BITS as usize - D)))
        .collect();
    let negative: Vec<u64> = positive
        .iter()
        .map(|&mask| rng.next_u64() & !mask & (u64::MAX >> (u64::BITS as usize - D)))
        .collect();
    Mat::compact_signed_unit_from_column_masks(D, blocks, &positive, &negative).unwrap()
}

fn dense(rng: &mut ChaCha8Rng) -> Mat<F> {
    let blocks = COLUMNS / D;
    Mat::from_row_major(
        D,
        blocks,
        (0..D * blocks)
            .map(|_| F::from_u64(rng.next_u64() % 7) - F::from_u64(3))
            .collect(),
    )
}

struct Run {
    messages: Vec<Vec<K>>,
    openings: Vec<V1_2OutputOpening>,
    visits_before_openings: usize,
}

fn prove(source: &dyn MatrixRows, relation: &Relation, seed: u64) -> Run {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let gate = SparsePoly::new(
        MATRICES,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![1, 1, 0, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 1, 0],
            },
            Term {
                coeff: F::ONE,
                exps: vec![0, 0, 0, 7],
            },
        ],
    );
    let structure = CcsStructure::new_verifier_artifact_header(ROWS, COLUMNS, MATRICES, gate).unwrap();
    let params = neo_params::NeoParams::nightstream_p64_k14();
    let fresh = vec![CcsWitness {
        Z: signed_unit(&mut rng),
    }];
    let running = vec![dense(&mut rng), signed_unit(&mut rng)];
    let dims = JointDims {
        assignment_width: COLUMNS,
        row_count: 1 << VARIABLES,
        variables: VARIABLES,
        matrix_count: MATRICES,
        degree: 9,
    };
    let challenges = Challenges::new((0..VARIABLES).map(|_| coin(&mut rng)).collect(), coin(&mut rng));
    let prior: Vec<_> = (0..VARIABLES).map(|_| coin(&mut rng)).collect();
    let payload = 2 * MATRICES * size_of::<K>() + size_of::<F>() + size_of::<K>();
    let workspace_bytes = 2 * MatrixWindow::required_workspace(&Runs(relation), 0..ROWS, payload).unwrap();
    relation.visits.store(0, Ordering::Relaxed);
    let mut oracle = OptimizedPaperJointOracle::new(
        &structure,
        &params,
        &fresh,
        &running,
        challenges,
        Some(&prior),
        dims,
        source,
        workspace_bytes,
        3,
    )
    .unwrap();
    let points: Vec<_> = (0..=dims.degree)
        .map(|point| K::from(F::from_usize(point)))
        .collect();
    let mut point = Vec::new();
    let mut messages = Vec::new();
    for _ in 0..VARIABLES {
        messages.push(oracle.evals_at(&points).unwrap());
        point.push(coin(&mut rng));
        oracle.fold(*point.last().unwrap()).unwrap();
    }
    let visits_before_openings = relation.visits.load(Ordering::Relaxed);
    let openings = oracle.output_openings(&point).unwrap().unwrap();
    Run {
        messages,
        openings,
        visits_before_openings,
    }
}

#[test]
fn factored_rows_give_the_same_round_messages_and_openings_as_runs() {
    let relation = Relation::new(&mut ChaCha8Rng::seed_from_u64(0x6661_6374));
    for seed in 0..4 {
        let runs = prove(&Runs(&relation), &relation, seed);
        let factored = prove(&Factored(&relation), &relation, seed);
        assert!(
            runs.messages
                .iter()
                .flatten()
                .any(|value| *value != K::ZERO),
            "seed {seed} has nonzero round messages"
        );
        assert_eq!(factored.messages, runs.messages, "round messages of seed {seed}");
        assert_eq!(factored.openings, runs.openings, "openings of seed {seed}");
        assert!(runs.visits_before_openings > 0, "runs build windows for the row values");
        assert_eq!(
            factored.visits_before_openings, 0,
            "factored rows build no window for the row values"
        );
    }
}

#[test]
fn factored_rows_reject_an_entry_without_a_slot_or_a_slot_past_the_columns() {
    let mut relation = Relation::new(&mut ChaCha8Rng::seed_from_u64(7));
    let unread = relation
        .widths
        .iter()
        .position(|&width| width == 0)
        .unwrap() as u32;
    let saved = relation.slots[0];
    relation.slots[0] = unread;
    assert!(FactoredRows::new(Relation::shape(), relation.entries(), relation.spans()).is_err());
    relation.slots[0] = saved;
    relation.starts[10] = (COLUMNS - 40) as u32;
    assert!(FactoredRows::new(Relation::shape(), relation.entries(), relation.spans()).is_err());
}

struct SlotRows {
    shape: MatrixShape,
    offsets: Vec<usize>,
    slots: Vec<u32>,
    coefficients: Vec<F>,
    starts: Vec<u32>,
    widths: Vec<u8>,
}

impl SlotRows {
    fn new(
        rng: &mut ChaCha8Rng,
        rows: usize,
        slots: usize,
        matrices: usize,
        read: impl Fn(&mut ChaCha8Rng) -> u32,
    ) -> Self {
        let mut offsets = vec![0];
        let (mut read_slots, mut coefficients) = (Vec::new(), Vec::new());
        for _ in 0..rows * matrices {
            for _ in 0..1 + rng.next_u32() % 8 {
                read_slots.push(read(rng));
                coefficients.push(match rng.next_u32() % 4 {
                    0 => F::ZERO,
                    1 => -F::ONE,
                    _ => F::from_u64(rng.next_u64()),
                });
            }
            offsets.push(read_slots.len());
        }
        Self {
            shape: MatrixShape {
                rows,
                columns: slots.next_multiple_of(D),
                matrices,
            },
            offsets,
            slots: read_slots,
            coefficients,
            starts: (0..slots as u32).collect(),
            widths: vec![1; slots],
        }
    }

    fn rows(&self) -> FactoredRows<'_> {
        FactoredRows::new(
            self.shape,
            FactoredEntries {
                ports: neo_ccs::sparse::FactoredPorts::Offsets(&self.offsets),
                slots: &self.slots,
                coefficients: &self.coefficients,
            },
            FactoredSlots {
                starts: &self.starts,
                widths: &self.widths,
                ratio: F::from_u64(3),
            },
        )
        .unwrap()
    }

    fn defined_weights(&self, row_weight: impl Fn(usize) -> K, matrix_coefficients: &[K]) -> Vec<K> {
        let mut weights = vec![K::ZERO; self.starts.len()];
        for row in 0..self.shape.rows {
            for (matrix, &coefficient) in matrix_coefficients.iter().enumerate() {
                let index = row * self.shape.matrices + matrix;
                for entry in self.offsets[index]..self.offsets[index + 1] {
                    weights[self.slots[entry] as usize] +=
                        row_weight(row) * coefficient * K::from(self.coefficients[entry]);
                }
            }
        }
        weights
    }
}

fn assert_slot_weights(rows: &SlotRows, rng: &mut ChaCha8Rng) {
    let row_weights: Vec<K> = (0..rows.shape.rows)
        .map(|row| if row % 3 == 1 { K::ZERO } else { coin(rng) })
        .collect();
    let mut matrix_coefficients: Vec<K> = (0..rows.shape.matrices).map(|_| coin(rng)).collect();
    if rows.shape.matrices > 1 {
        matrix_coefficients[1] = K::ZERO;
    }
    let expected = rows.defined_weights(|row| row_weights[row], &matrix_coefficients);
    assert!(expected.iter().any(|&weight| weight != K::ZERO));
    assert!(
        rows.rows()
            .slot_weights(&|row| row_weights[row], &matrix_coefficients)
            == expected
    );
    assert!(rows.rows().slot_weights(&|_| K::ZERO, &matrix_coefficients) == vec![K::ZERO; rows.starts.len()]);
}

#[test]
fn slot_weights_are_their_definition() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x736c_6f74);
    let spread = |slots: usize| move |rng: &mut ChaCha8Rng| rng.next_u32() % slots as u32;
    for matrices in 1..=MATRICES {
        for (rows, slots) in [(1, 5), (1, 3 * (1 << 14) + 5), (3 * (1 << 12) + 37, 3 * (1 << 14) + 5)] {
            let relation = SlotRows::new(&mut rng, rows, slots, matrices, spread(slots));
            assert_slot_weights(&relation, &mut rng);
        }
    }
    let one_band = SlotRows::new(
        &mut rng,
        2 * (1 << 12) + 1,
        1 << 15,
        MATRICES,
        |rng: &mut ChaCha8Rng| rng.next_u32() % 64,
    );
    assert_slot_weights(&one_band, &mut rng);
}

#[test]
fn shared_coefficients_must_cover_each_port() {
    let slots = FactoredSlots {
        starts: &[0],
        widths: &[1],
        ratio: F::ONE,
    };
    let shape = MatrixShape {
        rows: 3,
        columns: D,
        matrices: 1,
    };
    use neo_ccs::sparse::{FactoredPorts, SharedPort};
    let port = |slot_start, coefficient_start| SharedPort {
        slot_start,
        coefficient_start,
        len: 1,
    };
    let valid = [port(0, 1), port(0, 0), port(0, 1)];
    let entries = FactoredEntries {
        ports: FactoredPorts::Shared {
            ports: &valid,
            entries: 3,
        },
        slots: &[0],
        coefficients: &[F::ONE, F::NEG_ONE],
    };
    let rows = FactoredRows::new(shape, entries, slots).unwrap();
    assert_eq!(
        [0, 1, 2].map(|row| rows.row_value(0, row, &[F::ONE])),
        [F::NEG_ONE, F::ONE, F::NEG_ONE]
    );
    for ports in [
        &valid[..2],
        &[port(0, 1), port(1, 0), port(0, 1)],
        &[port(0, 1), port(0, 2), port(0, 1)],
    ] {
        assert!(FactoredRows::new(
            shape,
            FactoredEntries {
                ports: FactoredPorts::Shared { ports, entries: 3 },
                ..entries
            },
            slots
        )
        .is_err());
    }
}
