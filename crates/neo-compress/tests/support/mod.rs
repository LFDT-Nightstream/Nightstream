//! Small CE claims with their witnesses, computed from the definitions of
//! SPEC.md section 6 with schoolbook ring arithmetic.

#![allow(dead_code)]

use std::ops::Range;

use neo_ajtai::nightstream_fprime_setup::{coefficient_block, PRODUCTION_SEED};
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_compress::{CeClaim, Norm, Statement};
use neo_math::{from_complex, D, F, K};
use neo_reductions::superneo_eval::{
    FactoredEntries, FactoredRows, FactoredSlots, MatrixRowSink, MatrixRows, MatrixShape,
};
use neo_reductions::PiCcsError;
use p3_field::PrimeCharacteristicRing;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub const KAPPA: usize = 18;
pub const MATRICES: usize = 4;
pub const POINT_VARS: usize = 14;
pub const PUBLIC_BLOCKS: usize = 2;

pub fn random_extension(rng: &mut ChaCha8Rng) -> K {
    from_complex(F::new(rng.next_u64()), F::new(rng.next_u64()))
}

/// The product in `K[X]/(X^64 + 1)`.
pub fn ring_product(left: &[K], right: &[K]) -> [K; D] {
    let mut product = [K::ZERO; D];
    for (i, &left) in left.iter().enumerate() {
        for (j, &right) in right.iter().enumerate() {
            if i + j < D {
                product[i + j] += left * right;
            } else {
                product[i + j - D] -= left * right;
            }
        }
    }
    product
}

pub fn conjugate(element: &[K]) -> [K; D] {
    std::array::from_fn(|lane| if lane == 0 { element[0] } else { -element[D - lane] })
}

pub fn equality(point: &[K], index: usize) -> K {
    point
        .iter()
        .enumerate()
        .map(|(bit, &coordinate)| {
            if index >> bit & 1 == 1 {
                coordinate
            } else {
                K::ONE - coordinate
            }
        })
        .product()
}

/// Random matrix rows in factored form.
pub struct Rows {
    shape: MatrixShape,
    offsets: Vec<usize>,
    slots: Vec<u32>,
    coefficients: Vec<F>,
    starts: Vec<u32>,
    widths: Vec<u8>,
}

const RATIO: u64 = 3;

impl Rows {
    pub fn random(rng: &mut ChaCha8Rng, rows: usize, blocks: usize) -> Self {
        let columns = blocks * D;
        let (mut starts, mut widths) = (Vec::new(), Vec::new());
        let mut column = 0;
        while column < columns {
            let width = (1 + rng.next_u32() as usize % 9).min(columns - column);
            starts.push(column as u32);
            widths.push(width as u8);
            column += width + rng.next_u32() as usize % 3;
        }
        let (mut offsets, mut slots, mut coefficients) = (vec![0], Vec::new(), Vec::new());
        for _ in 0..rows * MATRICES {
            for _ in 0..rng.next_u32() % 4 {
                slots.push(rng.next_u32() % starts.len() as u32);
                coefficients.push(F::new(rng.next_u64()));
            }
            offsets.push(slots.len());
        }
        Self {
            shape: MatrixShape {
                rows,
                columns,
                matrices: MATRICES,
            },
            offsets,
            slots,
            coefficients,
            starts,
            widths,
        }
    }

    fn rows(&self) -> FactoredRows<'_> {
        FactoredRows::new(
            self.shape,
            FactoredEntries {
                offsets: &self.offsets,
                slots: &self.slots,
                coefficients: &self.coefficients,
            },
            FactoredSlots {
                starts: &self.starts,
                widths: &self.widths,
                ratio: F::from_u64(RATIO),
            },
        )
        .expect("factored rows")
    }

    /// `M_matrixᵀ · weights`, from the entries.
    fn transpose(&self, matrix: usize, weights: &[K]) -> Vec<K> {
        let mut columns = vec![K::ZERO; self.shape.columns];
        for (row, &weight) in weights.iter().enumerate() {
            let entries = self.offsets[MATRICES * row + matrix]..self.offsets[MATRICES * row + matrix + 1];
            for entry in entries {
                let slot = self.slots[entry] as usize;
                let mut term = weight * self.coefficients[entry];
                for column in 0..usize::from(self.widths[slot]) {
                    columns[self.starts[slot] as usize + column] += term;
                    term *= F::from_u64(RATIO);
                }
            }
        }
        columns
    }
}

impl MatrixRows for Rows {
    fn shape(&self) -> MatrixShape {
        self.shape
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        self.rows().visit_rows(rows, sink)
    }

    fn factored(&self) -> Result<Option<FactoredRows<'_>>, PiCcsError> {
        Ok(Some(self.rows()))
    }
}

/// A claim with its witness and rows.
pub struct Instance {
    pub rows: Rows,
    pub witness: Mat<F>,
    pub claim: CeClaim,
    pub norm: Norm,
}

/// A random witness with coefficients in `[-range, range]`.
pub fn random_witness(rng: &mut ChaCha8Rng, blocks: usize, range: u64) -> Mat<F> {
    let cells = (0..D * blocks)
        .map(|_| F::from_u64(rng.next_u64() % (2 * range + 1)) - F::from_u64(range))
        .collect();
    Mat::from_row_major(D, blocks, cells)
}

/// The claim that `witness` satisfies at `point` under `rows`.
pub fn claim_of(rows: &Rows, witness: &Mat<F>, point: &[K]) -> CeClaim {
    let blocks = witness.cols();
    let block = |block: usize| -> Vec<K> { (0..D).map(|lane| K::from(witness[(lane, block)])).collect() };
    let mut commitment = Vec::with_capacity(KAPPA * D);
    for row in 0..KAPPA {
        let mut sum = [K::ZERO; D];
        for index in 0..blocks {
            let key = coefficient_block(&PRODUCTION_SEED, row as u32, index as u64).map(|value| K::from(F::new(value)));
            for (sum, term) in sum.iter_mut().zip(ring_product(&key, &block(index))) {
                *sum += term;
            }
        }
        commitment.extend(sum.map(|value| {
            let [real, imaginary] = neo_math::KExtensions::as_coeffs(&value);
            assert_eq!(imaginary, F::ZERO);
            real
        }));
    }
    let (low, high) = point.split_at(D.trailing_zeros() as usize);
    let lanes: Vec<K> = (0..D).map(|lane| equality(low, lane)).collect();
    let mut folded = [K::ZERO; D];
    for index in 0..blocks {
        let weight = equality(high, index);
        for (sum, value) in folded.iter_mut().zip(block(index)) {
            *sum += weight * value;
        }
    }
    let row_weights: Vec<K> = (0..rows.shape.rows)
        .map(|row| equality(point, row))
        .collect();
    let eval_a = (0..MATRICES)
        .map(|matrix| {
            let columns = rows.transpose(matrix, &row_weights);
            let mut sum = [K::ZERO; D];
            for index in 0..blocks {
                let weights = conjugate(&columns[index * D..(index + 1) * D]);
                for (sum, term) in sum.iter_mut().zip(ring_product(&weights, &block(index))) {
                    *sum += term;
                }
            }
            sum.to_vec()
        })
        .collect();
    let public = (0..D)
        .flat_map(|lane| (0..PUBLIC_BLOCKS).map(move |block| (lane, block)))
        .map(|cell| witness[cell])
        .collect();
    CeClaim {
        c: Commitment::new(KAPPA, commitment).expect("commitment"),
        X: Mat::from_row_major(D, PUBLIC_BLOCKS, public),
        r: point.to_vec(),
        eval_k: ring_product(&conjugate(&lanes), &folded).to_vec(),
        eval_a,
        m_in: PUBLIC_BLOCKS * D,
        fold_digest: [0; 32],
        adv: None,
    }
}

impl Instance {
    pub fn random(rng: &mut ChaCha8Rng, blocks: usize, norm: Norm) -> Self {
        let rows = Rows::random(rng, 150, blocks);
        let range = match norm {
            Norm::Table { bound } => bound,
            Norm::Caller => 1,
        };
        let witness = random_witness(rng, blocks, range);
        let point: Vec<K> = (0..POINT_VARS).map(|_| random_extension(rng)).collect();
        let claim = claim_of(&rows, &witness, &point);
        Self {
            rows,
            witness,
            claim,
            norm,
        }
    }

    pub fn statement(&self) -> Statement<'_> {
        Statement {
            claim: &self.claim,
            rows: &self.rows,
            norm: self.norm,
        }
    }
}

/// Two claims with a table and one without, of three sizes.
pub fn instances() -> Vec<Instance> {
    let mut rng = ChaCha8Rng::seed_from_u64(0x6e65_6f63);
    vec![
        Instance::random(&mut rng, 61, Norm::Table { bound: 2176 }),
        Instance::random(&mut rng, 100, Norm::Table { bound: 300 }),
        Instance::random(&mut rng, 90, Norm::Caller),
    ]
}
