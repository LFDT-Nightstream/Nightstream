//! Crate-private tests of layer 1.

mod gkr;
mod layer1;
mod matrix;
mod mle;
mod setup;
mod whir;

use std::path::PathBuf;
use std::sync::Arc;

use neo_ajtai::nightstream_fprime_setup::{coefficient_block, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS};
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{cf_inv, Rq, D, F, K};
use neo_reductions::superneo_eval::{SuperneoEvalCache, SuperneoEvalCacheBuilder, SuperneoZBlocks};
use p3_field::PrimeCharacteristicRing;

use crate::Claim;

/// Deterministic pseudo-random words; the tests need variety, not secrecy.
pub(super) fn word(seed: u64, index: u64) -> u64 {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ index.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 31;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 29)
}

/// A scratch directory removed on drop.
pub(super) struct Scratch(pub(super) PathBuf);

impl Scratch {
    pub(super) fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("neo-spartan-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(super) fn small(seed: u64, index: u64, bound: u32) -> F {
    let span = 2 * u64::from(bound) + 1;
    let value = (word(seed, index) % span) as i64 - i64::from(bound);
    if value < 0 {
        -F::from_u64(value.unsigned_abs())
    } else {
        F::from_u64(value as u64)
    }
}

/// A CE(B) claim built only with repository code: random sparse matrices,
/// a witness bounded by `bound` (tail lanes included), the production key
/// prefix, and the reference openings.
pub(super) struct Toy {
    pub(super) cache: Arc<SuperneoEvalCache>,
    pub(super) witness: Mat<F>,
    pub(super) claim: Claim,
    pub(super) public_blocks: usize,
    pub(super) point_variables: usize,
}

impl Toy {
    pub(super) fn new(seed: u64, blocks: usize, rows: usize, matrices: usize, bound: u32) -> Self {
        let columns = blocks * D;
        let mut builder = SuperneoEvalCacheBuilder::new(rows, columns, matrices).unwrap();
        for row in 0..rows {
            for matrix in 0..matrices {
                // A few strictly increasing columns per row.
                let mut entries: Vec<(usize, F)> = (0..3)
                    .map(|k| {
                        let column = (word(seed + 1, (row * 31 + matrix * 7 + k) as u64) % columns as u64) as usize;
                        (column, F::from_u64(1 + word(seed + 2, (row + k) as u64) % 1000))
                    })
                    .collect();
                entries.sort_by_key(|&(column, _)| column);
                entries.dedup_by_key(|entry| entry.0);
                builder.push_row(matrix, row, entries).unwrap();
            }
        }
        let cache = Arc::new(builder.finish().unwrap());
        let witness = Mat::from_row_major(
            D,
            blocks,
            (0..D * blocks)
                .map(|i| small(seed + 3, i as u64, bound))
                .collect(),
        );
        let public_blocks = 1;
        let point_variables = (columns.max(rows)).next_power_of_two().trailing_zeros() as usize + 1;
        let point: Vec<K> = (0..point_variables)
            .map(|i| {
                K::new_complex(
                    F::from_u64(word(seed + 4, i as u64)),
                    F::from_u64(word(seed + 5, i as u64)),
                )
            })
            .collect();
        let blocks_view = SuperneoZBlocks::from_witness_mat(&witness, columns).unwrap();
        let opening = cache
            .eval_real_v1_1_openings(&point, &[blocks_view])
            .unwrap()
            .remove(0);
        let pad = |mut values: Vec<K>| {
            values.resize(D.next_power_of_two(), K::ZERO);
            values
        };
        let claim = Claim {
            c: commit(&witness),
            X: Mat::from_row_major(D, public_blocks, (0..D).map(|lane| witness[(lane, 0)]).collect()),
            r: point,
            eval_k: pad(opening.eval_k),
            eval_a: opening.eval_a.into_iter().map(pad).collect(),
            m_in: D * public_blocks,
            fold_digest: [0; 32],
            adv: None,
        };
        Self {
            cache,
            witness,
            claim,
            public_blocks,
            point_variables,
        }
    }
}

/// `c_i = Σ_b a_{i,b} · z_b mod Φ81` with the production key prefix.
pub(super) fn commit(witness: &Mat<F>) -> Commitment {
    let kappa = PRODUCTION_VERIFIER_ROWS as usize;
    let mut commitment = Commitment::zeros(D, kappa);
    for row in 0..kappa {
        let mut total = Rq([F::ZERO; D]);
        for block in 0..witness.cols() {
            let key = coefficient_block(&PRODUCTION_SEED, row as u32, block as u64).map(F::from_u64);
            let z = cf_inv(std::array::from_fn(|lane| witness[(lane, block)]));
            total = total + Rq(key).mul(&z);
        }
        commitment.col_mut(row).copy_from_slice(&total.0);
    }
    commitment
}
