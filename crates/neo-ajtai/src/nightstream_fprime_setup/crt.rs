use std::sync::{Mutex, OnceLock};

use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

use super::{SignedBlock, D, PRODUCTION_VERIFIER_ROWS, TASK_COLUMNS};

const P0: u64 = 268_425_089;
const P1: u64 = 268_428_161;
const P2: u64 = 268_432_897;
const Q: u128 = (1 << 64) - 59;
const M01: u128 = P0 as u128 * P1 as u128;
const M: u128 = M01 * P2 as u128;
const ROWS: usize = PRODUCTION_VERIFIER_ROWS as usize;
const TILE_COLUMNS: usize = TASK_COLUMNS as usize;
const CACHE_COLUMNS: usize = 524_288;
const GROUP: usize = 256;
const _: () = assert!(2 * D as u128 * TILE_COLUMNS as u128 * (u64::MAX as u128) < M);
const _: () = assert!(GROUP as u128 * (P2 - 1) as u128 * (P2 - 1) as u128 <= u64::MAX as u128);
const _: () = assert!(P0 < P2 && P1 < P2 && P2 < (1 << 28));
const _: () = assert!(CACHE_COLUMNS.is_multiple_of(TILE_COLUMNS));
const _: () = assert!(D == 64);

type Residues = [u32; D];
struct PackedResidues([[u32; 8]; 7]);

impl PackedResidues {
    fn new(values: Residues) -> Self {
        Self(std::array::from_fn(|group| {
            std::array::from_fn(|lane| {
                let high = values[56 + lane] >> (4 * group);
                values[8 * group + lane] | (high << 28)
            })
        }))
    }

    #[inline]
    fn unpack(&self) -> Residues {
        let mut values = [0; D];
        for (group, words) in self.0.iter().enumerate() {
            for (lane, &word) in words.iter().enumerate() {
                values[8 * group + lane] = word & 0x0fff_ffff;
                values[56 + lane] |= (word >> 28) << (4 * group);
            }
        }
        values
    }
}

type Output = Vec<[Fq; D]>;
type MaskSpectrum = (usize, Residues);

static PLANS: OnceLock<Plans> = OnceLock::new();

const fn pow_mod(mut value: u64, mut power: u64, m: u64) -> u64 {
    let mut result = 1;
    while power != 0 {
        if power & 1 != 0 {
            result = result * value % m;
        }
        value = value * value % m;
        power >>= 1;
    }
    result
}

const INVERSE_P0: u64 = pow_mod(P0 % P1, P1 - 2, P1);
const INVERSE_M01: u64 = pow_mod((P0 * P1) % P2, P2 - 2, P2);

#[inline]
fn mul<const P: u64>(a: u32, b: u32) -> u32 {
    (u64::from(a) * u64::from(b) % P) as u32
}

#[inline]
fn add<const P: u64>(a: u32, b: u32) -> u32 {
    let sum = a + b;
    if sum >= P as u32 {
        sum - P as u32
    } else {
        sum
    }
}

#[inline]
fn sub<const P: u64>(a: u32, b: u32) -> u32 {
    if a >= b {
        a - b
    } else {
        a + (P as u32 - b)
    }
}

struct Plan<const P: u64> {
    twist: Residues,
    inverse_twist: Residues,
    forward: [[u32; D / 2]; 6],
    inverse: [[u32; D / 2]; 6],
    bytes: Vec<Residues>,
}

impl<const P: u64> Plan<P> {
    fn new() -> Self {
        let psi = (2..256)
            .map(|generator| pow_mod(generator, (P - 1) / (2 * D as u64), P))
            .find(|&root| pow_mod(root, D as u64, P) == P - 1)
            .expect("a 128th root for the CRT prime");
        let omega = psi * psi % P;
        let table = |root| {
            std::array::from_fn(|stage| {
                std::array::from_fn(|i| pow_mod(root, (i * (D >> (stage + 1))) as u64, P) as u32)
            })
        };
        let mut plan = Self {
            twist: std::array::from_fn(|i| pow_mod(psi, i as u64, P) as u32),
            inverse_twist: std::array::from_fn(|i| {
                (pow_mod(psi, (2 * D - i) as u64, P) * pow_mod(D as u64, P - 2, P) % P) as u32
            }),
            forward: table(omega),
            inverse: table(pow_mod(omega, P - 2, P)),
            bytes: Vec::new(),
        };
        let mut bytes = vec![[0; D]; (D / 8) * 256];
        for (byte, patterns) in bytes.chunks_exact_mut(256).enumerate() {
            for bit in 0..8 {
                let lane = 8 * byte + bit;
                let mut transform = [0; D];
                transform[lane] = plan.twist[lane];
                Self::transform(&mut transform, &plan.forward);
                for low in 0..1 << bit {
                    let prior = patterns[low];
                    patterns[low + (1 << bit)] = std::array::from_fn(|lane| add::<P>(prior[lane], transform[lane]));
                }
            }
        }
        plan.bytes = bytes;
        plan
    }

    fn transform(values: &mut Residues, table: &[[u32; D / 2]; 6]) {
        for i in 0..D {
            let j = i.reverse_bits() >> (usize::BITS - 6);
            if i < j {
                values.swap(i, j);
            }
        }
        for (stage, twiddles) in table.iter().enumerate() {
            let half = 1 << stage;
            for window in values.chunks_exact_mut(2 * half) {
                let (left, right) = window.split_at_mut(half);
                for ((left, right), &twiddle) in left.iter_mut().zip(right).zip(twiddles) {
                    let b = mul::<P>(*right, twiddle);
                    let a = *left;
                    *left = add::<P>(a, b);
                    *right = sub::<P>(a, b);
                }
            }
        }
    }

    fn key(&self, words: &[u64]) -> Residues {
        let mut values = std::array::from_fn(|i| mul::<P>((words[i] % P) as u32, self.twist[i]));
        Self::transform(&mut values, &self.forward);
        values
    }

    fn mask(&self, positive: u64, negative: u64) -> Residues {
        let mut values = [0; D];
        for (byte, (positive, negative)) in positive
            .to_le_bytes()
            .into_iter()
            .zip(negative.to_le_bytes())
            .enumerate()
        {
            let patterns = &self.bytes[byte * 256..][..256];
            if positive != 0 {
                for (value, &term) in values.iter_mut().zip(&patterns[positive as usize]) {
                    *value = add::<P>(*value, term);
                }
            }
            if negative != 0 {
                for (value, &term) in values.iter_mut().zip(&patterns[negative as usize]) {
                    *value = sub::<P>(*value, term);
                }
            }
        }
        values
    }

    fn masks(&self, start: u64, blocks: &[SignedBlock]) -> Vec<MaskSpectrum> {
        blocks
            .iter()
            .map(|block| {
                (
                    (block.index() - start) as usize,
                    self.mask(block.positive(), block.negative()),
                )
            })
            .collect()
    }

    fn row(&self, key: &[PackedResidues], rhs: &[MaskSpectrum]) -> Residues {
        let mut residues = [0; D];
        for part in rhs.chunks(GROUP) {
            let mut sums = [0u64; D];
            for (column, witness) in part {
                for ((sum, left), &right) in sums.iter_mut().zip(key[*column].unpack()).zip(witness) {
                    *sum += u64::from(left) * u64::from(right);
                }
            }
            for (value, sum) in residues.iter_mut().zip(sums) {
                *value = add::<P>(*value, (sum % P) as u32);
            }
        }
        Self::transform(&mut residues, &self.inverse);
        for (residue, &twist) in residues.iter_mut().zip(&self.inverse_twist) {
            *residue = mul::<P>(*residue, twist);
        }
        residues
    }
}

struct Plans(Plan<P0>, Plan<P1>, Plan<P2>);

fn plans() -> &'static Plans {
    PLANS.get_or_init(|| Plans(Plan::new(), Plan::new(), Plan::new()))
}

fn combine(r0: u32, r1: u32, r2: u32) -> Fq {
    let t1 = (u64::from(r1) + P1 - u64::from(r0) % P1) * INVERSE_P0 % P1;
    let x01 = u128::from(r0) + u128::from(P0) * u128::from(t1);
    let t2 = (u64::from(r2) + P2 - (x01 % u128::from(P2)) as u64) * INVERSE_M01 % P2;
    let value = x01 + M01 * u128::from(t2);
    let residue = if value <= M / 2 {
        value % Q
    } else {
        (Q - (M - value) % Q) % Q
    };
    Fq::new(residue as u64)
}

pub(super) struct Tile {
    values: [Vec<PackedResidues>; 3],
}

impl Tile {
    fn new(words: &[u64], block_count: usize, start: usize) -> Self {
        let plans = plans();
        let mut values = std::array::from_fn(|_| Vec::with_capacity(ROWS * TILE_COLUMNS));
        let [a, b, c] = &mut values;
        for row in words.chunks_exact(block_count * D) {
            for words in row[start * D..][..TILE_COLUMNS * D].chunks_exact(D) {
                a.push(PackedResidues::new(plans.0.key(words)));
                b.push(PackedResidues::new(plans.1.key(words)));
                c.push(PackedResidues::new(plans.2.key(words)));
            }
        }
        Self { values }
    }

    pub(super) fn add_to(&self, start: u64, blocks: &[SignedBlock], output: &mut [[Fq; D]]) {
        let Plans(a, b, c) = plans();
        let (rhs_a, rhs_b, rhs_c) = (a.masks(start, blocks), b.masks(start, blocks), c.masks(start, blocks));
        let [key_a, key_b, key_c] = &self.values;
        for (row, output) in output.iter_mut().enumerate() {
            let range = row * TILE_COLUMNS..(row + 1) * TILE_COLUMNS;
            let values_a = a.row(&key_a[range.clone()], &rhs_a);
            let values_b = b.row(&key_b[range.clone()], &rhs_b);
            let values_c = c.row(&key_c[range], &rhs_c);
            for (((output, a), b), c) in output.iter_mut().zip(values_a).zip(values_b).zip(values_c) {
                *output += combine(a, b, c);
            }
        }
    }
}

#[derive(Default)]
struct CacheEntry {
    tile: OnceLock<Tile>,
    loading: Mutex<()>,
}

pub(super) struct Cache(Vec<CacheEntry>);

impl Cache {
    pub(super) fn new(block_count: u64) -> Self {
        let tiles = (block_count as usize).min(CACHE_COLUMNS) / TILE_COLUMNS;
        Self((0..tiles).map(|_| CacheEntry::default()).collect())
    }

    pub(super) fn tiles(&self) -> usize {
        self.0.len()
    }

    pub(super) fn tile(&self, index: usize, words: &[u64], block_count: u64) -> Option<&Tile> {
        let entry = self.0.get(index)?;
        if let Some(tile) = entry.tile.get() {
            return Some(tile);
        }
        let _loading = entry.loading.try_lock().ok()?;
        Some(
            entry
                .tile
                .get_or_init(|| Tile::new(words, block_count as usize, index * TILE_COLUMNS)),
        )
    }
}

pub(super) fn use_tile(blocks: &[SignedBlock]) -> bool {
    blocks
        .iter()
        .map(|block| (block.positive() | block.negative()).count_ones() as usize)
        .sum::<usize>()
        >= 8 * TILE_COLUMNS
}

pub(super) fn zero() -> Output {
    vec![[Fq::ZERO; D]; ROWS]
}

fn sum(mut left: Output, right: Output) -> Output {
    for (left, &right) in left.iter_mut().flatten().zip(right.iter().flatten()) {
        *left += right;
    }
    left
}

pub(super) fn sum_tiles(tiles: &[(&Tile, u64, &[SignedBlock])]) -> Output {
    let tiles_per_job = tiles.len().div_ceil(rayon::current_num_threads()).max(1);
    tiles
        .par_chunks(tiles_per_job)
        .map(|job| {
            let mut output = zero();
            for &(tile, start, blocks) in job {
                tile.add_to(start, blocks, &mut output);
            }
            output
        })
        .reduce_with(sum)
        .unwrap_or_else(zero)
}
