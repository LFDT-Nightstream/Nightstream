//! The Ajtai key as a committed polynomial (ePrint 2026/1983 section 9,
//! applied to the key of the claims; SPEC.md section 6).
//!
//! The key polynomial `A` has, lowest variable first, the lane, the block and
//! the row of a key coefficient: cell `h + 64·j + 64·n·i` holds coefficient
//! `h` of key element `(i, j)` for each row `i` below κ and each block `j`
//! below `n`, a power of two. Each cell of a row from κ is zero.
//! `ProverKey::new` commits `A` once with Akita, and the verifier holds the
//! commitment in place of the key (`VerificationKey`).
//!
//! The Ajtai rows of the claims add to the public weights at the point `ρ`
//! of the sum-check the value
//!
//! `Σ_claims c · Σ_{j < blocks} eq(β, j) · Σ_h u(h) · F(h, j)`, with
//! `F = Σ_i ω^i · A(·, ·, i)`,
//!
//! where `c`, `β` and `blocks` belong to a claim and `u` is the lane form
//! (`weights::Uses`). A claim reads a prefix of the blocks that is not a
//! power of two, so the sum is not a value of `F`. The prover sends the sum,
//! and a sum-check of degree 2 over the lanes and the blocks reduces it to
//! `F` at one point `(λ, β′)`. The verifier computes the weight
//! `u(λ) · Σ_claims c · eq_prefix(β, blocks, β′)` at that point in a number
//! of steps linear in the variables (`weight_at`). The powers of `ω` are a
//! scaled `eq` table (`row_point`), so `F(λ, β′) = scale · A(λ, β′, s)`: one
//! value of the committed polynomial, which Akita opens.

use std::fmt;
use std::sync::{Arc, OnceLock};

use neo_ajtai::nightstream_fprime_setup::production_key_pair;
use neo_math::{KExtensions, D, F, K};
use p3_field::{Field, PrimeCharacteristicRing};
use rayon::prelude::*;

use crate::akita::prover::{Committed, Prover};
use crate::akita::verifier::Verifier;
use crate::akita::{self, Base, Polynomial, Schedules, Shape, SCHEDULES};
use crate::layout::LANE_VARS;
use crate::ring::dot;
use crate::weights::{equality, powers, prefix_equality, Uses, KAPPA};
use crate::Error;

/// The row index of the key polynomial has this many variables.
pub(crate) const ROW_VARS: usize = 5;
const _: () = assert!(KAPPA <= 1 << ROW_VARS);

const CELLS_PER_TASK: usize = 1 << 12;
const MAGIC: &[u8; 16] = b"NS-COMPRESS-VK-1";

/// What the verifier holds in place of the Ajtai key: the Akita commitment
/// of the key polynomial, and Akita's commitments of the prefixes of its own
/// setup that the schedule rows of the key offload. Both are functions of
/// public seeds, and the verifier reads neither table.
///
/// The key of each size is pinned (`Schedules::key`), and a test computes
/// each pinned key again from the public seeds. `new` and `from_bytes` give
/// only a pinned key, so no key that they give is unchecked input.
#[derive(Clone)]
pub struct VerificationKey {
    schedules: &'static Schedules,
    commitment: Vec<u8>,
    setup_prefixes: Vec<u8>,
    /// The Akita verifier of each schedule row, made at the first proof of
    /// that row.
    akita: Arc<[OnceLock<Verifier>]>,
}

impl VerificationKey {
    /// The key for claims of at most `blocks` blocks: the pinned key of the
    /// smallest key polynomial that holds them. It expands no key element.
    pub fn new(blocks: usize) -> Result<Self, Error> {
        Ok(Self::pinned(schedules_for(blocks)?))
    }

    /// Decode untrusted bytes. They decode only when they are the pinned
    /// key of one size of the key polynomial.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        SCHEDULES
            .iter()
            .find(|schedules| schedules.key == bytes)
            .map(Self::pinned)
            .ok_or(Error::Key("the bytes are not a pinned key"))
    }

    fn pinned(schedules: &'static Schedules) -> Self {
        let mut input = &schedules.key[MAGIC.len() + 8..];
        let mut part = || {
            let (length, rest) = input.split_at(8);
            let (part, rest) = rest.split_at(u64::from_le_bytes(length.try_into().expect("a length")) as usize);
            input = rest;
            part.to_vec()
        };
        Self::of_parts(schedules, part(), part())
    }

    fn of_parts(schedules: &'static Schedules, commitment: Vec<u8>, setup_prefixes: Vec<u8>) -> Self {
        Self {
            schedules,
            commitment,
            setup_prefixes,
            akita: schedules.rows.iter().map(|_| OnceLock::new()).collect(),
        }
    }

    /// The key polynomial holds this many blocks of each key row, so the
    /// key proves claims of at most this many blocks.
    pub fn blocks(&self) -> usize {
        self.schedules.blocks()
    }

    pub(crate) fn commitment(&self) -> &[u8] {
        &self.commitment
    }

    /// The Akita verifier of the proofs with `shape`.
    pub(crate) fn akita(&self, shape: Shape) -> Result<&Verifier, Error> {
        let row = self
            .schedules
            .rows
            .iter()
            .position(|&row| shape.key == self.schedules.key_vars && row == (shape.witness, shape.inverse))
            .ok_or(Error::Layout("no Akita schedule row has these polynomials"))?;
        if let Some(verifier) = self.akita[row].get() {
            return Ok(verifier);
        }
        let verifier = Verifier::new(shape, &self.setup_prefixes)?;
        Ok(self.akita[row].get_or_init(|| verifier))
    }

    /// The key's strict bytes: a tag, the blocks, then each Akita byte
    /// string after its length.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend((self.blocks() as u64).to_le_bytes());
        for part in [&self.commitment, &self.setup_prefixes] {
            bytes.extend((part.len() as u64).to_le_bytes());
            bytes.extend_from_slice(part);
        }
        bytes
    }
}

impl PartialEq for VerificationKey {
    fn eq(&self, other: &Self) -> bool {
        (self.blocks(), &self.commitment, &self.setup_prefixes)
            == (other.blocks(), &other.commitment, &other.setup_prefixes)
    }
}

impl Eq for VerificationKey {}

impl fmt::Debug for VerificationKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerificationKey")
            .field("blocks", &self.blocks())
            .finish_non_exhaustive()
    }
}

/// What the prover holds: the key rows, their Akita commitment with the
/// polynomial that Akita opens, and the verification key.
pub struct ProverKey {
    pub(crate) akita: Prover,
    pub(crate) committed: Committed,
    /// Lane `h` of key element `(i, j)` at `(i·blocks + j)·64 + h`.
    table: Vec<F>,
    verification: VerificationKey,
}

impl ProverKey {
    /// The key for claims of at most `blocks` blocks. It expands the key
    /// polynomial from SHAKE128 once and commits it.
    pub fn new(blocks: usize) -> Result<Self, Error> {
        let schedules = schedules_for(blocks)?;
        Self::of_elements(schedules, production_element)
    }

    /// The key of the polynomial whose key elements are `element(row, pair)`
    /// for the blocks `2·pair` and `2·pair + 1`.
    pub(crate) fn of_elements(
        schedules: &'static Schedules,
        element: impl Fn(u32, u64) -> [[u64; D]; 2] + Sync,
    ) -> Result<Self, Error> {
        let table = table(schedules.blocks(), element, F::new);
        let cells = table.par_iter().copied().map(akita::base).collect();
        let (akita, committed, verification) = committed(schedules, cells)?;
        Ok(Self {
            akita,
            committed,
            table,
            verification,
        })
    }

    pub fn verification_key(&self) -> &VerificationKey {
        &self.verification
    }

    /// `F = Σ_i ω^i · A(·, ·, i)`: lane `h` of block `j` at `64j + h`.
    pub(crate) fn fold(&self, omega: K) -> Vec<K> {
        fold(&self.table, omega)
    }
}

/// The rows of `table` batched with the powers of `omega`.
fn fold(table: &[F], omega: K) -> Vec<K> {
    let (real, imaginary): (Vec<F>, Vec<F>) = powers(omega, KAPPA)
        .iter()
        .map(|power| {
            let [real, imaginary] = power.as_coeffs();
            (real, imaginary)
        })
        .unzip();
    let slab = table.len() / KAPPA;
    (0..slab)
        .into_par_iter()
        .with_min_len(CELLS_PER_TASK)
        .map(|cell| {
            let rows = table[cell..].iter().step_by(slab).copied();
            K::from_coeffs([
                F::sum_of_products(real.iter().copied().zip(rows.clone())),
                F::sum_of_products(imaginary.iter().copied().zip(rows)),
            ])
        })
        .collect()
}

/// The blocks of the key polynomial for claims of at most `blocks` blocks:
/// `ProverKey::new` and `VerificationKey::new` give the key of this size.
pub fn key_blocks(blocks: usize) -> Result<usize, Error> {
    Ok(schedules_for(blocks)?.blocks())
}

/// The schedule rows of the smallest key polynomial that holds `blocks`
/// blocks.
fn schedules_for(blocks: usize) -> Result<&'static Schedules, Error> {
    SCHEDULES
        .iter()
        .find(|schedules| schedules.blocks() >= blocks)
        .ok_or(Error::Layout("no Akita schedule row has a key for this many blocks"))
}

fn production_element(row: u32, pair: u64) -> [[u64; D]; 2] {
    production_key_pair(row, [2 * pair, 2 * pair + 1])
}

/// The cells of the rows below κ of the key polynomial of `blocks` blocks.
fn table<T: Send>(
    blocks: usize,
    element: impl Fn(u32, u64) -> [[u64; D]; 2] + Sync,
    cell: impl Fn(u64) -> T + Sync,
) -> Vec<T> {
    let pairs = blocks / 2;
    (0..KAPPA * pairs)
        .into_par_iter()
        .map(|index| element((index / pairs) as u32, (index % pairs) as u64).map(|block| block.map(&cell)))
        .collect::<Vec<_>>()
        .into_flattened()
        .into_flattened()
}

/// Commit the key polynomial with the cells `cells`.
fn committed(schedules: &'static Schedules, cells: Vec<Base>) -> Result<(Prover, Committed, VerificationKey), Error> {
    let akita = Prover::new(schedules)?;
    let committed = akita.commit(schedules.first(), Polynomial::Key, cells)?;
    let verification = VerificationKey::of_parts(schedules, committed.bytes.clone(), akita.setup_prefixes()?);
    Ok((akita, committed, verification))
}

/// `(s, scale)` with `ω^i = scale · eq(s, i)` for each row `i`:
/// `s_k = ω^(2^k) / (1 + ω^(2^k))` and `scale = Π_k (1 + ω^(2^k))`. No such
/// point exists when a factor of `scale` is zero: for 7 values of `ω`, since
/// 8 is the largest power of two that divides `|K*| = q² − 1`, so
/// `ω^(2^k) = −1` has `2^k` roots for `k ≤ 2` and none for `k > 2`.
pub(crate) fn row_point(omega: K) -> Result<([K; ROW_VARS], K), Error> {
    let mut point = [K::ZERO; ROW_VARS];
    let (mut scale, mut power) = (K::ONE, omega);
    for coordinate in &mut point {
        let sum = K::ONE + power;
        *coordinate = power * sum.try_inverse().ok_or(Error::Challenge)?;
        scale *= sum;
        power = power.square();
    }
    Ok((point, scale))
}

/// The prover's weights of the sum-check on `F`: cell `64j + h` has
/// `u(h) · Σ_claims c · eq(β, j)` for each block `j` that a claim reads.
pub(crate) fn weight_table(uses: &Uses) -> Vec<K> {
    let blocks = uses
        .claims
        .iter()
        .map(|claim| claim.blocks)
        .max()
        .unwrap_or(0);
    let mut by_block = vec![K::ZERO; blocks];
    for claim in &uses.claims {
        by_block
            .par_iter_mut()
            .zip(equality(claim.point, claim.blocks))
            .for_each(|(sum, weight)| *sum += claim.coefficient * weight);
    }
    let mut table = vec![K::ZERO; blocks * D];
    table
        .par_chunks_mut(D)
        .zip(&by_block)
        .for_each(|(lanes, &block)| {
            for (cell, &lane) in lanes.iter_mut().zip(&uses.form) {
                *cell = lane * block;
            }
        });
    table
}

/// The verifier's value of those weights at the point `(λ, β′)` where the
/// sum-check ends.
pub(crate) fn weight_at(uses: &Uses, point: &[K]) -> K {
    let (lanes, blocks) = point.split_at(LANE_VARS);
    let by_block: K = uses
        .claims
        .iter()
        .map(|claim| claim.coefficient * prefix_equality(claim.point, claim.blocks, blocks))
        .sum();
    dot(&equality(lanes, D), &uses.form) * by_block
}

#[cfg(test)]
#[path = "../tests/unit/key.rs"]
mod tests;
