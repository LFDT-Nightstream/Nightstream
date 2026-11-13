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

use akita_params::GroupCommitPhaseParams;
use jolt_field::Ring;
use neo_ajtai::nightstream_fprime_setup::production_key_pair;
use neo_math::{KExtensions, D, F, K};
use p3_field::{Field, PrimeCharacteristicRing};
use rayon::prelude::*;

use crate::akita::prover::{Committed, Prover};
use crate::akita::{self, Base, Polynomial, Schedules, SCHEDULES};
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
/// public seeds, and the verifier reads neither table. The key is authority:
/// its commitment must be the one that `VerificationKey::derive` makes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerificationKey {
    blocks: usize,
    commitment: Vec<u8>,
    setup_prefixes: Vec<u8>,
}

impl VerificationKey {
    /// The key for claims of at most `blocks` blocks, from the public seeds.
    /// It expands the key polynomial and commits it, and keeps neither.
    pub fn derive(blocks: usize) -> Result<Self, Error> {
        let schedules = schedules_for(blocks)?;
        Ok(committed(schedules, table(schedules.blocks(), production_element, Base::from_u64))?.2)
    }

    /// The key polynomial holds this many blocks of each key row, so the
    /// key proves claims of at most this many blocks.
    pub fn blocks(&self) -> usize {
        self.blocks
    }

    pub(crate) fn commitment(&self) -> &[u8] {
        &self.commitment
    }

    pub(crate) fn setup_prefixes(&self) -> &[u8] {
        &self.setup_prefixes
    }

    /// The key's strict bytes: a tag, the blocks, then each Akita byte
    /// string after its length.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend((self.blocks as u64).to_le_bytes());
        for part in [&self.commitment, &self.setup_prefixes] {
            bytes.extend((part.len() as u64).to_le_bytes());
            bytes.extend_from_slice(part);
        }
        bytes
    }

    /// Decode untrusted bytes. A key has one encoding: its size has
    /// schedule rows, its commitment has the profile of those rows, each
    /// Akita byte string is canonical, and no byte follows the key. The
    /// bytes do not show that the commitment is the one of the Ajtai key.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut input = bytes;
        if take(&mut input, MAGIC.len())? != MAGIC {
            return Err(Error::Key("format tag"));
        }
        let blocks = word(&mut input)?;
        let schedules = Schedules::of_blocks(blocks).ok_or(Error::Key("no schedule row has a key of this size"))?;
        let mut part = || -> Result<Vec<u8>, Error> {
            let length = word(&mut input)?;
            Ok(take(&mut input, length)?.to_vec())
        };
        let (commitment, setup_prefixes) = (part()?, part()?);
        if !input.is_empty() {
            return Err(Error::Key("bytes after the key"));
        }
        if *akita::commitment(&commitment)?.profile() != key_profile(schedules)? {
            return Err(Error::Key("the commitment has another profile"));
        }
        akita::setup_prefixes(&setup_prefixes)?;
        Ok(Self {
            blocks,
            commitment,
            setup_prefixes,
        })
    }
}

fn take<'a>(input: &mut &'a [u8], count: usize) -> Result<&'a [u8], Error> {
    let (taken, rest) = input
        .split_at_checked(count)
        .ok_or(Error::Key("the bytes end before the key does"))?;
    *input = rest;
    Ok(taken)
}

fn word(input: &mut &[u8]) -> Result<usize, Error> {
    let word = u64::from_le_bytes(take(input, 8)?.try_into().expect("eight bytes"));
    usize::try_from(word).map_err(|_| Error::Key("a length exceeds the address space"))
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

fn key_profile(schedules: &Schedules) -> Result<GroupCommitPhaseParams, Error> {
    let catalog = schedules.catalog()?;
    Ok(*akita::profile(
        akita::row(&catalog, schedules.first())?,
        Polynomial::Key,
    ))
}

/// Commit the key polynomial with the cells `cells`.
fn committed(schedules: &'static Schedules, cells: Vec<Base>) -> Result<(Prover, Committed, VerificationKey), Error> {
    let akita = Prover::new(schedules)?;
    let committed = akita.commit(schedules.first(), Polynomial::Key, cells)?;
    let verification = VerificationKey {
        blocks: schedules.blocks(),
        commitment: committed.bytes.clone(),
        setup_prefixes: akita.setup_prefixes()?,
    };
    Ok((akita, committed, verification))
}

/// `(s, scale)` with `ω^i = scale · eq(s, i)` for each row `i`:
/// `s_k = ω^(2^k) / (1 + ω^(2^k))` and `scale = Π_k (1 + ω^(2^k))`. No such
/// point exists when a factor of `scale` is zero.
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
