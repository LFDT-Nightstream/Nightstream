//! The Ajtai key as a committed polynomial (ePrint 2026/1983 section 9; SPEC.md section 6).

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

pub(crate) const ROW_VARS: usize = 5;
const _: () = assert!(KAPPA <= 1 << ROW_VARS);

const CELLS_PER_TASK: usize = 1 << 12;
const MAGIC: &[u8; 16] = b"NS-COMPRESS-VK-1";

#[derive(Clone)]
pub struct VerificationKey {
    schedules: &'static Schedules,
    commitment: Vec<u8>,
    setup_prefixes: Vec<u8>,
    akita: Arc<[OnceLock<Verifier>]>,
}

impl VerificationKey {
    pub fn new(blocks: usize) -> Result<Self, Error> {
        Ok(Self::pinned(schedules_for(blocks)?))
    }

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

    pub fn blocks(&self) -> usize {
        self.schedules.blocks()
    }

    pub(crate) fn commitment(&self) -> &[u8] {
        &self.commitment
    }

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

pub struct ProverKey {
    pub(crate) akita: Prover,
    pub(crate) committed: Committed,
    table: Vec<F>,
    verification: VerificationKey,
}

impl ProverKey {
    pub fn new(blocks: usize) -> Result<Self, Error> {
        let schedules = schedules_for(blocks)?;
        Self::of_elements(schedules, production_element)
    }

    pub(crate) fn of_elements(
        schedules: &'static Schedules,
        element: impl Fn(u32, u64) -> [[u64; D]; 2] + Sync,
    ) -> Result<Self, Error> {
        let table = table(schedules.blocks(), element, F::new);
        let cells = table.par_iter().copied().map(akita::base).collect();
        let table = by_cell(&table);
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

    pub(crate) fn fold(&self, omega: K) -> Vec<K> {
        fold(&self.table, omega)
    }
}

fn by_cell(rows: &[F]) -> Vec<F> {
    let slab = rows.len() / KAPPA;
    (0..rows.len())
        .into_par_iter()
        .map(|index| rows[index % KAPPA * slab + index / KAPPA])
        .collect()
}

fn fold(table: &[F], omega: K) -> Vec<K> {
    let (real, imaginary): (Vec<F>, Vec<F>) = powers(omega, KAPPA)
        .iter()
        .map(|power| {
            let [real, imaginary] = power.as_coeffs();
            (real, imaginary)
        })
        .unzip();
    table
        .par_chunks(KAPPA)
        .with_min_len(CELLS_PER_TASK)
        .map(|rows| {
            let rows = rows.iter().copied();
            K::from_coeffs([
                F::sum_of_products(real.iter().copied().zip(rows.clone())),
                F::sum_of_products(imaginary.iter().copied().zip(rows)),
            ])
        })
        .collect()
}

pub fn key_blocks(blocks: usize) -> Result<usize, Error> {
    Ok(schedules_for(blocks)?.blocks())
}

fn schedules_for(blocks: usize) -> Result<&'static Schedules, Error> {
    SCHEDULES
        .iter()
        .find(|schedules| schedules.blocks() >= blocks)
        .ok_or(Error::Layout("no Akita schedule row has a key for this many blocks"))
}

fn production_element(row: u32, pair: u64) -> [[u64; D]; 2] {
    production_key_pair(row, [2 * pair, 2 * pair + 1])
}

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

fn committed(schedules: &'static Schedules, cells: Vec<Base>) -> Result<(Prover, Committed, VerificationKey), Error> {
    let akita = Prover::new(schedules)?;
    let committed = akita.commit(schedules.first(), Polynomial::Key, cells)?;
    let verification = VerificationKey::of_parts(schedules, committed.bytes.clone(), akita.setup_prefixes()?);
    Ok((akita, committed, verification))
}

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
    (0..blocks * D)
        .into_par_iter()
        .map(|cell| uses.form[cell % D] * by_block[cell / D])
        .collect()
}

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
