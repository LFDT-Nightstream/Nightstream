//! The Ajtai key as a committed polynomial (ePrint 2026/1983 section 9; SPEC.md section 6).

use std::fmt;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use neo_ajtai::nightstream_fprime_setup::ExpandedKey;
use neo_math::{sum_of_scaled, D, F, K};
use neo_reductions::superneo_eval::EqualityWeights;
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

const CELLS_PER_TASK: usize = 1 << 10;
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
    key: Arc<ExpandedKey>,
    verification: VerificationKey,
}

impl ProverKey {
    pub fn new(blocks: usize) -> Result<Self, Error> {
        let schedules = schedules_for(blocks)?;
        Self::of_key(schedules, ExpandedKey::production(schedules.blocks() as u64))
    }

    pub fn shared(blocks: usize) -> Result<Arc<Self>, Error> {
        static KEYS: [Mutex<Weak<ProverKey>>; SCHEDULES.len()] = [const { Mutex::new(Weak::new()) }; SCHEDULES.len()];
        let index = schedule_index(blocks)?;
        let current = || {
            KEYS[index]
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        };
        if let Some(key) = current().upgrade() {
            return Ok(key);
        }
        let key = Arc::new(Self::new(blocks)?);
        let mut slot = current();
        if let Some(existing) = slot.upgrade() {
            return Ok(existing);
        }
        *slot = Arc::downgrade(&key);
        Ok(key)
    }

    fn of_key(schedules: &'static Schedules, key: Arc<ExpandedKey>) -> Result<Self, Error> {
        let (akita, committed, verification) = committed(schedules, cells(&key))?;
        Ok(Self {
            akita,
            committed,
            key,
            verification,
        })
    }

    pub fn verification_key(&self) -> &VerificationKey {
        &self.verification
    }

    pub(crate) fn fold(&self, omega: K) -> Vec<K> {
        fold(&self.key, omega)
    }
}

fn cells(key: &ExpandedKey) -> Vec<Base> {
    key.words()
        .par_iter()
        .map(|&word| akita::base(F::new(word)))
        .collect()
}

// SPEC.md section 6.
fn fold(key: &ExpandedKey, omega: K) -> Vec<K> {
    let powers = powers(omega, KAPPA);
    let cells = key.blocks() as usize * D;
    let mut folded = vec![K::ZERO; cells];
    folded
        .par_chunks_mut(CELLS_PER_TASK)
        .enumerate()
        .for_each_init(Vec::new, |words: &mut Vec<[F; KAPPA]>, (task, folded)| {
            let start = task * CELLS_PER_TASK;
            words.resize(folded.len(), [F::ZERO; KAPPA]);
            for (row, key_row) in key.words().chunks_exact(cells).enumerate() {
                for (cell, &word) in words.iter_mut().zip(&key_row[start..]) {
                    cell[row] = F::new(word);
                }
            }
            for (cell, words) in folded.iter_mut().zip(words.iter()) {
                *cell = sum_of_scaled(words.iter().copied().zip(powers.iter().copied()));
            }
        });
    folded
}

pub fn key_blocks(blocks: usize) -> Result<usize, Error> {
    Ok(schedules_for(blocks)?.blocks())
}

fn schedules_for(blocks: usize) -> Result<&'static Schedules, Error> {
    Ok(&SCHEDULES[schedule_index(blocks)?])
}

fn schedule_index(blocks: usize) -> Result<usize, Error> {
    SCHEDULES
        .iter()
        .position(|schedules| schedules.blocks() >= blocks)
        .ok_or(Error::Layout("no Akita schedule row has a key for this many blocks"))
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

/// SPEC.md Algorithm 9, step 10.
pub(crate) fn block_weights(uses: &Uses) -> Vec<K> {
    let blocks = uses
        .claims
        .iter()
        .map(|claim| claim.blocks)
        .max()
        .unwrap_or(0);
    let points: Vec<EqualityWeights> = uses
        .claims
        .iter()
        .map(|claim| EqualityWeights::new(claim.point))
        .collect();
    (0..blocks)
        .into_par_iter()
        .map(|block| {
            uses.claims
                .iter()
                .zip(&points)
                .filter(|(claim, _)| block < claim.blocks)
                .map(|(claim, point)| claim.coefficient * point.at(block))
                .sum()
        })
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
