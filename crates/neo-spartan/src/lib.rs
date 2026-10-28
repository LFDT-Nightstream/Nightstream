//! Layer 1 of Nightstream compression: an argument of knowledge for one
//! SuperNeo CE(B, L) claim (SuperNeo Definition 21) with a succinct verifier.
//!
//! Owns: every Plonky3 0.8 type in the workspace, the layer-1 Fiat-Shamir
//! schedule, the verifier key, the prover's setup, and the two per-proof
//! WHIR commitments. Does not own: how the claim was produced (layer 0:
//! Pi_CCS + Pi_RLC), the state binding, or the outer proof bytes.
//!
//! The witness `z` (54 lanes per block) is one column on the cube
//! `(block: high bits, lane: low 6 bits)`; lanes 54..64 are zero.
//! - Norm: a public histogram over `[-H, H]` and logUp by GKR.
//! - Commitment, public input, Eval_K and Eval_A: one batched ring row
//!   `Σ_b ω_b ⋆ z_b = Ȳ` checked at a random `ζ` through the quotient by Φ81,
//!   then one linear sum-check. Its last claim needs `Ω~(s_b)`, which the
//!   verifier assembles without the Ajtai key or the matrices:
//!   - the key part by the honest fold of the setup oracles (`setup.rs`);
//!   - Eval_K by long division (`mle.rs`);
//!   - Eval_A from the two-level scatter (`matrix.rs`), checked by logUp;
//!   - the public part directly.
//! - P0 (early) commits `z`, the run values `e`, the slot sums `U`, a copy of
//!   the slot blocks and the row counts. P1 (late) commits the two folded
//!   setup groups and the block weights `Ω^A`.
//!
//! - The prover is `prover.rs`. The relation, its WHIR plans and the points
//!   that both sides open are `relation.rs`.
//! - The verifier (`verifier.rs`) is written once over `circuit::Backend`:
//!   `verify` reads it on plain values, the shrink layer as rows.
//!
//! Invariants: all hashing is the workspace Poseidon2 permutation; no
//! Plonky3 0.8 type appears in the public API; the verifier reads only the
//! key, never the setup files or the matrices.

mod circuit;
mod device;
mod field;
mod gkr;
mod hash;
mod matrix;
mod merkle;
mod mle;
mod norm;
mod pcs;
mod prover;
mod relation;
mod ring;
mod scratch;
mod setup;
pub mod shrink;
mod sumcheck;
mod verifier;
mod whir;

#[cfg(test)]
#[path = "../tests/internal/mod.rs"]
mod internal_tests;

use std::path::Path;

use neo_ajtai::nightstream_fprime_setup::{coefficient_block, MAX_MESSAGE_COLUMNS, PRODUCTION_SEED};
use neo_ajtai::Commitment;
use neo_math::{D, F, K};
use neo_reductions::superneo_eval::MatrixRows;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing as _;
use p3_field_v08::PrimeCharacteristicRing;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::field::{from_word, gl, re_im, Ext, Gl};
use crate::gkr::GkrProof;
use crate::matrix::{EarlyValues, Structure, Tables};
use crate::relation::{kappa, Shape};
use crate::setup::{Leaf, Store, FOLD};
use crate::verifier::ProofView;

pub use crate::circuit::fold::FoldTranscript;
pub use crate::circuit::r1cs::R1cs;
pub use crate::circuit::{Backend, Native};
pub use crate::device::{Accelerator, Device, GkrLayer};
pub use crate::prover::prove;
pub use crate::relation::Relation;

/// The claim this crate proves.
pub type Claim = neo_ccs::CeClaim<Commitment, F, K>;

/// Lanes per block on the committed cube: `D = 54`, padded to a power of two.
const LANE_VARIABLES: usize = 6;
const LANES: usize = 1 << LANE_VARIABLES;
/// Setup oracles after the key rows: run rows, slots, coefficients, slot blocks.
const RUN_ORACLES: usize = 4;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("layer-1 setup: {0}")]
    Setup(&'static str),
    #[error("layer-1 shape: {0}")]
    Shape(&'static str),
    #[error("layer-1 witness: {0}")]
    Witness(&'static str),
    #[error("layer-1 proof rejected: {0}")]
    Rejected(&'static str),
    #[error("layer-1 proof bytes: {0}")]
    Codec(&'static str),
    #[error("layer-1 file I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("layer-1 device: {0}")]
    Device(String),
}

/// The verifier's constants of one circuit: the setup root, the sizes and the
/// matrix structure. Authority: derived from the matrices (`Key::derive`) or
/// a pinned copy. Never take a key from a prover.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Key {
    /// The setup root. Plonky3's encoding of a field word is canonical, so a
    /// word `>= p` does not decode.
    root: [Gl; 4],
    blocks: usize,
    rows: usize,
    structure: Structure,
}

impl Key {
    /// Derive the key from the matrices: the setup root, computed without files.
    pub fn derive(rows: &dyn MatrixRows) -> Result<Self, Error> {
        key(rows, &Tables::from_rows(rows)?, None)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("an in-memory key always encodes")
    }

    /// Strict: the bytes must be the canonical encoding of a well-formed key.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let key: Self = decode_canonical(bytes, "key")?;
        key.check()?;
        Ok(key)
    }

    /// The key is well formed: its structure is one that `derive` can
    /// produce, so the verifier's step functions and tables are defined.
    pub fn check(&self) -> Result<(), Error> {
        if self.blocks == 0 {
            return Err(Error::Codec("key without blocks"));
        }
        self.structure.check()
    }

    /// The key's matrix sizes are those of `rows`.
    pub fn fits(&self, rows: &dyn MatrixRows) -> bool {
        let shape = rows.shape();
        shape.columns == self.blocks * D && shape.rows == self.rows
    }

    fn cube_variables(&self) -> usize {
        block_variables(self.blocks) + LANE_VARIABLES
    }

    /// Every setup oracle has this many variables.
    fn setup_variables(&self) -> usize {
        self.cube_variables().max(self.structure.run_variables)
    }

    fn root(&self) -> [Gl; 4] {
        self.root
    }

    fn words(&self) -> Vec<Gl> {
        let mut words = self.root().to_vec();
        words.extend([self.blocks, self.rows].map(Gl::from_usize));
        words.extend(self.structure.words());
        words
    }
}

/// Decode `bytes` as the canonical encoding of a `T`. Reading stops at the
/// input length, so no length prefix can make the decoder allocate more than
/// the input holds.
pub(crate) fn decode_canonical<T: Serialize + serde::de::DeserializeOwned>(
    bytes: &[u8],
    what: &'static str,
) -> Result<T, Error> {
    use bincode::Options;
    let value: T = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .allow_trailing_bytes()
        .with_limit(bytes.len() as u64)
        .deserialize(bytes)
        .map_err(|_| Error::Codec(what))?;
    if bincode::serialize(&value).map_err(|_| Error::Codec(what))? != bytes {
        return Err(Error::Codec(what));
    }
    Ok(value)
}

fn block_variables(blocks: usize) -> usize {
    (blocks.next_power_of_two().trailing_zeros() as usize).max(1)
}

/// The key, and with `dir` the setup files, of `rows`.
fn key(rows: &dyn MatrixRows, tables: &Tables, dir: Option<&Path>) -> Result<Key, Error> {
    let shape = rows.shape();
    let blocks = shape.columns / D;
    if blocks == 0 || blocks as u64 > MAX_MESSAGE_COLUMNS {
        return Err(Error::Shape("matrix columns"));
    }
    let mut key = Key {
        root: [Gl::ZERO; 4],
        blocks,
        rows: shape.rows,
        structure: tables.structure().clone(),
    };
    let variables = key.setup_variables();
    let root = setup::build(
        variables,
        kappa() + RUN_ORACLES,
        |k| oracle(tables, blocks, k, variables),
        dir,
    )?;
    key.root = root;
    Ok(key)
}

/// Setup oracle `k`: key row `k` on the cube for `k < κ`, then the run columns.
fn oracle(tables: &Tables, blocks: usize, k: usize, variables: usize) -> Vec<Gl> {
    if k >= kappa() {
        return tables.setup_column(k - kappa(), variables);
    }
    let mut column = vec![Gl::ZERO; 1 << variables];
    column
        .par_chunks_mut(LANES)
        .take(blocks)
        .enumerate()
        .for_each(|(block, cells)| {
            let key = coefficient_block(&PRODUCTION_SEED, k as u32, block as u64);
            for (cell, &coefficient) in cells.iter_mut().zip(&key) {
                *cell = from_word(coefficient);
            }
        });
    column
}

/// The prover's setup: the key, the run and slot tables, and the setup files.
pub struct Setup {
    key: Key,
    tables: Tables,
    store: Store,
}

impl Setup {
    /// Write the setup files of `rows` into the existing directory `dir`.
    pub fn build(rows: &dyn MatrixRows, dir: &Path) -> Result<Self, Error> {
        let tables = Tables::from_rows(rows)?;
        let key = key(rows, &tables, Some(dir))?;
        let store = Store::open(dir, key.setup_variables(), kappa() + RUN_ORACLES, &key.root())?;
        Ok(Self { key, tables, store })
    }

    /// Reopen files built for `key`. The files' root must equal the key's: this
    /// detects stale files, it gives them no authority.
    pub fn open(rows: &dyn MatrixRows, dir: &Path, key: &Key) -> Result<Self, Error> {
        let tables = Tables::from_rows(rows)?;
        let shape = rows.shape();
        if tables.structure() != &key.structure || shape.rows != key.rows || shape.columns != D * key.blocks {
            return Err(Error::Setup("the matrices do not match the key"));
        }
        let store = Store::open(dir, key.setup_variables(), kappa() + RUN_ORACLES, &key.root())?;
        Ok(Self {
            key: key.clone(),
            tables,
            store,
        })
    }

    pub fn key(&self) -> &Key {
        &self.key
    }
}

/// Opaque layer-1 proof. It contains no witness coordinate.
#[derive(Clone, Serialize, Deserialize)]
pub struct Proof {
    p0: pcs::Commitment,
    histogram: Vec<u32>,
    early: GkrProof,
    early_values: EarlyValues<Ext>,
    quotient: Vec<Ext>,
    linear: Vec<Vec<Ext>>,
    /// `A_λ~` at the key point, `Ω^A~(s_b)`, `z~(s)`.
    finals: [Ext; 3],
    /// Honest-fold partial evaluations of the key group and the run group.
    partials: [[Ext; 1 << FOLD]; 2],
    p1: pcs::Commitment,
    leaves: Vec<Leaf>,
    late: GkrProof,
    late_values: Vec<[Ext; 2]>,
    p1_opening: pcs::Opening,
    p0_opening: pcs::Opening,
}

impl Proof {
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("an in-memory proof always encodes")
    }

    /// Strict: the bytes must be the canonical encoding of the decoded proof.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        decode_canonical(bytes, "layer-1 proof")
    }
}

/// Verify `proof` for `claim`, continuing `transcript`. Reads only the key.
pub fn verify(
    relation: &Relation<'_>,
    transcript: Poseidon2Transcript,
    claim: &Claim,
    proof: &Proof,
) -> Result<(), Error> {
    if transcript.absorbed() != 0 {
        return Err(Error::Shape("the fold transcript holds a partial chunk"));
    }
    let claim = ClaimWords::new(&relation.shape, claim)?;
    let transcript = FoldTranscript::from_state(transcript.state().map(gl));
    verify_with(&mut Native, relation, transcript, &claim, Some(proof))
}

/// Verify `proof` for `claim` over any backend, continuing the fold
/// `transcript`. With `None` the proof reads as zeros: a shape run.
pub fn verify_with<B: Backend>(
    b: &mut B,
    relation: &Relation<'_>,
    transcript: FoldTranscript<B>,
    claim: &ClaimWords<B::F>,
    proof: Option<&Proof>,
) -> Result<(), Error> {
    claim.check_shape(&relation.shape)?;
    let seed = transcript.handoff(b);
    let view = ProofView::read(b, relation, proof)?;
    verifier::verify(b, relation, seed, claim, &view)
}

/// The public part of a CE(B) claim as words; a `K` value is `[re, im]`.
/// `F` is a layer-1 value for the prover and a backend word for a verifier.
/// The words have no authority: the caller computes them or `new` checks
/// them.
pub struct ClaimWords<F> {
    /// The `κ` commitment rows, coefficient form.
    pub commitment: Vec<[F; D]>,
    /// The public input, one ring element per block.
    pub public: Vec<[F; D]>,
    pub point: Vec<[F; 2]>,
    pub eval_k: [[F; 2]; D],
    pub eval_a: Vec<[[F; 2]; D]>,
}

impl ClaimWords<Gl> {
    fn new(shape: &Shape, claim: &Claim) -> Result<Self, Error> {
        let c = &claim.c;
        if c.d() != D || c.kappa() != shape.kappa || c.data().len() != D * shape.kappa {
            return Err(Error::Shape("commitment"));
        }
        if claim.X.rows() != D || claim.X.cols() != shape.public_blocks || claim.m_in != D * shape.public_blocks {
            return Err(Error::Shape("public input"));
        }
        if claim.r.len() != shape.point_variables || claim.eval_a.len() != shape.matrices || claim.adv.is_some() {
            return Err(Error::Shape("evaluation claim"));
        }
        let ring = |values: &[K]| -> Result<[[Gl; 2]; D], Error> {
            if values.len() < D || values[D..].iter().any(|&value| value != K::ZERO) {
                return Err(Error::Shape("ring evaluation"));
            }
            Ok(std::array::from_fn(|lane| re_im(values[lane])))
        };
        Ok(Self {
            commitment: (0..shape.kappa)
                .map(|row| std::array::from_fn(|lane| gl(c.col(row)[lane])))
                .collect(),
            public: (0..shape.public_blocks)
                .map(|block| std::array::from_fn(|lane| gl(claim.X[(lane, block)])))
                .collect(),
            point: claim.r.iter().map(|&value| re_im(value)).collect(),
            eval_k: ring(&claim.eval_k)?,
            eval_a: claim
                .eval_a
                .iter()
                .map(|values| ring(values))
                .collect::<Result<_, _>>()?,
        })
    }
}

impl<F: Copy> ClaimWords<F> {
    fn check_shape(&self, shape: &Shape) -> Result<(), Error> {
        if self.commitment.len() != shape.kappa
            || self.public.len() != shape.public_blocks
            || self.point.len() != shape.point_variables
            || self.eval_a.len() != shape.matrices
        {
            return Err(Error::Shape("claim words"));
        }
        Ok(())
    }

    fn words(&self) -> Vec<F> {
        let rings = self.commitment.iter().chain(&self.public).flatten();
        let field = self
            .point
            .iter()
            .chain(&self.eval_k)
            .chain(self.eval_a.iter().flatten())
            .flatten();
        rings.chain(field).copied().collect()
    }
}
