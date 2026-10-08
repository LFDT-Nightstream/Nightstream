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
//! Invariants: all hashing is the workspace Poseidon2 permutation; no
//! Plonky3 0.8 type appears in the public API; the verifier reads only the
//! key, never the setup files or the matrices.

mod circuit;
mod field;
mod gkr;
mod hash;
mod matrix;
mod mle;
mod norm;
mod pcs;
mod ring;
mod setup;
mod sumcheck;
mod whir;

#[cfg(test)]
#[path = "../tests/internal/mod.rs"]
mod internal_tests;

use std::path::Path;

use neo_ajtai::nightstream_fprime_setup::{
    coefficient_block, MAX_MESSAGE_COLUMNS, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS,
};
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{D, F, K};
use neo_reductions::superneo_eval::MatrixRows;
use neo_transcript::Poseidon2Transcript;
use p3_challenger_v08::{CanObserve, CanSampleBits, FieldChallenger};
use p3_field::PrimeCharacteristicRing as _;
use p3_field_v08::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_security_v08::{ErrorBits, SecurityTerm};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::field::{coordinates, eq_table, from_coordinates, gl, re_im, Ext, Gl};
use crate::gkr::{GkrProof, Tree, TreeShape};
use crate::hash::Challenger;
use crate::matrix::{EarlyChallenges, EarlyValues, Structure, Tables};
use crate::mle::scaled_eq_point;
use crate::pcs::{Pcs, TablePlan};
use crate::ring::Mixing;
use crate::setup::{Leaf, Store, FOLD};

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
    #[error("layer-1 setup files: {0}")]
    Io(#[from] std::io::Error),
}

/// The verifier's constants of one circuit: the setup root, the sizes and the
/// matrix structure. Authority: derived from the matrices (`Key::derive`) or
/// a pinned copy. Never take a key from a prover.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Key {
    root: [u64; 4],
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

    /// Strict: the bytes must be the canonical encoding of the decoded key.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let key: Self = bincode::deserialize(bytes).map_err(|_| Error::Codec("key"))?;
        if key.to_bytes() != bytes {
            return Err(Error::Codec("non-canonical key"));
        }
        Ok(key)
    }

    fn cube_variables(&self) -> usize {
        block_variables(self.blocks) + LANE_VARIABLES
    }

    /// Every setup oracle has this many variables.
    fn setup_variables(&self) -> usize {
        self.cube_variables().max(self.structure.run_variables)
    }

    fn root(&self) -> [Gl; 4] {
        self.root.map(Gl::new)
    }

    fn words(&self) -> Vec<Gl> {
        let mut words = self.root().to_vec();
        words.extend([self.blocks, self.rows].map(Gl::from_usize));
        words.extend(self.structure.words());
        words
    }
}

fn block_variables(blocks: usize) -> usize {
    (blocks.next_power_of_two().trailing_zeros() as usize).max(1)
}

fn kappa() -> usize {
    PRODUCTION_VERIFIER_ROWS as usize
}

/// The key, and with `dir` the setup files, of `rows`.
fn key(rows: &dyn MatrixRows, tables: &Tables, dir: Option<&Path>) -> Result<Key, Error> {
    let shape = rows.shape();
    let blocks = shape.columns / D;
    if blocks == 0 || blocks as u64 > MAX_MESSAGE_COLUMNS {
        return Err(Error::Shape("matrix columns"));
    }
    let mut key = Key {
        root: [0; 4],
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
    key.root = root.map(|word| word.as_canonical_u64());
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
                *cell = Gl::new(coefficient);
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

/// One fixed CE(B, L) relation of a key: the production Ajtai key prefix, the
/// CCS matrices over the full carrier, the public width, the point length and
/// the norm bound. The WHIR configurations and the setup query count are
/// derived here, so prover and verifier cannot disagree on them.
pub struct Relation<'k> {
    key: &'k Key,
    shape: Shape,
    p0: Pcs,
    p1: Pcs,
    queries: usize,
}

pub(crate) struct Shape {
    pub(crate) blocks: usize,
    pub(crate) block_variables: usize,
    pub(crate) rows: usize,
    pub(crate) matrices: usize,
    pub(crate) kappa: usize,
    pub(crate) public_blocks: usize,
    pub(crate) point_variables: usize,
    pub(crate) norm_bound: u32,
}

impl Shape {
    fn cube_variables(&self) -> usize {
        self.block_variables + LANE_VARIABLES
    }

    fn words(&self) -> Vec<Gl> {
        [
            self.cube_variables(),
            self.blocks,
            self.rows,
            self.matrices,
            self.kappa,
            self.public_blocks,
            self.point_variables,
            self.norm_bound as usize,
        ]
        .map(Gl::from_usize)
        .to_vec()
    }
}

impl<'k> Relation<'k> {
    /// `norm_bound` is the honest witness bound `H` (every |z| ≤ H < B).
    /// `security_bits` is `-log2` of the soundness error this layer may add;
    /// each commitment, with the draws charged to it, gets half of it.
    pub fn new(
        key: &'k Key,
        public_blocks: usize,
        point_variables: usize,
        norm_bound: u32,
        security_bits: f64,
    ) -> Result<Self, Error> {
        let blocks = key.blocks;
        if public_blocks > blocks || norm_bound == 0 {
            return Err(Error::Shape("relation dimensions"));
        }
        if point_variables >= usize::BITS as usize || (1usize << point_variables) < (D * blocks).max(key.rows) {
            return Err(Error::Shape("point length"));
        }
        let shape = Shape {
            blocks,
            block_variables: block_variables(blocks),
            rows: key.rows,
            matrices: key.structure.matrices,
            kappa: kappa(),
            public_blocks,
            point_variables,
            norm_bound,
        };
        let target = security_bits + 1.0;
        let probe = Pcs::new(p1_plans(key, 0), target, &|_| Vec::new())?;
        let queries = query_count(target, probe.log2_candidates());
        let p1 = Pcs::new(p1_plans(key, queries), target, &|_| vec![query_term(queries)])?;
        if p1.log2_candidates() != probe.log2_candidates() {
            return Err(Error::Setup("P1 candidate count"));
        }
        let terms = outer_terms(&shape, &key.structure, p1.log2_candidates());
        let p0 = Pcs::new(p0_plans(&shape, &key.structure), target, &|_| terms.clone())?;
        Ok(Self {
            key,
            shape,
            p0,
            p1,
            queries,
        })
    }

    /// `-log2` of this layer's composed soundness error.
    pub fn security_bits(&self) -> f64 {
        let error = (-self.p0.security_bits()).exp2() + (-self.p1.security_bits()).exp2();
        -error.log2()
    }

    /// Hand the fold transcript over to layer 1 and bind the relation, the
    /// key and the statement before the first layer-1 challenge.
    fn start(&self, transcript: Poseidon2Transcript, statement: &Statement) -> Challenger {
        let mut challenger = hash::challenger(transcript);
        challenger.observe_slice(&self.shape.words());
        challenger.observe_slice(&self.key.words());
        challenger.observe_slice(&self.p0.profile_words());
        challenger.observe_slice(&self.p1.profile_words());
        challenger.observe(Gl::from_usize(self.queries));
        challenger.observe_slice(&statement.words());
        challenger
    }

    /// The setup leaves to query, drawn after the P1 root.
    fn query_indices(&self, challenger: &mut Challenger) -> Vec<usize> {
        let bits = self.key.setup_variables() + 1 - FOLD;
        (0..self.queries)
            .map(|_| challenger.sample_bits(bits))
            .collect()
    }
}

/// P0 tables and points: `z` at the norm leaf and the linear point; `e` at
/// the run point; `U` and the slot blocks at the pair point, the run point
/// and the late pair point; the row counts at the row point.
fn p0_plans(shape: &Shape, structure: &Structure) -> Vec<TablePlan> {
    let pairs = 2 * structure.matrices;
    vec![
        TablePlan {
            variables: shape.cube_variables(),
            width: 1,
            points: vec![vec![0]; 2],
        },
        TablePlan {
            variables: structure.run_variables,
            width: 2,
            points: vec![vec![0, 1]],
        },
        TablePlan {
            variables: structure.slot_variables,
            width: pairs + 1,
            points: vec![(0..pairs).collect(), vec![pairs], (0..=pairs).collect()],
        },
        TablePlan {
            variables: structure.row_variables,
            width: 1,
            points: vec![vec![0]],
        },
    ]
}

/// P1 tables and points: the folded key and run groups (three coordinates
/// each) at their fold claims and at every query point; `Ω^A` at the linear
/// block point and the late block point.
fn p1_plans(key: &Key, queries: usize) -> Vec<TablePlan> {
    let mut points = vec![vec![0, 1, 2], vec![3, 4, 5]];
    points.extend(std::iter::repeat_n((0..6).collect(), queries));
    vec![
        TablePlan {
            variables: key.setup_variables() - FOLD,
            width: 6,
            points,
        },
        TablePlan {
            variables: block_variables(key.blocks),
            width: 3,
            points: vec![vec![0, 1, 2]; 2],
        },
    ]
}

/// The setup query count. A wrong fold candidate passes one query with
/// probability below 1/2 (degree `2^{n-3}` against `2^{n-2}` points), so two
/// groups pass `t` queries with probability at most `2^{1-t}` per candidate.
/// Charged over the candidates `L`, this is at most half the commitment's
/// budget `2^{-target}` when `t = ⌈target + 2 + log2 L⌉`.
fn query_count(target: f64, log2_candidates: f64) -> usize {
    (target + 2.0 + log2_candidates).ceil() as usize
}

fn query_term(queries: usize) -> SecurityTerm {
    SecurityTerm::new("setup queries", ErrorBits::from_log2(queries as f64 - 1.0))
}

/// Opaque layer-1 proof. It contains no witness coordinate.
#[derive(Clone, Serialize, Deserialize)]
pub struct Proof {
    p0: pcs::Commitment,
    histogram: Vec<u32>,
    early: GkrProof,
    early_values: EarlyValues,
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
        let proof: Self = bincode::deserialize(bytes).map_err(|_| Error::Codec("decode"))?;
        if proof.to_bytes() != bytes {
            return Err(Error::Codec("non-canonical encoding"));
        }
        Ok(proof)
    }
}

/// Prove `claim` for `witness` (54 × blocks), continuing `transcript`.
pub fn prove(
    relation: &Relation<'_>,
    setup: &Setup,
    transcript: Poseidon2Transcript,
    claim: &Claim,
    witness: &Mat<F>,
) -> Result<Proof, Error> {
    let key = relation.key;
    if setup.key != *key {
        return Err(Error::Setup("the setup belongs to another key"));
    }
    let (shape, structure, tables) = (&relation.shape, &key.structure, &setup.tables);
    let statement = Statement::new(shape, claim)?;
    let z = witness_table(shape, witness)?;
    let histogram = norm::histogram(&z, shape.norm_bound)?;
    let early = tables.early(&statement.point);

    let mut challenger = relation.start(transcript, &statement);
    let slot_columns: Vec<Gl> = early
        .u
        .iter()
        .flat_map(|[re, im]| re.iter().chain(im))
        .copied()
        .chain(tables.blocks())
        .collect();
    let p0_tables = vec![
        z.clone(),
        [early.e[0].as_slice(), early.e[1].as_slice()].concat(),
        slot_columns,
        early.mult.clone(),
    ];
    let (p0, p0_data) = relation.p0.commit(p0_tables, &mut challenger);
    challenger.observe_slice(&norm::words(&histogram));
    let beta: Ext = challenger.sample_algebra_element();
    let challenges = early_challenges(&mut challenger);
    let mut trees = vec![Tree {
        factors: Vec::new(),
        denominator: z.par_iter().map(|&value| beta - value).collect(),
    }];
    trees.extend(tables.early_trees(&early, &statement.point, challenges));
    let (early_proof, claims) = gkr::prove(trees, &mut challenger);
    let early_values = tables.early_values(&early, &claims[1..]);
    observe_early(&early_values, &mut challenger);

    let mixing = Mixing::new(challenger.sample_algebra_element(), shape);
    let ubar = tables.slot_weights(&early, &mixing.eval_a());
    let key_weights = ring::key_weights(shape, &mixing);
    let eval_a = tables.column_weights(&ubar, shape.blocks);
    let weights = ring::block_weights(shape, &statement, &mixing, &eval_a, &key_weights);
    drop(eval_a);
    let (quotient, _) = ring::divide(&weights, &z, LANES, &ring::targets(&statement, &mixing));
    challenger.observe_algebra_slice(&quotient);
    let zeta: Ext = challenger.sample_algebra_element();
    let omega = padded(ring::evaluate(&weights, zeta), shape.block_variables);
    drop(weights);
    let (linear, linear_point, z_at_point) = prove_linear(omega, zeta, &z, &mut challenger);
    let block_point = &linear_point[LANE_VARIABLES..];

    let (_, lane_point) =
        scaled_eq_point(zeta, LANE_VARIABLES).ok_or(Error::Witness("ζ is a pole of the lane weights"))?;
    let variables = key.setup_variables();
    let key_point = setup_point(&[lane_point.as_slice(), block_point].concat(), variables);
    let key_table = key_table(&key_weights, variables);
    drop(key_weights);
    let lanes = structure.lane_tables(&ring::tau(zeta));
    let omega_a = tables.block_weights(&ubar, &lanes, shape.block_variables);
    let finals = [
        sumcheck::evaluate(&key_table, &key_point),
        sumcheck::evaluate(&omega_a, block_point),
        z_at_point,
    ];
    challenger.observe_algebra_slice(&finals);

    let mu: Ext = challenger.sample_algebra_element();
    let run_point = setup_point(&claims[1].point, variables);
    let run_table = run_table(tables, mu, variables);
    let partials = [
        setup::partials(&key_table, &key_point[FOLD..]),
        setup::partials(&run_table, &run_point[FOLD..]),
    ];
    for values in &partials {
        challenger.observe_algebra_slice(values);
    }
    let alpha: [Ext; FOLD] = std::array::from_fn(|_| challenger.sample_algebra_element());
    let fold_columns = [
        coordinates(&setup::fold(&key_table, &alpha)),
        coordinates(&setup::fold(&run_table, &alpha)),
    ]
    .concat();
    drop((key_table, run_table));
    let (p1, p1_data) = relation
        .p1
        .commit(vec![fold_columns, coordinates(&omega_a)], &mut challenger);
    let indices = relation.query_indices(&mut challenger);
    let leaves = indices
        .iter()
        .map(|&index| setup.store.read(index))
        .collect::<Result<Vec<_>, _>>()?;

    let beta: Ext = challenger.sample_algebra_element();
    let (late_proof, late) = gkr::prove(tables.late_trees(&ubar, &lanes, &omega_a, beta), &mut challenger);
    let slots = |point: &[Ext]| point[..structure.slot_variables].to_vec();
    let late_values = tables.slot_values(&early, &eq_table(&slots(&late[0].point)));
    observe_pairs(&late_values, &mut challenger);

    let p1_points = p1_points(&key_point, &run_point, &indices, block_point, &late[1].point, variables);
    let p1_opening = relation.p1.open(p1_data, &p1_points, &mut challenger);
    let p0_points = vec![
        claims[0].point.clone(),
        linear_point.clone(),
        claims[1].point.clone(),
        slots(&claims[4].point),
        slots(&claims[1].point),
        slots(&late[0].point),
        claims[2].point.clone(),
    ];
    let p0_opening = relation.p0.open(p0_data, &p0_points, &mut challenger);
    Ok(Proof {
        p0,
        histogram,
        early: early_proof,
        early_values,
        quotient: quotient.to_vec(),
        linear,
        finals,
        partials,
        p1,
        leaves,
        late: late_proof,
        late_values,
        p1_opening,
        p0_opening,
    })
}

/// Verify `proof` for `claim`, continuing `transcript`. Reads only the key.
pub fn verify(
    relation: &Relation<'_>,
    transcript: Poseidon2Transcript,
    claim: &Claim,
    proof: &Proof,
) -> Result<(), Error> {
    let (key, shape) = (relation.key, &relation.shape);
    let structure = &key.structure;
    let statement = Statement::new(shape, claim)?;
    let quotient: [Ext; D - 1] = proof
        .quotient
        .clone()
        .try_into()
        .map_err(|_| Error::Rejected("quotient length"))?;
    if proof.linear.len() != shape.cube_variables()
        || proof.histogram.len() != 2 * shape.norm_bound as usize + 1
        || proof.leaves.len() != relation.queries
        || proof.late_values.len() != structure.matrices
    {
        return Err(Error::Rejected("proof shape"));
    }

    let mut challenger = relation.start(transcript, &statement);
    relation.p0.observe(&proof.p0, &mut challenger);
    challenger.observe_slice(&norm::words(&proof.histogram));
    let beta: Ext = challenger.sample_algebra_element();
    let challenges = early_challenges(&mut challenger);
    let total = norm::table_sum(&proof.histogram, shape.norm_bound, beta)?;
    let mut shapes = vec![TreeShape {
        depth: shape.cube_variables(),
        factors: 0,
    }];
    shapes.extend(structure.early_shapes());
    let claims = gkr::verify(&proof.early, &shapes, &mut challenger)?;
    let [p, q] = claims[0].root;
    if q == Ext::ZERO || p != q * total {
        return Err(Error::Rejected("GKR root against the histogram"));
    }
    let z_leaf = beta - claims[0].values[0];
    let early = structure.check_early(&statement.point, challenges, &claims[1..], &proof.early_values)?;
    observe_early(&proof.early_values, &mut challenger);

    let mixing = Mixing::new(challenger.sample_algebra_element(), shape);
    challenger.observe_algebra_slice(&quotient);
    let zeta: Ext = challenger.sample_algebra_element();
    let value = ring::lifted_target(&ring::targets(&statement, &mixing), &quotient, zeta);
    let (point, last) = sumcheck::verify(&proof.linear, 2, value, &mut challenger)?;
    // The linear sum-check binds the block bits first, then the lane bits.
    let (block_point, lane_part) = point.split_at(shape.block_variables);
    let linear_point = [lane_part, block_point].concat();
    let [key_value, omega_value, z_value] = proof.finals;
    challenger.observe_algebra_slice(&proof.finals);

    // Ω~(s_b) = κ_ζ·A_λ~(x_ζ, s_b) + Eval_K + Ω^A~(s_b) + public.
    let (scale, lane_point) =
        scaled_eq_point(zeta, LANE_VARIABLES).ok_or(Error::Rejected("ζ is a pole of the lane weights"))?;
    let tau = ring::tau(zeta);
    let eval_k = mle::eval_k(&statement.point, block_point, &tau, shape.blocks);
    let [k_re, k_im] = mixing.eval_k();
    let public: Ext = (0..shape.public_blocks)
        .map(|block| eq_at(block_point, block) * mixing.public(block))
        .sum();
    let omega = scale * key_value + k_re * eval_k.re + k_im * eval_k.im + omega_value + public;
    if last != omega * sumcheck::evaluate(&lane_weights(zeta), lane_part) * z_value {
        return Err(Error::Rejected("linear claim"));
    }

    let mu: Ext = challenger.sample_algebra_element();
    for values in &proof.partials {
        challenger.observe_algebra_slice(values);
    }
    let alpha: [Ext; FOLD] = std::array::from_fn(|_| challenger.sample_algebra_element());
    let variables = key.setup_variables();
    let key_point = setup_point(&[lane_point.as_slice(), block_point].concat(), variables);
    let run_point = setup_point(&early.run_point, variables);
    let [row, slot, coefficient] = early.setup;
    let embedded: Ext = proof.early_values.block
        * early.run_point[structure.slot_variables..]
            .iter()
            .map(|&x| Ext::ONE - x)
            .product::<Ext>();
    let run_value = row + mu * (slot + mu * (coefficient + mu * embedded));
    let combine = |partials: &[Ext; 1 << FOLD], point: &[Ext]| -> Ext {
        eq_table(point)
            .iter()
            .zip(partials)
            .map(|(&e, &w)| e * w)
            .sum()
    };
    if combine(&proof.partials[0], &key_point[..FOLD]) != key_value
        || combine(&proof.partials[1], &run_point[..FOLD]) != run_value
    {
        return Err(Error::Rejected("honest-fold partials"));
    }

    relation.p1.observe(&proof.p1, &mut challenger);
    let indices = relation.query_indices(&mut challenger);
    let mut groups = vec![vec![Ext::ZERO; kappa() + RUN_ORACLES]; 2];
    for row in 0..kappa() {
        groups[0][row] = mixing.commitment(row);
    }
    let mut power = Ext::ONE;
    for weight in &mut groups[1][kappa()..] {
        *weight = power;
        power *= mu;
    }
    let root = key.root();
    let queried = indices
        .iter()
        .zip(&proof.leaves)
        .map(|(&index, leaf)| setup::check(&root, variables, index, leaf, &groups, &alpha))
        .collect::<Result<Vec<_>, _>>()?;

    let beta: Ext = challenger.sample_algebra_element();
    let late_claims = gkr::verify(
        &proof.late,
        &structure.late_shapes(shape.block_variables),
        &mut challenger,
    )?;
    let lanes = structure.lane_tables(&tau);
    let late = structure.check_late(beta, &mixing.eval_a(), &lanes, &late_claims, &proof.late_values)?;
    observe_pairs(&proof.late_values, &mut challenger);

    let p1_points = p1_points(
        &key_point,
        &run_point,
        &indices,
        block_point,
        &late.block_point,
        variables,
    );
    let opened = relation
        .p1
        .verify(&proof.p1, &proof.p1_opening, &p1_points, &mut challenger)?;
    let mut expected = vec![combine(&proof.partials[0], &alpha), combine(&proof.partials[1], &alpha)];
    expected.extend(queried.into_iter().flatten());
    expected.extend([omega_value, late.omega]);
    let values: Vec<Ext> = opened
        .iter()
        .flat_map(|batch| batch.chunks(3).map(from_coordinates))
        .collect();
    if values != expected {
        return Err(Error::Rejected("P1 openings"));
    }

    let slots = |point: &[Ext]| point[..structure.slot_variables].to_vec();
    let p0_points = vec![
        claims[0].point.clone(),
        linear_point,
        early.run_point.clone(),
        early.pair_point.clone(),
        slots(&early.run_point),
        late.pair_point.clone(),
        early.row_point.clone(),
    ];
    let opened = relation
        .p0
        .verify(&proof.p0, &proof.p0_opening, &p0_points, &mut challenger)?;
    let flat = |pairs: &[[Ext; 2]]| pairs.iter().flatten().copied().collect::<Vec<Ext>>();
    let [e_re, e_im, _] = proof.early_values.runs;
    let mut late_slots = flat(&proof.late_values);
    late_slots.push(late.block);
    let expected = vec![
        vec![z_leaf],
        vec![z_value],
        vec![e_re, e_im],
        flat(&proof.early_values.pairs),
        vec![proof.early_values.block],
        late_slots,
        vec![early.mult],
    ];
    if opened != expected {
        return Err(Error::Rejected("P0 openings"));
    }
    Ok(())
}

fn early_challenges(challenger: &mut Challenger) -> EarlyChallenges {
    EarlyChallenges {
        lookup: [challenger.sample_algebra_element(), challenger.sample_algebra_element()],
        scatter: [challenger.sample_algebra_element(), challenger.sample_algebra_element()],
    }
}

fn observe_early(values: &EarlyValues, challenger: &mut Challenger) {
    challenger.observe_algebra_slice(&values.runs);
    observe_pairs(&values.pairs, challenger);
    challenger.observe_algebra_element(values.block);
}

fn observe_pairs(pairs: &[[Ext; 2]], challenger: &mut Challenger) {
    for pair in pairs {
        challenger.observe_algebra_slice(pair);
    }
}

/// `eq(point, b)` for the bits of `b`.
fn eq_at(point: &[Ext], b: usize) -> Ext {
    point
        .iter()
        .enumerate()
        .map(|(t, &x)| if (b >> t) & 1 == 1 { x } else { Ext::ONE - x })
        .product()
}

/// A point on fewer variables, extended by zeros: an oracle embedded in the
/// low variables of a setup oracle has the same value there.
fn setup_point(point: &[Ext], variables: usize) -> Vec<Ext> {
    let mut point = point.to_vec();
    point.resize(variables, Ext::ZERO);
    point
}

/// `A_λ` on the cube, zero on padding lanes and blocks: the key group.
fn key_table(weights: &[[Ext; D]], variables: usize) -> Vec<Ext> {
    let mut table = vec![Ext::ZERO; 1 << variables];
    table
        .par_chunks_mut(LANES)
        .zip(weights)
        .for_each(|(cells, weight)| cells[..D].copy_from_slice(weight));
    table
}

/// `i + μ·σ + μ²·m + μ³·b'`: the run group.
fn run_table(tables: &Tables, mu: Ext, variables: usize) -> Vec<Ext> {
    let mut table = vec![Ext::ZERO; 1 << variables];
    let mut power = Ext::ONE;
    for k in 0..RUN_ORACLES {
        let column = tables.setup_column(k, variables);
        table
            .par_iter_mut()
            .zip(&column)
            .for_each(|(cell, &value)| *cell += power * value);
        power *= mu;
    }
    table
}

/// P1 points in plan order.
fn p1_points(
    key_point: &[Ext],
    run_point: &[Ext],
    indices: &[usize],
    block_point: &[Ext],
    late_block_point: &[Ext],
    variables: usize,
) -> Vec<Vec<Ext>> {
    let mut points = vec![key_point[FOLD..].to_vec(), run_point[FOLD..].to_vec()];
    points.extend(
        indices
            .iter()
            .map(|&index| setup::query_point(variables, index)),
    );
    points.extend([block_point.to_vec(), late_block_point.to_vec()]);
    points
}

/// The validated public part of a CE(B) claim, in layer-1 field types.
pub(crate) struct Statement {
    pub(crate) commitment: Vec<[Gl; D]>,
    pub(crate) public: Vec<[Gl; D]>,
    pub(crate) point: Vec<K>,
    pub(crate) eval_k: [K; D],
    pub(crate) eval_a: Vec<[K; D]>,
}

impl Statement {
    fn new(shape: &Shape, claim: &Claim) -> Result<Self, Error> {
        let c = &claim.c;
        if c.d != D || c.kappa != shape.kappa || c.data.len() != D * shape.kappa {
            return Err(Error::Shape("commitment"));
        }
        if claim.X.rows() != D || claim.X.cols() != shape.public_blocks || claim.m_in != D * shape.public_blocks {
            return Err(Error::Shape("public input"));
        }
        if claim.r.len() != shape.point_variables || claim.eval_a.len() != shape.matrices || claim.adv.is_some() {
            return Err(Error::Shape("evaluation claim"));
        }
        let ring = |values: &[K]| -> Result<[K; D], Error> {
            if values.len() < D || values[D..].iter().any(|&value| value != K::ZERO) {
                return Err(Error::Shape("ring evaluation"));
            }
            Ok(std::array::from_fn(|lane| values[lane]))
        };
        Ok(Self {
            commitment: (0..shape.kappa)
                .map(|row| std::array::from_fn(|lane| gl(c.col(row)[lane])))
                .collect(),
            public: (0..shape.public_blocks)
                .map(|block| std::array::from_fn(|lane| gl(claim.X[(lane, block)])))
                .collect(),
            point: claim.r.clone(),
            eval_k: ring(&claim.eval_k)?,
            eval_a: claim
                .eval_a
                .iter()
                .map(|values| ring(values))
                .collect::<Result<_, _>>()?,
        })
    }

    fn words(&self) -> Vec<Gl> {
        let rings = self
            .commitment
            .iter()
            .chain(&self.public)
            .flatten()
            .copied();
        let field = self
            .point
            .iter()
            .chain(&self.eval_k)
            .chain(self.eval_a.iter().flatten())
            .flat_map(|&value| re_im(value));
        rings.chain(field).collect()
    }
}

/// `z` on the cube: `z[64·block + lane] = Z[lane, block]`, zero elsewhere.
/// `z` on the cube: `z[64·block + lane] = Z[lane, block]`, zero elsewhere.
fn witness_table(shape: &Shape, witness: &Mat<F>) -> Result<Vec<Gl>, Error> {
    if witness.rows() != D || witness.cols() != shape.blocks {
        return Err(Error::Shape("witness dimensions"));
    }
    let mut table = vec![Gl::ZERO; 1 << shape.cube_variables()];
    table
        .par_chunks_mut(LANES)
        .take(shape.blocks)
        .enumerate()
        .for_each(|(block, cells)| {
            for (lane, cell) in cells.iter_mut().take(D).enumerate() {
                *cell = gl(witness[(lane, block)]);
            }
        });
    Ok(table)
}

/// `Λ(l) = ζ^l` for the 54 lanes, zero on the padding lanes.
fn lane_weights(zeta: Ext) -> Vec<Ext> {
    let mut power = Ext::ONE;
    (0..LANES)
        .map(|lane| {
            if lane >= D {
                return Ext::ZERO;
            }
            let value = power;
            power *= zeta;
            value
        })
        .collect()
}

fn padded(mut values: Vec<Ext>, variables: usize) -> Vec<Ext> {
    values.resize(1 << variables, Ext::ZERO);
    values
}

/// Prove `Σ_x Ω(x_block)·Λ(x_lane)·z(x)`, block bits first, then lane bits.
/// Returns the rounds, the point in cube order (lane bits first), and `z~`
/// at the point.
fn prove_linear(omega: Vec<Ext>, zeta: Ext, z: &[Gl], challenger: &mut Challenger) -> (Vec<Vec<Ext>>, Vec<Ext>, Ext) {
    let lanes = lane_weights(zeta);
    let combined: Vec<Ext> = (0..omega.len())
        .into_par_iter()
        .map(|block| {
            (0..D)
                .map(|lane| lanes[lane] * z[LANES * block + lane])
                .sum()
        })
        .collect();
    let (mut rounds, block_point, [omega_at_point, _]) = sumcheck::prove_product(omega, combined, challenger);
    let eq = eq_table(&block_point);
    let lane_values: Vec<Ext> = eq
        .par_iter()
        .enumerate()
        .fold(
            || vec![Ext::ZERO; LANES],
            |mut acc, (block, &weight)| {
                for (lane, value) in acc.iter_mut().enumerate() {
                    *value += weight * z[LANES * block + lane];
                }
                acc
            },
        )
        .reduce(
            || vec![Ext::ZERO; LANES],
            |a, b| a.iter().zip(&b).map(|(&x, &y)| x + y).collect(),
        );
    let scaled: Vec<Ext> = lanes
        .iter()
        .map(|&weight| omega_at_point * weight)
        .collect();
    let (lane_rounds, lane_point, [_, z_at_point]) = sumcheck::prove_product(scaled, lane_values, challenger);
    rounds.extend(lane_rounds);
    (rounds, [lane_point, block_point].concat(), z_at_point)
}

/// Every layer-1 draw made after P0, as a degree over |Ext|. Plonky3 charges
/// each over P0's candidates; `extra_log2` also charges it over P1's, since
/// a prover picks both candidates only at the openings.
fn outer_terms(shape: &Shape, structure: &Structure, extra_log2: f64) -> Vec<SecurityTerm> {
    let field_bits = (<Ext as Field>::bits() - 1) as f64;
    let term =
        |label, degree: f64| SecurityTerm::new(label, ErrorBits::from_log2(field_bits - degree.log2() - extra_log2));
    let size = |variables: usize| (1u64 << variables) as f64;
    // Per layer: rounds of degree at most 4, the batching draws and τ.
    let gkr = |depths: &[usize]| -> f64 {
        let deepest = depths.iter().copied().max().unwrap_or(0);
        (1..=deepest)
            .map(|k| (4 * k + depths.iter().filter(|&&depth| depth >= k).count()) as f64)
            .sum()
    };
    let (cube, runs, rows) = (shape.cube_variables(), structure.run_variables, structure.row_variables);
    let pairs = structure.slot_variables + structure.matrix_variables();
    let blocks = structure.slot_variables + 1;
    vec![
        term("norm logUp", size(cube) + f64::from(2 * shape.norm_bound + 1)),
        term("row lookup", 3.0 * (size(runs) + size(rows))),
        term("slot scatter", size(runs) + size(pairs) + 1.0),
        term("early GKR", gkr(&[cube, runs, rows, runs, pairs])),
        term("ring-row batching", (Mixing::rows(shape) - 1) as f64),
        term("quotient at zeta", (2 * D - 2) as f64),
        term("linear sum-check", 2.0 * cube as f64),
        term("run group", (RUN_ORACLES - 1) as f64),
        term("honest-fold partials", (2 * FOLD) as f64),
        term("block scatter", size(blocks) + size(shape.block_variables)),
        term("late GKR", gkr(&[blocks, shape.block_variables])),
    ]
}
