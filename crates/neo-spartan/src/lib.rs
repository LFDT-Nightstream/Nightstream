//! Layer 1 of Nightstream compression: an argument of knowledge for one
//! SuperNeo CE(B, L) claim (SuperNeo Definition 21).
//!
//! Owns: every Plonky3 0.8 type in the workspace, the layer-1 Fiat-Shamir
//! schedule, and the WHIR commitment to the claim's witness.
//! Does not own: how the claim was produced (layer 0: Pi_CCS + Pi_RLC), the
//! state binding, or the outer proof bytes.
//!
//! The witness `z` (54 lanes per block) is committed once, as one column on
//! the cube `(block: high bits, lane: low 6 bits)`; lanes 54..64 are zero.
//! - Norm: a public histogram over `[-H, H]` and a GKR proof of
//!   `Σ_x 1/(β - z(x)) = Σ_t m_t/(β - t)` (logUp).
//! - Commitment, public input, Eval_K and Eval_A: one batched ring row
//!   `Σ_b ω_b ⋆ z_b = Ȳ` checked at a random `ζ` through the quotient by Φ81,
//!   then one linear sum-check.
//! - WHIR opens `z` at the GKR leaf point and the linear point.
//!
//! Invariants: all hashing is the workspace Poseidon2 permutation; no
//! Plonky3 0.8 type appears in the public API.

mod field;
mod gkr;
mod hash;
mod norm;
mod pcs;
mod ring;
mod sumcheck;

#[cfg(test)]
#[path = "../tests/internal/mod.rs"]
mod internal_tests;

use neo_ajtai::nightstream_fprime_setup::{MAX_MESSAGE_COLUMNS, PRODUCTION_VERIFIER_ROWS};
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{D, F, K};
use neo_reductions::superneo_eval::MatrixRows;
use neo_transcript::Poseidon2Transcript;
use p3_challenger_v08::{CanObserve, FieldChallenger};
use p3_field::PrimeCharacteristicRing as _;
use p3_field_v08::{Field, PrimeCharacteristicRing};
use p3_security_v08::{ErrorBits, SecurityTerm};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::field::{eq_table, gl, re_im, Ext, Gl};
use crate::gkr::GkrProof;
use crate::hash::Challenger;
use crate::pcs::Pcs;
use crate::ring::Mixing;

/// The claim this crate proves.
pub type Claim = neo_ccs::CeClaim<Commitment, F, K>;

/// Lanes per block on the committed cube: `D = 54`, padded to a power of two.
const LANE_VARIABLES: usize = 6;
const LANES: usize = 1 << LANE_VARIABLES;

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
}

/// One fixed CE(B, L) relation: the production Ajtai key prefix, the CCS
/// matrices over the full carrier, the public width, the point length and the
/// norm bound. The WHIR configuration is derived here, so prover and verifier
/// cannot disagree on it.
pub struct Relation<'a> {
    matrices: &'a dyn MatrixRows,
    shape: Shape,
    pcs: Pcs,
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

impl<'a> Relation<'a> {
    /// `norm_bound` is the honest witness bound `H` (every |z| ≤ H < B).
    /// `security_bits` is `-log2` of the soundness error this layer may add.
    pub fn new(
        matrices: &'a dyn MatrixRows,
        public_blocks: usize,
        point_variables: usize,
        norm_bound: u32,
        security_bits: f64,
    ) -> Result<Self, Error> {
        let source = matrices.shape();
        if !source.columns.is_multiple_of(D) || source.rows == 0 || source.matrices == 0 {
            return Err(Error::Shape("matrix source"));
        }
        let blocks = source.columns / D;
        if blocks as u64 > MAX_MESSAGE_COLUMNS || public_blocks > blocks || norm_bound == 0 {
            return Err(Error::Shape("relation dimensions"));
        }
        if point_variables >= usize::BITS as usize || (1usize << point_variables) < source.columns.max(source.rows) {
            return Err(Error::Shape("point length"));
        }
        let shape = Shape {
            blocks,
            block_variables: blocks.next_power_of_two().trailing_zeros() as usize,
            rows: source.rows,
            matrices: source.matrices,
            kappa: PRODUCTION_VERIFIER_ROWS as usize,
            public_blocks,
            point_variables,
            norm_bound,
        };
        let pcs = Pcs::new(shape.cube_variables(), security_bits, &outer_terms(&shape))?;
        Ok(Self { matrices, shape, pcs })
    }

    /// `-log2` of this layer's composed soundness error.
    pub fn security_bits(&self) -> f64 {
        self.pcs.security_bits()
    }

    /// Hand the fold transcript over to layer 1 and bind the relation and the
    /// statement before the first layer-1 challenge.
    fn start(&self, transcript: Poseidon2Transcript, statement: &Statement) -> Challenger {
        let mut challenger = hash::challenger(transcript);
        challenger.observe_slice(&self.shape.words());
        challenger.observe_slice(&self.pcs.profile_words());
        challenger.observe_slice(&statement.words());
        challenger
    }
}

/// Opaque layer-1 proof. It contains no witness coordinate.
#[derive(Clone, Serialize, Deserialize)]
pub struct Proof {
    root: pcs::Commitment,
    histogram: Vec<u32>,
    gkr: GkrProof,
    quotient: Vec<Ext>,
    linear: Vec<Vec<Ext>>,
    opening: pcs::Opening,
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
    transcript: Poseidon2Transcript,
    claim: &Claim,
    witness: &Mat<F>,
) -> Result<Proof, Error> {
    let shape = &relation.shape;
    let statement = Statement::new(shape, claim)?;
    let z = witness_table(shape, witness)?;
    let histogram = norm::histogram(&z, shape.norm_bound)?;

    let mut challenger = relation.start(transcript, &statement);
    let (root, data) = relation.pcs.commit(z.clone(), &mut challenger);
    challenger.observe_slice(&norm::words(&histogram));
    let beta: Ext = challenger.sample_algebra_element();
    let (gkr, leaf_point) = gkr::prove(&z, beta, &mut challenger);

    let mixing = Mixing::new(challenger.sample_algebra_element(), shape);
    let weights = ring::block_weights(relation.matrices, shape, &statement, &mixing)?;
    let (quotient, _) = ring::divide(&weights, &z, LANES, &ring::targets(&statement, &mixing));
    challenger.observe_algebra_slice(&quotient);
    let zeta: Ext = challenger.sample_algebra_element();
    let omega = padded(ring::evaluate(&weights, zeta), shape.block_variables);
    drop(weights);

    let (linear, linear_point) = prove_linear(omega, zeta, &z, &mut challenger);
    let opening = relation
        .pcs
        .open(data, &[leaf_point, linear_point], &mut challenger);
    Ok(Proof {
        root,
        histogram,
        gkr,
        quotient: quotient.to_vec(),
        linear,
        opening,
    })
}

/// Verify `proof` for `claim`, continuing `transcript`.
pub fn verify(
    relation: &Relation<'_>,
    transcript: Poseidon2Transcript,
    claim: &Claim,
    proof: &Proof,
) -> Result<(), Error> {
    let shape = &relation.shape;
    let statement = Statement::new(shape, claim)?;
    let quotient: [Ext; D - 1] = proof
        .quotient
        .clone()
        .try_into()
        .map_err(|_| Error::Rejected("quotient length"))?;
    if proof.linear.len() != shape.cube_variables() || proof.histogram.len() != 2 * shape.norm_bound as usize + 1 {
        return Err(Error::Rejected("proof shape"));
    }

    let mut challenger = relation.start(transcript, &statement);
    relation.pcs.observe(&proof.root, &mut challenger);
    challenger.observe_slice(&norm::words(&proof.histogram));
    let beta: Ext = challenger.sample_algebra_element();
    let total = norm::table_sum(&proof.histogram, shape.norm_bound, beta)?;
    let leaf = gkr::verify(&proof.gkr, shape.cube_variables(), total, &mut challenger)?;

    let mixing = Mixing::new(challenger.sample_algebra_element(), shape);
    challenger.observe_algebra_slice(&quotient);
    let zeta: Ext = challenger.sample_algebra_element();
    let weights = ring::block_weights(relation.matrices, shape, &statement, &mixing)?;
    let value = ring::lifted_target(&ring::targets(&statement, &mixing), &quotient, zeta);
    let (point, last) = sumcheck::verify(&proof.linear, 2, value, &mut challenger)?;
    // The linear sum-check binds the block bits first, then the lane bits.
    let (block_point, lane_point) = point.split_at(shape.block_variables);
    let linear_point = [lane_point, block_point].concat();

    let [z_leaf, z_linear] = relation.pcs.verify(
        &proof.root,
        &proof.opening,
        &[leaf.point, linear_point],
        &mut challenger,
    )?;
    if z_leaf != beta - leaf.q {
        return Err(Error::Rejected("GKR leaf against the commitment"));
    }
    let omega = padded(ring::evaluate(&weights, zeta), shape.block_variables);
    let weight = sumcheck::evaluate(&omega, block_point) * sumcheck::evaluate(&lane_weights(zeta), lane_point);
    if last != weight * z_linear {
        return Err(Error::Rejected("linear claim against the commitment"));
    }
    Ok(())
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
/// Returns the rounds and the point in cube order (lane bits first).
fn prove_linear(omega: Vec<Ext>, zeta: Ext, z: &[Gl], challenger: &mut Challenger) -> (Vec<Vec<Ext>>, Vec<Ext>) {
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
    let (lane_rounds, lane_point, _) = sumcheck::prove_product(scaled, lane_values, challenger);
    rounds.extend(lane_rounds);
    (rounds, [lane_point, block_point].concat())
}

/// Every layer-1 draw made after the commitment, as a degree over |Ext|.
/// Plonky3 charges each over the commitment's candidate set.
fn outer_terms(shape: &Shape) -> Vec<SecurityTerm> {
    let field_bits = (<Ext as Field>::bits() - 1) as f64;
    let cube = shape.cube_variables();
    let term = |label, degree: f64| SecurityTerm::new(label, ErrorBits::from_log2(field_bits - degree.log2()));
    vec![
        term(
            "logUp at beta",
            (1u64 << cube) as f64 + f64::from(2 * shape.norm_bound + 1),
        ),
        term("GKR layers", (1..=cube).map(|k| 3.0 * k as f64 + 3.0).sum()),
        term("ring-row batching", (Mixing::rows(shape) - 1) as f64),
        term("quotient at zeta", (2 * D - 2) as f64),
        term("linear sum-check", 2.0 * cube as f64),
    ]
}
