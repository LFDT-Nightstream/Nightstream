//! The layer-1 relation: the shape of one CE(B, L) claim under a key, the
//! two WHIR table plans, the setup query count and the security terms
//! charged to each commitment, and the points that prover and verifier both
//! open, so the two follow one schedule.
//!
//! Does not own: the key (`lib.rs`), the prover's flow (`prover.rs`) or the
//! verifier's (`verifier.rs`).

use neo_ajtai::nightstream_fprime_setup::PRODUCTION_VERIFIER_ROWS;
use neo_math::D;
use neo_verified::degrees::{self, FOLD};
use p3_challenger_v08::{CanObserve, CanSampleBits};
use p3_field_v08::{Field, PrimeCharacteristicRing, PrimeField64, TwoAdicField};
use p3_security_v08::{ErrorBits, SecurityTerm};

use crate::circuit::{algebra, Backend};
use crate::field::{Ext, Gl};
use crate::hash::{self, Challenger};
use crate::matrix::Structure;
use crate::pcs::{self, Pcs, TablePlan};
use crate::{block_variables, ClaimWords, Error, Key, LANE_VARIABLES};

pub(crate) fn kappa() -> usize {
    PRODUCTION_VERIFIER_ROWS as usize
}

/// One fixed CE(B, L) relation of a key: the production Ajtai key prefix, the
/// CCS matrices over the full carrier, the public width, the point length and
/// the norm bound. The WHIR configurations and the setup query count are
/// derived here, so prover and verifier cannot disagree on them.
pub struct Relation<'k> {
    pub(crate) key: &'k Key,
    pub(crate) shape: Shape,
    pub(crate) p0: Pcs,
    pub(crate) p1: Pcs,
    pub(crate) queries: usize,
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
    pub(crate) fn cube_variables(&self) -> usize {
        self.block_variables + LANE_VARIABLES
    }

    pub(crate) fn words(&self) -> Vec<Gl> {
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
        // `setup::check` draws a root of unity of order 2^(setup variables + 1).
        if key.setup_variables() + 1 > Gl::TWO_ADICITY {
            return Err(Error::Setup("setup size"));
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
        let probe = Pcs::new(pcs::LAYER1, p1_plans(key, 0), target, &|_| Vec::new())?;
        let queries = query_count(target, probe.log2_candidates());
        let p1 = Pcs::new(pcs::LAYER1, p1_plans(key, queries), target, &|_| {
            vec![query_term(queries)]
        })?;
        if p1.log2_candidates() != probe.log2_candidates() {
            return Err(Error::Setup("P1 candidate count"));
        }
        let terms = outer_terms(&shape, &key.structure, p1.log2_candidates());
        let p0 = Pcs::new(pcs::LAYER1, p0_plans(&shape, &key.structure), target, &|_| {
            terms.clone()
        })?;
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

    /// `words` as canonical integers: what a transcript of a proof about
    /// this relation binds.
    pub fn key_words(&self) -> Vec<u64> {
        self.words()
            .iter()
            .map(|word| word.as_canonical_u64())
            .collect()
    }

    /// The relation as transcript words: the shape, the key, both WHIR
    /// profiles and the setup query count.
    pub(crate) fn words(&self) -> Vec<Gl> {
        let mut words = self.shape.words();
        words.extend(self.key.words());
        words.extend(self.p0.profile_words());
        words.extend(self.p1.profile_words());
        words.push(Gl::from_usize(self.queries));
        words
    }

    /// Bind the relation and the statement after the layer-1 seed, before
    /// the first layer-1 challenge.
    pub(crate) fn start(&self, seed: [Gl; 4], statement: &ClaimWords<Gl>) -> Challenger {
        let mut challenger = hash::challenger(seed);
        challenger.observe_slice(&self.words());
        challenger.observe_slice(&statement.words());
        challenger
    }

    /// The setup leaves to query, drawn after the P1 root.
    pub(crate) fn query_indices(&self, challenger: &mut Challenger) -> Vec<usize> {
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
pub(crate) fn p1_plans(key: &Key, queries: usize) -> Vec<TablePlan> {
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

/// A point on fewer variables, extended by zeros: an oracle embedded in the
/// low variables of a setup oracle has the same value there.
pub(crate) fn setup_point<B: Backend>(b: &mut B, point: &[B::E], variables: usize) -> Vec<B::E> {
    let mut point = point.to_vec();
    point.resize(variables, algebra::ext_zero(b));
    point
}

/// P1 points in plan order: the two fold claims, the setup query points,
/// the linear and the late block points.
pub(crate) fn p1_points<E: Clone>(
    key_point: &[E],
    run_point: &[E],
    query_points: Vec<Vec<E>>,
    block_point: &[E],
    late_block_point: &[E],
) -> Vec<Vec<E>> {
    let mut points = vec![key_point[FOLD..].to_vec(), run_point[FOLD..].to_vec()];
    points.extend(query_points);
    points.extend([block_point.to_vec(), late_block_point.to_vec()]);
    points
}

/// Every layer-1 draw made after P0, as a degree over |Ext|
/// (`neo_verified::degrees`). Plonky3 charges each over P0's candidates;
/// `extra_log2` also charges it over P1's, since a prover picks both
/// candidates only at the openings.
fn outer_terms(shape: &Shape, structure: &Structure, extra_log2: f64) -> Vec<SecurityTerm> {
    let field_bits = (<Ext as Field>::bits() - 1) as f64;
    let term = |label, degree: usize| {
        SecurityTerm::new(
            label,
            ErrorBits::from_log2(field_bits - (degree as f64).log2() - extra_log2),
        )
    };
    let relation = degrees::Relation {
        d: D,
        lane_variables: LANE_VARIABLES,
        block_variables: shape.block_variables,
        kappa: shape.kappa,
        matrices: shape.matrices,
        public_blocks: shape.public_blocks,
        norm_bound: shape.norm_bound,
        run_variables: structure.run_variables,
        row_variables: structure.row_variables,
        slot_variables: structure.slot_variables,
        matrix_variables: structure.matrix_variables(),
    };
    let [norm, lookup, scatter, early_gkr, slot_block] = degrees::early_terms(&relation);
    let [ring_rows, quotient, linear] = degrees::linear_terms(&relation);
    let [run_group, fold_partials] = degrees::FIXED;
    let [block_scatter, late_gkr] = degrees::late_terms(&relation);
    vec![
        term("norm logUp", norm),
        term("row lookup", lookup),
        term("slot scatter", scatter),
        term("early GKR", early_gkr),
        term("slot-block test", slot_block),
        term("ring-row batching", ring_rows),
        term("quotient at zeta", quotient),
        term("linear sum-check", linear),
        term("run group", run_group),
        term("honest-fold partials", fold_partials),
        term("block scatter", block_scatter),
        term("late GKR", late_gkr),
    ]
}
