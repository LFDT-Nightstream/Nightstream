//! The layer-1 prover: `prove` and the tables that only the prover builds.
//!
//! Owns: the prover's side of the layer-1 Fiat-Shamir schedule, which
//! `verifier.rs` replays step for step, and the order in which the prover
//! builds and frees its tables. Does not own: the relation, its plans or
//! its points (`relation.rs`).

use neo_ccs::Mat;
use neo_math::{D, F};
use neo_transcript::Poseidon2Transcript;
use neo_verified::degrees::{FOLD, RUN_ORACLES};
use p3_challenger_v08::{CanObserve, FieldChallenger};
use p3_field_v08::PrimeCharacteristicRing;
use rayon::prelude::*;

use crate::circuit::Native;
use crate::device::Accelerator;
use crate::field::{coordinates, eq_table, gl, Ext, Gl};
use crate::gkr::{self, Leaves, Tree};
use crate::hash::{self, Challenger};
use crate::matrix::{EarlyChallenges, EarlyValues, Tables};
use crate::relation::{p1_points, setup_point, Relation, Shape};
use crate::ring::{self, Mixing};
use crate::setup;
use crate::{mle, norm, sumcheck, Claim, ClaimWords, Error, Proof, Setup, LANES, LANE_VARIABLES};

/// Prove `claim` for `witness` (54 × blocks), continuing `transcript`. The
/// Merkle leaves hash on `accelerator`.
pub fn prove(
    relation: &Relation<'_>,
    setup: &Setup,
    accelerator: &Accelerator,
    transcript: Poseidon2Transcript,
    claim: &Claim,
    witness: &Mat<F>,
) -> Result<Proof, Error> {
    let key = relation.key;
    if setup.key != *key {
        return Err(Error::Setup("the setup belongs to another key"));
    }
    let (shape, structure, tables) = (&relation.shape, &key.structure, &setup.tables);
    let statement = ClaimWords::new(shape, claim)?;
    // The P0 commitment takes the only copies of its tables; layer 1 rebuilds
    // `z` and the early tables after it.
    let p0_values = p0_tables(shape, tables, claim, witness)?;
    let histogram = norm::histogram(&p0_values[0], shape.norm_bound)?;

    let mut challenger = relation.start(hash::seed(transcript), &statement);
    let (p0, mut p0_data) = relation
        .p0
        .commit(p0_values, &mut challenger, accelerator)?;
    // The commitment's layout holds the P0 tables until the opening. Layer 1
    // frees it now and rebuilds it for the opening, so the tables are not
    // held twice in between.
    drop(p0_data.layout);
    let z = witness_table(shape, witness)?;
    let early = tables.early(&claim.r);
    challenger.observe_slice(&norm::words(&histogram));
    let beta: Ext = challenger.sample_algebra_element();
    let challenges = early_challenges(&mut challenger);
    let mut trees = vec![Leaves::new(z.len(), Vec::new(), gkr::part(|x| beta - z[x]))];
    trees.extend(tables.early_trees(&early, &claim.r, challenges));
    let (early_proof, claims) = gkr::prove(&trees, &mut challenger, accelerator)?;
    drop(trees);
    let early_values = tables.early_values(&early, &claims[1..]);
    observe_early(&early_values, &mut challenger);

    let mixing = Mixing::new(&mut Native, challenger.sample_algebra_element(), shape);
    let ubar = tables.slot_weights(&early, &mixing.eval_a());
    let key_weights = ring::key_weights(shape, &mixing, accelerator);
    let eval_a = tables.column_weights(&ubar, shape.blocks);
    let weights = ring::block_weights(shape, &claim.r, &mixing, &eval_a, &key_weights);
    drop(eval_a);
    let (quotient, _) = ring::divide(&weights, &z, LANES, &ring::targets(&mut Native, &statement, &mixing));
    challenger.observe_algebra_slice(&quotient);
    let zeta: Ext = challenger.sample_algebra_element();
    let omega = padded(ring::evaluate(&weights, zeta), shape.block_variables);
    drop(weights);
    let (linear, linear_point, z_at_point) = prove_linear(omega, zeta, &z, &mut challenger);
    drop(z);
    let block_point = &linear_point[LANE_VARIABLES..];

    let (_, lane_point) = mle::scaled_eq_point(&mut Native, zeta, LANE_VARIABLES)
        .map_err(|_| Error::Witness("ζ is a pole of the lane weights"))?;
    let variables = key.setup_variables();
    let key_point = setup_point(&mut Native, &[lane_point.as_slice(), block_point].concat(), variables);
    let key_table = key_table(&key_weights, variables);
    drop(key_weights);
    let lanes = structure.lane_tables(&mut Native, &ring::tau(&mut Native, zeta));
    let omega_a = tables.block_weights(&ubar, &lanes, shape.block_variables);
    let finals = [
        sumcheck::evaluate(&key_table, &key_point),
        sumcheck::evaluate(&omega_a, block_point),
        z_at_point,
    ];
    challenger.observe_algebra_slice(&finals);

    let mu: Ext = challenger.sample_algebra_element();
    let run_point = setup_point(&mut Native, &claims[1].point, variables);
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
        .commit(vec![fold_columns, coordinates(&omega_a)], &mut challenger, accelerator)?;
    let indices = relation.query_indices(&mut challenger);
    let leaves = indices
        .iter()
        .map(|&index| setup.store.read(index))
        .collect::<Result<Vec<_>, _>>()?;

    let beta: Ext = challenger.sample_algebra_element();
    let late_trees = tables.late_trees(&ubar, &lanes, &omega_a, beta);
    let (late_proof, late) = gkr::prove(
        &late_trees.iter().map(Tree::leaves).collect::<Vec<_>>(),
        &mut challenger,
        accelerator,
    )?;
    drop(late_trees);
    let slots = |point: &[Ext]| point[..structure.slot_variables].to_vec();
    let late_values = tables.slot_values(&early, &eq_table(&slots(&late[0].point)));
    // The openings need none of the per-proof tables.
    drop((early, ubar, lanes, omega_a));
    observe_pairs(&late_values, &mut challenger);

    let query_points = indices
        .iter()
        .map(|&index| {
            let bits: Vec<Gl> = (0..variables + 1 - FOLD)
                .map(|t| Gl::from_usize((index >> t) & 1))
                .collect();
            setup::query_point(&mut Native, variables, &bits)
        })
        .collect();
    let p1_points = p1_points(&key_point, &run_point, query_points, block_point, &late[1].point);
    let p1_opening = relation
        .p1
        .open(p1_data, &p1_points, &mut challenger, accelerator)?;
    let p0_points = vec![
        claims[0].point.clone(),
        linear_point.clone(),
        claims[1].point.clone(),
        slots(&claims[4].point),
        slots(&claims[1].point),
        slots(&late[0].point),
        claims[2].point.clone(),
    ];
    p0_data.layout = relation
        .p0
        .layout(p0_tables(shape, tables, claim, witness)?);
    let p0_opening = relation
        .p0
        .open(p0_data, &p0_points, &mut challenger, accelerator)?;
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

fn early_challenges(challenger: &mut Challenger) -> EarlyChallenges<Ext> {
    EarlyChallenges {
        lookup: [challenger.sample_algebra_element(), challenger.sample_algebra_element()],
        scatter: [challenger.sample_algebra_element(), challenger.sample_algebra_element()],
    }
}

fn observe_early(values: &EarlyValues<Ext>, challenger: &mut Challenger) {
    challenger.observe_algebra_slice(&values.runs);
    observe_pairs(&values.pairs, challenger);
    challenger.observe_algebra_element(values.block);
}

fn observe_pairs(pairs: &[[Ext; 2]], challenger: &mut Challenger) {
    for pair in pairs {
        challenger.observe_algebra_slice(pair);
    }
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

/// `z` on the cube: `z[64·block + lane] = Z[lane, block]`, zero elsewhere.
/// The P0 tables: `z`, the run columns `e`, the slot columns, and the row
/// counts.
fn p0_tables(shape: &Shape, tables: &Tables, claim: &Claim, witness: &Mat<F>) -> Result<Vec<Vec<Gl>>, Error> {
    let early = tables.early(&claim.r);
    let slot_columns: Vec<Gl> = early
        .u
        .iter()
        .flat_map(|[re, im]| re.iter().chain(im))
        .copied()
        .chain(tables.blocks())
        .collect();
    let e = [early.e[0].as_slice(), early.e[1].as_slice()].concat();
    Ok(vec![witness_table(shape, witness)?, e, slot_columns, early.mult])
}

pub(crate) fn witness_table(shape: &Shape, witness: &Mat<F>) -> Result<Vec<Gl>, Error> {
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
