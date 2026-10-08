//! The layer-1 verifier, written once over `Backend`.
//!
//! Owns: the view of a layer-1 proof as backend words and the verifier's
//! schedule from the seed to the last P0 opening. `Native` reads it as the
//! plain verifier (`crate::verify`); `Recorder` reads it as rows. Does not own
//! the seed: the fold transcript (or a circuit replay of it) supplies it.

use neo_math::D;
use p3_field_v08::{BasedVectorSpace, PrimeCharacteristicRing};

use crate::circuit::algebra;
use crate::circuit::hash::Duplex;
use crate::circuit::Backend;
use crate::field::{Ext, Gl};
use crate::gkr::{self, GkrView, TreeShape};
use crate::matrix::{EarlyChallenges, EarlyValues};
use crate::ring::{self, Mixing};
use crate::setup::{self, LeafView, FOLD};
use crate::whir::{self, OpeningView};
use crate::{kappa, mle, p1_points, setup_point, sumcheck, ClaimWords, Error, Proof, Relation};
use crate::{LANE_VARIABLES, RUN_ORACLES};

/// A layer-1 proof as backend words, or zeros of its shape.
pub(crate) struct ProofView<'p, B: Backend> {
    p0: [B::F; 4],
    histogram: Vec<B::F>,
    early: GkrView<B>,
    early_values: EarlyValues<B::E>,
    quotient: Vec<B::E>,
    linear: Vec<Vec<B::E>>,
    finals: [B::E; 3],
    partials: [[B::E; 1 << FOLD]; 2],
    p1: [B::F; 4],
    leaves: Vec<LeafView<B>>,
    late: GkrView<B>,
    late_values: Vec<[B::E; 2]>,
    p1_opening: OpeningView<'p, B>,
    p0_opening: OpeningView<'p, B>,
}

impl<'p, B: Backend> ProofView<'p, B> {
    /// The words of `proof`, or zeros of the relation's shape when `None`.
    pub(crate) fn read(b: &mut B, relation: &Relation<'_>, proof: Option<&'p Proof>) -> Result<Self, Error> {
        let (shape, structure) = (&relation.shape, &relation.key.structure);
        let matrices = structure.matrices;
        if proof.is_some_and(|proof| {
            proof.histogram.len() != 2 * shape.norm_bound as usize + 1
                || proof.early_values.pairs.len() != matrices
                || proof.quotient.len() != D - 1
                || proof.linear.len() != shape.cube_variables()
                || proof.linear.iter().any(|message| message.len() != 2)
                || proof.leaves.len() != relation.queries
                || proof.late_values.len() != matrices
        }) {
            return Err(Error::Rejected("proof shape"));
        }
        let ext = |b: &mut B, value: Option<Ext>| algebra::private_ext(b, value.unwrap_or(Ext::ZERO));
        let exts = |b: &mut B, values: Option<&[Ext]>, count: usize| -> Vec<B::E> {
            (0..count)
                .map(|i| ext(b, values.map(|values| values[i])))
                .collect()
        };
        let pairs = |b: &mut B, values: Option<&[[Ext; 2]]>| -> Vec<[B::E; 2]> {
            (0..matrices)
                .map(|j| std::array::from_fn(|side| ext(b, values.map(|values| values[j][side]))))
                .collect()
        };
        let root = |b: &mut B, commitment: Option<&crate::pcs::Commitment>| -> Result<[B::F; 4], Error> {
            let root = match commitment.map(|commitment| commitment.roots()) {
                Some([root]) => *root,
                Some(_) => return Err(Error::Rejected("commitment cap")),
                None => [Gl::ZERO; 4],
            };
            Ok(root.map(|word| algebra::private(b, word)))
        };
        let p0 = root(b, proof.map(|proof| &proof.p0))?;
        let histogram = (0..2 * shape.norm_bound as usize + 1)
            .map(|slot| b.private(proof.map_or(0, |proof| u64::from(proof.histogram[slot]))))
            .collect();
        let early = GkrView::read(b, proof.map(|proof| &proof.early), &early_shapes(relation))?;
        let early_values = EarlyValues {
            runs: std::array::from_fn(|i| ext(b, proof.map(|proof| proof.early_values.runs[i]))),
            pairs: pairs(b, proof.map(|proof| proof.early_values.pairs.as_slice())),
            block: ext(b, proof.map(|proof| proof.early_values.block)),
        };
        let quotient = exts(b, proof.map(|proof| proof.quotient.as_slice()), D - 1);
        let linear = (0..shape.cube_variables())
            .map(|round| exts(b, proof.map(|proof| proof.linear[round].as_slice()), 2))
            .collect();
        let finals = std::array::from_fn(|i| ext(b, proof.map(|proof| proof.finals[i])));
        let partials =
            std::array::from_fn(|group| std::array::from_fn(|u| ext(b, proof.map(|proof| proof.partials[group][u]))));
        let p1 = root(b, proof.map(|proof| &proof.p1))?;
        let variables = relation.key.setup_variables();
        let leaves = (0..relation.queries)
            .map(|query| {
                let leaf = proof.map(|proof| &proof.leaves[query]);
                LeafView::read(b, leaf, kappa() + RUN_ORACLES, variables)
            })
            .collect::<Result<_, _>>()?;
        let late_shapes = structure.late_shapes(shape.block_variables);
        let late = GkrView::read(b, proof.map(|proof| &proof.late), &late_shapes)?;
        let late_values = pairs(b, proof.map(|proof| proof.late_values.as_slice()));
        let p1_opening = OpeningView::read(b, &relation.p1, proof.map(|proof| &proof.p1_opening))?;
        let p0_opening = OpeningView::read(b, &relation.p0, proof.map(|proof| &proof.p0_opening))?;
        Ok(Self {
            p0,
            histogram,
            early,
            early_values,
            quotient,
            linear,
            finals,
            partials,
            p1,
            leaves,
            late,
            late_values,
            p1_opening,
            p0_opening,
        })
    }
}

/// The norm tree over the cube, then the matrix early trees.
fn early_shapes(relation: &Relation<'_>) -> Vec<TreeShape> {
    let mut shapes = vec![TreeShape {
        depth: relation.shape.cube_variables(),
        factors: 0,
    }];
    shapes.extend(relation.key.structure.early_shapes());
    shapes
}

/// Verify `proof` for `statement` after the layer-1 `seed`.
pub(crate) fn verify<B: Backend>(
    b: &mut B,
    relation: &Relation<'_>,
    seed: [B::F; 4],
    statement: &ClaimWords<B::F>,
    proof: &ProofView<'_, B>,
) -> Result<(), Error> {
    let (key, shape) = (relation.key, &relation.shape);
    let structure = &key.structure;
    let mut duplex = Duplex::new(b);
    duplex.observe_slice(b, &seed);
    for word in relation.words() {
        let word = algebra::constant(b, word);
        duplex.observe(b, word);
    }
    duplex.observe_slice(b, &statement.words());

    whir::observe_root(b, &mut duplex, proof.p0);
    duplex.observe_slice(b, &proof.histogram);
    let beta = duplex.sample_ext(b);
    let challenges = EarlyChallenges {
        lookup: [duplex.sample_ext(b), duplex.sample_ext(b)],
        scatter: [duplex.sample_ext(b), duplex.sample_ext(b)],
    };
    let total = crate::norm::table_sum(b, &proof.histogram, shape.norm_bound, beta)?;
    let claims = gkr::verify(b, &proof.early, &early_shapes(relation), &mut duplex)?;
    let [p, q] = claims[0].root;
    b.ext_inverse(q, "GKR root against the histogram")?;
    let expected = b.ext_mul(q, total);
    b.assert_ext_equal(p, expected, "GKR root against the histogram")?;
    let z_leaf = b.ext_sub(beta, claims[0].values[0]);
    let early = structure.check_early(b, &statement.point, challenges, &claims[1..], &proof.early_values)?;
    observe_early(b, &mut duplex, &proof.early_values);

    let lambda = duplex.sample_ext(b);
    let mixing = Mixing::new(b, lambda, shape);
    for &value in &proof.quotient {
        duplex.observe_ext(b, value);
    }
    let zeta = duplex.sample_ext(b);
    let targets = ring::targets(b, statement, &mixing);
    let value = ring::lifted_target(b, &targets, &proof.quotient, zeta);
    let (point, last) = sumcheck::replay(b, &mut duplex, &proof.linear, value);
    // The linear sum-check binds the block bits first, then the lane bits.
    let (block_point, lane_part) = point.split_at(shape.block_variables);
    let linear_point = [lane_part, block_point].concat();
    let [key_value, omega_value, z_value] = proof.finals;
    for value in proof.finals {
        duplex.observe_ext(b, value);
    }

    // Ω~(s_b) = κ_ζ·A_λ~(x_ζ, s_b) + Eval_K + Ω^A~(s_b) + public.
    let (scale, lane_point) = mle::scaled_eq_point(b, zeta, LANE_VARIABLES)?;
    let tau = ring::tau(b, zeta);
    let eval_k = mle::eval_k(b, &statement.point, block_point, &tau, shape.blocks);
    let [k_re, k_im] = mixing.eval_k();
    let mut omega = b.ext_mul(scale, key_value);
    for (weight, value) in [(k_re, eval_k.re), (k_im, eval_k.im)] {
        let term = b.ext_mul(weight, value);
        omega = b.ext_add(omega, term);
    }
    omega = b.ext_add(omega, omega_value);
    for block in 0..shape.public_blocks {
        let eq = algebra::eq_at(b, block_point, block);
        let term = b.ext_mul(eq, mixing.public(block));
        omega = b.ext_add(omega, term);
    }
    let lanes = lane_value(b, zeta, lane_part);
    let weighted = b.ext_mul(omega, lanes);
    let expected = b.ext_mul(weighted, z_value);
    b.assert_ext_equal(last, expected, "linear claim")?;

    let mu = duplex.sample_ext(b);
    for values in &proof.partials {
        for &value in values {
            duplex.observe_ext(b, value);
        }
    }
    let alpha: [B::E; FOLD] = std::array::from_fn(|_| duplex.sample_ext(b));
    let variables = key.setup_variables();
    let key_point = setup_point(b, &[lane_point.as_slice(), block_point].concat(), variables);
    let run_point = setup_point(b, &early.run_point, variables);
    let [row, slot, coefficient] = early.setup;
    let mut embedded = proof.early_values.block;
    for &x in &early.run_point[structure.slot_variables..] {
        let low = algebra::one_minus(b, x);
        embedded = b.ext_mul(embedded, low);
    }
    let mut run_value = embedded;
    for value in [coefficient, slot, row] {
        let scaled = b.ext_mul(mu, run_value);
        run_value = b.ext_add(value, scaled);
    }
    let key_partial = combine(b, &proof.partials[0], &key_point[..FOLD]);
    b.assert_ext_equal(key_partial, key_value, "honest-fold partials")?;
    let run_partial = combine(b, &proof.partials[1], &run_point[..FOLD]);
    b.assert_ext_equal(run_partial, run_value, "honest-fold partials")?;

    whir::observe_root(b, &mut duplex, proof.p1);
    let index_bits = (0..relation.queries)
        .map(|_| duplex.sample_bits(b, variables + 1 - FOLD))
        .collect::<Result<Vec<_>, _>>()?;
    let commitment_rows = (0..kappa()).map(|row| mixing.commitment(row)).collect();
    let mut run_weights = vec![algebra::ext_one(b)];
    for _ in 1..RUN_ORACLES {
        let last = *run_weights.last().expect("one weight");
        run_weights.push(b.ext_mul(last, mu));
    }
    let groups = [commitment_rows, run_weights];
    let root = key.root().map(|word| algebra::constant(b, word));
    let mut queried = Vec::with_capacity(2 * relation.queries);
    for (bits, leaf) in index_bits.iter().zip(&proof.leaves) {
        queried.extend(setup::check(b, root, variables, bits, leaf, &groups, &alpha)?);
    }

    let beta = duplex.sample_ext(b);
    let late_claims = gkr::verify(
        b,
        &proof.late,
        &structure.late_shapes(shape.block_variables),
        &mut duplex,
    )?;
    let lanes = structure.lane_tables(b, &tau);
    let late = structure.check_late(b, beta, &mixing.eval_a(), &lanes, &late_claims, &proof.late_values)?;
    observe_pairs(b, &mut duplex, &proof.late_values);

    let query_points = index_bits
        .iter()
        .map(|bits| setup::query_point(b, variables, bits))
        .collect();
    let points = p1_points(&key_point, &run_point, query_points, block_point, &late.block_point);
    let opened = whir::verify(b, &relation.p1, proof.p1, &proof.p1_opening, &points, &mut duplex)?;
    let mut expected = vec![
        combine(b, &proof.partials[0], &alpha),
        combine(b, &proof.partials[1], &alpha),
    ];
    expected.extend(queried);
    expected.extend([omega_value, late.omega]);
    let values: Vec<B::E> = opened
        .iter()
        .flat_map(|batch| batch.chunks(3))
        .map(|coordinates| from_coordinates(b, coordinates))
        .collect();
    assert_eq!(values.len(), expected.len());
    for (value, expected) in values.into_iter().zip(expected) {
        b.assert_ext_equal(value, expected, "P1 openings")?;
    }

    let slots = |point: &[B::E]| point[..structure.slot_variables].to_vec();
    let points = vec![
        claims[0].point.clone(),
        linear_point,
        early.run_point.clone(),
        early.pair_point.clone(),
        slots(&early.run_point),
        late.pair_point.clone(),
        early.row_point.clone(),
    ];
    let opened = whir::verify(b, &relation.p0, proof.p0, &proof.p0_opening, &points, &mut duplex)?;
    let flat = |pairs: &[[B::E; 2]]| pairs.iter().flatten().copied().collect::<Vec<B::E>>();
    let [e_re, e_im, _] = proof.early_values.runs;
    let mut late_slots = flat(&proof.late_values);
    late_slots.push(late.block);
    let expected = [
        vec![z_leaf],
        vec![z_value],
        vec![e_re, e_im],
        flat(&proof.early_values.pairs),
        vec![proof.early_values.block],
        late_slots,
        vec![early.mult],
    ];
    assert_eq!(
        opened.iter().map(Vec::len).collect::<Vec<_>>(),
        expected.iter().map(Vec::len).collect::<Vec<_>>()
    );
    for (value, expected) in opened.iter().flatten().zip(expected.iter().flatten()) {
        b.assert_ext_equal(*value, *expected, "P0 openings")?;
    }
    Ok(())
}

fn observe_early<B: Backend>(b: &mut B, duplex: &mut Duplex<B>, values: &EarlyValues<B::E>) {
    for value in values.runs {
        duplex.observe_ext(b, value);
    }
    observe_pairs(b, duplex, &values.pairs);
    duplex.observe_ext(b, values.block);
}

fn observe_pairs<B: Backend>(b: &mut B, duplex: &mut Duplex<B>, pairs: &[[B::E; 2]]) {
    for pair in pairs {
        duplex.observe_ext(b, pair[0]);
        duplex.observe_ext(b, pair[1]);
    }
}

/// `Σ_u eq(point, u)·partials[u]`: a group's value from its partials.
fn combine<B: Backend>(b: &mut B, partials: &[B::E; 1 << FOLD], point: &[B::E]) -> B::E {
    let eq = algebra::eq_table(b, point);
    algebra::dot(b, &eq, partials)
}

/// `Λ~(x) = Σ_{l<D} ζ^l·eq(x, l)` at the lane part of the linear point.
fn lane_value<B: Backend>(b: &mut B, zeta: B::E, point: &[B::E]) -> B::E {
    let eq = algebra::eq_table(b, point);
    let mut total = algebra::ext_zero(b);
    let mut power = algebra::ext_one(b);
    for &weight in &eq[..D] {
        let term = b.ext_mul(weight, power);
        total = b.ext_add(total, term);
        power = b.ext_mul(power, zeta);
    }
    total
}

/// The `Ext` value whose coordinate columns opened to `values`.
fn from_coordinates<B: Backend>(b: &mut B, values: &[B::E]) -> B::E {
    let mut total = algebra::ext_zero(b);
    for (c, &value) in values.iter().enumerate() {
        let basis = <Ext as BasedVectorSpace<Gl>>::ith_basis_element(c).expect("three coordinates");
        let basis = algebra::ext_constant(b, basis);
        let term = b.ext_mul(basis, value);
        total = b.ext_add(total, term);
    }
    total
}
