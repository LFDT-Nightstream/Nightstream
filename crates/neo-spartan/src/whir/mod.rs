//! Our verifier for one p3-whir 0.8 prescribed-point opening of a `Pcs`,
//! written once over `Backend`.
//!
//! Owns: the replay of Plonky3's `verify_at` for our profile (stacked layout
//! of `SuffixProver`, Goldilocks with the cubic extension, no successor
//! openings, no grinding, non-stratified queries) and the view of an opening
//! as backend words. Differences from Plonky3, each stricter or equal: every
//! query is checked with a full Merkle path, and a sample equal to p − 1 is
//! rejected where Plonky3 draws again (completeness gap 2^-64 per draw).
//!
//! Points are in Plonky3 order (most significant variable first) inside this
//! file; callers pass low-bit-first points, as everywhere else.

pub(crate) mod paths;
pub(crate) mod seeds;

use p3_field_v08::{BasedVectorSpace, PrimeCharacteristicRing, TwoAdicField};
use p3_sumcheck_v08::layout::plan_stacked_layout;
use p3_whir::pcs::proof::QueryOpenings;
use p3_whir::transcript::WhirShape;

use crate::circuit::hash::{hash_leaf, merkle_root, Duplex};
use crate::circuit::Backend;
use crate::field::{Ext, Gl};
use crate::pcs::{Opening, Pcs};
use crate::Error;

/// One opened query set: the row of each draw, as base words (extension
/// rows as coordinates). Paths come from a hint at verification time.
struct Rows<B: Backend> {
    rows: Vec<Vec<B::F>>,
}

struct RoundView<B: Backend> {
    root: [B::F; 4],
    ood: Vec<B::E>,
    queries: Rows<B>,
    sumcheck: Vec<[B::E; 2]>,
}

/// One opening as backend words. Built from a Plonky3 proof, or as zeros of
/// the same shape for a shape run.
pub(crate) struct OpeningView<'p, B: Backend> {
    proof: Option<&'p Opening>,
    initial_ood: Vec<B::E>,
    evals: Vec<Vec<B::E>>,
    initial_sumcheck: Vec<[B::E; 2]>,
    rounds: Vec<RoundView<B>>,
    final_poly: Vec<B::E>,
    final_queries: Rows<B>,
    final_sumcheck: Vec<[B::E; 2]>,
}

fn ext_words(value: Ext) -> [Gl; 3] {
    let slice = <Ext as BasedVectorSpace<Gl>>::as_basis_coefficients_slice(&value);
    [slice[0], slice[1], slice[2]]
}

fn private_ext<B: Backend>(b: &mut B, value: Ext) -> B::E {
    let words = ext_words(value).map(|word| b.private(word));
    b.ext(words)
}

/// Read `count` extension values from `values` (zeros when `None`).
fn exts<B: Backend>(b: &mut B, values: Option<&[Ext]>, count: usize) -> Result<Vec<B::E>, Error> {
    if values.is_some_and(|values| values.len() != count) {
        return Err(Error::Rejected("WHIR proof shape"));
    }
    Ok((0..count)
        .map(|i| private_ext(b, values.map_or(Ext::ZERO, |values| values[i])))
        .collect())
}

fn rounds<B: Backend>(b: &mut B, values: Option<&[[Ext; 2]]>, count: usize) -> Result<Vec<[B::E; 2]>, Error> {
    if values.is_some_and(|values| values.len() != count) {
        return Err(Error::Rejected("WHIR sum-check shape"));
    }
    Ok((0..count)
        .map(|i| {
            let [a, c] = values.map_or([Ext::ZERO; 2], |values| values[i]);
            [private_ext(b, a), private_ext(b, c)]
        })
        .collect())
}

/// The rows of one query set: `draws` rows of `width` values (base, or
/// extension values as three coordinates each).
fn rows<B: Backend>(
    b: &mut B,
    openings: Option<&QueryOpenings<Gl, Ext, crate::pcs::MultiProof>>,
    draws: usize,
    width: usize,
    base: bool,
) -> Result<Rows<B>, Error> {
    let words: Option<Vec<Vec<Gl>>> = match (openings, base) {
        (None, _) => None,
        (Some(QueryOpenings::Base(opening)), true) => Some(opening.rows.clone()),
        (Some(QueryOpenings::Extension(opening)), false) => Some(
            opening
                .rows
                .iter()
                .map(|row| row.iter().flat_map(|&value| ext_words(value)).collect())
                .collect(),
        ),
        _ => return Err(Error::Rejected("WHIR query field")),
    };
    let row_len = if base { width } else { 3 * width };
    if words
        .as_ref()
        .is_some_and(|words| words.len() != draws || words.iter().any(|row| row.len() != row_len))
    {
        return Err(Error::Rejected("WHIR query shape"));
    }
    Ok(Rows {
        rows: (0..draws)
            .map(|d| {
                (0..row_len)
                    .map(|i| b.private(words.as_ref().map_or(Gl::ZERO, |words| words[d][i])))
                    .collect()
            })
            .collect(),
    })
}

fn draws(count: usize, index_bits: usize) -> usize {
    if count == 0 {
        1 << index_bits
    } else {
        count
    }
}

impl<'p, B: Backend> OpeningView<'p, B> {
    /// The words of `proof`, or zeros of its shape when `proof` is `None`.
    pub(crate) fn read(b: &mut B, pcs: &Pcs, proof: Option<&'p Opening>) -> Result<Self, Error> {
        let config = pcs.config();
        let shape = WhirShape::new(config, pcs.protocol().num_openings());
        let whir = proof.map(|proof| &proof.whir);
        if whir.is_some_and(|whir| {
            whir.rounds.len() != shape.rounds.len()
                || whir.final_pow_witness != Gl::ZERO
                || whir
                    .rounds
                    .iter()
                    .any(|round| round.pow_witness != Gl::ZERO)
                || proof.is_some_and(|proof| proof.evals.len() != pcs.protocol().num_openings())
        }) {
            return Err(Error::Rejected("WHIR proof shape"));
        }
        let initial_ood = exts(
            b,
            whir.map(|w| w.initial_ood_answers.as_slice()),
            shape.commitment_ood_samples,
        )?;
        let mut evals = Vec::new();
        for (index, (_, batch)) in pcs.protocol().iter_openings().enumerate() {
            let values = proof.map(|proof| &proof.evals[index]);
            if values.is_some_and(|values| !values.next().is_empty()) {
                return Err(Error::Rejected("WHIR successor openings"));
            }
            evals.push(exts(b, values.map(|v| v.current()), batch.current().len())?);
        }
        let initial_sumcheck = rounds(
            b,
            whir.map(|w| w.initial_sumcheck.polynomial_evaluations.as_slice()),
            shape.initial_sumcheck.rounds,
        )?;
        let mut views = Vec::with_capacity(shape.rounds.len());
        for (r, round) in shape.rounds.iter().enumerate() {
            let proof_round = whir.map(|w| &w.rounds[r]);
            let root = match proof_round.map(|round| round.commitment.as_ref()) {
                Some(None) => return Err(Error::Rejected("WHIR round commitment")),
                Some(Some(cap)) => cap.roots()[0],
                None => [Gl::ZERO; 4],
            };
            let width = 1 << config.round_folding_factor(r);
            views.push(RoundView {
                root: root.map(|word| b.private(word)),
                ood: exts(b, proof_round.map(|p| p.ood_answers.as_slice()), round.ood_samples)?,
                queries: rows(
                    b,
                    proof_round.map(|p| &p.openings),
                    draws(round.query_draws, round.index_bits),
                    width,
                    r == 0,
                )?,
                sumcheck: rounds(
                    b,
                    proof_round.map(|p| p.sumcheck.polynomial_evaluations.as_slice()),
                    round.sumcheck.rounds,
                )?,
            });
        }
        let final_poly = exts(
            b,
            match whir.map(|w| w.final_poly.as_ref()) {
                Some(None) => return Err(Error::Rejected("WHIR final polynomial")),
                Some(Some(poly)) => Some(poly.as_slice()),
                None => None,
            },
            shape.final_poly_len,
        )?;
        let final_width = 1 << config.final_round_config().folding_factor;
        let final_queries = rows(
            b,
            whir.map(|w| &w.final_openings),
            draws(shape.final_query_draws, shape.final_index_bits),
            final_width,
            shape.rounds.is_empty(),
        )?;
        let final_rounds = shape.final_sumcheck.rounds;
        let final_data = match whir.map(|w| w.final_sumcheck.as_ref()) {
            Some(Some(data)) => Some(data.polynomial_evaluations.as_slice()),
            Some(None) if final_rounds > 0 => return Err(Error::Rejected("WHIR final sum-check")),
            _ => None,
        };
        let final_sumcheck = rounds(b, final_data.or(proof.map(|_| &[][..])), final_rounds)?;
        Ok(Self {
            proof,
            initial_ood,
            evals,
            initial_sumcheck,
            rounds: views,
            final_poly,
            final_queries,
            final_sumcheck,
        })
    }
}

fn absorb<B: Backend>(b: &mut B, duplex: &mut Duplex<B>, words: &[Gl]) {
    for &word in words {
        let constant = b.constant(word);
        duplex.observe(b, constant);
    }
}

/// Bind a commitment root as Plonky3's `observe_commitment` does.
pub(crate) fn observe_root<B: Backend>(b: &mut B, duplex: &mut Duplex<B>, root: [B::F; 4]) {
    absorb(b, duplex, &seeds::commitment());
    duplex.observe_slice(b, &root);
}

/// `(x^{2^{n-1}}, ..., x^2, x)`: `Point::expand_from_univariate`.
fn expand<B: Backend>(b: &mut B, x: B::E, n: usize) -> Vec<B::E> {
    let mut point = Vec::with_capacity(n);
    let mut current = x;
    for _ in 0..n {
        point.push(current);
        current = b.ext_mul(current, current);
    }
    point.reverse();
    point
}

/// `Π (2·q·p − p − q + 1)`: `Point::eval_eq`.
fn eval_eq<B: Backend>(b: &mut B, p: &[B::E], q: &[B::E]) -> B::E {
    assert_eq!(p.len(), q.len());
    let one = b.ext_constant(Ext::ONE);
    let mut total = one;
    for (&p, &q) in p.iter().zip(q) {
        let product = b.ext_mul(p, q);
        let twice = b.ext_add(product, product);
        let sum = b.ext_add(p, q);
        let difference = b.ext_sub(twice, sum);
        let term = b.ext_add(difference, one);
        total = b.ext_mul(total, term);
    }
    total
}

/// `Π_j (1 + r_j·(var^{2^j} − 1))` over the coordinates of `point` from the
/// last: `Point::eval_select`.
fn eval_select<B: Backend>(b: &mut B, var: B::F, point: &[B::E]) -> B::E {
    let one = b.ext_constant(Ext::ONE);
    let mut power = var;
    let mut total = one;
    for &r in point.iter().rev() {
        let one_base = b.constant(Gl::ONE);
        let minus = b.sub(power, one_base);
        let scaled = b.ext_scale(r, minus);
        let term = b.ext_add(scaled, one);
        total = b.ext_mul(total, term);
        power = b.mul(power, power);
    }
    total
}

/// The multilinear extension of `values` at `r` (low bit first).
fn mle<B: Backend>(b: &mut B, values: &[B::E], r: &[B::E]) -> B::E {
    assert_eq!(values.len(), 1 << r.len());
    let mut layer = values.to_vec();
    for &x in r {
        layer = layer
            .chunks(2)
            .map(|pair| {
                let difference = b.ext_sub(pair[1], pair[0]);
                let step = b.ext_mul(x, difference);
                b.ext_add(pair[0], step)
            })
            .collect();
    }
    layer[0]
}

/// `Σ_i values_i·var^i` by Horner.
fn horner<B: Backend>(b: &mut B, values: &[B::E], var: B::F) -> B::E {
    let mut total = b.ext_constant(Ext::ZERO);
    for &value in values.iter().rev() {
        let scaled = b.ext_scale(total, var);
        total = b.ext_add(scaled, value);
    }
    total
}

/// A quadratic sum-check: seed, then per round observe `(h(0), h(∞))`, draw
/// `r` and move the claim to `h(r) = h(0)(1−r) + h(1)r + h(∞)r(r−1)`.
fn sumcheck<B: Backend>(b: &mut B, duplex: &mut Duplex<B>, rounds: &[[B::E; 2]], claim: &mut B::E) -> Vec<B::E> {
    if rounds.is_empty() {
        return Vec::new();
    }
    absorb(b, duplex, &seeds::sumcheck(rounds.len()));
    let one = b.ext_constant(Ext::ONE);
    rounds
        .iter()
        .map(|&[at_zero, at_infinity]| {
            duplex.observe_ext(b, at_zero);
            duplex.observe_ext(b, at_infinity);
            let r = duplex.sample_ext(b);
            let at_one = b.ext_sub(*claim, at_zero);
            let one_minus = b.ext_sub(one, r);
            let r_minus = b.ext_sub(r, one);
            let quadratic = b.ext_mul(r, r_minus);
            let a = b.ext_mul(at_zero, one_minus);
            let c = b.ext_mul(at_one, r);
            let d = b.ext_mul(at_infinity, quadratic);
            let sum = b.ext_add(a, c);
            *claim = b.ext_add(sum, d);
            r
        })
        .collect()
}

/// One batch of constraints: a challenge, its first power, and statements of
/// eq points and select variables with their claimed values.
struct Constraint<B: Backend> {
    challenge: B::E,
    initial_power: u64,
    variables: usize,
    eq: Vec<(Vec<B::E>, B::E)>,
    select: Vec<(B::F, B::E)>,
}

impl<B: Backend> Constraint<B> {
    /// The challenge powers in statement order, from `challenge^initial_power`.
    fn powers(&self, b: &mut B) -> Vec<B::E> {
        let mut power = b.ext_constant(Ext::ONE);
        for _ in 0..self.initial_power {
            power = b.ext_mul(power, self.challenge);
        }
        (0..self.eq.len() + self.select.len())
            .map(|_| {
                let current = power;
                power = b.ext_mul(power, self.challenge);
                current
            })
            .collect()
    }

    fn claimed(&self, b: &mut B, powers: &[B::E]) -> B::E {
        let mut total = b.ext_constant(Ext::ZERO);
        let values = self
            .eq
            .iter()
            .map(|(_, v)| *v)
            .chain(self.select.iter().map(|(_, v)| *v));
        for (value, &power) in values.zip(powers) {
            let term = b.ext_mul(value, power);
            total = b.ext_add(total, term);
        }
        total
    }

    /// The batched weight at the last `variables` folding challenges.
    fn weight(&self, b: &mut B, powers: &[B::E], randomness: &[B::E]) -> B::E {
        // Plonky3: reverse the whole folding point, keep the first `variables`.
        let local: Vec<B::E> = randomness
            .iter()
            .rev()
            .take(self.variables)
            .copied()
            .collect();
        let mut total = b.ext_constant(Ext::ZERO);
        let mut powers = powers.iter();
        for (point, _) in &self.eq {
            let weight = eval_eq(b, point, &local);
            let term = b.ext_mul(weight, *powers.next().expect("one power per statement"));
            total = b.ext_add(total, term);
        }
        for &(var, _) in &self.select {
            let weight = eval_select(b, var, &local);
            let term = b.ext_mul(weight, *powers.next().expect("one power per statement"));
            total = b.ext_add(total, term);
        }
        total
    }
}

/// Draw one query index of `width` bits: reject the sample p − 1, take the low
/// bits of the canonical value. Returns the bits and the index as a word.
fn query_index<B: Backend>(b: &mut B, duplex: &mut Duplex<B>, width: usize) -> Result<(Vec<B::F>, B::F), Error> {
    let sample = duplex.sample(b);
    let one = b.constant(Gl::ONE);
    let shifted = b.add(sample, one);
    let zero = b.constant(Gl::ZERO);
    let embedded = b.ext([shifted, zero, zero]);
    b.ext_inverse(embedded, "WHIR query sample p - 1")?;
    let mut bits = b.bits(sample, 64, "WHIR query bits")?;
    bits.truncate(width);
    let mut index = b.constant(Gl::ZERO);
    for (i, &bit) in bits.iter().enumerate() {
        let weighted = b.scale(bit, Gl::from_u64(1u64 << i));
        index = b.add(index, weighted);
    }
    Ok((bits, index))
}

/// `generator^index` from the index bits.
fn power_from_bits<B: Backend>(b: &mut B, generator: Gl, bits: &[B::F]) -> B::F {
    let mut total = b.constant(Gl::ONE);
    let mut base = generator;
    for &bit in bits {
        let one = b.constant(Gl::ONE);
        let factor_minus = b.scale(bit, base - Gl::ONE);
        let factor = b.add(one, factor_minus);
        total = b.mul(total, factor);
        base = base.square();
    }
    total
}

/// The STIR checks of one query set against `root`: indices, full paths,
/// folds at `folding` (low bit first). Returns the select statements.
#[allow(clippy::too_many_arguments)]
fn stir<B: Backend>(
    b: &mut B,
    duplex: &mut Duplex<B>,
    view_proof: Option<&QueryOpenings<Gl, Ext, crate::pcs::MultiProof>>,
    queries: &Rows<B>,
    draws: usize,
    index_bits: usize,
    log_folded_domain: usize,
    root: [B::F; 4],
    folding: &[B::E],
    base: bool,
) -> Result<Vec<(B::F, B::E)>, Error> {
    let mut indices = Vec::with_capacity(queries.rows.len());
    if draws == 0 {
        for index in 0..1usize << index_bits {
            let bits: Vec<B::F> = (0..index_bits)
                .map(|i| b.constant(Gl::from_u64(((index >> i) & 1) as u64)))
                .collect();
            let word = b.constant(Gl::from_usize(index));
            indices.push((bits, word));
        }
    } else {
        for _ in 0..draws {
            indices.push(query_index(b, duplex, index_bits)?);
        }
    }
    let words: Vec<B::F> = indices.iter().map(|(_, word)| *word).collect();
    let rows_native: Option<Vec<Vec<Gl>>> = view_proof.map(|openings| match openings {
        QueryOpenings::Base(opening) => opening.rows.clone(),
        QueryOpenings::Extension(opening) => opening
            .rows
            .iter()
            .map(|row| row.iter().flat_map(|&value| ext_words(value)).collect())
            .collect(),
    });
    let siblings: Option<Vec<[Gl; 4]>> = view_proof.map(|openings| match openings {
        QueryOpenings::Base(opening) => opening.proof.sibling_hashes.clone(),
        QueryOpenings::Extension(opening) => opening.proof.sibling_hashes.clone(),
    });
    let path_words = b.hint(&words, words.len() * index_bits * 4, &|values| {
        let zeros = vec![Gl::ZERO; values.len() * index_bits * 4];
        let (Some(rows), Some(siblings)) = (&rows_native, &siblings) else {
            return zeros;
        };
        let indices: Vec<usize> = values
            .iter()
            .map(|v| p3_field_v08::PrimeField64::as_canonical_u64(v) as usize)
            .collect();
        let leaves: Vec<[Gl; 4]> = rows.iter().map(|row| crate::hash::hash_leaf(row)).collect();
        match paths::expand(&indices, &leaves, siblings, index_bits) {
            Some(paths) => paths.into_iter().flatten().flatten().collect(),
            None => zeros,
        }
    });
    let generator = Gl::two_adic_generator(log_folded_domain);
    let mut statements = Vec::with_capacity(indices.len());
    for (d, (bits, _)) in indices.iter().enumerate() {
        let row = &queries.rows[d];
        let leaf = hash_leaf(b, row);
        let path: Vec<[B::F; 4]> = (0..index_bits)
            .map(|level| std::array::from_fn(|lane| path_words[(d * index_bits + level) * 4 + lane]))
            .collect();
        let computed = merkle_root(b, leaf, bits, &path);
        for lane in 0..4 {
            b.assert_equal(computed[lane], root[lane], "WHIR Merkle path")?;
        }
        let values: Vec<B::E> = if base {
            row.iter()
                .map(|&value| {
                    let zero = b.constant(Gl::ZERO);
                    b.ext([value, zero, zero])
                })
                .collect()
        } else {
            row.chunks(3).map(|c| b.ext([c[0], c[1], c[2]])).collect()
        };
        let fold = mle(b, &values, folding);
        let var = power_from_bits(b, generator, bits);
        statements.push((var, fold));
    }
    Ok(statements)
}

/// Verify `view` against `root` for points `points` (one per batch, low bit
/// first, each of its table's arity). Returns the opened batch values.
pub(crate) fn verify<B: Backend>(
    b: &mut B,
    pcs: &Pcs,
    root: [B::F; 4],
    view: &OpeningView<'_, B>,
    points: &[Vec<B::E>],
    duplex: &mut Duplex<B>,
) -> Result<Vec<Vec<B::E>>, Error> {
    let config = pcs.config();
    let protocol = pcs.protocol();
    let tables = protocol.table_shapes();
    let (k, placements) = plan_stacked_layout(&tables);
    let shape = WhirShape::new(config, protocol.num_openings());
    if points.len() != protocol.num_openings() {
        return Err(Error::Rejected("WHIR point count"));
    }

    // Out-of-domain claims on the stacked polynomial.
    let mut virtual_claims = Vec::new();
    let virtual_seed = seeds::virtual_claim(k, &tables);
    for &answer in &view.initial_ood {
        absorb(b, duplex, &virtual_seed);
        let x = duplex.sample_ext(b);
        let point = expand(b, x, k);
        duplex.observe_ext(b, answer);
        virtual_claims.push((point, answer));
    }

    // Opening claims, recorded per table.
    let mut claims: Vec<Vec<(usize, Vec<B::E>, B::E)>> = vec![Vec::new(); tables.len()];
    let mut count = 0;
    for (((table, batch), values), point) in protocol.iter_openings().zip(&view.evals).zip(points) {
        absorb(b, duplex, &seeds::opening(k, &tables, table, batch.current()));
        for &value in values {
            duplex.observe_ext(b, value);
        }
        let arity = tables[table].num_variables();
        assert!(point.len() <= arity);
        let zero = b.ext_constant(Ext::ZERO);
        let mut padded = point.clone();
        padded.resize(arity, zero);
        padded.reverse();
        for (&column, &value) in batch.current().iter().zip(values) {
            claims[table].push((column, padded.clone(), value));
            count += 1;
        }
    }

    // The WHIR run.
    absorb(b, duplex, &seeds::whir(config, protocol.num_openings()));
    absorb(b, duplex, &seeds::batching(k, &tables, count, virtual_claims.len()));
    let alpha = duplex.sample_ext(b);
    let mut eq = Vec::new();
    for placement in &placements {
        for (column, point, value) in &claims[placement.idx()] {
            let selector = placement.selectors()[*column];
            let mut lifted: Vec<B::E> = (0..selector.num_variables())
                .map(|i| {
                    let bit = (selector.index() >> (selector.num_variables() - 1 - i)) & 1;
                    b.ext_constant(Ext::from(Gl::from_usize(bit)))
                })
                .collect();
            lifted.extend_from_slice(point);
            eq.push((lifted, *value));
        }
    }
    eq.extend(virtual_claims);
    let mut constraints = vec![Constraint {
        challenge: alpha,
        initial_power: 0,
        variables: k,
        eq,
        select: Vec::new(),
    }];
    let powers = constraints[0].powers(b);
    let mut claimed = constraints[0].claimed(b, &powers);
    let mut all_powers = vec![powers];
    let mut randomness = sumcheck(b, duplex, &view.initial_sumcheck, &mut claimed);
    let mut last_folding = randomness.clone();
    let mut previous_root = root;

    for (r, (round, round_view)) in shape.rounds.iter().zip(&view.rounds).enumerate() {
        let parameters = &config.round_parameters()[r];
        duplex.observe_slice(b, &round_view.root);
        let mut ood = Vec::with_capacity(round_view.ood.len());
        for &answer in &round_view.ood {
            let x = duplex.sample_ext(b);
            let point = expand(b, x, parameters.num_variables);
            duplex.observe_ext(b, answer);
            ood.push((point, answer));
        }
        let select = stir(
            b,
            duplex,
            view.proof.map(|p| &p.whir.rounds[r].openings),
            &round_view.queries,
            round.query_draws,
            round.index_bits,
            parameters.log_folded_domain_size,
            previous_root,
            &last_folding,
            r == 0,
        )?;
        let beta = duplex.sample_ext(b);
        let constraint = Constraint {
            challenge: beta,
            initial_power: 1,
            variables: parameters.num_variables,
            eq: ood,
            select,
        };
        let powers = constraint.powers(b);
        let added = constraint.claimed(b, &powers);
        claimed = b.ext_add(claimed, added);
        constraints.push(constraint);
        all_powers.push(powers);
        last_folding = sumcheck(b, duplex, &round_view.sumcheck, &mut claimed);
        randomness.extend_from_slice(&last_folding);
        previous_root = round_view.root;
    }

    for &value in &view.final_poly {
        duplex.observe_ext(b, value);
    }
    let final_config = config.final_round_config();
    let final_select = stir(
        b,
        duplex,
        view.proof.map(|p| &p.whir.final_openings),
        &view.final_queries,
        shape.final_query_draws,
        shape.final_index_bits,
        final_config.log_folded_domain_size,
        previous_root,
        &last_folding,
        shape.rounds.is_empty(),
    )?;
    for (var, value) in final_select {
        let evaluated = horner(b, &view.final_poly, var);
        b.assert_ext_equal(evaluated, value, "WHIR final polynomial query")?;
    }
    let final_randomness = sumcheck(b, duplex, &view.final_sumcheck, &mut claimed);
    randomness.extend_from_slice(&final_randomness);

    let mut weights = b.ext_constant(Ext::ZERO);
    for (constraint, powers) in constraints.iter().zip(&all_powers) {
        let weight = constraint.weight(b, powers, &randomness);
        weights = b.ext_add(weights, weight);
    }
    let final_value = mle(b, &view.final_poly, &final_randomness);
    let expected = b.ext_mul(weights, final_value);
    b.assert_ext_equal(claimed, expected, "WHIR final sum-check")?;
    Ok(view.evals.clone())
}
