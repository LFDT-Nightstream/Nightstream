//! The PiCCS replay over any backend: `pi_ccs_joint_protocol::verify_with_trace`
//! for one fresh source and sixteen running sources.
//!
//! Owns: the statement binding, the coin reads, the initial and terminal
//! claims, the sum-check rounds (coefficient form) and the output message.
//! The output instances are the inputs' commitments and public inputs at the
//! new point, so only their evaluations are read.

use neo_ccs::crypto::poseidon2_goldilocks::RATE;
use neo_ccs::SparsePoly;
use neo_math::{D, F};
use neo_spartan::{Backend, Error, FoldTranscript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::k::{self, Kw};
use super::words::Evaluations;

/// Coins per state: one rate-lane pair each.
const COINS_PER_CHUNK: usize = RATE / 2;

/// The coin at `position`, from rate-lane pair `position mod 6`; after the
/// sixth pair of a state, absorb one zero chunk.
fn read_coin<B: Backend>(b: &mut B, transcript: &mut FoldTranscript<B>, position: usize) -> Kw<B> {
    let pair = position % COINS_PER_CHUNK;
    let value = transcript.read_pair(pair);
    if pair == COINS_PER_CHUNK - 1 {
        transcript.absorb_constants(b, &[0; RATE]);
    }
    value
}

/// `Σ_{r,c} γ^{r + R·c}·Eval_K[r][c]` and
/// `Σ_{r,j,c} γ^{r + R·j + R·t·c}·Eval_A[r][j][c]`, with `R` sources.
fn family_sums<B: Backend>(b: &mut B, gamma: Kw<B>, sources: &[&Evaluations<B>]) -> (Kw<B>, Kw<B>) {
    let matrices = sources[0].eval_a.len();
    let per_source = k::power(b, gamma, sources.len());
    let per_matrix = k::power(b, per_source, matrices);
    let over_sources = |b: &mut B, pick: &dyn Fn(&Evaluations<B>) -> Kw<B>| {
        let values: Vec<Kw<B>> = sources.iter().map(|source| pick(source)).collect();
        k::horner(b, &values, gamma)
    };
    let eval_k: Vec<Kw<B>> = (0..D)
        .map(|c| over_sources(b, &|source| source.eval_k[c]))
        .collect();
    let eval_k = k::horner(b, &eval_k, per_source);
    let eval_a: Vec<Kw<B>> = (0..D)
        .map(|c| {
            let by_matrix: Vec<Kw<B>> = (0..matrices)
                .map(|j| over_sources(b, &|source| source.eval_a[j][c]))
                .collect();
            k::horner(b, &by_matrix, per_source)
        })
        .collect();
    let eval_a = k::horner(b, &eval_a, per_matrix);
    (eval_k, eval_a)
}

/// The CCS polynomial `f` at `values`, over `K`.
fn ccs<B: Backend>(b: &mut B, f: &SparsePoly<F>, values: &[Kw<B>]) -> Kw<B> {
    let mut total = k::zero(b);
    for term in f.terms() {
        let mut product = [b.constant(term.coeff.as_canonical_u64()), b.constant(0)];
        for (&value, &exponent) in values.iter().zip(&term.exps) {
            for _ in 0..exponent {
                product = k::mul(b, product, value);
            }
        }
        total = k::add(b, total, product);
    }
    total
}

/// `Π_{i = -(base-1)}^{base-1} (y - i)`.
fn range_product<B: Backend>(b: &mut B, y: Kw<B>, base: u32) -> Kw<B> {
    let bound = i64::from(base) - 1;
    let mut product = k::one(b);
    for integer in -bound..=bound {
        let value = k::constant(b, neo_math::K::from(F::from_i64(integer)));
        let factor = k::sub(b, y, value);
        product = k::mul(b, product, factor);
    }
    product
}

/// The fixed inputs of one replay.
pub(super) struct Inputs<'a, B: Backend> {
    pub(super) f: &'a SparsePoly<F>,
    pub(super) base: u32,
    /// The state digest, which names the running claims.
    pub(super) digest: [B::F; 4],
    pub(super) fresh_commitment: &'a [[B::F; D]],
    pub(super) fresh_public: &'a [B::F],
    pub(super) running: &'a [Evaluations<B>],
    pub(super) prior_point: &'a [Kw<B>],
    pub(super) rounds: &'a [Vec<Kw<B>>],
    /// The fresh source, then the running ones.
    pub(super) outputs: &'a [Evaluations<B>],
}

/// Replay PiCCS from a reset transcript; returns the new point.
pub(super) fn replay<B: Backend>(
    b: &mut B,
    transcript: &mut FoldTranscript<B>,
    inputs: &Inputs<'_, B>,
) -> Result<Vec<Kw<B>>, Error> {
    let domain = neo_transcript::fold_domain_chunk_v1_1().map(|word| word.as_canonical_u64());
    transcript.absorb_constants(b, &domain);
    let mut statement = inputs.digest.to_vec();
    statement.extend(inputs.fresh_commitment.iter().flatten());
    statement.extend_from_slice(inputs.fresh_public);
    transcript.absorb(b, &statement);
    let variables = inputs.prior_point.len();
    let alpha: Vec<Kw<B>> = (0..variables)
        .map(|index| read_coin(b, transcript, index))
        .collect();
    let gamma = read_coin(b, transcript, variables);

    let running: Vec<&Evaluations<B>> = inputs.running.iter().collect();
    let (eval_k, eval_a) = family_sums(b, gamma, &running);
    let eval_a_shift = k::power(b, gamma, running.len() * D);
    let shifted = k::mul(b, eval_a_shift, eval_a);
    let mut claim = k::add(b, eval_k, shifted);

    let mut point = Vec::with_capacity(variables);
    for coefficients in inputs.rounds {
        // h(0) + h(1) = 2·c_0 + Σ_{i≥1} c_i.
        let mut sum = coefficients[0];
        for &c in coefficients {
            sum = k::add(b, sum, c);
        }
        k::assert_equal(b, sum, claim, "PiCCS round sum")?;
        let fields: Vec<B::F> = coefficients.iter().flatten().copied().collect();
        transcript.absorb(b, &fields);
        let challenge = read_coin(b, transcript, 0);
        claim = k::horner(b, coefficients, challenge);
        point.push(challenge);
    }

    let outputs = inputs.outputs;
    let fresh = &outputs[0];
    let first: Vec<Kw<B>> = fresh.eval_a.iter().map(|values| values[0]).collect();
    let residual = ccs(b, inputs.f, &first);
    let ranges: Vec<Kw<B>> = outputs
        .iter()
        .map(|output| range_product(b, output.eval_k[0], inputs.base))
        .collect();
    let norm = k::horner(b, &ranges, gamma);
    let carried: Vec<&Evaluations<B>> = outputs[1..].iter().collect();
    let (out_k, out_a) = family_sums(b, gamma, &carried);
    let prior = k::equality(b, &point, inputs.prior_point);
    let matrices = fresh.eval_a.len();
    let constraint_shift = k::power(b, gamma, carried.len() * D * (matrices + 1));
    let at_alpha = k::equality(b, &point, &alpha);
    let weighted_norm = k::mul(b, gamma, norm);
    let constraint = k::add(b, residual, weighted_norm);
    let constraint = k::mul(b, at_alpha, constraint);
    let constraint = k::mul(b, constraint_shift, constraint);
    let out_k = k::mul(b, prior, out_k);
    let out_a = k::mul(b, prior, out_a);
    let out_a = k::mul(b, eval_a_shift, out_a);
    let terminal = k::add(b, out_k, out_a);
    let terminal = k::add(b, terminal, constraint);
    k::assert_equal(b, claim, terminal, "PiCCS terminal claim")?;

    let mut message = Vec::new();
    for output in outputs {
        message.extend(output.eval_k.iter().flatten());
        message.extend(output.eval_a.iter().flatten().flatten());
    }
    transcript.absorb(b, &message);
    Ok(point)
}
