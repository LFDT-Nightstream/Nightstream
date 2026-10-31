//! The programs that the lowering check (`lowering.rs`) records: `Edges`,
//! which takes every branch of `record.rs`, `Shared` and `Early`, which call
//! the verifier's helpers, `LaneSplit`, which builds lane tables that the
//! toy relation does not have, and one program per stage of the layer-1
//! verifier on the toy proof. A stage program starts from the verifier's
//! state at the stage's start, taken from a native run of the stages before
//! it, with every value it reads as a proof word. Each program has a mirror
//! in `CompressionLeanExe/Fixtures.lean`.

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use neo_math::D;
use neo_verified::degrees::{FOLD, RUN_ORACLES};
use p3_field_v08::{PrimeCharacteristicRing, PrimeField64};

use super::verifier::statement_input;
use crate::circuit::algebra::{self, Kx, K};
use crate::circuit::fold::FoldTranscript;
use crate::circuit::hash::{hash_leaf, merkle_root, Duplex};
use crate::circuit::{Backend, Fast, Hint};
use crate::field::{Ext, Gl};
use crate::gkr::{self, GkrView, TreeShape};
use crate::matrix::{Class, EarlyChallenges, EarlyOpenings, EarlyValues, LateOpenings, Structure};
use crate::relation::{kappa, p1_points, setup_point};
use crate::ring::{self, Mixing};
use crate::setup::{self, LeafView};
use crate::shrink::Program;
use crate::verifier::{combine, early_shapes, from_coordinates, lane_value, observe_early, observe_pairs};
use crate::whir::OpeningView;
use crate::R1cs;
use crate::{mle, norm, sumcheck, whir, Claim, ClaimWords, Error, Proof, Relation, LANE_VARIABLES};
use neo_verified::hash::compress;

/// Every branch of `record.rs`: merged, cancelled, zero-weight and long
/// forms; constant sides of products; extension products by and from
/// constants; a folded, a recorded and a failing constant inverse; a skipped
/// and a failing constant assertion; short and 64-bit splits; a selection; a
/// hint; a fixed R1CS system over forms with constants; and a permutation of
/// mixed lanes. Some checks fail: the rows do not depend on that.
pub(super) struct Edges;

impl Program for Edges {
    fn statement(&self) -> Vec<u64> {
        vec![7, 11]
    }

    fn key(&self) -> Vec<u64> {
        Vec::new()
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error> {
        let t = b.public(7);
        let t2 = b.public(11);
        let s = b.private(5);
        let u = b.private(3);
        let w = b.private(9);
        let zero = b.constant(0);
        let two = b.constant(2);
        let double = b.add(s, s);
        let sum = b.add(s, u);
        let sum = b.add(sum, t);
        let four = b.add(sum, t2);
        let shifted = b.add(four, two);
        let five = b.add(shifted, w);
        let cancelled = b.sub(four, s);
        let scaled = b.scale(cancelled, 3);
        let vanished = b.scale(double, 0);
        let left = b.mul(two, s);
        let right = b.mul(scaled, two);
        let constant = b.mul(two, two);
        let product = b.mul(shifted, five);
        let e = b.ext([s, shifted, product]);
        let c = b.ext_constant([3, 0, 1]);
        let by_constant = b.ext_mul(c, e);
        let from_constant = b.ext_mul(e, c);
        let both = b.ext_mul(e, by_constant);
        let folded = b.ext_inverse(c, "constant inverse")?;
        let probe = b.ext_mul(folded, e);
        b.assert_ext_zero(probe, "probe")?;
        let inverse = b.ext_inverse(e, "inverse")?;
        let sum3 = b.ext_add(e, inverse);
        let difference3 = b.ext_sub(sum3, from_constant);
        let scaled3 = b.ext_scale(difference3, u);
        let zero3 = b.ext_constant([0, 0, 0]);
        b.ext_inverse(zero3, "zero inverse")?;
        b.assert_zero(zero, "zero constant")?;
        b.assert_zero(two, "nonzero constant")?;
        b.assert_zero(vanished, "vanished")?;
        b.assert_equal(left, right, "difference")?;
        b.assert_equal(constant, two, "constants")?;
        let low = b.bits(s, 8, "short split")?;
        let all = b.bits(five, 64, "full split")?;
        let chosen = b.select(low[0], u, w);
        let hinted = b.hint(&[s], 2, &Hint::Words(vec![6, 9]));
        let mut system = R1cs::new(4);
        system.push(&[(0, 1)], &[(0, 1)], &[(1, 1)]);
        system.push(&[(2, 3), (3, 1), (2, 1)], &[(2, 1)], &[(3, 5)]);
        b.r1cs(&system, &[s, product, two, shifted], "system")?;
        let [_, _, scaled3] = b.coordinates(scaled3);
        let [_, both, _] = b.coordinates(both);
        let mut state = [zero; WIDTH];
        let lanes = [chosen, two, shifted, hinted[0], all[63], scaled3, both, five];
        state[..lanes.len()].copy_from_slice(&lanes);
        let digest = b.permute(state);
        b.assert_equal(digest[0], t, "digest")?;
        Ok(())
    }
}

/// Every helper that the layer-1 verifier shares, over proof words: the
/// algebra (`neo_verified::algebra`), the hashes and the duplex
/// (`neo_verified::hash`), the fold transcript and its handoff
/// (`circuit/fold.rs`) and `sumcheck::replay`. The duplex absorbs a full and
/// a partial chunk, reads its output, and duplexes once with no input.
/// `shared` in `CompressionLeanExe/Fixtures.lean` writes the same program.
pub(super) struct Shared;

impl Program for Shared {
    fn statement(&self) -> Vec<u64> {
        Vec::new()
    }

    fn key(&self) -> Vec<u64> {
        Vec::new()
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error> {
        let words: Vec<B::F> = (0..14).map(|word| b.private(word)).collect();
        let x: Vec<B::E> = (0..4).map(|_| algebra::private_ext(b, Ext::ZERO)).collect();
        let scaled = algebra::scale(b, words[0], Gl::from_u64(3));
        let lifted = algebra::lift_word(b, scaled);
        let weighted = algebra::ext_scale_constant(b, lifted, Gl::from_u64(9));
        let dot = algebra::dot(b, &x[..2], &x[2..]);
        let eq = algebra::eq_eval(b, &x[..2], &x[2..]);
        let table = algebra::eq_table(b, &x[..2]);
        let at = algebra::eq_at(b, &x[1..], 5);
        let value = algebra::interpolate(b, &table, weighted);
        let horner = algebra::horner(b, &x, dot);
        let k = Kx::from_k(b, [scaled, words[1]]);
        let one = Kx::one(b);
        let zero = Kx::zero(b);
        let k = k.mul(b, Kx { re: eq, im: at });
        let k = k.add(b, one).scale(b, horner).one_minus(b).add(b, zero);
        let bits = b.bits(words[2], 5, "bits")?;
        let power = algebra::power_from_bits(b, Gl::from_u64(7), &bits);
        let leaf = hash_leaf(b, &words);
        let short = hash_leaf(b, &words[..3]);
        let node = compress(b, leaf, short);
        let root = merkle_root(b, leaf, &bits[..2], &[short, node]);
        let mut duplex = Duplex::new(b);
        duplex.observe_slice(b, &words);
        duplex.observe_ext(b, k.re);
        let first = duplex.sample(b);
        let drawn = duplex.sample_ext(b);
        let index = duplex.sample_bits(b, 10)?;
        let more: Vec<B::E> = (0..3).map(|_| duplex.sample_ext(b)).collect();
        duplex.observe(b, power);
        let rounds = [vec![x[0], x[1]], vec![x[2], x[3], value]];
        let (point, last) = sumcheck::replay(b, &mut duplex, &rounds, drawn);
        let mut fold = FoldTranscript::new(b);
        fold.absorb(b, &words);
        fold.absorb_constants(b, &[1, 2, 3]);
        let digest = fold.squeeze_digest(b);
        let seed = FoldTranscript::from_state(std::array::from_fn(|lane| words[lane % words.len()])).handoff(b);
        b.assert_ext_equal(last, point[1], "replay")?;
        b.assert_ext_equal(k.im, more[2], "draws")?;
        b.assert_equal(first, index[3], "index")?;
        for lane in 0..4 {
            b.assert_equal(root[lane], seed[lane], "root")?;
            b.assert_equal(digest[lane], fold.prefix()[lane], "digest")?;
        }
        Ok(())
    }
}

/// The early stage's helpers in `verify`'s order (`verifier.rs:154-167`), on
/// a structure that the toy relation does not have: three matrices, an
/// empty run range, a run count of `2^run_variables`, a GKR tree of depth
/// one and a row point shorter than `r`. The program then reads every
/// early opening. `early` in `CompressionLeanExe/Fixtures.lean` writes the same program.
pub(super) struct Early;

impl Program for Early {
    fn statement(&self) -> Vec<u64> {
        Vec::new()
    }

    fn key(&self) -> Vec<u64> {
        Vec::new()
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error> {
        let structure = Structure {
            matrices: 3,
            row_variables: 1,
            run_variables: 3,
            slot_variables: 1,
            run_bounds: vec![0, 2, 2, 8],
            classes: Vec::new(),
            segments: Vec::new(),
            slots: 2,
        };
        let mut shapes = vec![TreeShape { depth: 2, factors: 0 }];
        shapes.extend(structure.early_shapes());
        let histogram: Vec<B::F> = (0..5).map(|count| b.private(count)).collect();
        let r: Vec<K<B>> = (0..3).map(|_| [b.private(1), b.private(2)]).collect();
        let view = GkrView::read(b, None, &shapes)?;
        let values = EarlyValues {
            runs: std::array::from_fn(|_| algebra::private_ext(b, Ext::ZERO)),
            pairs: (0..structure.matrices)
                .map(|_| std::array::from_fn(|_| algebra::private_ext(b, Ext::ZERO)))
                .collect(),
            block: algebra::private_ext(b, Ext::ZERO),
        };
        let mut duplex = Duplex::new(b);
        duplex.observe_slice(b, &histogram);
        let beta = duplex.sample_ext(b);
        let challenges = EarlyChallenges {
            lookup: [duplex.sample_ext(b), duplex.sample_ext(b)],
            scatter: [duplex.sample_ext(b), duplex.sample_ext(b)],
        };
        let total = norm::table_sum(b, &histogram, 2, beta)?;
        let claims = gkr::verify(b, &view, &shapes, &mut duplex)?;
        let early = structure.check_early(b, &r, challenges, &claims[1..], &values)?;
        for &x in early
            .run_point
            .iter()
            .chain(&early.pair_point)
            .chain(&early.row_point)
        {
            duplex.observe_ext(b, x);
        }
        let [row, slot, coefficient] = early.setup;
        b.assert_ext_equal(slot, total, "slot")?;
        b.assert_ext_equal(row, coefficient, "setup")?;
        Ok(b.assert_ext_equal(early.mult, claims[0].point[1], "mult")?)
    }
}

/// The linear stage of the layer-1 verifier on the toy proof, from the state
/// after the early stage: the statement as statement words, then the duplex
/// after `observe_early` and the proof's quotient, linear rounds and finals as
/// proof words. The duplex's unread output is the first `output` lanes of its
/// state. `run` makes the calls of `verifier.rs:170-206` in its order.
/// `linear` in `CompressionLeanExe/Fixtures.lean` writes the same program.
pub(super) struct Linear<'a> {
    relation: &'a Relation<'a>,
    statement: ClaimWords<Gl>,
    state: [Gl; WIDTH],
    pub(super) input: Vec<Gl>,
    pub(super) output: usize,
    proof: &'a Proof,
}

impl<'a> Linear<'a> {
    /// The duplex after `observe_early` in a native run of `verifier.rs:143-168`
    /// on `proof`: the early stage's transcript calls, without its checks.
    pub(super) fn new(relation: &'a Relation<'a>, seed: [Gl; 4], claim: &Claim, proof: &'a Proof) -> Self {
        let statement = ClaimWords::new(&relation.shape, claim).unwrap();
        let b = &mut Fast;
        let mut duplex = Duplex::new(b);
        duplex.observe_slice(b, &seed);
        for word in relation.words() {
            let word = algebra::constant(b, word);
            duplex.observe(b, word);
        }
        duplex.observe_slice(b, &statement.words());
        whir::observe_root(b, &mut duplex, proof.p0.roots()[0]);
        let histogram: Vec<Gl> = proof
            .histogram
            .iter()
            .map(|&count| Gl::from_u64(u64::from(count)))
            .collect();
        duplex.observe_slice(b, &histogram);
        for _ in 0..5 {
            let _ = duplex.sample_ext(b);
        }
        let shapes = early_shapes(relation);
        let early = GkrView::read(b, Some(&proof.early), &shapes).unwrap();
        gkr::verify(b, &early, &shapes, &mut duplex).unwrap();
        observe_early(b, &mut duplex, &proof.early_values);
        let output = duplex.output.len();
        assert_eq!(
            duplex.output,
            duplex.state[..output],
            "the output is the state's rate lanes"
        );
        Self {
            relation,
            statement,
            state: duplex.state,
            input: duplex.input,
            output,
            proof,
        }
    }
}

impl Program for Linear<'_> {
    fn statement(&self) -> Vec<u64> {
        self.statement
            .words()
            .iter()
            .map(Gl::as_canonical_u64)
            .collect()
    }

    fn key(&self) -> Vec<u64> {
        Vec::new()
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error> {
        let statement = statement_input(b, &self.statement);
        let state = self.state.map(|word| algebra::private(b, word));
        let mut duplex = Duplex {
            state,
            input: self
                .input
                .iter()
                .map(|&word| algebra::private(b, word))
                .collect(),
            output: state[..self.output].to_vec(),
        };
        let exts = |b: &mut B, values: &[Ext]| -> Vec<B::E> {
            values
                .iter()
                .map(|&value| algebra::private_ext(b, value))
                .collect()
        };
        let quotient = exts(b, &self.proof.quotient);
        let linear: Vec<Vec<B::E>> = self
            .proof
            .linear
            .iter()
            .map(|message| exts(b, message))
            .collect();
        let finals = self
            .proof
            .finals
            .map(|value| algebra::private_ext(b, value));
        let shape = &self.relation.shape;

        let lambda = duplex.sample_ext(b);
        let mixing = Mixing::new(b, lambda, shape);
        for &value in &quotient {
            duplex.observe_ext(b, value);
        }
        let zeta = duplex.sample_ext(b);
        let targets = ring::targets(b, &statement, &mixing);
        let value = ring::lifted_target(b, &targets, &quotient, zeta);
        let (point, last) = sumcheck::replay(b, &mut duplex, &linear, value);
        let (block_point, lane_part) = point.split_at(shape.block_variables);
        let [key_value, omega_value, z_value] = finals;
        for value in finals {
            duplex.observe_ext(b, value);
        }
        let (scale, _) = mle::scaled_eq_point(b, zeta, LANE_VARIABLES)?;
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
        Ok(b.assert_ext_equal(last, expected, "linear claim")?)
    }
}

/// The honest fold of the layer-1 verifier on the toy proof, from the state
/// before `μ`: the duplex and every value that the fold reads of the earlier
/// stages and of the proof, as proof words. The duplex's unread output is the
/// first `output` lanes of its state. `run` makes the calls of
/// `verifier.rs:208-249`, then those of `verifier.rs:262-265`, in their
/// order. `honest` in `CompressionLeanExe/Fixtures.lean` writes the same program.
pub(super) struct Honest<'a> {
    relation: &'a Relation<'a>,
    proof: &'a Proof,
    state: [Gl; WIDTH],
    pub(super) input: Vec<Gl>,
    pub(super) output: usize,
    lane_point: Vec<Ext>,
    block_point: Vec<Ext>,
    run_point: Vec<Ext>,
    setup: [Ext; 3],
    commitment_rows: Vec<Ext>,
}

impl<'a> Honest<'a> {
    /// The state at `verifier.rs:208` in a native run of `verifier.rs:143-206`
    /// on `proof`: the transcript calls and the values that the honest fold
    /// reads, without the earlier checks.
    pub(super) fn new(relation: &'a Relation<'a>, seed: [Gl; 4], claim: &Claim, proof: &'a Proof) -> Self {
        let (shape, structure) = (&relation.shape, &relation.key.structure);
        let statement = ClaimWords::new(shape, claim).unwrap();
        let b = &mut Fast;
        let mut duplex = Duplex::new(b);
        duplex.observe_slice(b, &seed);
        for word in relation.words() {
            duplex.observe(b, word);
        }
        duplex.observe_slice(b, &statement.words());
        whir::observe_root(b, &mut duplex, proof.p0.roots()[0]);
        let histogram: Vec<Gl> = proof
            .histogram
            .iter()
            .map(|&count| Gl::from_u64(u64::from(count)))
            .collect();
        duplex.observe_slice(b, &histogram);
        let _beta = duplex.sample_ext(b);
        let challenges = EarlyChallenges {
            lookup: [duplex.sample_ext(b), duplex.sample_ext(b)],
            scatter: [duplex.sample_ext(b), duplex.sample_ext(b)],
        };
        let shapes = early_shapes(relation);
        let view = GkrView::read(b, Some(&proof.early), &shapes).unwrap();
        let claims = gkr::verify(b, &view, &shapes, &mut duplex).unwrap();
        let early = structure
            .check_early(b, &statement.point, challenges, &claims[1..], &proof.early_values)
            .unwrap();
        observe_early(b, &mut duplex, &proof.early_values);
        let lambda = duplex.sample_ext(b);
        let mixing = Mixing::new(b, lambda, shape);
        for &value in &proof.quotient {
            duplex.observe_ext(b, value);
        }
        let zeta = duplex.sample_ext(b);
        let targets = ring::targets(b, &statement, &mixing);
        let value = ring::lifted_target(b, &targets, &proof.quotient, zeta);
        let (point, _) = sumcheck::replay(b, &mut duplex, &proof.linear, value);
        for value in proof.finals {
            duplex.observe_ext(b, value);
        }
        let (_, lane_point) = mle::scaled_eq_point(b, zeta, LANE_VARIABLES).unwrap();
        let output = duplex.output.len();
        assert_eq!(
            duplex.output,
            duplex.state[..output],
            "the output is the state's rate lanes"
        );
        Self {
            relation,
            proof,
            state: duplex.state,
            input: duplex.input,
            output,
            lane_point,
            block_point: point[..shape.block_variables].to_vec(),
            run_point: early.run_point,
            setup: early.setup,
            commitment_rows: (0..kappa()).map(|row| mixing.commitment(row)).collect(),
        }
    }
}

impl Program for Honest<'_> {
    fn statement(&self) -> Vec<u64> {
        Vec::new()
    }

    fn key(&self) -> Vec<u64> {
        Vec::new()
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error> {
        let (relation, proof) = (self.relation, self.proof);
        let key = relation.key;
        let state = self.state.map(|word| algebra::private(b, word));
        let mut duplex = Duplex {
            state,
            input: self
                .input
                .iter()
                .map(|&word| algebra::private(b, word))
                .collect(),
            output: state[..self.output].to_vec(),
        };
        let exts = |b: &mut B, values: &[Ext]| -> Vec<B::E> {
            values
                .iter()
                .map(|&value| algebra::private_ext(b, value))
                .collect()
        };
        let lane_point = exts(b, &self.lane_point);
        let block_point = exts(b, &self.block_point);
        let early_run_point = exts(b, &self.run_point);
        let [row, slot, coefficient] = self.setup.map(|value| algebra::private_ext(b, value));
        let block = algebra::private_ext(b, proof.early_values.block);
        let commitment_rows = exts(b, &self.commitment_rows);
        let key_value = algebra::private_ext(b, proof.finals[0]);
        let partials = proof
            .partials
            .map(|values| values.map(|value| algebra::private_ext(b, value)));
        let p1 = proof.p1.roots()[0].map(|word| algebra::private(b, word));
        let variables = key.setup_variables();
        let leaves = proof
            .leaves
            .iter()
            .map(|leaf| LeafView::read(b, Some(leaf), kappa() + RUN_ORACLES, variables))
            .collect::<Result<Vec<_>, _>>()?;

        let mu = duplex.sample_ext(b);
        for values in &partials {
            for &value in values {
                duplex.observe_ext(b, value);
            }
        }
        let alpha: [B::E; FOLD] = std::array::from_fn(|_| duplex.sample_ext(b));
        let key_point = setup_point(b, &[lane_point.as_slice(), &block_point].concat(), variables);
        let run_point = setup_point(b, &early_run_point, variables);
        let mut embedded = block;
        for &x in &early_run_point[key.structure.slot_variables..] {
            let low = algebra::one_minus(b, x);
            embedded = b.ext_mul(embedded, low);
        }
        let mut run_value = embedded;
        for value in [coefficient, slot, row] {
            let scaled = b.ext_mul(mu, run_value);
            run_value = b.ext_add(value, scaled);
        }
        let key_partial = combine(b, &partials[0], &key_point[..FOLD]);
        b.assert_ext_equal(key_partial, key_value, "honest-fold partials")?;
        let run_partial = combine(b, &partials[1], &run_point[..FOLD]);
        b.assert_ext_equal(run_partial, run_value, "honest-fold partials")?;

        whir::observe_root(b, &mut duplex, p1);
        let index_bits = (0..relation.queries)
            .map(|_| duplex.sample_bits(b, variables + 1 - FOLD))
            .collect::<Result<Vec<_>, _>>()?;
        let mut run_weights = vec![algebra::ext_one(b)];
        for _ in 1..RUN_ORACLES {
            let last = *run_weights.last().expect("one weight");
            run_weights.push(b.ext_mul(last, mu));
        }
        let groups = [commitment_rows, run_weights];
        let root = key.root().map(|word| algebra::constant(b, word));
        for (bits, leaf) in index_bits.iter().zip(&leaves) {
            setup::check(b, root, variables, bits, leaf, &groups, &alpha)?;
        }
        for bits in &index_bits {
            setup::query_point(b, variables, bits);
        }
        Ok(())
    }
}

/// The late stage of the layer-1 verifier on the toy proof, from the
/// verifier's state after the setup queries: the duplex, the lane weights `τ`
/// and the Eval_A weights as proof words, then the proof's late GKR and late
/// values. The duplex's unread output is the first `output` lanes of its
/// state. `run` makes the calls of `verifier.rs:251-260` in its order, then
/// binds the late openings. `late` in `CompressionLeanExe/Fixtures.lean` writes the same
/// program.
pub(super) struct Late<'a> {
    relation: &'a Relation<'a>,
    state: [Gl; WIDTH],
    pub(super) input: Vec<Gl>,
    pub(super) output: usize,
    tau: [Ext; D],
    eval_a: Vec<[Ext; 2]>,
    proof: &'a Proof,
}

impl<'a> Late<'a> {
    /// The state after `verifier.rs:249` in a native run of the transcript
    /// calls of `verifier.rs:143-249` on `proof`, with no check outside
    /// `gkr::verify`.
    pub(super) fn new(relation: &'a Relation<'a>, seed: [Gl; 4], claim: &Claim, proof: &'a Proof) -> Self {
        let statement = ClaimWords::new(&relation.shape, claim).unwrap();
        let b = &mut Fast;
        let mut duplex = Duplex::new(b);
        duplex.observe_slice(b, &seed);
        for word in relation.words() {
            let word = algebra::constant(b, word);
            duplex.observe(b, word);
        }
        duplex.observe_slice(b, &statement.words());
        whir::observe_root(b, &mut duplex, proof.p0.roots()[0]);
        let histogram: Vec<Gl> = proof
            .histogram
            .iter()
            .map(|&count| Gl::from_u64(u64::from(count)))
            .collect();
        duplex.observe_slice(b, &histogram);
        for _ in 0..5 {
            let _ = duplex.sample_ext(b);
        }
        let shapes = early_shapes(relation);
        let early = GkrView::read(b, Some(&proof.early), &shapes).unwrap();
        gkr::verify(b, &early, &shapes, &mut duplex).unwrap();
        observe_early(b, &mut duplex, &proof.early_values);
        let lambda = duplex.sample_ext(b);
        for &value in &proof.quotient {
            duplex.observe_ext(b, value);
        }
        let zeta = duplex.sample_ext(b);
        let _ = sumcheck::replay(b, &mut duplex, &proof.linear, Ext::ZERO);
        for value in proof.finals {
            duplex.observe_ext(b, value);
        }
        let _ = duplex.sample_ext(b);
        for &value in proof.partials.iter().flatten() {
            duplex.observe_ext(b, value);
        }
        for _ in 0..FOLD {
            let _ = duplex.sample_ext(b);
        }
        whir::observe_root(b, &mut duplex, proof.p1.roots()[0]);
        let variables = relation.key.setup_variables();
        for _ in 0..relation.queries {
            duplex
                .sample_bits(b, variables + 1 - FOLD)
                .map_err(Error::from)
                .unwrap();
        }
        let output = duplex.output.len();
        assert_eq!(
            duplex.output,
            duplex.state[..output],
            "the output is the state's rate lanes"
        );
        Self {
            relation,
            state: duplex.state,
            input: duplex.input,
            output,
            tau: ring::tau(b, zeta),
            eval_a: Mixing::new(b, lambda, &relation.shape).eval_a(),
            proof,
        }
    }
}

impl Program for Late<'_> {
    fn statement(&self) -> Vec<u64> {
        Vec::new()
    }

    fn key(&self) -> Vec<u64> {
        Vec::new()
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error> {
        let state = self.state.map(|word| algebra::private(b, word));
        let input = self
            .input
            .iter()
            .map(|&word| algebra::private(b, word))
            .collect();
        let mut duplex = Duplex {
            state,
            input,
            output: state[..self.output].to_vec(),
        };
        let tau = self.tau.map(|value| algebra::private_ext(b, value));
        let pairs = |b: &mut B, values: &[[Ext; 2]]| -> Vec<[B::E; 2]> {
            values
                .iter()
                .map(|pair| pair.map(|value| algebra::private_ext(b, value)))
                .collect()
        };
        let eval_a = pairs(b, &self.eval_a);
        let structure = &self.relation.key.structure;
        let shapes = structure.late_shapes(self.relation.shape.block_variables);
        let view = GkrView::read(b, Some(&self.proof.late), &shapes)?;
        let late_values = pairs(b, &self.proof.late_values);

        let beta = duplex.sample_ext(b);
        let late_claims = gkr::verify(b, &view, &shapes, &mut duplex)?;
        let lanes = structure.lane_tables(b, &tau);
        let late = structure.check_late(b, beta, &eval_a, &lanes, &late_claims, &late_values)?;
        observe_pairs(b, &mut duplex, &late_values);
        let openings = late.pair_point.iter().chain([&late.block]);
        for &value in openings.chain(&late.block_point).chain([&late.omega]) {
            duplex.observe_ext(b, value);
        }
        Ok(())
    }
}

/// `Structure::lane_tables` on classes that the toy relation does not have:
/// a run of `D` lanes, and a run of 30 lanes whose ratio is `p - 2`. The toy's
/// one class has length one and ratio one, so it never skips the dropped
/// lane, never fills the tail and multiplies only by one. The program then
/// binds every table value. `laneSplit` in `CompressionLeanExe/Fixtures.lean` writes the
/// same program.
pub(super) struct LaneSplit;

impl Program for LaneSplit {
    fn statement(&self) -> Vec<u64> {
        Vec::new()
    }

    fn key(&self) -> Vec<u64> {
        Vec::new()
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error> {
        let tau: [B::E; D] = std::array::from_fn(|_| algebra::private_ext(b, Ext::ZERO));
        let structure = Structure {
            matrices: 1,
            row_variables: 1,
            run_variables: 1,
            slot_variables: 1,
            run_bounds: vec![0, 1],
            classes: vec![
                Class {
                    len: D as u64,
                    ratio: 3,
                },
                Class {
                    len: 30,
                    ratio: Gl::ORDER_U64 - 2,
                },
            ],
            segments: Vec::new(),
            slots: 1,
        };
        let lanes = structure.lane_tables(b, &tau);
        let mut duplex = Duplex::new(b);
        for &value in lanes.iter().flatten().flatten() {
            duplex.observe_ext(b, value);
        }
        duplex.sample(b);
        Ok(())
    }
}

/// The openings of the layer-1 verifier on the toy proof, from the state
/// after the query points: the duplex, what the earlier stages leave to the
/// openings and the proof parts that the openings read, all as proof words.
/// `run` makes the calls of `verifier.rs:266-315` in its order. `openings`
/// in `CompressionLeanExe/Fixtures.lean` writes the same program.
pub(super) struct Openings<'a> {
    relation: &'a Relation<'a>,
    proof: &'a Proof,
    state: [Gl; WIDTH],
    pub(super) input: Vec<Gl>,
    pub(super) output: usize,
    query_points: Vec<Vec<Ext>>,
    key_point: Vec<Ext>,
    run_point: Vec<Ext>,
    block_point: Vec<Ext>,
    alpha: [Ext; FOLD],
    queried: Vec<Ext>,
    norm_point: Vec<Ext>,
    linear_point: Vec<Ext>,
    z_leaf: Ext,
    early: EarlyOpenings<Ext>,
    late: LateOpenings<Ext>,
}

impl<'a> Openings<'a> {
    /// The verifier's state after the query points in a native run of
    /// `verifier.rs:143-265` on `proof`: its transcript calls and the values
    /// that the openings read, without its checks.
    pub(super) fn new(relation: &'a Relation<'a>, seed: [Gl; 4], claim: &Claim, proof: &'a Proof) -> Self {
        let (key, shape) = (relation.key, &relation.shape);
        let structure = &key.structure;
        let statement = ClaimWords::new(shape, claim).unwrap();
        let b = &mut Fast;
        let mut duplex = Duplex::new(b);
        duplex.observe_slice(b, &seed);
        for word in relation.words() {
            let word = algebra::constant(b, word);
            duplex.observe(b, word);
        }
        duplex.observe_slice(b, &statement.words());
        whir::observe_root(b, &mut duplex, proof.p0.roots()[0]);
        let histogram: Vec<Gl> = proof
            .histogram
            .iter()
            .map(|&count| Gl::from_u64(u64::from(count)))
            .collect();
        duplex.observe_slice(b, &histogram);
        let beta = duplex.sample_ext(b);
        let challenges = EarlyChallenges {
            lookup: [duplex.sample_ext(b), duplex.sample_ext(b)],
            scatter: [duplex.sample_ext(b), duplex.sample_ext(b)],
        };
        let shapes = early_shapes(relation);
        let view = GkrView::read(b, Some(&proof.early), &shapes).unwrap();
        let claims = gkr::verify(b, &view, &shapes, &mut duplex).unwrap();
        let early = structure
            .check_early(b, &statement.point, challenges, &claims[1..], &proof.early_values)
            .unwrap();
        observe_early(b, &mut duplex, &proof.early_values);

        let lambda = duplex.sample_ext(b);
        let mixing = Mixing::new(b, lambda, shape);
        for &value in &proof.quotient {
            duplex.observe_ext(b, value);
        }
        let zeta = duplex.sample_ext(b);
        let (point, _) = sumcheck::replay(b, &mut duplex, &proof.linear, Ext::ZERO);
        let (block_point, lane_part) = point.split_at(shape.block_variables);
        for value in proof.finals {
            duplex.observe_ext(b, value);
        }
        let (_, lane_point) = mle::scaled_eq_point(b, zeta, LANE_VARIABLES).unwrap();

        let mu = duplex.sample_ext(b);
        for values in &proof.partials {
            for &value in values {
                duplex.observe_ext(b, value);
            }
        }
        let alpha: [Ext; FOLD] = std::array::from_fn(|_| duplex.sample_ext(b));
        let variables = key.setup_variables();
        let key_point = setup_point(b, &[lane_point.as_slice(), block_point].concat(), variables);
        let run_point = setup_point(b, &early.run_point, variables);
        whir::observe_root(b, &mut duplex, proof.p1.roots()[0]);
        let index_bits: Vec<Vec<Gl>> = (0..relation.queries)
            .map(|_| {
                duplex
                    .sample_bits(b, variables + 1 - FOLD)
                    .map_err(Error::from)
                    .unwrap()
            })
            .collect();
        let mut run_weights = vec![Ext::ONE];
        for _ in 1..RUN_ORACLES {
            run_weights.push(*run_weights.last().unwrap() * mu);
        }
        let groups = [(0..kappa()).map(|row| mixing.commitment(row)).collect(), run_weights];
        let mut queried = Vec::new();
        for (bits, leaf) in index_bits.iter().zip(&proof.leaves) {
            let leaf = LeafView::read(b, Some(leaf), kappa() + RUN_ORACLES, variables).unwrap();
            queried.extend(setup::check(b, key.root(), variables, bits, &leaf, &groups, &alpha).unwrap());
        }

        let late_beta = duplex.sample_ext(b);
        let late_shapes = structure.late_shapes(shape.block_variables);
        let view = GkrView::read(b, Some(&proof.late), &late_shapes).unwrap();
        let late_claims = gkr::verify(b, &view, &late_shapes, &mut duplex).unwrap();
        let tau = ring::tau(b, zeta);
        let lanes = structure.lane_tables(b, &tau);
        let late = structure
            .check_late(b, late_beta, &mixing.eval_a(), &lanes, &late_claims, &proof.late_values)
            .unwrap();
        observe_pairs(b, &mut duplex, &proof.late_values);
        let output = duplex.output.len();
        assert_eq!(
            duplex.output,
            duplex.state[..output],
            "the output is the state's rate lanes"
        );
        let query_points = index_bits
            .iter()
            .map(|bits| setup::query_point(b, variables, bits))
            .collect();
        Self {
            relation,
            proof,
            state: duplex.state,
            input: duplex.input,
            output,
            query_points,
            key_point,
            run_point,
            block_point: block_point.to_vec(),
            alpha,
            queried,
            norm_point: claims[0].point.clone(),
            linear_point: [lane_part, block_point].concat(),
            z_leaf: beta - claims[0].values[0],
            early,
            late,
        }
    }
}

impl Program for Openings<'_> {
    fn statement(&self) -> Vec<u64> {
        Vec::new()
    }

    fn key(&self) -> Vec<u64> {
        Vec::new()
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error> {
        let (relation, proof) = (self.relation, self.proof);
        let words = |b: &mut B, values: &[Gl]| -> Vec<B::F> {
            values
                .iter()
                .map(|&word| algebra::private(b, word))
                .collect()
        };
        let exts = |b: &mut B, values: &[Ext]| -> Vec<B::E> {
            values
                .iter()
                .map(|&value| algebra::private_ext(b, value))
                .collect()
        };
        let pairs = |b: &mut B, values: &[[Ext; 2]]| -> Vec<[B::E; 2]> {
            values
                .iter()
                .map(|pair| pair.map(|value| algebra::private_ext(b, value)))
                .collect()
        };
        let state = self.state.map(|word| algebra::private(b, word));
        let mut duplex = Duplex {
            state,
            input: words(b, &self.input),
            output: state[..self.output].to_vec(),
        };
        let query_points = self
            .query_points
            .iter()
            .map(|point| exts(b, point))
            .collect();
        let key_point = exts(b, &self.key_point);
        let run_point = exts(b, &self.run_point);
        let block_point = exts(b, &self.block_point);
        let late_block_point = exts(b, &self.late.block_point);
        let alpha = exts(b, &self.alpha);
        let queried = exts(b, &self.queried);
        let omega_value = algebra::private_ext(b, proof.finals[1]);
        let late_omega = algebra::private_ext(b, self.late.omega);
        let norm_point = exts(b, &self.norm_point);
        let linear_point = exts(b, &self.linear_point);
        let early_run_point = exts(b, &self.early.run_point);
        let pair_point = exts(b, &self.early.pair_point);
        let late_pair_point = exts(b, &self.late.pair_point);
        let row_point = exts(b, &self.early.row_point);
        let z_leaf = algebra::private_ext(b, self.z_leaf);
        let z_value = algebra::private_ext(b, proof.finals[2]);
        let late_block = algebra::private_ext(b, self.late.block);
        let mult = algebra::private_ext(b, self.early.mult);
        let p1 = proof.p1.roots()[0].map(|word| algebra::private(b, word));
        let p0 = proof.p0.roots()[0].map(|word| algebra::private(b, word));
        let partials = proof
            .partials
            .map(|values| values.map(|value| algebra::private_ext(b, value)));
        let runs = proof
            .early_values
            .runs
            .map(|value| algebra::private_ext(b, value));
        let early_pairs = pairs(b, &proof.early_values.pairs);
        let early_block = algebra::private_ext(b, proof.early_values.block);
        let late_values = pairs(b, &proof.late_values);
        let p1_opening = OpeningView::read(b, &relation.p1, Some(&proof.p1_opening))?;
        let p0_opening = OpeningView::read(b, &relation.p0, Some(&proof.p0_opening))?;

        let points = p1_points(&key_point, &run_point, query_points, &block_point, &late_block_point);
        let opened = whir::verify(b, &relation.p1, p1, &p1_opening, &points, &mut duplex)?;
        let mut expected = vec![combine(b, &partials[0], &alpha), combine(b, &partials[1], &alpha)];
        expected.extend(queried);
        expected.extend([omega_value, late_omega]);
        let values: Vec<B::E> = opened
            .iter()
            .flat_map(|batch| batch.chunks(3))
            .map(|coordinates| from_coordinates(b, coordinates))
            .collect();
        assert_eq!(values.len(), expected.len());
        for (value, expected) in values.into_iter().zip(expected) {
            b.assert_ext_equal(value, expected, "P1 openings")?;
        }

        let slots = early_run_point[..relation.key.structure.slot_variables].to_vec();
        let points = vec![
            norm_point,
            linear_point,
            early_run_point,
            pair_point,
            slots,
            late_pair_point,
            row_point,
        ];
        let opened = whir::verify(b, &relation.p0, p0, &p0_opening, &points, &mut duplex)?;
        let flat = |pairs: &[[B::E; 2]]| pairs.iter().flatten().copied().collect::<Vec<B::E>>();
        let [e_re, e_im, _] = runs;
        let mut late_slots = flat(&late_values);
        late_slots.push(late_block);
        let expected = [
            vec![z_leaf],
            vec![z_value],
            vec![e_re, e_im],
            flat(&early_pairs),
            vec![early_block],
            late_slots,
            vec![mult],
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
}
