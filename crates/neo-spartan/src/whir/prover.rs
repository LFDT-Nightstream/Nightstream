//! Our WHIR prover for the stacked suffix layout: Plonky3 0.8's `commit`,
//! `open_at` and `prove`, with each codeword committed one coset at a time.
//!
//! Owns: the order of the prover's transcript steps, which is Plonky3's (the
//! same transcript, layout and sum-check calls in the same order), so
//! commitments and proofs are byte-identical to Plonky3's (test
//! `tests/internal/whir_prover.rs`); the coset encoding of each codeword;
//! and the prover data between a commitment and its opening. Does not own:
//! the configuration (`pcs.rs`), the Merkle tree (`merkle.rs`), the layout
//! and sum-check arithmetic (Plonky3), or verification (`whir/mod.rs`).
//!
//! Row `k` of a codeword of `n` rows is `Σ_i ω^{ik}·row_i` over the message
//! rows, with `ω` of order `n`. Its rows `a, a + c, a + 2c, ...` (coset `a`
//! of `c`) are the DFT of length `n / c` of the message twisted by
//! `ω^{ia}` and folded modulo `n / c`. Each coset goes straight into the
//! tree, so no codeword is ever whole in memory.

use p3_commit_v08::Mmcs as _;
use p3_dft_v08::TwoAdicSubgroupDft;
use p3_field_v08::{BasedVectorSpace, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix_v08::dense::RowMajorMatrix;
use p3_multilinear_util_v08::point::Point;
use p3_multilinear_util_v08::poly::{Poly, PolyMaybePacked};
use p3_sumcheck_v08::constraints::statement::{EqStatement, SelectStatement};
use p3_sumcheck_v08::constraints::{Constraint, Statements};
use p3_sumcheck_v08::layout::{observe_commitment, plan_stacked_layout, Layout as _, Witness};
use p3_sumcheck_v08::strategy::SumcheckProver;
use p3_sumcheck_v08::{OpeningProtocol, SumcheckData};
use p3_whir::pcs::proof::{PcsProof, QueryOpenings, SharedProofOpening, WhirProof, WhirRoundProof};
use p3_whir::transcript::{WhirProverTranscript, WhirShape};
use p3_whir::{WhirDomain, WhirQueryPoint};
use rayon::prelude::*;

use crate::field::{as_base, Ext, Gl};
use crate::hash::Challenger;
use crate::merkle::{self, Mmcs};
use crate::pcs::{Commitment, Config, Dft, Layout, MultiProof, Opening};
use crate::scratch;

/// Cosets per codeword: one coset, a sixteenth of the codeword, is in memory
/// at a time. Measured on a P0-like table set at 2^27 cells on the CPU
/// (`p0_like_heap_and_time`, 2026-10-20): the commitment adds 0.22 GB to
/// its tables (4 cosets: 0.67 GB, 8: 0.37 GB, 32: 0.14 GB), below the
/// opening's 0.56 GB, which Plonky3's layout phase sets at any count. The
/// fold of the stacked message, one pass per coset, takes 0.15 s of the
/// 3.4 s commitment.
const COSETS: usize = 16;

type Tree = merkle::Tree<RowMajorMatrix<Gl>>;
type Transcript<'a> = WhirProverTranscript<'a, Challenger, Gl, Ext>;

/// What a commitment leaves for its opening.
pub(crate) struct ProverData {
    /// The stacked tables and the claims recorded on them: the opening needs
    /// them, the commitment does not.
    pub(crate) layout: Layout,
    tree: Tree,
}

/// Plonky3's `WhirProver::commit`: the root of the stacked codeword, bound
/// in `challenger`.
pub(crate) fn commit(
    config: &Config,
    dft: &Dft,
    mmcs: &Mmcs,
    witness: Witness<Gl>,
    challenger: &mut Challenger,
) -> (Commitment, ProverData) {
    assert_eq!(witness.num_variables(), config.num_variables());
    let layout = Layout::from_witness(witness);
    let message = stacked(&layout, config.round_folding_factor(0));
    let (commitment, tree) = encode(dft, mmcs, &message, config.starting_log_inv_rate);
    observe_commitment::<Gl, _, _>(challenger, commitment.clone());
    (commitment, ProverData { layout, tree })
}

/// Plonky3's `open_at`: every batch at its point, then the WHIR run.
pub(crate) fn open(
    config: &Config,
    dft: &Dft,
    mmcs: &Mmcs,
    data: ProverData,
    protocol: &OpeningProtocol,
    points: &[Point<Ext>],
    challenger: &mut Challenger,
) -> Opening {
    let claims = protocol
        .checked_num_claims()
        .and_then(|claims| claims.checked_add(config.commitment_ood_samples()))
        .expect("the claim count fits");
    config
        .validate_initial_claims(claims)
        .expect("the plan fixes the claims");
    assert_eq!(protocol.num_openings(), points.len(), "one point per batch");
    let ProverData { mut layout, tree } = data;
    let initial_ood_answers = (0..config.commitment_ood_samples())
        .map(|_| layout.add_virtual_eval(challenger))
        .collect();
    let evals = protocol
        .iter_openings()
        .zip(points)
        .map(|((table, batch), point)| layout.eval_at(table, batch, point, challenger))
        .collect();
    let run = Run { config, dft, mmcs };
    PcsProof {
        whir: run.prove(initial_ood_answers, challenger, layout, tree, protocol.num_openings()),
        evals,
    }
}

/// What every step of one WHIR run reads.
struct Run<'a> {
    config: &'a Config,
    dft: &'a Dft,
    mmcs: &'a Mmcs,
}

/// The state between rounds: the sum-check that folds the polynomial, its
/// last challenges, and the codeword the next queries open.
struct RoundState {
    sumcheck: SumcheckProver<Gl, Ext>,
    folding: Point<Ext>,
    codeword: Codeword,
}

/// A committed codeword: the stacked one has base rows, a folded one
/// extension rows.
enum Codeword {
    Base(Tree),
    Extension(Tree),
}

impl Run<'_> {
    /// Plonky3's `WhirProver::prove`.
    fn prove(
        &self,
        initial_ood_answers: Vec<Ext>,
        challenger: &mut Challenger,
        layout: Layout,
        tree: Tree,
        num_opening_claims: usize,
    ) -> WhirProof<Gl, Ext, Mmcs> {
        let config = self.config;
        assert_eq!(config.round_folding_factor(0), layout.folding());
        let mut transcript = Transcript::new(challenger, WhirShape::new(config, num_opening_claims));
        let mut initial_sumcheck = SumcheckData::default();
        let (sumcheck, folding) = transcript.delegate_initial_fold(|challenger| {
            layout.into_sumcheck(&mut initial_sumcheck, config.starting_folding_pow_bits(), challenger)
        });
        let mut state = RoundState {
            sumcheck,
            folding,
            codeword: Codeword::Base(tree),
        };
        let rounds = (0..config.n_rounds())
            .map(|index| self.round(index, &mut transcript, &mut state))
            .collect();
        let (final_poly, final_pow_witness, final_openings, final_sumcheck) =
            self.final_round(&mut transcript, &mut state);
        transcript.finish();
        WhirProof {
            initial_ood_answers,
            initial_sumcheck,
            rounds,
            final_poly,
            final_pow_witness,
            final_openings,
            final_sumcheck,
        }
    }

    /// Plonky3's `WhirProver::round`: commit to the folded polynomial,
    /// answer its out-of-domain points, open the last codeword at the drawn
    /// queries, and fold again under both claim groups.
    fn round(
        &self,
        index: usize,
        transcript: &mut Transcript<'_>,
        state: &mut RoundState,
    ) -> WhirRoundProof<Gl, Ext, Mmcs> {
        let config = self.config;
        let num_variables = config.num_variables() - config.total_folded_through(index);
        assert_eq!(num_variables, state.sumcheck.num_variables());
        let params = &config.round_parameters()[index];
        let folding = config.round_folding_factor(index + 1);
        let (root, tree) = self.encode_folded(&mut state.sumcheck, folding, params.log_inv_rate);
        transcript.commitment(root.clone());

        let mut ood_statement = EqStatement::initialize(num_variables);
        let mut ood_answers = Vec::with_capacity(params.ood_samples);
        for _ in 0..params.ood_samples {
            let point = Point::expand_from_univariate(transcript.ood_point(), num_variables);
            let eval = state.sumcheck.eval(&point);
            transcript.ood_answer(eval);
            ood_answers.push(eval);
            ood_statement.add_evaluated_constraint(point, eval);
        }
        let pow_witness = transcript.query_pow(index);
        let queries = transcript.query_indices(index);

        // Each opened row folds at the last challenges, in suffix order.
        let randomness = state.folding.reversed();
        let openings = open_rows(self.mmcs, &state.codeword, &queries);
        let evals: Vec<Ext> = match &openings {
            QueryOpenings::Base(opening) => opening
                .rows
                .iter()
                .map(|row| Poly::new(row.as_slice()).eval_base(&randomness))
                .collect(),
            QueryOpenings::Extension(opening) => opening
                .rows
                .iter()
                .map(|row| Poly::new(row.as_slice()).eval_ext::<Gl>(&randomness))
                .collect(),
        };
        let mut stir_statement = SelectStatement::initialize(num_variables);
        for (&query, eval) in queries.iter().zip(evals) {
            match WhirDomain::<Gl, Ext>::query_point(
                self.dft,
                params.log_folded_domain_size,
                params.num_variables,
                query,
            ) {
                WhirQueryPoint::Univariate(var) => stir_statement.add_constraint(var, eval),
                WhirQueryPoint::Multilinear(point) => stir_statement.add_point_constraint(point, eval),
            }
        }
        let constraint = Constraint::new_with_existing_claim(
            transcript.round_batching(),
            ood_statement.num_variables(),
            vec![Statements::Eq(ood_statement), Statements::Select(stir_statement)],
        );
        let mut sumcheck = SumcheckData::default();
        state.folding = transcript.delegate_round_fold(|challenger| {
            state.sumcheck.compute_sumcheck_polynomials(
                &mut sumcheck,
                challenger,
                folding,
                params.folding_pow_bits,
                Some(constraint),
            )
        });
        state.codeword = Codeword::Extension(tree);
        WhirRoundProof {
            commitment: Some(root),
            ood_answers,
            pow_witness,
            openings,
            sumcheck,
        }
    }

    /// Plonky3's `WhirProver::final_round`: the folded polynomial in the
    /// clear, the last queries, and the closing sum-check.
    fn final_round(
        &self,
        transcript: &mut Transcript<'_>,
        state: &mut RoundState,
    ) -> (
        Option<Poly<Ext>>,
        Gl,
        QueryOpenings<Gl, Ext, MultiProof>,
        Option<SumcheckData<Gl, Ext>>,
    ) {
        let config = self.config;
        let index = config.n_rounds();
        let final_poly = state.sumcheck.evals();
        assert_eq!(
            final_poly.num_variables(),
            config.num_variables() - config.total_folded_through(index)
        );
        transcript.final_poly(final_poly.as_slice());
        let final_pow_witness = transcript.query_pow(index);
        let openings = open_rows(self.mmcs, &state.codeword, &transcript.query_indices(index));
        let final_sumcheck = transcript.delegate_final_fold(|challenger| {
            let mut sumcheck = SumcheckData::default();
            // The prover keeps the folded state, so the folding point is not needed.
            let _ = state.sumcheck.compute_sumcheck_polynomials(
                &mut sumcheck,
                challenger,
                config.final_sumcheck_rounds(),
                config.final_folding_pow_bits(),
                None,
            );
            sumcheck
        });
        (Some(final_poly), final_pow_witness, openings, final_sumcheck)
    }

    /// Commit to the polynomial of `sumcheck` as rows of `2^folding` values
    /// at `log_inv_rate`, read in place.
    fn encode_folded(
        &self,
        sumcheck: &mut SumcheckProver<Gl, Ext>,
        folding: usize,
        log_inv_rate: usize,
    ) -> (Commitment, Tree) {
        let PolyMaybePacked::Scalar(evals) = sumcheck.evals_view() else {
            unreachable!("Plonky3 packs only prefix-order tables; the stacked layout is in suffix order")
        };
        let evals = evals.into_slice();
        let message = Message {
            width: 3 << folding,
            rows: evals.len() >> folding,
            runs: vec![(0, as_base(evals))],
        };
        encode(self.dft, self.mmcs, &message, log_inv_rate)
    }
}

/// The rows at `queries` of `codeword` and one pruned multiproof, as
/// Plonky3's `SharedProofOpening::open`.
fn open_rows(mmcs: &Mmcs, codeword: &Codeword, queries: &[usize]) -> QueryOpenings<Gl, Ext, MultiProof> {
    let (Codeword::Base(tree) | Codeword::Extension(tree)) = codeword;
    let (values, proof) = mmcs.open_multi_batch(queries, tree);
    // A codeword is one matrix, so each query opens one row.
    let rows = values
        .into_iter()
        .map(|mut matrices| matrices.swap_remove(0));
    match codeword {
        Codeword::Base(_) => QueryOpenings::Base(SharedProofOpening {
            rows: rows.collect(),
            proof,
        }),
        Codeword::Extension(_) => QueryOpenings::Extension(SharedProofOpening {
            rows: rows.map(Ext::reconstitute_from_base).collect(),
            proof,
        }),
    }
}

/// Commit the Reed-Solomon codeword of `message` at `log_inv_rate`, one
/// coset at a time.
fn encode(dft: &Dft, mmcs: &Mmcs, message: &Message<'_>, log_inv_rate: usize) -> (Commitment, Tree) {
    let rows = message.rows << log_inv_rate;
    let log_rows = rows.trailing_zeros() as usize;
    mmcs.commit_cosets(rows, message.width, COSETS.min(rows), |coset, values| {
        message.twist(log_rows, coset, values);
        *values = dft
            .dft_batch(RowMajorMatrix::new(std::mem::take(values), message.width))
            .values;
    })
}

/// The stacked polynomial of `layout` as rows of `2^folding` values: every
/// table column in its slot, read in place.
fn stacked(layout: &Layout, folding: usize) -> Message<'_> {
    let shapes = layout.table_shapes();
    let (variables, placements) = plan_stacked_layout(&shapes);
    assert_eq!(variables, layout.num_variables(), "the planner placed these tables");
    let mut runs: Vec<(usize, &[Gl])> = placements
        .iter()
        .flat_map(|placement| {
            let table = layout.table(placement.idx());
            let arity = shapes[placement.idx()].num_variables();
            placement
                .selectors()
                .iter()
                .enumerate()
                .map(move |(column, selector)| {
                    ((selector.index() << arity) >> folding, table.poly(column).into_slice())
                })
        })
        .collect();
    runs.sort_unstable_by_key(|&(row, _)| row);
    Message {
        width: 1 << folding,
        rows: 1 << (variables - folding),
        runs,
    }
}

/// The message of a codeword: `rows` rows of `width` base words, the given
/// runs and zeros elsewhere.
struct Message<'a> {
    width: usize,
    rows: usize,
    /// Runs of whole rows, `(first row, words)`, in row order.
    runs: Vec<(usize, &'a [Gl])>,
}

impl Message<'_> {
    /// The input of coset `coset` of the codeword of `2^log_rows` rows: row
    /// `u` of `values` becomes `Σ_{i ≡ u mod l} ω^{i·coset}·row_i`, with `l`
    /// the rows of `values` and `ω` of order `2^log_rows`.
    fn twist(&self, log_rows: usize, coset: usize, values: &mut [Gl]) {
        let length = values.len() / self.width;
        let step = Gl::two_adic_generator(log_rows).exp_u64(coset as u64);
        let block = (scratch::BLOCK_WORDS / self.width).max(1);
        values
            .par_chunks_mut(block * self.width)
            .enumerate()
            .for_each(|(b, out)| {
                out.fill(Gl::ZERO);
                let first = b * block;
                let count = out.len() / self.width;
                for start in (first..self.rows).step_by(length) {
                    self.for_each_run(start, (start + count).min(self.rows), |row, words| {
                        let mut twiddle = step.exp_u64(row as u64);
                        for (r, words) in words.chunks_exact(self.width).enumerate() {
                            let out = &mut out[(row + r - start) * self.width..][..self.width];
                            for (value, &word) in out.iter_mut().zip(words) {
                                *value += word * twiddle;
                            }
                            twiddle *= step;
                        }
                    });
                }
            });
    }

    /// `visit(row, words)` for the part of every run in rows `start..end`.
    fn for_each_run(&self, start: usize, end: usize, mut visit: impl FnMut(usize, &[Gl])) {
        let first = self
            .runs
            .partition_point(|&(row, words)| row + words.len() / self.width <= start);
        for &(row, words) in &self.runs[first..] {
            if row >= end {
                break;
            }
            let (from, to) = (row.max(start), (row + words.len() / self.width).min(end));
            visit(from, &words[(from - row) * self.width..(to - row) * self.width]);
        }
    }
}
