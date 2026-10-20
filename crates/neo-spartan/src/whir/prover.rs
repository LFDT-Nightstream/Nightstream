//! Our WHIR prover for the stacked suffix layout: Plonky3 0.8's `commit`
//! and `open_at`, with the stacked codeword committed one coset at a time.
//!
//! Owns: the order of the prover's transcript steps, which is Plonky3's (the
//! same transcript and layout calls in the same order), so commitments and
//! proofs are byte-identical to Plonky3's (test
//! `tests/internal/whir_prover.rs`); the coset encoding of the codeword; and
//! the prover data between a commitment and its opening. Does not own: the
//! configuration (`pcs.rs`), the Merkle tree (`merkle.rs`), the rounds of
//! the opening and the layout arithmetic (Plonky3), or verification
//! (`whir/mod.rs`).
//!
//! Row `k` of a codeword of `n` rows is `Σ_i ω^{ik}·row_i` over the message
//! rows, with `ω` of order `n`. Its rows `a, a + c, a + 2c, ...` (coset `a`
//! of `c`) are the DFT of length `n / c` of the message twisted by
//! `ω^{ia}` and folded modulo `n / c`. Each coset goes straight into the
//! tree, so the stacked codeword is never whole in memory.

use p3_dft_v08::TwoAdicSubgroupDft;
use p3_field_v08::{PrimeCharacteristicRing, TwoAdicField};
use p3_matrix_v08::dense::RowMajorMatrix;
use p3_multilinear_util_v08::point::Point;
use p3_sumcheck_v08::layout::{observe_commitment, plan_stacked_layout, Layout as _, Witness};
use p3_sumcheck_v08::OpeningProtocol;
use p3_whir::pcs::proof::PcsProof;
use rayon::prelude::*;

use crate::field::{Ext, Gl};
use crate::hash::Challenger;
use crate::merkle::{self, Mmcs};
use crate::pcs::{Commitment, Config, Dft, Layout, Opening, Whir};
use crate::scratch;

/// Cosets per codeword: one coset, a sixteenth of the codeword, is in memory
/// at a time.
const COSETS: usize = 16;

type Tree = merkle::Tree<RowMajorMatrix<Gl>>;

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
    whir: &Whir,
    data: ProverData,
    protocol: &OpeningProtocol,
    points: &[Point<Ext>],
    challenger: &mut Challenger,
) -> Opening {
    let claims = protocol
        .checked_num_claims()
        .and_then(|claims| claims.checked_add(whir.commitment_ood_samples()))
        .expect("the claim count fits");
    whir.validate_initial_claims(claims)
        .expect("the plan fixes the claims");
    assert_eq!(protocol.num_openings(), points.len(), "one point per batch");
    let ProverData { mut layout, tree } = data;
    let initial_ood_answers = (0..whir.commitment_ood_samples())
        .map(|_| layout.add_virtual_eval(challenger))
        .collect();
    let evals = protocol
        .iter_openings()
        .zip(points)
        .map(|((table, batch), point)| layout.eval_at(table, batch, point, challenger))
        .collect();
    let proof = whir
        .prove(initial_ood_answers, challenger, layout, tree, protocol.num_openings())
        .expect("the plan fixes the claims");
    PcsProof { whir: proof, evals }
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
