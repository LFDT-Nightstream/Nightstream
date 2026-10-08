//! The final verifier, written once over `neo_spartan::Backend`.
//!
//! Owns: the order of the terminal checks of a `FinalProof` against an
//! expected state: the statement and its state hash, the canonical child
//! split, PiCCS, PiRLC, then the layer-1 argument for the parent claim.
//! `Native` reads it as `verify_final`; the shrink layer reads it as rows.
//!
//! Invariant: every derived value (the fresh public input, the PiCCS output
//! instances, the PiRLC parent) is recomputed here. The proof's copies of
//! them are not read and have no authority.

mod k;
mod pi_ccs;
mod pi_rlc;
mod state;
mod words;

#[cfg(test)]
#[path = "../../../tests/lifecycle_native/terminal_program.rs"]
mod tests;

use neo_ccs::SparsePoly;
use neo_math::{D, F};
use neo_spartan::{Backend, ClaimWords, Error, FoldTranscript, Relation};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

use self::pi_rlc::Source;
use self::state::State;
use self::words::FinalWords;
use super::{FinalProof, Stage1State};

/// The final verifier of one circuit for one expected state. With no proof
/// it reads zeros: a shape run.
pub(crate) struct FinalProgram<'a> {
    /// The verifier context digest of the circuit.
    pub(crate) context: [u64; 4],
    /// The CCS polynomial.
    pub(crate) f: &'a SparsePoly<F>,
    /// The decomposition base `b`.
    pub(crate) base: u32,
    pub(crate) relation: &'a Relation<'a>,
    pub(crate) state: &'a Stage1State,
    pub(crate) proof: Option<&'a FinalProof>,
}

impl FinalProgram<'_> {
    /// The statement words: the iteration, `z0` and the current state.
    pub(crate) fn statement(&self) -> Vec<u64> {
        let mut words = vec![self.state.iteration()];
        words.extend(self.state.z0().map(|word| word.as_canonical_u64()));
        words.extend(self.state.current().map(|word| word.as_canonical_u64()));
        words
    }

    pub(crate) fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error> {
        let statement: Vec<B::F> = self
            .statement()
            .into_iter()
            .map(|word| b.public(word))
            .collect();
        let state = State {
            iteration: statement[0],
            z0: std::array::from_fn(|i| statement[1 + i]),
            current: std::array::from_fn(|i| statement[5 + i]),
        };
        // An active proof has a positive iteration.
        let inverse = b.hint(&[state.iteration], 1, &|values| {
            vec![F::from_u64(values[0])
                .try_inverse()
                .map_or(0, |v| v.as_canonical_u64())]
        })[0];
        let product = b.mul(state.iteration, inverse);
        let one = b.constant(1);
        b.assert_equal(product, one, "positive iteration")?;

        let words = FinalWords::read(b, self.proof)?;
        let parent_public = state::canonical_parent(b, &words.digits)?;
        let digest = state::state_hash(b, self.context, &state, &words, &parent_public);
        let fresh_public = state::encode(b, digest)?;

        let mut transcript = FoldTranscript::new(b);
        let point = pi_ccs::replay(
            b,
            &mut transcript,
            &pi_ccs::Inputs {
                f: self.f,
                base: self.base,
                digest,
                fresh_commitment: &words.fresh,
                fresh_public: &fresh_public,
                running: &words.evaluations,
                prior_point: &words.point,
                rounds: &words.rounds,
                outputs: &words.outputs,
            },
        )?;

        let columns = |public: &[B::F]| -> Vec<[B::F; D]> {
            public
                .chunks_exact(D)
                .map(|c| std::array::from_fn(|lane| c[lane]))
                .collect()
        };
        let mut sources = vec![Source {
            commitment: &words.fresh,
            public: columns(&fresh_public),
            evaluations: &words.outputs[0],
        }];
        for (i, digits) in words.digits.iter().enumerate() {
            sources.push(Source {
                commitment: &words.commitments[i],
                public: columns(digits),
                evaluations: &words.outputs[1 + i],
            });
        }
        let parent = pi_rlc::parent(b, &mut transcript, &sources)?;
        let claim = ClaimWords {
            commitment: parent.commitment,
            public: parent.public,
            point,
            eval_k: parent.eval_k,
            eval_a: parent.eval_a,
        };
        neo_spartan::verify_with(
            b,
            self.relation,
            transcript,
            &claim,
            self.proof.map(|proof| &proof.layer1),
        )
    }
}
