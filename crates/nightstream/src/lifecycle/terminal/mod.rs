//! The final verifier, written once over `neo_spartan::Backend`.
//!
//! Owns: the order of the terminal checks of a `FinalProof` against an
//! expected state: the statement, the Lean terminal package (prior-state
//! hash, PiCCS, PiRLC), then the layer-1 argument for the parent claim.
//! `Native` reads it as `verify_final`; the shrink layer reads it as rows.
//!
//! Invariant: the parent claim and the transcript that layer 1 reads are the
//! package's output columns, which its rows fix. The proof carries only the
//! package inputs and the layer-1 proof.

mod package;

#[cfg(test)]
pub(crate) use package::IDENTIFIER as TERMINAL_PACKAGE_IDENTIFIER;
pub(crate) use package::{package, preload, Package};

use neo_math::F;
use neo_spartan::{Backend, Error, Relation};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

use super::finish::TerminalProof;
use super::Stage1State;

/// The final verifier of one circuit for one expected state. With no proof
/// it reads zeros: a shape run.
pub(crate) struct FinalProgram<'a> {
    /// The verifier context digest of the circuit.
    pub(crate) context: [u64; 4],
    pub(crate) package: &'a Package,
    pub(crate) relation: &'a Relation<'a>,
    pub(crate) state: &'a Stage1State,
    pub(crate) proof: Option<&'a TerminalProof>,
}

impl neo_spartan::shrink::Program for FinalProgram<'_> {
    /// The layer-1 relation (its key, shape and WHIR profiles, which fix
    /// the bound and the security bits) and the circuit's context.
    fn key(&self) -> Vec<u64> {
        let mut words = self.relation.key_words();
        words.extend(self.context);
        words
    }

    /// The statement words: the iteration, `z0` and the current state.
    fn statement(&self) -> Vec<u64> {
        let mut words = vec![self.state.iteration()];
        words.extend(self.state.z0().map(|word| word.as_canonical_u64()));
        words.extend(self.state.current().map(|word| word.as_canonical_u64()));
        words
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error> {
        let statement: Vec<B::F> = self
            .statement()
            .into_iter()
            .map(|word| b.public(word))
            .collect();
        // An active proof has a positive iteration.
        let inverse = b.hint(&[statement[0]], 1, &|values| {
            vec![F::from_u64(values[0])
                .try_inverse()
                .map_or(0, |v| v.as_canonical_u64())]
        })[0];
        let product = b.mul(statement[0], inverse);
        let one = b.constant(1);
        b.assert_equal(product, one, "positive iteration")?;

        let (claim, transcript) =
            self.package
                .apply(b, &statement, self.context, self.proof.map(|proof| &proof.inputs))?;
        neo_spartan::verify_with(
            b,
            self.relation,
            transcript,
            &claim,
            self.proof.map(|proof| &proof.layer1),
        )
    }
}
