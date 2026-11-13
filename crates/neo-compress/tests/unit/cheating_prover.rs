//! Provers that commit a false lookup and then follow the protocol, and
//! lookup challenges that are table values.

#[path = "../support/mod.rs"]
mod support;

use neo_math::{from_complex, F, K};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use support::{instances, Instance};

use super::{commit, table_inverses, Committed};
use crate::layout::{Layout, LOOKUPS};
use crate::weights::{self, Challenges};
use crate::{akita, verify, Error, Proof, Statement};

fn statements(instances: &[Instance]) -> Vec<Statement<'_>> {
    instances.iter().map(Instance::statement).collect()
}

fn layout(instances: &[Instance]) -> Layout {
    let claims: Vec<_> = instances
        .iter()
        .map(|instance| (instance.witness.cols(), instance.norm))
        .collect();
    Layout::new(&claims).expect("layout")
}

/// The verifier's answer to the proof of a prover that applies `change`
/// after it committed the honest `W`.
fn verdict(change: impl FnOnce(&mut Committed)) -> Result<(), Error> {
    let instances = instances();
    let witnesses: Vec<_> = instances.iter().map(|instance| &instance.witness).collect();
    let mut committed = commit(layout(&instances), &witnesses).expect("commitment");
    change(&mut committed);
    let (proof, _): (Proof, _) = committed
        .prove(&mut Poseidon2Transcript::new_v1_2(), &statements(&instances))
        .expect("the prover does not check its tables");
    verify(&mut Poseidon2Transcript::new_v1_2(), &statements(&instances), &proof).map(|_| ())
}

/// Two cells of the first table's witness with different codes.
fn cells_with_different_codes(committed: &Committed) -> (usize, usize) {
    let (_, slice, _, _) = committed.layout.tables().next().expect("a table");
    let first = slice.offset;
    let second = slice
        .range()
        .find(|&cell| committed.codes[cell] != committed.codes[first])
        .expect("a cell with another value");
    (first, second)
}

#[test]
fn the_prover_of_these_tests_makes_an_accepted_proof() {
    verdict(|_| ()).expect("the honest proof verifies");
}

/// `W` holds counts that move one cell from its value to another value. The
/// number of counted cells does not change.
#[test]
fn wrong_counts_do_not_verify() {
    let result = verdict(|committed| {
        let (_, _, _, counts) = committed.layout.tables().next().expect("a table");
        let (first, second) = cells_with_different_codes(committed);
        let count = |cell: usize| counts.offset + usize::from(committed.codes[cell]);
        let (from, to) = (count(first), count(second));
        committed.witness[from] -= F::ONE;
        committed.witness[to] += F::ONE;
        let cells = committed.witness.iter().copied().map(akita::base).collect();
        committed.committed = committed
            .akita
            .commit(committed.layout.witness_vars, cells)
            .expect("commitment");
    });
    assert!(matches!(result, Err(Error::Rejected(_))), "{result:?}");
}

/// Two cells with different values exchange their inverses. Each column of
/// `V` keeps its sum, and no inverse of the two cells is correct.
#[test]
fn inverses_with_the_correct_sum_on_the_wrong_cells_do_not_verify() {
    let result = verdict(|committed| {
        let (first, second) = cells_with_different_codes(committed);
        committed.codes.swap(first, second);
    });
    assert!(matches!(result, Err(Error::Rejected(_))), "{result:?}");
}

/// The tables of the instances are `[-2176, 2176]` and `[-300, 300]`.
#[test]
fn a_lookup_challenge_in_a_table_gives_no_proof_and_no_acceptance() {
    let instances = instances();
    let (statements, layout) = (statements(&instances), layout(&instances));
    let extension = |seed: u64| from_complex(F::from_u64(seed), F::from_u64(seed + 1));
    let outside = F::from_u64(2177);
    let run = |tau: F| {
        let mut taus = [outside; LOOKUPS];
        taus[LOOKUPS - 1] = tau;
        let tables = layout.tables().count();
        let challenges = Challenges {
            taus,
            omega: extension(2),
            delta: extension(4),
            gamma: extension(6),
            mu: extension(8),
            psi: std::array::from_fn(|k| extension(10 + k as u64)),
            xi: (0..tables as u64)
                .map(|table| extension(20 + table))
                .collect(),
            lambda: (0..tables as u64)
                .map(|table| extension(30 + table))
                .collect(),
            beta: (0..layout.span_vars as u64)
                .map(|bit| extension(40 + bit))
                .collect(),
        };
        let rho: Vec<K> = (0..layout.witness_vars as u64)
            .map(|bit| extension(80 + bit))
            .collect();
        (
            table_inverses(&layout, &taus).map(|_| ()),
            weights::table(&statements, &layout, &challenges).map(|_| ()),
            weights::evaluate(&statements, &layout, &challenges, &rho).map(|_| ()),
        )
    };
    for tau in [
        F::ZERO,
        F::from_u64(300),
        -F::from_u64(300),
        F::from_u64(301),
        -F::from_u64(2176),
    ] {
        let (inverses, prover, verifier) = run(tau);
        assert!(matches!(inverses, Err(Error::Challenge)), "inverses, {tau}");
        assert!(matches!(prover, Err(Error::Challenge)), "prover weights, {tau}");
        assert!(matches!(verifier, Err(Error::Challenge)), "verifier weights, {tau}");
    }
    for tau in [outside, -outside] {
        let (inverses, prover, verifier) = run(tau);
        assert!(inverses.is_ok() && prover.is_ok() && verifier.is_ok(), "{tau}");
    }
}
