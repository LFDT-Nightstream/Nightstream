use crate::support::{claim_of, instances, key, layout, statements, Instance, POINT_VARS};
use neo_ccs::Mat;
use neo_math::{from_complex, D, F, K};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::{commit, table_inverses, Committed};
use crate::akita::Polynomial;
use crate::layout::LOOKUPS;
use crate::weights::{self, Challenges, Uses};
use crate::{akita, verify, Error, Norm, Proof};

fn verdict(change: impl FnOnce(&mut Committed)) -> Result<(), Error> {
    let instances = instances();
    let witnesses: Vec<_> = instances.iter().map(|instance| &instance.witness).collect();
    let mut committed = commit(key(), layout(&instances), &witnesses).expect("commitment");
    change(&mut committed);
    let (proof, _): (Proof, _) = committed
        .prove(&mut Poseidon2Transcript::new_v1_2(), &statements(&instances))
        .expect("the prover does not check its tables");
    verify(
        &mut Poseidon2Transcript::new_v1_2(),
        key().verification_key(),
        &statements(&instances),
        &proof,
    )
    .map(|_| ())
}

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
fn signed_span_commitment_keeps_padded_slices_and_counts() {
    let layout = crate::Layout::new(
        &[(3, Norm::Table { bound: 1 }), (3, Norm::BoundByCaller)],
        key().verification_key().blocks(),
    )
    .unwrap();
    let (_, counts) = layout.cells[0].table.unwrap();
    let fresh = layout.cells[1].witness;
    let count_cell = counts.offset + 1;
    let count = F::from_u64(layout.cells[0].witness.len() as u64);
    let signs = ([0b101, 0, 1 << 63], [0b10, 1 << 63, 0]);

    let mut dense = vec![F::ZERO; layout.witness_len];
    dense[fresh.offset] = F::ONE;
    dense[fresh.offset + 1] = -F::ONE;
    dense[fresh.offset + 2] = F::ONE;
    dense[fresh.offset + 127] = -F::ONE;
    dense[fresh.offset + 191] = F::ONE;
    dense[count_cell] = count;
    let expected = key()
        .akita
        .commit(
            layout.shape(),
            Polynomial::Witness,
            dense.into_iter().map(akita::base).collect(),
        )
        .unwrap();
    let actual = key()
        .akita
        .commit_signed_witness(
            layout.shape(),
            fresh.offset,
            (&signs.0, &signs.1),
            (count_cell, akita::base(count)),
        )
        .unwrap();
    assert_eq!(actual.bytes, expected.bytes);
}

#[test]
fn the_prover_of_these_tests_makes_an_accepted_proof() {
    verdict(|_| ()).expect("the honest proof verifies");
}

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
            .key
            .akita
            .commit(committed.layout.shape(), Polynomial::Witness, cells)
            .expect("commitment");
    });
    assert!(matches!(result, Err(Error::Rejected(_))), "{result:?}");
}

#[test]
fn inverses_with_the_correct_sum_on_the_wrong_cells_do_not_verify() {
    let result = verdict(|committed| {
        let (first, second) = cells_with_different_codes(committed);
        std::sync::Arc::make_mut(&mut committed.codes).swap(first, second);
    });
    assert!(matches!(result, Err(Error::Rejected(_))), "{result:?}");
}

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
            beta: (0..layout.table_witness_vars as u64)
                .map(|bit| extension(40 + bit))
                .collect(),
        };
        let rho: Vec<K> = (0..layout.witness_vars as u64)
            .map(|bit| extension(80 + bit))
            .collect();
        let uses = Uses::new(&layout, &challenges, &rho);
        (
            table_inverses(&layout, &taus).map(|_| ()),
            weights::table(&statements, &layout, &challenges, key(), false).map(|_| ()),
            weights::closed_part(&statements, &layout, &challenges, &uses, &rho).map(|_| ()),
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

#[test]
fn a_virtual_zero_table_has_the_dense_proof() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x7a65_726f);
    for (blocks, point) in [(61, K::ZERO), (64, from_complex(F::from_u64(3), F::from_u64(7)))] {
        let mut zero = Instance::random(&mut rng, blocks, Norm::Table { bound: 52 });
        zero.witness = Mat::virtual_constant(D, blocks, F::ZERO);
        zero.claim = claim_of(&zero.rows, &zero.witness, &[point; POINT_VARS]);
        let dense_zero = Mat::from_row_major(D, blocks, zero.witness.to_dense_vec());
        let fresh = Instance::random(&mut rng, blocks, Norm::BoundByCaller);
        let instances = [zero, fresh];
        let statements = statements(&instances);
        let layout = layout(&instances);
        let prove = |zero: &Mat<F>| {
            let witnesses = [zero, &instances[1].witness];
            let committed = commit(key(), layout.clone(), &witnesses).unwrap();
            assert_eq!(committed.zero_prefix.is_some(), zero.virtual_constant_value().is_some());
            let mut transcript = Poseidon2Transcript::new_v1_2();
            committed.prove(&mut transcript, &statements).unwrap().0
        };
        let proof = prove(&instances[0].witness);
        assert_eq!(proof, prove(&dense_zero));
        verify(
            &mut Poseidon2Transcript::new_v1_2(),
            key().verification_key(),
            &statements,
            &proof,
        )
        .unwrap();
    }
}
