//! Primitive conformance: the Rust Poseidon2 permutation and sponge hash must
//! equal the Lean reference `NightstreamFPrime.Spec.Poseidon2` on fixed vectors.
//! Expected values are the `#eval` output of the Lean definitions
//! (`formal/nightstream-fprime/NightstreamFPrime/Spec/Poseidon2.lean`).

use neo_ccs::crypto::poseidon2_goldilocks as p2;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

fn u64s(xs: &[Goldilocks]) -> Vec<u64> {
    xs.iter().map(|x| x.as_canonical_u64()).collect()
}

#[test]
fn permutation_matches_lean_reference() {
    let state: [Goldilocks; 8] = core::array::from_fn(|i| Goldilocks::from_u64(i as u64));
    assert_eq!(
        u64s(&p2::permute_state(state)),
        [
            5488585136800735649,
            15788908966630034145,
            16079255763572629144,
            18141420471353752734,
            15609380996481485392,
            2967186249397356049,
            3405496945520602869,
            15650816165326155396
        ]
    );
}

#[test]
fn sponge_hash_matches_lean_reference() {
    assert_eq!(
        u64s(&p2::poseidon2_hash(&[1u64, 2, 3].map(Goldilocks::from_u64))),
        [
            11606899037157615988,
            14420500567323833487,
            14641874967252747720,
            5178821719885745993
        ]
    );
    assert_eq!(
        u64s(&p2::poseidon2_hash(&[Goldilocks::from_u64(42); 10])),
        [
            8060843777812792290,
            6964801764726245452,
            2244548803069042284,
            4702553374630865470
        ]
    );
    assert_eq!(
        u64s(&p2::poseidon2_hash(&[])),
        [
            18381645674097109552,
            17323949295411050332,
            1008456283436807093,
            14351770501940631280
        ]
    );
}
