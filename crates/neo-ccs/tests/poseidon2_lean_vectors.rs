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
    let state: [Goldilocks; 16] = core::array::from_fn(|i| Goldilocks::from_u64(i as u64));
    assert_eq!(
        u64s(&p2::permute_state(state)),
        [
            2765442853011447597,
            7143063655483819115,
            13081757537038180257,
            7227677949497923249,
            3439930679990181944,
            5758471765018571805,
            15963820721376813919,
            12462245079697881918,
            2208502778189471335,
            11034615862719689705,
            10550941392938205181,
            9937460284766256728,
            10826235081420509452,
            15876015485749169348,
            1075711347785640356,
            10097266676967410534,
        ]
    );
}

#[test]
fn sponge_hash_matches_lean_reference() {
    assert_eq!(
        u64s(&p2::poseidon2_hash(&[1u64, 2, 3].map(Goldilocks::from_u64))),
        [
            10174139661590874945,
            682071588285257240,
            2146088507292232401,
            7825976598430147546,
        ]
    );
    assert_eq!(
        u64s(&p2::poseidon2_hash(&[Goldilocks::from_u64(42); 10])),
        [
            14573620483189275490,
            47946682125460792,
            1809112248114747742,
            2910711220625124551,
        ]
    );
    assert_eq!(
        u64s(&p2::poseidon2_hash(&[])),
        [
            8278139043587206520,
            10537167997139471771,
            12238973609469370520,
            4874705429328367957,
        ]
    );
    // Three rate-12 blocks, the last one partial.
    let input: Vec<Goldilocks> = (0..25).map(Goldilocks::from_u64).collect();
    assert_eq!(
        u64s(&p2::poseidon2_hash(&input)),
        [
            12330712987123350331,
            8313004511586635200,
            624835263297322767,
            13941719899968475042,
        ]
    );
}
