//! Primitive conformance: the Rust Poseidon2 permutations must equal each known answer of
//! `scripts/poseidon2/vectors.json`, and the sponge hash must equal the Lean reference
//! `NightstreamFPrime.Spec.Poseidon2` on fixed vectors.

use neo_ccs::crypto::poseidon2 as p2;
use neo_math::Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn u64s(xs: &[Fq]) -> Vec<u64> {
    xs.iter().map(|x| x.as_canonical_u64()).collect()
}

fn state<const T: usize>(vector: &serde_json::Value, key: &str) -> [Fq; T] {
    let words = vector[key].as_array().expect("state");
    assert_eq!(words.len(), T);
    core::array::from_fn(|lane| Fq::new(words[lane].as_str().expect("decimal").parse().expect("u64")))
}

#[test]
fn permutations_match_every_known_answer() {
    let vectors: serde_json::Value =
        serde_json::from_str(include_str!("../../../scripts/poseidon2/vectors.json")).expect("vectors.json");
    assert_eq!(vectors["q"], Fq::ORDER_U64.to_string());
    let mut checked = [0, 0];
    for vector in vectors["vectors"].as_array().expect("vectors") {
        let name = &vector["name"];
        match vector["t"].as_u64() {
            Some(16) => {
                assert_eq!(
                    p2::permute_state(state(vector, "input")),
                    state(vector, "output"),
                    "{name}"
                );
                assert_eq!(
                    p2::INSTANCE_16.permute(state(vector, "input")),
                    state(vector, "output"),
                    "{name}"
                );
                checked[0] += 1;
            }
            Some(12) => {
                assert_eq!(
                    p2::INSTANCE_12.permute(state::<12>(vector, "input")),
                    state(vector, "output"),
                    "{name}"
                );
                checked[1] += 1;
            }
            width => panic!("unexpected width {width:?}"),
        }
    }
    assert_eq!(checked, [5, 5]);
}

#[test]
fn sponge_hash_matches_lean_reference() {
    assert_eq!(
        u64s(&p2::poseidon2_hash(&[1u64, 2, 3].map(Fq::from_u64))),
        [
            10174139661590874945,
            682071588285257240,
            2146088507292232401,
            7825976598430147546,
        ]
    );
    assert_eq!(
        u64s(&p2::poseidon2_hash(&[Fq::from_u64(42); 10])),
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
    let input: Vec<Fq> = (0..25).map(Fq::from_u64).collect();
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
