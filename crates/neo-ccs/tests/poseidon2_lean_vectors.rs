//! Primitive conformance: the Rust Poseidon2 permutations must equal each known answer of
//! `scripts/poseidon2/vectors.json`, and the permutation and the sponge hash must equal the Lean
//! references `NightstreamFPrime.Spec.Poseidon2.permute` and `hash` on the inputs that
//! `Export/Poseidon2VectorsMain.lean` emits to
//! `formal/nightstream-fprime/artifacts/nightstream-fprime-poseidon2-vectors-v1.json`. ProtogaLattice
//! can check its own code against the same file. From `formal/nightstream-fprime`, emit them again with
//! `bash scripts/validate.sh lean-executable .lake/build/bin/emitPoseidon2Vectors
//! artifacts/nightstream-fprime-poseidon2-vectors-v1.json`.

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

fn words(value: &serde_json::Value) -> Vec<u64> {
    value
        .as_array()
        .expect("words")
        .iter()
        .map(|word| word.as_u64().expect("u64"))
        .collect()
}

fn fields(value: &serde_json::Value) -> Vec<Fq> {
    words(value).into_iter().map(Fq::from_u64).collect()
}

#[test]
fn permutation_and_sponge_hash_match_lean_cross_check_vectors() {
    let vectors: serde_json::Value = serde_json::from_str(include_str!(
        "../../../formal/nightstream-fprime/artifacts/nightstream-fprime-poseidon2-vectors-v1.json"
    ))
    .expect("Lean cross-check vectors");
    assert_eq!(vectors[0], Fq::ORDER_U64);
    assert_eq!(
        words(&vectors[1]),
        [
            p2::WIDTH as u64,
            p2::RATE as u64,
            p2::DIGEST_LEN as u64,
            2 * p2::HALF_FULL_ROUNDS as u64,
            p2::PARTIAL_ROUNDS as u64
        ]
    );
    let permutations = vectors[2].as_array().expect("permutations");
    assert_eq!(permutations.len(), 6);
    for vector in permutations {
        let input: [Fq; p2::WIDTH] = fields(&vector[0]).try_into().expect("a state");
        assert_eq!(
            u64s(&p2::permute_state(input)),
            words(&vector[1]),
            "permutation of {input:?}"
        );
    }
    let mut lengths = Vec::new();
    for vector in vectors[3].as_array().expect("hashes") {
        let input = fields(&vector[0]);
        assert_eq!(
            u64s(&p2::poseidon2_hash(&input)),
            words(&vector[1]),
            "length {}",
            input.len()
        );
        lengths.push(input.len());
    }
    assert!((0..=25).all(|length| lengths.contains(&length)));
}
