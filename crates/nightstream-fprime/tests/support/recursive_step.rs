//! Bind the recursive caller packet to the checked prior proof and children.
//! The complete independent row evaluator runs after these input checks.

use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde_json::{json, Value};

const PI_CCS_V1_1_ROUND_COUNT: usize = 28;
const STATE_PREIMAGE_WORDS: usize = 49_393;
const PUBLIC: usize = 270;
const MODULUS: u64 = 0xffff_ffff_0000_0001;

#[path = "pi_ccs_parent.rs"]
mod parent;

fn read(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).expect("recursive caller JSON")
}

fn words(value: &Value) -> Vec<u64> {
    match value {
        Value::Array(values) => values.iter().flat_map(words).collect(),
        Value::Number(value) => {
            let word = value.as_u64().expect("unsigned field word");
            assert!(word < MODULUS, "canonical field encoding");
            vec![word]
        }
        _ => panic!("numeric array encoding"),
    }
}

fn hash(values: &[u64]) -> [u64; 4] {
    poseidon2_hash(
        &values
            .iter()
            .map(|&word| Goldilocks::from_u64(word))
            .collect::<Vec<_>>(),
    )
    .map(|word| word.as_canonical_u64())
}

fn encode(digest: [u64; 4]) -> Vec<u64> {
    let mut result = vec![1];
    for word in digest {
        result.extend((0..u64::BITS).map(|bit| (word >> bit) & 1));
    }
    result.resize(PUBLIC, 0);
    result
}

pub fn check_fixture(fixture: &[u8], base: &[u8], input: &[u8], children: &[u8], previous: &[u8], folded: &[u8]) {
    let fixture = read(fixture);
    let base = read(base);
    let input = read(input);
    let children = read(children);
    let previous = read(previous);
    let folded = read(folded);
    assert_eq!(fixture.as_array().expect("caller packet").len(), 5);
    assert_eq!(fixture[0], 1);
    assert_eq!(base[0], 1);
    assert_eq!(input[0], 2);
    assert_eq!(previous[0], 1);
    assert_eq!(folded[0], 1);
    assert_eq!(fixture[1], base[1]);
    assert_eq!(fixture[1], folded[2]);
    assert_eq!(previous[1], input, "exact preceding PiCCS input and proof");
    assert_eq!(previous[5][0], 1, "preceding acceptance record");
    assert_eq!(children[0], previous[5][6], "exact preceding output point");
    assert_eq!(children[0], folded[12]);
    let private = words(&fixture[2]);
    let public = words(&fixture[3]);
    let base_private = words(&base[2]);
    assert_eq!(private.len(), 177_326);
    assert_eq!(public.len(), 278);
    let prior = &private[..STATE_PREIMAGE_WORDS];
    let output = &private[STATE_PREIMAGE_WORDS..2 * STATE_PREIMAGE_WORDS];
    assert_eq!(
        prior,
        &base_private[STATE_PREIMAGE_WORDS..2 * STATE_PREIMAGE_WORDS],
        "actual base output is the next prior"
    );
    assert_eq!(prior[28], 1, "recursive branch");
    assert_eq!(
        parent::with_running(prior, &input[6]),
        prior,
        "same PiCCS running input"
    );
    assert_eq!(words(&input[2]), encode(hash(prior)), "same fresh public input");
    let message = &private[private.len() - 4..];
    assert_eq!(message, &base_private[base_private.len() - 4..]);
    let mut application = b"Nightstream/Stage1/Poseidon2HashChain/v1"
        .iter()
        .map(|&byte| u64::from(byte))
        .collect::<Vec<_>>();
    application.extend_from_slice(&prior[35..39]);
    application.extend_from_slice(message);
    let application_output = hash(&application);
    let mut next = prior.to_vec();
    next[28] = 2;
    next[35..39].copy_from_slice(&application_output);
    assert_eq!(
        output,
        parent::with_running(&next, &children),
        "actual application and all child claims in the output preimage"
    );
    let digest = hash(output);
    assert_eq!(words(&fixture[4][0]), application_output);
    assert_eq!(words(&fixture[4][1]), digest);
    assert_eq!(words(&fixture[4][2]), encode(digest));
    assert_eq!(public[..PUBLIC], encode(hash(prior)));
    assert_eq!(public[PUBLIC..PUBLIC + 4], digest);
    assert_eq!(public[PUBLIC + 4..], words(&fixture[1]));
    let mut proof_words = words(&input[1]);
    proof_words.extend(words(&input[3]));
    // PiCCSProofInputs.serializeOutput places K then A within each source.
    assert_eq!(input[4].as_array().expect("PiCCS Eval_K sources").len(), 17);
    assert_eq!(input[5].as_array().expect("PiCCS Eval_A sources").len(), 17);
    for source in 0..17 {
        proof_words.extend(words(&input[4][source]));
        proof_words.extend(words(&input[5][source]));
    }
    let mut expected_private = prior.to_vec();
    expected_private.extend_from_slice(output);
    expected_private.extend(proof_words);
    for index in [1, 3, 4, 2] {
        expected_private.extend(words(&children[index]));
    }
    expected_private.extend_from_slice(message);
    assert_eq!(private.len(), expected_private.len(), "complete caller slots");
    for (index, (actual, expected)) in private.iter().zip(&expected_private).enumerate() {
        assert_eq!(actual, expected, "proof and child caller slot {index}");
    }
    assert_eq!(fixture[4][3], children[0]);
    assert_eq!(fixture[4][4], previous[5][14]);
    assert_eq!(fixture[4][4], folded[5]);
    assert_eq!(fixture[4][5], folded[7]);
    let centered: Vec<i64> = serde_json::from_value(folded[10].clone()).expect("folded public integers");
    assert!(
        centered.iter().all(|value| value.unsigned_abs() < 1 << 16),
        "strict parent bound"
    );
    let parent_public = centered
        .iter()
        .map(|&value| {
            if value < 0 {
                MODULUS - value.unsigned_abs()
            } else {
                value as u64
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(fixture[4][6], json!(parent_public));
    println!("recursive_caller_binding=passed prior_iteration=1 output_iteration=2 children=16 matrix_families=14");
}
