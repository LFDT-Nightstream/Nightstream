//! Canonical parent-preimage input for the PiCCS conformance tests.
//! This assembles caller data in Lifecycle.XOut.serializeRunning order;
//! the hash and canonical rows must separately validate that data.

use super::{PI_CCS_V1_1_ROUND_COUNT, STATE_PREIMAGE_WORDS};

const RUNNING_COUNT: usize = 16;
const MATRIX_COUNT: usize = 14;
const PRIOR_PUBLIC_WORDS: usize = 270;
const RUNNING_POINT_WORDS: usize = 2 * PI_CCS_V1_1_ROUND_COUNT;

pub fn with_running(base: &[u64], running: &serde_json::Value) -> Vec<u64> {
    // Lifecycle.XOut.serializePreimage places serializeRunning after the
    // fixed tag, key, iteration and two application-state blocks.
    assert_eq!(base.len(), STATE_PREIMAGE_WORDS);
    let fields = running.as_array().expect("schema-2 running statement");
    assert_eq!(fields.len(), 5);
    let point: Vec<[u64; 2]> = serde_json::from_value(fields[0].clone()).expect("running point");
    let commitments: Vec<Vec<u64>> = serde_json::from_value(fields[1].clone()).expect("running commitments");
    let public: Vec<Vec<u64>> = serde_json::from_value(fields[2].clone()).expect("running public inputs");
    let eval_k: Vec<Vec<[u64; 2]>> = serde_json::from_value(fields[3].clone()).expect("running Eval_K");
    let eval_a: Vec<Vec<Vec<[u64; 2]>>> = serde_json::from_value(fields[4].clone()).expect("running Eval_A");
    assert_eq!(point.len(), PI_CCS_V1_1_ROUND_COUNT);
    for count in [commitments.len(), public.len(), eval_k.len(), eval_a.len()] {
        assert_eq!(count, RUNNING_COUNT);
    }
    let mut words = base[..39].to_vec();
    words.push(RUNNING_POINT_WORDS as u64);
    words.extend(point.into_iter().flatten());
    for source in 0..RUNNING_COUNT {
        assert_eq!(commitments[source].len(), 1_188);
        assert_eq!(public[source].len(), PRIOR_PUBLIC_WORDS);
        assert_eq!(eval_k[source].len(), 54);
        assert_eq!(eval_a[source].len(), MATRIX_COUNT);
        assert!(eval_a[source].iter().all(|matrix| matrix.len() == 54));
        words.push(commitments[source].len() as u64);
        words.extend_from_slice(&commitments[source]);
        words.push(public[source].len() as u64);
        words.extend_from_slice(&public[source]);
        words.push(((MATRIX_COUNT + 1) * 54 * 2) as u64);
        words.extend(eval_k[source].iter().flatten().copied());
        words.extend(eval_a[source].iter().flatten().flatten().copied());
    }
    words.push(base[STATE_PREIMAGE_WORDS - 1]);
    assert_eq!(words.len(), STATE_PREIMAGE_WORDS);
    words
}
