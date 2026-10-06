//! Canonical parent-preimage input for the PiCCS conformance tests.
//! This assembles caller data in Lifecycle.XOut.serializePreimage order and
//! the PiCCS prior child region; the hash and canonical rows must separately
//! validate that data.

use super::{PI_CCS_V1_1_ROUND_COUNT, STATE_PREIMAGE_WORDS};

const RUNNING_COUNT: usize = 16;
const MATRIX_COUNT: usize = 4;
const PRIOR_PUBLIC_WORDS: usize = 270;
const DOMAIN_WORDS: usize = 12;
const TAIL_WORDS: usize = 13;
const MODULUS: u128 = 0xffff_ffff_0000_0001;

struct Running {
    point: Vec<[u64; 2]>,
    commitments: Vec<Vec<u64>>,
    public: Vec<Vec<u64>>,
    eval_k: Vec<Vec<[u64; 2]>>,
    eval_a: Vec<Vec<Vec<[u64; 2]>>>,
}

fn parse(running: &serde_json::Value) -> Running {
    let fields = running.as_array().expect("schema-2 running statement");
    assert_eq!(fields.len(), 5);
    let running = Running {
        point: serde_json::from_value(fields[0].clone()).expect("running point"),
        commitments: serde_json::from_value(fields[1].clone()).expect("running commitments"),
        public: serde_json::from_value(fields[2].clone()).expect("running public inputs"),
        eval_k: serde_json::from_value(fields[3].clone()).expect("running Eval_K"),
        eval_a: serde_json::from_value(fields[4].clone()).expect("running Eval_A"),
    };
    assert_eq!(running.point.len(), PI_CCS_V1_1_ROUND_COUNT);
    for count in [
        running.commitments.len(),
        running.public.len(),
        running.eval_k.len(),
        running.eval_a.len(),
    ] {
        assert_eq!(count, RUNNING_COUNT);
    }
    for source in 0..RUNNING_COUNT {
        assert_eq!(running.commitments[source].len(), 1_188);
        assert_eq!(running.public[source].len(), PRIOR_PUBLIC_WORDS);
        assert_eq!(running.eval_k[source].len(), 54);
        assert_eq!(running.eval_a[source].len(), MATRIX_COUNT);
        assert!(running.eval_a[source]
            .iter()
            .all(|matrix| matrix.len() == 54));
    }
    running
}

/// Lean `parentPublic`: `Σ_j 2^j x_j` for each public-input column.
fn parent(running: &Running) -> Vec<u64> {
    (0..PRIOR_PUBLIC_WORDS)
        .map(|column| {
            let mut value = 0_u128;
            for (child, public) in running.public.iter().enumerate() {
                value = (value + ((public[column] as u128) << child) % MODULUS) % MODULUS;
            }
            value as u64
        })
        .collect()
}

/// Replace the running instance of `base` and keep its domain chunk and tail.
pub fn with_running(base: &[u64], running: &serde_json::Value) -> Vec<u64> {
    assert_eq!(base.len(), STATE_PREIMAGE_WORDS);
    let running = parse(running);
    let mut words = base[..DOMAIN_WORDS].to_vec();
    words.extend(running.commitments.iter().flatten().copied());
    words.extend(running.eval_k.iter().flatten().flatten().copied());
    words.extend(running.eval_a.iter().flatten().flatten().flatten().copied());
    words.extend(running.point.iter().flatten().copied());
    for lanes in parent(&running).chunks_exact(3) {
        let packed = (lanes[0] as u128 + ((lanes[1] as u128) << 17) + ((lanes[2] as u128) << 34) % MODULUS) % MODULUS;
        words.push(packed as u64);
    }
    words.extend_from_slice(&base[STATE_PREIMAGE_WORDS - TAIL_WORDS..]);
    assert_eq!(words.len(), STATE_PREIMAGE_WORDS);
    words
}

/// The PiCCS prior child region: the child public inputs, child-major, then
/// one sign bit per parent coordinate, one when the parent is negative.
pub fn prior_children(running: &serde_json::Value) -> Vec<u64> {
    let running = parse(running);
    let mut words: Vec<u64> = running.public.iter().flatten().copied().collect();
    words.extend(
        parent(&running)
            .into_iter()
            .map(|value| u64::from(value as u128 > MODULUS / 2)),
    );
    words
}
