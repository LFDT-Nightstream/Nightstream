//! Load the checked opening prefixes and produce the honest scalar rounds.
//! Full ring output families remain a separate construction step.

use std::{
    fs,
    io::{BufReader, Read},
    path::Path,
    time::Instant,
};

use neo_ccs::{SparsePoly, Term};
use neo_math::{from_complex, KExtensions, F, K};
use neo_reductions::engines::{pi_ccs_joint::ProtocolTrace, pi_ccs_joint_protocol::prove_phase};
use neo_transcript::Poseidon2Transcript;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;
use serde_json::{json, Value};

use super::folded_opening::{child_assignments, is_zero};
use super::oracle::{Oracle, LIVE_MATRICES, MATRICES, ROUNDS};

const MODULUS: u64 = 0xffff_ffff_0000_0001;

// Numeric cache schema, destructured once into named values below.
type Opening = (
    u64,
    [u64; 4],
    [u64; 4],
    usize,
    usize,
    usize,
    usize,
    usize,
    Vec<(u64, Vec<u32>)>,
    Vec<u64>,
    Vec<u64>,
);

fn field(word: u64) -> F {
    assert!(word < MODULUS, "canonical field word");
    F::from_u64(word)
}

fn extension(words: [u64; 2]) -> K {
    from_complex(field(words[0]), field(words[1]))
}

fn words(value: K) -> [u64; 2] {
    value.to_limbs_u64().into()
}

pub fn generate(
    cache: &Path,
    lean_prelude: &Path,
    running_prefix: Option<&Path>,
    folded_children: Option<&Path>,
    output: &Path,
) {
    let started = Instant::now();
    assert!(
        folded_children.is_none() || running_prefix.is_some(),
        "child openings require their linear evaluation prefix"
    );
    assert!(!output.exists(), "use a fresh external round output");
    let (
        schema,
        identity,
        context,
        logical_width,
        carrier_width,
        row_count,
        matrix_count,
        round_count,
        terms,
        public,
        commitment,
    ): Opening = serde_json::from_slice(&fs::read(cache.join("opening.json")).expect("checked opening metadata"))
        .expect("opening cache schema");
    assert_eq!((schema, matrix_count, round_count), (1, MATRICES, ROUNDS));
    assert_eq!(carrier_width, logical_width.div_ceil(54) * 54);
    assert!(carrier_width <= 1 << ROUNDS && row_count <= 1 << ROUNDS);
    assert_eq!((public.len(), commitment.len()), (270, 1_188));
    let polynomial = SparsePoly::new(
        MATRICES,
        terms
            .into_iter()
            .map(|(coefficient, exps)| {
                assert_eq!(exps.len(), MATRICES);
                Term {
                    coeff: field(coefficient),
                    exps,
                }
            })
            .collect(),
    );
    let prelude: Value = serde_json::from_slice(&fs::read(lean_prelude).expect("executable Lean prelude result"))
        .expect("Lean result JSON");
    assert_eq!(prelude[0], json!(1));
    assert_eq!(prelude[1][0], json!(2));
    assert_eq!(
        prelude[1][1],
        json!(commitment),
        "same fresh commitment as the checked opening"
    );
    assert_eq!(
        prelude[1][2],
        json!(public),
        "same public projection as the checked opening"
    );
    let alpha: Vec<[u64; 2]> = serde_json::from_value(prelude[5][1].clone()).expect("Lean alpha");
    let alpha = alpha.into_iter().map(extension).collect::<Vec<_>>();
    assert_eq!(alpha.len(), ROUNDS);
    let gamma = extension(serde_json::from_value(prelude[5][2].clone()).expect("Lean gamma"));
    let initial = extension(serde_json::from_value(prelude[5][7].clone()).expect("Lean initial claim"));
    let state: [u64; 8] = serde_json::from_value(prelude[5][3].clone()).expect("Lean pre-SumCheck state");
    let state = state.map(field);

    let carrier = fs::read(cache.join("carrier.i8")).expect("complete carrier prefix");
    assert_eq!(carrier.len(), carrier_width);
    assert!(
        carrier[logical_width..].iter().all(|&value| value == 0),
        "carrier alignment zeros"
    );
    assert_eq!(carrier[0], 1);
    let values = carrier
        .par_iter()
        .map(|value| match value {
            0 => K::ZERO,
            1 => K::ONE,
            255 => -K::ONE,
            _ => panic!("carrier coordinate is not a signed unit"),
        })
        .collect();
    drop(carrier);
    let image_path = cache.join("matrix-images.u64");
    assert_eq!(
        fs::metadata(&image_path).expect("matrix image size").len(),
        (row_count * LIVE_MATRICES * 8) as u64
    );
    let mut reader = BufReader::new(fs::File::open(image_path).expect("matrix image prefix"));
    let mut images = Vec::with_capacity(row_count);
    for _ in 0..row_count {
        let mut row = [0u8; LIVE_MATRICES * 8];
        reader
            .read_exact(&mut row)
            .expect("complete scalar matrix row");
        images.push(std::array::from_fn(|matrix| {
            K::from(field(u64::from_le_bytes(
                row[matrix * 8..matrix * 8 + 8]
                    .try_into()
                    .expect("field word width"),
            )))
        }));
    }
    println!("checked prefixes loaded: {:?}", started.elapsed());
    let mut oracle = Oracle::new(values, images, &polynomial, alpha.clone(), gamma);
    if let Some(path) = running_prefix {
        let prior_point: Vec<[u64; 2]> = serde_json::from_value(prelude[2][0].clone()).expect("running prior point");
        let prefix_length = carrier_width.max(row_count);
        assert_eq!(
            fs::metadata(path).expect("running prefix size").len(),
            (prefix_length * 16) as u64
        );
        let mut reader = BufReader::new(fs::File::open(path).expect("combined running evaluation prefix"));
        let mut running = Vec::with_capacity(prefix_length);
        for _ in 0..prefix_length {
            let mut bytes = [0u8; 16];
            reader
                .read_exact(&mut bytes)
                .expect("complete running prefix word");
            running.push(extension([
                u64::from_le_bytes(bytes[..8].try_into().expect("base limb width")),
                u64::from_le_bytes(bytes[8..].try_into().expect("base limb width")),
            ]));
        }
        let prior_point = prior_point.into_iter().map(extension).collect();
        oracle = if let Some(children) = folded_children {
            let sources = child_assignments(children, identity, context, logical_width, carrier_width, &prelude);
            oracle.with_distinct_running(running, prior_point, sources)
        } else {
            let norm_weight = K::ONE + gamma * super::running_prefix::sign_sum(&prelude, &commitment, &public, gamma);
            oracle.with_running(running, prior_point, norm_weight)
        };
    } else {
        assert!(
            is_zero(&prelude[2]),
            "nonzero running statements require running-rounds and their canonical prefix"
        );
    }
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(state, 0);
    let mut trace = ProtocolTrace {
        alpha,
        gamma,
        pre_sumcheck_state: state,
        ..ProtocolTrace::default()
    };
    let (rounds, challenges, final_claim) =
        prove_phase(&mut transcript, &mut trace, initial, &mut oracle).expect("honest fixed-profile SumCheck chain");
    assert_eq!(rounds.len(), ROUNDS);
    assert!(rounds.iter().all(|round| round.len() == 10));
    assert_eq!(challenges.len(), ROUNDS);
    assert_eq!(final_claim, oracle.terminal(), "canonical scalar terminal identity");
    let (eval_k_constant, eval_a_constants) = oracle.scalar_outputs();
    let result = json!([
        1,
        identity,
        context,
        public,
        commitment,
        trace.alpha.iter().copied().map(words).collect::<Vec<_>>(),
        words(gamma),
        state.map(|value| value.as_canonical_u64()),
        rounds
            .iter()
            .map(|round| round.iter().copied().map(words).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        challenges.iter().copied().map(words).collect::<Vec<_>>(),
        trace
            .round_claims
            .iter()
            .copied()
            .map(words)
            .collect::<Vec<_>>(),
        transcript.state().map(|value| value.as_canonical_u64()),
        words(eval_k_constant),
        eval_a_constants.map(words)
    ]);
    let mut encoded = serde_json::to_vec(&result).expect("canonical scalar-round output");
    encoded.push(b'\n');
    fs::write(output, encoded).expect("complete scalar-round sink");
    println!("honest_scalar_rounds={} elapsed={:?}", ROUNDS, started.elapsed());
    println!("full_ring_output_families_and_positive_phase_conformance=still_required");
}
