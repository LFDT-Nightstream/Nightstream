#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::PathBuf;
use std::process::Command;

use neo_ccs::crypto::poseidon2_goldilocks as p2;
use neo_math::{from_complex, KExtensions, F};
use neo_prover_metal::poseidon2::round_constant_words;
use neo_prover_metal::{MetalDevice, MetalTranscriptOp};
use neo_transcript::Poseidon2Transcript;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const Q: u64 = 0xffff_ffff_0000_0001;

#[test]
fn metal_field_and_poseidon2_match_canonical_rust() {
    let metallib = build_macos_metallib();
    let metal = unsafe { MetalDevice::from_metallib(&metallib) }.expect("open Metal parity device");

    let lhs = [0, 1, 2, Q - 1, Q - 2, 0x1234_5678_9abc_def0, 0xffff_fffe_ffff_ffff];
    let rhs = [Q - 1, 2, Q - 2, Q - 1, 17, 0xfedc_ba98_7654_3210, 0x0000_0001_0000_0001];
    let expected_add = field_binary(&lhs, &rhs, |a, b| a + b);
    let expected_sub = field_binary(&lhs, &rhs, |a, b| a - b);
    let expected_mul = field_binary(&lhs, &rhs, |a, b| a * b);
    assert_eq!(metal.goldilocks_add(&lhs, &rhs).unwrap(), expected_add);
    assert_eq!(metal.goldilocks_sub(&lhs, &rhs).unwrap(), expected_sub);
    assert_eq!(metal.goldilocks_mul(&lhs, &rhs).unwrap(), expected_mul);

    let low_norm = [0, 1, 2, Q - 1, Q - 2, u32::MAX as u64, Q - u32::MAX as u64];
    let expected_low_norm = field_binary(&lhs, &low_norm, |a, b| a * b);
    assert_eq!(
        metal.goldilocks_mul_low_norm(&lhs, &low_norm).unwrap(),
        expected_low_norm
    );

    let lhs_k: Vec<[u64; 2]> = lhs
        .iter()
        .zip(rhs.iter().rev())
        .map(|(&c0, &c1)| [c0, c1])
        .collect();
    let rhs_k: Vec<[u64; 2]> = rhs
        .iter()
        .zip(lhs.iter().rev())
        .map(|(&c0, &c1)| [c0, c1])
        .collect();
    let expected_k: Vec<[u64; 2]> = lhs_k
        .iter()
        .zip(&rhs_k)
        .map(|(a, b)| {
            let product =
                from_complex(F::from_u64(a[0]), F::from_u64(a[1])) * from_complex(F::from_u64(b[0]), F::from_u64(b[1]));
            let [c0, c1] = product.as_coeffs();
            [c0.as_canonical_u64(), c1.as_canonical_u64()]
        })
        .collect();
    assert_eq!(metal.extension_mul(&lhs_k, &rhs_k).unwrap(), expected_k);

    let states: Vec<[u64; p2::WIDTH]> = (0..64)
        .map(|case| {
            core::array::from_fn(|lane| {
                ((case as u128 * 0x9e37_79b9_7f4a_7c15 + lane as u128 * 0x1000_0001) % Q as u128) as u64
            })
        })
        .collect();
    let expected_states: Vec<[u64; p2::WIDTH]> = states
        .iter()
        .map(|state| {
            let state = core::array::from_fn(|lane| F::from_u64(state[lane]));
            let out = p2::permute_state(state);
            core::array::from_fn(|lane| out[lane].as_canonical_u64())
        })
        .collect();
    let constants = round_constant_words();
    assert_eq!(metal.poseidon2_permute(&states, &constants).unwrap(), expected_states);

    let hash_inputs = [
        Vec::new(),
        vec![1],
        vec![1, 2, 3, 4],
        vec![1, 2, 3, 4, 5],
        (0..17).map(|word| word * 0x1000_0001).collect(),
    ];
    let input_refs: Vec<&[u64]> = hash_inputs.iter().map(Vec::as_slice).collect();
    let expected_hashes: Vec<[u64; p2::DIGEST_LEN]> = hash_inputs
        .iter()
        .map(|input| {
            let fields: Vec<F> = input.iter().map(|&word| F::from_u64(word)).collect();
            let digest = p2::poseidon2_hash(&fields);
            core::array::from_fn(|lane| digest[lane].as_canonical_u64())
        })
        .collect();
    assert_eq!(metal.poseidon2_hash(&input_refs, &constants).unwrap(), expected_hashes);

    let seed = [F::from_u64(7), F::from_u64(11), F::from_u64(13)];
    let mut canonical_transcript = Poseidon2Transcript::new_raw_fields(&seed);
    let initial_state = canonical_transcript.state();
    let initial_absorbed = canonical_transcript.absorbed();
    let mut transcript_ops = Vec::new();
    let mut expected_challenges = Vec::new();
    for round in 0..32 {
        let fields: Vec<F> = (0..round % 7)
            .map(|lane| F::from_u64((round * 17 + lane * 29) as u64))
            .collect();
        let mut absorb = vec![fields.len() as u64];
        absorb.extend(fields.iter().map(|field| field.as_canonical_u64()));
        canonical_transcript.append_fields_raw(&fields);
        transcript_ops.push(MetalTranscriptOp::AbsorbRaw(absorb));

        let count = 1 + round % 6;
        expected_challenges.extend(
            canonical_transcript
                .challenge_fields_raw(count)
                .iter()
                .map(|field| field.as_canonical_u64()),
        );
        transcript_ops.push(MetalTranscriptOp::Challenge(count));
    }
    let transcript_out = metal
        .poseidon2_transcript(
            initial_state.map(|field| field.as_canonical_u64()),
            initial_absorbed,
            &transcript_ops,
            &constants,
        )
        .unwrap();
    assert_eq!(transcript_out.challenges, expected_challenges);
    assert_eq!(
        transcript_out.state,
        canonical_transcript
            .state()
            .map(|field| field.as_canonical_u64())
    );
    assert_eq!(transcript_out.absorbed, canonical_transcript.absorbed());

    eprintln!(
        "[metal parity] OK on {}: Goldilocks, K, Poseidon2 hash/transcript",
        metal.name()
    );
}

fn field_binary(lhs: &[u64], rhs: &[u64], op: impl Fn(F, F) -> F) -> Vec<u64> {
    lhs.iter()
        .zip(rhs)
        .map(|(&lhs, &rhs)| op(F::from_u64(lhs), F::from_u64(rhs)).as_canonical_u64())
        .collect()
}

fn build_macos_metallib() -> Vec<u8> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = manifest.join("../../scripts/build_metal_shaders.sh");
    let output = std::env::temp_dir().join(format!("nightstream-metal-parity-{}.metallib", std::process::id()));
    let status = Command::new(script)
        .args(["--sdk", "macosx", "--out"])
        .arg(&output)
        .status()
        .expect("run Metal shader build");
    assert!(status.success(), "Metal shader build failed");
    let bytes = std::fs::read(&output).expect("read parity metallib");
    let _ = std::fs::remove_file(output);
    bytes
}
