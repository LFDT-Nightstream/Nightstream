//! Phase 1.1-mini-3b — sponge-mode Poseidon transcript paths.
//!
//! This covers the Table B paths from the Fibonacci F' layout spec:
//! transcript state is preserved across interleaved absorbs and squeezes,
//! unlike the one-shot digest paths covered by mini-1, mini-2, and mini-3a.
//! The tests compare the bit-backed sponge trace builder against
//! `neo_transcript::Poseidon2Transcript` operation-for-operation.
//!
//! Out of scope: ring action, lifecycle, Spartan, generic AppStep, and any
//! change that turns `ivc_invariants` green.

use neo_fold_clean::engine::ccs_native::poseidon2_transcript::{
    decode_squeezed_lanes, SpongeTraceBuilder, SpongeTraceImage,
};
use neo_fold_clean::paper::digest::digest32_as_fields;
use neo_fold_clean::paper::f_prime::native::F_PRIME_STEP_TRANSCRIPT_LABEL;
use neo_fold_clean::paper::f_prime::poseidon_trace::assert_committed_coords_are_bits;
use neo_math::F;
use neo_reductions::engines::pi_ccs_joint::{
    ALPHA_TAG, GAMMA_TAG, PROTOCOL_VERSION, PUBLIC_INPUT_TAG, ROUND_CHALLENGE_TAG, ROUND_TAG, STATEMENT_TAG,
};
use neo_transcript::{Poseidon2Transcript, Transcript as _};
use p3_field::PrimeCharacteristicRing;

const APP: &[u8] = F_PRIME_STEP_TRANSCRIPT_LABEL;

fn deterministic_digest(seed: u64) -> [F; 4] {
    std::array::from_fn(|i| F::from_u64(seed + 17 * i as u64))
}

fn deterministic_fields(len: usize, seed: u64) -> Vec<F> {
    (0..len)
        .map(|i| F::from_u64(seed.wrapping_add(97 * i as u64)))
        .collect()
}

fn deterministic_k_rounds(rounds: usize, coeffs_per_round: usize, seed: u64) -> Vec<Vec<[F; 2]>> {
    (0..rounds)
        .map(|round| {
            (0..coeffs_per_round)
                .map(|coeff| {
                    [
                        F::from_u64(seed + 101 * round as u64 + 7 * coeff as u64),
                        F::from_u64(seed + 103 * round as u64 + 11 * coeff as u64 + 1),
                    ]
                })
                .collect()
        })
        .collect()
}

fn flatten_k_round(round: &[[F; 2]]) -> Vec<F> {
    let mut out = Vec::with_capacity(round.len() * 2);
    for &[c0, c1] in round {
        out.push(c0);
        out.push(c1);
    }
    out
}

fn assert_trace_matches_native(image: &SpongeTraceImage, native: &Poseidon2Transcript, expected_squeezes: &[F]) {
    assert_committed_coords_are_bits(&image.values);
    assert_eq!(
        image.final_state,
        native.state(),
        "builder final state must match native"
    );
    assert_eq!(
        image.absorbed,
        native.absorbed(),
        "builder absorbed cursor must match native"
    );
    assert_eq!(image.squeezed_values, expected_squeezes, "builder value-side squeezes");
    assert_eq!(
        decode_squeezed_lanes(image),
        expected_squeezes,
        "bit-decoded squeeze lanes"
    );
    assert_eq!(
        image.layout.squeeze_lane_offsets.len(),
        expected_squeezes.len(),
        "one offset per squeezed field"
    );
}

fn append_f_prime_state_prefix(native: &mut Poseidon2Transcript, builder: &mut SpongeTraceBuilder) {
    let labels: [&'static [u8]; 6] = [
        b"f_prime/vk_fs",
        b"f_prime/structure",
        b"f_prime/z_0",
        b"f_prime/z_i_in",
        b"f_prime/public_trace_in",
        b"f_prime/chunk_digest",
    ];
    for (idx, label) in labels.iter().enumerate() {
        let fields = deterministic_digest(1_000 + 100 * idx as u64);
        native.append_fields(label, &fields);
        builder.append_fields(label, &fields);
    }
}

fn append_engine_prefix(native: &mut Poseidon2Transcript, builder: &mut SpongeTraceBuilder) {
    let public = deterministic_digest(2_000);
    let statement = deterministic_digest(3_000);
    let sequences: [Vec<F>; 2] = [
        vec![
            F::from_u64(PUBLIC_INPUT_TAG),
            F::from_u64(PROTOCOL_VERSION),
            public[0],
            public[1],
            public[2],
            public[3],
        ],
        vec![
            F::from_u64(STATEMENT_TAG),
            statement[0],
            statement[1],
            statement[2],
            statement[3],
        ],
    ];
    for fields in &sequences {
        native.append_fields_raw(fields);
        builder.append_fields_raw(fields);
    }
}

fn start_prefixed_transcripts() -> (Poseidon2Transcript, SpongeTraceBuilder) {
    let mut native = Poseidon2Transcript::new(APP);
    let mut builder = SpongeTraceBuilder::new(APP);
    append_f_prime_state_prefix(&mut native, &mut builder);
    append_engine_prefix(&mut native, &mut builder);
    (native, builder)
}

#[test]
fn phase_1_mini_3b_f_prime_state_absorbs_preserve_sponge_state() {
    let mut native = Poseidon2Transcript::new(APP);
    let mut builder = SpongeTraceBuilder::new(APP);

    append_f_prime_state_prefix(&mut native, &mut builder);
    let image = builder.finish();

    assert_trace_matches_native(&image, &native, &[]);
    assert!(
        !image.layout.permute_offsets.is_empty(),
        "six labelled digest absorbs must cross the rate boundary"
    );
    eprintln!(
        "mini-3b R-11: {} permutes, {} trace bits",
        image.layout.permute_offsets.len(),
        image.values.len() - 1
    );
}

#[test]
fn phase_1_mini_3b_engine_challenge_batch_decodes() {
    let (mut native, mut builder) = start_prefixed_transcripts();
    let variables = 3usize;
    let mut expected = Vec::with_capacity(2 * (variables + 1));

    for index in 0..variables {
        let tag = [F::from_u64(ALPHA_TAG), F::from_u64(index as u64)];
        native.append_fields_raw(&tag);
        builder.append_fields_raw(&tag);
        let challenge = native.challenge_fields_raw(2);
        assert_eq!(builder.challenge_fields_raw(2), challenge);
        expected.extend(challenge);
    }
    let tag = [F::from_u64(GAMMA_TAG)];
    native.append_fields_raw(&tag);
    builder.append_fields_raw(&tag);
    let challenge = native.challenge_fields_raw(2);
    assert_eq!(builder.challenge_fields_raw(2), challenge);
    expected.extend(challenge);
    let image = builder.finish();

    assert_trace_matches_native(&image, &native, &expected);
    eprintln!(
        "mini-3b R-12: {} alpha/gamma K challenges, {} squeezed F lanes, {} permutes",
        variables + 1,
        expected.len(),
        image.layout.permute_offsets.len()
    );
}

#[test]
fn phase_1_mini_3b_joint_sumcheck_round_squeezes_decode() {
    let (mut native, mut builder) = start_prefixed_transcripts();
    let mut expected = Vec::new();

    for (round_index, round) in deterministic_k_rounds(5, 10, 5_000).iter().enumerate() {
        let mut fields = vec![
            F::from_u64(ROUND_TAG),
            F::from_u64(round_index as u64),
            F::from_u64(round.len() as u64),
        ];
        fields.extend(flatten_k_round(round));
        native.append_fields_raw(&fields);
        builder.append_fields_raw(&fields);
        let challenge_tag = [F::from_u64(ROUND_CHALLENGE_TAG), F::from_u64(round_index as u64)];
        native.append_fields_raw(&challenge_tag);
        builder.append_fields_raw(&challenge_tag);
        let challenge = native.challenge_fields_raw(2);
        assert_eq!(builder.challenge_fields_raw(2), challenge);
        expected.extend(challenge);
    }

    let image = builder.finish();
    assert_trace_matches_native(&image, &native, &expected);
    eprintln!(
        "mini-3b R-22: {} joint SumCheck rounds, {} squeezed F lanes",
        5,
        expected.len()
    );
}

#[test]
fn phase_1_mini_3b_header_digest_catch_up_decodes() {
    let (mut native, mut builder) = start_prefixed_transcripts();
    let prefix = deterministic_fields(9, 7_000);
    native.append_fields_raw(&prefix);
    builder.append_fields_raw(&prefix);

    let expected = digest32_as_fields(native.digest32()).to_vec();
    let got = builder.digest_fields().to_vec();
    let image = builder.finish();

    assert_eq!(got, expected, "builder digest fields");
    assert_trace_matches_native(&image, &native, &expected);
    eprintln!(
        "mini-3b R-25: header digest catch-up, {} permutes, {} trace bits",
        image.layout.permute_offsets.len(),
        image.values.len() - 1
    );
}
