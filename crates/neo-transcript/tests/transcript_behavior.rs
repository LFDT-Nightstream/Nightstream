//! Transcript framing, isolation, determinism, and challenge behavior.

use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

#[test]
fn framing_distinguishes_splits() {
    let mut t1 = Poseidon2Transcript::new(b"test/app");
    t1.append_message(b"a", b"bc");
    let d1 = t1.digest32();

    let mut t2 = Poseidon2Transcript::new(b"test/app");
    t2.append_message(b"ab", b"c");
    let d2 = t2.digest32();

    assert_ne!(d1, d2, "framing must distinguish different label/byte splits");
}

#[test]
fn framing_distinguishes_message_bytes_from_field_elements() {
    let mut bytes = Poseidon2Transcript::new(b"test/typed-framing");
    bytes.append_message(b"value", &[42]);
    let bytes_digest = bytes.digest32();

    let mut fields = Poseidon2Transcript::new(b"test/typed-framing");
    fields.append_fields(b"value", &[F::from_u64(42)]);
    let fields_digest = fields.digest32();

    assert_ne!(
        bytes_digest, fields_digest,
        "append_message and append_fields must have distinct operation tags"
    );
}

#[test]
fn label_changes_challenge() {
    let mut t = Poseidon2Transcript::new(b"neo/tests");
    t.append_message(b"m", b"data");
    let c1 = t.challenge_field(b"alpha");

    let mut t2 = Poseidon2Transcript::new(b"neo/tests");
    t2.append_message(b"m", b"data");
    let c2 = t2.challenge_field(b"beta");

    assert_ne!(c1.as_canonical_u64(), c2.as_canonical_u64());
}

#[test]
fn fork_isolated() {
    let t = Poseidon2Transcript::new(b"neo/tests");
    let mut a = t.fork(b"A");
    let mut b = t.fork(b"B");
    let ca = a.challenge_field(b"rho");
    let cb = b.challenge_field(b"rho");
    assert_ne!(ca.as_canonical_u64(), cb.as_canonical_u64());
}

#[test]
fn fork_does_not_affect_parent() {
    let mut t1 = Poseidon2Transcript::new(b"neo/tests");
    t1.append_message(b"m", b"data");

    let mut t2 = t1.clone();

    let mut child = t1.fork(b"child");
    let _ = child.challenge_field(b"x");

    let c1 = t1.challenge_field(b"rho");
    let c2 = t2.challenge_field(b"rho");
    assert_eq!(c1.as_canonical_u64(), c2.as_canonical_u64());
}

#[test]
fn determinism_identical_operations() {
    let run = || {
        let mut t = Poseidon2Transcript::new(b"neo/determinism");
        t.append_message(b"step", b"hello");
        t.append_fields(b"vals", &[F::from_u64(42), F::from_u64(99)]);
        let c = t.challenge_field(b"alpha");
        let d = t.digest32();
        (c, d)
    };

    let (c1, d1) = run();
    let (c2, d2) = run();

    assert_eq!(c1.as_canonical_u64(), c2.as_canonical_u64());
    assert_eq!(d1, d2);
}

#[test]
fn domain_separation_app_labels() {
    let mut t1 = Poseidon2Transcript::new(b"app/alpha");
    t1.append_message(b"m", b"data");
    let c1 = t1.challenge_field(b"x");

    let mut t2 = Poseidon2Transcript::new(b"app/beta");
    t2.append_message(b"m", b"data");
    let c2 = t2.challenge_field(b"x");

    assert_ne!(
        c1.as_canonical_u64(),
        c2.as_canonical_u64(),
        "different app labels must produce different challenges"
    );
}

#[test]
fn domain_gate_squeeze_changes_output() {
    let mut t = Poseidon2Transcript::new(b"neo/gate");
    t.append_message(b"m", b"data");

    let c1 = t.challenge_field(b"x");
    let c2 = t.challenge_field(b"x");

    assert_ne!(
        c1.as_canonical_u64(),
        c2.as_canonical_u64(),
        "consecutive squeezes must differ due to domain gate"
    );
}

#[test]
fn challenge_nonzero_field_never_zero() {
    let mut t = Poseidon2Transcript::new(b"neo/nonzero");
    for i in 0..100u64 {
        t.append_message(b"i", &i.to_le_bytes());
        let c = t.challenge_nonzero_field(b"nz");
        assert_ne!(c, F::ZERO, "challenge_nonzero_field returned zero at iteration {i}");
    }
}

#[test]
#[should_panic(expected = "iterator length mismatch")]
fn append_fields_iter_length_mismatch_panics() {
    let mut t = Poseidon2Transcript::new(b"neo/iter");
    let fields = vec![F::from_u64(1), F::from_u64(2)];
    t.append_fields_iter(b"bad", 5, fields.into_iter());
}

#[test]
fn append_fields_batch_determinism() {
    let fields = [F::from_u64(10), F::from_u64(20), F::from_u64(30)];

    let mut t1 = Poseidon2Transcript::new(b"neo/batch");
    t1.append_fields(b"vals", &fields);
    let d1 = t1.digest32();

    let mut t2 = Poseidon2Transcript::new(b"neo/batch");
    t2.append_fields(b"vals", &fields);
    let d2 = t2.digest32();

    assert_eq!(d1, d2);
}

#[test]
fn append_fields_different_data_different_digest() {
    let mut t1 = Poseidon2Transcript::new(b"neo/diff");
    t1.append_fields(b"v", &[F::from_u64(1), F::from_u64(2)]);
    let d1 = t1.digest32();

    let mut t2 = Poseidon2Transcript::new(b"neo/diff");
    t2.append_fields(b"v", &[F::from_u64(1), F::from_u64(3)]);
    let d2 = t2.digest32();

    assert_ne!(d1, d2);
}

#[test]
fn challenge_bytes_exact_length() {
    let mut t = Poseidon2Transcript::new(b"neo/bytes");
    t.append_message(b"m", b"data");

    for len in [1, 7, 8, 16, 31, 32, 33, 64, 100] {
        let mut out = vec![0u8; len];
        let mut t2 = t.clone();
        t2.challenge_bytes(b"c", &mut out);
        assert!(out.iter().any(|&b| b != 0), "challenge_bytes({len}) produced all zeros");
    }
}

#[test]
fn challenge_output_arity_is_bound_into_transcript_state() {
    let mut one = Poseidon2Transcript::new(b"neo/arity");
    let mut four = one.clone();
    let c1 = one.challenge_fields(b"alpha", 1);
    let c4 = four.challenge_fields(b"alpha", 4);

    assert_eq!(c1[0], c4[0], "XOF prefix precondition");
    assert_ne!(
        one.challenge_field(b"next"),
        four.challenge_field(b"next"),
        "challenge-field arity must be transcript metadata"
    );

    let mut byte_one = Poseidon2Transcript::new(b"neo/byte-arity");
    let mut byte_32 = byte_one.clone();
    let mut out1 = [0u8; 1];
    let mut out32 = [0u8; 32];
    byte_one.challenge_bytes(b"alpha", &mut out1);
    byte_32.challenge_bytes(b"alpha", &mut out32);

    assert_eq!(out1[0], out32[0], "XOF prefix precondition");
    assert_ne!(
        byte_one.challenge_field(b"next"),
        byte_32.challenge_field(b"next"),
        "byte-challenge length must be transcript metadata"
    );
}

#[test]
fn challenge_output_type_is_bound_into_transcript_state() {
    let mut bytes = Poseidon2Transcript::new(b"neo/challenge-type");
    let mut fields = bytes.clone();

    let mut byte_output = [0u8; 32];
    bytes.challenge_bytes(b"alpha", &mut byte_output);
    let field_output = fields.challenge_fields(b"alpha", 4);

    let mut encoded_fields = [0u8; 32];
    for (chunk, field) in encoded_fields.chunks_exact_mut(8).zip(&field_output) {
        chunk.copy_from_slice(&field.as_canonical_u64().to_le_bytes());
    }
    assert_eq!(byte_output, encoded_fields, "typed-query collision precondition");
    assert_ne!(
        bytes.challenge_field(b"next"),
        fields.challenge_field(b"next"),
        "byte and field challenge operations must have distinct query tags"
    );
}
