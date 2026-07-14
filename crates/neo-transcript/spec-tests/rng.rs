//! Spec-derived tests for Transcript.spec.md RNG invariant obligations.
//!
//! Covers: RNG binding (witness + entropy sensitivity) and RNG determinism.

use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript, TranscriptRngBuilder};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};

// ---------------------------------------------------------------------------
// Obligation 8: RNG binding — different witness/entropy -> different output
// ---------------------------------------------------------------------------

/// Transcript.spec.md: RNG binding — different witness data or external
/// entropy produces different RNG output.
#[test]
fn rng_binding_changes_on_inputs() {
    let mut tr = Poseidon2Transcript::new(b"rng/test");
    tr.append_message(b"m", b"public");

    let base = TranscriptRngBuilder::from_transcript(&tr);
    let ws1 = [F::from_u64(123)];
    let ws2 = [F::from_u64(124)];

    let mut rng1 = ChaCha8Rng::seed_from_u64(42);
    let mut rng2 = ChaCha8Rng::seed_from_u64(42);
    let mut rng3 = ChaCha8Rng::seed_from_u64(43);

    let mut trrng1 = base
        .clone()
        .rekey_with_witness_fields(b"wit", &ws1)
        .finalize(&mut rng1);
    let mut trrng2 = base
        .clone()
        .rekey_with_witness_fields(b"wit", &ws2)
        .finalize(&mut rng2);
    let mut trrng3 = base
        .clone()
        .rekey_with_witness_fields(b"wit", &ws1)
        .finalize(&mut rng3);

    let mut out1 = [0u8; 32];
    let mut out2 = [0u8; 32];
    let mut out3 = [0u8; 32];
    trrng1.fill_bytes(&mut out1);
    trrng2.fill_bytes(&mut out2);
    trrng3.fill_bytes(&mut out3);

    // Witness change -> output changes
    assert_ne!(out1, out2, "different witness data must produce different RNG output");
    // External entropy change -> output changes
    assert_ne!(
        out1, out3,
        "different external entropy must produce different RNG output"
    );
}

// ---------------------------------------------------------------------------
// Obligation 9: RNG determinism — same inputs -> same output
// ---------------------------------------------------------------------------

/// Transcript.spec.md: RNG determinism — identical transcript state, witness
/// data, and external entropy produce identical RNG output.
#[test]
fn rng_determinism_same_inputs() {
    let mut tr = Poseidon2Transcript::new(b"rng/test2");
    tr.append_message(b"m", b"public");
    let base = TranscriptRngBuilder::from_transcript(&tr);
    let ws = [F::from_u64(777)];
    let mut rng_a = ChaCha8Rng::seed_from_u64(100);
    let mut rng_b = ChaCha8Rng::seed_from_u64(100);

    let mut trrng_a = base
        .clone()
        .rekey_with_witness_fields(b"wit", &ws)
        .finalize(&mut rng_a);
    let mut trrng_b = base
        .clone()
        .rekey_with_witness_fields(b"wit", &ws)
        .finalize(&mut rng_b);
    let mut out_a = [0u8; 64];
    let mut out_b = [0u8; 64];
    trrng_a.fill_bytes(&mut out_a);
    trrng_b.fill_bytes(&mut out_b);
    assert_eq!(out_a, out_b, "same inputs must produce identical RNG output");
}

// ---------------------------------------------------------------------------
// Additional: RNG field() produces non-trivial output
// ---------------------------------------------------------------------------

/// TranscriptRng::field() produces field elements (basic sanity check).
#[test]
fn rng_field_produces_elements() {
    let mut tr = Poseidon2Transcript::new(b"rng/field");
    tr.append_message(b"m", b"data");
    let builder = TranscriptRngBuilder::from_transcript(&tr);
    let mut ext_rng = ChaCha8Rng::seed_from_u64(0);
    let mut trrng = builder
        .rekey_with_witness_fields(b"w", &[F::from_u64(1)])
        .finalize(&mut ext_rng);

    let mut seen_nonzero = false;
    for _ in 0..10 {
        let f = trrng.field();
        if f != F::ZERO {
            seen_nonzero = true;
        }
    }
    assert!(seen_nonzero, "TranscriptRng::field() produced only zeros in 10 calls");
}

// ---------------------------------------------------------------------------
// Additional: RNG is transcript-bound (different transcript -> different RNG)
// ---------------------------------------------------------------------------

/// Different transcript states produce different RNG outputs, even with
/// identical witness and entropy.
#[test]
fn rng_transcript_binding() {
    let mut tr1 = Poseidon2Transcript::new(b"rng/bind");
    tr1.append_message(b"m", b"data1");

    let mut tr2 = Poseidon2Transcript::new(b"rng/bind");
    tr2.append_message(b"m", b"data2");

    let ws = [F::from_u64(42)];

    let mut rng1 = ChaCha8Rng::seed_from_u64(99);
    let mut rng2 = ChaCha8Rng::seed_from_u64(99);

    let mut trrng1 = TranscriptRngBuilder::from_transcript(&tr1)
        .rekey_with_witness_fields(b"w", &ws)
        .finalize(&mut rng1);
    let mut trrng2 = TranscriptRngBuilder::from_transcript(&tr2)
        .rekey_with_witness_fields(b"w", &ws)
        .finalize(&mut rng2);

    let mut out1 = [0u8; 32];
    let mut out2 = [0u8; 32];
    trrng1.fill_bytes(&mut out1);
    trrng2.fill_bytes(&mut out2);

    assert_ne!(
        out1, out2,
        "different transcript states must produce different RNG output"
    );
}

/// Red-team regression: witness-rekey labels and vectors must be framed as
/// separate inputs. These two different `(label, witness)` pairs currently
/// absorb the same field-element sequence: `[b'L', 2, 1, x]`.
#[test]
fn rng_rekey_frames_label_separately_from_witness_fields() {
    let tr = Poseidon2Transcript::new(b"rng/rekey-framing");
    let base = TranscriptRngBuilder::from_transcript(&tr);
    let x = F::from_u64(0x1234_5678);
    let mut entropy_a = ChaCha8Rng::seed_from_u64(404);
    let mut entropy_b = ChaCha8Rng::seed_from_u64(404);

    let mut rng_a = base
        .clone()
        .rekey_with_witness_fields(b"L", &[F::ONE, x])
        .finalize(&mut entropy_a);
    let mut rng_b = base
        .rekey_with_witness_fields(b"L\x02", &[x])
        .finalize(&mut entropy_b);
    let mut out_a = [0u8; 32];
    let mut out_b = [0u8; 32];
    rng_a.fill_bytes(&mut out_a);
    rng_b.fill_bytes(&mut out_b);

    assert_ne!(
        out_a, out_b,
        "different witness-rekey labels and vectors must not share an RNG stream"
    );
}
