use neo_transcript::{Poseidon2Transcript, Transcript, TranscriptRngBuilder};
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use p3_field::PrimeCharacteristicRing;
use neo_math::F;

#[test]
fn rng_binding_changes_on_inputs() {
    // Base transcript with same public data
    let mut tr = Poseidon2Transcript::new(b"rng/test");
    tr.append_message(b"m", b"public");

    // Builder from transcript state
    let base = TranscriptRngBuilder::from_transcript(&tr);
    let ws1 = [F::from_u64(123)];
    let ws2 = [F::from_u64(124)];

    let mut rng1 = ChaCha8Rng::seed_from_u64(42);
    let mut rng2 = ChaCha8Rng::seed_from_u64(42);
    let mut rng3 = ChaCha8Rng::seed_from_u64(43);

    let mut trrng1 = base.clone().rekey_with_witness_fields(b"wit", &ws1).finalize(&mut rng1);
    let mut trrng2 = base.clone().rekey_with_witness_fields(b"wit", &ws2).finalize(&mut rng2);
    let mut trrng3 = base.clone().rekey_with_witness_fields(b"wit", &ws1).finalize(&mut rng3);

    let mut out1 = [0u8; 32];
    let mut out2 = [0u8; 32];
    let mut out3 = [0u8; 32];
    trrng1.fill_bytes(&mut out1);
    trrng2.fill_bytes(&mut out2);
    trrng3.fill_bytes(&mut out3);

    // ws change -> output changes
    assert_ne!(out1, out2);
    // external entropy change -> output changes
    assert_ne!(out1, out3);
}

#[test]
fn rng_determinism_same_inputs() {
    let mut tr = Poseidon2Transcript::new(b"rng/test2");
    tr.append_message(b"m", b"public");
    let base = TranscriptRngBuilder::from_transcript(&tr);
    let ws = [F::from_u64(777)];
    let mut rng_a = ChaCha8Rng::seed_from_u64(100);
    let mut rng_b = ChaCha8Rng::seed_from_u64(100);

    let mut trrng_a = base.clone().rekey_with_witness_fields(b"wit", &ws).finalize(&mut rng_a);
    let mut trrng_b = base.clone().rekey_with_witness_fields(b"wit", &ws).finalize(&mut rng_b);
    let mut out_a = [0u8; 64];
    let mut out_b = [0u8; 64];
    trrng_a.fill_bytes(&mut out_a);
    trrng_b.fill_bytes(&mut out_b);
    assert_eq!(out_a, out_b);
}
