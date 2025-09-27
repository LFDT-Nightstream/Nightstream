use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeField64;

#[test]
fn label_changes_challenge() {
    let mut t = Poseidon2Transcript::new(b"neo/tests");
    t.append_message(b"m", b"data");
    let c1 = t.challenge_field(b"alpha");
    // Restart same absorption
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
