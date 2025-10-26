// crates/neo-fold/tests/transcript_red_team.rs
use neo_transcript::{Poseidon2Transcript, labels as tr_labels, Transcript};

#[test]
fn domain_separation_is_binding() {
    let mut t1 = Poseidon2Transcript::new(b"test1");
    let mut t2 = Poseidon2Transcript::new(b"test2");

    t1.append_message(tr_labels::PI_CCS, b"");
    t1.append_u64s(b"u", &[1,2,3]);

    t2.append_message(tr_labels::PI_RLC, b""); // different domain, same payload
    t2.append_u64s(b"u", &[1,2,3]);

    let c1 = t1.challenge_field(b"chal/f");
    let c2 = t2.challenge_field(b"chal/f");
    assert_ne!(c1, c2, "different domains must yield different challenges");
}

#[test]
fn order_matters_in_transcript() {
    let mut t1 = Poseidon2Transcript::new(b"order_test1");
    let mut t2 = Poseidon2Transcript::new(b"order_test2");

    t1.append_message(tr_labels::PI_CCS, b"");
    t1.append_u64s(b"u", &[42]);
    t1.append_message(tr_labels::PI_DEC, b"");
    t1.append_u64s(b"u", &[7]);

    t2.append_message(tr_labels::PI_DEC, b"");
    t2.append_u64s(b"u", &[7]);
    t2.append_message(tr_labels::PI_CCS, b"");
    t2.append_u64s(b"u", &[42]);

    let c1 = t1.challenge_field(b"chal/f");
    let c2 = t2.challenge_field(b"chal/f");
    assert_ne!(c1, c2, "challenge must bind to message ordering");
}
