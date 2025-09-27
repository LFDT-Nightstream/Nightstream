use neo_transcript::{Poseidon2Transcript, Transcript};

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

