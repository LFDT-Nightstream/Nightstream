use bellpepper_core::{test_cs::TestConstraintSystem, ConstraintSystem};
use neo_fold_next::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use neo_math::F as GoldilocksF;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use spartan2::provider::goldi::F as SpartanF;

fn field_to_u64(value: SpartanF) -> u64 {
    value.to_canonical_u64()
}

#[test]
fn rv64im_main_relation_transcript_matches_native_poseidon2_transcript() {
    let mut native = Poseidon2Transcript::new(b"neo.fold.next/test/main_relation");
    native.append_message(b"label/one", b"message/one");
    native.append_u64s(b"meta", &[1, 2, u64::MAX, 17]);
    native.append_fields(
        b"fields",
        &[
            GoldilocksF::from_u64(3),
            GoldilocksF::from_u64(9),
            GoldilocksF::from_u64(27),
            GoldilocksF::from_u64(81),
            GoldilocksF::from_u64(243),
        ],
    );
    let native_challenges = native.challenge_fields(b"challenge/fields", 6);
    let native_digest = native.digest32();

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let mut transcript = Poseidon2TranscriptCircuit::new(cs.namespace(|| "new"), b"neo.fold.next/test/main_relation")
        .expect("circuit transcript");
    transcript
        .append_message(cs.namespace(|| "message"), b"label/one", b"message/one")
        .expect("append message");
    transcript
        .append_u64s(cs.namespace(|| "u64s"), b"meta", &[1, 2, u64::MAX, 17])
        .expect("append u64s");

    let allocated_fields = [3u64, 9, 27, 81, 243]
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            bellpepper_core::num::AllocatedNum::alloc(cs.namespace(|| format!("field_{idx}")), || {
                Ok(SpartanF::from_canonical_u64(value))
            })
            .expect("allocate field")
        })
        .collect::<Vec<_>>();
    transcript
        .append_fields(
            cs.namespace(|| "fields"),
            b"fields",
            &allocated_fields,
            &[3u64, 9, 27, 81, 243].map(SpartanF::from_canonical_u64),
        )
        .expect("append fields");
    let gadget_challenges = transcript
        .challenge_fields(cs.namespace(|| "challenge"), b"challenge/fields", 6)
        .expect("challenge fields");
    let gadget_digest = transcript
        .digest32(cs.namespace(|| "digest"))
        .expect("digest32");

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
    assert_eq!(
        gadget_challenges
            .iter()
            .map(|value| field_to_u64(value.get_value().expect("challenge witness")))
            .collect::<Vec<_>>(),
        native_challenges
            .iter()
            .map(|value| value.as_canonical_u64())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        gadget_digest
            .iter()
            .map(|value| field_to_u64(value.get_value().expect("digest witness")).to_le_bytes())
            .flatten()
            .collect::<Vec<_>>(),
        native_digest.to_vec()
    );
}
