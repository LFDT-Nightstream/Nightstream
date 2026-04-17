use bellpepper_core::{num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem};
use neo_fold_next::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use neo_math::F as GoldilocksF;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use spartan2::provider::goldi::F as SpartanF;

fn alloc_constant(
    cs: &mut TestConstraintSystem<SpartanF>,
    value: SpartanF,
    label: &str,
) -> Result<AllocatedNum<SpartanF>, bellpepper_core::SynthesisError> {
    let out = AllocatedNum::alloc(cs.namespace(|| label.to_string()), || Ok(value))?;
    cs.enforce(
        || format!("{label}_constant"),
        |lc| lc + out.get_variable(),
        |lc| lc + <TestConstraintSystem<SpartanF> as ConstraintSystem<SpartanF>>::one(),
        |lc| {
            lc + (
                value,
                <TestConstraintSystem<SpartanF> as ConstraintSystem<SpartanF>>::one(),
            )
        },
    );
    Ok(out)
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_transcript_resume_matches_native() {
    let mut native = Poseidon2Transcript::new(b"neo.fold.next/test/transcript_resume");
    native.append_message(b"label0", b"payload0");
    native.append_fields(b"fields0", &[GoldilocksF::from_u64(11), GoldilocksF::from_u64(13)]);
    let snapshot_state = native.state();
    let snapshot_absorbed = native.absorbed();
    native.append_message(b"label1", b"payload1");
    let expected_digest = native.digest32();

    let mut resumed = Poseidon2Transcript::from_state_and_absorbed(snapshot_state, snapshot_absorbed);
    resumed.append_message(b"label1", b"payload1");
    assert_eq!(resumed.digest32(), expected_digest);
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_transcript_circuit_resume_matches_native() {
    let mut native = Poseidon2Transcript::new(b"neo.fold.next/test/transcript_resume");
    native.append_message(b"label0", b"payload0");
    native.append_fields(b"fields0", &[GoldilocksF::from_u64(11), GoldilocksF::from_u64(13)]);
    let snapshot_state = native.state();
    let snapshot_absorbed = native.absorbed();
    native.append_message(b"label1", b"payload1");
    let expected_digest = native.digest32();

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let state_vars = core::array::from_fn(|idx| {
        alloc_constant(
            &mut cs,
            SpartanF::from_canonical_u64(snapshot_state[idx].as_canonical_u64()),
            &format!("state_{idx}"),
        )
        .expect("state")
    });
    let mut transcript = Poseidon2TranscriptCircuit::from_state(
        state_vars,
        snapshot_state.map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
        snapshot_absorbed,
    )
    .expect("resume");
    transcript
        .append_message(cs.namespace(|| "label1"), b"label1", b"payload1")
        .expect("append");
    let digest = transcript
        .digest32(cs.namespace(|| "digest"))
        .expect("digest");
    for (idx, limb) in digest.iter().enumerate() {
        let expected = alloc_constant(
            &mut cs,
            SpartanF::from_canonical_u64(u64::from_le_bytes(
                expected_digest[idx * 8..(idx + 1) * 8]
                    .try_into()
                    .expect("limb"),
            )),
            &format!("expected_{idx}"),
        )
        .expect("expected");
        cs.enforce(
            || format!("digest_eq_{idx}"),
            |lc| lc + limb.get_variable(),
            |lc| lc + <TestConstraintSystem<SpartanF> as ConstraintSystem<SpartanF>>::one(),
            |lc| lc + expected.get_variable(),
        );
    }
    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}
