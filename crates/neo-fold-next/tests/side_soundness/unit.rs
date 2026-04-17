use neo_fold_next::rv64im::{derive_phase0_point, CommitmentContextId, FamilyEvalSchemaId, OpenedAjtaiObjectId};

#[test]
fn rv64im_side_soundness_phase0_point_depends_on_full_target_key() {
    let commitment_context = CommitmentContextId::new([1; 32], [2; 32]);
    let opened_object = OpenedAjtaiObjectId::new(
        FamilyEvalSchemaId::Stage1Rows.family_kind(),
        &commitment_context,
        [3; 32],
        1,
        4,
    );

    let slot_zero = derive_phase0_point(
        &opened_object,
        &commitment_context,
        FamilyEvalSchemaId::Stage1Rows,
        0,
        [4; 32],
    );
    let slot_one = derive_phase0_point(
        &opened_object,
        &commitment_context,
        FamilyEvalSchemaId::Stage1Rows,
        1,
        [4; 32],
    );
    let rebound = derive_phase0_point(
        &opened_object,
        &commitment_context,
        FamilyEvalSchemaId::Stage1Rows,
        0,
        [5; 32],
    );
    let other_schema = derive_phase0_point(
        &opened_object,
        &commitment_context,
        FamilyEvalSchemaId::Stage2RegisterWrites,
        0,
        [4; 32],
    );

    assert_ne!(slot_zero, slot_one, "slot changes must change the Phase 0 point");
    assert_ne!(
        slot_zero, rebound,
        "binding-digest changes must change the Phase 0 point"
    );
    assert_ne!(slot_zero, other_schema, "schema changes must change the Phase 0 point");
}
