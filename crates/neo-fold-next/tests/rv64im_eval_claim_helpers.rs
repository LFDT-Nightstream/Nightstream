use neo_fold_next::rv64im::{
    derive_phase0_point, encode_packed_column_evals_k, encode_words_to_field_evals_k, phase0_full_width_for_schema,
    reconstruct_words_from_field_evals, unpack_column_evals_k, CommitmentContextId, EvalClaimError, FamilyEvalSchemaId,
    OpenedAjtaiObjectId,
};
use neo_math::{from_complex, D, F, K};
use p3_field::PrimeCharacteristicRing;

fn digest(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn k(real: u64, imag: u64) -> K {
    from_complex(F::from_u64(real), F::from_u64(imag))
}

fn opened_object(
    schema: FamilyEvalSchemaId,
    commitment_context: &CommitmentContextId,
    commitment_root_digest: [u8; 32],
    row_domain_log_size: u32,
) -> OpenedAjtaiObjectId {
    OpenedAjtaiObjectId::new(
        schema.family_kind(),
        commitment_context,
        commitment_root_digest,
        1,
        row_domain_log_size,
    )
}

#[test]
fn phase0_same_claim_inputs_rederive_same_point() {
    let commitment_context = CommitmentContextId::new(digest(1), digest(2));
    let opened_object = opened_object(FamilyEvalSchemaId::Stage1Rows, &commitment_context, digest(3), 4);

    let point_a = derive_phase0_point(
        &opened_object,
        &commitment_context,
        FamilyEvalSchemaId::Stage1Rows,
        0,
        digest(4),
    );
    let point_b = derive_phase0_point(
        &opened_object,
        &commitment_context,
        FamilyEvalSchemaId::Stage1Rows,
        0,
        digest(4),
    );

    assert_eq!(point_a, point_b);
}

#[test]
fn phase0_distinct_slot_changes_point() {
    let commitment_context = CommitmentContextId::new(digest(10), digest(11));
    let opened_object = opened_object(
        FamilyEvalSchemaId::Stage2RegisterReads,
        &commitment_context,
        digest(12),
        3,
    );

    let slot0_point = derive_phase0_point(
        &opened_object,
        &commitment_context,
        FamilyEvalSchemaId::Stage2RegisterReads,
        0,
        digest(13),
    );
    let slot1_point = derive_phase0_point(
        &opened_object,
        &commitment_context,
        FamilyEvalSchemaId::Stage2RegisterReads,
        1,
        digest(13),
    );

    assert_ne!(slot0_point, slot1_point);
}

#[test]
fn phase0_point_len_matches_row_domain_log_size() {
    let commitment_context = CommitmentContextId::new(digest(20), digest(21));
    let opened_object = opened_object(FamilyEvalSchemaId::Stage3Continuity, &commitment_context, digest(22), 5);
    let point = derive_phase0_point(
        &opened_object,
        &commitment_context,
        FamilyEvalSchemaId::Stage3Continuity,
        0,
        digest(23),
    );

    assert_eq!(point.len(), opened_object.row_domain_log_size as usize);
}

#[test]
fn unpack_column_evals_recovers_first_full_width_coeffs() {
    let words = (0..23)
        .map(|idx| 0x1000_0000_0000_0000u64 + idx as u64 * 0x0101)
        .collect::<Vec<_>>();
    let field_evals =
        encode_words_to_field_evals_k(FamilyEvalSchemaId::Stage1Rows, &words).expect("encode stage1 words");
    let packed =
        encode_packed_column_evals_k(FamilyEvalSchemaId::Stage1Rows, &field_evals).expect("pack stage1 field evals");
    let unpacked = unpack_column_evals_k(FamilyEvalSchemaId::Stage1Rows, &packed).expect("unpack stage1 payload");

    assert_eq!(unpacked, field_evals);
}

#[test]
fn packed_padding_positions_are_zero() {
    let words = (0..23)
        .map(|idx| 0x2000_0000_0000_0000u64 + idx as u64 * 0x0202)
        .collect::<Vec<_>>();
    let field_evals =
        encode_words_to_field_evals_k(FamilyEvalSchemaId::Stage1Rows, &words).expect("encode stage1 words");
    let packed =
        encode_packed_column_evals_k(FamilyEvalSchemaId::Stage1Rows, &field_evals).expect("pack stage1 field evals");
    let full_width = phase0_full_width_for_schema(FamilyEvalSchemaId::Stage1Rows);
    let used_coeffs_in_last_column = full_width % D;
    let last_column = packed
        .last()
        .expect("stage1 has at least one packed column");

    for coeff in &last_column.coeffs[used_coeffs_in_last_column..] {
        assert_eq!(*coeff, K::ZERO);
    }
}

#[test]
fn word_reconstruction_matches_original_words() {
    let cases = [
        (
            FamilyEvalSchemaId::Stage2RegisterWrites,
            vec![
                0x0123_4567_89ab_cdef,
                0x1111_2222_3333_4444,
                0x5555_6666_7777_8888,
                0x9999_aaaa_bbbb_cccc,
                0xddd0_eee1_fff2_0003,
            ],
        ),
        (
            FamilyEvalSchemaId::Stage3Continuity,
            vec![
                0x0001_0002_0003_0004,
                0x0102_0304_0506_0708,
                0x1112_1314_1516_1718,
                0x2122_2324_2526_2728,
                0x3132_3334_3536_3738,
                0x4142_4344_4546_4748,
            ],
        ),
    ];

    for (schema, words) in cases {
        let field_evals = encode_words_to_field_evals_k(schema, &words).expect("encode words");
        let reconstructed = reconstruct_words_from_field_evals(schema, &field_evals).expect("reconstruct words");
        assert_eq!(reconstructed, words, "schema {schema:?}");
    }
}

#[test]
fn reconstruct_words_rejects_non_base_field_limb() {
    let words = vec![
        0x0123_4567_89ab_cdef,
        0x1111_2222_3333_4444,
        0x5555_6666_7777_8888,
        0x9999_aaaa_bbbb_cccc,
        0xddd0_eee1_fff2_0003,
    ];
    let mut field_evals =
        encode_words_to_field_evals_k(FamilyEvalSchemaId::Stage2RegisterReads, &words).expect("encode words");
    field_evals[1] = k(7, 1);

    let err = reconstruct_words_from_field_evals(FamilyEvalSchemaId::Stage2RegisterReads, &field_evals)
        .expect_err("non-base-field limb must reject");

    assert_eq!(
        err,
        EvalClaimError::NonBaseFieldLimb {
            schema: FamilyEvalSchemaId::Stage2RegisterReads,
            word_index: 0,
            limb_index: 0,
        }
    );
}
