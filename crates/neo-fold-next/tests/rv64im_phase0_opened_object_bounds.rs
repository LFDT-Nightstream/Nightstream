use neo_ajtai::{set_global_pp_seeded, AjtaiSModule, Commitment};
use neo_ccs::{traits::SModuleHomomorphism, Mat};
use neo_fold_next::rv64im::kernel::{
    phase0_full_width_for_schema, CommitmentContextId, EvalClaimError, FamilyEvalSchemaId, OpenedAjtaiObjectId,
    OpenedAjtaiObjectWitness, PackedColumnOracleRef,
};
use neo_math::{D, F};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

fn phase0_pp_seed_digest(seed: [u8; 32]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase0/pp_seed");
    tr.append_message(b"neo.fold.next/rv64im/opening_convergence/phase0/pp_seed/value", &seed);
    tr.digest32()
}

fn phase0_module_shape_digest(d: u64, packed_column_count: u64) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase0/module_shape");
    tr.append_u64s(
        b"neo.fold.next/rv64im/opening_convergence/phase0/module_shape/value",
        &[d, packed_column_count],
    );
    tr.digest32()
}

fn phase0_commitment_root_digest(commitment_vector: &[Commitment]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase0/commitment_root");
    tr.append_fields_raw(&[F::from_u64(commitment_vector.len() as u64)]);
    for commitment in commitment_vector {
        tr.append_fields_raw(&commitment.data);
    }
    tr.digest32()
}

#[test]
fn rv64im_phase0_witness_rejects_coefficients_outside_phase0_limb_range() {
    let schema = FamilyEvalSchemaId::Stage1Rows;
    let full_width = phase0_full_width_for_schema(schema);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(full_width).expect("phase0 params");
    let seed = [0x5Au8; 32];
    let padded_time_len = 1usize;
    set_global_pp_seeded(D, params.kappa as usize, padded_time_len, seed)
        .expect("register deterministic Ajtai PP for bound test");
    let committer = AjtaiSModule::from_global_for_dims(D, padded_time_len).expect("phase0 committer");

    let wide_coeff = F::from_u64(1u64 << 40);
    let mut matrix = Mat::zero(D, padded_time_len, F::ZERO);
    matrix[(0, 0)] = wide_coeff;
    let commitment_vector = vec![committer.commit(&matrix)];

    let mut row = [F::ZERO; D];
    row[0] = wide_coeff;
    let packed_columns = vec![PackedColumnOracleRef {
        column_index: 0,
        rows: vec![row],
    }];

    let commitment_context = CommitmentContextId::new(
        phase0_pp_seed_digest(seed),
        phase0_module_shape_digest(D as u64, schema.packed_column_count() as u64),
    );
    let opened_object = OpenedAjtaiObjectId::new(
        schema.family_kind(),
        &commitment_context,
        phase0_commitment_root_digest(&commitment_vector),
        1,
        0,
    );

    let err = OpenedAjtaiObjectWitness::new(opened_object, commitment_context, packed_columns, commitment_vector)
        .expect_err("wide Phase 0 coefficients must be rejected");

    assert!(
        matches!(
            err,
            EvalClaimError::WitnessCoeffOutOfRange {
                column_index: 0,
                row_index: 0,
                coeff_index: 0,
                ..
            }
        ),
        "unexpected range rejection error: {err}"
    );
}
