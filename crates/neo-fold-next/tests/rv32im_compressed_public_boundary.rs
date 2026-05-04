use neo_fold_next::finalize::FixedShapeChunkSummary;
use neo_fold_next::rv32im::{
    Rv32imChunkStepIvcStatement, Rv32imChunkStepPublic, Rv32imEncodedPublicInput, Rv32imIvcPublicImage,
    Rv32imMainRecursionConstruction2PublicBoundary,
};
use neo_math::{D, F};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

fn canonical_construction2_boundary(x_i: Rv32imEncodedPublicInput) -> Rv32imMainRecursionConstruction2PublicBoundary {
    let boundary = Rv32imMainRecursionConstruction2PublicBoundary {
        fresh_instance_digest: [0; 32],
        commitment_digest: [0; 32],
        commitment_d: D as u64,
        commitment_kappa: 1,
        commitment_data: vec![F::from_u64(11); D],
        x_i,
    };
    Rv32imMainRecursionConstruction2PublicBoundary {
        commitment_digest: boundary.expected_commitment_digest(),
        fresh_instance_digest: boundary.expected_fresh_instance_digest(),
        ..boundary
    }
}

fn verified_step_statement_digest(statement: &Rv32imChunkStepIvcStatement) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement");
    tr.append_message(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/version",
        b"v2",
    );
    tr.append_u64s(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/meta",
        &[
            statement.step_public.chunk_index,
            statement.step_public.step_lo,
            statement.step_public.step_hi,
            u64::from(statement.step_public.halted_out),
        ],
    );
    tr.append_fields(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/state_in",
        &digest32_as_fields(statement.step_public.state_in),
    );
    tr.append_fields(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/state_out",
        &digest32_as_fields(statement.step_public.state_out),
    );
    tr.append_fields(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/public_chunk_digest",
        &digest32_as_fields(statement.chunk_summary.public_chunk_digest),
    );
    tr.append_fields(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/chunk_relation_digest",
        &digest32_as_fields(statement.chunk_summary.chunk_relation_digest),
    );
    tr.digest32()
}

fn digest32_as_fields(digest: [u8; 32]) -> [F; 4] {
    [
        F::from_u64(u64::from_le_bytes(digest[0..8].try_into().expect("digest limb 0"))),
        F::from_u64(u64::from_le_bytes(digest[8..16].try_into().expect("digest limb 1"))),
        F::from_u64(u64::from_le_bytes(digest[16..24].try_into().expect("digest limb 2"))),
        F::from_u64(u64::from_le_bytes(digest[24..32].try_into().expect("digest limb 3"))),
    ]
}

fn canonical_public_image() -> Rv32imIvcPublicImage {
    let x_i = Rv32imEncodedPublicInput::from_digest_bytes([4; 32]);
    let terminal_statement = Rv32imChunkStepIvcStatement {
        step_public: Rv32imChunkStepPublic {
            program_digest: [7; 32],
            chunk_index: 0,
            step_lo: 0,
            step_hi: 2,
            state_in: [2; 32],
            state_out: [3; 32],
            halted_out: true,
        },
        chunk_summary: FixedShapeChunkSummary {
            start_index: 0,
            public_step_count: 2,
            public_chunk_digest: [8; 32],
            chunk_relation_digest: [9; 32],
        },
    };
    Rv32imIvcPublicImage {
        vk_fs_digest: [1; 32],
        chunk_count: 1,
        step_count: 2,
        z_0: [2; 32],
        z_i: [3; 32],
        pc: 1,
        x_i: x_i.clone(),
        construction2_u_i: canonical_construction2_boundary(x_i),
        folded_accumulator_digest: [5; 32],
        terminal_bridge_handoff_digest: [6; 32],
        terminal_verified_step_statement_digest: verified_step_statement_digest(&terminal_statement),
        terminal_statement: Some(terminal_statement),
    }
}

fn make_noncanonical_digest(digest: &mut [u8; 32]) {
    digest[..8].copy_from_slice(&u64::MAX.to_le_bytes());
}

fn expect_terminal_statement_digest_rejected(label: &str, mutate: fn(&mut Rv32imIvcPublicImage)) {
    let mut image = canonical_public_image();
    mutate(&mut image);
    let err = match image.validate_final_construction2_public_boundary() {
        Ok(()) => panic!("{label} must be rejected as a non-canonical field-limb digest"),
        Err(err) => err,
    };
    assert!(
        format!("{err}").contains("not a canonical four-limb field encoding"),
        "{label}: expected canonical-limb error, got: {err}"
    );
}

#[test]
fn rv32im_compressed_public_boundary_rejects_terminal_metadata_tamper() {
    let image = canonical_public_image();
    image
        .validate_final_construction2_public_boundary()
        .expect("canonical compressed public boundary must validate");

    let mut missing_terminal = image.clone();
    missing_terminal.terminal_statement = None;
    missing_terminal
        .validate_final_construction2_public_boundary()
        .expect_err("terminal metadata must be present");

    let mut unhalted = image.clone();
    unhalted
        .terminal_statement
        .as_mut()
        .expect("terminal statement")
        .step_public
        .halted_out = false;
    unhalted
        .validate_final_construction2_public_boundary()
        .expect_err("terminal chunk must be halted");

    let mut wrong_terminal_state = image.clone();
    wrong_terminal_state
        .terminal_statement
        .as_mut()
        .expect("terminal statement")
        .step_public
        .state_out[0] ^= 1;
    wrong_terminal_state
        .validate_final_construction2_public_boundary()
        .expect_err("terminal state_out must bind z_i");

    let mut wrong_step_hi = image.clone();
    wrong_step_hi
        .terminal_statement
        .as_mut()
        .expect("terminal statement")
        .step_public
        .step_hi ^= 1;
    wrong_step_hi
        .validate_final_construction2_public_boundary()
        .expect_err("terminal step_hi must close step_count");

    let mut zero_step_count = image.clone();
    zero_step_count.step_count = 0;
    zero_step_count
        .terminal_statement
        .as_mut()
        .expect("terminal statement")
        .step_public
        .step_hi = 0;
    zero_step_count
        .terminal_statement
        .as_mut()
        .expect("terminal statement")
        .chunk_summary
        .public_step_count = 0;
    zero_step_count.terminal_verified_step_statement_digest = verified_step_statement_digest(
        zero_step_count
            .terminal_statement
            .as_ref()
            .expect("terminal statement"),
    );
    zero_step_count
        .validate_final_construction2_public_boundary()
        .expect_err("terminal boundary must close at least one semantic step");

    let mut wrong_verified_step_digest = image.clone();
    wrong_verified_step_digest.terminal_verified_step_statement_digest[0] ^= 1;
    wrong_verified_step_digest
        .validate_final_construction2_public_boundary()
        .expect_err("terminal verified-step digest must bind terminal statement");
}

#[test]
fn rv32im_compressed_public_boundary_rejects_noncanonical_terminal_statement_digest_limbs() {
    expect_terminal_statement_digest_rejected("program_digest", |image| {
        make_noncanonical_digest(
            &mut image
                .terminal_statement
                .as_mut()
                .expect("terminal statement")
                .step_public
                .program_digest,
        );
    });
    expect_terminal_statement_digest_rejected("state_in", |image| {
        make_noncanonical_digest(
            &mut image
                .terminal_statement
                .as_mut()
                .expect("terminal statement")
                .step_public
                .state_in,
        );
    });
    expect_terminal_statement_digest_rejected("state_out", |image| {
        make_noncanonical_digest(
            &mut image
                .terminal_statement
                .as_mut()
                .expect("terminal statement")
                .step_public
                .state_out,
        );
    });
    expect_terminal_statement_digest_rejected("public_chunk_digest", |image| {
        make_noncanonical_digest(
            &mut image
                .terminal_statement
                .as_mut()
                .expect("terminal statement")
                .chunk_summary
                .public_chunk_digest,
        );
    });
    expect_terminal_statement_digest_rejected("chunk_relation_digest", |image| {
        make_noncanonical_digest(
            &mut image
                .terminal_statement
                .as_mut()
                .expect("terminal statement")
                .chunk_summary
                .chunk_relation_digest,
        );
    });
}

#[test]
fn rv32im_compressed_public_boundary_rejects_construction2_u_i_tamper() {
    let image = canonical_public_image();

    let mut wrong_x = image.clone();
    wrong_x.construction2_u_i.x_i = Rv32imEncodedPublicInput::from_digest_bytes([12; 32]);
    wrong_x
        .validate_final_construction2_public_boundary()
        .expect_err("Construction-2 u_i.x_i must match public x_i");

    let mut wrong_commitment_data = image.clone();
    wrong_commitment_data.construction2_u_i.commitment_data[0] += F::ONE;
    wrong_commitment_data
        .validate_final_construction2_public_boundary()
        .expect_err("Construction-2 u_i commitment digest must bind commitment data");

    let mut wrong_commitment_digest = image.clone();
    wrong_commitment_digest.construction2_u_i.commitment_digest[0] ^= 1;
    wrong_commitment_digest
        .validate_final_construction2_public_boundary()
        .expect_err("Construction-2 u_i fresh-instance digest must bind the commitment digest");

    let mut wrong_fresh_digest = image.clone();
    wrong_fresh_digest.construction2_u_i.fresh_instance_digest[0] ^= 1;
    wrong_fresh_digest
        .validate_final_construction2_public_boundary()
        .expect_err("Construction-2 u_i fresh-instance digest must bind commitment and x_i");

    let mut wrong_pc = image.clone();
    wrong_pc.pc = 2;
    wrong_pc
        .validate_final_construction2_public_boundary()
        .expect_err("pc must match the single RV32IM recursion lane");

    let mut wrong_commitment_d = image.clone();
    wrong_commitment_d.construction2_u_i.commitment_d = (D as u64) + 1;
    wrong_commitment_d.construction2_u_i.commitment_digest = wrong_commitment_d
        .construction2_u_i
        .expected_commitment_digest();
    wrong_commitment_d.construction2_u_i.fresh_instance_digest = wrong_commitment_d
        .construction2_u_i
        .expected_fresh_instance_digest();
    wrong_commitment_d
        .validate_final_construction2_public_boundary()
        .expect_err("Construction-2 u_i commitment d must match SuperNeo D");

    let mut zero_commitment_kappa = image.clone();
    zero_commitment_kappa.construction2_u_i.commitment_kappa = 0;
    zero_commitment_kappa
        .construction2_u_i
        .commitment_data
        .clear();
    zero_commitment_kappa.construction2_u_i.commitment_digest = zero_commitment_kappa
        .construction2_u_i
        .expected_commitment_digest();
    zero_commitment_kappa
        .construction2_u_i
        .fresh_instance_digest = zero_commitment_kappa
        .construction2_u_i
        .expected_fresh_instance_digest();
    zero_commitment_kappa
        .validate_final_construction2_public_boundary()
        .expect_err("Construction-2 u_i commitment kappa must be nonzero");

    let mut short_commitment_data = image.clone();
    short_commitment_data
        .construction2_u_i
        .commitment_data
        .pop();
    short_commitment_data.construction2_u_i.commitment_digest = short_commitment_data
        .construction2_u_i
        .expected_commitment_digest();
    short_commitment_data
        .construction2_u_i
        .fresh_instance_digest = short_commitment_data
        .construction2_u_i
        .expected_fresh_instance_digest();
    short_commitment_data
        .validate_final_construction2_public_boundary()
        .expect_err("Construction-2 u_i commitment data length must equal D * kappa");
}
