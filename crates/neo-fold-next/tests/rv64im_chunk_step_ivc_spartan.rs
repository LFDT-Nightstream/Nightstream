#![allow(dead_code)]

#[path = "support/rv64im_n2.rs"]
mod rv64im_n2_support;

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::{
    build_rv64im_chunk_step_ivc_published_target, build_rv64im_chunk_step_ivc_recursive_step_cover_shape,
    build_rv64im_chunk_step_ivc_recursive_step_padding, build_rv64im_chunk_step_ivc_relations,
    build_rv64im_chunk_step_ivc_shape, prove_rv64im_chunk_step_ivc_spartan, prove_rv64im_chunk_step_ivc_spartan_chain,
    setup_rv64im_chunk_step_ivc_spartan_cached, validate_rv64im_chunk_step_ivc_published_statement,
    verify_rv64im_chunk_step_ivc_spartan, verify_rv64im_chunk_step_ivc_spartan_chain,
    Rv64imChunkStepIvcPublishedTarget, Rv64imChunkStepIvcShape, Rv64imChunkStepIvcSpartanError,
    Rv64imChunkStepIvcStatement, Rv64imChunkStepIvcWitness,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, parity_source_cases, prove_rv64im_public_proof_with_options,
    Rv64imProofInput, Rv64imPublicProofOptions,
};
use neo_transcript::{Poseidon2Transcript, Transcript};

fn chunk_step_setup_identity(statement: &Rv64imChunkStepIvcStatement, witness: &Rv64imChunkStepIvcWitness) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/tests/rv64im/chunk_step_ivc/setup_identity");
    tr.append_message(
        b"neo.fold.next/tests/rv64im/chunk_step_ivc/setup_identity/shape",
        &build_rv64im_chunk_step_ivc_shape(statement, witness)
            .expect("build chunk-step shape")
            .expected_digest(),
    );
    tr.append_message(
        b"neo.fold.next/tests/rv64im/chunk_step_ivc/setup_identity/statement",
        &bincode::serialize(statement).expect("serialize chunk-step statement"),
    );
    tr.append_message(
        b"neo.fold.next/tests/rv64im/chunk_step_ivc/setup_identity/state_in",
        &bincode::serialize(&witness.state_in).expect("serialize chunk-step state_in"),
    );
    tr.append_message(
        b"neo.fold.next/tests/rv64im/chunk_step_ivc/setup_identity/state_out",
        &bincode::serialize(&witness.state_out).expect("serialize chunk-step state_out"),
    );
    tr.append_message(
        b"neo.fold.next/tests/rv64im/chunk_step_ivc/setup_identity/public_chunk_digest",
        &witness.handoff.public_chunk_digest,
    );
    tr.append_message(
        b"neo.fold.next/tests/rv64im/chunk_step_ivc/setup_identity/bridge_handoff_digest",
        &witness.handoff.bridge_handoff.digest,
    );
    for digest in &witness.handoff.prepared_step_digests {
        tr.append_message(
            b"neo.fold.next/tests/rv64im/chunk_step_ivc/setup_identity/prepared_step_digest",
            digest,
        );
    }
    tr.append_message(
        b"neo.fold.next/tests/rv64im/chunk_step_ivc/setup_identity/replay_witness",
        &bincode::serialize(&witness.replay_witness).expect("serialize replay witness"),
    );
    tr.digest32()
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_chunk_step_ivc_spartan_round_trip() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relation = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations")
        .into_iter()
        .next()
        .expect("first relation");

    let keys = setup_rv64im_chunk_step_ivc_spartan_cached(&relation.statement, &relation.witness)
        .expect("setup chunk-step IVC");
    let (pk, vk) = &*keys;
    let proof =
        prove_rv64im_chunk_step_ivc_spartan(pk, &relation.statement, &relation.witness).expect("prove chunk-step IVC");
    verify_rv64im_chunk_step_ivc_spartan(vk, &relation.statement, &proof).expect("verify chunk-step IVC");
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_chunk_step_ivc_spartan_rejects_tampered_public_io() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relation = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations")
        .into_iter()
        .next()
        .expect("first relation");

    let keys = setup_rv64im_chunk_step_ivc_spartan_cached(&relation.statement, &relation.witness)
        .expect("setup chunk-step IVC");
    let (pk, vk) = &*keys;
    let proof =
        prove_rv64im_chunk_step_ivc_spartan(pk, &relation.statement, &relation.witness).expect("prove chunk-step IVC");

    let mut tampered_statement = relation.statement.clone();
    tampered_statement.step_public.state_out[0] ^= 1;

    let err = verify_rv64im_chunk_step_ivc_spartan(vk, &tampered_statement, &proof)
        .expect_err("tampered step public IO must fail");
    assert!(matches!(err, Rv64imChunkStepIvcSpartanError::PublicIoMismatch));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_chunk_step_ivc_spartan_cached_setup_reuses_same_relation() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relation = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations")
        .into_iter()
        .next()
        .expect("first relation");

    let first = setup_rv64im_chunk_step_ivc_spartan_cached(&relation.statement, &relation.witness)
        .expect("setup cached chunk-step IVC first");
    let second = setup_rv64im_chunk_step_ivc_spartan_cached(&relation.statement, &relation.witness)
        .expect("setup cached chunk-step IVC second");
    assert_eq!(
        chunk_step_setup_identity(&relation.statement, &relation.witness),
        chunk_step_setup_identity(&relation.statement, &relation.witness),
        "exact same relation must produce the same deterministic setup identity"
    );
    let proof =
        prove_rv64im_chunk_step_ivc_spartan(&first.as_ref().0, &relation.statement, &relation.witness).expect("prove");
    verify_rv64im_chunk_step_ivc_spartan(&second.as_ref().1, &relation.statement, &proof).expect("verify");
}

#[test]
fn rv64im_chunk_step_ivc_shape_ignores_tampered_legacy_statement_halted_out() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relation = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations")
        .into_iter()
        .next()
        .expect("first relation");
    let original_shape =
        build_rv64im_chunk_step_ivc_shape(&relation.statement, &relation.witness).expect("build original shape");
    let mut tampered_statement = relation.statement.clone();
    tampered_statement.step_public.halted_out = !tampered_statement.step_public.halted_out;
    let tampered_shape =
        build_rv64im_chunk_step_ivc_shape(&tampered_statement, &relation.witness).expect("build tampered shape");

    assert_eq!(
        original_shape, tampered_shape,
        "chunk-step IVC shape must derive terminality from the authoritative witness, not legacy statement.step_public.halted_out"
    );
}

#[test]
fn rv64im_chunk_step_ivc_shape_rejects_statement_chunk_index_drift() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relation = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations")
        .into_iter()
        .next()
        .expect("first relation");
    let mut tampered_statement = relation.statement.clone();
    tampered_statement.step_public.chunk_index ^= 1;

    assert!(
        build_rv64im_chunk_step_ivc_shape(&tampered_statement, &relation.witness).is_err(),
        "chunk-step IVC shape builder must reject step_public.chunk_index drift from the authoritative bridge handoff"
    );
}

#[test]
fn rv64im_chunk_step_ivc_shape_rejects_legacy_statement_shell_drift() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relation = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations")
        .into_iter()
        .next()
        .expect("first relation");
    let mut tampered_statement = relation.statement.clone();
    tampered_statement.step_public.step_hi ^= 1;
    tampered_statement.step_public.state_out[0] ^= 1;
    tampered_statement.chunk_summary.public_chunk_digest[0] ^= 1;

    assert!(
        build_rv64im_chunk_step_ivc_shape(&tampered_statement, &relation.witness).is_err(),
        "chunk-step IVC shape builder must reject legacy statement shell drift from the authoritative published statement"
    );
}

#[test]
fn rv64im_chunk_step_ivc_published_statement_rejects_internal_summary_drift() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relation = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations")
        .into_iter()
        .next()
        .expect("first relation");
    let mut tampered_statement = relation.statement.clone();
    tampered_statement.chunk_summary.start_index ^= 1;

    assert!(
        validate_rv64im_chunk_step_ivc_published_statement(&tampered_statement).is_err(),
        "chunk-step IVC published statement validator must reject summary start drift from step_public.step_lo"
    );
}

#[test]
fn rv64im_chunk_step_ivc_published_target_matches_statement_shell() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relation = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations")
        .into_iter()
        .next()
        .expect("first relation");

    let target: Rv64imChunkStepIvcPublishedTarget =
        build_rv64im_chunk_step_ivc_published_target(&relation.statement).expect("build published target");

    assert_eq!(target.program_digest, relation.statement.step_public.program_digest);
    assert_eq!(target.chunk_index, relation.statement.step_public.chunk_index);
    assert_eq!(target.step_lo, relation.statement.step_public.step_lo);
    assert_eq!(target.step_hi, relation.statement.step_public.step_hi);
    assert_eq!(target.halted_out, relation.statement.step_public.halted_out);
    assert_eq!(target.state_in, relation.statement.step_public.state_in);
    assert_eq!(target.state_out, relation.statement.step_public.state_out);
    assert_eq!(target.summary_start, relation.statement.chunk_summary.start_index);
    assert_eq!(
        target.summary_step_count,
        relation.statement.chunk_summary.public_step_count
    );
    assert_eq!(
        target.public_chunk_digest,
        relation.statement.chunk_summary.public_chunk_digest
    );
    assert_eq!(
        target.chunk_relation_digest,
        relation.statement.chunk_summary.chunk_relation_digest
    );
    assert_eq!(target.chunk_summary(), relation.statement.chunk_summary);
    assert_eq!(target.expected_digest(), relation.statement.expected_digest());
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_chunk_step_ivc_spartan_cached_setup_does_not_reuse_when_only_shape_matches() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relation = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations")
        .into_iter()
        .next()
        .expect("first relation");
    let mut tampered_statement = relation.statement.clone();
    tampered_statement.step_public.program_digest[0] ^= 1;

    let original_shape: Rv64imChunkStepIvcShape =
        build_rv64im_chunk_step_ivc_shape(&relation.statement, &relation.witness).expect("build original shape");
    let tampered_shape =
        build_rv64im_chunk_step_ivc_shape(&tampered_statement, &relation.witness).expect("build tampered shape");
    assert_eq!(
        original_shape, tampered_shape,
        "changing only the non-shape public surface should not alter the coarse chunk-step shape"
    );

    assert_ne!(
        chunk_step_setup_identity(&relation.statement, &relation.witness),
        chunk_step_setup_identity(&tampered_statement, &relation.witness),
        "distinct relations with the same coarse shape must still produce distinct deterministic setup identities"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_chunk_step_ivc_spartan_rejects_internally_inconsistent_public_statement() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relation = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations")
        .into_iter()
        .next()
        .expect("first relation");

    let keys = setup_rv64im_chunk_step_ivc_spartan_cached(&relation.statement, &relation.witness)
        .expect("setup chunk-step IVC");
    let (pk, vk) = &*keys;
    let proof =
        prove_rv64im_chunk_step_ivc_spartan(pk, &relation.statement, &relation.witness).expect("prove chunk-step IVC");

    let mut tampered_statement = relation.statement.clone();
    tampered_statement.chunk_summary.start_index ^= 1;

    let err = verify_rv64im_chunk_step_ivc_spartan(vk, &tampered_statement, &proof)
        .expect_err("internally inconsistent public statement must fail before public IO comparison");
    assert!(
        matches!(err, Rv64imChunkStepIvcSpartanError::Verify(_)),
        "internally inconsistent public statement must be rejected as an invalid published statement, not only as a public IO mismatch"
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_chunk_step_ivc_spartan_chunked_chain_round_trip() {
    let source = parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == "control_flow_jal_skip_ecall")
        .expect("control-flow parity source case");
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof = prove_rv64im_public_proof_with_options(&input, options).expect("prove chunked public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
    let relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step IVC relations");
    assert!(relations.len() > 1, "expected multiple chunk-step relations");

    let proof = prove_rv64im_chunk_step_ivc_spartan_chain(&relations).expect("prove chunk-step IVC chain");
    verify_rv64im_chunk_step_ivc_spartan_chain(&relations, &proof).expect("verify chunk-step IVC chain");
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_chunk_step_ivc_spartan_chain_rejects_length_mismatch() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations");
    let mut proof = prove_rv64im_chunk_step_ivc_spartan_chain(&relations).expect("prove chunk-step IVC chain");
    proof.step_proofs.pop();

    let err = verify_rv64im_chunk_step_ivc_spartan_chain(&relations, &proof)
        .expect_err("chunk-step IVC chain length mismatch must fail");
    assert!(matches!(err, Rv64imChunkStepIvcSpartanError::ChainLengthMismatch));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_chunk_step_ivc_recursive_cover_shape_covers_multi_step_chain() {
    let source = parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == "control_flow_jal_skip_ecall")
        .expect("control-flow parity source case");
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof = prove_rv64im_public_proof_with_options(&input, options).expect("prove chunked public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
    let relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step IVC relations");
    assert!(relations.len() > 1, "expected multiple chunk-step relations");

    let cover_shape =
        build_rv64im_chunk_step_ivc_recursive_step_cover_shape(&relations).expect("build recursive-step cover shape");
    assert!(
        !cover_shape.terminal_step,
        "recursive-step cover should treat terminality as a selector, not a separate shape"
    );

    let mut saw_terminal = false;
    let mut saw_non_terminal = false;
    for relation in &relations {
        let step_shape =
            build_rv64im_chunk_step_ivc_shape(&relation.statement, &relation.witness).expect("build per-step shape");
        saw_terminal |= step_shape.terminal_step;
        saw_non_terminal |= !step_shape.terminal_step;
        assert!(
            cover_shape.covers_recursive_step_shape(&step_shape),
            "recursive-step cover must dominate every per-step chunk shape"
        );
    }
    assert!(saw_terminal, "expected a terminal chunk-step shape");
    assert!(saw_non_terminal, "expected a non-terminal chunk-step shape");
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_chunk_step_ivc_recursive_padding_lifts_each_step_to_cover_shape() {
    let source = parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == "control_flow_jal_skip_ecall")
        .expect("control-flow parity source case");
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof = prove_rv64im_public_proof_with_options(&input, options).expect("prove chunked public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
    let relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step IVC relations");
    let cover_shape =
        build_rv64im_chunk_step_ivc_recursive_step_cover_shape(&relations).expect("build recursive-step cover shape");

    for relation in &relations {
        let step_shape =
            build_rv64im_chunk_step_ivc_shape(&relation.statement, &relation.witness).expect("build per-step shape");
        let padding =
            build_rv64im_chunk_step_ivc_recursive_step_padding(&relation.statement, &relation.witness, &cover_shape)
                .expect("build recursive-step padding");

        assert_eq!(
            step_shape.state_in_claim_count + padding.state_in_claim_pad,
            cover_shape.state_in_claim_count
        );
        assert_eq!(
            step_shape.state_out_claim_count + padding.state_out_claim_pad,
            cover_shape.state_out_claim_count
        );
        assert_eq!(
            step_shape.fresh_claim_count + padding.fresh_claim_pad,
            cover_shape.fresh_claim_count
        );
        assert_eq!(
            step_shape.fresh_witness_count + padding.fresh_witness_pad,
            cover_shape.fresh_witness_count
        );
        assert_eq!(
            step_shape.ccs_output_count + padding.ccs_output_pad,
            cover_shape.ccs_output_count
        );
        assert_eq!(step_shape.child_count + padding.child_pad, cover_shape.child_count);
        assert_eq!(
            step_shape.fe_round_lengths.len() as u64 + padding.fe_round_count_pad,
            cover_shape.fe_round_lengths.len() as u64
        );
        assert_eq!(
            step_shape.nc_round_lengths.len() as u64 + padding.nc_round_count_pad,
            cover_shape.nc_round_lengths.len() as u64
        );
        for (idx, cover_len) in cover_shape.fe_round_lengths.iter().enumerate() {
            assert_eq!(
                step_shape.fe_round_lengths.get(idx).copied().unwrap_or(0) + padding.fe_round_coeff_pad[idx],
                *cover_len
            );
        }
        for (idx, cover_len) in cover_shape.nc_round_lengths.iter().enumerate() {
            assert_eq!(
                step_shape.nc_round_lengths.get(idx).copied().unwrap_or(0) + padding.nc_round_coeff_pad[idx],
                *cover_len
            );
        }
    }
}
