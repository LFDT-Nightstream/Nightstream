#[path = "support/rv64im_n2.rs"]
mod rv64im_n2_support;

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::{
    build_rv64im_chunk_step_ivc_published_target, build_rv64im_chunk_step_ivc_recursive_step_cover_shape,
    build_rv64im_chunk_step_ivc_recursive_step_padding, build_rv64im_chunk_step_ivc_relations,
    build_rv64im_chunk_step_ivc_shape, validate_rv64im_chunk_step_ivc_published_statement,
    Rv64imChunkStepIvcPublishedTarget,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, parity_source_cases, prove_rv64im_public_proof_with_options,
    Rv64imProofInput, Rv64imPublicProofOptions,
};

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
