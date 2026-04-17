#![allow(dead_code)]

#[path = "support/rv64im_n2.rs"]
mod rv64im_n2_support;

use neo_fold_next::rv64im::final_relation::build_rv64im_chunk_fold_freshs;
use neo_fold_next::rv64im::final_relation::build_rv64im_chunk_fold_step_traces;
use neo_fold_next::rv64im::final_relation::build_rv64im_chunk_step_publics;
use neo_fold_next::rv64im::final_relation::rv64im_chunk_fold_initial_transcript_snapshot;
use neo_fold_next::rv64im::rv64im_chunk_fold_seed;

#[test]
fn rv64im_chunk_step_publics_chain_terminal_handles() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let step_publics =
        build_rv64im_chunk_step_publics(&fixture.final_statement, &fixture.final_proof).expect("build step publics");

    assert_eq!(
        step_publics.len(),
        fixture.final_proof.steps.len(),
        "chunk-step public count must match the recursive step count"
    );
    assert!(
        !step_publics.is_empty(),
        "the mixed-opcode fixture should produce at least one recursive step"
    );
    assert_eq!(
        step_publics[0].state_in,
        rv64im_chunk_fold_seed(),
        "the first chunk-step must start from the fixed recursive seed handle"
    );
    assert_eq!(
        step_publics.last().expect("last step").state_out,
        fixture
            .final_statement
            .folded
            .final_accumulator
            .terminal_handle
            .0,
        "the final chunk-step must end at the folded statement terminal handle"
    );

    for (chunk_index, step) in step_publics.iter().enumerate() {
        assert_eq!(
            step.chunk_index as usize, chunk_index,
            "chunk indices must be canonical"
        );
        assert!(
            step.step_hi >= step.step_lo,
            "chunk-step bounds must be monotone within each chunk"
        );
        if chunk_index + 1 < step_publics.len() {
            assert_eq!(
                step.state_out,
                step_publics[chunk_index + 1].state_in,
                "chunk-step handles must chain across the recursive carry"
            );
        }
    }
}

#[test]
fn rv64im_chunk_fold_freshs_match_verified_chunk_layout() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let freshs =
        build_rv64im_chunk_fold_freshs(&fixture.final_statement, &fixture.final_proof).expect("build chunk freshs");
    let step_publics =
        build_rv64im_chunk_step_publics(&fixture.final_statement, &fixture.final_proof).expect("build step publics");

    assert_eq!(
        freshs.len(),
        fixture.final_proof.chunk_summaries.len(),
        "chunk-fold fresh count must match the finalized chunk summary count"
    );
    assert_eq!(
        freshs.len(),
        step_publics.len(),
        "chunk-fold freshs and chunk-step publics must describe the same verified chunks"
    );

    for (chunk_index, fresh) in freshs.iter().enumerate() {
        let summary = &fixture.final_proof.chunk_summaries[chunk_index];
        let step_public = &step_publics[chunk_index];
        assert_eq!(
            fresh.public_chunk.start_index as u64, summary.start_index,
            "chunk-fold fresh start index must match the verified chunk summary"
        );
        assert_eq!(
            fresh.public_chunk.steps.len() as u64,
            summary.public_step_count,
            "chunk-fold fresh step count must match the verified chunk summary"
        );
        assert_eq!(
            fresh.public_chunk_digest, summary.public_chunk_digest,
            "chunk-fold fresh digest must match the verified chunk summary"
        );
        assert_eq!(
            fresh.public_chunk.start_index as u64, step_public.step_lo,
            "chunk-fold fresh start index must match the chunk-step lower bound"
        );
        assert_eq!(
            fresh.public_chunk.start_index as u64 + fresh.public_chunk.steps.len() as u64,
            step_public.step_hi,
            "chunk-fold fresh step span must match the chunk-step upper bound"
        );
        assert_eq!(
            fresh.fresh_claims.len(),
            fresh.public_chunk.steps.len(),
            "chunk-fold fresh claim count must match the carried public chunk steps"
        );
        assert_eq!(
            fresh.fresh_witnesses.len(),
            fresh.public_chunk.steps.len(),
            "chunk-fold fresh witness count must match the carried public chunk steps"
        );
    }
}

#[test]
fn rv64im_chunk_fold_step_traces_chain_verified_carries() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let traces = build_rv64im_chunk_fold_step_traces(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-fold step traces");

    assert_eq!(
        traces.len(),
        fixture.final_proof.steps.len(),
        "chunk-fold step trace count must match the carried replay witness count"
    );
    assert!(!traces.is_empty(), "expected at least one verified chunk-fold step");
    assert_eq!(
        traces[0].carry_in.terminal_handle.0,
        rv64im_chunk_fold_seed(),
        "the first verified chunk-fold step must begin at the fixed recursive seed"
    );
    assert_eq!(
        traces.last().expect("last step").carry_out.terminal_handle,
        fixture
            .final_statement
            .folded
            .final_accumulator
            .terminal_handle,
        "the last verified chunk-fold step must end at the folded statement terminal handle"
    );

    for (chunk_index, trace) in traces.iter().enumerate() {
        assert_eq!(
            trace.step_public.chunk_index as usize, chunk_index,
            "verified chunk-fold step indices must stay canonical"
        );
        assert_eq!(
            trace.chunk_summary.start_index, trace.step_public.step_lo,
            "verified chunk summary and step public must agree on step_lo"
        );
        assert_eq!(
            trace.chunk_summary.start_index + trace.chunk_summary.public_step_count,
            trace.step_public.step_hi,
            "verified chunk summary and step public must agree on step_hi"
        );
        assert_eq!(
            trace.carry_in.terminal_handle.0, trace.step_public.state_in,
            "verified carry_in must match the chunk-step public input state"
        );
        assert_eq!(
            trace.carry_out.terminal_handle.0, trace.step_public.state_out,
            "verified carry_out must match the chunk-step public output state"
        );
        if chunk_index + 1 < traces.len() {
            assert_eq!(
                trace.carry_out.terminal_handle,
                traces[chunk_index + 1].carry_in.terminal_handle,
                "verified chunk-fold terminal handles must chain between adjacent steps"
            );
            assert_eq!(
                trace.carry_out.main.claims,
                traces[chunk_index + 1].carry_in.main.claims,
                "verified chunk-fold carried claims must chain between adjacent steps"
            );
            assert_eq!(
                trace.carry_out.main.witnesses,
                traces[chunk_index + 1].carry_in.main.witnesses,
                "verified chunk-fold carried witnesses must chain between adjacent steps"
            );
        }
    }
}

#[test]
fn rv64im_chunk_fold_step_traces_chain_transcript_state() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let traces = build_rv64im_chunk_fold_step_traces(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-fold step traces");

    assert!(!traces.is_empty(), "expected at least one verified chunk-fold step");
    assert_eq!(
        traces[0].transcript_in,
        rv64im_chunk_fold_initial_transcript_snapshot(),
        "the first verified step must start from the canonical chunk-fold transcript seed"
    );

    for (chunk_index, trace) in traces.iter().enumerate() {
        assert_ne!(
            trace.transcript_in, trace.transcript_out,
            "verified chunk-fold step {chunk_index} must advance the running transcript state"
        );
        assert_eq!(
            trace.state_in().transcript,
            trace.transcript_in,
            "state_in() must expose the same transcript snapshot captured before verification"
        );
        assert_eq!(
            trace.state_out().transcript,
            trace.transcript_out,
            "state_out() must expose the same transcript snapshot captured after verification"
        );
        if chunk_index + 1 < traces.len() {
            assert_eq!(
                trace.transcript_out,
                traces[chunk_index + 1].transcript_in,
                "verified chunk-fold transcript snapshots must chain between adjacent steps"
            );
        }
    }
}
