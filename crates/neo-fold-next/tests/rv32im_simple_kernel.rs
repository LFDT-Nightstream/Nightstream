//! Focused tests for the live RV32IM simple-kernel proof boundary.

use neo_fold_next::core::proof::StepInput;
use neo_fold_next::rv32im::ccs::RV32IM_ROOT_ROW_WIDTH;
use neo_fold_next::rv32im::kernel::{AjtaiFamilyKind, OpeningPointLabel, SelectedOpeningRef};
use neo_fold_next::rv32im::{
    parity_source_cases, prove_packaged_simple_kernel, prove_simple_kernel, verify_packaged_simple_kernel,
    verify_simple_kernel, SimpleKernelProverInput, SimpleKernelPublicInput, SimpleKernelVerifierInput,
};

fn source_case(name: &str) -> neo_fold_next::rv32im::Rv32imParitySourceCase {
    parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == name)
        .unwrap_or_else(|| panic!("missing parity source case {name}"))
}

fn public_input(name: &str) -> SimpleKernelPublicInput {
    let source = source_case(name);
    let max_steps = source.program_words.len();
    SimpleKernelPublicInput { source, max_steps }
}

fn same_public_step(
    lhs: &neo_fold_next::core::proof::PublicStep,
    rhs: &neo_fold_next::core::proof::PublicStep,
) -> bool {
    lhs.label == rhs.label
        && lhs.mcs.m_in == rhs.mcs.m_in
        && lhs.mcs.x == rhs.mcs.x
        && lhs.mcs.c.d == rhs.mcs.c.d
        && lhs.mcs.c.kappa == rhs.mcs.c.kappa
        && lhs.mcs.c.data == rhs.mcs.c.data
}

#[test]
fn simple_kernel_roundtrip_exports_one_prepared_step_per_execution_row() {
    let public = public_input("vertical_add_sw_lw_ecall");
    let prover = SimpleKernelProverInput { public: public.clone() };
    let verifier = SimpleKernelVerifierInput { public };

    let (output, proof) = prove_simple_kernel(&prover).expect("prove simple kernel");
    let verified = verify_simple_kernel(&verifier, &proof).expect("verify simple kernel");
    let public_steps = output
        .prepared_steps
        .iter()
        .map(StepInput::instance)
        .collect::<Vec<_>>();
    let main_lane_family = neo_fold_next::rv32im::kernel::build_main_lane_family_summary(&public_steps);
    assert_eq!(proof.trace, output.trace);
    assert_eq!(proof.stages, output.stages);
    assert_eq!(proof.stage_claims, output.stage_claims);
    assert_eq!(proof.stage_packages.digest, output.stage_packages.digest);
    assert_eq!(proof.kernel_opening.digest, output.kernel_opening.digest);
    assert_eq!(proof.kernel_claims, output.kernel_claims);
    assert_eq!(proof.root_lane_columns, output.root_lane_columns);
    assert_eq!(proof.root_lane_commitment, output.root_lane_commitment);
    assert_eq!(verified.trace, output.trace);
    assert_eq!(verified.stages, output.stages);
    assert_eq!(verified.stage_claims, output.stage_claims);
    assert_eq!(verified.stage_packages.digest, output.stage_packages.digest);
    assert_eq!(verified.kernel_opening.digest, output.kernel_opening.digest);
    assert_eq!(verified.kernel_claims, output.kernel_claims);
    assert_eq!(verified.root_lane_columns, output.root_lane_columns);
    assert_eq!(verified.root_lane_commitment, output.root_lane_commitment);
    assert_eq!(output.prepared_steps.len(), output.trace.execution_rows.len());
    assert_eq!(public_steps.len(), output.prepared_steps.len());
    assert_eq!(
        output.kernel_claims.prepared_step_bindings.bindings.len(),
        output.prepared_steps.len()
    );

    for ((row, step), binding) in output
        .trace
        .execution_rows
        .iter()
        .zip(output.prepared_steps.iter())
        .zip(output.kernel_claims.prepared_step_bindings.bindings.iter())
    {
        assert_eq!(binding.trace_index, row.trace_index);
        let expected_row_opening = SelectedOpeningRef::from_parts(
            output.root_lane_columns.object.family,
            output.root_lane_columns.object.commitment_digest,
            output.root_lane_columns.object.layout_version,
            row.trace_index as u64,
            binding.row_digest,
        );
        assert_eq!(binding.row_opening_digest, expected_row_opening.digest);
        assert_ne!(binding.row_digest, [0; 32]);
        assert!(same_public_step(&step.instance(), &public_steps[row.trace_index]));
    }

    assert_eq!(
        output.stage_claims.stage1.claim.row_count,
        output.stages.stage1.rows.len()
    );
    assert_eq!(
        output.stage_claims.stage2.claim.register_read_count,
        output.stages.stage2.register_reads.len()
    );
    assert_eq!(
        output.stage_claims.stage2.claim.register_write_count,
        output.stages.stage2.register_writes.len()
    );
    assert_eq!(
        output.stage_claims.stage2.claim.ram_event_count,
        output.stages.stage2.ram_events.len()
    );
    assert_eq!(
        output.stage_claims.stage2.claim.twist_link_count,
        output.stages.stage2.twist_links.len()
    );
    assert_eq!(
        output.stage_claims.stage3.claim.continuity_count,
        output.stages.stage3.continuity.len()
    );
    assert_eq!(
        output.stage_claims.transcript.claim.event_count,
        output.stages.transcript.events.len()
    );
    assert!(output
        .stages
        .stage3
        .continuity
        .iter()
        .all(|event| event.continuity_holds));
    assert!(output.stage_claims.stage3.claim.all_continuity_hold);
    assert_ne!(output.stage_claims.stage1.rows.rows_digest, [0; 32]);
    assert_ne!(output.stage_claims.stage1.rows.digest, [0; 32]);
    assert_ne!(output.stage_claims.stage2.families.register_reads_digest, [0; 32]);
    assert_ne!(output.stage_claims.stage2.families.register_writes_digest, [0; 32]);
    assert_ne!(output.stage_claims.stage2.families.ram_events_digest, [0; 32]);
    assert_ne!(output.stage_claims.stage2.families.twist_links_digest, [0; 32]);
    assert_ne!(output.stage_claims.stage2.families.digest, [0; 32]);
    assert_ne!(output.stage_claims.stage3.continuity.continuity_digest, [0; 32]);
    assert_ne!(output.stage_claims.stage3.continuity.digest, [0; 32]);
    assert_ne!(output.stage_claims.digest, [0; 32]);
    assert_ne!(output.stage_packages.stage1.digest, [0; 32]);
    assert_ne!(output.stage_packages.stage1.claim.digest(), [0; 32]);
    assert_eq!(
        output.stage_packages.stage1.claim.labels(),
        vec![
            OpeningPointLabel::Stage1First,
            OpeningPointLabel::Stage1Effect,
            OpeningPointLabel::Stage1Commit,
            OpeningPointLabel::Stage1Last,
        ]
    );
    assert_ne!(output.stage_packages.stage2.digest, [0; 32]);
    assert_ne!(output.stage_packages.stage2.claim.digest(), [0; 32]);
    assert_ne!(output.stage_packages.stage3.digest, [0; 32]);
    assert_ne!(output.stage_packages.stage3.claim.digest(), [0; 32]);
    assert_ne!(output.stage_packages.digest, [0; 32]);
    assert_eq!(
        output
            .kernel_opening
            .claim
            .bindings
            .stage_claim_bundle_digest,
        output.stage_claims.digest
    );
    assert_eq!(
        output
            .kernel_opening
            .claim
            .bindings
            .stage_package_bundle_digest,
        output.stage_packages.digest
    );
    assert_eq!(
        output
            .kernel_opening
            .claim
            .prepared_steps
            .prepared_step_count as usize,
        output.prepared_steps.len()
    );
    assert_eq!(
        output
            .kernel_opening
            .claim
            .prepared_steps
            .points
            .first_prepared_step
            .as_ref()
            .map(|reference| reference.id.object.family),
        Some(AjtaiFamilyKind::RootMainLaneCommittedRows)
    );
    assert_eq!(
        output
            .kernel_opening
            .claim
            .prepared_steps
            .points
            .first_prepared_step
            .as_ref()
            .map(|reference| reference.id.object.commitment_digest),
        Some(output.root_lane_commitment.commitments.digest)
    );
    assert_eq!(
        output
            .kernel_opening
            .claim
            .prepared_steps
            .points
            .last_prepared_step
            .as_ref()
            .map(|reference| reference.id.object.family),
        Some(AjtaiFamilyKind::RootMainLaneCommittedRows)
    );
    assert_ne!(output.kernel_opening.claim.digest(), [0; 32]);
    assert_ne!(output.kernel_opening.bindings.digest, [0; 32]);
    assert_ne!(output.kernel_opening.prepared_steps.digest, [0; 32]);
    assert_eq!(
        output.root_lane_columns.time_len as usize,
        output.trace.execution_rows.len()
    );
    assert_eq!(output.root_lane_columns.row_width, RV32IM_ROOT_ROW_WIDTH as u64);
    assert_eq!(output.root_lane_columns.column_digests.len(), RV32IM_ROOT_ROW_WIDTH);
    assert_eq!(output.root_lane_commitment.row_width, RV32IM_ROOT_ROW_WIDTH as u64);
    assert_eq!(
        output.root_lane_commitment.time_len as usize,
        output.trace.execution_rows.len()
    );
    assert_eq!(
        output.root_lane_commitment.commitments.commitments.len(),
        RV32IM_ROOT_ROW_WIDTH
    );
    assert!(output
        .root_lane_commitment
        .padded_time_len
        .is_power_of_two());
    assert_ne!(output.root_lane_commitment.commitments.digest, [0; 32]);
    assert_ne!(output.root_lane_commitment.digest, [0; 32]);
    assert_eq!(
        output
            .root_lane_commitment
            .first_opening
            .as_ref()
            .map(|opening| opening.logical_index),
        Some(0)
    );
    assert_eq!(
        output
            .root_lane_commitment
            .last_opening
            .as_ref()
            .map(|opening| opening.logical_index),
        Some(output.trace.execution_rows.len().saturating_sub(1) as u64)
    );
    assert_eq!(
        output
            .root_lane_columns
            .first_row
            .as_ref()
            .map(|reference| reference.id.logical_index),
        Some(0)
    );
    assert_eq!(
        output
            .root_lane_columns
            .last_row
            .as_ref()
            .map(|reference| reference.id.logical_index),
        Some(output.trace.execution_rows.len().saturating_sub(1) as u64)
    );
    assert_eq!(main_lane_family.public_step_count as usize, public_steps.len());
    assert_eq!(main_lane_family.row_width, RV32IM_ROOT_ROW_WIDTH as u64);
    assert_eq!(
        main_lane_family
            .first_public_step
            .as_ref()
            .map(|reference| reference.id.logical_index),
        Some(0)
    );
    assert_eq!(
        main_lane_family
            .last_public_step
            .as_ref()
            .map(|reference| reference.id.logical_index),
        Some(public_steps.len().saturating_sub(1) as u64)
    );
    assert_eq!(
        output.kernel_opening.claim.labels(),
        vec![
            OpeningPointLabel::KernelFirstBinding,
            OpeningPointLabel::KernelLastBinding,
            OpeningPointLabel::KernelFirstPreparedStep,
            OpeningPointLabel::KernelLastPreparedStep,
        ]
    );
    assert_ne!(output.kernel_opening.digest, [0; 32]);
}

#[test]
fn simple_kernel_packaged_roundtrip_matches_exported_public_steps() {
    let public = public_input("multiply_high_mulh_mulhu_mulhsu_ecall");
    let prover = SimpleKernelProverInput { public: public.clone() };
    let verifier = SimpleKernelVerifierInput { public };

    let (output, packaged) = prove_packaged_simple_kernel(&prover).expect("prove packaged simple kernel");
    let verified = verify_packaged_simple_kernel(&verifier, &packaged).expect("verify packaged simple kernel");
    assert_eq!(verified.trace, output.trace);
    assert_eq!(verified.stages, output.stages);
    assert_eq!(verified.stage_claims, output.stage_claims);
    assert_eq!(verified.stage_packages.digest, output.stage_packages.digest);
    assert_eq!(verified.kernel_opening.digest, output.kernel_opening.digest);
    assert_eq!(verified.kernel_claims, output.kernel_claims);
    assert_eq!(verified.root_lane_columns, output.root_lane_columns);
    assert_eq!(verified.root_lane_commitment, output.root_lane_commitment);
    assert!(output
        .kernel_claims
        .prepared_step_bindings
        .bindings
        .is_empty());
    assert!(verified
        .kernel_claims
        .prepared_step_bindings
        .bindings
        .is_empty());
    assert_eq!(
        packaged.main_lane.binding.root_lane_columns_digest,
        output.root_lane_columns.digest
    );
    assert_eq!(
        packaged.main_lane.binding.root_lane_commitment_digest,
        output.root_lane_commitment.digest
    );
    assert_eq!(
        packaged.main_lane.binding.public_step_count as usize,
        output.root_lane_columns.time_len as usize
    );
    assert_eq!(
        output.root_lane_columns.time_len as usize,
        output.trace.execution_rows.len()
    );
    assert!(output
        .trace
        .execution_rows
        .iter()
        .any(|row| row.trace_virtual_opcode.is_some()));
}

#[test]
fn simple_kernel_rejects_tampered_kernel_and_packaged_boundaries() {
    let public = public_input("unsigned_divrem_chain_ecall");
    let prover = SimpleKernelProverInput { public: public.clone() };
    let verifier = SimpleKernelVerifierInput { public };

    let (_output, proof) = prove_simple_kernel(&prover).expect("prove simple kernel");
    let mut tampered_kernel = proof.clone();
    tampered_kernel.trace.execution_rows[0].is_commit_row = !tampered_kernel.trace.execution_rows[0].is_commit_row;
    assert!(verify_simple_kernel(&verifier, &tampered_kernel).is_err());

    let mut tampered_stage_claims = proof.clone();
    tampered_stage_claims.stage_claims.stage1.claim.row_count += 1;
    assert!(verify_simple_kernel(&verifier, &tampered_stage_claims).is_err());

    let mut tampered_stage2_families = proof.clone();
    tampered_stage2_families
        .stage_claims
        .stage2
        .families
        .register_reads_digest[0] ^= 1;
    assert!(verify_simple_kernel(&verifier, &tampered_stage2_families).is_err());

    let mut tampered_stage1_rows = proof.clone();
    tampered_stage1_rows.stage_claims.stage1.rows.rows_digest[0] ^= 1;
    assert!(verify_simple_kernel(&verifier, &tampered_stage1_rows).is_err());

    let mut tampered_stage_package = proof.clone();
    tampered_stage_package
        .stage_packages
        .stage1
        .claim
        .first_digest_mut()[0] ^= 1;
    assert!(verify_simple_kernel(&verifier, &tampered_stage_package).is_err());

    let mut tampered_stage1_family = proof.clone();
    tampered_stage1_family
        .stage_packages
        .stage1
        .claim
        .rows_family_digest[0] ^= 1;
    assert!(verify_simple_kernel(&verifier, &tampered_stage1_family).is_err());

    let mut tampered_stage2_family = proof.clone();
    tampered_stage2_family
        .stage_packages
        .stage2
        .claim
        .register_reads_family_digest[0] ^= 1;
    assert!(verify_simple_kernel(&verifier, &tampered_stage2_family).is_err());

    let mut tampered_stage3_continuity = proof.clone();
    tampered_stage3_continuity
        .stage_claims
        .stage3
        .continuity
        .continuity_digest[0] ^= 1;
    assert!(verify_simple_kernel(&verifier, &tampered_stage3_continuity).is_err());

    let mut tampered_stage3_family = proof.clone();
    tampered_stage3_family
        .stage_packages
        .stage3
        .claim
        .continuity_family_digest[0] ^= 1;
    assert!(verify_simple_kernel(&verifier, &tampered_stage3_family).is_err());

    let mut tampered_kernel_opening = proof.clone();
    tampered_kernel_opening
        .kernel_opening
        .claim
        .first_digest_mut()[0] ^= 1;
    assert!(verify_simple_kernel(&verifier, &tampered_kernel_opening).is_err());

    let mut tampered_kernel_prepared_opening = proof.clone();
    tampered_kernel_prepared_opening
        .kernel_opening
        .claim
        .prepared_steps
        .points
        .first_prepared_step
        .as_mut()
        .expect("kernel prepared-step opening ref")
        .id
        .object
        .commitment_digest[0] ^= 1;
    assert!(verify_simple_kernel(&verifier, &tampered_kernel_prepared_opening).is_err());

    let mut tampered_kernel_opening_package = proof.clone();
    tampered_kernel_opening_package
        .kernel_opening
        .bindings
        .packaged
        .proof
        .proof_digest[0] ^= 1;
    assert!(verify_simple_kernel(&verifier, &tampered_kernel_opening_package).is_err());

    let mut tampered_root_lane_columns = proof.clone();
    tampered_root_lane_columns.root_lane_columns.family_digest[0] ^= 1;
    assert!(verify_simple_kernel(&verifier, &tampered_root_lane_columns).is_err());

    let mut tampered_root_lane_commitment = proof.clone();
    tampered_root_lane_commitment
        .root_lane_commitment
        .commitments
        .digest[0] ^= 1;
    assert!(verify_simple_kernel(&verifier, &tampered_root_lane_commitment).is_err());

    let (_output, packaged) = prove_packaged_simple_kernel(&prover).expect("prove packaged simple kernel");
    let mut tampered_main_lane_binding = packaged.clone();
    tampered_main_lane_binding
        .main_lane
        .binding
        .root_lane_commitment_digest[0] ^= 1;
    assert!(verify_packaged_simple_kernel(&verifier, &tampered_main_lane_binding).is_err());

    let mut tampered_main_lane_digest = packaged.clone();
    tampered_main_lane_digest.main_lane.digest[0] ^= 1;
    assert!(verify_packaged_simple_kernel(&verifier, &tampered_main_lane_digest).is_err());
}
