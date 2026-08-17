//! Exact two-arm selective profile for the streaming lifecycle relation.

use neo_fold_clean::frontends::nebula::f_prime::{
    prepare_streaming_lifecycle_preprocessing, production_streaming_lifecycle_full_source_fixed_point_audit,
    production_streaming_lifecycle_profile, synthesize_streaming_lifecycle_source_arms,
    NebulaFPrimeStreamingLifecycleArm, STREAMING_LIFECYCLE_BASE_SOURCE_ARTIFACT_ID,
    STREAMING_LIFECYCLE_FINAL_ARTIFACT_ID, STREAMING_LIFECYCLE_PROFILE_ID,
    STREAMING_LIFECYCLE_RECURSIVE_SOURCE_ARTIFACT_ID,
};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::r1cs_f_prime::build_multi_branch_selective_low_norm_r1cs_with_alignment;
use neo_fold_clean::frontends::r1cs_f_prime::SelectiveEmittedRowFamily;
use neo_fold_clean::paper::params::Params;
use neo_math::D;

#[test]
#[ignore = "expensive exact streaming lifecycle fixed-point audit"]
fn lifecycle_full_source_profile_reaches_a_recursive_shape_fixed_point() {
    let reference_params = Params::production();
    let memory = NebulaParams::new(0, 0, 1, 2, 1).expect("one-step memory profile");
    let plan = NebulaPlan::new(memory, vec![7], [0xD9; 32], reference_params.kappa() as usize).expect("Nebula plan");

    let audit = production_streaming_lifecycle_full_source_fixed_point_audit(&plan)
        .expect("Appendix B.2 streaming lifecycle full-source fixed point");
    for (index, round) in audit.rounds().iter().copied().enumerate() {
        eprintln!(
            "streaming lifecycle full-source fixed point round {index}: input=({},{},{},{}) lambda={} base=({},{},{}) recursive=({},{},{}) output=({},{},{},{})",
            round.input().rows(),
            round.input().columns(),
            round.input().matrix_count(),
            round.input().max_degree(),
            round.effective_lambda(),
            round.base_source().rows(),
            round.base_source().columns(),
            round.base_source().public_columns(),
            round.recursive_source().rows(),
            round.recursive_source().columns(),
            round.recursive_source().public_columns(),
            round.output().rows(),
            round.output().columns(),
            round.output().matrix_count(),
            round.output().max_degree(),
        );
    }
    let fixed = *audit.fixed_point();
    assert!(fixed.is_closed());
    assert_eq!(fixed.base_source().public_columns(), 641);
    assert_eq!(fixed.recursive_source().public_columns(), 641);
    assert_eq!(fixed.output().public_columns(), 648);
    let width = audit.width();
    eprintln!(
        "streaming lifecycle full-source fixed-point width: total={} branch_start={} base={} recursive={}",
        width.total_coordinates,
        width.branch_start,
        width.arms[0].total_branch_coordinates,
        width.arms[1].total_branch_coordinates,
    );
    let mut stages = width.arms[1].physical_stages.iter().collect::<Vec<_>>();
    stages.sort_unstable_by_key(|stage| std::cmp::Reverse(stage.allocated_coordinates));
    for stage in stages.into_iter().take(20) {
        eprintln!(
            "streaming lifecycle full-source recursive width stage: path={} source_columns={} direct={} aliases={} linear={} trace_eliminated={} eliminated={} allocated={}",
            stage.path,
            stage.source_column_count,
            stage.direct_columns,
            stage.decomposition_alias_columns + stage.equality_alias_columns,
            stage.linear_definition_columns,
            stage.trace_eliminated_columns,
            stage.eliminated_columns,
            stage.allocated_coordinates,
        );
    }
    assert!(
        audit.fits_joint_domain(),
        "streaming lifecycle full-source fixed point needs 2^{} but the phased target is 2^24",
        audit.joint_domain_bits()
    );
}

#[test]
#[ignore = "expensive exact lifecycle selective profile"]
fn lifecycle_profile_binds_every_source_stage_and_both_x_out_frames() {
    let reference_params = Params::production();
    let memory = NebulaParams::new(0, 0, 1, 2, 1).expect("one-step memory profile");
    let plan = NebulaPlan::new(memory, vec![7], [0xD9; 32], reference_params.kappa() as usize).expect("Nebula plan");
    let params = Params::for_ccs_shape(
        plan.circuit().structure().n,
        plan.circuit().structure().m,
        plan.circuit().structure().t(),
        plan.circuit().structure().max_degree(),
    )
    .expect("shape-specific Nightstream Goldilocks k_rho=16 parameters");
    assert!(params.has_production_core());
    let effective_lambda = params.lambda();
    let log = neo_fold_clean::frontends::direct_ccs::ajtai::setup_seeded(
        &params,
        plan.circuit().structure(),
        0x5354_5245_414d,
    );
    let preprocessing =
        neo_fold_clean::lifecycle::preprocess_with_test_log(params, plan.circuit().structure().clone(), log, Some(648))
            .expect("verifier-owned lifecycle preprocessing");
    let preprocessing =
        prepare_streaming_lifecycle_preprocessing(preprocessing, &plan).expect("fixed streaming lifecycle policy");
    let lifecycle = synthesize_streaming_lifecycle_source_arms(&preprocessing, &plan)
        .expect("exact streaming lifecycle source arms");
    let source_arms = [
        lifecycle
            .arm(NebulaFPrimeStreamingLifecycleArm::Base)
            .clone(),
        lifecycle
            .arm(NebulaFPrimeStreamingLifecycleArm::Recursive)
            .clone(),
    ];
    let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&source_arms, 0, D, 0)
        .expect("exact two-arm selective lifecycle relation");
    let width = relation
        .selective_compiler_audit()
        .expect("selective compiler audit")
        .width();
    let mut recursive_stages = width.arms[1].physical_stages.iter().collect::<Vec<_>>();
    recursive_stages.sort_unstable_by_key(|stage| std::cmp::Reverse(stage.allocated_coordinates));
    for stage in recursive_stages.into_iter().take(12) {
        eprintln!(
            "streaming lifecycle recursive width: path={} source={} direct={} aliases={} linear={} eliminated={} allocated={}",
            stage.path,
            stage.source_column_count,
            stage.direct_columns,
            stage.decomposition_alias_columns + stage.equality_alias_columns,
            stage.linear_definition_columns,
            stage.eliminated_columns + stage.trace_eliminated_columns,
            stage.allocated_coordinates,
        );
    }
    let profile = production_streaming_lifecycle_profile(&lifecycle, &relation)
        .expect("exact lifecycle source-to-selective profile");

    eprintln!(
        "streaming lifecycle profile: base=({},{},{}) recursive=({},{},{}) final=({},{},{}) lambda={}",
        profile
            .arm(NebulaFPrimeStreamingLifecycleArm::Base)
            .source_rows(),
        profile
            .arm(NebulaFPrimeStreamingLifecycleArm::Base)
            .source_columns(),
        profile
            .arm(NebulaFPrimeStreamingLifecycleArm::Base)
            .source_public_columns(),
        profile
            .arm(NebulaFPrimeStreamingLifecycleArm::Recursive)
            .source_rows(),
        profile
            .arm(NebulaFPrimeStreamingLifecycleArm::Recursive)
            .source_columns(),
        profile
            .arm(NebulaFPrimeStreamingLifecycleArm::Recursive)
            .source_public_columns(),
        profile.final_rows(),
        profile.final_columns(),
        profile.final_public_columns(),
        effective_lambda,
    );

    assert_eq!(profile.profile_id(), STREAMING_LIFECYCLE_PROFILE_ID);
    assert_eq!(profile.final_artifact_identity(), STREAMING_LIFECYCLE_FINAL_ARTIFACT_ID);
    assert_eq!(profile.final_rows(), relation.structure().n);
    assert_eq!(profile.final_columns(), relation.structure().m);
    assert_eq!(profile.final_public_columns(), relation.public_input_len());
    assert_eq!(profile.final_rows(), 10_306_243);
    assert_eq!(profile.final_columns(), 28_033_344);
    assert_eq!(profile.final_public_columns(), 648);
    assert!(profile.final_rows() <= 1 << 24);
    assert!(
        profile.final_columns() > 1 << 24,
        "the monolithic reference must not be mistaken for the final phased relation"
    );

    let columns = profile.column_layout();
    assert_eq!(columns.logical_public_columns().start, 0);
    assert_eq!(columns.public_columns(), 0..relation.public_input_len());
    assert_eq!(columns.ring_alignment_padding_columns().end, relation.structure().m);
    assert_eq!(columns.selector_columns().len(), 2);
    for required in [
        SelectiveEmittedRowFamily::SelectorDomain,
        SelectiveEmittedRowFamily::SharedDomain,
        SelectiveEmittedRowFamily::OneHot,
    ] {
        assert_eq!(
            profile
                .global_row_runs()
                .iter()
                .filter(|run| run.family() == required)
                .count(),
            1,
            "missing exact global lifecycle owner {required:?}",
        );
    }
    assert!(profile
        .global_row_runs()
        .iter()
        .find(|run| run.family() == SelectiveEmittedRowFamily::SharedDomain)
        .is_some_and(|run| run.rows().is_empty()));
    assert!(profile
        .global_row_runs()
        .iter()
        .all(|run| run.rewrite_id().is_none()));

    for (arm, expected_identity) in [
        (
            NebulaFPrimeStreamingLifecycleArm::Base,
            STREAMING_LIFECYCLE_BASE_SOURCE_ARTIFACT_ID,
        ),
        (
            NebulaFPrimeStreamingLifecycleArm::Recursive,
            STREAMING_LIFECYCLE_RECURSIVE_SOURCE_ARTIFACT_ID,
        ),
    ] {
        let source = lifecycle.arm(arm);
        let branch = profile.arm(arm);
        assert_eq!(branch.arm(), arm);
        assert_eq!(branch.source_artifact_identity(), expected_identity);
        assert_eq!(branch.source_rows(), source.n);
        assert_eq!(branch.source_columns(), source.m);
        assert_eq!(branch.source_public_columns(), source.m_in);
        assert_eq!(branch.stages().len(), source.physical_stage_ranges().len());
        assert!(!branch.compiler_row_runs().is_empty());
        assert!(branch
            .compiler_row_runs()
            .iter()
            .all(|run| run.rows().end <= profile.final_rows()));

        let mut source_row_cursor = 0usize;
        let mut source_column_cursor = source.m_in;
        for stage in branch.stages() {
            assert_eq!(stage.source_rows().start, source_row_cursor);
            assert_eq!(stage.source_columns().start, source_column_cursor);
            assert_eq!(stage.path(), source.physical_stage_ranges()[stage.occurrence()].path());
            let mut stage_run_cursor = stage.source_rows().start;
            for run in stage.source_runs() {
                assert_eq!(run.source_rows().start, stage_run_cursor);
                stage_run_cursor = run.source_rows().end;
            }
            assert_eq!(stage_run_cursor, stage.source_rows().end);
            for run in stage.final_row_runs() {
                assert!(run.rows().end <= profile.final_rows());
            }
            for rewrite in stage.rewrites() {
                assert!(rewrite.final_rows().end <= profile.final_rows());
                assert!(rewrite
                    .source_rows()
                    .iter()
                    .all(|rows| stage.source_rows().start <= rows.start && rows.end <= stage.source_rows().end));
            }
            source_row_cursor = stage.source_rows().end;
            source_column_cursor = stage.source_columns().end;
        }
        assert_eq!(source_row_cursor, source.n);
        assert_eq!(source_column_cursor, source.m);

        for x_out in [branch.before_x_out(), branch.after_x_out()] {
            assert_eq!(x_out.source_columns().len(), 32);
            assert_eq!(x_out.fields().len(), 32);
            for (source_column, field) in x_out.source_columns().iter().copied().zip(x_out.fields()) {
                assert_eq!(field.source_column(), source_column);
                assert!(!field.decoder_terms().is_empty());
                assert!(field
                    .decoder_terms()
                    .iter()
                    .all(|term| term.final_column() < profile.final_columns()));
            }
        }
    }

    let base = profile.arm(NebulaFPrimeStreamingLifecycleArm::Base);
    assert_eq!((base.source_rows(), base.source_columns()), (741_068, 740_549));
    let recursive = profile.arm(NebulaFPrimeStreamingLifecycleArm::Recursive);
    assert_eq!(
        (recursive.source_rows(), recursive.source_columns()),
        (31_339_296, 31_063_352)
    );
}
