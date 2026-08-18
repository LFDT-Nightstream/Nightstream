//! Frozen identities for the exact two-arm lifecycle selective profile.

use neo_fold_clean::frontends::nebula::f_prime::{
    prepare_streaming_lifecycle_preprocessing, synthesize_streaming_lifecycle_source_arms,
    NebulaFPrimeStreamingLifecycleArm,
};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::r1cs_f_prime::build_multi_branch_selective_low_norm_r1cs_with_alignment;
use neo_fold_clean::paper::f_prime::stage as fprime_stage;
use neo_fold_clean::paper::params::Params;
use neo_math::D;
use nightstream_constraint_exporter::export_complete_streaming_lifecycle_problem;

#[test]
#[ignore = "expensive exact lifecycle artifact hashing"]
fn exact_streaming_lifecycle_profile_has_frozen_artifact_identities() {
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
    let problem_export = export_complete_streaming_lifecycle_problem(
        &lifecycle,
        &relation,
        NebulaFPrimeStreamingLifecycleArm::Base,
        "nightstream/goldilocks/streaming-lifecycle-base/v1",
    )
    .expect("complete base lifecycle problem export");
    let export = problem_export.profile_export();
    let recursive_profile = export
        .profile()
        .arm(NebulaFPrimeStreamingLifecycleArm::Recursive);
    for arm in [
        NebulaFPrimeStreamingLifecycleArm::Base,
        NebulaFPrimeStreamingLifecycleArm::Recursive,
    ] {
        let arm_profile = export.profile().arm(arm);
        let stage_path = match arm {
            NebulaFPrimeStreamingLifecycleArm::Base => fprime_stage::BASE_SEMANTIC_LINKS,
            NebulaFPrimeStreamingLifecycleArm::Recursive => fprime_stage::RECURSIVE_SEMANTIC_LINKS,
        };
        let semantic_link = arm_profile
            .stages()
            .iter()
            .find(|stage| stage.path() == stage_path)
            .expect("lifecycle semantic-link physical stage");
        let retained_source_runs = semantic_link
            .source_runs()
            .iter()
            .filter(|run| run.emitted_start().is_some())
            .count();
        let first_final_rows = semantic_link.final_row_runs().first().map(|run| run.rows());
        let last_final_rows = semantic_link.final_row_runs().last().map(|run| run.rows());
        let nonempty_rewrites = semantic_link
            .rewrites()
            .iter()
            .filter(|rewrite| !rewrite.final_rows().is_empty())
            .count();
        let expected = match arm {
            NebulaFPrimeStreamingLifecycleArm::Base => (
                8,
                50_723..715_863,
                50_543..715_683,
                7_659,
                4_772_321..4_774_490,
                4_875_885..4_875_971,
                7_657,
            ),
            NebulaFPrimeStreamingLifecycleArm::Recursive => (
                11_883,
                30_676_324..31_339_295,
                30_400_381..31_063_352,
                5_490,
                5_008_421..5_010_590,
                5_190_239..5_190_325,
                5_488,
            ),
        };
        assert_eq!(semantic_link.occurrence(), expected.0);
        assert_eq!(semantic_link.source_rows(), expected.1);
        assert_eq!(semantic_link.source_columns(), expected.2);
        assert_eq!(semantic_link.source_runs().len(), expected.3);
        assert_eq!(retained_source_runs, 2);
        assert_eq!(semantic_link.final_row_runs().len(), 1_096);
        assert_eq!(first_final_rows, Some(expected.4));
        assert_eq!(last_final_rows, Some(expected.5));
        assert_eq!(semantic_link.rewrites().len(), expected.6);
        assert_eq!(nonempty_rewrites, 1_094);
        eprintln!(
            "streaming lifecycle semantic-link ownership: scope={} occurrence={} source_rows={:?} source_row_count={} source_columns={:?} source_runs={} retained_source_runs={} final_row_runs={} first_final_rows={:?} last_final_rows={:?} rewrites={} nonempty_rewrites={}",
            arm_profile.lifecycle_scope(),
            semantic_link.occurrence(),
            semantic_link.source_rows(),
            semantic_link.source_rows().len(),
            semantic_link.source_columns(),
            semantic_link.source_runs().len(),
            retained_source_runs,
            semantic_link.final_row_runs().len(),
            first_final_rows,
            last_final_rows,
            semantic_link.rewrites().len(),
            nonempty_rewrites,
        );
    }
    let recursive_x_out = lifecycle.x_out_preimage_columns(NebulaFPrimeStreamingLifecycleArm::Recursive);
    for (scope, binding, source_columns) in [
        ("before", recursive_profile.before_x_out(), recursive_x_out.before()),
        ("after", recursive_profile.after_x_out(), recursive_x_out.after()),
    ] {
        assert_eq!(binding.source_columns(), source_columns);
        assert_eq!(binding.fields().len(), source_columns.len());
        for (field, (&source_column, field_binding)) in source_columns.iter().zip(binding.fields()).enumerate() {
            assert_eq!(field_binding.source_column(), source_column);
            assert!(
                !field_binding.decoder_terms().is_empty(),
                "recursive {scope} XOut field {field} has no final decoder"
            );
            let stage = recursive_profile
                .stages()
                .iter()
                .find(|stage| stage.source_columns().contains(&source_column))
                .unwrap_or_else(|| panic!("recursive {scope} XOut field {field} has no physical source owner"));
            assert!(
                !stage.final_row_runs().is_empty(),
                "recursive {scope} XOut field {field} has no final row owner"
            );
        }
    }

    eprintln!(
        "streaming lifecycle identities: base={} recursive={} plan={} final={}",
        export.source_artifact_digest(NebulaFPrimeStreamingLifecycleArm::Base),
        export.source_artifact_digest(NebulaFPrimeStreamingLifecycleArm::Recursive),
        export.final_plan_digest(),
        export.final_relation_digest(),
    );
    eprintln!(
        "streaming lifecycle dimensions: final=({},{},{}) base=({},{},{}) recursive=({},{},{}) problem=({},{})",
        export.profile().final_rows(),
        export.profile().final_columns(),
        export.profile().final_public_columns(),
        export
            .profile()
            .arm(NebulaFPrimeStreamingLifecycleArm::Base)
            .source_rows(),
        export
            .profile()
            .arm(NebulaFPrimeStreamingLifecycleArm::Base)
            .source_columns(),
        export
            .profile()
            .arm(NebulaFPrimeStreamingLifecycleArm::Base)
            .source_public_columns(),
        export
            .profile()
            .arm(NebulaFPrimeStreamingLifecycleArm::Recursive)
            .source_rows(),
        export
            .profile()
            .arm(NebulaFPrimeStreamingLifecycleArm::Recursive)
            .source_columns(),
        export
            .profile()
            .arm(NebulaFPrimeStreamingLifecycleArm::Recursive)
            .source_public_columns(),
        problem_export.problem().rows.len(),
        problem_export.problem().column_count,
    );

    assert_eq!(export.profile().final_rows(), 10_306_243);
    assert_eq!(export.profile().final_columns(), 28_033_344);
    assert_eq!(
        export.source_artifact_digest(NebulaFPrimeStreamingLifecycleArm::Base),
        "sha256:c36cf17e8c4fba6dce24bf928d67f43106cb0af2bbada6a985860e79a8deb67d"
    );
    assert_eq!(
        export.source_artifact_digest(NebulaFPrimeStreamingLifecycleArm::Recursive),
        "sha256:3269aa2706f04e52b50b9bb92b09a9a36dfd7e68d5918ba7cec4df7aa6f4477f"
    );
    assert_eq!(
        export.final_plan_digest(),
        "sha256:7a7bf7804e6573d50836500bb59a4f773259ff4829ef70b574dbdc54ce6f0d52"
    );
    assert_eq!(
        export.final_relation_digest(),
        "sha256:734880081882f75c1c3e4b826201088c505fa9f468b802347b181d45e96282e5"
    );
    assert_eq!(problem_export.problem().rows.len(), 741_068);
    assert_eq!(problem_export.problem().source.total_rows, 741_068);
    assert_eq!(problem_export.problem().column_count, 740_549);
    assert_eq!(
        problem_export.problem().source.artifact_digest,
        export.source_artifact_digest(NebulaFPrimeStreamingLifecycleArm::Base)
    );
    assert_eq!(
        problem_export.arm_profile().source_rows(),
        problem_export.problem().rows.len()
    );
}
