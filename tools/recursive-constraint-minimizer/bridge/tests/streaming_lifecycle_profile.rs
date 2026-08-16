//! Frozen identities for the exact two-arm lifecycle selective profile.

use neo_fold_clean::frontends::nebula::f_prime::{
    prepare_streaming_lifecycle_preprocessing, synthesize_streaming_lifecycle_source_arms,
    NebulaFPrimeStreamingLifecycleArm,
};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::r1cs_f_prime::build_multi_branch_selective_low_norm_r1cs_with_alignment;
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
    .expect("shape-specific Appendix B.2 parameters");
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

    eprintln!(
        "streaming lifecycle identities: base={} recursive={} plan={} final={}",
        export.source_artifact_digest(NebulaFPrimeStreamingLifecycleArm::Base),
        export.source_artifact_digest(NebulaFPrimeStreamingLifecycleArm::Recursive),
        export.final_plan_digest(),
        export.final_relation_digest(),
    );

    assert_eq!(export.profile().final_rows(), 1_346_348);
    assert_eq!(export.profile().final_columns(), 8_755_452);
    assert_eq!(
        export.source_artifact_digest(NebulaFPrimeStreamingLifecycleArm::Base),
        "sha256:1cf6e47c3b6cf308056af0af2a3cba18b5513c0df26121d5453933f8e22ccb6c"
    );
    assert_eq!(
        export.source_artifact_digest(NebulaFPrimeStreamingLifecycleArm::Recursive),
        "sha256:e2ab53022f3b7598898ffc24c0daa4213df1db0a9f19225d9c0af1f026b2b83e"
    );
    assert_eq!(
        export.final_plan_digest(),
        "sha256:d9e9a8835c6788be21554bce015af84526da625737d6e01efca4ed51a20fd23f"
    );
    assert_eq!(
        export.final_relation_digest(),
        "sha256:7cd1be6de8b25c32c23971d7136108c90310bd8633c992e89a6f8f280940ebd0"
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
