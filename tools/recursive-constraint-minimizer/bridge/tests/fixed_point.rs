use neo_ccs::Mat;
use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::direct_ccs::R1cs;
use neo_fold_clean::frontends::f_prime::recursive_plan::{RecursiveStepImagePlan, StateXOutPlanOptions};
use neo_fold_clean::frontends::r1cs_f_prime::ivc::{R1csIvcBranch, R1csIvcRelation};
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_fold_clean::paper::params::Params;
use neo_math::F;
use neo_params::{goldilocks_paper_b2, NeoParams};
use nightstream_constraint_exporter::{
    analyze_fixed_point_branch, export_complete_fixed_point_problem, export_fixed_point_problem,
    fixed_point_family_census, render_bound_artifact_lean, render_complete_bound_artifact_lean, ExportRequest,
    FixedPointFamilySearch,
};
use p3_field::PrimeCharacteristicRing;
use recursive_constraint_minimizer::{Scope, SolverConfig};

fn one_product_r1cs() -> R1cs {
    let mut a = Mat::zero(1, neo_math::D, F::ZERO);
    let mut b = Mat::zero(1, neo_math::D, F::ZERO);
    let mut c = Mat::zero(1, neo_math::D, F::ZERO);
    a[(0, 1)] = F::ONE;
    b[(0, 2)] = F::ONE;
    c[(0, 0)] = F::ONE;
    R1cs { a, b, c, m_in: 1 }
}

fn lifecycle_plan(m: usize, m_in: usize) -> RecursiveStepImagePlan {
    RecursiveStepImagePlan {
        limbs: m * POSEIDON2_GOLDILOCKS_BITS + 1,
        app_private_var_widths: Vec::new(),
        boundary_bits: 4 * POSEIDON2_GOLDILOCKS_BITS,
        kmul_count: 0,
        ring_action_pair_count: 0,
        projection_batches: Vec::new(),
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
        ),
        sponge_transcript_permutes: 0,
        nifs_payload_shapes: Vec::new(),
        accumulator: None,
        state_x_out: Some(StateXOutPlanOptions {
            pc: 1,
            public_x_out_lane_bit_starts: std::array::from_fn(|lane| lane * POSEIDON2_GOLDILOCKS_BITS),
            app_public_input_var_indices: (0..m_in).collect(),
            app_public_input_bit_var_indices: Vec::new(),
            semantic_state_in_var_indices: Vec::new(),
            semantic_state_out_var_indices: Vec::new(),
            initial_semantic_state_digest_anchor: None,
        }),
    }
}

fn tiny_params() -> Params {
    let inner = NeoParams::new(
        goldilocks_paper_b2::Q,
        goldilocks_paper_b2::ETA as u32,
        goldilocks_paper_b2::D as u32,
        4,
        1u64 << 24,
        goldilocks_paper_b2::B_BASE,
        goldilocks_paper_b2::K_RHO,
        goldilocks_paper_b2::T,
        goldilocks_paper_b2::EXTENSION_DEGREE,
        60,
    )
    .expect("test parameters satisfy the reduction guard");
    Params::test_only_from_neo_params(inner)
}

#[test]
fn fixed_point_arms_have_complete_exportable_family_censuses() {
    let params = tiny_params();
    let app = one_product_r1cs();
    let plan = lifecycle_plan(app.m(), app.m_in);
    let audit = R1csIvcRelation::audit_fixed_point_constraint_sources(&params, &app.into(), &plan)
        .expect("discover fixed-point source arms");

    for branch in [
        R1csIvcBranch::Base,
        R1csIvcBranch::BootstrapRecursive,
        R1csIvcBranch::Recursive,
    ] {
        let arm = audit.arm(branch);
        let census = fixed_point_family_census(&audit, branch).expect("complete reviewed family ownership");
        assert!(!census.is_empty());
        assert_eq!(
            census
                .iter()
                .map(|family| family.source_rows().len())
                .sum::<usize>(),
            arm.n
        );
        assert!(census
            .windows(2)
            .all(|pair| pair[0].name() < pair[1].name()));

        let rewritten_source_rows = audit.fixed_point().rows().arms()[branch_index(branch)]
            .source_runs()
            .iter()
            .filter(|run| run.disposition().rewrite_id().is_some())
            .flat_map(|run| run.source_rows())
            .collect::<std::collections::BTreeSet<_>>();
        let family = census
            .iter()
            .filter(|family| {
                family
                    .source_rows()
                    .iter()
                    .any(|row| rewritten_source_rows.contains(row))
            })
            .min_by_key(|family| family.source_rows().len())
            .expect("one rewrite-owned family");
        let export = export_fixed_point_problem(
            &audit,
            branch,
            ExportRequest {
                profile: format!("fixed-point-test-{branch:?}"),
                scope: Scope::Branch,
                public_input_count: arm.m_in,
                source_rows: family.source_rows().to_vec(),
                complete_families: vec![family.name().to_owned()],
            },
        )
        .expect("export one complete fixed-point family");
        let problem = export.problem();
        assert_eq!(problem.rows.len(), family.source_rows().len());
        assert!(problem.rows.iter().all(|row| row.family == family.name()));
        let binding = export.binding();
        assert_eq!(binding.branch(), branch_name(branch));
        assert_eq!(binding.requested_source_rows(), family.source_rows());
        assert!(binding
            .requested_source_rows()
            .iter()
            .all(|row| binding.closure_source_rows().contains(row)));
        assert!(!binding.rewrites().is_empty());
        assert!(binding.rewrites().iter().all(|rewrite| rewrite
            .source_rows()
            .iter()
            .flat_map(|range| range.clone())
            .all(|row| binding.closure_source_rows().contains(&row))));
        assert_eq!(
            binding
                .projected_rows()
                .iter()
                .map(|row| row.emitted_row())
                .collect::<Vec<_>>(),
            binding.emitted_rows()
        );
        assert_eq!(binding.final_rows(), audit.fixed_point().rows().total_rows());
        assert_eq!(binding.final_columns(), audit.fixed_point().layout().total_columns());
        assert_eq!(
            binding.final_public_input_count(),
            audit.fixed_point().layout().public_input_len()
        );
        assert!(binding.final_plan_digest().starts_with("sha256:"));
        assert!(binding.projected_slice_digest().starts_with("sha256:"));
        let lean = render_bound_artifact_lean(&export, &format!("Generated.FixedPoint{branch:?}"))
            .expect("render exact bound Lean data");
        assert!(lean.contains("def boundArtifact : BoundArtifact"));
        assert!(lean.contains("theorem boundArtifact_coherent"));
        assert!(lean.contains("scope := \"branch\""));
        assert!(lean.contains("fieldModulus := \"18446744069414584321\""));
    }
}

#[test]
fn branch_driver_records_every_family_when_analysis_is_inconclusive() {
    let params = tiny_params();
    let app = one_product_r1cs();
    let plan = lifecycle_plan(app.m(), app.m_in);
    let audit = R1csIvcRelation::audit_fixed_point_constraint_sources(&params, &app.into(), &plan)
        .expect("discover fixed-point source arms");
    let branch = R1csIvcBranch::Base;
    let census = fixed_point_family_census(&audit, branch).expect("complete reviewed family ownership");

    // The wrong assignment length fails before cvc5 starts. The branch driver
    // must still produce one fail-closed result for every reviewed family.
    let report = analyze_fixed_point_branch(
        &audit,
        branch,
        &[],
        "fixed-point-exhaustive-inconclusive-test",
        &SolverConfig::default(),
        1,
    )
    .expect("build exact exhaustive branch ledger");

    assert_eq!(report.profile(), "fixed-point-exhaustive-inconclusive-test");
    assert_eq!(report.branch(), branch);
    assert!(report.source_artifact_digest().starts_with("sha256:"));
    assert!(report.final_plan_digest().starts_with("sha256:"));
    assert_eq!(report.source_rows(), audit.arm(branch).n);
    assert_eq!(report.source_columns(), audit.arm(branch).m);
    assert_eq!(report.source_public_columns(), audit.arm(branch).m_in);
    assert_eq!(report.final_rows(), audit.fixed_point().rows().total_rows());
    assert_eq!(report.final_columns(), audit.fixed_point().layout().total_columns());
    assert_eq!(
        report.final_public_columns(),
        audit.fixed_point().layout().public_input_len()
    );
    assert_eq!(report.families().len(), census.len());
    for (record, expected) in report.families().iter().zip(&census) {
        assert_eq!(record.name(), expected.name());
        assert_eq!(record.source_rows(), expected.source_rows());
        assert_eq!(record.search().family(), expected.name());
        assert!(matches!(
            record.search(),
            FixedPointFamilySearch::Inconclusive { reason, .. }
                if reason.contains("background assignment")
        ));
    }
}

#[test]
fn complete_branch_export_contains_every_owned_source_row() {
    let params = tiny_params();
    let app = one_product_r1cs();
    let plan = lifecycle_plan(app.m(), app.m_in);
    let audit = R1csIvcRelation::audit_fixed_point_constraint_sources(&params, &app.into(), &plan)
        .expect("discover fixed-point source arms");
    let branch = R1csIvcBranch::Base;
    let census = fixed_point_family_census(&audit, branch).expect("complete reviewed family ownership");
    let export = export_complete_fixed_point_problem(&audit, branch, "complete-fixed-point-test")
        .expect("export the complete branch relation");

    assert_eq!(export.problem().rows.len(), audit.arm(branch).n);
    assert_eq!(
        export
            .problem()
            .rows
            .iter()
            .map(|row| row.source_index)
            .collect::<Vec<_>>(),
        (0..audit.arm(branch).n).collect::<Vec<_>>()
    );
    assert_eq!(
        export.problem().complete_families,
        census
            .iter()
            .map(|family| family.name().to_owned())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        export.binding().requested_source_rows(),
        (0..audit.arm(branch).n).collect::<Vec<_>>()
    );
    assert_eq!(
        export.binding().closure_source_rows(),
        export.binding().requested_source_rows()
    );
    assert!(export.binding().additional_source_rows().is_empty());
    let lean = render_complete_bound_artifact_lean(&export, "Generated.CompleteFixedPoint")
        .expect("render complete relation coverage proof");
    assert!(lean.contains("theorem sourceArtifact_row_count"));
    assert!(lean.contains("theorem boundArtifact_coversFullRelation"));
}

fn branch_index(branch: R1csIvcBranch) -> usize {
    match branch {
        R1csIvcBranch::Base => 0,
        R1csIvcBranch::BootstrapRecursive => 1,
        R1csIvcBranch::Recursive => 2,
    }
}

fn branch_name(branch: R1csIvcBranch) -> &'static str {
    match branch {
        R1csIvcBranch::Base => "base",
        R1csIvcBranch::BootstrapRecursive => "bootstrap_recursive",
        R1csIvcBranch::Recursive => "recursive",
    }
}
