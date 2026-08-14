use neo_ccs::Mat;
use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::direct_ccs::R1cs;
use neo_fold_clean::frontends::f_prime::recursive_plan::{RecursiveStepImagePlan, StateXOutPlanOptions};
use neo_fold_clean::frontends::r1cs_f_prime::ivc::{R1csIvcBranch, R1csIvcRelation};
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_fold_clean::paper::params::Params;
use neo_math::F;
use neo_params::{goldilocks_paper_b2, NeoParams};
use nightstream_constraint_exporter::{export_fixed_point_problem, fixed_point_family_census, ExportRequest};
use p3_field::PrimeCharacteristicRing;
use recursive_constraint_minimizer::Scope;

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

        let family = census
            .iter()
            .min_by_key(|family| family.source_rows().len())
            .expect("one owned family");
        let problem = export_fixed_point_problem(
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
        assert_eq!(problem.rows.len(), family.source_rows().len());
        assert!(problem.rows.iter().all(|row| row.family == family.name()));
    }
}
