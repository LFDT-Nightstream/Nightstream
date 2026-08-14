#![cfg(all(feature = "metal", target_vendor = "apple", neo_metal_shaders))]

//! Metal adapter boundary tests.
//!
//! The adapter owns fresh-instance commitments and the one-joint table
//! evaluator. The canonical prover owns PiRLC, PiDEC, transcript order, round
//! checks, terminal checks, and proof bytes.

use std::sync::Arc;

use neo_ccs::{CcsMatrix, CscMat, GeometricRowRun, Mat, SeededPhi81LinearBlock};
use neo_fold_clean::engine::r1cs_circuit::boolean::enforce_bit;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::frontends::r1cs_f_prime::{
    build_multi_branch_selective_low_norm_r1cs_with_alignment, lower_field_r1cs,
};
use neo_fold_clean::paper::construction2::LaneCommitmentMode;
use neo_fold_clean::paper::nifs::{
    self, NifsFreshInstancesRequest, NifsFreshSignedUnitAssignment, NifsFreshSignedUnitInstancesRequest,
    NifsProverAdapter,
};
use neo_fold_clean::paper::reductions::accumulator_sis_circuit::{
    enforce_commit_fields, ACCUMULATOR_CE_CLAIM_SIS_CONFIG,
};
use neo_fold_clean::paper::relations::{CcsInstance, LaneRanges, LaneScheme};
use neo_fold_clean::{FinalWitnessOpeningBackend, RunningInstance};
use neo_math::{D, F, K};
use neo_prover_metal::MetalNifsProver;
use p3_field::PrimeCharacteristicRing;

fn relation(columns: usize) -> R1cs {
    let mut a = Mat::zero(1, columns, F::ZERO);
    a.set(0, 1, F::ONE);
    a.set(0, 2, F::ONE);
    let mut b = Mat::zero(1, columns, F::ZERO);
    b.set(0, 0, F::ONE);
    let mut c = Mat::zero(1, columns, F::ZERO);
    c.set(0, 3, F::ONE);
    R1cs { a, b, c, m_in: D }
}

fn assignment(columns: usize, lhs: u64, rhs: u64) -> Vec<F> {
    let mut values = vec![F::ZERO; columns];
    values[0] = F::ONE;
    values[1] = F::from_u64(lhs);
    values[2] = F::from_u64(rhs);
    values[3] = F::from_u64(lhs + rhs);
    values
}

fn radix_four_params(structure: &neo_ccs::CcsStructure<F>) -> neo_fold_clean::Params {
    let base = neo_fold_clean::config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("radix-four base parameters");
    neo_fold_clean::Params::test_only_from_neo_params(
        neo_params::NeoParams::new(
            base.q(),
            base.eta(),
            base.d(),
            base.kappa(),
            base.m(),
            4,
            7,
            base.T(),
            base.extension_degree(),
            114,
        )
        .expect("radix-four parameters"),
    )
}

fn with_extra_sparse_term(matrix: &CcsMatrix<F>, row: usize, column: usize, coefficient: F) -> CcsMatrix<F> {
    let csc = matrix
        .sparse_component()
        .expect("test matrix must have a sparse component");
    let mut terms = Vec::with_capacity(csc.vals.len() + 1);
    for source_column in 0..csc.ncols {
        for entry in csc.column_range(source_column) {
            terms.push((csc.row_index(entry), source_column, csc.vals[entry]));
        }
    }
    terms.push((row, column, coefficient));
    CcsMatrix::csc_with_compact_rows(
        CscMat::from_triplets(terms, csc.nrows, csc.ncols),
        matrix.seeded_phi81_blocks().to_vec(),
        matrix.geometric_runs().to_vec(),
    )
    .expect("rebuild test matrix")
}

#[test]
fn metal_joint_plan_keeps_its_structure_cache_alive() {
    let r1cs = relation(2 * D);
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4d45_5441_4c41).expect("preprocess");
    let superneo = prep.optimized_cache().superneo_arc();
    let weak = Arc::downgrade(&superneo);
    let mut metal = MetalNifsProver::new().expect("Metal adapter");

    metal
        .prepare_static(&prep.log, prep.structure(), prep.optimized_cache(), None)
        .expect("prepare static Metal plan");
    drop(superneo);
    drop(prep);

    assert!(
        weak.upgrade().is_some(),
        "the Metal plan must own the cache used to identify its matrix buffers"
    );
    drop(metal);
    assert!(
        weak.upgrade().is_none(),
        "dropping the Metal plan must release its cache owner"
    );
}

#[test]
fn metal_one_joint_oracle_matches_the_canonical_host_without_running_claims() {
    let r1cs = relation(2 * D);
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4d45_5441_4c32).expect("preprocess");
    let fresh = direct_ccs::build_instance(&prep, &r1cs, &assignment(2 * D, 1, 0)).expect("fresh instance");
    let fresh_claims = vec![fresh.claim.clone()];
    let running = RunningInstance::default();

    let mut cpu_transcript = Transcript::session();
    let cpu = nifs::prove(
        &mut cpu_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh.clone()],
        &running,
    )
    .expect("canonical host proof");

    let mut metal = MetalNifsProver::new().expect("Metal adapter");
    let mut metal_transcript = Transcript::session();
    let delegated = nifs::prove_with_adapter(
        &mut metal,
        &mut metal_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh.clone()],
        &running,
    )
    .expect("delegated host proof");

    assert_eq!(delegated.0.claims, cpu.0.claims);
    assert_eq!(delegated.0.witnesses, cpu.0.witnesses);
    assert_eq!(delegated.0.parent_authority, cpu.0.parent_authority);
    assert_eq!(delegated.1.pi_ccs.outputs, cpu.1.pi_ccs.outputs);
    assert_eq!(delegated.1.pi_ccs.outputs_digest, cpu.1.pi_ccs.outputs_digest);
    assert_eq!(delegated.1.pi_rlc.combined, cpu.1.pi_rlc.combined);
    assert_eq!(delegated.1.pi_dec.children, cpu.1.pi_dec.children);
    assert_eq!(
        serde_json::to_vec(&delegated.1.pi_ccs.sumcheck).expect("delegated SumCheck bytes"),
        serde_json::to_vec(&cpu.1.pi_ccs.sumcheck).expect("host SumCheck bytes"),
    );

    let mut verifier_transcript = Transcript::session();
    let verified = nifs::verify(
        &mut verifier_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &running,
        &delegated.1,
    )
    .expect("verify delegated proof");
    assert_eq!(verified.claims, delegated.0.claims);
}

#[test]
fn metal_one_joint_oracle_matches_the_canonical_host_with_compact_geometric_rows() {
    let r1cs = relation(2 * D);
    let baseline = direct_ccs::preprocess_seeded(&r1cs, 0x4745_4f4d_4554).expect("baseline preprocessing");
    let mut structure = baseline.structure().clone();
    structure.matrices[0] = CcsMatrix::csc_with_compact_rows(
        CscMat::from_triplets(Vec::new(), structure.n, structure.m),
        Vec::new(),
        vec![GeometricRowRun::new(0, 1, 2, F::ONE, F::ONE)],
    )
    .expect("compact geometric A matrix");
    let params = neo_fold_clean::config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("geometric parameters");
    let log = direct_ccs::ajtai::setup_seeded(&params, &structure, 0x4745_4f4d_4554);
    let prep = neo_fold_clean::lifecycle::preprocess_with_test_log(params, structure, log, Some(r1cs.m_in))
        .expect("geometric preprocessing");
    assert_eq!(
        prep.optimized_cache()
            .superneo()
            .matrix(0)
            .expect("geometric matrix cache")
            .compact_geometric_run_count(),
        1,
    );
    let values = assignment(2 * D, 1, 0);
    let fresh = CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &values, r1cs.m_in)
        .expect("geometric fresh instance");
    let running = RunningInstance::default();

    let mut cpu_transcript = Transcript::session();
    let cpu = nifs::prove(
        &mut cpu_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh.clone()],
        &running,
    )
    .expect("canonical geometric proof");

    let mut metal = MetalNifsProver::new().expect("Metal adapter");
    let mut metal_transcript = Transcript::session();
    let accelerated = nifs::prove_with_adapter(
        &mut metal,
        &mut metal_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh],
        &running,
    )
    .expect("Metal geometric proof");

    assert_eq!(accelerated.0.claims, cpu.0.claims);
    assert_eq!(accelerated.1.pi_ccs.outputs, cpu.1.pi_ccs.outputs);
    assert_eq!(
        accelerated.1.pi_ccs.sumcheck.canonical_bytes(),
        cpu.1.pi_ccs.sumcheck.canonical_bytes(),
    );
}

#[test]
fn metal_radix_four_geometric_running_oracle_matches_the_host() {
    let columns = 11 * D;
    let r1cs = relation(columns);
    let mut structure = r1cs.to_structure();
    structure.matrices[0] = CcsMatrix::csc_with_compact_rows(
        CscMat::from_triplets(Vec::new(), structure.n, structure.m),
        Vec::new(),
        vec![GeometricRowRun::new(0, 1, 2, F::ONE, F::from_u64(2))],
    )
    .expect("radix-four geometric A matrix");
    let params = radix_four_params(&structure);
    let log = direct_ccs::ajtai::setup_seeded(&params, &structure, 0x5241_4434_4745);
    let prep = neo_fold_clean::lifecycle::preprocess_with_test_log(params, structure, log, Some(D))
        .expect("radix-four geometric preprocessing");
    let values = |lhs: u64, rhs: u64| {
        let mut values = vec![F::ZERO; columns];
        values[0] = F::ONE;
        values[1] = F::from_u64(lhs);
        values[2] = F::from_u64(rhs);
        values[3] = F::from_u64(lhs + 2 * rhs);
        values
    };

    let initial = CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &values(1, 1), D)
        .expect("initial radix-four geometric instance");
    let mut initial_transcript = Transcript::session();
    let (running, _) = nifs::prove(
        &mut initial_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![initial],
        &RunningInstance::default(),
    )
    .expect("initial radix-four geometric fold");
    assert!(running.witnesses.iter().any(|witness| {
        witness.to_dense_vec().iter().any(|value| {
            *value == F::from_u64(2)
                || *value == -F::from_u64(2)
                || *value == F::from_u64(3)
                || *value == -F::from_u64(3)
        })
    }));

    let fresh = CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &values(3, 0), D)
        .expect("recursive radix-four geometric instance");
    let mut cpu_transcript = Transcript::session();
    let cpu = nifs::prove(
        &mut cpu_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh.clone()],
        &running,
    )
    .expect("radix-four geometric host proof");
    let mut metal = MetalNifsProver::new().expect("Metal adapter");
    let mut metal_transcript = Transcript::session();
    let accelerated = nifs::prove_with_adapter(
        &mut metal,
        &mut metal_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh],
        &running,
    )
    .expect("radix-four geometric Metal proof");
    assert_eq!(
        accelerated.1.pi_ccs.sumcheck.canonical_bytes(),
        cpu.1.pi_ccs.sumcheck.canonical_bytes(),
    );
}

#[test]
fn metal_partial_carrier_identity_opening_matches_the_canonical_host() {
    let columns = 2 * D - 23;
    let r1cs = relation(columns);
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4d45_5441_4c42).expect("preprocess");
    let mut values = assignment(columns, 1, 0);
    for (index, value) in values.iter_mut().enumerate().skip(4) {
        *value = match index % 3 {
            0 => F::ONE,
            1 => F::ZERO - F::ONE,
            _ => F::ZERO,
        };
    }
    let fresh = direct_ccs::build_instance(&prep, &r1cs, &values).expect("fresh instance");
    let variables = prep
        .structure()
        .n
        .max(neo_reductions::common::superneo_carrier_width(prep.structure().m))
        .next_power_of_two()
        .trailing_zeros() as usize;
    let point = (0..variables)
        .map(|index| K::from(F::from_u64(index as u64 + 2)))
        .collect::<Vec<_>>();
    let expected = neo_reductions::common::compute_y_from_Z_and_r(
        prep.structure(),
        &fresh.witness.Z,
        &point,
        D.next_power_of_two().trailing_zeros() as usize,
        prep.params.b(),
    )
    .0;
    let mut metal = MetalNifsProver::new().expect("Metal adapter");
    let actual = metal
        .final_witness_openings(
            prep.optimized_cache(),
            std::slice::from_ref(&fresh.witness.Z),
            &point,
            prep.structure().m,
        )
        .expect("Metal openings")
        .expect("supported Metal openings");

    assert_eq!(&expected[0][..D], &actual[0][0], "partial-carrier identity opening");
}

#[test]
fn metal_one_joint_oracle_skips_canonical_zero_running_planes() {
    let r1cs = relation(2 * D);
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4d45_5441_4c40).expect("preprocess");
    let fresh = direct_ccs::build_instance(&prep, &r1cs, &assignment(2 * D, 1, 0)).expect("fresh instance");
    let running = RunningInstance::canonical_zero(&prep.params, prep.structure(), r1cs.m_in, LaneCommitmentMode::Plain)
        .expect("canonical zero running accumulator");

    let mut cpu_transcript = Transcript::session();
    let cpu = nifs::prove(
        &mut cpu_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh.clone()],
        &running,
    )
    .expect("canonical host proof");

    let mut metal = MetalNifsProver::new().expect("Metal adapter");
    let mut metal_transcript = Transcript::session();
    let accelerated = nifs::prove_with_adapter(
        &mut metal,
        &mut metal_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh],
        &running,
    )
    .expect("Metal proof with canonical zero running accumulator");

    assert_eq!(accelerated.0, cpu.0);
    assert_eq!(accelerated.1.pi_ccs.outputs, cpu.1.pi_ccs.outputs);
    assert_eq!(
        accelerated.1.pi_ccs.sumcheck.canonical_bytes(),
        cpu.1.pi_ccs.sumcheck.canonical_bytes(),
    );
}

#[test]
fn metal_selective_f_prime_oracle_matches_the_canonical_host() {
    let fixtures = (0..3)
        .map(|arm| {
            let mut builder = R1csBuilder::new();
            let bit = builder.alloc(F::from_u64(arm & 1));
            enforce_bit(&mut builder, bit);
            lower_field_r1cs(builder, &[bit])
                .expect("lower selective Boolean arm")
                .into_parts()
        })
        .collect::<Vec<_>>();
    let shapes = fixtures
        .iter()
        .map(|(shape, _)| shape.clone())
        .collect::<Vec<_>>();
    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 0, D, 0).expect("build selective relation");
    assert!(
        neo_fold_clean::frontends::r1cs_f_prime::is_canonical_selective_low_norm_polynomial(&relation.structure().f)
    );
    let assignment = relation
        .encode(1, &fixtures[1].1)
        .expect("encode selected arm");
    assert!(relation.is_satisfied(&assignment));

    let structure = relation.structure().clone();
    let params = neo_fold_clean::config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("selective parameters");
    let log = direct_ccs::ajtai::setup_seeded(&params, &structure, 0x5345_4c45_4354);
    let prep =
        neo_fold_clean::lifecycle::preprocess_with_test_log(params, structure, log, Some(relation.public_input_len()))
            .expect("selective preprocessing");
    let fresh = CcsInstance::from_low_norm_assignment(
        &prep.params,
        &prep.log,
        prep.structure(),
        &assignment,
        relation.public_input_len(),
    )
    .expect("selective fresh instance");
    let running = RunningInstance::default();

    let mut cpu_transcript = Transcript::session();
    let cpu = nifs::prove(
        &mut cpu_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh.clone()],
        &running,
    )
    .expect("canonical selective proof");

    let mut metal = MetalNifsProver::new().expect("Metal adapter");
    let variables = prep
        .structure()
        .n
        .max(neo_reductions::common::superneo_carrier_width(prep.structure().m))
        .next_power_of_two()
        .trailing_zeros() as usize;
    let point = (0..variables)
        .map(|index| K::from(F::from_u64(index as u64 + 2)))
        .collect::<Vec<_>>();
    let expected = neo_reductions::common::compute_y_from_Z_and_r(
        prep.structure(),
        &fresh.witness.Z,
        &point,
        D.next_power_of_two().trailing_zeros() as usize,
        prep.params.b(),
    )
    .0;
    let actual = metal
        .final_witness_openings(
            prep.optimized_cache(),
            std::slice::from_ref(&fresh.witness.Z),
            &point,
            prep.structure().m,
        )
        .expect("Metal selective openings")
        .expect("supported selective openings");
    for (matrix, (expected, actual)) in expected.iter().zip(&actual[0]).enumerate() {
        assert_eq!(&expected[..D], actual, "selective opening matrix {matrix}");
    }
    let mut metal_transcript = Transcript::session();
    let accelerated = nifs::prove_with_adapter(
        &mut metal,
        &mut metal_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh.clone()],
        &running,
    )
    .expect("Metal selective proof");

    assert_eq!(accelerated.0.claims, cpu.0.claims);
    assert_eq!(accelerated.1.pi_ccs.outputs, cpu.1.pi_ccs.outputs);
    assert_eq!(
        accelerated.1.pi_ccs.sumcheck.canonical_bytes(),
        cpu.1.pi_ccs.sumcheck.canonical_bytes(),
    );

    let mut recursive_cpu_transcript = Transcript::session();
    let recursive_cpu = nifs::prove(
        &mut recursive_cpu_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh.clone()],
        &cpu.0,
    )
    .expect("canonical recursive selective proof");
    let mut recursive_metal_transcript = Transcript::session();
    let recursive_metal = nifs::prove_with_adapter(
        &mut metal,
        &mut recursive_metal_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh],
        &cpu.0,
    )
    .expect("Metal recursive selective proof");
    assert_eq!(
        recursive_metal.1.pi_ccs.sumcheck.canonical_bytes(),
        recursive_cpu.1.pi_ccs.sumcheck.canonical_bytes(),
    );
}

#[test]
fn metal_selective_seeded_phi81_satisfied_rows_match_the_canonical_host() {
    let fixtures = (0..3)
        .map(|arm| {
            let mut builder = R1csBuilder::new();
            let bit = builder.alloc(F::from_u64(arm & 1));
            enforce_bit(&mut builder, bit);
            let field = builder.alloc(F::from_u64(arm as u64 + 3));
            enforce_commit_fields(
                &mut builder,
                ACCUMULATOR_CE_CLAIM_SIS_CONFIG,
                std::slice::from_ref(&field),
            )
            .expect("seeded Phi81 fixture");
            lower_field_r1cs(builder, &[bit])
                .expect("lower selective seeded arm")
                .into_parts()
        })
        .collect::<Vec<_>>();
    let shapes = fixtures
        .iter()
        .map(|(shape, _)| shape.clone())
        .collect::<Vec<_>>();
    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 0, D, 0).expect("build selective relation");
    assert!(
        neo_fold_clean::frontends::r1cs_f_prime::is_canonical_selective_low_norm_polynomial(&relation.structure().f)
    );
    assert!(relation.structure().matrices.iter().any(|matrix| matches!(
        matrix,
        CcsMatrix::CscWithSeededPhi81 { blocks, .. } if !blocks.is_empty()
    )));
    let assignment = relation
        .encode(1, &fixtures[1].1)
        .expect("encode selected seeded arm");
    assert!(relation.is_satisfied(&assignment));

    let structure = relation.structure().clone();
    let mut fallback_structure = structure.clone();
    let seeded_row = fallback_structure.matrices[2]
        .seeded_phi81_blocks()
        .first()
        .expect("seeded A block")
        .row_start();
    let zero_column = assignment
        .iter()
        .enumerate()
        .skip(1)
        .find_map(|(column, &value)| (value == F::ZERO).then_some(column))
        .expect("fixture must contain a zero assignment coordinate");
    fallback_structure.matrices[3] =
        with_extra_sparse_term(&fallback_structure.matrices[3], seeded_row, zero_column, F::ONE);
    assert!(
        neo_ccs::check_ccs_rowwise_zero(&fallback_structure, &assignment, &[]).is_ok(),
        "the fallback fixture must remain a valid selective assignment"
    );
    let params = neo_fold_clean::config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("selective parameters");
    let log = direct_ccs::ajtai::setup_seeded(&params, &structure, 0x5345_4544_4544);
    let prep =
        neo_fold_clean::lifecycle::preprocess_with_test_log(params, structure, log, Some(relation.public_input_len()))
            .expect("selective preprocessing");
    let fresh = CcsInstance::from_low_norm_assignment(
        &prep.params,
        &prep.log,
        prep.structure(),
        &assignment,
        relation.public_input_len(),
    )
    .expect("selective fresh instance");
    let running = RunningInstance::default();

    let mut cpu_transcript = Transcript::session();
    let cpu = nifs::prove(
        &mut cpu_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh.clone()],
        &running,
    )
    .expect("canonical selective seeded proof");
    let mut metal = MetalNifsProver::new().expect("Metal adapter");
    metal.session().reset_activity();
    let mut metal_transcript = Transcript::session();
    let accelerated = nifs::prove_with_adapter(
        &mut metal,
        &mut metal_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh],
        &running,
    )
    .expect("Metal selective seeded proof");
    let copy_activity = metal.session().activity();

    assert_eq!(accelerated.0.claims, cpu.0.claims);
    assert_eq!(accelerated.1.pi_ccs.outputs, cpu.1.pi_ccs.outputs);
    assert_eq!(
        accelerated.1.pi_ccs.sumcheck.canonical_bytes(),
        cpu.1.pi_ccs.sumcheck.canonical_bytes(),
    );

    let fallback_params = neo_fold_clean::config::ccs_params(
        fallback_structure.n,
        fallback_structure.m,
        fallback_structure.t(),
        fallback_structure.max_degree(),
    )
    .expect("fallback parameters");
    let fallback_log = direct_ccs::ajtai::setup_seeded(&fallback_params, &fallback_structure, 0x5345_4544_4642);
    let fallback_prep = neo_fold_clean::lifecycle::preprocess_with_test_log(
        fallback_params,
        fallback_structure,
        fallback_log,
        Some(relation.public_input_len()),
    )
    .expect("fallback preprocessing");
    let fallback_fresh = CcsInstance::from_low_norm_assignment(
        &fallback_prep.params,
        &fallback_prep.log,
        fallback_prep.structure(),
        &assignment,
        relation.public_input_len(),
    )
    .expect("fallback fresh instance");
    let mut fallback_cpu_transcript = Transcript::session();
    let fallback_cpu = nifs::prove(
        &mut fallback_cpu_transcript,
        &fallback_prep.params,
        fallback_prep.structure(),
        fallback_prep.optimized_cache(),
        &fallback_prep.log,
        None,
        fallback_prep.mix_rhos_commits(),
        fallback_prep.combine_b_pows(),
        vec![fallback_fresh.clone()],
        &running,
    )
    .expect("canonical fallback proof");
    let mut fallback_metal = MetalNifsProver::new().expect("fallback Metal adapter");
    fallback_metal.session().reset_activity();
    let mut fallback_metal_transcript = Transcript::session();
    let fallback_accelerated = nifs::prove_with_adapter(
        &mut fallback_metal,
        &mut fallback_metal_transcript,
        &fallback_prep.params,
        fallback_prep.structure(),
        fallback_prep.optimized_cache(),
        &fallback_prep.log,
        None,
        fallback_prep.mix_rhos_commits(),
        fallback_prep.combine_b_pows(),
        vec![fallback_fresh],
        &running,
    )
    .expect("Metal fallback proof");
    let fallback_activity = fallback_metal.session().activity();
    assert_eq!(
        fallback_accelerated.1.pi_ccs.sumcheck.canonical_bytes(),
        fallback_cpu.1.pi_ccs.sumcheck.canonical_bytes(),
    );
    assert!(
        fallback_activity.dispatches > copy_activity.dispatches,
        "a noncanonical seeded row must use the complete Metal evaluation path"
    );
}

#[test]
fn metal_one_joint_oracle_matches_the_canonical_host_with_running_claims() {
    let r1cs = relation(11 * D);
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4d45_5441_4c39).expect("preprocess");
    let initial_fresh =
        direct_ccs::build_instance(&prep, &r1cs, &assignment(11 * D, 1, 0)).expect("initial fresh instance");
    let mut initial_transcript = Transcript::session();
    let (running, _) = nifs::prove(
        &mut initial_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![initial_fresh],
        &RunningInstance::default(),
    )
    .expect("initial canonical fold");

    let fresh = direct_ccs::build_instance(&prep, &r1cs, &assignment(11 * D, 0, 1)).expect("second fresh instance");
    let fresh_claims = vec![fresh.claim.clone()];
    let mut cpu_transcript = Transcript::session();
    let cpu = nifs::prove(
        &mut cpu_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh.clone()],
        &running,
    )
    .expect("canonical host proof");

    let mut metal = MetalNifsProver::new().expect("Metal adapter");
    metal.session().reset_activity();
    let mut metal_transcript = Transcript::session();
    let accelerated = nifs::prove_with_adapter(
        &mut metal,
        &mut metal_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh.clone()],
        &running,
    )
    .expect("Metal proof");

    let activity = metal.session().activity();
    assert!(activity.dispatches > 20, "one-joint proof must execute on Metal");
    assert!(activity.host_waits > 1, "one-joint rounds must synchronize with Metal");
    assert_eq!(accelerated.0.claims, cpu.0.claims);
    assert_eq!(accelerated.0.witnesses, cpu.0.witnesses);
    assert_eq!(accelerated.1.pi_ccs.outputs, cpu.1.pi_ccs.outputs);
    assert_eq!(accelerated.1.pi_ccs.outputs_digest, cpu.1.pi_ccs.outputs_digest);
    assert_eq!(
        accelerated.1.pi_ccs.sumcheck.canonical_bytes(),
        cpu.1.pi_ccs.sumcheck.canonical_bytes(),
    );

    let mut verifier_transcript = Transcript::session();
    nifs::verify(
        &mut verifier_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &running,
        &accelerated.1,
    )
    .expect("verify Metal proof");
}

#[test]
fn metal_radix_four_one_joint_matches_the_host_with_running_digits() {
    let r1cs = relation(11 * D);
    let structure = r1cs.to_structure();
    let params = radix_four_params(&structure);
    let log = direct_ccs::ajtai::setup_seeded(&params, &structure, 0x5241_4449_5834);
    let prep = neo_fold_clean::lifecycle::preprocess_with_test_log(params, structure, log, Some(r1cs.m_in))
        .expect("radix-four preprocessing");

    let initial_fresh =
        direct_ccs::build_instance(&prep, &r1cs, &assignment(11 * D, 1, 2)).expect("initial radix-four instance");
    let mut initial_transcript = Transcript::session();
    let (running, _) = nifs::prove(
        &mut initial_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![initial_fresh],
        &RunningInstance::default(),
    )
    .expect("initial radix-four fold");
    assert!(running.witnesses.iter().any(|witness| {
        witness.to_dense_vec().iter().any(|value| {
            *value == F::from_u64(2)
                || *value == -F::from_u64(2)
                || *value == F::from_u64(3)
                || *value == -F::from_u64(3)
        })
    }));

    let fresh =
        direct_ccs::build_instance(&prep, &r1cs, &assignment(11 * D, 2, 1)).expect("second radix-four instance");
    let fresh_claims = vec![fresh.claim.clone()];
    let mut cpu_transcript = Transcript::session();
    let cpu = nifs::prove(
        &mut cpu_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh.clone()],
        &running,
    )
    .expect("canonical radix-four proof");

    let mut metal = MetalNifsProver::new().expect("Metal adapter");
    metal.session().reset_activity();
    let mut metal_transcript = Transcript::session();
    let accelerated = nifs::prove_with_adapter(
        &mut metal,
        &mut metal_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh],
        &running,
    )
    .expect("Metal radix-four proof");

    assert!(metal.session().activity().dispatches > 20);
    assert_eq!(accelerated.0.claims, cpu.0.claims);
    assert_eq!(accelerated.0.witnesses, cpu.0.witnesses);
    assert_eq!(accelerated.1.pi_ccs.outputs, cpu.1.pi_ccs.outputs);
    assert_eq!(
        accelerated.1.pi_ccs.sumcheck.canonical_bytes(),
        cpu.1.pi_ccs.sumcheck.canonical_bytes(),
    );

    let mut verifier_transcript = Transcript::session();
    nifs::verify(
        &mut verifier_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &running,
        &accelerated.1,
    )
    .expect("verify Metal radix-four proof");
}

#[test]
fn metal_seeded_one_joint_oracle_matches_the_canonical_host() {
    let columns = 2 * D;
    let mut a = Mat::zero(D, columns, F::ZERO);
    let mut b = Mat::zero(D, columns, F::ZERO);
    let mut c = Mat::zero(D, columns, F::ZERO);
    for row in 0..D {
        a.set(row, 1, F::ONE);
        a.set(row, 2, F::ONE);
        b.set(row, 0, F::ONE);
        c.set(row, 3, F::ONE);
    }
    let r1cs = R1cs { a, b, c, m_in: D };
    let assignment = assignment(columns, 1, 0);
    r1cs.is_satisfied_by(&assignment)
        .expect("satisfied seeded fixture");

    let mut structure = r1cs.to_structure();
    let seeded = SeededPhi81LinearBlock::new_with_word_width(0, vec![1], 1, 1, 1, 1, vec![vec![[0xa5; 32]]])
        .expect("seeded active-column block");
    let sparse = structure.matrices[0]
        .sparse_component()
        .expect("R1CS matrix has a sparse component")
        .clone();
    structure.matrices[0] =
        CcsMatrix::csc_with_seeded_phi81(sparse, vec![seeded.clone()]).expect("attach seeded block to A");
    let sparse = structure.matrices[2]
        .sparse_component()
        .expect("R1CS matrix has a sparse component")
        .clone();
    structure.matrices[2] = CcsMatrix::csc_with_seeded_phi81(sparse, vec![seeded]).expect("attach seeded block to C");
    let params = neo_fold_clean::config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("seeded parameters");
    let log = direct_ccs::ajtai::setup_seeded(&params, &structure, 0x4d45_5441_4c5f);
    let prep = neo_fold_clean::lifecycle::preprocess_with_test_log(params, structure, log, Some(r1cs.m_in))
        .expect("seeded preprocessing");
    let fresh =
        CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &assignment, r1cs.m_in)
            .expect("seeded fresh instance");
    let fresh_claims = vec![fresh.claim.clone()];
    let running = RunningInstance::default();

    let mut cpu_transcript = Transcript::session();
    let cpu = nifs::prove(
        &mut cpu_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh.clone()],
        &running,
    )
    .expect("canonical seeded proof");

    let mut metal = MetalNifsProver::new().expect("Metal adapter");
    let mut metal_transcript = Transcript::session();
    let accelerated = nifs::prove_with_adapter(
        &mut metal,
        &mut metal_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh.clone()],
        &running,
    )
    .expect("Metal seeded proof");

    assert_eq!(accelerated.0.claims, cpu.0.claims);
    assert_eq!(accelerated.0.witnesses, cpu.0.witnesses);
    assert_eq!(accelerated.1.pi_ccs.outputs, cpu.1.pi_ccs.outputs);
    assert_eq!(accelerated.1.pi_rlc.combined, cpu.1.pi_rlc.combined);
    assert_eq!(accelerated.1.pi_dec.children, cpu.1.pi_dec.children);
    assert_eq!(
        accelerated.1.pi_ccs.sumcheck.canonical_bytes(),
        cpu.1.pi_ccs.sumcheck.canonical_bytes(),
    );

    let mut verifier_transcript = Transcript::session();
    nifs::verify(
        &mut verifier_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &running,
        &accelerated.1,
    )
    .expect("verify Metal seeded proof");

    let mut recursive_cpu_transcript = Transcript::session();
    let recursive_cpu = nifs::prove(
        &mut recursive_cpu_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh.clone()],
        &cpu.0,
    )
    .expect("canonical recursive seeded proof");
    let mut recursive_metal_transcript = Transcript::session();
    let recursive_metal = nifs::prove_with_adapter(
        &mut metal,
        &mut recursive_metal_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh],
        &cpu.0,
    )
    .expect("Metal recursive seeded proof");
    assert_eq!(
        recursive_metal.1.pi_ccs.sumcheck.canonical_bytes(),
        recursive_cpu.1.pi_ccs.sumcheck.canonical_bytes(),
    );
}

#[test]
fn metal_fresh_commitment_matches_the_canonical_constructor() {
    let r1cs = relation(2 * D);
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4d45_5441_4c34).expect("preprocess");
    let values = assignment(2 * D, 1, 0);
    let canonical = direct_ccs::build_instance(&prep, &r1cs, &values).expect("canonical instance");
    let assignments = [values.as_slice()];
    let mut metal = MetalNifsProver::new().expect("Metal adapter");
    let built = metal
        .build_fresh_instances(NifsFreshInstancesRequest {
            pp: &prep.params,
            s: prep.structure(),
            cache: prep.optimized_cache(),
            log: &prep.log,
            m_in: r1cs.m_in,
            assignments: &assignments,
            lane_scheme: None,
        })
        .expect("Metal fresh-instance build")
        .expect("signed-unit assignment");

    assert_eq!(built.len(), 1);
    assert_eq!(built[0].claim.c, canonical.claim.c);
    assert_eq!(built[0].claim.x, canonical.claim.x);
    assert_eq!(built[0].claim.m_in, canonical.claim.m_in);
    assert_eq!(built[0].claim.adv, canonical.claim.adv);
    assert_eq!(built[0].witness.w, canonical.witness.w);
    assert_eq!(built[0].witness.Z, canonical.witness.Z);
    assert!(built[0].witness.Z.is_packed_signed_unit());
}

#[test]
fn metal_fresh_lane_commitments_match_the_canonical_constructor() {
    let r1cs = relation(3 * D);
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4d45_5441_4c37).expect("preprocess");
    let values = assignment(3 * D, 1, 0);
    let canonical = direct_ccs::build_instance(&prep, &r1cs, &values).expect("canonical instance");
    let lanes = LaneScheme::from_seeds(
        prep.params.kappa() as usize,
        LaneRanges {
            ops: 0..1,
            is: 1..2,
            fs: 2..3,
        },
        [0xA7; 32],
        [0x7A; 32],
    )
    .expect("lane scheme");
    let expected = lanes
        .commit(&canonical.witness.Z)
        .expect("canonical lane commitments");
    let assignments = [NifsFreshSignedUnitAssignment::from_dense(&values).expect("signed-unit assignment")];
    let mut metal = MetalNifsProver::new().expect("Metal adapter");
    let built = metal
        .build_fresh_signed_unit_instances(NifsFreshSignedUnitInstancesRequest {
            pp: &prep.params,
            s: prep.structure(),
            cache: prep.optimized_cache(),
            log: &prep.log,
            m_in: r1cs.m_in,
            assignments: &assignments,
            lane_scheme: Some(&lanes),
        })
        .expect("Metal fresh-instance build")
        .expect("signed-unit assignment");

    assert_eq!(built.len(), 1);
    assert_eq!(built[0].claim.c, canonical.claim.c);
    assert_eq!(built[0].claim.adv.as_ref(), Some(&expected));
    assert_eq!(built[0].witness.Z, canonical.witness.Z);
}

#[test]
#[ignore = "requires Apple Metal hardware; checks GPU fresh commitments and complete optimized NIFS parity"]
fn metal_selected_nifs_crosschecks_after_gpu_fresh_commitment() {
    let r1cs = relation(2 * D);
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4d45_5441_4c38).expect("preprocess");
    let values = assignment(2 * D, 1, 0);
    let assignments = [values.as_slice()];
    let mut prover = MetalNifsProver::new()
        .expect("Metal adapter")
        .crosschecked();

    prover.accelerator().session().reset_activity();
    let fresh = prover
        .build_fresh_instances(NifsFreshInstancesRequest {
            pp: &prep.params,
            s: prep.structure(),
            cache: prep.optimized_cache(),
            log: &prep.log,
            m_in: r1cs.m_in,
            assignments: &assignments,
            lane_scheme: None,
        })
        .expect("Metal fresh-instance build")
        .expect("signed-unit assignment");
    let activity = prover.accelerator().session().activity();
    assert!(activity.dispatches > 0, "fresh commitment must execute on Metal");

    let mut transcript = Transcript::session();
    nifs::prove_with_adapter(
        &mut prover,
        &mut transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &RunningInstance::default(),
    )
    .expect("Metal-selected NIFS matches optimized CPU");
}
