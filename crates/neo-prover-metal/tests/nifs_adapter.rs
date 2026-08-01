#![cfg(all(feature = "metal", target_vendor = "apple", neo_metal_shaders))]

use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::boolean::enforce_bit;
use neo_fold_clean::engine::r1cs_circuit::field_ext::{enforce_k_dot_product, KVar};
use neo_fold_clean::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder};
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::frontends::r1cs_f_prime::{
    build_multi_branch_selective_low_norm_r1cs_with_alignment, lower_field_r1cs, SparseR1cs,
};
use neo_fold_clean::paper::nifs::{
    self, NifsFreshInstancesRequest, NifsFreshSignedUnitAssignment, NifsFreshSignedUnitInstancesRequest,
    NifsProverAdapter, NifsProverRequest,
};
use neo_fold_clean::paper::relations::{CcsInstance, LaneRanges, LaneScheme};
use neo_fold_clean::RunningInstance;
use neo_math::{D, F, K};
use neo_prover_metal::MetalNifsProver;
use p3_field::PrimeCharacteristicRing;

fn relation() -> R1cs {
    let mut a = Mat::zero(1, 2 * D, F::ZERO);
    a.set(0, 1, F::ONE);
    a.set(0, 2, F::ONE);
    let mut b = Mat::zero(1, 2 * D, F::ZERO);
    b.set(0, 0, F::ONE);
    let mut c = Mat::zero(1, 2 * D, F::ZERO);
    c.set(0, 3, F::ONE);
    R1cs { a, b, c, m_in: 3 }
}

fn assignment(lhs: u64, rhs: u64) -> Vec<F> {
    let mut values = vec![F::ZERO; 2 * D];
    values[0] = F::ONE;
    values[1] = F::from_u64(lhs);
    values[2] = F::from_u64(rhs);
    values[3] = F::from_u64(lhs + rhs);
    values
}

fn lane_relation() -> R1cs {
    let mut relation = relation();
    relation.a = Mat::zero(1, 3 * D, F::ZERO);
    relation.a.set(0, 1, F::ONE);
    relation.a.set(0, 2, F::ONE);
    relation.b = Mat::zero(1, 3 * D, F::ZERO);
    relation.b.set(0, 0, F::ONE);
    relation.c = Mat::zero(1, 3 * D, F::ZERO);
    relation.c.set(0, 3, F::ONE);
    relation
}

fn lane_assignment(lhs: u64, rhs: u64) -> Vec<F> {
    let mut values = assignment(lhs, rhs);
    values.resize(3 * D, F::ZERO);
    values
}

fn wide_relation(columns: usize) -> R1cs {
    let mut relation = relation();
    relation.a = Mat::zero(1, columns, F::ZERO);
    relation.a.set(0, 1, F::ONE);
    relation.a.set(0, 2, F::ONE);
    relation.b = Mat::zero(1, columns, F::ZERO);
    relation.b.set(0, 0, F::ONE);
    relation.c = Mat::zero(1, columns, F::ZERO);
    relation.c.set(0, 3, F::ONE);
    relation
}

fn wide_assignment(columns: usize, tail: usize) -> Vec<F> {
    let mut values = vec![F::ZERO; columns];
    values[0] = F::ONE;
    values[1] = F::ONE;
    values[3] = F::ONE;
    values[tail] = F::ONE;
    values
}

fn repeated_row_relation(rows: usize) -> R1cs {
    let columns = 2 * D;
    let mut a = Mat::zero(rows, columns, F::ZERO);
    let mut b = Mat::zero(rows, columns, F::ZERO);
    let mut c = Mat::zero(rows, columns, F::ZERO);
    for row in 0..rows {
        a.set(row, 1, F::ONE);
        b.set(row, 0, F::ONE);
        c.set(row, 1, F::ONE);
    }
    R1cs { a, b, c, m_in: 1 }
}

fn repeated_row_assignment() -> Vec<F> {
    let mut values = vec![F::ZERO; 2 * D];
    values[0] = F::ONE;
    values[1] = F::ONE;
    values
}

fn selective_factored_arm(seed: u64) -> (SparseR1cs, Vec<F>) {
    let mut builder = R1csBuilder::new();
    let public = builder.alloc(F::ONE);
    enforce_bit(&mut builder, public);
    let public_copy = builder.alloc(F::ONE);
    enforce_bit(&mut builder, public_copy);
    builder.enforce_eq(&Lc::from_var(public), &Lc::from_var(public_copy));
    let lhs = (0..6)
        .map(|index| {
            KVar::alloc(
                &mut builder,
                F::from_u64(seed + 2 * index + 1),
                F::from_u64(seed + 2 * index + 2),
            )
        })
        .collect::<Vec<_>>();
    let rhs = (0..6)
        .map(|index| {
            KVar::alloc(
                &mut builder,
                F::from_u64(seed + 3 * index + 3),
                F::from_u64(seed + 3 * index + 4),
            )
        })
        .collect::<Vec<_>>();
    let output = enforce_k_dot_product(&mut builder, &lhs, &rhs);
    let equal_copy = builder.alloc(builder.witness()[output.c1.col()]);
    builder.enforce_eq(&Lc::from_var(output.c1), &Lc::from_var(equal_copy));
    let canonical_copy = builder.alloc(builder.witness()[output.c0.col()]);
    builder.enforce_eq(&Lc::from_var(output.c0), &Lc::from_var(canonical_copy));
    let _output_bits = decompose_var_to_u64_bits(&mut builder, output.c0);
    let _copy_bits = decompose_var_to_u64_bits(&mut builder, canonical_copy);
    lower_field_r1cs(builder, &[public])
        .expect("lower selective factored arm")
        .into_parts()
}

#[test]
fn metal_nifs_matches_cpu_and_verifies() {
    let r1cs = relation();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4d45_5441_4c32).expect("preprocess");
    let fresh = direct_ccs::build_instance(&prep, &r1cs, &assignment(1, 0)).expect("fresh instance");
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
    .expect("CPU NIFS proof");

    let mut metal = MetalNifsProver::new().expect("Metal prover");
    let mut metal_transcript = Transcript::session();
    let metal_output = nifs::prove_with_adapter(
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
    .expect("Metal NIFS proof");

    assert_eq!(metal_output.0.claims, cpu.0.claims);
    assert_eq!(metal_output.0.witnesses, cpu.0.witnesses);
    assert_eq!(metal_output.0.parent_authority, cpu.0.parent_authority);
    assert_eq!(metal_output.1.pi_ccs.outputs, cpu.1.pi_ccs.outputs);
    assert_eq!(metal_output.1.pi_ccs.outputs_digest, cpu.1.pi_ccs.outputs_digest);
    assert_eq!(metal_output.1.pi_rlc.combined, cpu.1.pi_rlc.combined);
    assert_eq!(metal_output.1.pi_dec.children, cpu.1.pi_dec.children);
    assert_eq!(
        serde_json::to_vec(&metal_output.1.pi_ccs.sumcheck).expect("Metal sumcheck JSON"),
        serde_json::to_vec(&cpu.1.pi_ccs.sumcheck).expect("CPU sumcheck JSON"),
    );

    let profile = metal.last_profile().expect("Metal profile");
    assert!(profile.pi_ccs.fe.rounds > 0);
    assert!(profile.pi_ccs.fe.mcs_tables > 0);
    assert!(profile.pi_ccs.fe.on_metal);
    assert!(profile.pi_ccs.ajtai.y_eval_on_metal);
    assert!(!profile.pi_ccs.ajtai.y_eval.is_zero());
    assert!(profile.pi_ccs.nc.rounds > 0);
    assert!(profile.pi_ccs.nc.on_metal);
    assert!(profile.pi_ccs.nc.mask_native_on_metal);
    assert!(profile.pi_ccs.witness_masks_shared);
    assert!(profile.pi_rlc.witness_on_metal);
    assert!(profile.pi_rlc.witness_resident_only);
    assert!(profile.pi_rlc.witness_masks_reused);
    assert!(profile.pi_rlc.rho_small_coefficients);
    assert!(profile.pi_dec.split_on_metal);
    assert!(profile.pi_dec.recomposition_on_metal);
    assert!(profile.pi_dec.forms_on_metal);
    assert!(profile.pi_dec.y_on_metal);
    assert!(profile.pi_dec.commit_on_metal);
    assert!(profile.residency.proof_deferred);
    assert!(profile.residency.running_deferred);
    assert!(!profile.residency.recursive_compile_reverify_required);
    assert!(!metal.requires_recursive_compile_reverify());
    assert!(profile.activity.dispatches > 0);
    assert!(profile.activity.host_waits + 2 <= profile.activity.command_buffers);

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
        &metal_output.1,
    )
    .expect("verify Metal proof");
    assert_eq!(verified.claims, metal_output.0.claims);
}

#[test]
fn metal_factored_fe_zero_selector_groups_match_cpu_and_verify() {
    let fixtures = [
        selective_factored_arm(13),
        selective_factored_arm(29),
        selective_factored_arm(47),
    ];
    let arms = fixtures
        .iter()
        .map(|(shape, _)| shape.clone())
        .collect::<Vec<_>>();
    let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&arms, 0, D, 0)
        .expect("build selective factored relation");
    let assignment = relation
        .encode(1, &fixtures[1].1)
        .expect("encode second selective arm");
    assert!(relation.is_satisfied(&assignment));

    let structure = relation.structure().clone();
    let params = neo_fold_clean::config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("selective parameters");
    let log = direct_ccs::ajtai::setup_seeded(&params, &structure, 0x4d45_5441_4c5f);
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
    .expect("CPU selective proof");

    let mut metal = MetalNifsProver::new().expect("Metal prover");
    let mut metal_transcript = Transcript::session();
    let metal_output = nifs::prove_with_adapter(
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
    .expect("Metal selective proof");

    assert_eq!(metal_output.0.claims, cpu.0.claims);
    assert_eq!(metal_output.0.witnesses, cpu.0.witnesses);
    assert_eq!(metal_output.1.pi_ccs.outputs, cpu.1.pi_ccs.outputs);
    assert_eq!(metal_output.1.pi_rlc.combined, cpu.1.pi_rlc.combined);
    assert_eq!(metal_output.1.pi_dec.children, cpu.1.pi_dec.children);
    assert_eq!(
        serde_json::to_vec(&metal_output.1.pi_ccs.sumcheck).expect("Metal selective sumcheck JSON"),
        serde_json::to_vec(&cpu.1.pi_ccs.sumcheck).expect("CPU selective sumcheck JSON"),
    );
    let profile = metal.last_profile().expect("selective Metal profile");
    assert!(profile.pi_ccs.fe.on_metal);
    assert_eq!(profile.pi_ccs.fe.mcs_tables, 13);

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
        &metal_output.1,
    )
    .expect("verify Metal selective proof");
    assert_eq!(verified.claims, metal_output.0.claims);
}

#[test]
fn metal_nifs_tamper_is_rejected_by_canonical_verifier() {
    let r1cs = relation();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4d45_5441_4c33).expect("preprocess");
    let fresh = direct_ccs::build_instance(&prep, &r1cs, &assignment(0, 1)).expect("fresh instance");
    let fresh_claims = vec![fresh.claim.clone()];
    let running = RunningInstance::default();
    let mut metal = MetalNifsProver::new().expect("Metal prover");
    let mut transcript = Transcript::session();
    let (_, proof) = nifs::prove_with_adapter(
        &mut metal,
        &mut transcript,
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
    .expect("Metal NIFS proof");

    let mut commitment_tamper = proof.clone();
    commitment_tamper.pi_dec.children[0].c.data[0] += F::ONE;
    let mut verifier_transcript = Transcript::session();
    assert!(nifs::verify(
        &mut verifier_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &running,
        &commitment_tamper,
    )
    .is_err());

    let mut opening_tamper = proof;
    opening_tamper.pi_dec.children[0].y_ring[0][0] += K::ONE;
    let mut verifier_transcript = Transcript::session();
    assert!(nifs::verify(
        &mut verifier_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &running,
        &opening_tamper,
    )
    .is_err());
}

#[test]
fn metal_fresh_commitment_matches_canonical_constructor() {
    let r1cs = relation();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4d45_5441_4c34).expect("preprocess");
    let values = assignment(1, 0);
    let canonical = direct_ccs::build_instance(&prep, &r1cs, &values).expect("canonical instance");
    let mut metal = MetalNifsProver::new().expect("Metal prover");
    let assignments = [values.as_slice()];
    let built = metal
        .build_fresh_instances(NifsFreshInstancesRequest {
            pp: &prep.params,
            s: prep.structure(),
            cache: prep.optimized_cache(),
            log: &prep.log,
            m_in: r1cs.m_in,
            assignments: &assignments,
            image_overlay: None,
            lane_scheme: None,
        })
        .expect("Metal fresh-instance build")
        .expect("Metal accepted low-norm assignment");
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
fn metal_fresh_nebula_lanes_match_canonical_commitments() {
    let r1cs = lane_relation();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4d45_5441_4c37).expect("preprocess");
    let values = lane_assignment(1, 0);
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
    let mut metal = MetalNifsProver::new().expect("Metal prover");
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
        .expect("Metal accepted low-norm assignment");
    assert_eq!(built.len(), 1);
    assert_eq!(built[0].claim.c, canonical.claim.c);
    assert_eq!(built[0].claim.adv.as_ref(), Some(&expected));
    assert_eq!(built[0].witness.Z, canonical.witness.Z);
}

#[test]
fn metal_nifs_matches_cpu_across_bootstrap_and_steady_folds() {
    let r1cs = relation();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4d45_5441_4c35).expect("preprocess");
    let batches = [assignment(1, 0), assignment(0, 1), assignment(1, 0)];
    let mut cpu_running = RunningInstance::default();
    let mut metal_running = RunningInstance::default();
    let mut cpu_transcript = Transcript::session();
    let mut metal_transcript = Transcript::session();
    let mut metal = MetalNifsProver::new().expect("Metal prover");

    for (fold, values) in batches.iter().enumerate() {
        let instance = direct_ccs::build_instance(&prep, &r1cs, values).expect("fresh instance");
        let cpu = nifs::prove(
            &mut cpu_transcript,
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            &prep.log,
            None,
            prep.mix_rhos_commits(),
            prep.combine_b_pows(),
            vec![instance.clone()],
            &cpu_running,
        )
        .expect("CPU fold");
        let metal_output = nifs::prove_with_adapter(
            &mut metal,
            &mut metal_transcript,
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            &prep.log,
            None,
            prep.mix_rhos_commits(),
            prep.combine_b_pows(),
            vec![instance],
            &metal_running,
        )
        .expect("Metal fold");
        assert_eq!(metal_output.0.claims, cpu.0.claims, "running claims at fold {fold}");
        assert_eq!(
            metal_output.0.witnesses, cpu.0.witnesses,
            "running witnesses at fold {fold}"
        );
        assert_eq!(metal_output.0.parent_authority, cpu.0.parent_authority);
        assert_eq!(metal_output.1.pi_ccs.outputs, cpu.1.pi_ccs.outputs);
        assert_eq!(metal_output.1.pi_rlc.combined, cpu.1.pi_rlc.combined);
        assert_eq!(metal_output.1.pi_dec.children, cpu.1.pi_dec.children);
        assert_eq!(
            serde_json::to_vec(&metal_output.1.pi_ccs.sumcheck).expect("Metal sumcheck JSON"),
            serde_json::to_vec(&cpu.1.pi_ccs.sumcheck).expect("CPU sumcheck JSON"),
        );
        let profile = metal.last_profile().expect("Metal fold profile");
        assert!(
            profile.pi_ccs.ajtai.y_eval_on_metal,
            "Ajtai Y_eval was not selected at fold {fold}"
        );
        assert!(
            profile.pi_ccs.nc.on_metal && profile.pi_ccs.nc.mask_native_on_metal,
            "mask-native NC was not selected at fold {fold}"
        );
        if fold > 0 {
            assert!(
                profile.pi_ccs.nc.active_witnesses < profile.pi_ccs.nc.input_witnesses,
                "steady fold {fold} must exercise canonical expansion around zero witnesses"
            );
        }
        cpu_running = cpu.0;
        metal_running = metal_output.0;
    }
}

#[test]
fn metal_nc_live_prefix_ignores_recycled_suffix_across_witnesses() {
    let mut metal = MetalNifsProver::new().expect("Metal prover");
    let running = RunningInstance::default();

    // Both relations occupy twelve mask blocks and a 1024-row padded table.
    // The first leaves nonzero data in dense row five; the second has only five
    // live dense rows, so its odd final pair must treat that recycled row as zero.
    let poison_columns = 12 * D;
    let poison_r1cs = wide_relation(poison_columns);
    let poison_prep = direct_ccs::preprocess_seeded(&poison_r1cs, 0x4d45_5441_4c5b).expect("poison preprocess");
    let poison_instances = [
        direct_ccs::build_instance(
            &poison_prep,
            &poison_r1cs,
            &wide_assignment(poison_columns, poison_columns - 1),
        )
        .expect("first poison instance"),
        direct_ccs::build_instance(
            &poison_prep,
            &poison_r1cs,
            &wide_assignment(poison_columns, poison_columns - 2),
        )
        .expect("second poison instance"),
    ];
    let mut cpu_transcript = Transcript::session();
    let poison_cpu = nifs::prove(
        &mut cpu_transcript,
        &poison_prep.params,
        poison_prep.structure(),
        poison_prep.optimized_cache(),
        &poison_prep.log,
        None,
        poison_prep.mix_rhos_commits(),
        poison_prep.combine_b_pows(),
        poison_instances.to_vec(),
        &running,
    )
    .expect("CPU poison fold");
    let mut metal_transcript = Transcript::session();
    let poison_metal = nifs::prove_with_adapter(
        &mut metal,
        &mut metal_transcript,
        &poison_prep.params,
        poison_prep.structure(),
        poison_prep.optimized_cache(),
        &poison_prep.log,
        None,
        poison_prep.mix_rhos_commits(),
        poison_prep.combine_b_pows(),
        poison_instances.to_vec(),
        &running,
    )
    .expect("Metal poison fold");
    assert_eq!(poison_metal.1.pi_ccs.outputs, poison_cpu.1.pi_ccs.outputs);
    assert_eq!(
        serde_json::to_vec(&poison_metal.1.pi_ccs.sumcheck).expect("Metal poison sumcheck JSON"),
        serde_json::to_vec(&poison_cpu.1.pi_ccs.sumcheck).expect("CPU poison sumcheck JSON"),
    );

    let live_columns = 11 * D + 1;
    let live_r1cs = wide_relation(live_columns);
    let live_prep = direct_ccs::preprocess_seeded(&live_r1cs, 0x4d45_5441_4c5c).expect("live-prefix preprocess");
    let live_instances = [
        direct_ccs::build_instance(&live_prep, &live_r1cs, &wide_assignment(live_columns, live_columns - 1))
            .expect("first live-prefix instance"),
        direct_ccs::build_instance(&live_prep, &live_r1cs, &wide_assignment(live_columns, live_columns - 2))
            .expect("second live-prefix instance"),
    ];
    let mut cpu_transcript = Transcript::session();
    let live_cpu = nifs::prove(
        &mut cpu_transcript,
        &live_prep.params,
        live_prep.structure(),
        live_prep.optimized_cache(),
        &live_prep.log,
        None,
        live_prep.mix_rhos_commits(),
        live_prep.combine_b_pows(),
        live_instances.to_vec(),
        &running,
    )
    .expect("CPU live-prefix fold");
    let mut metal_transcript = Transcript::session();
    let live_metal = nifs::prove_with_adapter(
        &mut metal,
        &mut metal_transcript,
        &live_prep.params,
        live_prep.structure(),
        live_prep.optimized_cache(),
        &live_prep.log,
        None,
        live_prep.mix_rhos_commits(),
        live_prep.combine_b_pows(),
        live_instances.to_vec(),
        &running,
    )
    .expect("Metal live-prefix fold");
    assert_eq!(live_metal.0.claims, live_cpu.0.claims);
    assert_eq!(live_metal.0.witnesses, live_cpu.0.witnesses);
    assert_eq!(live_metal.1.pi_ccs.outputs, live_cpu.1.pi_ccs.outputs);
    assert_eq!(live_metal.1.pi_rlc.combined, live_cpu.1.pi_rlc.combined);
    assert_eq!(live_metal.1.pi_dec.children, live_cpu.1.pi_dec.children);
    assert_eq!(
        serde_json::to_vec(&live_metal.1.pi_ccs.sumcheck).expect("Metal live-prefix sumcheck JSON"),
        serde_json::to_vec(&live_cpu.1.pi_ccs.sumcheck).expect("CPU live-prefix sumcheck JSON"),
    );
    let profile = metal.last_profile().expect("live-prefix Metal profile");
    assert!(profile.pi_ccs.nc.mask_native_on_metal);
    assert!(profile.pi_ccs.nc.active_witnesses >= 2);
}

#[test]
fn metal_fe_live_prefix_ignores_recycled_odd_suffix() {
    let mut metal = MetalNifsProver::new().expect("Metal prover");
    let running = RunningInstance::default();

    // The six-row proof leaves a nonzero sixth MCS row behind. The five-row
    // proof has the same padded domain and two live witnesses, but its odd
    // final pair must consume the zero sentinel instead of that stale row.
    let poison_r1cs = repeated_row_relation(6);
    let poison_prep = direct_ccs::preprocess_seeded(&poison_r1cs, 0x4d45_5441_4c5d).expect("poison preprocess");
    let poison_instances = (0..2)
        .map(|_| {
            direct_ccs::build_instance(&poison_prep, &poison_r1cs, &repeated_row_assignment()).expect("poison instance")
        })
        .collect::<Vec<_>>();
    let mut poison_transcript = Transcript::session();
    nifs::prove_with_adapter(
        &mut metal,
        &mut poison_transcript,
        &poison_prep.params,
        poison_prep.structure(),
        poison_prep.optimized_cache(),
        &poison_prep.log,
        None,
        poison_prep.mix_rhos_commits(),
        poison_prep.combine_b_pows(),
        poison_instances,
        &running,
    )
    .expect("Metal poison fold");

    let live_r1cs = repeated_row_relation(5);
    let live_prep = direct_ccs::preprocess_seeded(&live_r1cs, 0x4d45_5441_4c5e).expect("live preprocess");
    let live_instances = (0..2)
        .map(|_| direct_ccs::build_instance(&live_prep, &live_r1cs, &repeated_row_assignment()).expect("live instance"))
        .collect::<Vec<_>>();
    let live_claims = live_instances
        .iter()
        .map(|instance| instance.claim.clone())
        .collect::<Vec<_>>();
    let mut cpu_transcript = Transcript::session();
    let cpu = nifs::prove(
        &mut cpu_transcript,
        &live_prep.params,
        live_prep.structure(),
        live_prep.optimized_cache(),
        &live_prep.log,
        None,
        live_prep.mix_rhos_commits(),
        live_prep.combine_b_pows(),
        live_instances.clone(),
        &running,
    )
    .expect("CPU live fold");
    let mut metal_transcript = Transcript::session();
    let metal_output = nifs::prove_with_adapter(
        &mut metal,
        &mut metal_transcript,
        &live_prep.params,
        live_prep.structure(),
        live_prep.optimized_cache(),
        &live_prep.log,
        None,
        live_prep.mix_rhos_commits(),
        live_prep.combine_b_pows(),
        live_instances,
        &running,
    )
    .expect("Metal live fold");

    assert_eq!(metal_output.0.claims, cpu.0.claims);
    assert_eq!(metal_output.0.witnesses, cpu.0.witnesses);
    assert_eq!(metal_output.1.pi_ccs.outputs, cpu.1.pi_ccs.outputs);
    assert_eq!(
        serde_json::to_vec(&metal_output.1.pi_ccs.sumcheck).expect("Metal live sumcheck JSON"),
        serde_json::to_vec(&cpu.1.pi_ccs.sumcheck).expect("CPU live sumcheck JSON"),
    );
    assert!(
        metal
            .last_profile()
            .expect("Metal profile")
            .pi_ccs
            .fe
            .on_metal
    );

    let mut verifier_transcript = Transcript::session();
    let verified = nifs::verify(
        &mut verifier_transcript,
        &live_prep.params,
        live_prep.structure(),
        live_prep.optimized_cache(),
        live_prep.mix_rhos_commits(),
        live_prep.combine_b_pows(),
        &live_claims,
        &running,
        &metal_output.1,
    )
    .expect("verify Metal live-prefix proof");
    assert_eq!(verified.claims, metal_output.0.claims);
}

#[test]
fn metal_nifs_all_zero_running_witnesses_match_cpu() {
    let r1cs = relation();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4d45_5441_4c5a).expect("preprocess");
    let running = RunningInstance::canonical_zero(
        &prep.params,
        prep.structure(),
        r1cs.m_in,
        neo_fold_clean::paper::construction2::LaneCommitmentMode::Plain,
    )
    .expect("canonical zero running instance");
    let zero_assignment = vec![F::ZERO; 2 * D];
    let fresh = direct_ccs::build_instance(&prep, &r1cs, &zero_assignment).expect("all-zero fresh instance");
    assert!(!running.witnesses.is_empty());
    assert!((0..fresh.witness.Z.rows())
        .all(|row| (0..fresh.witness.Z.cols()).all(|column| fresh.witness.Z[(row, column)] == F::ZERO)));
    assert!(running.witnesses.iter().all(|witness| {
        (0..witness.rows()).all(|row| (0..witness.cols()).all(|column| witness[(row, column)] == F::ZERO))
    }));

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
    .expect("CPU all-zero fold");

    let mut metal = MetalNifsProver::new().expect("Metal prover");
    let mut metal_transcript = Transcript::session();
    let metal_output = nifs::prove_with_adapter(
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
    .expect("Metal all-zero fold");

    assert_eq!(metal_output.0.claims, cpu.0.claims);
    assert_eq!(metal_output.0.witnesses, cpu.0.witnesses);
    assert_eq!(metal_output.0.parent_authority, cpu.0.parent_authority);
    assert_eq!(metal_output.1.pi_ccs.outputs, cpu.1.pi_ccs.outputs);
    assert_eq!(metal_output.1.pi_rlc.combined, cpu.1.pi_rlc.combined);
    assert_eq!(metal_output.1.pi_dec.children, cpu.1.pi_dec.children);
    assert_eq!(
        serde_json::to_vec(&metal_output.1.pi_ccs.sumcheck).expect("Metal sumcheck JSON"),
        serde_json::to_vec(&cpu.1.pi_ccs.sumcheck).expect("CPU sumcheck JSON"),
    );
    assert!(metal_output
        .1
        .pi_ccs
        .outputs
        .iter()
        .flat_map(|claim| &claim.y_ring)
        .flatten()
        .all(|&value| value == K::ZERO));
    let profile = metal.last_profile().expect("Metal all-zero profile");
    assert!(profile.pi_ccs.ajtai.y_eval_on_metal);
    assert_eq!(profile.pi_ccs.nc.input_witnesses, running.witnesses.len() + 1);
}

#[test]
fn metal_resident_snapshot_survives_a_later_generation() {
    let r1cs = relation();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4d45_5441_4c37).expect("preprocess");
    let first = direct_ccs::build_instance(&prep, &r1cs, &assignment(1, 0)).expect("first instance");
    let second = direct_ccs::build_instance(&prep, &r1cs, &assignment(0, 1)).expect("second instance");
    let empty = RunningInstance::default();

    let mut cpu_transcript = Transcript::session();
    let expected_first = nifs::prove(
        &mut cpu_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![first.clone()],
        &empty,
    )
    .expect("CPU first fold")
    .0;

    let mut metal = MetalNifsProver::new().expect("Metal prover");
    let mut metal_transcript = Transcript::session();
    let first_output = metal
        .prove(NifsProverRequest {
            tr: &mut metal_transcript,
            pp: &prep.params,
            s: prep.structure(),
            cache: prep.optimized_cache(),
            log: &prep.log,
            lanes: None,
            mix_rhos_commits: prep.mix_rhos_commits(),
            combine_b_pows: prep.combine_b_pows(),
            fresh: vec![first],
            running_carrier: None,
            running: &empty,
            cache_output_for_next_step: true,
        })
        .expect("first resident Metal fold");
    let (first_carrier, _first_proof) = first_output.into_carriers();
    let first_running_input = first_carrier
        .materialize_prover_input()
        .expect("first resident prover input");

    let _second_output = metal
        .prove(NifsProverRequest {
            tr: &mut metal_transcript,
            pp: &prep.params,
            s: prep.structure(),
            cache: prep.optimized_cache(),
            log: &prep.log,
            lanes: None,
            mix_rhos_commits: prep.mix_rhos_commits(),
            combine_b_pows: prep.combine_b_pows(),
            fresh: vec![second],
            running_carrier: Some(&first_carrier),
            running: &first_running_input,
            cache_output_for_next_step: true,
        })
        .expect("second resident Metal fold");

    let materialized_first = first_carrier
        .materialize()
        .expect("materialize first carrier after advancing the session");
    assert_eq!(materialized_first.claims, expected_first.claims);
    assert_eq!(materialized_first.witnesses, expected_first.witnesses);
    assert_eq!(materialized_first.parent_authority, expected_first.parent_authority);
}

#[test]
fn metal_nifs_rejects_earlier_running_and_transcript_tampering() {
    let r1cs = relation();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4d45_5441_4c36).expect("preprocess");
    let first = direct_ccs::build_instance(&prep, &r1cs, &assignment(1, 0)).expect("first instance");
    let second = direct_ccs::build_instance(&prep, &r1cs, &assignment(0, 1)).expect("second instance");
    let empty = RunningInstance::default();
    let mut metal = MetalNifsProver::new().expect("Metal prover");
    let mut prover_transcript = Transcript::session();
    let first_output = nifs::prove_with_adapter(
        &mut metal,
        &mut prover_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![first.clone()],
        &empty,
    )
    .expect("first Metal fold");
    let second_output = nifs::prove_with_adapter(
        &mut metal,
        &mut prover_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![second.clone()],
        &first_output.0,
    )
    .expect("second Metal fold");

    let mut transcript_tamper = Transcript::session();
    transcript_tamper.append_message(b"metal/m5/tamper", b"different transcript prefix");
    assert!(nifs::verify(
        &mut transcript_tamper,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &[first.claim.clone()],
        &empty,
        &first_output.1,
    )
    .is_err());

    let mut verifier_transcript = Transcript::session();
    let verified_first = nifs::verify(
        &mut verifier_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &[first.claim],
        &empty,
        &first_output.1,
    )
    .expect("verify first fold");
    let pending_projection = verified_first.pending_projection().cloned();
    let mut earlier_running_tamper = RunningInstance::new(
        verified_first.claims,
        first_output.0.witnesses.clone(),
        verified_first.parent_authority,
        pending_projection,
    );
    earlier_running_tamper.claims[0].c.data[0] += F::ONE;
    assert!(nifs::verify(
        &mut verifier_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &[second.claim],
        &earlier_running_tamper,
        &second_output.1,
    )
    .is_err());
}
