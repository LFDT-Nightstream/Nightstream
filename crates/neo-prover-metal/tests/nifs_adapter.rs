#![cfg(all(feature = "metal", target_vendor = "apple", neo_metal_shaders))]

//! Metal adapter boundary tests.
//!
//! The adapter owns fresh-instance commitments. The canonical host prover
//! owns PiCCS, PiRLC, PiDEC, transcript order, and proof bytes.

use neo_ccs::Mat;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::nifs::{
    self, NifsFreshInstancesRequest, NifsFreshSignedUnitAssignment, NifsFreshSignedUnitInstancesRequest,
    NifsProverAdapter,
};
use neo_fold_clean::paper::relations::{LaneRanges, LaneScheme};
use neo_fold_clean::RunningInstance;
use neo_math::{D, F};
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

#[test]
fn metal_adapter_delegates_the_one_joint_proof_to_the_canonical_host() {
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
        vec![fresh],
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
            image_overlay: None,
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
            image_overlay: None,
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
