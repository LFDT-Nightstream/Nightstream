#![cfg(target_vendor = "apple")]

use neo_ccs::Mat;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::nifs::{self, NifsFreshInstancesRequest, NifsProverAdapter};
use neo_fold_clean::paper::relations::{LaneRanges, LaneScheme};
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
    let assignments = [values.as_slice()];
    let mut metal = MetalNifsProver::new().expect("Metal prover");
    let built = metal
        .build_fresh_instances(NifsFreshInstancesRequest {
            pp: &prep.params,
            s: prep.structure(),
            cache: prep.optimized_cache(),
            log: &prep.log,
            m_in: r1cs.m_in,
            assignments: &assignments,
            image_overlay: None,
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
        assert!(
            metal
                .last_profile()
                .is_some_and(|profile| profile.pi_ccs.ajtai.y_eval_on_metal),
            "Ajtai Y_eval was not selected at fold {fold}"
        );
        assert!(
            metal
                .last_profile()
                .is_some_and(|profile| profile.pi_ccs.nc.on_metal && profile.pi_ccs.nc.mask_native_on_metal),
            "mask-native NC was not selected at fold {fold}"
        );
        cpu_running = cpu.0;
        metal_running = metal_output.0;
    }
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
    let mut earlier_running_tamper = RunningInstance {
        claims: verified_first.claims,
        witnesses: first_output.0.witnesses.clone(),
        parent_authority: verified_first.parent_authority,
    };
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
