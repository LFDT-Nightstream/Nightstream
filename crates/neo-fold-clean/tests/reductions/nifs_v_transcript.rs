//! NIFS.V transcript-phase red-team tests.
//!
//! These tests build coherent lower-layer proof fragments under the wrong
//! Fiat-Shamir phase. They are meant to catch verifier gadgets that run the
//! right algebra with challenges sampled from the wrong transcript state.

use neo_ajtai::Commitment;
use neo_ccs::LaneCommitments;
use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, TranscriptGadget};
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::nifs;
use neo_fold_clean::paper::nifs::circuit::{
    enforce_nifs_v_circuit_with_transcript, NifsVCircuitConfig, NifsVCircuitMessages, NifsVOutputs,
};
use neo_fold_clean::paper::nifs::NifsProof;
use neo_fold_clean::paper::reductions::pi_ccs_circuit::{enforce_pi_ccs, PiCcsVerifierConfig, PiCcsVerifierMessages};
use neo_fold_clean::paper::relations::{superneo_public_x_cols, CcsClaim, LaneRanges, LaneScheme};
use neo_fold_clean::paper::{pi_dec, pi_rlc};
use neo_fold_clean::{CeClaim, Preprocessing};
use neo_math::ring::D;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

const SESSION_LABEL: &[u8] = b"neo.fold.clean/session/v1";

struct Fixture {
    prep: Preprocessing,
    fresh_claims: Vec<CcsClaim>,
    running: RunningInstance,
    proof: NifsProof,
    combined: CeClaim,
    children: Vec<CeClaim>,
}

fn three_term_addition() -> R1cs {
    // Three packed ring columns leave one whole column per Nebula lane in
    // the product-commitment fixture below. The first complete ring column
    // is public, as required by the selected PaddedRowIdentity carrier.
    let m = 3 * D;
    let mut a = Mat::zero(1, m, F::ZERO);
    a.set(0, 1, F::ONE);
    a.set(0, 2, F::ONE);
    let mut b = Mat::zero(1, m, F::ZERO);
    b.set(0, 0, F::ONE);
    let mut c = Mat::zero(1, m, F::ZERO);
    c.set(0, 3, F::ONE);
    R1cs { a, b, c, m_in: D }
}

fn assignment(a: u64, b: u64) -> Vec<F> {
    let mut z = vec![F::ZERO; 3 * D];
    z[0] = F::ONE;
    z[1] = F::from_u64(a);
    z[2] = F::from_u64(b);
    z[3] = F::from_u64(a + b);
    z
}

fn forged_adv(kappa: usize) -> LaneCommitments<Commitment> {
    let commitment = |marker: u64| {
        let mut data = vec![F::ZERO; D * kappa];
        data[0] = F::from_u64(marker);
        Commitment { d: D, kappa, data }
    };
    LaneCommitments {
        ops: commitment(1),
        is: commitment(2),
        fs: commitment(3),
    }
}

fn build_honest_fixture() -> Fixture {
    build_honest_fixture_with_adv(false)
}

fn build_honest_fixture_with_adv(with_adv: bool) -> Fixture {
    let r1cs = three_term_addition();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");
    let lanes = with_adv.then(|| {
        LaneScheme::from_seeds(
            prep.params.kappa() as usize,
            LaneRanges {
                ops: 0..1,
                is: 1..2,
                fs: 2..3,
            },
            [0xA5; 32],
            [0x5A; 32],
        )
        .expect("test lane scheme")
    });

    let mut first = direct_ccs::build_instance(&prep, &r1cs, &assignment(1, 0)).expect("first instance");
    if let Some(lanes) = &lanes {
        first.claim.adv = Some(lanes.commit(&first.witness.Z).expect("first adv"));
    }
    let mut first_tr = Transcript::session();
    let (running, _first_proof) = nifs::prove(
        &mut first_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        lanes.as_ref(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![first],
        &RunningInstance::default(),
    )
    .expect("first NIFS.P");

    let mut second = direct_ccs::build_instance(&prep, &r1cs, &assignment(0, 1)).expect("second instance");
    if let Some(lanes) = &lanes {
        second.claim.adv = Some(lanes.commit(&second.witness.Z).expect("second adv"));
    }
    let fresh_claims = vec![second.claim.clone()];

    let mut second_tr = Transcript::session();
    let (next_running, proof) = nifs::prove(
        &mut second_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        lanes.as_ref(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![second],
        &running,
    )
    .expect("second NIFS.P");

    Fixture {
        prep,
        fresh_claims,
        running,
        combined: proof.pi_rlc.combined.clone(),
        children: next_running.claims,
        proof,
    }
}

fn build_wrong_rlc_phase_fixture() -> Fixture {
    let r1cs = three_term_addition();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 43).expect("preprocess");

    let first = direct_ccs::build_instance(&prep, &r1cs, &assignment(1, 0)).expect("first instance");
    let mut first_tr = Transcript::session();
    let (running, _first_proof) = nifs::prove(
        &mut first_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![first],
        &RunningInstance::default(),
    )
    .expect("first NIFS.P");

    let second = direct_ccs::build_instance(&prep, &r1cs, &assignment(0, 1)).expect("second instance");
    let second_witness = second.witness.Z.clone();
    let fresh_claims = vec![second.claim.clone()];

    let mut honest_tr = Transcript::session();
    let (_honest_next, proof) = nifs::prove(
        &mut honest_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![second],
        &running,
    )
    .expect("second NIFS.P");

    let mut all_witnesses = Vec::with_capacity(1 + running.witnesses.len());
    all_witnesses.push(second_witness);
    all_witnesses.extend(running.witnesses.iter().cloned());

    // Adversarial construction: keep the honest Π_CCS proof and outputs, but
    // regenerate the Π_RLC parent from a fresh transcript that omitted the
    // Π_CCS header/instance/ME absorbs, sumcheck transcript, and header-digest
    // catch-up. Then regenerate Π_DEC children coherently from that wrong
    // parent. The composed NIFS.V circuit must reject because its ρ values are
    // verifier-derived from the post-Π_CCS transcript state.
    let mut wrong_rlc_tr = Transcript::session();
    let (wrong_rlc, _wrong_rlc_proof) = pi_rlc::prove(
        &mut wrong_rlc_tr,
        &prep.params,
        prep.structure(),
        prep.mix_rhos_commits(),
        &proof.pi_ccs.outputs,
        &all_witnesses,
    )
    .expect("wrong-phase Π_RLC.P");

    let (wrong_dec, _wrong_dec_proof) = pi_dec::prove(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.combine_b_pows(),
        &wrong_rlc.claim,
        &wrong_rlc.witness,
    )
    .expect("wrong-phase Π_DEC.P");

    Fixture {
        prep,
        fresh_claims,
        running,
        proof,
        combined: wrong_rlc.claim,
        children: wrong_dec.claims,
    }
}

fn pi_ccs_config<'a>(prep: &'a Preprocessing) -> PiCcsVerifierConfig<'a> {
    PiCcsVerifierConfig {
        params: &prep.params,
        structure: prep.structure().into(),
        matrix_digest: prep.pi_ccs_header_bundle(),
    }
}

fn emit_verifier(f: &Fixture) -> Result<(R1csBuilder, NifsVOutputs), neo_fold_clean::paper::nifs::circuit::Error> {
    let mut builder = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut builder, SESSION_LABEL);
    let cfg = NifsVCircuitConfig {
        pi_ccs: pi_ccs_config(&f.prep),
    };
    let outputs = enforce_nifs_v_circuit_with_transcript(
        &mut builder,
        &f.prep.params,
        &cfg,
        &mut tr,
        &NifsVCircuitMessages {
            fresh: &f.fresh_claims,
            running: &f.running.claims,
            running_parent_authority: f.running.parent_authority.as_ref(),
            pi_ccs: &f.proof.pi_ccs,
            combined: &f.combined,
            children: &f.children,
        },
    )?;
    Ok((builder, outputs))
}

#[test]
fn nifs_v_transcript_phase_accepts_honest_native_tail() {
    let fixture = build_honest_fixture();
    let (builder, outputs) = emit_verifier(&fixture).expect("NIFS.V synthesis");
    assert!(
        builder.is_satisfied(),
        "honest native NIFS tail must satisfy the composed verifier"
    );
    let unconstrained = builder.unconstrained_columns();
    assert!(
        unconstrained.is_empty(),
        "composed NIFS.V verifier left unconstrained columns: {unconstrained:?}"
    );
    let _ = outputs;
}

#[test]
fn nifs_v_projection_census_binds_every_rlc_ring_action_client() {
    let fixture = build_honest_fixture();
    let (mut builder, outputs) = emit_verifier(&fixture).expect("NIFS.V synthesis");
    assert!(builder.is_satisfied(), "baseline");

    assert_eq!(outputs.projection_q_lanes.len(), fixture.combined.c.kappa);
    assert_eq!(
        outputs.projection_x_q_lanes.len(),
        superneo_public_x_cols(fixture.combined.m_in)
    );
    assert_eq!(
        outputs.projection_y_ring_q_lanes.len(),
        fixture.prep.structure().t() + 1
    );
    assert!(outputs.projection_adv_q_lanes.is_none());

    let load_bearing = [
        outputs.projection_q_lanes[0][0],
        outputs.projection_x_q_lanes[0][0],
        outputs.projection_y_ring_q_lanes[0][0][0],
        outputs.projection_y_ring_q_lanes[0][1][0],
    ];
    for wire in load_bearing {
        let original = builder.witness()[wire.col()];
        builder.tamper_witness(wire.col(), original + F::ONE);
        assert!(
            !builder.is_satisfied(),
            "every transcript-bound c/X/y projection quotient must feed an enforced identity"
        );
        builder.tamper_witness(wire.col(), original);
        assert!(
            builder.is_satisfied(),
            "restoring one quotient must restore the baseline"
        );
    }
}

#[test]
fn nifs_v_accepts_honest_adv_product_commitment_tail() {
    let fixture = build_honest_fixture_with_adv(true);
    let (mut builder, outputs) = emit_verifier(&fixture).expect("NIFS.V synthesis with adv");
    assert!(
        builder.is_satisfied(),
        "honest native NIFS tail with the full product commitment must satisfy"
    );
    let expected = fixture.fresh_claims[0].adv.as_ref().expect("fresh adv");
    let surfaced = outputs.fresh_adv[0]
        .as_ref()
        .expect("surfaced fresh adv wires");
    for (expected_component, surfaced_component) in [
        (&expected.ops, &surfaced.ops),
        (&expected.is, &surfaced.is),
        (&expected.fs, &surfaced.fs),
    ] {
        assert_eq!(surfaced_component.d, expected_component.d);
        assert_eq!(surfaced_component.kappa, expected_component.kappa);
        let wire_values: Vec<F> = surfaced_component
            .data
            .iter()
            .map(|wire| builder.witness()[wire.col()])
            .collect();
        assert_eq!(wire_values, expected_component.data);
    }

    let projection = outputs
        .projection_adv_q_lanes
        .as_ref()
        .expect("adv projection quotients");
    for coordinate in [&projection.ops, &projection.is, &projection.fs] {
        assert_eq!(coordinate.len(), expected.ops.kappa);
    }
    let quotient_wire = projection.ops[0][0];
    let original = builder.witness()[quotient_wire.col()];
    builder.tamper_witness(quotient_wire.col(), original + F::ONE);
    assert!(
        !builder.is_satisfied(),
        "the transcript-bound adv quotient must feed the enforced projection identity"
    );
}

#[test]
fn nifs_v_rejects_rlc_and_dec_tail_proved_under_fresh_transcript() {
    let fixture = build_wrong_rlc_phase_fixture();
    let (builder, _outputs) = emit_verifier(&fixture).expect("NIFS.V synthesis");
    assert!(
        !builder.is_satisfied(),
        "NIFS.V accepted a coherent Π_RLC/Π_DEC tail proved under a fresh transcript; \
         Π_RLC ρ must be sampled from the post-Π_CCS transcript state"
    );
}

/// The circuit must bind the fresh claim's `adv` tuple and enforce Pi_CCS
/// forwarding. Rejection may happen structurally during synthesis or as an
/// unsatisfied equality row.
#[test]
fn nifs_v_must_reject_unforwarded_fresh_adv() {
    let mut fixture = build_honest_fixture();
    let kappa = fixture.fresh_claims[0].c.kappa;
    fixture.fresh_claims[0].adv = Some(forged_adv(kappa));

    if let Ok((builder, _outputs)) = emit_verifier(&fixture) {
        assert!(
            !builder.is_satisfied(),
            "NIFS.V accepted a fresh adv tuple that is absent from the proved Pi_CCS output"
        );
    }
}

#[test]
fn recursive_pi_ccs_rejects_unforwarded_output_adv() {
    for mutate_running_output in [false, true] {
        let mut fixture = build_honest_fixture_with_adv(true);
        let output_index = if mutate_running_output {
            fixture.fresh_claims.len()
        } else {
            0
        };
        fixture.proof.pi_ccs.outputs[output_index]
            .adv
            .as_mut()
            .expect("output adv")
            .ops
            .data[0] += F::ONE;

        let mut native_tr = Transcript::session();
        assert!(
            neo_fold_clean::paper::pi_ccs::verify(
                &mut native_tr,
                &fixture.prep.params,
                fixture.prep.structure(),
                fixture.prep.optimized_cache(),
                &fixture.fresh_claims,
                &fixture.running,
                &fixture.proof.pi_ccs,
            )
            .is_err(),
            "fixture precondition: native PiCCS.V must reject output adv that was not forwarded"
        );

        let mut builder = R1csBuilder::new();
        let mut circuit_tr = TranscriptGadget::new(&mut builder, SESSION_LABEL);
        let messages = PiCcsVerifierMessages {
            fresh: &fixture.fresh_claims,
            running: &fixture.running.claims,
            running_parent_authority: fixture.running.parent_authority.as_ref(),
            outputs: &fixture.proof.pi_ccs.outputs,
            sumcheck_rounds: &fixture.proof.pi_ccs.sumcheck.sumcheck_rounds,
        };
        enforce_pi_ccs(&mut builder, &mut circuit_tr, &pi_ccs_config(&fixture.prep), &messages)
            .expect("recursive PiCCS.V synthesis");
        assert!(
            !builder.is_satisfied(),
            "native/recursive differential: recursive PiCCS.V accepted output adv that differs from its input"
        );
    }
}

#[test]
fn nifs_v_rejects_coherent_dec_adv_parent_unlinked_from_rlc_outputs() {
    let mut fixture = build_honest_fixture_with_adv(true);
    fixture
        .combined
        .adv
        .as_mut()
        .expect("combined adv")
        .ops
        .data[0] += F::ONE;
    fixture.children[0]
        .adv
        .as_mut()
        .expect("child adv")
        .ops
        .data[0] += F::ONE;

    if let Ok((builder, _outputs)) = emit_verifier(&fixture) {
        assert!(
            !builder.is_satisfied(),
            "NIFS.V accepted an adv parent/child mutation unrelated to the Pi_CCS outputs"
        );
    }
}

#[test]
fn recursive_nifs_v_proof_values_keep_one_relation_shape() {
    let honest = build_honest_fixture();
    let honest_builder = emit_verifier(&honest).expect("honest verifier synthesis").0;
    assert!(honest_builder.is_satisfied());

    let mut forged_values = build_honest_fixture();
    forged_values.proof.pi_ccs.sumcheck.sumcheck_rounds[0][0] += K::ONE;
    let forged_builder = emit_verifier(&forged_values)
        .expect("forged proof values still synthesize the fixed relation")
        .0;
    assert!(!forged_builder.is_satisfied());
    let honest_snapshot = honest_builder.snapshot();
    let forged_snapshot = forged_builder.snapshot();
    assert!(
        honest_snapshot.has_same_relation(&forged_snapshot),
        "Π_CCS proof values must not change the verifier relation"
    );
}
