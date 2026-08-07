//! NIFS.V composition — hard-gate parity test.
//!
//! A real native `nifs::prove` proof must satisfy the composed in-circuit
//! [`enforce_nifs_v_circuit_with_transcript`], and targeted mutations of
//! the proof must cause `R1csBuilder::is_satisfied()` to return false.
//!
//! Tests:
//! - `nifs_v_accepts_native_proof`
//! - `nifs_v_rejects_fresh_count_above_rlc_guard_native`
//! - `nifs_v_circuit_rejects_fresh_count_above_rlc_guard`
//! - `nifs_v_rejects_tampered_fe_round`
//! - `nifs_v_rejects_parent_authority_when_running_is_empty`
//! - `nifs_v_rejects_nonempty_running_without_parent_authority`
//! - `nifs_v_rejects_tampered_running_parent_authority`
//! - `nifs_v_rejects_tampered_running_parent_authority_r_c1_limb`
//! - `nifs_v_rejects_tampered_running_parent_authority_y_ring_c1_limb`
//! - `nifs_v_rejects_tampered_running_parent_authority_ct_c1_limb`
//! - `nifs_v_rejects_tampered_running_parent_authority_fold_digest`
//! - `nifs_v_rejects_tampered_running_child_r_c1_limb`
//! - `nifs_v_rejects_tampered_running_child_y_ring_c1_limb`
//! - `nifs_v_rejects_tampered_running_child_ct_c1_limb`
//! - `nifs_v_rejects_tampered_running_child_fold_digest`
//! - `nifs_v_rejects_tampered_pi_ccs_fresh_output_y_ring_non_ct_lane`
//! - `nifs_v_rejects_tampered_pi_ccs_fresh_output_y_ring_padding_lane`
//! - `nifs_v_rejects_tampered_combined_y_ring_non_ct_c1_limb`
//! - `nifs_v_rejects_tampered_combined_r_point`
//! - `nifs_v_rejects_tampered_combined_r_c1_limb`
//! - `nifs_v_rejects_tampered_child_commitment_lane`
//! - `nifs_v_rejects_tampered_child_x_active_lane`
//! - `nifs_v_rejects_noncanonical_x_width_in_dec_child`
//! - `nifs_v_rejects_noncanonical_x_width_in_running`
//! - `nifs_v_rejects_tampered_child_y_ring_lane`
//! - `nifs_v_rejects_canceling_child_y_ring_padding_lanes`
//! - `nifs_v_rejects_tampered_combined_ct_lane`
//! - `nifs_v_rejects_tampered_child_ct_lane`
//! - `nifs_v_rejects_tampered_child_ct_c1_limb`
//! - `nifs_v_rejects_tampered_child_r_point`
//! - `nifs_v_rejects_extra_self_consistent_y_ring_row`
//! - `nifs_v_rejects_extra_self_consistent_running_y_ring_row`
//! - `nifs_v_rejects_child_m_in_drift`
//! - `nifs_v_rejects_parent_m_in_drift`
//! - `nifs_v_rejects_tampered_pi_ccs_output_fold_digest`
//! - `nifs_v_rejects_tampered_combined_fold_digest`
//! - `nifs_v_rejects_tampered_child_fold_digest`
//! - `nifs_v_rejects_tampered_header_digest`
//! - `nifs_v_rejects_coherent_forged_fold_digest_chain`

#![allow(non_snake_case)]

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, TranscriptGadget};
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::nifs::circuit::{
    enforce_nifs_v_circuit_with_transcript, NifsVCircuitConfig, NifsVCircuitMessages, NifsVOutputs,
};
use neo_fold_clean::paper::nifs::NifsProof;
use neo_fold_clean::paper::reductions::pi_ccs_circuit::PiCcsVerifierConfig;
use neo_fold_clean::paper::relations::{CcsClaim, CeClaim};
use neo_math::ring::D;
use neo_math::{KExtensions, F, K};
use neo_transcript::{Poseidon2Transcript, Transcript as NeoTranscript};
use p3_field::{Field, PrimeCharacteristicRing};

const SESSION_LABEL: &[u8] = b"neo.fold.clean/session/v1";

fn k_c1_one() -> K {
    K::from_coeffs([F::ZERO, F::ONE])
}

fn append_noncanonical_x_column(claim: &mut CeClaim) {
    let old_cols = claim.X.cols();
    let mut widened = Mat::zero(D, old_cols + 1, F::ZERO);
    for row in 0..D {
        for column in 0..old_cols {
            widened[(row, column)] = claim.X[(row, column)];
        }
    }
    claim.X = widened;
}

fn three_term_addition() -> R1cs {
    let m = D;
    let mut a = Mat::zero(1, m, F::ZERO);
    a.set(0, 1, F::ONE);
    a.set(0, 2, F::ONE);
    let mut b = Mat::zero(1, m, F::ZERO);
    b.set(0, 0, F::ONE);
    let mut c = Mat::zero(1, m, F::ZERO);
    c.set(0, 3, F::ONE);
    // The selected SuperNeo profile exposes complete degree-D ring slots.
    // Coordinates 3..D are explicit public zeros in this fixture.
    R1cs { a, b, c, m_in: D }
}

fn assignment(a: u64, b: u64) -> Vec<F> {
    let mut z = vec![F::ZERO; D];
    z[0] = F::ONE;
    z[1] = F::from_u64(a);
    z[2] = F::from_u64(b);
    z[3] = F::from_u64(a + b);
    z
}

/// Test fixture: one NIFS fold seeds the running accumulator, then a second
/// NIFS step produces the proof we'll feed to the in-circuit verifier.
struct Fixture {
    prep: neo_fold_clean::Preprocessing,
    fresh_claims: Vec<CcsClaim>,
    running: RunningInstance,
    proof: NifsProof,
    combined: neo_fold_clean::CeClaim,
    children: Vec<neo_fold_clean::CeClaim>,
}

fn build_fixture() -> Fixture {
    let r1cs = three_term_addition();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");

    // First fold: seed running accumulator.
    let first = direct_ccs::build_instance(&prep, &r1cs, &assignment(1, 0)).expect("first instance");
    let mut first_tr = Transcript::session();
    let (running, _first_proof) = neo_fold_clean::paper::nifs::prove(
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

    // Second fold: this is the proof the circuit must accept.
    let second = direct_ccs::build_instance(&prep, &r1cs, &assignment(0, 1)).expect("second instance");
    let fresh_claims = vec![second.claim.clone()];

    let mut tr = Transcript::session();
    let (next_running, proof) = neo_fold_clean::paper::nifs::prove(
        &mut tr,
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

    // Π_RLC produces `combined` (parent of Π_DEC); Π_DEC produces children
    // (the new running accumulator). Both are part of `proof`; we surface
    // them as standalone refs so the circuit driver can wire them as
    // `msg.combined` / `msg.children`.
    let combined = proof.pi_rlc.combined.clone();
    let children: Vec<_> = next_running.claims.clone();

    Fixture {
        prep,
        fresh_claims,
        running,
        proof,
        combined,
        children,
    }
}

fn append_zero_commitment_column(claim: &mut neo_fold_clean::CeClaim) {
    claim.c.kappa += 1;
    claim.c.data.extend(std::iter::repeat(F::ZERO).take(D));
}

fn trivial_public_dec_children(
    prep: &neo_fold_clean::Preprocessing,
    parent: &neo_fold_clean::CeClaim,
) -> Vec<neo_fold_clean::CeClaim> {
    let k = prep.params.k_rho() as usize;
    let d_pad = D.next_power_of_two();
    let mut children = Vec::with_capacity(k);
    for idx in 0..k {
        if idx == 0 {
            children.push(parent.clone());
            continue;
        }
        children.push(neo_fold_clean::CeClaim {
            adv: None,
            c: Commitment::zeros(parent.c.d, parent.c.kappa),
            X: Mat::zero(parent.X.rows(), parent.X.cols(), F::ZERO),
            r: parent.r.clone(),
            y_ring: vec![vec![K::ZERO; d_pad]; parent.y_ring.len()],
            ct: vec![K::ZERO; parent.ct.len()],
            m_in: parent.m_in,
            fold_digest: parent.fold_digest,
        });
    }
    children
}

fn widen_x_with_zero_col(x: &Mat<F>) -> Mat<F> {
    let mut widened = Mat::zero(x.rows(), x.cols() + 1, F::ZERO);
    for r in 0..x.rows() {
        for c in 0..x.cols() {
            widened.set(r, c, x[(r, c)]);
        }
    }
    widened
}

fn widen_claim_x_with_zero_col(claim: &mut neo_fold_clean::CeClaim) {
    claim.X = widen_x_with_zero_col(&claim.X);
    claim.m_in += D;
}

fn pi_ccs_config<'a>(prep: &'a neo_fold_clean::Preprocessing) -> PiCcsVerifierConfig<'a> {
    prep.nifs_v_circuit_config()
        .expect("NIFS.V circuit configuration")
        .pi_ccs
}

fn emit_verifier(f: &Fixture) -> Result<R1csBuilder, neo_fold_clean::paper::nifs::circuit::Error> {
    Ok(emit_verifier_outputs(f)?.0)
}

fn emit_verifier_outputs(
    f: &Fixture,
) -> Result<(R1csBuilder, NifsVOutputs), neo_fold_clean::paper::nifs::circuit::Error> {
    emit_verifier_with_running_outputs(f, &f.running.claims, f.running.parent_authority.as_ref(), &f.children)
}

fn emit_verifier_with_running(
    f: &Fixture,
    running: &[CeClaim],
    running_parent_authority: Option<&CeClaim>,
    children: &[CeClaim],
) -> Result<R1csBuilder, neo_fold_clean::paper::nifs::circuit::Error> {
    Ok(emit_verifier_with_running_outputs(f, running, running_parent_authority, children)?.0)
}

fn emit_verifier_with_running_outputs(
    f: &Fixture,
    running: &[CeClaim],
    running_parent_authority: Option<&CeClaim>,
    children: &[CeClaim],
) -> Result<(R1csBuilder, NifsVOutputs), neo_fold_clean::paper::nifs::circuit::Error> {
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
            running,
            running_parent_authority,
            pi_ccs: &f.proof.pi_ccs,
            combined: &f.combined,
            children,
        },
    )?;
    Ok((builder, outputs))
}

#[test]
fn nifs_v_accepts_native_proof() {
    let fixture = build_fixture();
    // Sanity: native verifier must accept the proof we built. If this
    // fails, the fixture itself is wrong, not the circuit.
    {
        let mut native_tr = Transcript::session();
        let _next_running_claims = neo_fold_clean::paper::nifs::verify(
            &mut native_tr,
            &fixture.prep.params,
            fixture.prep.structure(),
            fixture.prep.optimized_cache(),
            fixture.prep.mix_rhos_commits(),
            fixture.prep.combine_b_pows(),
            &fixture.fresh_claims,
            &fixture.running,
            &fixture.proof,
        )
        .expect("native NIFS.V must accept its own proof");
    }
    let builder = emit_verifier(&fixture).expect("emit verifier");
    let first_bad_row = builder.first_unsatisfied_row();
    let containing_families = first_bad_row.map(|row| {
        builder
            .row_family_ranges()
            .iter()
            .filter(|range| range.row_start <= row && row < range.row_end)
            .copied()
            .collect::<Vec<_>>()
    });
    assert!(
        first_bad_row.is_none(),
        "native nifs::prove proof must satisfy NIFS.V circuit; first bad row {first_bad_row:?}; families {containing_families:?}"
    );
}

fn make_oversized_fresh_claims(fixture: &Fixture) -> Vec<CcsClaim> {
    vec![fixture.fresh_claims[0].clone(); fixture.prep.params.max_fresh_count() + 1]
}

fn pad_pi_ccs_outputs_for_current_fresh_len(fixture: &mut Fixture) {
    let target = fixture.fresh_claims.len() + fixture.running.claims.len();
    let template = fixture.proof.pi_ccs.outputs[0].clone();
    while fixture.proof.pi_ccs.outputs.len() < target {
        fixture.proof.pi_ccs.outputs.push(template.clone());
    }
    fixture.proof.pi_ccs.outputs.truncate(target);
}

#[test]
fn nifs_v_rejects_fresh_count_above_rlc_guard_native() {
    let mut fixture = build_fixture();
    fixture.fresh_claims = make_oversized_fresh_claims(&fixture);
    pad_pi_ccs_outputs_for_current_fresh_len(&mut fixture);

    let mut tr = Transcript::session();
    let err = neo_fold_clean::paper::nifs::verify(
        &mut tr,
        &fixture.prep.params,
        fixture.prep.structure(),
        fixture.prep.optimized_cache(),
        fixture.prep.mix_rhos_commits(),
        fixture.prep.combine_b_pows(),
        &fixture.fresh_claims,
        &fixture.running,
        &fixture.proof,
    )
    .expect_err("NIFS.V must reject K above the SuperNeo RLC guard");
    assert!(
        err.to_string().contains("max_fresh_count"),
        "expected max_fresh_count shape rejection, got {err}"
    );
}

#[test]
fn nifs_v_circuit_rejects_fresh_count_above_rlc_guard() {
    let mut fixture = build_fixture();
    fixture.fresh_claims = make_oversized_fresh_claims(&fixture);
    pad_pi_ccs_outputs_for_current_fresh_len(&mut fixture);

    let err = emit_verifier(&fixture)
        .err()
        .expect("NIFS.V circuit synthesis must reject K above the SuperNeo RLC guard");
    assert!(
        err.to_string().contains("fresh source count"),
        "expected max_fresh_count shape rejection, got {err}"
    );
}

#[test]
fn nifs_v_rejects_tampered_fe_round() {
    let mut fixture = build_fixture();
    fixture.proof.pi_ccs.sumcheck.sumcheck_rounds[0][0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered FE round must be rejected");
}

#[test]
fn nifs_v_rejects_parent_authority_when_running_is_empty() {
    let mut fixture = build_fixture();
    // The selected verifier checks Π_CCS output count
    // against `fresh.len() + running.len()`. Truncate the proof outputs to
    // the empty-running width so this test reaches the intended
    // parent-authority guard rather than failing early on output count.
    let output_count_for_empty_running = fixture.fresh_claims.len();
    fixture
        .proof
        .pi_ccs
        .outputs
        .truncate(output_count_for_empty_running);
    let parent = fixture
        .running
        .parent_authority
        .as_ref()
        .expect("fixture has running parent authority");

    let err = emit_verifier_with_running(
        &fixture,
        &[],
        Some(parent),
        &fixture.children[..output_count_for_empty_running],
    )
    .err()
    .expect("parent authority with empty running must fail closed");
    assert!(
        err.to_string()
            .contains("empty running accumulator carries a parent authority"),
        "expected empty-running parent-authority shape error, got {err}"
    );
}

#[test]
fn nifs_v_rejects_nonempty_running_without_parent_authority() {
    let fixture = build_fixture();

    let err = emit_verifier_with_running(&fixture, &fixture.running.claims, None, &fixture.children)
        .err()
        .expect("non-empty running without parent authority must fail closed");
    assert!(
        err.to_string()
            .contains("nonempty running accumulator is missing its parent authority"),
        "expected missing-parent-authority shape error, got {err}"
    );
}

#[test]
fn nifs_v_rejects_tampered_running_parent_authority() {
    let mut fixture = build_fixture();
    fixture
        .running
        .parent_authority
        .as_mut()
        .expect("running parent authority")
        .c
        .data[0] += F::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered running Pi_RLC parent authority must be rejected"
    );
}

#[test]
fn nifs_v_rejects_proof_generated_with_inconsistent_running_parent_authority() {
    // This is stronger than post-hoc tampering: construct the next NIFS proof
    // using a running parent authority that does not decompose to the running
    // children. If NIFS.V treats the parent sidecar as HyperNova authority, it
    // must reject the malformed input rather than merely binding it into the
    // Fiat-Shamir transcript self-consistently.
    let mut fixture = build_fixture();
    let parent = fixture
        .running
        .parent_authority
        .as_mut()
        .expect("running parent authority");
    parent.X.set(0, 0, parent.X[(0, 0)] + F::ONE);

    let r1cs = three_term_addition();
    let second = direct_ccs::build_instance(&fixture.prep, &r1cs, &assignment(0, 1)).expect("second instance");
    fixture.fresh_claims = vec![second.claim.clone()];
    let mut tr = Transcript::session();
    let (next_running, proof) = neo_fold_clean::paper::nifs::prove(
        &mut tr,
        &fixture.prep.params,
        fixture.prep.structure(),
        fixture.prep.optimized_cache(),
        &fixture.prep.log,
        None,
        fixture.prep.mix_rhos_commits(),
        fixture.prep.combine_b_pows(),
        vec![second],
        &fixture.running,
    )
    .expect("NIFS.P over malformed running");
    fixture.combined = proof.pi_rlc.combined.clone();
    fixture.children = next_running.claims.clone();
    fixture.proof = proof;

    let mut native_tr = Transcript::session();
    let native = neo_fold_clean::paper::nifs::verify(
        &mut native_tr,
        &fixture.prep.params,
        fixture.prep.structure(),
        fixture.prep.optimized_cache(),
        fixture.prep.mix_rhos_commits(),
        fixture.prep.combine_b_pows(),
        &fixture.fresh_claims,
        &fixture.running,
        &fixture.proof,
    );
    assert!(
        native.is_err(),
        "native NIFS.V accepted a proof generated from inconsistent running parent authority"
    );

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "in-circuit NIFS.V accepted a proof generated from inconsistent running parent authority"
    );
}

#[test]
fn nifs_v_rejects_tampered_running_parent_authority_r_c1_limb() {
    // HyperNova's `U_i` includes the evaluation point carried by the CE
    // accumulator. Probe the extension-field c1 limb so a c0-only digest or
    // allocation path cannot satisfy this test.
    let mut fixture = build_fixture();
    let parent = fixture
        .running
        .parent_authority
        .as_mut()
        .expect("running parent authority");
    assert!(!parent.r.is_empty(), "fixture must expose parent r");
    let original = parent.r[0];
    parent.r[0] = original + k_c1_one();
    assert_eq!(
        parent.r[0].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave c0 unchanged"
    );
    assert_ne!(
        parent.r[0].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change c1"
    );

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered running parent authority r c1 limb must be rejected"
    );
}

#[test]
fn nifs_v_rejects_tampered_running_parent_authority_y_ring_c1_limb() {
    // Probe a non-ct y_ring lane on the running parent authority. This is
    // not covered by the ct equality row, so rejection must come from the
    // full accumulator authority binding consumed by NIFS.V.
    let mut fixture = build_fixture();
    let parent = fixture
        .running
        .parent_authority
        .as_mut()
        .expect("running parent authority");
    assert!(
        !parent.y_ring.is_empty() && parent.y_ring[0].len() > 1,
        "fixture must expose a non-ct parent y_ring lane"
    );
    let original = parent.y_ring[0][1];
    parent.y_ring[0][1] = original + k_c1_one();
    assert_eq!(
        parent.y_ring[0][1].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave c0 unchanged"
    );
    assert_ne!(
        parent.y_ring[0][1].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change c1"
    );

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered running parent authority y_ring c1 limb must be rejected"
    );
}

#[test]
fn nifs_v_rejects_tampered_running_parent_authority_ct_c1_limb() {
    // `ct` is denormalized from lane 0 of y_ring, but it is also carried as
    // a public CE field inside U_i. A c1-only ct tamper should fail before a
    // prover can use a stale or prefix-only accumulator handle as authority.
    let mut fixture = build_fixture();
    let parent = fixture
        .running
        .parent_authority
        .as_mut()
        .expect("running parent authority");
    assert!(!parent.ct.is_empty(), "fixture must expose parent ct");
    let original = parent.ct[0];
    parent.ct[0] = original + k_c1_one();
    assert_eq!(
        parent.ct[0].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave c0 unchanged"
    );
    assert_ne!(
        parent.ct[0].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change c1"
    );

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered running parent authority ct c1 limb must be rejected"
    );
}

#[test]
fn nifs_v_rejects_tampered_running_parent_authority_fold_digest() {
    // The running parent authority's fold_digest is part of the carried
    // accumulator authority representation. If omitted from the running handle, a
    // prover could relabel this transcript boundary while leaving algebraic
    // c/X/y rows unchanged.
    let mut fixture = build_fixture();
    let parent = fixture
        .running
        .parent_authority
        .as_mut()
        .expect("running parent authority");
    parent.fold_digest[0] ^= 1;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered running parent authority fold_digest must be rejected"
    );
}

#[test]
fn nifs_v_rejects_tampered_running_child_r_c1_limb() {
    let mut fixture = build_fixture();
    let running = fixture.running.claims.get_mut(0).expect("running CE claim");
    assert!(!running.r.is_empty(), "fixture must expose running r");
    let original = running.r[0];
    running.r[0] = original + k_c1_one();
    assert_eq!(running.r[0].as_coeffs()[0], original.as_coeffs()[0]);
    assert_ne!(running.r[0].as_coeffs()[1], original.as_coeffs()[1]);

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "NIFS.V accepted a c1-only running child r tamper"
    );
}

#[test]
fn nifs_v_rejects_tampered_running_child_y_ring_c1_limb() {
    let mut fixture = build_fixture();
    let running = fixture.running.claims.get_mut(0).expect("running CE claim");
    assert!(
        !running.y_ring.is_empty() && running.y_ring[0].len() > 1,
        "fixture must expose a non-ct running y_ring lane"
    );
    let original = running.y_ring[0][1];
    running.y_ring[0][1] = original + k_c1_one();
    assert_eq!(running.y_ring[0][1].as_coeffs()[0], original.as_coeffs()[0]);
    assert_ne!(running.y_ring[0][1].as_coeffs()[1], original.as_coeffs()[1]);

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "NIFS.V accepted a c1-only running child y_ring tamper"
    );
}

#[test]
fn nifs_v_rejects_tampered_running_child_ct_c1_limb() {
    let mut fixture = build_fixture();
    let running = fixture.running.claims.get_mut(0).expect("running CE claim");
    assert!(!running.ct.is_empty(), "fixture must expose running ct");
    let original = running.ct[0];
    running.ct[0] = original + k_c1_one();
    assert_eq!(running.ct[0].as_coeffs()[0], original.as_coeffs()[0]);
    assert_ne!(running.ct[0].as_coeffs()[1], original.as_coeffs()[1]);

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "NIFS.V accepted a c1-only running child ct tamper"
    );
}

#[test]
fn nifs_v_rejects_tampered_running_child_fold_digest() {
    let mut fixture = build_fixture();
    fixture.running.claims[0].fold_digest[0] ^= 0x80;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "NIFS.V accepted a running child fold_digest tamper"
    );
}

#[test]
fn nifs_v_rejects_tampered_pi_ccs_fresh_output_y_ring_non_ct_lane() {
    // A standalone Π_CCS verifier only consumes fresh outputs' constant
    // terms in its FE terminal identity. NIFS.V must still reject a forged
    // non-ct lane because Π_RLC immediately folds the full output y_ring
    // into the DEC parent.
    let mut fixture = build_fixture();
    assert!(
        !fixture.proof.pi_ccs.outputs[0].y_ring.is_empty() && fixture.proof.pi_ccs.outputs[0].y_ring[0].len() > 1,
        "fixture must have a non-ct y_ring lane"
    );
    fixture.proof.pi_ccs.outputs[0].y_ring[0][1] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered Π_CCS fresh-output non-ct y_ring lane must be rejected by NIFS.V"
    );
}

#[test]
fn nifs_v_rejects_coherent_fresh_output_non_ct_y_ring_relabel() {
    // A single stale-field tamper is easy to catch. This is the adversarial
    // version: mutate a fresh Π_CCS output lane that standalone Π_CCS does
    // not consume, then recompute the Π_RLC parent and a public Π_DEC
    // decomposition coherently under the same transcript-derived ρ values.
    let mut fixture = build_fixture();
    assert!(
        !fixture.proof.pi_ccs.outputs[0].y_ring.is_empty() && fixture.proof.pi_ccs.outputs[0].y_ring[0].len() > 1,
        "fixture must have a fresh output non-ct y_ring lane"
    );

    fixture.proof.pi_ccs.outputs[0].y_ring[0][1] += K::ONE;

    let mut tr = Poseidon2Transcript::new(SESSION_LABEL);
    let ccs_ok = neo_fold_clean::engine::optimized::verify_pi_ccs(
        &mut tr,
        &fixture.prep.params,
        fixture.prep.structure(),
        fixture.prep.optimized_cache(),
        &fixture.fresh_claims,
        &fixture.running,
        &fixture.proof.pi_ccs.outputs,
        &fixture.proof.pi_ccs.sumcheck,
    )
    .expect("Π_CCS verifier should run");
    assert!(
        ccs_ok,
        "fixture no longer probes the intended gap: Π_CCS alone rejected the non-ct fresh-output relabel"
    );

    let rhos = neo_fold_clean::engine::optimized::sample_rho_n(
        &mut tr,
        &fixture.prep.params,
        fixture.proof.pi_ccs.outputs.len(),
    )
    .expect("sample rho");
    let combined = neo_reductions::api::rlc_public(
        fixture.prep.structure(),
        fixture.prep.params.inner(),
        &rhos,
        &fixture.proof.pi_ccs.outputs,
        |rho_mats, commitments| (fixture.prep.mix_rhos_commits())(rho_mats, commitments),
        D.next_power_of_two().trailing_zeros() as usize,
    )
    .expect("public RLC recompute");

    fixture.proof.pi_rlc.combined = combined.clone();
    fixture.combined = combined.clone();
    fixture.proof.pi_dec.children = trivial_public_dec_children(&fixture.prep, &combined);
    fixture.children = fixture.proof.pi_dec.children.clone();

    let mut native_tr = Transcript::session();
    let native = neo_fold_clean::paper::nifs::verify(
        &mut native_tr,
        &fixture.prep.params,
        fixture.prep.structure(),
        fixture.prep.optimized_cache(),
        fixture.prep.mix_rhos_commits(),
        fixture.prep.combine_b_pows(),
        &fixture.fresh_claims,
        &fixture.running,
        &fixture.proof,
    );
    assert!(
        native.is_err(),
        "native NIFS.V accepted a coherent non-ct y_ring relabel"
    );

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "in-circuit NIFS.V accepted a coherent non-ct y_ring relabel"
    );
}

#[test]
fn nifs_v_rejects_tampered_pi_ccs_fresh_output_y_ring_padding_lane() {
    // Native SuperNeo computes y_ring as D real ring coefficients padded to
    // d_pad = next_power_of_two(D). The padding lanes are not semantic
    // degrees of freedom; if this wire can change without breaking NIFS.V,
    // an intermediate CE claim is carrying non-canonical data through the
    // verifier circuit.
    let mut fixture = build_fixture();
    let d_pad = D.next_power_of_two();
    assert!(d_pad > D, "fixture must have padded y_ring lanes");
    assert!(
        !fixture.proof.pi_ccs.outputs[0].y_ring.is_empty() && fixture.proof.pi_ccs.outputs[0].y_ring[0].len() == d_pad,
        "fixture must expose full padded y_ring rows"
    );
    fixture.proof.pi_ccs.outputs[0].y_ring[0][D] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered Π_CCS fresh-output y_ring padding lane must be rejected by NIFS.V"
    );
}

#[test]
fn nifs_v_rejects_tampered_combined_y_ring_non_ct_c1_limb() {
    // Probe the Π_RLC parent y_ring away from lane 0, so the rejection
    // cannot be explained only by `ct == constant_term(y_ring)`. NIFS.V
    // must fold and carry both K limbs of the full parent y_ring row.
    let mut fixture = build_fixture();
    assert!(
        !fixture.combined.y_ring.is_empty() && fixture.combined.y_ring[0].len() > 1,
        "fixture must have a non-ct combined y_ring lane"
    );
    let original = fixture.combined.y_ring[0][1];
    fixture.combined.y_ring[0][1] = original + k_c1_one();
    assert_eq!(
        fixture.combined.y_ring[0][1].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave c0 unchanged"
    );
    assert_ne!(
        fixture.combined.y_ring[0][1].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change c1"
    );

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "NIFS.V accepted a c1-only combined.y_ring non-ct-lane tamper"
    );
}

#[test]
fn nifs_v_rejects_tampered_combined_r_point() {
    // Π_DEC's parent evaluation point must be the Π_CCS verifier's
    // r_prime. If parent.r is just allocated and never bound to the
    // sumcheck point, a prover can move the decomposition to a different
    // evaluation point while keeping the rest of the message self-consistent.
    let mut fixture = build_fixture();
    assert!(!fixture.combined.r.is_empty(), "fixture must have a DEC parent r");
    fixture.combined.r[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered Π_DEC parent r must be rejected");
}

#[test]
fn nifs_v_rejects_tampered_combined_r_c1_limb() {
    // The Π_DEC parent point must equal the Π_CCS output point in both
    // extension limbs. This catches a half-bound KVar point equality.
    let mut fixture = build_fixture();
    assert!(!fixture.combined.r.is_empty(), "fixture must have a DEC parent r");
    let original = fixture.combined.r[0];
    fixture.combined.r[0] = original + k_c1_one();
    assert_eq!(
        fixture.combined.r[0].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave c0 unchanged"
    );
    assert_ne!(
        fixture.combined.r[0].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change c1"
    );

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "NIFS.V accepted a c1-only combined.r tamper");
}

#[test]
fn nifs_v_rejects_tampered_child_commitment_lane() {
    // Π_DEC children are the next running accumulator. Their commitments
    // must b-ary recombine to the Π_RLC parent commitment lane-by-lane.
    let mut fixture = build_fixture();
    assert!(
        !fixture.children.is_empty() && !fixture.children[0].c.data.is_empty(),
        "fixture must have child commitment lanes"
    );
    fixture.children[0].c.data[0] += F::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered Π_DEC child commitment lane must be rejected"
    );
}

#[test]
fn nifs_v_rejects_tampered_child_x_active_lane() {
    // Active X columns are the public-input projection carried by the CE
    // claim. Mutating one active child lane must break Π_DEC's X
    // recomposition, not be hidden in shape-only metadata.
    let mut fixture = build_fixture();
    assert!(
        !fixture.children.is_empty() && fixture.children[0].X.rows() > 0 && fixture.children[0].X.cols() > 0,
        "fixture must have a non-empty child X"
    );
    let old = fixture.children[0].X[(0, 0)];
    fixture.children[0].X.set(0, 0, old + F::ONE);

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered Π_DEC child X lane must be rejected");
}

#[test]
fn nifs_v_rejects_recomposition_preserving_out_of_alphabet_child_x() {
    // SuperNeo §7.5 outputs CE(b) children. Checking only
    // `parent.X = Σ b^i child_i.X` is not enough: child X values must still
    // be compatible with low-norm child witnesses. This tamper preserves the
    // b-ary X recomposition (`child0 += 2b`, `child1 -= 2`) while moving an
    // active child public-input lane outside the balanced CE(b) alphabet.
    let mut fixture = build_fixture();
    assert!(
        fixture.children.len() >= 2,
        "fixture must expose at least two DEC children"
    );
    assert!(
        fixture.children[0].X.rows() > 0 && fixture.children[0].X.cols() > 0,
        "fixture must have a non-empty child X"
    );

    let b = F::from_u64(fixture.prep.params.b() as u64);
    let child0 = fixture.children[0].X[(0, 0)] + b + b;
    let child1 = fixture.children[1].X[(0, 0)] - F::from_u64(2);
    fixture.children[0].X.set(0, 0, child0);
    fixture.children[1].X.set(0, 0, child1);

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "NIFS.V circuit accepted recomposition-preserving active child X outside CE(b)"
    );
}

#[test]
fn nifs_v_rejects_noncanonical_x_width_in_dec_child() {
    let mut fixture = build_fixture();
    append_noncanonical_x_column(&mut fixture.children[0]);
    assert!(
        emit_verifier(&fixture).is_err(),
        "NIFS.V circuit accepted a Π_DEC child wider than the SuperNeo coefficient embedding"
    );
}

#[test]
fn nifs_v_rejects_noncanonical_x_width_in_running() {
    let mut fixture = build_fixture();
    append_noncanonical_x_column(&mut fixture.running.claims[0]);
    assert!(
        emit_verifier(&fixture).is_err(),
        "NIFS.V circuit accepted a running claim wider than the SuperNeo coefficient embedding"
    );
}

#[test]
fn nifs_v_rejects_tampered_child_y_ring_lane() {
    // y_ring is the SuperNeo CE evaluation output. A child y_ring lane
    // must participate in Π_DEC recomposition; otherwise the next running
    // accumulator can carry forged evaluations.
    let mut fixture = build_fixture();
    assert!(
        !fixture.children.is_empty()
            && !fixture.children[0].y_ring.is_empty()
            && !fixture.children[0].y_ring[0].is_empty(),
        "fixture must have child y_ring lanes"
    );
    fixture.children[0].y_ring[0][0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered Π_DEC child y_ring lane must be rejected"
    );
}

#[test]
fn nifs_v_rejects_canceling_child_y_ring_padding_lanes() {
    // A single child y_ring tamper is easy: Π_DEC recomposition catches it.
    // The sneaky version changes two child padding lanes so their b-ary
    // weighted sum is still zero and the parent recomposition is unchanged.
    // Children are the next running accumulator, so their padded lanes must
    // be canonical too, not merely parent-canceling.
    let mut fixture = build_fixture();
    let d_pad = D.next_power_of_two();
    assert!(d_pad > D, "fixture must have padded y_ring lanes");
    assert!(
        fixture.children.len() >= 2,
        "fixture must have at least two DEC children"
    );
    assert!(
        !fixture.children[0].y_ring.is_empty() && fixture.children[0].y_ring[0].len() == d_pad,
        "fixture must expose full padded child y_ring rows"
    );

    let b_inv = K::from_u64(fixture.prep.params.b() as u64).inverse();
    fixture.children[0].y_ring[0][D] += K::ONE;
    fixture.children[1].y_ring[0][D] -= b_inv;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "canceling nonzero Π_DEC child y_ring padding lanes must be rejected"
    );
}

#[test]
fn nifs_v_rejects_tampered_combined_ct_lane() {
    // The Π_RLC parent is the authority that Π_DEC decomposes into
    // children. Its denormalized ct must be tied back to parent.y_ring
    // before DEC uses it; otherwise the parent can carry a shadow scalar
    // view even while y_ring recomposition remains valid.
    let mut fixture = build_fixture();
    assert!(!fixture.combined.ct.is_empty(), "fixture must have combined ct lanes");
    fixture.combined.ct[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered Π_RLC parent ct lane must be rejected"
    );
}

#[test]
fn nifs_v_rejects_tampered_child_ct_lane() {
    // ct is denormalized from y_ring's constant term and is consumed by
    // scalar folding. Strict DEC must bind it back to y_ring instead of
    // letting it become a shadow authoritative field.
    let mut fixture = build_fixture();
    assert!(
        !fixture.children.is_empty() && !fixture.children[0].ct.is_empty(),
        "fixture must have child ct lanes"
    );
    fixture.children[0].ct[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered Π_DEC child ct lane must be rejected");
}

#[test]
fn nifs_v_rejects_tampered_child_ct_c1_limb() {
    // Child ct is denormalized from child.y_ring. Strict DEC must bind both
    // K limbs, not just the scalar-looking c0 component.
    let mut fixture = build_fixture();
    assert!(
        !fixture.children.is_empty() && !fixture.children[0].ct.is_empty(),
        "fixture must have child ct lanes"
    );
    let original = fixture.children[0].ct[0];
    fixture.children[0].ct[0] = original + k_c1_one();
    assert_eq!(
        fixture.children[0].ct[0].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave c0 unchanged"
    );
    assert_ne!(
        fixture.children[0].ct[0].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change c1"
    );

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "NIFS.V accepted a c1-only child.ct tamper");
}

#[test]
fn nifs_v_rejects_tampered_child_r_point() {
    // Parent and all children share the same CE evaluation point r in
    // Π_DEC. A child-specific r would let the decomposition output a
    // running accumulator at a different point.
    let mut fixture = build_fixture();
    assert!(
        !fixture.children.is_empty() && !fixture.children[0].r.is_empty(),
        "fixture must have child r lanes"
    );
    fixture.children[0].r[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered Π_DEC child r must be rejected");
}

#[test]
fn nifs_v_rejects_extra_self_consistent_y_ring_row() {
    // The selected circuit owns the identity row plus exactly `structure.t()`
    // application-matrix evaluation rows. A proof must not smuggle an extra y_ring/ct row
    // through Π_DEC, even if the extra row is self-consistent and zero, since
    // no Π_CCS/Π_RLC matrix owns it. This is a structural error, not an
    // unsatisfied-row case.
    let mut fixture = build_fixture();
    let d_pad = D.next_power_of_two();
    let extra_row = vec![K::ZERO; d_pad];
    fixture.combined.y_ring.push(extra_row.clone());
    fixture.combined.ct.push(K::ZERO);
    for child in &mut fixture.children {
        child.y_ring.push(extra_row.clone());
        child.ct.push(K::ZERO);
    }

    let err = emit_verifier(&fixture)
        .err()
        .expect("extra y_ring row must fail NIFS.V synthesis");
    assert!(
        err.to_string().contains("y_ring count"),
        "expected y_ring outer length shape error, got {err}"
    );
}

#[test]
fn native_nifs_verify_rejects_extra_self_consistent_y_ring_row() {
    // Native NIFS.V must fail closed on the same structural boundary as the
    // circuit. The extra row is self-consistent across Π_CCS output,
    // Π_RLC parent, and Π_DEC children, so a `< s.t()`-only shape check plus
    // decomposition over `parent.y_ring.len()` would accept it even though no
    // CCS matrix owns that row.
    let mut fixture = build_fixture();
    let d_pad = D.next_power_of_two();
    let extra_row = vec![K::ZERO; d_pad];
    fixture.proof.pi_ccs.outputs[0]
        .y_ring
        .push(extra_row.clone());
    fixture.proof.pi_ccs.outputs[0].ct.push(K::ZERO);
    fixture.proof.pi_rlc.combined.y_ring.push(extra_row.clone());
    fixture.proof.pi_rlc.combined.ct.push(K::ZERO);
    for child in &mut fixture.proof.pi_dec.children {
        child.y_ring.push(extra_row.clone());
        child.ct.push(K::ZERO);
    }

    let mut tr = Transcript::session();
    let result = neo_fold_clean::paper::nifs::verify(
        &mut tr,
        &fixture.prep.params,
        fixture.prep.structure(),
        fixture.prep.optimized_cache(),
        fixture.prep.mix_rhos_commits(),
        fixture.prep.combine_b_pows(),
        &fixture.fresh_claims,
        &fixture.running,
        &fixture.proof,
    );
    assert!(
        result.is_err(),
        "native NIFS.V accepted an extra self-consistent y_ring/ct row"
    );
}

#[test]
fn nifs_v_rejects_extra_self_consistent_running_y_ring_row() {
    // Same smuggling attack, but on the running CE input U_i rather than
    // the proof outputs. HyperNova's recursive link hashes the full U_i;
    // a prefix-only ME-input digest would ignore this extra row, while the
    // circuit must reject because the CCS structure owns exactly t rows.
    let mut fixture = build_fixture();
    let d_pad = D.next_power_of_two();
    let extra_row = vec![K::ZERO; d_pad];
    let running = fixture
        .running
        .claims
        .get_mut(0)
        .expect("fixture must carry a running CE input");
    running.y_ring.push(extra_row);
    running.ct.push(K::ZERO);

    let err = emit_verifier(&fixture)
        .err()
        .expect("extra running y_ring row must fail NIFS.V synthesis");
    assert!(
        err.to_string()
            .contains("running[0] does not have the selected CE shape"),
        "expected running y_ring outer length shape error, got {err}"
    );
}

#[test]
fn native_nifs_verify_rejects_extra_self_consistent_running_y_ring_row() {
    // Native parity for the running-input tail attack above. Even if an
    // older standalone CE shape helper permits y_ring.len() > t, the real
    // NIFS verifier must not accept a running accumulator with extra
    // non-structure-owned rows.
    let mut fixture = build_fixture();
    let d_pad = D.next_power_of_two();
    let extra_row = vec![K::ZERO; d_pad];
    let running = fixture
        .running
        .claims
        .get_mut(0)
        .expect("fixture must carry a running CE input");
    running.y_ring.push(extra_row);
    running.ct.push(K::ZERO);

    let mut tr = Transcript::session();
    let result = neo_fold_clean::paper::nifs::verify(
        &mut tr,
        &fixture.prep.params,
        fixture.prep.structure(),
        fixture.prep.optimized_cache(),
        fixture.prep.mix_rhos_commits(),
        fixture.prep.combine_b_pows(),
        &fixture.fresh_claims,
        &fixture.running,
        &fixture.proof,
    );
    assert!(
        result.is_err(),
        "native NIFS.V accepted an extra self-consistent running y_ring/ct row"
    );
}

#[test]
fn nifs_v_rejects_child_m_in_drift() {
    // Native Π_DEC verifies child.m_in == parent.m_in. The circuit has to
    // fail closed on the same metadata, because m_in is part of the public
    // CE claim shape and accumulator digest even though it is not a field
    // wire.
    let mut fixture = build_fixture();
    assert!(!fixture.children.is_empty(), "fixture must have children");
    fixture.children[0].m_in += 1;

    let err = emit_verifier(&fixture)
        .err()
        .expect("child m_in drift must fail synthesis");
    assert!(
        err.to_string().contains("child m_in"),
        "expected child m_in shape error, got {err}"
    );
}

#[test]
fn nifs_v_rejects_parent_m_in_drift() {
    // Parent.m_in must agree with parent.X.cols just like native
    // verify_dec_public. Otherwise a proof could alter CE shape metadata
    // without changing allocated X wires.
    let mut fixture = build_fixture();
    fixture.combined.m_in += D;

    let err = emit_verifier(&fixture)
        .err()
        .expect("parent m_in drift must fail synthesis");
    assert!(
        err.to_string().contains("parent X cols vs m_in"),
        "expected parent X/m_in shape error, got {err}"
    );
}

#[test]
fn nifs_v_rejects_combined_parent_m_in_wider_than_rlc_outputs() {
    // Make the Π_DEC parent and every child self-consistently wider than
    // the Π_CCS outputs that Π_RLC actually folds. Without a cross-boundary
    // parent-shape check, DEC can accept the widened shape while RLC only
    // constrains the old output width.
    let mut fixture = build_fixture();
    widen_claim_x_with_zero_col(&mut fixture.combined);
    for child in &mut fixture.children {
        widen_claim_x_with_zero_col(child);
    }

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "widened DEC parent m_in must be rejected by in-circuit Π_RLC parent-shape rows"
    );
}

#[test]
fn nifs_v_rejects_combined_parent_commitment_kappa_wider_than_rlc_outputs() {
    // Same attack shape for commitments: append a zero Ajtai column to the
    // Π_DEC parent and all children. Π_DEC recomposition would preserve
    // that extra zero column, but Π_RLC's commitment fold is defined over
    // the κ coming from Π_CCS outputs and must reject the larger DEC parent.
    let mut fixture = build_fixture();
    append_zero_commitment_column(&mut fixture.combined);
    for child in &mut fixture.children {
        append_zero_commitment_column(child);
    }

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "widened DEC parent commitment kappa must be rejected by in-circuit Π_RLC parent-shape rows"
    );
}

#[test]
fn nifs_v_rejects_tampered_pi_ccs_output_fold_digest() {
    // Π_CCS outputs carry the catch-up transcript digest. If this field is
    // not pinned to `proof.header_digest`, a later compact accumulator handle
    // could treat a self-consistent but fake fold_digest as authority.
    let mut fixture = build_fixture();
    fixture.proof.pi_ccs.outputs[0].fold_digest[0] ^= 1;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered Π_CCS output fold_digest must be rejected"
    );
}

#[test]
fn nifs_v_rejects_tampered_combined_fold_digest() {
    // Π_RLC carries the fold_digest through unchanged from the Π_CCS
    // outputs. It is not an RLC-linear field, so the circuit must bind it
    // with equality rather than ignore it.
    let mut fixture = build_fixture();
    fixture.combined.fold_digest[0] ^= 1;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered Π_RLC parent fold_digest must be rejected"
    );
}

#[test]
fn nifs_v_rejects_tampered_child_fold_digest() {
    // Π_DEC children become the next running accumulator. They must not be
    // able to mint a new fold_digest while still decomposing the parent on
    // c/X/y_ring.
    let mut fixture = build_fixture();
    fixture.children[0].fold_digest[0] ^= 1;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered Π_DEC child fold_digest must be rejected"
    );
}

#[test]
fn nifs_v_rejects_coherent_forged_fold_digest_chain() {
    // Rewrite every carried fold_digest to the same forged value. That keeps
    // the Π_CCS output -> Π_RLC parent ->
    // Π_DEC child equality chain self-consistent. The only row that should
    // reject is the verifier-driven transcript digest bound to the outputs.
    let mut fixture = build_fixture();
    let mut forged_fold_digest = fixture.proof.pi_ccs.outputs[0].fold_digest;
    forged_fold_digest[0] ^= 1;

    for output in &mut fixture.proof.pi_ccs.outputs {
        output.fold_digest = forged_fold_digest;
    }
    fixture.combined.fold_digest = forged_fold_digest;
    for child in &mut fixture.children {
        child.fold_digest = forged_fold_digest;
    }

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "NIFS.V accepted a self-consistent forged fold_digest chain; the header digest must be recomputed"
    );
}

#[test]
fn native_nifs_verify_rejects_tampered_dec_child_fold_digest() {
    let mut fixture = build_fixture();
    fixture.proof.pi_dec.children[0].fold_digest[0] ^= 1;

    let mut tr = Transcript::session();
    let result = neo_fold_clean::paper::nifs::verify(
        &mut tr,
        &fixture.prep.params,
        fixture.prep.structure(),
        fixture.prep.optimized_cache(),
        fixture.prep.mix_rhos_commits(),
        fixture.prep.combine_b_pows(),
        &fixture.fresh_claims,
        &fixture.running,
        &fixture.proof,
    );
    assert!(result.is_err(), "native NIFS.V must reject child fold_digest tamper");
}

#[test]
fn native_nifs_verify_rejects_noncanonical_x_width_in_dec_child() {
    let mut fixture = build_fixture();
    append_noncanonical_x_column(&mut fixture.proof.pi_dec.children[0]);

    let mut tr = Transcript::session();
    let result = neo_fold_clean::paper::nifs::verify(
        &mut tr,
        &fixture.prep.params,
        fixture.prep.structure(),
        fixture.prep.optimized_cache(),
        fixture.prep.mix_rhos_commits(),
        fixture.prep.combine_b_pows(),
        &fixture.fresh_claims,
        &fixture.running,
        &fixture.proof,
    );
    assert!(
        result.is_err(),
        "native nifs::verify accepted a Π_DEC child wider than the SuperNeo coefficient embedding"
    );
}

#[test]
fn native_nifs_verify_rejects_recomposition_preserving_out_of_alphabet_child_x() {
    let mut fixture = build_fixture();
    assert!(
        fixture.proof.pi_dec.children.len() >= 2,
        "fixture must expose at least two DEC children"
    );

    let b = F::from_u64(fixture.prep.params.b() as u64);
    let child0 = fixture.proof.pi_dec.children[0].X[(0, 0)] + b + b;
    let child1 = fixture.proof.pi_dec.children[1].X[(0, 0)] - F::from_u64(2);
    fixture.proof.pi_dec.children[0].X.set(0, 0, child0);
    fixture.proof.pi_dec.children[1].X.set(0, 0, child1);

    let mut tr = Transcript::session();
    let result = neo_fold_clean::paper::nifs::verify(
        &mut tr,
        &fixture.prep.params,
        fixture.prep.structure(),
        fixture.prep.optimized_cache(),
        fixture.prep.mix_rhos_commits(),
        fixture.prep.combine_b_pows(),
        &fixture.fresh_claims,
        &fixture.running,
        &fixture.proof,
    );
    assert!(
        result.is_err(),
        "native NIFS.V accepted recomposition-preserving active child X outside CE(b)"
    );
}

#[test]
fn native_nifs_verify_rejects_noncanonical_x_width_in_running() {
    let mut fixture = build_fixture();
    append_noncanonical_x_column(&mut fixture.running.claims[0]);

    let mut tr = Transcript::session();
    let result = neo_fold_clean::paper::nifs::verify(
        &mut tr,
        &fixture.prep.params,
        fixture.prep.structure(),
        fixture.prep.optimized_cache(),
        fixture.prep.mix_rhos_commits(),
        fixture.prep.combine_b_pows(),
        &fixture.fresh_claims,
        &fixture.running,
        &fixture.proof,
    );
    assert!(
        result.is_err(),
        "native nifs::verify accepted a running claim wider than the SuperNeo coefficient embedding"
    );
}
