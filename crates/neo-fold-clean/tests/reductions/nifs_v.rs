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
//! - `nifs_v_rejects_tampered_running_parent_authority_s_col`
//! - `nifs_v_accepts_tampered_running_parent_authority_y_zcol_non_authority`
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
//! - `nifs_v_rejects_tampered_pi_ccs_output_y_zcol_padding_lane`
//! - `nifs_v_rejects_tampered_combined_y_zcol_lane`
//! - `nifs_v_rejects_tampered_combined_y_zcol_c1_limb`
//! - `nifs_v_rejects_tampered_combined_y_ring_non_ct_c1_limb`
//! - `nifs_v_rejects_tampered_combined_r_point`
//! - `nifs_v_rejects_tampered_combined_r_c1_limb`
//! - `nifs_v_rejects_tampered_combined_s_col_c1_limb`
//! - `nifs_v_rejects_tampered_child_commitment_lane`
//! - `nifs_v_rejects_tampered_child_x_active_lane`
//! - `nifs_v_rejects_nonzero_inactive_x_in_dec_child`
//! - `nifs_v_rejects_nonzero_inactive_x_in_running`
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
//! - `nifs_v_rejects_incoming_running_sidecars`
//! - `nifs_v_rejects_combined_aux_openings_sidecar`
//! - `nifs_v_rejects_child_aux_openings_sidecar`
//! - `nifs_v_rejects_combined_pattern_a_sidecar`
//! - `nifs_v_rejects_child_pattern_a_sidecar`
//! - `nifs_v_rejects_tampered_child_s_col`
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
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsVConfig;
use neo_fold_clean::paper::relations::{CcsClaim, CeClaim};
use neo_math::ring::D;
use neo_math::{KExtensions, F, K};
use neo_transcript::{Poseidon2Transcript, Transcript as NeoTranscript};
use p3_field::{Field, PrimeCharacteristicRing};

const SESSION_LABEL: &[u8] = b"neo.fold.clean/session/v1";

fn k_c1_one() -> K {
    K::from_coeffs([F::ZERO, F::ONE])
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
    R1cs { a, b, c, m_in: 3 }
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
            s_col: parent.s_col.clone(),
            y_ring: vec![vec![K::ZERO; d_pad]; parent.y_ring.len()],
            ct: vec![K::ZERO; parent.ct.len()],
            aux_openings: Vec::new(),
            y_zcol: vec![K::ZERO; parent.y_zcol.len()],
            m_in: parent.m_in,
            fold_digest: parent.fold_digest,
            c_step_coords: Vec::new(),
            u_offset: 0,
            u_len: 0,
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
    claim.m_in += 1;
}

fn pi_ccs_config<'a>(prep: &'a neo_fold_clean::Preprocessing) -> SplitNcPiCcsVConfig<'a> {
    let raw_params = neo_params::NeoParams::goldilocks_auto_r1cs_ccs_with(
        prep.structure().n.max(prep.structure().m),
        neo_fold_clean::config::MIN_EFFECTIVE_LAMBDA,
        neo_fold_clean::config::EXTENSION_SAFETY_MARGIN_BITS,
    )
    .expect("raw params reconstruction");
    let dims =
        neo_reductions::engines::utils::build_dims_and_policy(&raw_params, prep.structure()).expect("engine dims");
    let mat_digest = neo_reductions::engines::utils::digest_ccs_matrices_with_sparse_cache(prep.structure(), None);
    let header_bundle = neo_reductions::engines::utils::pi_ccs_header_bundle_digest_fields(
        &raw_params,
        prep.structure(),
        dims,
        &mat_digest,
    )
    .expect("header bundle digest");

    SplitNcPiCcsVConfig {
        params: &prep.params,
        structure: prep.structure().into(),
        header_bundle,
        ell_d: dims.ell_d,
        ell_n: dims.ell_n,
        ell_m: dims.ell_m,
        d_sc: dims.d_sc,
    }
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

fn expect_incoming_sidecar_rejected(name: &'static str, mutate: fn(&mut Fixture), needle: &str) {
    let mut fixture = build_fixture();
    mutate(&mut fixture);
    let err = emit_verifier(&fixture)
        .err()
        .unwrap_or_else(|| panic!("{name} sidecar must fail NIFS.V synthesis"));
    assert!(
        err.to_string().contains(needle),
        "expected `{needle}` shape error for {name}, got {err}"
    );
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
    assert!(
        builder.is_satisfied(),
        "native nifs::prove proof must satisfy NIFS.V circuit; first bad row {:?}",
        builder.first_unsatisfied_row()
    );
}

#[test]
fn nifs_v_omits_child_and_running_y_zcol_without_changing_the_relation() {
    let baseline_fixture = build_fixture();
    let (baseline_builder, baseline_outputs) =
        emit_verifier_outputs(&baseline_fixture).expect("emit baseline verifier");
    assert!(baseline_builder.is_satisfied(), "baseline NIFS.V must satisfy");
    assert!(
        baseline_outputs
            .running
            .iter()
            .all(|claim| claim.y_zcol.is_empty()),
        "Π_CCS running claims must not allocate y_zcol"
    );
    assert!(
        baseline_outputs
            .children
            .iter()
            .all(|claim| claim.y_zcol.is_empty()),
        "Π_DEC children must not allocate y_zcol"
    );
    assert!(
        !baseline_outputs.parent.y_zcol.is_empty(),
        "authoritative Π_RLC parent must retain y_zcol"
    );
    assert!(
        baseline_outputs
            .running_parent_authority
            .as_ref()
            .is_some_and(|parent| !parent.y_zcol.is_empty()),
        "incoming Π_RLC parent authority must retain y_zcol"
    );

    let baseline = baseline_builder.snapshot();

    let mut running_mutation = build_fixture();
    running_mutation.running.claims[0].y_zcol[0] += K::ONE;
    let (running_builder, _) = emit_verifier_outputs(&running_mutation).expect("emit running-y_zcol mutation");
    let running = running_builder.snapshot();
    assert!(baseline.has_same_relation(&running));
    assert_eq!(
        baseline.witness(),
        running.witness(),
        "native running y_zcol leaked into the NIFS.V witness"
    );

    let mut child_mutation = build_fixture();
    child_mutation.children[0].y_zcol[0] += K::ONE;
    let (child_builder, _) = emit_verifier_outputs(&child_mutation).expect("emit child-y_zcol mutation");
    let child = child_builder.snapshot();
    assert!(baseline.has_same_relation(&child));
    assert_eq!(
        baseline.witness(),
        child.witness(),
        "native Π_DEC child y_zcol leaked into the NIFS.V witness"
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
        err.to_string().contains("max_fresh_count"),
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
    // The SplitNc verifier's first shape guard checks Π_CCS output count
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
            .contains("parent authority present while running is empty"),
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
            .contains("non-empty running accumulator missing Pi_RLC parent authority"),
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
fn nifs_v_rejects_tampered_running_parent_authority_s_col() {
    // `s_col` is not part of the legacy Π_CCS parent-authority digest, but
    // it is part of HyperNova's carried running accumulator `U_i`. The
    // running-accumulator authority handle absorbed by SplitNc Π_CCS.V must
    // bind it before Fiat-Shamir challenges are sampled.
    let mut fixture = build_fixture();
    let parent = fixture
        .running
        .parent_authority
        .as_mut()
        .expect("running parent authority");
    assert!(!parent.s_col.is_empty(), "fixture must expose parent s_col");
    parent.s_col[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered running parent authority s_col must be rejected"
    );
}

#[test]
fn nifs_v_records_current_unbound_running_parent_y_zcol() {
    // This acceptance pins a known authority gap, not a desired boundary.
    // Hashing the sidecar would bind a prover value without proving its source;
    // a delayed-NC refinement must make this mutation fail semantically.
    let mut fixture = build_fixture();
    let parent = fixture
        .running
        .parent_authority
        .as_mut()
        .expect("running parent authority");
    assert!(!parent.y_zcol.is_empty(), "fixture must expose parent y_zcol");
    parent.y_zcol[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        builder.is_satisfied(),
        "current-gap regression: NIFS.V unexpectedly started binding parent y_zcol; update the delayed-authority audit"
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
    let raw_params = neo_params::NeoParams::goldilocks_auto_r1cs_ccs_with(
        fixture.prep.structure().n.max(fixture.prep.structure().m),
        neo_fold_clean::config::MIN_EFFECTIVE_LAMBDA,
        neo_fold_clean::config::EXTENSION_SAFETY_MARGIN_BITS,
    )
    .expect("raw params reconstruction");
    let dims =
        neo_reductions::engines::utils::build_dims_and_policy(&raw_params, fixture.prep.structure()).expect("dims");
    let combined = neo_reductions::api::rlc_public(
        fixture.prep.structure(),
        &raw_params,
        &rhos,
        &fixture.proof.pi_ccs.outputs,
        |rho_mats, commitments| (fixture.prep.mix_rhos_commits())(rho_mats, commitments),
        dims.ell_d,
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
fn nifs_v_rejects_tampered_pi_ccs_output_y_zcol_padding_lane() {
    // Same padding-canonicality check for the NC output column. The NC
    // terminal identity consumes the full d_pad prefix, so this should fail
    // even before considering the downstream RLC fold.
    let mut fixture = build_fixture();
    let d_pad = D.next_power_of_two();
    assert!(d_pad > D, "fixture must have padded y_zcol lanes");
    assert_eq!(
        fixture.proof.pi_ccs.outputs[0].y_zcol.len(),
        d_pad,
        "fixture must expose full padded y_zcol"
    );
    fixture.proof.pi_ccs.outputs[0].y_zcol[D] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered Π_CCS output y_zcol padding lane must be rejected by NIFS.V"
    );
}

#[test]
fn nifs_v_rejects_tampered_combined_y_zcol_lane() {
    // Tamper a lane of the Π_RLC combined parent's y_zcol. The padded RLC
    // fold `parent.y_zcol = Σ ρ_i · output_i.y_zcol` (rotation on [0, D),
    // zero on tail) must break.
    let mut fixture = build_fixture();
    fixture.combined.y_zcol[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered combined.y_zcol[0] must be rejected");
}

#[test]
fn nifs_v_rejects_tampered_combined_y_zcol_c1_limb() {
    // Full NIFS.V must consume both K limbs of the Π_RLC combined
    // y_zcol. A c0-only NC fold or digest binding would miss this.
    let mut fixture = build_fixture();
    assert!(!fixture.combined.y_zcol.is_empty(), "fixture must have combined y_zcol");
    let original = fixture.combined.y_zcol[0];
    fixture.combined.y_zcol[0] = original + k_c1_one();
    assert_eq!(
        fixture.combined.y_zcol[0].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave c0 unchanged"
    );
    assert_ne!(
        fixture.combined.y_zcol[0].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change c1"
    );

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "NIFS.V accepted a c1-only combined.y_zcol tamper"
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
fn nifs_v_rejects_tampered_combined_s_col_c1_limb() {
    // Π_RLC does not mix `s_col`; it propagates the SplitNc output column
    // point into the combined parent by equality. Mutate only c1 so a
    // c0-only consistency row would miss it.
    let mut fixture = build_fixture();
    assert!(
        !fixture.combined.s_col.is_empty(),
        "fixture must have a DEC parent s_col"
    );
    let original = fixture.combined.s_col[0];
    fixture.combined.s_col[0] = original + k_c1_one();
    assert_eq!(
        fixture.combined.s_col[0].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave c0 unchanged"
    );
    assert_ne!(
        fixture.combined.s_col[0].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change c1"
    );

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "NIFS.V accepted a c1-only combined.s_col tamper"
    );
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
fn nifs_v_rejects_nonzero_inactive_x_in_dec_child() {
    // Inactive X columns are not part of the SuperNeo public projection.
    // They must be pinned to zero in-circuit, otherwise the next running
    // accumulator could carry hidden data skipped by packed-X projections.
    let mut fixture = build_fixture();
    let active_cols = neo_fold_clean::paper::relations::superneo_public_x_cols(fixture.children[0].m_in);
    let total_cols = fixture.children[0].X.cols();
    assert!(active_cols < total_cols, "fixture must have at least one inactive col");

    fixture.children[0].X.set(0, active_cols, F::ONE);

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "NIFS.V circuit accepted a non-zero inactive X column in a Π_DEC child"
    );
}

#[test]
fn nifs_v_rejects_nonzero_inactive_x_in_running() {
    // Running CE claims are consumed by Π_CCS and then carried into the
    // next accumulator handle. Inactive X columns must not become a
    // self-consistent side channel there either.
    let mut fixture = build_fixture();
    let active_cols = neo_fold_clean::paper::relations::superneo_public_x_cols(fixture.running.claims[0].m_in);
    let total_cols = fixture.running.claims[0].X.cols();
    assert!(active_cols < total_cols, "fixture must have at least one inactive col");

    fixture.running.claims[0].X.set(0, active_cols, F::ONE);

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "NIFS.V circuit accepted a non-zero inactive X column in a running CE claim"
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
    // The clean SplitNc/NIFS circuit owns exactly `structure.t()` matrix
    // evaluation rows. A proof must not smuggle an extra y_ring/ct row
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
        err.to_string().contains("y_ring outer length"),
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
        err.to_string().contains("running[0].y_ring.len"),
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
fn nifs_v_rejects_incoming_running_sidecars() {
    // Incoming `running` is HyperNova's carried U_i; its Π_RLC parent is a
    // separately checked cache. The clean circuit does not allocate
    // aux_openings or Pattern-A metadata for either, so those sidecars fail
    // closed before any digest can consume them.
    expect_incoming_sidecar_rejected(
        "running aux_openings",
        |f| f.running.claims[0].aux_openings.push(K::ONE),
        "running[0].aux_openings",
    );
    expect_incoming_sidecar_rejected(
        "running parent aux_openings",
        |f| {
            f.running
                .parent_authority
                .as_mut()
                .expect("running parent authority")
                .aux_openings
                .push(K::ONE);
        },
        "running_parent_authority.aux_openings",
    );
    expect_incoming_sidecar_rejected(
        "running Pattern-A",
        |f| f.running.claims[0].u_len = 1,
        "running[0] carries unsupported Pattern-A",
    );
    expect_incoming_sidecar_rejected(
        "running parent Pattern-A",
        |f| {
            f.running
                .parent_authority
                .as_mut()
                .expect("running parent authority")
                .u_offset = 1;
        },
        "running_parent_authority carries unsupported Pattern-A",
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
    fixture.combined.m_in += 1;

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
fn nifs_v_rejects_combined_aux_openings_sidecar() {
    // Native Π_DEC supports aux_openings decomposition, but the clean
    // SplitNc/NIFS circuit does not allocate or constrain that sidecar.
    // It must fail closed rather than silently drop it.
    let mut fixture = build_fixture();
    fixture.combined.aux_openings.push(K::ONE);

    let err = emit_verifier(&fixture)
        .err()
        .expect("combined aux_openings sidecar must fail synthesis");
    assert!(
        err.to_string().contains("aux_openings"),
        "expected aux_openings shape error, got {err}"
    );
}

#[test]
fn nifs_v_rejects_child_aux_openings_sidecar() {
    let mut fixture = build_fixture();
    fixture.children[0].aux_openings.push(K::ONE);

    let err = emit_verifier(&fixture)
        .err()
        .expect("child aux_openings sidecar must fail synthesis");
    assert!(
        err.to_string().contains("aux_openings"),
        "expected aux_openings shape error, got {err}"
    );
}

#[test]
fn nifs_v_rejects_combined_pattern_a_sidecar() {
    // Pattern-A fields are part of the CE struct but unsupported in the
    // clean circuit path. A non-zero value must not disappear during
    // allocation.
    let mut fixture = build_fixture();
    fixture.combined.c_step_coords.push(F::ONE);

    let err = emit_verifier(&fixture)
        .err()
        .expect("combined Pattern-A sidecar must fail synthesis");
    assert!(
        err.to_string().contains("c_step_coords"),
        "expected c_step_coords shape error, got {err}"
    );
}

#[test]
fn nifs_v_rejects_child_pattern_a_sidecar() {
    let mut fixture = build_fixture();
    fixture.children[0].u_len = 1;

    let err = emit_verifier(&fixture)
        .err()
        .expect("child Pattern-A sidecar must fail synthesis");
    assert!(
        err.to_string().contains("u_len"),
        "expected u_len shape error, got {err}"
    );
}

#[test]
fn nifs_v_rejects_tampered_child_s_col() {
    // Tamper one child's s_col. Π_DEC strict `enforce_s_col_consistency`
    // requires every child s_col equal parent.s_col lane-wise.
    let mut fixture = build_fixture();
    assert!(
        !fixture.children.is_empty() && !fixture.children[0].s_col.is_empty(),
        "test fixture must expose child s_col lanes"
    );
    fixture.children[0].s_col[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered child.s_col must be rejected");
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
fn nifs_v_rejects_tampered_header_digest() {
    let mut fixture = build_fixture();
    fixture.proof.pi_ccs.sumcheck.header_digest[0] ^= 1;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered header_digest must be rejected");
}

#[test]
fn nifs_v_rejects_coherent_forged_fold_digest_chain() {
    // Sneakier than the single-field fold_digest tests: rewrite the
    // recorded Π_CCS header digest and every carried fold_digest to the
    // same forged value. That keeps the Π_CCS output -> Π_RLC parent ->
    // Π_DEC child equality chain self-consistent. The only row that should
    // reject is the transcript catch-up squeeze that recomputes the real
    // header digest from the verifier-driven transcript.
    let mut fixture = build_fixture();
    let mut forged = fixture.proof.pi_ccs.sumcheck.header_digest.clone();
    forged[0] ^= 1;
    let forged_fold_digest: [u8; 32] = forged
        .as_slice()
        .try_into()
        .expect("Pi_CCS header_digest is always 32 bytes");

    fixture.proof.pi_ccs.sumcheck.header_digest = forged.clone();
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

/// Native Π_DEC verify path must reject a children-side CE claim whose `X`
/// has a non-zero entry in an inactive column. The Π_DEC children become
/// the next running accumulator; without this guard, a terminal state
/// could carry a non-canonical accumulator that no downstream Π_CCS step
/// would catch.
#[test]
fn native_nifs_verify_rejects_nonzero_inactive_x_in_dec_child() {
    let mut fixture = build_fixture();
    let active_cols = neo_fold_clean::paper::relations::superneo_public_x_cols(fixture.children[0].m_in);
    let total_cols = fixture.children[0].X.cols();
    assert!(active_cols < total_cols, "fixture must have at least one inactive col");

    // Mutate one inactive slot in proof.pi_dec.children[0].X (this is the
    // verifier-side proof copy that `nifs::verify` walks).
    fixture.proof.pi_dec.children[0]
        .X
        .set(0, active_cols, F::ONE);

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
        "native nifs::verify must reject non-zero inactive X in Π_DEC child"
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

/// Native Π_CCS shape check must reject a `running` CE claim whose `X` has
/// a non-zero entry in an inactive column. The circuit-side verifier
/// enforces the same invariant and the v2 `ce_claim_digest` skips inactive
/// columns; without this guard, the column would not be transcript-bound
/// natively, so an attacker could smuggle data there.
#[test]
fn native_nifs_verify_rejects_nonzero_inactive_x_in_running() {
    let mut fixture = build_fixture();
    let active_cols = neo_fold_clean::paper::relations::superneo_public_x_cols(fixture.running.claims[0].m_in);
    let total_cols = fixture.running.claims[0].X.cols();
    assert!(active_cols < total_cols, "fixture must have at least one inactive col");

    // Mutate one inactive slot in running[0].X.
    fixture.running.claims[0].X.set(0, active_cols, F::ONE);

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
        "native nifs::verify must reject non-zero inactive X in running CE claim"
    );
}
