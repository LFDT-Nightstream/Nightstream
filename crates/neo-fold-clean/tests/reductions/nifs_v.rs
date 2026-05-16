//! NIFS.V composition — hard-gate parity test.
//!
//! A real native `nifs::prove` proof must satisfy the composed in-circuit
//! [`enforce_nifs_v_circuit_with_transcript`], and targeted mutations of
//! the proof must cause `R1csBuilder::is_satisfied()` to return false.
//!
//! Tests:
//! - `nifs_v_accepts_native_proof`
//! - `nifs_v_rejects_tampered_fe_round`
//! - `nifs_v_rejects_tampered_running_parent_authority`
//! - `nifs_v_rejects_tampered_combined_y_zcol_lane`
//! - `nifs_v_rejects_tampered_child_s_col`
//! - `nifs_v_rejects_tampered_header_digest`

#![allow(non_snake_case)]

use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, TranscriptGadget};
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::nifs::circuit::{
    enforce_nifs_v_circuit_with_transcript, NifsVCircuitConfig, NifsVCircuitMessages,
};
use neo_fold_clean::paper::nifs::NifsProof;
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsVConfig;
use neo_fold_clean::paper::relations::CcsClaim;
use neo_math::ring::D;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

const SESSION_LABEL: &[u8] = b"neo.fold.clean/session/v1";

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
        prep.mix_rhos_commits,
        prep.combine_b_pows,
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
        prep.mix_rhos_commits,
        prep.combine_b_pows,
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
        structure: prep.structure(),
        header_bundle,
        ell_d: dims.ell_d,
        ell_n: dims.ell_n,
        ell_m: dims.ell_m,
        d_sc: dims.d_sc,
    }
}

fn emit_verifier(f: &Fixture) -> Result<R1csBuilder, neo_fold_clean::paper::nifs::circuit::Error> {
    let mut builder = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut builder, SESSION_LABEL);
    let cfg = NifsVCircuitConfig {
        pi_ccs: pi_ccs_config(&f.prep),
    };
    enforce_nifs_v_circuit_with_transcript(
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
    Ok(builder)
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
            fixture.prep.mix_rhos_commits,
            fixture.prep.combine_b_pows,
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
fn nifs_v_rejects_tampered_fe_round() {
    let mut fixture = build_fixture();
    fixture.proof.pi_ccs.sumcheck.sumcheck_rounds[0][0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered FE round must be rejected");
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
fn nifs_v_rejects_tampered_child_s_col() {
    // Tamper one child's s_col. Π_DEC strict `enforce_s_col_consistency`
    // requires every child s_col equal parent.s_col lane-wise.
    let mut fixture = build_fixture();
    if fixture.children.is_empty() || fixture.children[0].s_col.is_empty() {
        eprintln!("test fixture has no children/s_col — skipping");
        return;
    }
    fixture.children[0].s_col[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered child.s_col must be rejected");
}

#[test]
fn nifs_v_rejects_tampered_header_digest() {
    let mut fixture = build_fixture();
    fixture.proof.pi_ccs.sumcheck.header_digest[0] ^= 1;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered header_digest must be rejected");
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
        fixture.prep.mix_rhos_commits,
        fixture.prep.combine_b_pows,
        &fixture.fresh_claims,
        &fixture.running,
        &fixture.proof,
    );
    assert!(
        result.is_err(),
        "native nifs::verify must reject non-zero inactive X in Π_DEC child"
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
        fixture.prep.mix_rhos_commits,
        fixture.prep.combine_b_pows,
        &fixture.fresh_claims,
        &fixture.running,
        &fixture.proof,
    );
    assert!(
        result.is_err(),
        "native nifs::verify must reject non-zero inactive X in running CE claim"
    );
}
