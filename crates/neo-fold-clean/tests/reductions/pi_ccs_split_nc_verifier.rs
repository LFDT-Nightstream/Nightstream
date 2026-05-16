//! SplitNcV1 — Π_CCS.V composition: hard-gate parity test.
//!
//! This is the gating test for sub-step J: a real native `pi_ccs::prove`
//! proof must satisfy the composed in-circuit SplitNc Π_CCS.V verifier
//! (`enforce_split_nc_pi_ccs_v`), and targeted mutations of the proof must
//! cause `R1csBuilder::is_satisfied()` to return false.
//!
//! Without this, all the FE/NC/digest/transcript-binding sub-gadget parity
//! tests only prove the *pieces* work; the *composition* is unverified.
//!
//! Tests:
//! - `split_nc_pi_ccs_v_accepts_native_proof`
//! - `split_nc_pi_ccs_v_rejects_tampered_fe_round`
//! - `split_nc_pi_ccs_v_rejects_tampered_nc_y_zcol`
//! - `split_nc_pi_ccs_v_rejects_tampered_header_digest`
//! - `split_nc_pi_ccs_v_rejects_output_m_in_mismatch`

#![allow(non_snake_case)]

use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, TranscriptGadget};
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::reductions::pi_ccs;
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::{
    enforce_split_nc_pi_ccs_v, Error, SplitNcPiCcsVConfig, SplitNcPiCcsVMessages,
};
use neo_fold_clean::paper::relations::CcsClaim;
use neo_math::ring::D;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

// The paper-layer `Transcript::session()` initializes its inner sponge with
// `b"neo.fold.clean/session/v1"`. The in-circuit `TranscriptGadget` must use
// the same label so the prove- and verify-side sponge states stay in sync.
const SESSION_LABEL: &[u8] = b"neo.fold.clean/session/v1";

// ── R1CS fixture: z[0]·(z[1] + z[2]) = z[3] (three-term addition) ────────

/// One-constraint R1CS: `(z[1] + z[2]) · z[0] = z[3]`. With `z[0] = 1` this
/// degenerates into `z[1] + z[2] = z[3]`, satisfied by any `(a, b, a+b)`.
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

/// Satisfying assignment `z = [1, a, b, a+b, 0, ..., 0]`.
fn assignment(a: u64, b: u64) -> Vec<F> {
    let mut z = vec![F::ZERO; D];
    z[0] = F::ONE;
    z[1] = F::from_u64(a);
    z[2] = F::from_u64(b);
    z[3] = F::from_u64(a + b);
    z
}

// ── Fixture: native NIFS step + standalone Π_CCS proof ────────────────────

/// Test fixture. `running` is the running accumulator after one NIFS fold,
/// and `proof` is a fresh `pi_ccs::prove` output that the in-circuit verifier
/// must accept.
struct Fixture {
    prep: neo_fold_clean::Preprocessing,
    fresh_claims: Vec<CcsClaim>,
    running: RunningInstance,
    proof: pi_ccs::Proof,
}

fn build_fixture() -> Fixture {
    let r1cs = three_term_addition();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");

    // Step 1: seed the running accumulator with one NIFS fold so that the
    // second Π_CCS proof has a non-empty `running`. Without this step the
    // verifier path skips the eq(α', α)·eq(r', r_in)·γ^k_total·eval_sum
    // branch in the FE terminal identity, which we want to exercise.
    let first = direct_ccs::build_instance(&prep, &r1cs, &assignment(1, 0)).expect("first instance");
    let mut first_tr = Transcript::session();
    let (running, _) = neo_fold_clean::paper::nifs::prove(
        &mut first_tr,
        &prep.params,
        &prep.structure,
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        vec![first],
        &RunningInstance::default(),
    )
    .expect("first NIFS.P");

    // Step 2: standalone Π_CCS proof. This is what the in-circuit verifier
    // mirrors. Uses a fresh session transcript with the same label as the
    // in-circuit `TranscriptGadget::new(...)` will use.
    let second = direct_ccs::build_instance(&prep, &r1cs, &assignment(0, 1)).expect("second instance");
    let fresh_claims = vec![second.claim.clone()];

    let mut tr = Transcript::session();
    let proof = pi_ccs::prove(
        &mut tr,
        &prep.params,
        &prep.structure,
        &prep.log,
        vec![second],
        &running,
    )
    .expect("pi_ccs.prove");

    Fixture {
        prep,
        fresh_claims,
        running,
        proof,
    }
}

// ── Verifier driver: emit the SplitNc Π_CCS.V circuit on the fixture ─────

/// Build the `SplitNcPiCcsVConfig` for this fixture by recomputing the
/// engine's dims + matrix digest + header bundle from the same params and
/// structure the prover used. Mirrors the native verifier wrapper exactly.
fn split_nc_config<'a>(prep: &'a neo_fold_clean::Preprocessing) -> SplitNcPiCcsVConfig<'a> {
    // The paper-layer `Params` keeps its `NeoParams` private; reconstruct
    // it from the same shape the production `r1cs_params` derives.
    let raw_params = neo_params::NeoParams::goldilocks_auto_r1cs_ccs_with(
        prep.structure.n.max(prep.structure.m),
        neo_fold_clean::config::MIN_EFFECTIVE_LAMBDA,
        neo_fold_clean::config::EXTENSION_SAFETY_MARGIN_BITS,
    )
    .expect("raw params reconstruction");

    let dims =
        neo_reductions::engines::utils::build_dims_and_policy(&raw_params, &prep.structure).expect("engine dims");
    let mat_digest = neo_reductions::engines::utils::digest_ccs_matrices_with_sparse_cache(&prep.structure, None);
    let header_bundle = neo_reductions::engines::utils::pi_ccs_header_bundle_digest_fields(
        &raw_params,
        &prep.structure,
        dims,
        &mat_digest,
    )
    .expect("header bundle digest");

    SplitNcPiCcsVConfig {
        params: &prep.params,
        structure: &prep.structure,
        header_bundle,
        ell_d: dims.ell_d,
        ell_n: dims.ell_n,
        ell_m: dims.ell_m,
        d_sc: dims.d_sc,
    }
}

/// Build a fresh R1cs circuit, allocate the SplitNc Π_CCS.V composition for
/// the fixture's proof, and return the populated `R1csBuilder`. Caller
/// inspects `builder.is_satisfied()`.
fn emit_verifier(f: &Fixture) -> Result<R1csBuilder, Error> {
    let mut builder = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut builder, SESSION_LABEL);
    let cfg = split_nc_config(&f.prep);

    enforce_split_nc_pi_ccs_v(
        &mut builder,
        &mut tr,
        &cfg,
        &SplitNcPiCcsVMessages {
            fresh: &f.fresh_claims,
            running: &f.running.claims,
            running_parent_authority: f.running.parent_authority.as_ref(),
            outputs: &f.proof.outputs,
            sumcheck_rounds_fe: &f.proof.sumcheck.sumcheck_rounds,
            sumcheck_rounds_nc: &f.proof.sumcheck.sumcheck_rounds_nc,
            header_digest: &f.proof.sumcheck.header_digest,
        },
    )?;
    Ok(builder)
}

// ── Tests ────────────────────────────────────────────────────────────────

#[test]
fn split_nc_pi_ccs_v_accepts_native_proof() {
    let fixture = build_fixture();
    let builder = emit_verifier(&fixture).expect("emit verifier");

    assert!(
        builder.is_satisfied(),
        "native pi_ccs::prove proof must satisfy SplitNc Π_CCS.V circuit; first bad row {:?}",
        builder.first_unsatisfied_row()
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_fe_round() {
    let mut fixture = build_fixture();
    // Bump the leading coeff of the first FE round. The chain identity
    // `g(0) + g(1) == claim_q` and downstream sumcheck challenges diverge.
    fixture.proof.sumcheck.sumcheck_rounds[0][0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered FE round must be rejected");
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_nc_y_zcol() {
    let mut fixture = build_fixture();
    // Mutate one y_zcol lane of the first output. The NC terminal identity
    // recomputes `⟨y_zcol, χ_{α'}⟩` from this wire, so its pin-to-rhs_nc
    // must break.
    fixture.proof.outputs[0].y_zcol[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered y_zcol must be rejected");
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_header_digest() {
    let mut fixture = build_fixture();
    // Flip one byte of the captured header digest. The catch-up squeeze
    // computes the real digest and pins each lane to the recorded value,
    // so any byte flip breaks at least one lane's pin.
    fixture.proof.sumcheck.header_digest[0] ^= 1;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered header_digest must be rejected");
}

#[test]
fn split_nc_pi_ccs_v_rejects_output_m_in_mismatch() {
    // `m_in` is a structural field; the verifier rejects with `Err(Shape)`
    // *before* emitting any constraints when it disagrees with the input
    // claim's m_in. (Mirrors native `validate_me_outputs_against_inputs`.)
    let mut fixture = build_fixture();
    fixture.proof.outputs[0].m_in += 1;

    let err = match emit_verifier(&fixture) {
        Ok(_) => panic!("m_in mismatch must surface as Err(Shape)"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(msg.contains("m_in"), "expected 'm_in' in error, got: {msg}");
}
