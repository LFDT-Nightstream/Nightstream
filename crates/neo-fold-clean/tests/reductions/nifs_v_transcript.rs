//! NIFS.V transcript-phase red-team tests.
//!
//! These tests build coherent lower-layer proof fragments under the wrong
//! Fiat-Shamir phase. They are meant to catch verifier gadgets that run the
//! right algebra with challenges sampled from the wrong transcript state.

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
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsVConfig;
use neo_fold_clean::paper::relations::CcsClaim;
use neo_fold_clean::paper::{pi_dec, pi_rlc};
use neo_fold_clean::{CeClaim, Preprocessing};
use neo_math::ring::D;
use neo_math::F;
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

fn build_honest_fixture() -> Fixture {
    let r1cs = three_term_addition();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");

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
    let fresh_claims = vec![second.claim.clone()];

    let mut second_tr = Transcript::session();
    let (next_running, proof) = nifs::prove(
        &mut second_tr,
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

fn pi_ccs_config<'a>(prep: &'a Preprocessing) -> SplitNcPiCcsVConfig<'a> {
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
    let mut allowed = Vec::new();
    allowed.extend(
        outputs
            .running
            .iter()
            .flat_map(|claim| claim.y_zcol.iter().flat_map(|v| [v.c0.col(), v.c1.col()])),
    );
    if let Some(parent) = &outputs.running_parent_authority {
        allowed.extend(parent.y_zcol.iter().flat_map(|v| [v.c0.col(), v.c1.col()]));
    }
    allowed.extend(
        outputs
            .children
            .iter()
            .flat_map(|child| child.y_zcol.iter().map(|v| v.col())),
    );
    allowed.sort_unstable();
    assert!(
        unconstrained == allowed,
        "composed NIFS.V verifier left unexpected unconstrained columns: got {unconstrained:?}, \
         expected only non-authority y_zcol sidecar limbs {allowed:?}"
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
