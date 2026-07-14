use std::sync::Arc;

use neo_ajtai::{setup as ajtai_setup, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, Mat, SparsePoly};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::optimized_engine::PiCcsProofVariant;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;

fn identity_left(n: usize, m: usize) -> Mat<F> {
    let mut mat = Mat::zero(n, m, F::ZERO);
    for i in 0..n.min(m) {
        mat.set(i, i, F::ONE);
    }
    mat
}

fn zero_poly(t: usize) -> SparsePoly<F> {
    SparsePoly::new(t, Vec::new())
}

fn build_fixture(
    label: &'static [u8],
    n: usize,
    m: usize,
) -> (
    NeoParams,
    CcsStructure<F>,
    AjtaiSModule,
    CcsClaim<neo_ajtai::Commitment, F>,
    CcsWitness<F>,
    Poseidon2Transcript,
) {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let s = CcsStructure::new(vec![identity_left(n, m)], zero_poly(1)).expect("ccs");

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(123);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, m / D).expect("ajtai setup");
    let l = AjtaiSModule::new(Arc::new(pp));

    // Z entries are all zero digits (in-range for any b>=2).
    let z = Mat::from_row_major(D, m / D, vec![F::ZERO; D * (m / D)]);
    let c = l.commit(&z);
    let mcs_inst = CcsClaim {
        adv: None,
        c,
        x: vec![],
        m_in: 0,
    };
    let mcs_wit = CcsWitness {
        w: vec![F::ZERO; m],
        Z: z,
    };

    let tr = Poseidon2Transcript::new(label);
    (params, s, l, mcs_inst, mcs_wit, tr)
}

#[test]
fn split_nc_rejects_missing_y_zcol() {
    let (params, s, l, mcs_inst, mcs_wit, mut tr_p) = build_fixture(b"test/split_nc/missing_y_zcol", 4, D);

    let (mut out_me, proof) = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        core::slice::from_ref(&mcs_wit),
        &[],
        &[],
        &l,
    )
    .expect("prove");

    assert_eq!(proof.variant, PiCcsProofVariant::SplitNcV1);

    for out in out_me.iter_mut() {
        out.y_zcol.clear();
    }

    let mut tr_v = Poseidon2Transcript::new(b"test/split_nc/missing_y_zcol");
    let res = neo_reductions::api::verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        &[],
        &out_me,
        &proof,
    );
    assert!(res.is_err(), "expected verifier to reject missing y_zcol");
}

#[test]
fn split_nc_rejects_s_col_mismatch() {
    let (params, s, l, mcs_inst, mcs_wit, mut tr_p) = build_fixture(b"test/split_nc/s_col_mismatch", 4, D);

    let (mut out_me, proof) = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        core::slice::from_ref(&mcs_wit),
        &[],
        &[],
        &l,
    )
    .expect("prove");

    assert_eq!(proof.variant, PiCcsProofVariant::SplitNcV1);
    assert!(!out_me.is_empty());
    out_me[0].s_col[0] += K::ONE;

    let mut tr_v = Poseidon2Transcript::new(b"test/split_nc/s_col_mismatch");
    let res = neo_reductions::api::verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        &[],
        &out_me,
        &proof,
    );
    assert!(res.is_err(), "expected verifier to reject mismatched s_col");
}

#[test]
fn split_nc_rejects_missing_nc_sumcheck_rounds() {
    let (params, s, l, mcs_inst, mcs_wit, mut tr_p) = build_fixture(b"test/split_nc/missing_nc_rounds", 4, D);

    let (out_me, mut proof) = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        core::slice::from_ref(&mcs_wit),
        &[],
        &[],
        &l,
    )
    .expect("prove");

    assert_eq!(proof.variant, PiCcsProofVariant::SplitNcV1);
    proof.sumcheck_rounds_nc.clear();

    let mut tr_v = Poseidon2Transcript::new(b"test/split_nc/missing_nc_rounds");
    let res = neo_reductions::api::verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        &[],
        &out_me,
        &proof,
    );
    assert!(res.is_err(), "expected verifier to reject missing NC rounds");
}

#[test]
fn split_nc_uses_ell_m_for_s_col_when_rectangular() {
    let (params, s, l, mcs_inst, mcs_wit, mut tr_p) = build_fixture(b"test/split_nc/ell_m", 4, D);
    let dims = neo_reductions::engines::utils::build_dims_and_policy(&params, &s).expect("dims");
    assert_ne!(dims.ell_n, dims.ell_m, "test requires ell_n != ell_m");

    let (out_me, proof) = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        core::slice::from_ref(&mcs_wit),
        &[],
        &[],
        &l,
    )
    .expect("prove");

    assert_eq!(proof.variant, PiCcsProofVariant::SplitNcV1);
    assert_eq!(proof.sumcheck_rounds_nc.len(), dims.ell_nc);
    assert!(!out_me.is_empty());
    assert_eq!(out_me[0].s_col.len(), dims.ell_m);

    let mut tr_v = Poseidon2Transcript::new(b"test/split_nc/ell_m");
    let ok = neo_reductions::api::verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        &[],
        &out_me,
        &proof,
    )
    .expect("verify should not error");
    assert!(ok);
}

#[test]
fn split_nc_verify_ignores_stale_ct_shell() {
    let (params, s, l, mcs_inst, mcs_wit, mut tr_p) = build_fixture(b"test/split_nc/stale_ct", 4, D);

    let (mut out_me, proof) = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        core::slice::from_ref(&mcs_wit),
        &[],
        &[],
        &l,
    )
    .expect("prove");

    assert_eq!(proof.variant, PiCcsProofVariant::SplitNcV1);
    assert!(!out_me.is_empty());
    out_me[0].ct[0] += K::ONE;

    let mut tr_v = Poseidon2Transcript::new(b"test/split_nc/stale_ct");
    let ok = neo_reductions::api::verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        &[],
        &out_me,
        &proof,
    )
    .expect("verify should not error");
    assert!(ok, "stale ct shell must not fail optimized verify");
}

#[test]
fn split_nc_raw_verify_rejects_unbound_output_ct() {
    let label = b"test/split_nc/redteam/unbound_output_ct";
    let (params, s, l, mcs_inst, mcs_wit, mut tr_p) = build_fixture(label, 4, D);

    let (mut outputs, proof) = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        core::slice::from_ref(&mcs_wit),
        &[],
        &[],
        &l,
    )
    .expect("prove");

    assert_eq!(outputs[0].ct[0], outputs[0].y_ring[0][0]);
    outputs[0].ct[0] += K::ONE;
    assert_ne!(outputs[0].ct[0], outputs[0].y_ring[0][0]);

    let mut tr_v = Poseidon2Transcript::new(label);
    let accepted = neo_reductions::api::verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        &[],
        &outputs,
        &proof,
    )
    .expect("malformed output must receive a verdict");

    assert!(
        !accepted,
        "raw Pi_CCS accepted an output CE claim whose ct disagrees with y_ring"
    );
}

#[test]
fn split_nc_tampered_y_zcol_is_rejected() {
    // If `eq((α',s'),β)` happens to be zero, a single attempt may not detect tampering.
    // Try a few transcript labels and require that at least one detects it.
    const LABELS: [&[u8]; 16] = [
        b"test/split_nc/tamper/0",
        b"test/split_nc/tamper/1",
        b"test/split_nc/tamper/2",
        b"test/split_nc/tamper/3",
        b"test/split_nc/tamper/4",
        b"test/split_nc/tamper/5",
        b"test/split_nc/tamper/6",
        b"test/split_nc/tamper/7",
        b"test/split_nc/tamper/8",
        b"test/split_nc/tamper/9",
        b"test/split_nc/tamper/10",
        b"test/split_nc/tamper/11",
        b"test/split_nc/tamper/12",
        b"test/split_nc/tamper/13",
        b"test/split_nc/tamper/14",
        b"test/split_nc/tamper/15",
    ];

    for &label in LABELS.iter() {
        let (params, s, l, mcs_inst, mcs_wit, mut tr_p) = build_fixture(label, 4, D);
        let (mut out_me, proof) = neo_reductions::api::prove(
            FoldingMode::Optimized,
            &mut tr_p,
            &params,
            &s,
            core::slice::from_ref(&mcs_inst),
            core::slice::from_ref(&mcs_wit),
            &[],
            &[],
            &l,
        )
        .expect("prove");

        let mut tr_v_ok = Poseidon2Transcript::new(label);
        let ok = neo_reductions::api::verify(
            FoldingMode::Optimized,
            &mut tr_v_ok,
            &params,
            &s,
            core::slice::from_ref(&mcs_inst),
            &[],
            &out_me,
            &proof,
        )
        .expect("verify should not error");
        assert!(ok, "baseline proof must verify");

        // Tamper y_zcol without breaking structural checks by setting it to a constant
        // out-of-range value `b` (so ⟨y_zcol, χ_α⟩ == b for any α).
        let b = K::from(F::from_u64(params.b as u64));
        for out in out_me.iter_mut() {
            out.y_zcol.fill(b);
        }

        let mut tr_v_bad = Poseidon2Transcript::new(label);
        let res_bad = neo_reductions::api::verify(
            FoldingMode::Optimized,
            &mut tr_v_bad,
            &params,
            &s,
            core::slice::from_ref(&mcs_inst),
            &[],
            &out_me,
            &proof,
        );
        match res_bad {
            Ok(false) | Err(_) => return,
            Ok(true) => continue,
        }
    }

    panic!("tampered y_zcol was accepted in all attempts");
}

#[test]
fn split_nc_raw_verify_rejects_unbound_inactive_output_x() {
    let label = b"test/split_nc/redteam/inactive_output_x";
    let (params, s, l, mut mcs, mut wit, mut tr_p) = build_fixture(label, 4, D);
    mcs.m_in = 2;
    mcs.x = vec![F::ZERO; 2];
    wit.w = vec![F::ZERO; D - 2];

    let (mut outputs, proof) = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&mcs),
        core::slice::from_ref(&wit),
        &[],
        &[],
        &l,
    )
    .expect("prove");

    assert_eq!(outputs[0].X.rows(), D);
    assert_eq!(outputs[0].X.cols(), 2);
    outputs[0].X[(0, 1)] = F::ONE;

    let mut tr_v = Poseidon2Transcript::new(label);
    let accepted = neo_reductions::api::verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&mcs),
        &[],
        &outputs,
        &proof,
    )
    .expect("malformed output must receive a verdict");

    assert!(
        !accepted,
        "raw Pi_CCS accepted an output CE claim with unbound inactive X data"
    );
}

#[test]
fn raw_pi_ccs_rejects_fresh_count_above_parameter_profile() {
    let label = b"test/split_nc/redteam/fresh_count_policy";
    let (params, s, l, claim, witness, mut tr_p) = build_fixture(label, 4, D);
    let count = neo_params::goldilocks_paper_b2::MAX_FRESH_K as usize + 1;
    let claims = vec![claim; count];
    let witnesses = vec![witness; count];

    let (outputs, proof) = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &s,
        &claims,
        &witnesses,
        &[],
        &[],
        &l,
    )
    .expect("raw Pi_CCS currently accepts a fresh count above its installed profile");

    let mut tr_v = Poseidon2Transcript::new(label);
    let accepted = neo_reductions::api::verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &s,
        &claims,
        &[],
        &outputs,
        &proof,
    )
    .expect("raw verifier must return a verdict");

    assert!(
        !accepted,
        "raw Pi_CCS accepted {} fresh claims under a profile capped at {}",
        count,
        neo_params::goldilocks_paper_b2::MAX_FRESH_K
    );
}

#[test]
fn raw_pi_ccs_rejects_running_count_above_parameter_profile() {
    let seed_label = b"test/split_nc/redteam/running_count_seed";
    let (params, s, l, claim, witness, mut seed_tr) = build_fixture(seed_label, 4, D);
    let running_witness = witness.Z.clone();
    let (seed_outputs, _) = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut seed_tr,
        &params,
        &s,
        core::slice::from_ref(&claim),
        core::slice::from_ref(&witness),
        &[],
        &[],
        &l,
    )
    .expect("seed one valid running claim");

    let count = params.k_rho as usize + 1;
    let running = vec![seed_outputs[0].clone(); count];
    let running_witnesses = vec![running_witness; count];
    let label = b"test/split_nc/redteam/running_count_policy";
    let mut tr_p = Poseidon2Transcript::new(label);
    let (outputs, proof) = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&claim),
        core::slice::from_ref(&witness),
        &running,
        &running_witnesses,
        &l,
    )
    .expect("raw Pi_CCS currently accepts a running count above its installed profile");

    let mut tr_v = Poseidon2Transcript::new(label);
    let accepted = neo_reductions::api::verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&claim),
        &running,
        &outputs,
        &proof,
    )
    .expect("raw verifier must return a verdict");

    assert!(
        !accepted,
        "raw Pi_CCS accepted {count} running claims under a profile fixed to k_rho={}",
        params.k_rho
    );
}
