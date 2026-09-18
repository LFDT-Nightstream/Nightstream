//! `adv` tuple through a real fold — spec §5.2 R2/R3, M1b (live) slice:
//! Π_CCS forwards, Π_RLC ρ-mixes, Π_DEC recomposes and re-commits child
//! lane slices, and the terminal slice-opening pins tuples to witnesses.
//! Every tamper lands on the specific check the spec names, never on a
//! host replay comparison.

use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_fold_clean::config;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::lifecycle::{preprocess, Preprocessing};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::nifs;
use neo_fold_clean::paper::relations::{CcsInstance, LaneRanges, LaneScheme, Structure};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

/// Four ring columns: column 0 hosts `x`/app, columns 1–3 are the ops, is,
/// and fs lanes (whole ring columns — L-ALIGN is the unit of the ranges).
const COLS: usize = 4;

fn lane_ranges() -> LaneRanges {
    LaneRanges {
        ops: 1..2,
        is: 2..3,
        fs: 3..4,
    }
}

fn lane_scheme(prep: &Preprocessing) -> LaneScheme {
    LaneScheme::from_seeds(prep.params.kappa() as usize, lane_ranges(), [0xA5; 32], [0x5A; 32])
        .expect("lane scheme from test seeds")
}

/// A satisfiable structure wide enough to host the three lanes: identity
/// matrix, zero polynomial — every low-norm assignment satisfies it, so
/// the tests exercise the fold plumbing, not app semantics.
fn wide_preprocessing() -> Preprocessing {
    let m = COLS * D;
    let structure: Structure =
        CcsStructure::new(vec![Mat::identity(m)], SparsePoly::new(1, vec![])).expect("wide test structure");
    let params = config::r1cs_params(structure.n, structure.m).expect("params for wide structure");
    support::install_ajtai_module(&params, &structure);
    preprocess(params, structure, Some(1)).expect("wide preprocessing")
}

#[path = "../support/mod.rs"]
mod support;

/// A low-norm instance whose lane columns carry seed-dependent ±1 bits,
/// with its honest `adv` tuple committed by the scheme.
fn adv_instance(prep: &Preprocessing, scheme: &LaneScheme, seed: u64) -> CcsInstance {
    let mut z = vec![F::ZERO; prep.structure().m];
    for (i, slot) in z.iter_mut().enumerate().skip(D) {
        if (seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left((i % 61) as u32)
            >> (i % 7))
            & 1
            == 1
        {
            *slot = F::ONE;
        }
    }
    let mut instance = CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &z, 1)
        .expect("low-norm adv instance");
    instance.claim.adv = Some(scheme.commit(&instance.witness.Z).expect("lane commit"));
    instance
}

fn prove_fold(
    prep: &Preprocessing,
    scheme: &LaneScheme,
    fresh: Vec<CcsInstance>,
) -> Result<(RunningInstance, nifs::NifsProof), nifs::Error> {
    let mut tr = Transcript::session();
    nifs::prove(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        Some(scheme),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &RunningInstance::default(),
    )
}

fn verify_fold(
    prep: &Preprocessing,
    fresh_claims: &[neo_fold_clean::paper::relations::CcsClaim],
    proof: &nifs::NifsProof,
) -> Result<RunningInstance, nifs::Error> {
    let mut tr = Transcript::session();
    nifs::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh_claims,
        &RunningInstance::default(),
        proof,
    )
}

/// The full R2 + R3 loop: fresh tuples ride Π_CCS unchanged, ρ-mix through
/// Π_RLC, split into child tuples by Π_DEC — and every terminal child's
/// tuple opens against its own witness's lane slices (the decider check).
/// This is the algebra security-note Lemma 1 composes, exercised end to end.
#[test]
fn adv_mirrors_through_a_real_fold_and_children_open() {
    let prep = wide_preprocessing();
    let scheme = lane_scheme(&prep);
    let fresh = vec![adv_instance(&prep, &scheme, 3), adv_instance(&prep, &scheme, 17)];
    let fresh_claims: Vec<_> = fresh.iter().map(|i| i.claim.clone()).collect();

    let (next_running, proof) = prove_fold(&prep, &scheme, fresh).expect("NIFS.P with lanes");
    let verified = verify_fold(&prep, &fresh_claims, &proof).expect("NIFS.V");
    assert_eq!(verified.claims, next_running.claims);

    let parent = next_running.parent_authority.as_ref().expect("RLC parent");
    assert!(parent.adv.is_some(), "combined claim must carry the mixed tuple");
    for (claim, witness) in next_running.claims.iter().zip(&next_running.witnesses) {
        let adv = claim.adv.as_ref().expect("every child carries a tuple");
        assert!(
            scheme.open_matches(adv, witness).expect("openable shapes"),
            "terminal slice-opening (R3): child tuple must open to its lane slices"
        );
    }
}

/// Post-challenge tamper on a child tuple: Π_DEC's recomposition rejects.
#[test]
fn verify_rejects_tampered_child_adv() {
    let prep = wide_preprocessing();
    let scheme = lane_scheme(&prep);
    let fresh = vec![adv_instance(&prep, &scheme, 3)];
    let fresh_claims: Vec<_> = fresh.iter().map(|i| i.claim.clone()).collect();

    let (_, mut proof) = prove_fold(&prep, &scheme, fresh).expect("NIFS.P");
    let adv = proof.pi_dec.children[0].adv.as_mut().expect("child tuple");
    adv.fs.data[0] += F::ONE;
    assert!(
        verify_fold(&prep, &fresh_claims, &proof).is_err(),
        "tampered child adv must fail Π_DEC recomposition"
    );
}

/// Post-challenge tamper on the combined tuple: Π_RLC's mix check rejects.
#[test]
fn verify_rejects_tampered_combined_adv() {
    let prep = wide_preprocessing();
    let scheme = lane_scheme(&prep);
    let fresh = vec![adv_instance(&prep, &scheme, 3)];
    let fresh_claims: Vec<_> = fresh.iter().map(|i| i.claim.clone()).collect();

    let (_, mut proof) = prove_fold(&prep, &scheme, fresh).expect("NIFS.P");
    let adv = proof.pi_rlc.combined.adv.as_mut().expect("combined tuple");
    adv.ops.data[0] += F::ONE;
    assert!(
        verify_fold(&prep, &fresh_claims, &proof).is_err(),
        "tampered combined adv must fail Π_RLC's mix recomputation"
    );
}

/// Π_CCS forwarding is checked, not trusted: an output tuple differing
/// from its input claim's is rejected before any mixing.
#[test]
fn verify_rejects_tampered_pi_ccs_output_adv() {
    let prep = wide_preprocessing();
    let scheme = lane_scheme(&prep);
    let fresh = vec![adv_instance(&prep, &scheme, 3)];
    let fresh_claims: Vec<_> = fresh.iter().map(|i| i.claim.clone()).collect();

    let (_, mut proof) = prove_fold(&prep, &scheme, fresh).expect("NIFS.P");
    proof.pi_ccs.outputs[0].adv = None;
    assert!(
        verify_fold(&prep, &fresh_claims, &proof).is_err(),
        "dropping adv on a Π_CCS output must fail forwarding validation"
    );
}

/// Fail-closed: an adv-bearing fold without a LaneScheme cannot produce
/// child tuples and must error rather than silently drop them.
#[test]
fn prove_without_lane_scheme_fails_closed() {
    let prep = wide_preprocessing();
    let scheme = lane_scheme(&prep);
    let fresh = vec![adv_instance(&prep, &scheme, 3)];

    let mut tr = Transcript::session();
    let result = nifs::prove(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &RunningInstance::default(),
    );
    assert!(result.is_err(), "adv-bearing parent without a LaneScheme must fail");
}

/// All-or-nothing presence across a fold's inputs (§5.1 lifted to the
/// fold): one tuple-bearing and one plain claim cannot mix.
#[test]
fn mixed_adv_presence_is_rejected() {
    let prep = wide_preprocessing();
    let scheme = lane_scheme(&prep);
    let with_adv = adv_instance(&prep, &scheme, 3);
    let plain = {
        let z = vec![F::ZERO; prep.structure().m];
        CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &z, 1).expect("plain instance")
    };

    let result = prove_fold(&prep, &scheme, vec![with_adv, plain]);
    assert!(result.is_err(), "mixed adv presence must be rejected before mixing");
}

/// The Π_CCS input transcript binds `adv` before the output challenge. The
/// pre-ρ output absorb then commits only the newly sent evaluation messages;
/// deterministic forwarded fields must not be hashed a second time.
#[test]
fn rho_authority_binds_adv_once_at_pi_ccs_input() {
    let prep = wide_preprocessing();
    let scheme = lane_scheme(&prep);
    let with_adv = adv_instance(&prep, &scheme, 3);
    let mut without_adv = with_adv.clone();
    without_adv.claim.adv = None;

    assert_ne!(
        neo_fold_clean::paper::digest::ccs_claim_digest(&with_adv.claim),
        neo_fold_clean::paper::digest::ccs_claim_digest(&without_adv.claim),
        "Π_CCS input authority must bind the tuple before its challenges"
    );
    let (_, proof_a) = prove_fold(&prep, &scheme, vec![with_adv]).expect("adv fold");

    // Π_CCS.V constrains output adv to the already-bound input adv. Removing
    // only that deterministic forwarding from a copied output therefore does
    // not alter the later digest of the new evaluation messages.
    let mut outputs_without = proof_a.pi_ccs.outputs.clone();
    for output in &mut outputs_without {
        output.adv = None;
    }
    assert_eq!(
        neo_fold_clean::paper::digest::pi_ccs_outputs_digest(&proof_a.pi_ccs.outputs),
        neo_fold_clean::paper::digest::pi_ccs_outputs_digest(&outputs_without),
        "pre-ρ output digest must not redundantly absorb forwarded adv"
    );
}
