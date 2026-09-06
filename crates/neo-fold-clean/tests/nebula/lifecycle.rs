//! Nebula through the lifecycle: a chain with a `NebulaConfig` runs the
//! lane transition on every extend, carries the
//! lane through `x_out`, closes its segment, survives finalization, and
//! is accepted by the audit verifier — which replays the identical
//! transition, checks the terminal slice openings, and applies the
//! finalization rule.

use neo_ajtai::Commitment;
use neo_ccs::LaneCommitments;
use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_fold_clean::config;
use neo_fold_clean::frontends::nebula::layout::StepPublicInput;
use neo_fold_clean::lifecycle::{
    self, extend, extend_nebula_open, finish_uncompressed_with_audit, preprocess, verify_uncompressed,
    verify_uncompressed_audit, Error, Preprocessing, UncompressedAudit,
};
use neo_fold_clean::paper::construction2::{NebulaConfig, StackShape};
use neo_fold_clean::paper::digest;
use neo_fold_clean::paper::relations::{CcsInstance, LaneRanges, LaneScheme};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

#[path = "../support/mod.rs"]
mod support;

/// Steps per segment for this fixture.
const N: u64 = 2;
/// `x = [1 ‖ 1,400 bits ‖ 3 zero completion coefficients]` occupies the
/// complete public ring columns `[0, 26)`. The three lanes follow on whole
/// columns.
const M_IN: usize = 26 * D;
const LANE_COLS: LaneRanges = LaneRanges {
    ops: 26..27,
    is: 27..28,
    fs: 28..29,
};
/// Total width: x region + 3 lane columns.
const M: usize = 29 * D;

fn wide_preprocessing() -> Preprocessing {
    let structure =
        CcsStructure::new(vec![Mat::identity(M)], SparsePoly::new(1, vec![])).expect("lifecycle test structure");
    let params = config::r1cs_params(structure.n, structure.m).expect("params");
    support::install_ajtai_module(&params, &structure);
    preprocess(params, structure, Some(M_IN)).expect("preprocessing")
}

fn lane_scheme(prep: &Preprocessing) -> LaneScheme {
    LaneScheme::from_seeds(prep.params().kappa() as usize, LANE_COLS, [0xA7; 32], [0x7A; 32]).expect("scheme")
}

/// Deterministic lane bits for one step: the tail 3·54 slots of `z`.
fn lane_bits(step: u64) -> Vec<F> {
    (0..(3 * D) as u64)
        .map(|i| {
            F::from_u64(
                (step
                    .wrapping_mul(0x9E37)
                    .wrapping_add(i)
                    .rotate_left((i % 13) as u32))
                    & 1,
            )
        })
        .collect()
}

/// Phase 1 of the two-pass discipline: the lane commitments and
/// their `D_pre` chains, before any x (hence any γ) exists. Only the lane
/// columns matter to `LaneScheme::commit`, so a zero-x assignment yields
/// the same tuples the real instances will carry.
fn precommit(prep: &Preprocessing, scheme: &LaneScheme) -> (Vec<LaneCommitments<Commitment>>, [[F; 4]; 3]) {
    let mut advs = Vec::new();
    for step in 0..N {
        let mut z = vec![F::ZERO; M];
        z[26 * D..].copy_from_slice(&lane_bits(step));
        let inst =
            CcsInstance::from_low_norm_assignment(prep.params(), prep.commitment_scheme(), prep.structure(), &z, M_IN)
                .expect("dummy");
        advs.push(scheme.commit(&inst.witness.Z).expect("lane commit"));
    }
    let mem = digest::nebula_chain_mem_header();
    let mut chains = [digest::nebula_chain_ops_header(), mem, mem];
    let tags: [&[u8]; 3] = [
        digest::NEBULA_CHAIN_OPS_TAG,
        digest::NEBULA_CHAIN_MEM_TAG,
        digest::NEBULA_CHAIN_MEM_TAG,
    ];
    for adv in &advs {
        let leaves = digest::nebula_lane_leaf_digests(adv);
        for lane_id in 0..3 {
            chains[lane_id] = digest::nebula_chain_link(&chains[lane_id], tags[lane_id], &leaves[lane_id]);
        }
    }
    (advs, chains)
}

/// A full Nebula preprocessing whose `d_init` matches the fixture's
/// segment-0 IS chain (boundary continuity for the one-segment chain).
fn nebula_preprocessing() -> (Preprocessing, Vec<LaneCommitments<Commitment>>, [[F; 4]; 3]) {
    let prep = wide_preprocessing();
    let scheme = lane_scheme(&prep);
    let (advs, d_pre) = precommit(&prep, &scheme);
    let cfg = NebulaConfig {
        scheme,
        steps_per_segment: N,
        seg_max: 1,
        stacks: StackShape::NONE,
        initial_semantic_state_digest: [F::from_u64(10); 4],
        plan_digest: [F::from_u64(11); 4],
        d_init: d_pre[1],
    };
    (prep.with_nebula(cfg), advs, d_pre)
}

/// Phase 2: the real instance for one step — γ and the running products
/// in `x`, the precommitted lane bits in the lane columns.
fn step_instance(prep: &Preprocessing, gamma: [K; 2], step: u64, adv: &LaneCommitments<Commitment>) -> CcsInstance {
    let x = StepPublicInput {
        seg_idx: 0,
        idx: step,
        ts_in: step,
        ts_out: step + 1,
        gamma,
        h_in: [K::ONE; 4],
        h_out: [K::ONE; 4],
        sp_in: [0; 2],
        sp_out: [0; 2],
    };
    let bits = x.encode(StackShape::NONE).expect("x encode");
    let mut z = vec![F::ZERO; M];
    z[0] = F::ONE;
    z[1..1 + bits.len()].copy_from_slice(&bits);
    z[26 * D..].copy_from_slice(&lane_bits(step));
    let mut inst =
        CcsInstance::from_low_norm_assignment(prep.params(), prep.commitment_scheme(), prep.structure(), &z, M_IN)
            .expect("instance");
    inst.claim.adv = Some(adv.clone());
    inst
}

/// γ exactly as `open_segment` will squeeze it at the chain's first
/// extend. This is the segment prover's pre-derivation.
fn derive_gamma(prep: &Preprocessing, audit: &UncompressedAudit, d_pre: [[F; 4]; 3]) -> [K; 2] {
    let state = &audit.proof.state;
    let mut lane = state.nebula.clone().expect("nebula chain carries a lane");
    lane.open_segment(
        prep.nebula().expect("config"),
        prep.verifier_key().digest(),
        state.z_i,
        state.acc_digest,
        d_pre,
    )
    .expect("gamma derivation");
    lane.gamma.expect("squeezed")
}

/// Run the honest one-segment chain to a finalized audit.
fn honest_chain() -> (Preprocessing, UncompressedAudit) {
    let (prep, advs, d_pre) = nebula_preprocessing();
    let audit = lifecycle::prove(&prep, Vec::<Vec<CcsInstance>>::new()).expect("base");
    let gamma = derive_gamma(&prep, &audit, d_pre);

    let audit = extend_nebula_open(&prep, audit, vec![step_instance(&prep, gamma, 0, &advs[0])], d_pre)
        .expect("segment-open extend");
    let audit = extend(&prep, audit, vec![step_instance(&prep, gamma, 1, &advs[1])]).expect("mid-segment extend");

    let lane = audit.proof.state.nebula.as_ref().expect("lane carried");
    assert!(lane.is_closed(), "segment must close at N steps");
    assert_eq!(lane.seg_idx, 1);
    assert_eq!(lane.ts, N, "global ts carried through close");

    let audit = finish_uncompressed_with_audit(&prep, audit).expect("finalize");
    (prep, audit)
}

/// The full loop: honest segment proves, finalizes, and verifies —
/// including the audit replay of every lane transition, the finalization
/// rule, and the terminal lane slice openings.
#[test]
fn nebula_chain_proves_and_verifies_end_to_end() {
    let (prep, audit) = honest_chain();
    verify_uncompressed_audit(&prep, &audit).expect("audit verification");
}

#[test]
fn terminal_verifier_rejects_another_program_binding() {
    let (prep, mut audit) = honest_chain();
    audit
        .proof
        .state
        .nebula
        .as_mut()
        .expect("Nebula lane")
        .program_binding_digest[0] += F::ONE;
    assert!(matches!(
        verify_uncompressed(&prep, &audit.proof),
        Err(Error::NebulaProgramBindingMismatch)
    ));
}

/// A chain that stops mid-segment is
/// prover resume material, never an externally accepted proof.
#[test]
fn mid_segment_terminal_state_is_rejected() {
    let (prep, advs, d_pre) = nebula_preprocessing();
    let audit = lifecycle::prove(&prep, Vec::<Vec<CcsInstance>>::new()).expect("base");
    let gamma = derive_gamma(&prep, &audit, d_pre);
    let audit = extend_nebula_open(&prep, audit, vec![step_instance(&prep, gamma, 0, &advs[0])], d_pre)
        .expect("segment-open extend");
    let audit = finish_uncompressed_with_audit(&prep, audit).expect("finalize");
    assert!(matches!(
        verify_uncompressed_audit(&prep, &audit),
        Err(Error::NebulaSegmentOpenAtTerminal)
    ));
}

/// The prover-side transition fails at the named lane-transition check: folding a
/// step before any segment opened.
#[test]
fn extend_before_open_is_rejected() {
    let (prep, advs, d_pre) = nebula_preprocessing();
    let audit = lifecycle::prove(&prep, Vec::<Vec<CcsInstance>>::new()).expect("base");
    let gamma = derive_gamma(&prep, &audit, d_pre);
    let result = extend(&prep, audit, vec![step_instance(&prep, gamma, 0, &advs[0])]);
    assert!(result.is_err(), "advance before open_segment must fail");
}

/// A claim whose x carries the wrong γ is caught by the per-step lane
/// equality at extend time (commit-then-challenge, prover self-check).
#[test]
fn extend_rejects_wrong_gamma_in_x() {
    let (prep, advs, d_pre) = nebula_preprocessing();
    let audit = lifecycle::prove(&prep, Vec::<Vec<CcsInstance>>::new()).expect("base");
    let mut gamma = derive_gamma(&prep, &audit, d_pre);
    gamma[0] += K::ONE;
    let result = extend_nebula_open(&prep, audit, vec![step_instance(&prep, gamma, 0, &advs[0])], d_pre);
    assert!(result.is_err(), "γ mismatch must fail the lane equality");
}

/// A lane-bit flip in a final folded witness fails the terminal
/// slice-opening even though every digest was recomputed consistently.
#[test]
fn tampered_terminal_witness_fails_slice_opening() {
    let (prep, mut audit) = honest_chain();
    let neo_fold_clean::paper::construction2::ProofState::Active { running, .. } = &mut audit.proof.state.proof else {
        panic!("finalized chain is Active");
    };
    // Flip one lane-column coefficient in one terminal witness.
    let witness = &mut running.witnesses[0];
    let flip = (0, 26); // row 0, first ops-lane column
    witness[flip] += F::ONE;
    let result = verify_uncompressed_audit(&prep, &audit);
    assert!(result.is_err(), "lane-bit flip must be rejected");
}
