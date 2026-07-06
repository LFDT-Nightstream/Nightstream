//! `NebulaLane` transition — spec §6.3, one rejection test per check
//! (M2 acceptance, spec §13 step 4) plus the honest segment walk and γ
//! determinism. Pure state-machine tests: folding is exercised by
//! `nebula_adv_fold`, lifecycle wiring by M2b's tests.

use neo_ajtai::Commitment;
use neo_ccs::LaneCommitments;
use neo_fold_clean::paper::construction2::{NebulaConfig, NebulaError, NebulaLane, NebulaStepX};
use neo_fold_clean::paper::digest;
use neo_fold_clean::paper::relations::{LaneRanges, LaneScheme};
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

/// Steps per segment for these tests (spec `N`, tiny).
const N: u64 = 3;

fn commitment(seed: u64) -> Commitment {
    Commitment {
        d: 54,
        kappa: 2,
        data: (0..108u64)
            .map(|i| F::from_u64(seed.wrapping_mul(0x9E37).wrapping_add(i)))
            .collect(),
    }
}

fn adv(step: u64) -> LaneCommitments<Commitment> {
    LaneCommitments {
        ops: commitment(step * 3),
        is: commitment(step * 3 + 1),
        fs: commitment(step * 3 + 2),
    }
}

/// The honest `D_pre` chains for the segment's adv tuples — the same
/// leaf/link/header formulas the lane itself applies.
fn honest_d_pre(advs: &[LaneCommitments<Commitment>]) -> [[F; 4]; 3] {
    let mem = digest::nebula_chain_mem_header();
    let mut chains = [digest::nebula_chain_ops_header(), mem, mem];
    let tags: [&[u8]; 3] = [
        digest::NEBULA_CHAIN_OPS_TAG,
        digest::NEBULA_CHAIN_MEM_TAG,
        digest::NEBULA_CHAIN_MEM_TAG,
    ];
    for tuple in advs {
        let leaves = digest::nebula_lane_leaf_digests(tuple);
        for lane_id in 0..3 {
            chains[lane_id] = digest::nebula_chain_link(&chains[lane_id], tags[lane_id], &leaves[lane_id]);
        }
    }
    chains
}

/// A config whose `d_init` matches the segment's honest IS chain, so the
/// boundary check can close segment 0.
fn config(d_init: [F; 4]) -> NebulaConfig {
    let scheme = LaneScheme::from_seeds(
        2,
        LaneRanges {
            ops: 1..2,
            is: 2..3,
            fs: 3..4,
        },
        [1; 32],
        [2; 32],
    )
    .expect("test lane scheme");
    NebulaConfig {
        scheme,
        steps_per_segment: N,
        plan_digest: [F::from_u64(7); 4],
        d_init,
    }
}

fn open(lane: &mut NebulaLane, cfg: &NebulaConfig, d_pre: [[F; 4]; 3]) {
    lane.open_segment(cfg, [3; 32], [4; 32], [5; 32], d_pre)
        .expect("open");
}

/// The honest x for the lane's current position: products stay at `1_K`
/// throughout (balance holds trivially: 1·1 == 1·1), timestamps advance
/// by one per step.
fn honest_x(lane: &NebulaLane) -> NebulaStepX {
    NebulaStepX {
        seg_idx: lane.seg_idx,
        idx: lane.idx,
        ts_in: lane.ts,
        ts_out: lane.ts + 1,
        gamma: lane.gamma.expect("segment open"),
        h_in: lane.h,
        h_out: [K::ONE; 4],
    }
}

/// Walk one honest segment to close; returns the lane afterwards.
fn walk_honest_segment() -> (NebulaConfig, NebulaLane) {
    let advs: Vec<_> = (0..N).map(adv).collect();
    let d_pre = honest_d_pre(&advs);
    let cfg = config(d_pre[1]); // boundary: segment 0's IS chain must equal D_init
    let mut lane = NebulaLane::base(&cfg);
    open(&mut lane, &cfg, d_pre);
    for tuple in &advs {
        let x = honest_x(&lane);
        lane.advance(&cfg, &x, Some(tuple)).expect("honest advance");
    }
    (cfg, lane)
}

#[test]
fn honest_segment_closes_and_resets_without_ts() {
    let (_, lane) = walk_honest_segment();
    assert_eq!(lane.seg_idx, 1, "close must advance the segment counter");
    assert_eq!(lane.idx, 0);
    assert_eq!(lane.gamma, None, "γ must reset to ⊥");
    assert_eq!(lane.ts, N, "the global timestamp must NOT reset");
    assert!(lane.is_closed(), "a closed lane satisfies the finalization rule");
    // The boundary handle now carries segment 0's FS chain.
    let advs: Vec<_> = (0..N).map(adv).collect();
    assert_eq!(lane.d_mem, honest_d_pre(&advs)[2]);
}

#[test]
fn gamma_is_deterministic_and_binds_d_pre() {
    let advs: Vec<_> = (0..N).map(adv).collect();
    let d_pre = honest_d_pre(&advs);
    let cfg = config(d_pre[1]);

    let mut a = NebulaLane::base(&cfg);
    let mut b = NebulaLane::base(&cfg);
    open(&mut a, &cfg, d_pre);
    open(&mut b, &cfg, d_pre);
    assert_eq!(a.gamma, b.gamma, "same open context must squeeze the same γ");

    let mut c = NebulaLane::base(&cfg);
    let mut other = d_pre;
    other[0][0] += F::ONE;
    open(&mut c, &cfg, other);
    assert_ne!(
        a.gamma, c.gamma,
        "γ must bind the claimed D_pre (commit-then-challenge)"
    );
}

#[test]
fn open_twice_is_rejected() {
    let (_, _) = walk_honest_segment(); // sanity: the honest path exists
    let d_pre = honest_d_pre(&(0..N).map(adv).collect::<Vec<_>>());
    let cfg = config(d_pre[1]);
    let mut lane = NebulaLane::base(&cfg);
    open(&mut lane, &cfg, d_pre);
    assert_eq!(
        lane.open_segment(&cfg, [3; 32], [4; 32], [5; 32], d_pre),
        Err(NebulaError::SegmentAlreadyOpen)
    );
}

#[test]
fn advance_before_open_is_rejected() {
    let d_pre = honest_d_pre(&(0..N).map(adv).collect::<Vec<_>>());
    let cfg = config(d_pre[1]);
    let mut lane = NebulaLane::base(&cfg);
    let x = NebulaStepX {
        seg_idx: 0,
        idx: 0,
        ts_in: 0,
        ts_out: 1,
        gamma: [K::ONE; 2],
        h_in: [K::ONE; 4],
        h_out: [K::ONE; 4],
    };
    assert_eq!(lane.advance(&cfg, &x, Some(&adv(0))), Err(NebulaError::SegmentNotOpen));
}

#[test]
fn per_step_equalities_reject_tampered_x() {
    let advs: Vec<_> = (0..N).map(adv).collect();
    let d_pre = honest_d_pre(&advs);
    let cfg = config(d_pre[1]);
    let mut lane = NebulaLane::base(&cfg);
    open(&mut lane, &cfg, d_pre);

    let honest = honest_x(&lane);
    let tuple = adv(0);

    let mut x = honest.clone();
    x.idx += 1;
    assert!(matches!(
        lane.clone().advance(&cfg, &x, Some(&tuple)),
        Err(NebulaError::CounterMismatch { .. })
    ));

    let mut x = honest.clone();
    x.ts_in += 5;
    assert!(matches!(
        lane.clone().advance(&cfg, &x, Some(&tuple)),
        Err(NebulaError::TsMismatch { .. })
    ));

    let mut x = honest.clone();
    x.gamma[0] += K::ONE;
    assert_eq!(
        lane.clone().advance(&cfg, &x, Some(&tuple)),
        Err(NebulaError::GammaMismatch)
    );

    let mut x = honest.clone();
    x.h_in[0] += K::ONE;
    assert_eq!(
        lane.clone().advance(&cfg, &x, Some(&tuple)),
        Err(NebulaError::ProductThreadMismatch)
    );

    assert_eq!(lane.clone().advance(&cfg, &honest, None), Err(NebulaError::MissingAdv));
}

/// Fold a different tuple than the pre-committed one: the segment closes
/// on `D_seen != D_pre` — the retroactive authority of the L0b claim.
#[test]
fn close_rejects_swapped_lane_commitments() {
    let advs: Vec<_> = (0..N).map(adv).collect();
    let d_pre = honest_d_pre(&advs);
    let cfg = config(d_pre[1]);
    let mut lane = NebulaLane::base(&cfg);
    open(&mut lane, &cfg, d_pre);

    for (i, tuple) in advs.iter().enumerate() {
        let x = honest_x(&lane);
        let swapped = adv(99); // not the pre-committed tuple
        let result = if i as u64 == N - 1 {
            lane.advance(&cfg, &x, Some(&swapped))
        } else {
            lane.advance(&cfg, &x, Some(tuple))
        };
        if i as u64 == N - 1 {
            assert_eq!(result, Err(NebulaError::PreSeenMismatch));
            return;
        }
        result.expect("honest prefix");
    }
    unreachable!("close must have run");
}

/// Unbalanced products at close: the Nebula multiset equation rejects.
#[test]
fn close_rejects_unbalanced_products() {
    let advs: Vec<_> = (0..N).map(adv).collect();
    let d_pre = honest_d_pre(&advs);
    let cfg = config(d_pre[1]);
    let mut lane = NebulaLane::base(&cfg);
    open(&mut lane, &cfg, d_pre);

    for (i, tuple) in advs.iter().enumerate() {
        let mut x = honest_x(&lane);
        if i as u64 == N - 1 {
            x.h_out = [K::ONE, K::ONE, K::ONE, K::ONE + K::ONE]; // h_fs ≠ balance
            assert_eq!(lane.advance(&cfg, &x, Some(tuple)), Err(NebulaError::ProductEquation));
            return;
        }
        lane.advance(&cfg, &x, Some(tuple)).expect("honest prefix");
    }
}

/// Fresh memory at segment start (wrong `D_init`): boundary check rejects.
#[test]
fn close_rejects_broken_memory_continuity() {
    let advs: Vec<_> = (0..N).map(adv).collect();
    let d_pre = honest_d_pre(&advs);
    let mut wrong_init = d_pre[1];
    wrong_init[0] += F::ONE;
    let cfg = config(wrong_init);
    let mut lane = NebulaLane::base(&cfg);
    open(&mut lane, &cfg, d_pre);

    for (i, tuple) in advs.iter().enumerate() {
        let x = honest_x(&lane);
        let result = lane.advance(&cfg, &x, Some(tuple));
        if i as u64 == N - 1 {
            assert_eq!(result, Err(NebulaError::BoundaryMismatch));
            return;
        }
        result.expect("honest prefix");
    }
}

/// Mid-segment lanes violate the finalization rule (spec §6.3): a proof
/// may not end here.
#[test]
fn finalization_rule_rejects_open_segments() {
    let advs: Vec<_> = (0..N).map(adv).collect();
    let d_pre = honest_d_pre(&advs);
    let cfg = config(d_pre[1]);
    let mut lane = NebulaLane::base(&cfg);
    assert!(lane.is_closed(), "base lane is a valid terminal state");
    open(&mut lane, &cfg, d_pre);
    assert!(!lane.is_closed(), "opened-but-unused segment is not terminal");
    let x = honest_x(&lane);
    lane.advance(&cfg, &x, Some(&adv(0))).expect("advance");
    assert!(!lane.is_closed(), "mid-segment is not terminal");
}
