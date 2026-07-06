//! Nebula lane-transition gadgets — parity against the native §6.3
//! transition and its `paper::digest` material (spec §13 step 9).
//!
//! The native side (`NebulaLane` + `digest::nebula_*`) is called
//! directly, never inlined: the in-circuit mirrors must produce
//! byte-identical digests and the identical carried state for the same
//! inputs, and every close check must fail as a row when its native
//! twin would reject.

use neo_ajtai::Commitment;
use neo_ccs::LaneCommitments;
use neo_fold_clean::engine::r1cs_circuit::field_ext::KVar;
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, Var};
use neo_fold_clean::paper::construction2::{NebulaConfig, NebulaLane, NebulaStepX, StackShape};
use neo_fold_clean::paper::digest;
use neo_fold_clean::paper::f_prime::nebula_lane_circuit::{
    enforce_nebula_advance_circuit, enforce_nebula_chain_link_circuit, enforce_nebula_close_circuit,
    enforce_nebula_lane_digest_circuit, enforce_nebula_lane_leaf_digests_circuit, GammaWires, NebulaLaneWires,
    NebulaStepXWires,
};
use neo_fold_clean::paper::relations::{LaneRanges, LaneScheme};
use neo_math::field::KExtensions;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

/// Steps per segment for these tests (spec `N`, tiny).
const N: u64 = 2;

// ── Native fixture (mirrors tests/nebula/lane.rs) ────────────────────────

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

fn honest_d_pre(advs: &[LaneCommitments<Commitment>]) -> [[F; 4]; 3] {
    digest::nebula_lane_chains(advs.iter())
}

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
        stacks: StackShape::NONE,
        plan_digest: [F::from_u64(7); 4],
        d_init,
    }
}

fn honest_x(lane: &NebulaLane) -> NebulaStepX {
    NebulaStepX {
        seg_idx: lane.seg_idx,
        idx: lane.idx,
        ts_in: lane.ts,
        ts_out: lane.ts + 1,
        gamma: lane.gamma.expect("segment open"),
        h_in: lane.h,
        h_out: [K::ONE; 4],
        sp_in: [0; 2],
        sp_out: [0; 2],
    }
}

// ── Wire allocation helpers ──────────────────────────────────────────────

fn alloc_digest(b: &mut R1csBuilder, values: [F; 4]) -> [Var; 4] {
    values.map(|v| b.alloc(v))
}

fn alloc_k(b: &mut R1csBuilder, value: K) -> KVar {
    let (c0, c1) = value.to_limbs_u64();
    KVar::alloc(b, F::from_u64(c0), F::from_u64(c1))
}

fn alloc_lane(b: &mut R1csBuilder, lane: &NebulaLane) -> NebulaLaneWires {
    let gamma = lane.gamma.unwrap_or([K::ONE; 2]);
    NebulaLaneWires {
        seg_idx: b.alloc(F::from_u64(lane.seg_idx)),
        idx: b.alloc(F::from_u64(lane.idx)),
        ts: b.alloc(F::from_u64(lane.ts)),
        gamma: gamma.map(|g| alloc_k(b, g)),
        h: lane.h.map(|h| alloc_k(b, h)),
        sp: lane.sp.map(|s| b.alloc(F::from_u64(s))),
        d_pre: lane.d_pre.map(|d| alloc_digest(b, d)),
        d_seen: lane.d_seen.map(|d| alloc_digest(b, d)),
        d_mem: alloc_digest(b, lane.d_mem),
    }
}

fn alloc_x(b: &mut R1csBuilder, x: &NebulaStepX) -> NebulaStepXWires {
    NebulaStepXWires {
        seg_idx: b.alloc(F::from_u64(x.seg_idx)),
        idx: b.alloc(F::from_u64(x.idx)),
        ts_in: b.alloc(F::from_u64(x.ts_in)),
        ts_out: b.alloc(F::from_u64(x.ts_out)),
        gamma: x.gamma.map(|g| alloc_k(b, g)),
        h_in: x.h_in.map(|h| alloc_k(b, h)),
        h_out: x.h_out.map(|h| alloc_k(b, h)),
        sp_in: x.sp_in.map(|s| b.alloc(F::from_u64(s))),
        sp_out: x.sp_out.map(|s| b.alloc(F::from_u64(s))),
    }
}

fn alloc_adv_leaves(b: &mut R1csBuilder, tuple: &LaneCommitments<Commitment>) -> [[Var; 4]; 3] {
    let ops: Vec<Var> = tuple.ops.data.iter().map(|&v| b.alloc(v)).collect();
    let is: Vec<Var> = tuple.is.data.iter().map(|&v| b.alloc(v)).collect();
    let fs: Vec<Var> = tuple.fs.data.iter().map(|&v| b.alloc(v)).collect();
    enforce_nebula_lane_leaf_digests_circuit(b, tuple.ops.d, tuple.ops.kappa, &ops, &is, &fs)
}

fn extract_digest(b: &R1csBuilder, vars: [Var; 4]) -> [F; 4] {
    vars.map(|v| b.witness()[v.col()])
}

fn assert_lane_parity(b: &R1csBuilder, wires: &NebulaLaneWires, native: &NebulaLane) {
    let value = |v: Var| b.witness()[v.col()];
    assert_eq!(value(wires.seg_idx), F::from_u64(native.seg_idx));
    assert_eq!(value(wires.idx), F::from_u64(native.idx));
    assert_eq!(value(wires.ts), F::from_u64(native.ts));
    for (wire, sp) in wires.sp.iter().zip(native.sp.iter()) {
        assert_eq!(value(*wire), F::from_u64(*sp));
    }
    for (wire, h) in wires.h.iter().zip(native.h.iter()) {
        let (c0, c1) = h.to_limbs_u64();
        assert_eq!(value(wire.c0), F::from_u64(c0));
        assert_eq!(value(wire.c1), F::from_u64(c1));
    }
    for (wire, native) in wires.d_pre.iter().zip(native.d_pre.iter()) {
        assert_eq!(extract_digest(b, *wire), *native);
    }
    for (wire, native) in wires.d_seen.iter().zip(native.d_seen.iter()) {
        assert_eq!(extract_digest(b, *wire), *native);
    }
    assert_eq!(extract_digest(b, wires.d_mem), native.d_mem);
}

// ── Poseidon2 mirror parity ──────────────────────────────────────────────

#[test]
fn leaf_digests_match_native() {
    let tuple = adv(0);
    let native = digest::nebula_lane_leaf_digests(&tuple);

    let mut b = R1csBuilder::new();
    let leaves = alloc_adv_leaves(&mut b, &tuple);
    for (circuit, native) in leaves.iter().zip(native.iter()) {
        assert_eq!(extract_digest(&b, *circuit), *native);
    }
    assert!(b.is_satisfied(), "honest leaf traces satisfy the rows");
}

#[test]
fn chain_link_matches_native_for_both_tags() {
    let prev = digest::nebula_chain_ops_header();
    let leaf = digest::nebula_chain_mem_header(); // any 4-lane value works
    for tag in [digest::NEBULA_CHAIN_OPS_TAG, digest::NEBULA_CHAIN_MEM_TAG] {
        let native = digest::nebula_chain_link(&prev, tag, &leaf);
        let mut b = R1csBuilder::new();
        let prev_w = alloc_digest(&mut b, prev);
        let leaf_w = alloc_digest(&mut b, leaf);
        let out = enforce_nebula_chain_link_circuit(&mut b, prev_w, tag, leaf_w);
        assert_eq!(extract_digest(&b, out), native);
        assert!(b.is_satisfied());
    }
}

#[test]
fn lane_digest_matches_native_for_both_gamma_shapes() {
    let advs: Vec<_> = (0..N).map(adv).collect();
    let d_pre = honest_d_pre(&advs);
    let cfg = config(d_pre[1]);

    // γ = ⊥ (base lane).
    let base = NebulaLane::base(&cfg);
    let mut b = R1csBuilder::new();
    let wires = alloc_lane(&mut b, &base);
    let out = enforce_nebula_lane_digest_circuit(&mut b, &wires, GammaWires::Absent);
    assert_eq!(extract_digest(&b, out), base.digest());
    assert!(b.is_satisfied());

    // γ present (opened segment).
    let mut opened = base;
    opened
        .open_segment(&cfg, [3; 32], [4; 32], [5; 32], d_pre)
        .expect("open");
    let mut b = R1csBuilder::new();
    let wires = alloc_lane(&mut b, &opened);
    let gamma = wires.gamma;
    let out = enforce_nebula_lane_digest_circuit(&mut b, &wires, GammaWires::Present(gamma));
    assert_eq!(extract_digest(&b, out), opened.digest());
    assert!(b.is_satisfied());
}

// ── The §6.3 transition, mirrored end to end ─────────────────────────────

/// Walk one honest N = 2 segment natively and in-circuit side by side:
/// advance (mid-segment), then advance + close. Every carried coordinate
/// and every digest must agree at every point, and the composed circuit
/// must be satisfied.
#[test]
fn advance_and_close_mirror_the_native_transition() {
    let advs: Vec<_> = (0..N).map(adv).collect();
    let d_pre = honest_d_pre(&advs);
    let cfg = config(d_pre[1]); // boundary: segment 0's IS chain is D_init
    let mut native = NebulaLane::base(&cfg);
    native
        .open_segment(&cfg, [3; 32], [4; 32], [5; 32], d_pre)
        .expect("open");

    let mut b = R1csBuilder::new();
    let mut wires = alloc_lane(&mut b, &native);
    let d_mem_in = wires.d_mem;

    // Step 1: mid-segment advance.
    let x1 = honest_x(&native);
    native
        .advance(&cfg, &x1, Some(&advs[0]))
        .expect("native advance 1");
    let x1_wires = alloc_x(&mut b, &x1);
    let leaves1 = alloc_adv_leaves(&mut b, &advs[0]);
    wires = enforce_nebula_advance_circuit(&mut b, &wires, &x1_wires, leaves1);
    assert_lane_parity(&b, &wires, &native);

    // Step 2: the segment's N-th claim — advance, then close.
    let x2 = honest_x(&native);
    native
        .advance(&cfg, &x2, Some(&advs[1]))
        .expect("native advance 2 + close");
    assert!(native.is_closed(), "native segment closed");
    let x2_wires = alloc_x(&mut b, &x2);
    let leaves2 = alloc_adv_leaves(&mut b, &advs[1]);
    wires = enforce_nebula_advance_circuit(&mut b, &wires, &x2_wires, leaves2);
    wires = enforce_nebula_close_circuit(&mut b, &wires);
    assert_lane_parity(&b, &wires, &native);

    // The closed lane's compact handle also agrees (γ back to ⊥).
    let handle = enforce_nebula_lane_digest_circuit(&mut b, &wires, GammaWires::Absent);
    assert_eq!(extract_digest(&b, handle), native.digest());

    assert!(b.is_satisfied(), "the honest two-step segment satisfies every row");

    // Rejection sweep: tampering any load-bearing input wire breaks a
    // row; restoring it restores satisfaction (so each rejection is
    // attributable to exactly the tampered value).
    let mut expect_reject = |col: usize, bad: F, restore: F, what: &str| {
        b.tamper_witness(col, bad);
        assert!(!b.is_satisfied(), "{what} must be enforced as rows");
        b.tamper_witness(col, restore);
        assert!(b.is_satisfied(), "restoring after `{what}` must re-satisfy");
    };
    // (a) x.ts_in continuity (the advance equality).
    expect_reject(
        x1_wires.ts_in.col(),
        F::from_u64(99),
        F::from_u64(x1.ts_in),
        "ts continuity",
    );
    // (b) γ equality against the lane.
    let (g0, _) = x1.gamma[0].to_limbs_u64();
    expect_reject(
        x1_wires.gamma[0].c0.col(),
        F::from_u64(99),
        F::from_u64(g0),
        "γ equality",
    );
    // (c) sp = 0 at close (v3.1 segment-local stacks).
    expect_reject(x2_wires.sp_out[0].col(), F::ONE, F::ZERO, "sp = 0 at close");
    // (d) the product equation at close (h_fs coefficient).
    expect_reject(
        x2_wires.h_out[3].c0.col(),
        F::from_u64(2),
        F::ONE,
        "the product equation",
    );
    // (e) the boundary handoff `D_seen[is] == D_mem` at close.
    expect_reject(
        d_mem_in[0].col(),
        F::from_u64(99),
        cfg.d_init[0],
        "the boundary handoff",
    );
    // (f) a leaf digest lane: the chain links inherit the change and
    //     D_seen diverges from D_pre at close.
    let leaf_native = digest::nebula_lane_leaf_digests(&advs[1])[0][0];
    expect_reject(
        leaves2[0][0].col(),
        F::from_u64(99),
        leaf_native,
        "the leaf-chain binding",
    );
}
