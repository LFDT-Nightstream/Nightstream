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
use neo_fold_clean::frontends::nebula::layout::encode_delayed_f_prime_suffix;
use neo_fold_clean::paper::construction2::{NebulaConfig, NebulaLane, NebulaStepX, StackShape};
use neo_fold_clean::paper::digest;
use neo_fold_clean::paper::f_prime::nebula_lane_circuit::{
    alloc_nebula_lane_wires, decode_delayed_nebula_public_suffix_circuit, decode_nebula_step_x_bits_circuit,
    enforce_delayed_nebula_claim_circuit, enforce_nebula_advance_circuit, enforce_nebula_chain_link_circuit,
    enforce_nebula_close_circuit, enforce_nebula_lane_constant_circuit, enforce_nebula_lane_digest_circuit,
    enforce_nebula_lane_digest_selected_circuit, enforce_nebula_lane_leaf_digests_circuit,
    enforce_nebula_maybe_close_circuit, enforce_nebula_maybe_open_circuit, GammaWires, NebulaLaneWires,
    NebulaOpenContextWires, NebulaStepXWires,
};
use neo_fold_clean::paper::relations::product_commitment_circuit::alloc_adv;
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
        seg_max: 1,
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
        open: b.alloc(if lane.gamma.is_some() { F::ONE } else { F::ZERO }),
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
    assert_eq!(value(wires.open), if native.gamma.is_some() { F::ONE } else { F::ZERO });
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

#[test]
fn step_x_bit_decoder_matches_native_layout_and_rejects_non_bits() {
    let stacks = StackShape { count: 2, sigma: 5 };
    let k = |a, b| K::from_coeffs([F::from_u64(a), F::from_u64(b)]);
    let native = NebulaStepX {
        seg_idx: 9,
        idx: 17,
        ts_in: 123,
        ts_out: 129,
        gamma: [k(3, 5), k(7, 11)],
        h_in: [k(13, 17), k(19, 23), k(29, 31), k(37, 41)],
        h_out: [k(43, 47), k(53, 59), k(61, 67), k(71, 73)],
        sp_in: [4, 9],
        sp_out: [5, 8],
    };
    let encoded = native.encode(stacks).expect("encode native step x");
    let mut builder = R1csBuilder::new();
    let bit_wires = builder.alloc_vec(&encoded);
    let decoded = decode_nebula_step_x_bits_circuit(&mut builder, &bit_wires, stacks).expect("decode circuit step x");
    let value = |wire: Var| builder.witness()[wire.col()];
    let kvar = |wire: KVar| K::from_coeffs([value(wire.c0), value(wire.c1)]);

    assert_eq!(value(decoded.seg_idx), F::from_u64(native.seg_idx));
    assert_eq!(value(decoded.idx), F::from_u64(native.idx));
    assert_eq!(value(decoded.ts_in), F::from_u64(native.ts_in));
    assert_eq!(value(decoded.ts_out), F::from_u64(native.ts_out));
    assert_eq!(decoded.gamma.map(kvar), native.gamma);
    assert_eq!(decoded.h_in.map(kvar), native.h_in);
    assert_eq!(decoded.h_out.map(kvar), native.h_out);
    assert_eq!(decoded.sp_in.map(value), native.sp_in.map(F::from_u64));
    assert_eq!(decoded.sp_out.map(value), native.sp_out.map(F::from_u64));
    assert!(builder.is_satisfied(), "honest encoded step x must satisfy");

    builder.tamper_witness(bit_wires[0].col(), F::from_u64(2));
    assert!(!builder.is_satisfied(), "non-bit public suffix coordinate must reject");
}

#[test]
fn delayed_suffix_decoder_binds_open_d_pre_and_canonicalizes_absence() {
    let stacks = StackShape::NONE;
    let step = NebulaStepX {
        seg_idx: 2,
        idx: 0,
        ts_in: 10,
        ts_out: 11,
        gamma: [K::ONE; 2],
        h_in: [K::ONE; 4],
        h_out: [K::ONE; 4],
        sp_in: [0; 2],
        sp_out: [0; 2],
    };
    let d_pre = std::array::from_fn(|digest| std::array::from_fn(|lane| F::from_u64(100 + (digest * 4 + lane) as u64)));
    let encoded = encode_delayed_f_prime_suffix(&step, stacks, Some(d_pre)).expect("encode delayed suffix");
    let mut builder = R1csBuilder::new();
    let suffix_wires = builder.alloc_vec(&encoded);
    let decoded = decode_delayed_nebula_public_suffix_circuit(&mut builder, &suffix_wires, stacks)
        .expect("decode delayed suffix");
    assert_eq!(builder.witness()[decoded.open.col()], F::ONE);
    for (decoded_digest, expected_digest) in decoded.d_pre.iter().zip(d_pre.iter()) {
        assert_eq!(extract_digest(&builder, *decoded_digest), *expected_digest);
    }
    assert!(builder.is_satisfied(), "present D_pre suffix must satisfy");

    let encoded = encode_delayed_f_prime_suffix(&step, stacks, None).expect("encode absent D_pre");
    let mut builder = R1csBuilder::new();
    let suffix_wires = builder.alloc_vec(&encoded);
    let decoded =
        decode_delayed_nebula_public_suffix_circuit(&mut builder, &suffix_wires, stacks).expect("decode absent suffix");
    assert_eq!(builder.witness()[decoded.open.col()], F::ZERO);
    assert!(builder.is_satisfied(), "canonical absent D_pre suffix must satisfy");
    let first_d_pre_bit = stacks.x_bits() + 1;
    builder.tamper_witness(suffix_wires[first_d_pre_bit].col(), F::ONE);
    assert!(!builder.is_satisfied(), "open=0 must force every D_pre bit to zero");
}

#[test]
fn maybe_open_replays_native_gamma_and_carries_open_segments() {
    let advs: Vec<_> = (0..N).map(adv).collect();
    let d_pre = honest_d_pre(&advs);
    let cfg = config(d_pre[1]);
    let vk = [3u8; 32];
    let z_i = [4u8; 32];
    let acc = [5u8; 32];
    let mut native = NebulaLane::base(&cfg);
    let first_x = NebulaStepX {
        seg_idx: 0,
        idx: 0,
        ts_in: 0,
        ts_out: 1,
        gamma: [K::ONE; 2],
        h_in: [K::ONE; 4],
        h_out: [K::ONE; 4],
        sp_in: [0; 2],
        sp_out: [0; 2],
    };
    let suffix = encode_delayed_f_prime_suffix(&first_x, cfg.stacks, Some(d_pre)).expect("encode open suffix");

    let mut builder = R1csBuilder::new();
    let lane = alloc_lane(&mut builder, &native);
    let suffix_wires = builder.alloc_vec(&suffix);
    let delayed = decode_delayed_nebula_public_suffix_circuit(&mut builder, &suffix_wires, cfg.stacks)
        .expect("decode open suffix");
    let context = NebulaOpenContextWires {
        vk_fs: alloc_digest(&mut builder, digest::digest32_as_fields(vk)),
        z_i: alloc_digest(&mut builder, digest::digest32_as_fields(z_i)),
        acc_digest: alloc_digest(&mut builder, digest::digest32_as_fields(acc)),
        plan_digest: alloc_digest(&mut builder, cfg.plan_digest),
    };
    let opened = enforce_nebula_maybe_open_circuit(&mut builder, &lane, &delayed, &context, cfg.seg_max);
    native
        .open_segment(&cfg, vk, z_i, acc, d_pre)
        .expect("native open");
    assert_lane_parity(&builder, &opened, &native);
    assert!(builder.is_satisfied(), "open branch must match native gamma transcript");

    let carry_suffix = encode_delayed_f_prime_suffix(&first_x, cfg.stacks, None).expect("encode carry suffix");
    let carry_wires = builder.alloc_vec(&carry_suffix);
    let carry_input = decode_delayed_nebula_public_suffix_circuit(&mut builder, &carry_wires, cfg.stacks)
        .expect("decode carry suffix");
    let carried = enforce_nebula_maybe_open_circuit(&mut builder, &opened, &carry_input, &context, cfg.seg_max);
    assert_lane_parity(&builder, &carried, &native);
    assert!(
        builder.is_satisfied(),
        "already-open branch must carry the lane unchanged"
    );
}

#[test]
fn maybe_open_rejects_a_segment_at_the_plan_limit() {
    let advs: Vec<_> = (0..N).map(adv).collect();
    let d_pre = honest_d_pre(&advs);
    let cfg = config(d_pre[1]);
    let mut exhausted = NebulaLane::base(&cfg);
    exhausted.seg_idx = cfg.seg_max;
    let x = NebulaStepX {
        seg_idx: exhausted.seg_idx,
        idx: 0,
        ts_in: 0,
        ts_out: 1,
        gamma: [K::ONE; 2],
        h_in: [K::ONE; 4],
        h_out: [K::ONE; 4],
        sp_in: [0; 2],
        sp_out: [0; 2],
    };
    let suffix = encode_delayed_f_prime_suffix(&x, cfg.stacks, Some(d_pre)).expect("encode open suffix");

    let mut builder = R1csBuilder::new();
    let lane = alloc_lane(&mut builder, &exhausted);
    let suffix_wires = builder.alloc_vec(&suffix);
    let delayed = decode_delayed_nebula_public_suffix_circuit(&mut builder, &suffix_wires, cfg.stacks)
        .expect("decode open suffix");
    let context = NebulaOpenContextWires {
        vk_fs: alloc_digest(&mut builder, [F::ZERO; 4]),
        z_i: alloc_digest(&mut builder, [F::ZERO; 4]),
        acc_digest: alloc_digest(&mut builder, [F::ZERO; 4]),
        plan_digest: alloc_digest(&mut builder, cfg.plan_digest),
    };
    let _ = enforce_nebula_maybe_open_circuit(&mut builder, &lane, &delayed, &context, cfg.seg_max);
    assert!(!builder.is_satisfied(), "seg_idx == seg_max must fail in circuit");
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
    let selected = enforce_nebula_lane_digest_selected_circuit(&mut b, &wires);
    assert_eq!(extract_digest(&b, selected), base.digest());
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
    let selected = enforce_nebula_lane_digest_selected_circuit(&mut b, &wires);
    assert_eq!(extract_digest(&b, selected), opened.digest());
    assert!(b.is_satisfied());
}

#[test]
fn base_lane_wires_are_pinned_to_the_verifier_owned_constant() {
    let cfg = config([F::from_u64(41); 4]);
    let lane = NebulaLane::base(&cfg);
    let mut builder = R1csBuilder::new();
    let wires = alloc_nebula_lane_wires(&mut builder, &lane);
    enforce_nebula_lane_constant_circuit(&mut builder, &wires, &lane);
    assert!(builder.is_satisfied(), "honest base lane constant must satisfy");

    let column = wires.d_mem[0].col();
    let original = builder.witness()[column];
    builder.tamper_witness(column, original + F::ONE);
    assert!(!builder.is_satisfied(), "base lane D_mem must be verifier-pinned");
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

#[test]
fn maybe_close_derives_segment_boundary_and_matches_native_reset() {
    let advs: Vec<_> = (0..N).map(adv).collect();
    let d_pre = honest_d_pre(&advs);
    let cfg = config(d_pre[1]);
    let mut native = NebulaLane::base(&cfg);
    native
        .open_segment(&cfg, [3; 32], [4; 32], [5; 32], d_pre)
        .expect("open");

    let mut builder = R1csBuilder::new();
    let mut wires = alloc_lane(&mut builder, &native);
    for (step, tuple) in advs.iter().enumerate() {
        let x = honest_x(&native);
        native
            .advance(&cfg, &x, Some(tuple))
            .expect("native advance");
        let x_wires = alloc_x(&mut builder, &x);
        let leaves = alloc_adv_leaves(&mut builder, tuple);
        wires = enforce_nebula_advance_circuit(&mut builder, &wires, &x_wires, leaves);
        let closed = enforce_nebula_maybe_close_circuit(&mut builder, &wires, N);
        assert_eq!(
            builder.witness()[closed.closed.col()],
            if step + 1 == N as usize { F::ONE } else { F::ZERO }
        );
        wires = closed.lane;
        assert_lane_parity(&builder, &wires, &native);
    }
    assert!(native.is_closed());
    assert!(
        builder.is_satisfied(),
        "dynamic close path must match native segment walk"
    );
}

#[test]
fn delayed_claim_transition_matches_native_end_to_end() {
    let advs: Vec<_> = (0..N).map(adv).collect();
    let d_pre = honest_d_pre(&advs);
    let cfg = config(d_pre[1]);
    let vk = [3u8; 32];
    let z_i = [4u8; 32];
    let acc = [5u8; 32];
    let base = NebulaLane::base(&cfg);
    let mut native = base.clone();
    native
        .open_segment(&cfg, vk, z_i, acc, d_pre)
        .expect("native open");

    let mut builder = R1csBuilder::new();
    let mut wires = alloc_lane(&mut builder, &base);
    let context = NebulaOpenContextWires {
        vk_fs: alloc_digest(&mut builder, digest::digest32_as_fields(vk)),
        z_i: alloc_digest(&mut builder, digest::digest32_as_fields(z_i)),
        acc_digest: alloc_digest(&mut builder, digest::digest32_as_fields(acc)),
        plan_digest: alloc_digest(&mut builder, cfg.plan_digest),
    };

    for (step, tuple) in advs.iter().enumerate() {
        let x = honest_x(&native);
        let suffix = encode_delayed_f_prime_suffix(&x, cfg.stacks, if step == 0 { Some(d_pre) } else { None })
            .expect("encode delayed claim");
        let suffix_wires = builder.alloc_vec(&suffix);
        let input = decode_delayed_nebula_public_suffix_circuit(&mut builder, &suffix_wires, cfg.stacks)
            .expect("decode delayed claim");
        let adv_wires = alloc_adv(&mut builder, Some(tuple)).expect("adv wires");
        let out =
            enforce_delayed_nebula_claim_circuit(&mut builder, &wires, &input, &adv_wires, &context, N, cfg.seg_max)
                .expect("delayed transition");
        native
            .advance(&cfg, &x, Some(tuple))
            .expect("native advance");
        wires = out.lane;
        assert_lane_parity(&builder, &wires, &native);
    }
    assert!(builder.is_satisfied(), "composed delayed transition must match native");
}
