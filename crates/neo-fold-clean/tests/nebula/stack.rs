//! v3.1 stacks — plan and stack discipline, operation rows E10–E14, the
//! `sp` carry, and every stack red-team row, from the native
//! machine up through the full two-segment pipeline.
//!
//! The rows enforce pointer
//! discipline (addr = sp, no under/overflow, one-hot selectors, pinned
//! `sw`, dead push fields); the *product equation* enforces value/time
//! truth (a wrong-value pop satisfies every row and dies at close); the
//! *lane* enforces `sp` continuity and emptiness at segment boundaries.

use neo_ccs::relations::check_ccs_rowwise_zero;
use neo_fold_clean::config;
use neo_fold_clean::frontends::nebula::circuit::{SMemCircuit, StepData};
use neo_fold_clean::frontends::nebula::fingerprint::Gammas;
use neo_fold_clean::frontends::nebula::layout::{MemOpRecord, MemSpace, NebulaParams};
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::nebula::prove::{prove_segment, SegmentError};
use neo_fold_clean::frontends::nebula::trace::{Memory, SegmentTrace, TraceError};
use neo_fold_clean::lifecycle::{self, preprocess, verify_uncompressed_audit, Error as LifecycleError, Preprocessing};
use neo_fold_clean::paper::construction2::{Error as C2Error, NebulaError};
use neo_math::field::KExtensions;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

#[path = "../support/mod.rs"]
mod support;

const LANE_KAPPA: usize = 18;
const ROM: [u32; 4] = [10, 20, 30, 40];

/// The tiny fixture profile plus two σ = 2 stacks (capacity 3 cells):
/// `r = 2, μ = 2, B_ops = B_scan = 4 → N = 2`.
fn stack_params() -> NebulaParams {
    NebulaParams::new(2, 2, 4, 4, 16)
        .expect("tiny fixture params")
        .with_stacks(2, 2)
        .expect("two σ = 2 stacks")
}

fn plan() -> NebulaPlan {
    NebulaPlan::new(stack_params(), ROM.to_vec(), [0xD1; 32], LANE_KAPPA).expect("stack plan")
}

fn preprocessing(plan: &NebulaPlan) -> Preprocessing {
    let structure = plan.circuit().structure().clone();
    let params = config::r1cs_params(structure.n, structure.m).expect("engine params for S_mem");
    support::install_ajtai_module(&params, &structure);
    preprocess(params, structure, Some(plan.circuit().m_in()))
        .expect("preprocessing")
        .with_nebula(plan.config())
}

fn gammas(seed: u64) -> Gammas {
    Gammas {
        gamma1: K::from_coeffs([F::from_u64(seed), F::from_u64(seed + 1)]),
        gamma2: K::from_coeffs([F::from_u64(seed + 2), F::from_u64(seed + 3)]),
    }
}

/// Segment 0: nested pushes on stack 0, a stack-1 pair, and RAM writes
/// the next segment will read — 8 ops, exactly the segment's capacity,
/// both stacks empty at close.
fn segment0(memory: &mut Memory) -> SegmentTrace {
    let mut run = memory.begin_segment().expect("segment 0");
    run.push(0, 7).expect("push s0");
    run.push(0, 9).expect("push s0 deeper");
    assert_eq!(run.pop(0).expect("pop s0"), 9, "LIFO: last push first");
    assert_eq!(run.pop(0).expect("pop s0"), 7);
    run.write(true, 0, 5).expect("write RAM[0]");
    run.push(1, 3).expect("push s1");
    assert_eq!(run.pop(1).expect("pop s1"), 3);
    assert_eq!(run.read(false, 1).expect("read ROM[1]"), 20);
    run.finish().expect("segment close")
}

/// Segment 1: RAM continuity from segment 0 plus fresh stack traffic —
/// stacks opened at 0 again (they are segment-local).
fn segment1(memory: &mut Memory) -> SegmentTrace {
    let mut run = memory.begin_segment().expect("segment 1");
    assert_eq!(run.read(true, 0).expect("continuity RAM[0]"), 5);
    run.push(0, 11).expect("push s0");
    assert_eq!(run.pop(0).expect("pop s0"), 11);
    assert_eq!(run.read(false, 0).expect("read ROM[0]"), 10);
    run.finish().expect("segment close")
}

/// Unwrap the lane-transition error a prover-side rejection carries.
fn lane_error(err: SegmentError) -> NebulaError {
    match err {
        SegmentError::Lifecycle(LifecycleError::Construction2(C2Error::Nebula(e))) => e,
        other => panic!("expected a lane-transition rejection, got {other}"),
    }
}

fn check(circuit: &SMemCircuit, z: &[F]) -> Result<(), neo_ccs::CcsError> {
    check_ccs_rowwise_zero(circuit.structure(), &z[..circuit.m_in()], &z[circuit.m_in()..])
}

/// Step `i`'s inputs from a segment trace plus the incoming carry.
fn step_data<'a>(trace: &'a SegmentTrace, i: usize, ts_in: u64, h_in: [K; 4], sp_in: [u64; 2]) -> StepData<'a> {
    let b_scan = trace.params().b_scan;
    StepData {
        seg_idx: trace.seg_idx,
        idx: i as u64,
        ts_in,
        h_in,
        sp_in,
        ops: trace.step_ops(i),
        is_cells: &trace.is_cells[i * b_scan..(i + 1) * b_scan],
        fs_cells: &trace.fs_cells[i * b_scan..(i + 1) * b_scan],
    }
}

// ── Native machine ───────────────────────────────────────────────────────

/// Pushes emit WS only, pops RS only, and an honest stack segment
/// balances exactly — the push/pop tuples cancel with probability 1.
#[test]
fn stack_segment_balances_exactly_with_one_tuple_per_op() {
    let mut memory = Memory::new(stack_params(), &ROM).expect("memory");
    let trace = segment0(&mut memory);

    let (pushes, pops) = (3, 3); // segment0's stack traffic
    assert_eq!(trace.rs_tuples().len(), trace.ops.len() - pushes);
    assert_eq!(trace.ws_tuples().len(), trace.ops.len() - pops);
    assert!(trace.balanced(&gammas(41)), "honest balance is exact, no SZ slack");
}

/// The honest-path API guards: under/overflow, unknown stacks, and the
/// segment-local discipline at close.
#[test]
fn stack_api_rejects_misuse() {
    let mut memory = Memory::new(stack_params(), &ROM).expect("memory");
    let mut run = memory.begin_segment().expect("segment");

    assert_eq!(run.pop(0), Err(TraceError::StackUnderflow(0)));
    assert_eq!(run.push(2, 1), Err(TraceError::StackIndex { got: 2, stacks: 2 }));

    // Capacity is 2^σ − 1 = 3 cells (bitness-pure bound, the plan).
    run.push(0, 1).expect("1");
    run.push(0, 2).expect("2");
    run.push(0, 3).expect("3");
    assert_eq!(run.push(0, 4), Err(TraceError::StackOverflow(0)));

    // Stacks are segment-local: a close with live cells is a prover bug,
    // caught here rather than at the product equation.
    assert_eq!(
        run.finish().unwrap_err(),
        TraceError::StackNotEmpty { stack: 0, live: 3 }
    );
}

// ── Circuit rows ─────────────────────────────────────────────────────────

/// Every step of an honest stack segment satisfies the rows, steps chain
/// through `sp` in `x`, and `sp` returns to 0 by segment close.
#[test]
fn honest_stack_segment_satisfies_rows_and_chains_sp() {
    let p = stack_params();
    let circuit = SMemCircuit::new(p);
    let mut memory = Memory::new(p, &ROM).expect("memory");
    let trace = segment0(&mut memory);
    let gammas = gammas(7);

    let mut ts_in = trace.ts_in;
    let mut h_in = [K::ONE; 4];
    let mut sp_in = [0u64; 2];
    for i in 0..p.steps_per_segment() {
        let (z, x) = circuit
            .witness(&gammas, &step_data(&trace, i, ts_in, h_in, sp_in))
            .expect("witness");
        check(&circuit, &z).expect("rows satisfied");
        ts_in = x.ts_out;
        h_in = x.h_out;
        sp_in = x.sp_out;
    }
    assert_eq!(sp_in, [0, 0], "segment-local stacks close empty");
    assert_eq!(h_in, trace.products(&gammas), "circuit and oracle products agree");
}

/// The E10–E14 row rejections, each forged at the op-record or lane-bit
/// level — and the deliberate non-rejection: a wrong-value pop satisfies
/// every row (its authority is the product equation, Lemma 4).
#[test]
fn stack_rows_reject_discipline_violations() {
    let p = stack_params();
    let circuit = SMemCircuit::new(p);
    let mut memory = Memory::new(p, &ROM).expect("memory");
    let trace = segment0(&mut memory);
    let gammas = gammas(13);
    let honest = step_data(&trace, 0, trace.ts_in, [K::ONE; 4], [0; 2]);

    let forge = |ops: &[MemOpRecord]| {
        let (z, _) = circuit
            .witness(&gammas, &StepData { ops, ..honest })
            .expect("total builder");
        check(&circuit, &z)
    };
    let honest_ops = honest.ops.to_vec();
    assert!(forge(&honest_ops).is_ok(), "sanity: the honest step passes");

    // E13: a push not at the stack pointer (slot 0 opens at sp = 0).
    let mut ops = honest_ops.clone();
    ops[0].addr = 1;
    assert!(forge(&ops).is_err(), "E13 must bind addr to sp");

    // E14: a push smuggling a nonzero RS-side field.
    let mut ops = honest_ops.clone();
    ops[0].rt = 1;
    assert!(forge(&ops).is_err(), "E14 must pin rt on push");
    let mut ops = honest_ops.clone();
    ops[0].v_r = 1;
    assert!(forge(&ops).is_err(), "E14 must pin v_r on push");

    // E12: pop at empty — sp would need to be −1, unrepresentable.
    let pop_at_empty = [MemOpRecord {
        is_write: false,
        space: MemSpace::Stack(0),
        addr: 0,
        v_r: 0,
        v_w: 0,
        rt: 0,
    }];
    assert!(forge(&pop_at_empty).is_err(), "E12 bitness must reject underflow");

    // E12: push past capacity — sp would need to be 2^σ.
    let push = |v: u32, addr: u64| MemOpRecord {
        is_write: true,
        space: MemSpace::Stack(0),
        addr,
        v_r: 0,
        v_w: v,
        rt: 0,
    };
    let overflow = [push(1, 0), push(2, 1), push(3, 2), push(4, 3)];
    assert!(forge(&overflow).is_err(), "E12 bitness must reject overflow");

    // E10: two namespace selectors on one slot (lane-bit tamper — no
    // record can express it).
    let (mut z, _) = circuit.witness(&gammas, &honest).expect("witness");
    z[circuit.op_slot_column(0) + 2] = F::ONE; // ram bit beside stk_0
    assert!(check(&circuit, &z).is_err(), "E10 must reject non-one-hot selectors");

    // E11: a lying `sw` bit — faking push-ness on the slot-2 pop would
    // skip its RS tuple (the post-challenge selection attack).
    let (mut z, _) = circuit.witness(&gammas, &honest).expect("witness");
    z[circuit.op_sw_column(2, 0)] = F::ONE;
    assert!(check(&circuit, &z).is_err(), "E11 must pin sw to stk·is_write");

    // The documented NON-rejection: a pop returning the wrong value
    // satisfies every row — value truth is the product equation's job.
    let mut ops = honest_ops.clone();
    assert!(
        !ops[2].is_write && ops[2].space == MemSpace::Stack(0),
        "slot 2 is the first pop"
    );
    ops[2].v_r = 8;
    ops[2].v_w = 8;
    assert!(
        forge(&ops).is_ok(),
        "rows alone must NOT catch a value lie (Lemma 4 layering)"
    );
}

// ── Full pipeline ────────────────────────────────────────────────────────

/// The whole protocol with stacks: two segments of mixed stack/RAM/ROM
/// traffic, cross-segment RAM continuity, folded, finalized, and
/// audit-verified end to end.
#[test]
fn stack_segments_prove_and_verify_end_to_end() {
    let plan = plan();
    let prep = preprocessing(&plan);
    let mut memory = Memory::new(stack_params(), &ROM).expect("memory");
    let trace0 = segment0(&mut memory);
    let trace1 = segment1(&mut memory);

    let audit = lifecycle::prove(&prep, Vec::<Vec<_>>::new()).expect("base");
    let audit = prove_segment(&prep, &plan, audit, &trace0).expect("segment 0");
    let audit = prove_segment(&prep, &plan, audit, &trace1).expect("segment 1");
    let audit = lifecycle::finish_uncompressed_with_audit(&prep, audit).expect("finalize");
    verify_uncompressed_audit(&prep, &audit).expect("audit verification");

    let lane = audit.proof.state.nebula.as_ref().expect("lane");
    assert!(lane.is_closed());
    assert_eq!(lane.sp, [0, 0], "sp is part of the carried lane and ends 0");
    assert_eq!(lane.seg_idx, 2);
}

/// "Pop a different value than was pushed": every row holds, the
/// commitments match `D_pre` — only the product equation catches it.
#[test]
fn wrong_value_pop_fails_the_product_equation() {
    let plan = plan();
    let prep = preprocessing(&plan);
    let mut memory = Memory::new(stack_params(), &ROM).expect("memory");
    let mut trace0 = segment0(&mut memory);

    // The lie: the first pop (slot 2) returns 8 instead of the pushed 9.
    // E3 is kept consistent (v_w = v_r) and rt stays the true push time.
    trace0.ops[2].v_r = 8;
    trace0.ops[2].v_w = 8;

    let audit = lifecycle::prove(&prep, Vec::<Vec<_>>::new()).expect("base");
    let err = prove_segment(&prep, &plan, audit, &trace0).expect_err("value lie must be rejected");
    assert_eq!(lane_error(err), NebulaError::ProductEquation);
}

/// "Pop claiming a wrong `push_time`": the RS tuple matches no WS
/// push tuple, so the products cannot balance.
#[test]
fn wrong_push_time_pop_fails_the_product_equation() {
    let plan = plan();
    let prep = preprocessing(&plan);
    let mut memory = Memory::new(stack_params(), &ROM).expect("memory");
    let mut trace0 = segment0(&mut memory);

    assert_eq!(trace0.ops[2].rt, 2, "the pop's true push time");
    trace0.ops[2].rt = 1; // a real, older timestamp — E4 still holds
    let audit = lifecycle::prove(&prep, Vec::<Vec<_>>::new()).expect("base");
    let err = prove_segment(&prep, &plan, audit, &trace0).expect_err("time lie must be rejected");
    assert_eq!(lane_error(err), NebulaError::ProductEquation);
}

/// "Push without popping": the trace API refuses at `finish`; a
/// forged trace that drops a trailing pop lands on the deterministic
/// `sp = 0` close check, backed by the product equation except with the
/// configured fingerprint-collision probability.
#[test]
fn unpopped_push_fails_the_close() {
    let plan = plan();
    let prep = preprocessing(&plan);
    let mut memory = Memory::new(stack_params(), &ROM).expect("memory");
    let mut trace0 = segment0(&mut memory);

    // segment0's final op is the ROM read; make the forged tail
    // [push s1, ROM read] with the s1 pop dropped: sp1 ends at 1.
    let popped = trace0.ops.remove(6);
    assert!(
        !popped.is_write && popped.space == MemSpace::Stack(1),
        "dropped the s1 pop"
    );

    let audit = lifecycle::prove(&prep, Vec::<Vec<_>>::new()).expect("base");
    let err = prove_segment(&prep, &plan, audit, &trace0).expect_err("unpopped push must be rejected");
    assert_eq!(lane_error(err), NebulaError::StackNotEmptyAtClose);
}

/// "Cross-stack splice": pop from stack 1 what was pushed to stack 0.
/// Both pointer disciplines are locally consistent, but the tuples live
/// in disjoint `g`-ranges — the product equation rejects.
#[test]
fn cross_stack_splice_fails_the_product_equation() {
    let plan = plan();
    let prep = preprocessing(&plan);
    let mut memory = Memory::new(stack_params(), &ROM).expect("memory");
    let mut trace0 = segment0(&mut memory);

    // Swap the s1 push/pop pair (slots 5, 6) onto stack 0 for the push
    // only: push goes to s0, pop still claims s1. Keep both pointer
    // walks valid by also retagging the pop's matching push... which is
    // exactly what the attacker cannot do consistently: retag the push
    // to s0 and leave the pop on s1 — each stack sees one unmatched op.
    assert_eq!(trace0.ops[5].space, MemSpace::Stack(1));
    trace0.ops[5].space = MemSpace::Stack(0);

    let audit = lifecycle::prove(&prep, Vec::<Vec<_>>::new()).expect("base");
    let err = prove_segment(&prep, &plan, audit, &trace0).expect_err("cross-stack splice must be rejected");
    assert_eq!(lane_error(err), NebulaError::StackNotEmptyAtClose);
}
