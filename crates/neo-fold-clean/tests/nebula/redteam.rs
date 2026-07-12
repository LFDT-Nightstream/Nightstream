//! Red-team suite — spec §12, against the real pipeline (real `S_mem`
//! structure, real fingerprints, real folding). Every attack lands on
//! the specific named check the spec's table points at — never a host
//! replay comparison.
//!
//! Row-level attacks on the op block (`rt ≥ wt` → E4, pad misuse → E7,
//! ROM write → E5, ROM address range → E6) are pinned at circuit level
//! by `nebula_circuit::forged_ops_are_rejected_by_rows`; this suite owns
//! the protocol-level rows: multiset lies, boundary/continuity attacks,
//! challenge tampering, and terminal slice-openings.

#[path = "fixture.rs"]
mod fixture;

use fixture::{honest_two_segment_chain, plan, preprocessing, segment0, tiny_params, LANE_KAPPA, ROM};
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::nebula::prove::{prove_segment, SegmentError};
use neo_fold_clean::frontends::nebula::trace::Memory;
use neo_fold_clean::lifecycle::{self, verify_uncompressed_audit, Error as LifecycleError};
use neo_fold_clean::paper::construction2::{Error as C2Error, NebulaError, ProofState};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

/// Unwrap the lane-transition error a prover-side rejection carries.
fn lane_error(err: SegmentError) -> NebulaError {
    match err {
        SegmentError::Lifecycle(LifecycleError::Construction2(C2Error::Nebula(e))) => e,
        other => panic!("expected a §6.3 lane rejection, got {other}"),
    }
}

/// §12 "Stale read (classic memory lie)": a read claiming the initial
/// value after it was overwritten. The multiset product equation fails at
/// segment close, except with the §9 probability.
#[test]
fn stale_read_fails_the_product_equation() {
    let plan = plan();
    let prep = preprocessing(&plan);
    let mut memory = Memory::new(tiny_params(), &ROM).expect("memory");
    let trace0 = segment0(&mut memory);
    let mut run = memory.begin_segment().expect("segment 1");
    assert_eq!(run.read(true, 0).expect("read"), 7);
    run.write(true, 2, 5).expect("write");
    assert_eq!(run.read(true, 2).expect("read back"), 5);
    assert_eq!(run.read(false, 0).expect("rom"), 10);
    let mut trace1 = run.finish().expect("segment close");

    // The lie: RAM[0] read returns the pre-segment-0 value (0 at t = 0)
    // instead of the 7 written in segment 0. Row-level checks all pass
    // (rt < wt holds, read consistency holds); only the multiset argument
    // catches it.
    trace1.ops[0].v_r = 0;
    trace1.ops[0].v_w = 0;
    trace1.ops[0].rt = 0;

    let audit = lifecycle::prove(&prep, Vec::<Vec<_>>::new()).expect("base");
    let audit = prove_segment(&prep, &plan, audit, &trace0).expect("segment 0");
    let err = prove_segment(&prep, &plan, audit, &trace1).expect_err("stale read must be rejected");
    assert_eq!(lane_error(err), NebulaError::ProductEquation);
}

/// §12 "Fresh memory at segment start": segment 1 is internally
/// consistent against a *different* memory history. Its products balance,
/// its chains match its own precommitments — only the boundary check
/// (`D_seen[is] == D_mem`) catches it.
#[test]
fn fresh_memory_at_segment_start_fails_the_boundary() {
    let plan = plan();
    let prep = preprocessing(&plan);
    let mut memory = Memory::new(tiny_params(), &ROM).expect("memory");
    let trace0 = segment0(&mut memory);

    // A parallel universe: a different segment 0, then its segment 1 —
    // internally honest, but its opening memory is not our chain's
    // closing memory.
    let mut other = Memory::new(tiny_params(), &ROM).expect("other memory");
    let mut run = other.begin_segment().expect("other segment 0");
    for _ in 0..4 {
        run.write(true, 3, 1).expect("write");
    }
    let _other_trace0 = run.finish().expect("segment close");
    let mut run = other.begin_segment().expect("other segment 1");
    for _ in 0..4 {
        assert_eq!(run.read(true, 3).expect("read"), 1);
    }
    let spliced_trace1 = run.finish().expect("segment close");

    let audit = lifecycle::prove(&prep, Vec::<Vec<_>>::new()).expect("base");
    let audit = prove_segment(&prep, &plan, audit, &trace0).expect("segment 0");
    let err = prove_segment(&prep, &plan, audit, &spliced_trace1).expect_err("spliced history must be rejected");
    assert_eq!(lane_error(err), NebulaError::BoundaryMismatch);
}

/// §12 "Swap IS and FS lanes of one step" (whole-segment variant): the
/// tuples are committed consistently with the swap, so `D_seen == D_pre`
/// holds — the product identity is what rejects.
#[test]
fn swapped_is_fs_snapshots_fail_the_product_equation() {
    let plan = plan();
    let prep = preprocessing(&plan);
    let mut memory = Memory::new(tiny_params(), &ROM).expect("memory");
    let mut trace0 = segment0(&mut memory);
    std::mem::swap(&mut trace0.is_cells, &mut trace0.fs_cells);

    let audit = lifecycle::prove(&prep, Vec::<Vec<_>>::new()).expect("base");
    let err = prove_segment(&prep, &plan, audit, &trace0).expect_err("swapped snapshots must be rejected");
    assert_eq!(lane_error(err), NebulaError::ProductEquation);
}

/// §12 "Reset timestamps between segments": a segment relabeled to start
/// at ts 0 disagrees with the carried global counter.
#[test]
fn timestamp_reset_between_segments_is_rejected() {
    let plan = plan();
    let prep = preprocessing(&plan);
    let mut memory = Memory::new(tiny_params(), &ROM).expect("memory");
    let trace0 = segment0(&mut memory);
    let mut run = memory.begin_segment().expect("segment 1");
    assert_eq!(run.read(true, 0).expect("read"), 7);
    let mut trace1 = run.finish().expect("segment close");
    trace1.ts_in = 0; // pretend history restarted

    let audit = lifecycle::prove(&prep, Vec::<Vec<_>>::new()).expect("base");
    let audit = prove_segment(&prep, &plan, audit, &trace0).expect("segment 0");
    let err = prove_segment(&prep, &plan, audit, &trace1).expect_err("ts reset must be rejected");
    assert!(matches!(err, SegmentError::ChainPositionMismatch { .. }));
}

/// §12 "Squeeze γ before absorbing all lane commitments" (payload form):
/// tampering the recorded segment-open `D_pre` diverges the verifier's γ
/// replay, so the folded claims' x fails the §6.3 γ equality.
#[test]
fn tampered_segment_open_payload_fails_verification() {
    let (_, prep, mut audit) = honest_two_segment_chain();
    let open = audit.steps[0]
        .nebula_open
        .as_mut()
        .expect("segment-open payload");
    open[0][0] += F::ONE;
    assert!(
        verify_uncompressed_audit(&prep, &audit).is_err(),
        "tampered D_pre must diverge the γ replay"
    );
}

/// §12 "Flip a lane bit after committing" at the terminal. Defense in
/// depth: the decider preflight's full-`z` opening rejects first (the
/// flip breaks `commit(Z) == c` before the lane checks even run), and
/// the R3 slice-opening — Lemma 1's extraction anchor — independently
/// pins each tuple to the same witness, probed here directly against the
/// real chain's terminal children.
#[test]
fn terminal_lane_bit_flip_fails_the_openings() {
    let (plan, prep, mut audit) = honest_two_segment_chain();
    let ProofState::Active { running, .. } = &mut audit.proof.state.proof else {
        panic!("finalized chain is Active");
    };
    let running = running
        .as_materialized_mut()
        .expect("CPU Nebula fixture has materialized running state");

    // R3 probe on the untampered chain: every terminal child's tuple
    // opens against its own lane slices...
    for (claim, witness) in running.claims.iter().zip(&running.witnesses) {
        let adv = claim.adv.as_ref().expect("terminal children carry tuples");
        assert!(plan.scheme().open_matches(adv, witness).expect("openable"));
    }

    // ...and the flip breaks exactly the ops-lane opening.
    let ops_col = plan.circuit().lane_ranges().ops.start;
    running.witnesses[0][(3, ops_col)] += F::ONE;
    let adv = running.claims[0].adv.as_ref().expect("tuple");
    assert!(
        !plan
            .scheme()
            .open_matches(adv, &running.witnesses[0])
            .expect("openable"),
        "R3 slice-opening must reject the flipped lane"
    );

    // Full-pipeline rejection (whichever layered check fires first).
    assert!(
        verify_uncompressed_audit(&prep, &audit).is_err(),
        "a lane-bit flip must be rejected end to end"
    );
}

/// §12 "Wrong ROM image with correct shape": the same proof verified
/// against a different program's plan diverges at the plan-bound γ and
/// the `D_init` boundary.
#[test]
fn wrong_rom_plan_rejects_the_chain() {
    let (_, _, audit) = honest_two_segment_chain();
    let mut other_rom = ROM;
    other_rom[0] += 1;
    let other_plan = NebulaPlan::new(tiny_params(), other_rom.to_vec(), [0xC3; 32], LANE_KAPPA).expect("other plan");
    let other_prep = preprocessing(&other_plan);
    assert!(
        verify_uncompressed_audit(&other_prep, &audit).is_err(),
        "a different ROM is a different plan; the chain must not verify against it"
    );
}
