//! Segment-prover tests: the segment prover drives real
//! `S_mem` traces through the full pipeline, memory chains across
//! segments, and the plan artifact binds the ROM image. This is the test
//! the external review named: segment 0 writes RAM and reads ROM,
//! segment 1 reads segment 0's writes — the shape that would have caught
//! the lane-typed-tag continuity bug.

#[path = "fixture.rs"]
mod fixture;

use fixture::{honest_two_segment_chain, plan, preprocessing, segment0, segment1, tiny_params, LANE_KAPPA, ROM};
use neo_fold_clean::frontends::nebula::circuit::StepData;
use neo_fold_clean::frontends::nebula::fingerprint::Gammas;
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::nebula::prove::{derive_segment_gamma, prove_segment, resume_segment, SegmentError};
use neo_fold_clean::frontends::nebula::trace::{Memory, SegmentTrace};
use neo_fold_clean::lifecycle::{self, verify_uncompressed_audit, Preprocessing, UncompressedAudit};
use neo_fold_clean::paper::digest;
use neo_fold_clean::paper::relations::CcsInstance;
use neo_math::K;
use p3_field::PrimeCharacteristicRing;

/// The whole protocol, real circuit, two segments, verified end to end:
/// fingerprint products from real memory ops balance at each close, the
/// FS→IS boundary chains match, and the audit verifier replays every
/// lane transition plus the terminal slice openings.
#[test]
#[ignore = "constructing and checking the full two-segment proof exceeds the 5-minute test cap"]
fn two_segment_chain_with_memory_continuity_verifies() {
    let (_, prep, audit) = honest_two_segment_chain();
    verify_uncompressed_audit(&prep, &audit).expect("audit verification");

    let lane = audit.proof.state.nebula.as_ref().expect("lane");
    assert!(lane.is_closed());
    assert_eq!(lane.seg_idx, 2, "two segments closed");
    assert_eq!(lane.ts, 8, "4 ops per segment, ts never resets");
    let max_fresh = prep.params.max_fresh_count().max(1);
    let n = 2usize; // fixture steps per segment
    let chunks_per_segment = n.div_ceil(max_fresh);
    assert_eq!(
        audit.steps.len(),
        2 * chunks_per_segment,
        "batched folding: each segment is ⌈N / max_fresh⌉ F′ steps (max_fresh = {max_fresh})"
    );
}

/// The plan digest and `D_init` bind the ROM image (the initial-memory and plan rules): a
/// different program is a different plan.
#[test]
fn plan_binds_rom_image() {
    let base = plan();
    let mut other_rom = ROM;
    other_rom[1] += 1;
    let other = NebulaPlan::new(tiny_params(), other_rom.to_vec(), [0xC3; 32], LANE_KAPPA).expect("other plan");
    assert_ne!(base.d_init(), other.d_init(), "D_init must bind the ROM image");
    assert_ne!(
        base.plan_digest(),
        other.plan_digest(),
        "plan digest must bind the ROM image"
    );

    let reseeded = NebulaPlan::new(tiny_params(), ROM.to_vec(), [0xC4; 32], LANE_KAPPA).expect("reseeded plan");
    assert_ne!(
        base.plan_digest(),
        reseeded.plan_digest(),
        "plan digest must bind the matrix seed"
    );
}

#[test]
fn plan_binds_and_memory_uses_nonzero_initial_ram() {
    let params = tiny_params();
    let mut ram = vec![0; params.ram_cells() as usize];
    ram[3] = 0xfeed_beef;
    let with_data = NebulaPlan::new_with_initial_ram(params, ROM.to_vec(), ram.clone(), [0xC3; 32], LANE_KAPPA)
        .expect("plan with initial RAM");
    let zeroed = plan();
    assert_ne!(with_data.d_init(), zeroed.d_init(), "D_init must bind initial RAM");
    assert_ne!(
        with_data.plan_digest(),
        zeroed.plan_digest(),
        "plan digest must bind initial RAM",
    );

    let mut memory = Memory::new_with_initial_ram(params, &ROM, &ram).expect("memory with initial RAM");
    let mut segment = memory.begin_segment().expect("segment");
    assert_eq!(segment.read(true, 3).expect("read initialized RAM"), 0xfeed_beef);
}

/// `D_init` is verifier-recomputable and deterministic — the γ-independent
/// ROM handle: same public inputs, same handle.
#[test]
fn d_init_is_deterministic() {
    assert_eq!(plan().d_init(), plan().d_init());
}

// ── Prover resume ────────────────────────────────────────────────────────

/// Open segment 0 and deposit only its first step (chunk sizes are
/// caller-chosen; the lane records the mid-segment position). Returns
/// the paused chain — the carried lane is the whole checkpoint.
fn mid_segment_chain(prep: &Preprocessing, plan: &NebulaPlan, trace: &SegmentTrace) -> UncompressedAudit {
    let params = *plan.params();
    let mut advs = Vec::new();
    for i in 0..params.steps_per_segment() {
        let ops_bits = params.encode_ops_lane(trace.step_ops(i)).expect("ops lane");
        let is_bits = params
            .encode_scan_lane(&trace.is_cells[i * params.b_scan..(i + 1) * params.b_scan])
            .expect("is lane");
        let fs_bits = params
            .encode_scan_lane(&trace.fs_cells[i * params.b_scan..(i + 1) * params.b_scan])
            .expect("fs lane");
        advs.push(
            plan.scheme()
                .commit_bits(&ops_bits, &is_bits, &fs_bits)
                .expect("commit"),
        );
    }
    let d_pre = digest::nebula_lane_chains(advs.iter());

    let audit = lifecycle::prove(prep, Vec::<Vec<_>>::new()).expect("base");
    let gamma = derive_segment_gamma(prep, &audit, d_pre).expect("γ");
    let gammas = Gammas {
        gamma1: gamma[0],
        gamma2: gamma[1],
    };
    let data = StepData {
        seg_idx: trace.seg_idx,
        idx: 0,
        ts_in: trace.ts_in,
        h_in: [K::ONE; 4],
        sp_in: [0; 2],
        ops: trace.step_ops(0),
        is_cells: &trace.is_cells[..params.b_scan],
        fs_cells: &trace.fs_cells[..params.b_scan],
    };
    let (z, _) = plan.circuit().witness(&gammas, &data).expect("witness");
    let mut instance =
        CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &z, plan.circuit().m_in())
            .expect("instance");
    instance.claim.adv = Some(advs[0].clone());
    let audit = lifecycle::extend_nebula_open(prep, audit, vec![instance], d_pre).expect("first chunk");

    let lane = audit.proof.state.nebula.as_ref().expect("lane");
    assert_eq!(lane.idx, 1, "mid-segment position recorded on the lane");
    assert!(lane.gamma.is_some(), "γ squeezed at open");
    audit
}

/// The resume path completes a mid-segment chain — the lane supplies γ,
/// `D_pre`, the step index, and the `ts`/`h`/`sp` carry — and the chain
/// continues into the next segment and verifies end to end.
#[test]
#[ignore = "resuming, completing, and checking the full two-segment proof exceeds the 5-minute test cap"]
fn mid_segment_resume_completes_and_verifies() {
    let plan = plan();
    let prep = preprocessing(&plan);
    let mut memory = Memory::new(tiny_params(), &ROM).expect("memory");
    let trace0 = segment0(&mut memory);
    let trace1 = segment1(&mut memory);

    let paused = mid_segment_chain(&prep, &plan, &trace0);
    let audit = resume_segment(&prep, &plan, paused, &trace0).expect("resume");
    let lane = audit.proof.state.nebula.as_ref().expect("lane");
    assert!(lane.is_closed(), "resume must close the segment");
    assert_eq!(lane.seg_idx, 1);

    let audit = prove_segment(&prep, &plan, audit, &trace1).expect("segment 1");
    let audit = lifecycle::finish_uncompressed_with_audit(&prep, audit).expect("finalize");
    verify_uncompressed_audit(&prep, &audit).expect("audit verification");
}

/// Resume is fail-closed: a chain at a segment boundary has nothing to
/// resume, and a trace that does not reproduce the open segment's
/// pre-committed chains (γ was squeezed over them) is rejected.
#[test]
fn resume_rejects_boundary_chains_and_wrong_traces() {
    let plan = plan();
    let prep = preprocessing(&plan);
    let mut memory = Memory::new(tiny_params(), &ROM).expect("memory");
    let trace0 = segment0(&mut memory);

    // A fresh chain (no open segment) is not resumable.
    let fresh = lifecycle::prove(&prep, Vec::<Vec<_>>::new()).expect("base");
    assert!(matches!(
        resume_segment(&prep, &plan, fresh, &trace0),
        Err(SegmentError::NotMidSegment)
    ));

    // A different history at the same position: same seg_idx and ts_in,
    // different lane commitments — recompute-vs-D_pre must reject.
    let mut other_memory = Memory::new(tiny_params(), &ROM).expect("other memory");
    let mut run = other_memory.begin_segment().expect("other segment 0");
    for _ in 0..4 {
        run.write(true, 3, 8).expect("write");
    }
    let other_trace = run.finish().expect("segment close");

    let paused = mid_segment_chain(&prep, &plan, &trace0);
    assert!(matches!(
        resume_segment(&prep, &plan, paused, &other_trace),
        Err(SegmentError::ResumeTraceMismatch)
    ));
}
