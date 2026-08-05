//! Shared real-pipeline fixture: a tiny but complete Nebula plan
//! (`r = 2, μ = 2, B_ops = B_scan = 4 → N = 2`), its preprocessing over
//! the real `S_mem` structure, and a two-segment program with
//! cross-segment memory continuity. Used by the segment-prover tests
//! (`segment.rs`) and the red-team suite (`redteam.rs`).

use neo_fold_clean::config;
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::nebula::prove::prove_segment;
use neo_fold_clean::frontends::nebula::trace::{Memory, SegmentTrace};
use neo_fold_clean::lifecycle::{self, preprocess, Preprocessing, UncompressedAudit};

#[path = "../support/mod.rs"]
mod support;

/// Ajtai module dimension for the lane matrices (`A_ops`/`A_mem` are
/// their own MSIS instances, independent of the engine's `A`).
pub const LANE_KAPPA: usize = 18;
/// The fixture's ROM image (R = 4 cells).
pub const ROM: [u32; 4] = [10, 20, 30, 40];

pub fn tiny_params() -> NebulaParams {
    NebulaParams::new(2, 2, 4, 4, 16).expect("tiny fixture params")
}

pub fn plan() -> NebulaPlan {
    NebulaPlan::new(tiny_params(), ROM.to_vec(), [0xC3; 32], LANE_KAPPA).expect("fixture plan")
}

/// Preprocessing over the plan's real `S_mem` structure, with the plan
/// attached — the production shape in miniature.
pub fn preprocessing(plan: &NebulaPlan) -> Preprocessing {
    let structure = plan.circuit().structure().clone();
    let params = config::r1cs_params(structure.n, structure.m).expect("engine params for S_mem");
    support::install_ajtai_module(&params, &structure);
    preprocess(params, structure, Some(plan.circuit().m_in()))
        .expect("preprocessing")
        .with_nebula(plan.config())
}

/// Segment 0: write RAM, read ROM — the writes segment 1 must see.
pub fn segment0(memory: &mut Memory) -> SegmentTrace {
    let mut run = memory.begin_segment().expect("segment 0");
    run.write(true, 0, 7).expect("write RAM[0]");
    run.write(true, 1, 9).expect("write RAM[1]");
    assert_eq!(run.read(false, 2).expect("read ROM[2]"), 30);
    assert_eq!(run.read(true, 0).expect("read back RAM[0]"), 7);
    run.finish().expect("segment close")
}

/// Segment 1: read segment 0's writes (cross-segment continuity) and the
/// ROM, then write again.
pub fn segment1(memory: &mut Memory) -> SegmentTrace {
    let mut run = memory.begin_segment().expect("segment 1");
    assert_eq!(run.read(true, 0).expect("continuity RAM[0]"), 7);
    assert_eq!(run.read(true, 1).expect("continuity RAM[1]"), 9);
    run.write(true, 2, 5).expect("write RAM[2]");
    assert_eq!(run.read(false, 0).expect("read ROM[0]"), 10);
    run.finish().expect("segment close")
}

/// The honest two-segment chain, proved and finalized.
pub fn honest_two_segment_chain() -> (NebulaPlan, Preprocessing, UncompressedAudit) {
    let plan = plan();
    let prep = preprocessing(&plan);
    let mut memory = Memory::new(tiny_params(), &ROM).expect("memory");

    let trace0 = segment0(&mut memory);
    let trace1 = segment1(&mut memory);

    let audit = lifecycle::prove(&prep, Vec::<Vec<_>>::new()).expect("base");
    let audit = prove_segment(&prep, &plan, audit, &trace0).expect("segment 0");
    let audit = prove_segment(&prep, &plan, audit, &trace1).expect("segment 1");
    let audit = lifecycle::finish_uncompressed_with_audit(&prep, audit).expect("finalize");
    (plan, prep, audit)
}
