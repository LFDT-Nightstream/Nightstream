//! M3 gate tests (spec §13 steps 5–6): the segment prover drives real
//! `S_mem` traces through the full pipeline, memory chains across
//! segments, and the plan artifact binds the ROM image. This is the test
//! the external review named: segment 0 writes RAM and reads ROM,
//! segment 1 reads segment 0's writes — the shape that would have caught
//! the lane-typed-tag continuity bug.

#[path = "fixture.rs"]
mod fixture;

use fixture::{honest_two_segment_chain, plan, tiny_params, LANE_KAPPA, ROM};
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::lifecycle::verify_uncompressed_audit;

/// The whole protocol, real circuit, two segments, verified end to end:
/// fingerprint products from real memory ops balance at each close, the
/// FS→IS boundary chains match, and the audit verifier replays every
/// §6.3 transition plus the terminal slice-openings.
#[test]
fn two_segment_chain_with_memory_continuity_verifies() {
    let (_, prep, audit) = honest_two_segment_chain();
    verify_uncompressed_audit(&prep, &audit).expect("audit verification");

    let lane = audit.proof.state.nebula.as_ref().expect("lane");
    assert!(lane.is_closed());
    assert_eq!(lane.seg_idx, 2, "two segments closed");
    assert_eq!(lane.ts, 8, "4 ops per segment, ts never resets");
}

/// The plan digest and `D_init` bind the ROM image (spec §7/§11): a
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

/// `D_init` is verifier-recomputable and deterministic — the γ-independent
/// ROM handle (spec §7): same public inputs, same handle.
#[test]
fn d_init_is_deterministic() {
    assert_eq!(plan().d_init(), plan().d_init());
}
