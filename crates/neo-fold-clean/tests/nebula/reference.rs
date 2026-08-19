//! Reference-model tests — the protocol logic of spec §1/§3/§6 exercised
//! natively, honest and adversarial. The trace is the test oracle (never
//! verifier authority); every attack here must reappear in the red-team
//! suite against the real pipeline once it exists (spec §12).
//!
//! For honest traces the Blum invariant `IS ∪ WS = RS ∪ FS` holds exactly,
//! so balance assertions carry no soundness slack; tampered traces fail
//! except with the negligible §9 probability (fixed challenge seeds keep
//! that deterministic here).

use neo_fold_clean::frontends::nebula::fingerprint::{self, Gammas};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::trace::{Memory, SegmentTrace, TraceError};
use neo_math::field::KExtensions;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

/// Deterministic SplitMix64 — keeps tests reproducible with zero deps.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn gammas(&mut self) -> Gammas {
        Gammas {
            gamma1: K::from_coeffs([F::from_u64(self.next()), F::from_u64(self.next())]),
            gamma2: K::from_coeffs([F::from_u64(self.next()), F::from_u64(self.next())]),
        }
    }
}

const RAM: bool = true;
const ROM: bool = false;

fn rom_image(p: &NebulaParams) -> Vec<u32> {
    (0..p.rom_cells() as u32)
        .map(|i| i.wrapping_mul(2654435761).wrapping_add(17))
        .collect()
}

/// One busy honest segment: RAM writes, overwrites, read-backs, ROM reads.
fn honest_segment(mem: &mut Memory, rng: &mut Rng) -> SegmentTrace {
    let p = *mem.params();
    let mut seg = mem.begin_segment().unwrap();
    for i in 0..40 {
        let addr = rng.next() % p.ram_cells();
        match rng.next() % 3 {
            0 => {
                seg.write(RAM, addr, rng.next() as u32).unwrap();
            }
            1 => {
                seg.read(RAM, addr).unwrap();
            }
            _ => {
                seg.read(ROM, rng.next() % p.rom_cells()).unwrap();
            }
        }
        // Every few ops, hammer one cell to exercise rt chains.
        if i % 7 == 0 {
            seg.write(RAM, 3, i as u32).unwrap();
            assert_eq!(seg.read(RAM, 3).unwrap(), i as u32);
        }
    }
    seg.finish().expect("segment close")
}

#[test]
fn honest_segment_balances_exactly() {
    let p = NebulaParams::test_profile();
    let mut mem = Memory::new(p, &rom_image(&p)).unwrap();
    let mut rng = Rng(11);
    let trace = honest_segment(&mut mem, &mut rng);

    assert_eq!(trace.ts_out, trace.ts_in + trace.ops.len() as u64);
    for seed in [1u64, 2, 3] {
        assert!(trace.balanced(&Rng(seed).gammas()));
    }
}

#[test]
fn reads_return_last_write_and_rom_returns_image() {
    let p = NebulaParams::test_profile();
    let image = rom_image(&p);
    let mut mem = Memory::new(p, &image).unwrap();
    let mut seg = mem.begin_segment().unwrap();

    seg.write(RAM, 7, 5).unwrap();
    seg.write(RAM, 7, 9).unwrap();
    assert_eq!(seg.read(RAM, 7).unwrap(), 9);
    assert_eq!(seg.read(RAM, 8).unwrap(), 0); // untouched RAM starts zeroed
    assert_eq!(seg.read(ROM, 3).unwrap(), image[3]);
    assert!(seg
        .finish()
        .expect("segment close")
        .balanced(&Rng(1).gammas()));
}

#[test]
fn stale_read_is_detected() {
    let p = NebulaParams::test_profile();
    let mut mem = Memory::new(p, &rom_image(&p)).unwrap();
    let mut rng = Rng(12);
    let trace = honest_segment(&mut mem, &mut rng);

    // The classic memory lie: claim a read returned a different value.
    // Only RS changes (WS carries v_w), so the multisets stop balancing.
    let mut tampered = trace.clone();
    tampered.ops[5].v_r ^= 1;
    assert!(!tampered.balanced(&Rng(1).gammas()));
}

#[test]
fn timestamp_tamper_is_detected() {
    let p = NebulaParams::test_profile();
    let mut mem = Memory::new(p, &rom_image(&p)).unwrap();
    let mut rng = Rng(13);
    let trace = honest_segment(&mut mem, &mut rng);

    let mut tampered = trace.clone();
    tampered.ops[9].rt += 1;
    assert!(!tampered.balanced(&Rng(1).gammas()));
}

#[test]
fn dropped_op_is_detected() {
    let p = NebulaParams::test_profile();
    let mut mem = Memory::new(p, &rom_image(&p)).unwrap();
    let mut rng = Rng(14);
    let trace = honest_segment(&mut mem, &mut rng);

    let mut tampered = trace.clone();
    tampered.ops.pop();
    assert!(!tampered.balanced(&Rng(1).gammas()));
}

#[test]
fn products_are_order_independent() {
    // The argument is over multisets: permuting a tuple list must not
    // change its product (this is what makes op-to-step chunking free).
    let p = NebulaParams::test_profile();
    let mut mem = Memory::new(p, &rom_image(&p)).unwrap();
    let mut rng = Rng(15);
    let trace = honest_segment(&mut mem, &mut rng);
    let gammas = Rng(1).gammas();

    let ws = trace.ws_tuples();
    let mut reversed = ws.clone();
    reversed.reverse();
    assert_eq!(
        fingerprint::product(&gammas, &ws),
        fingerprint::product(&gammas, &reversed)
    );
}

#[test]
fn rom_is_write_protected_and_bounds_are_checked() {
    let p = NebulaParams::test_profile();
    let mut mem = Memory::new(p, &rom_image(&p)).unwrap();
    let mut seg = mem.begin_segment().unwrap();

    assert_eq!(seg.write(ROM, 3, 1), Err(TraceError::RomWrite(3)));
    assert!(matches!(
        seg.read(RAM, p.ram_cells()),
        Err(TraceError::AddrRange { .. })
    ));
    assert!(matches!(
        seg.read(ROM, p.rom_cells()),
        Err(TraceError::AddrRange { .. })
    ));
    // Failed ops must not consume timestamps.
    assert_eq!(seg.read(RAM, 0).unwrap(), 0);
    let trace = seg.finish().expect("segment close");
    assert_eq!(trace.ops.len(), 1);
    assert!(trace.balanced(&Rng(1).gammas()));
}

#[test]
fn segment_capacity_is_enforced() {
    let p = NebulaParams::test_profile();
    let mut mem = Memory::new(p, &rom_image(&p)).unwrap();
    let mut seg = mem.begin_segment().unwrap();
    for _ in 0..p.ops_per_segment() {
        seg.read(RAM, 0).unwrap();
    }
    assert!(matches!(seg.read(RAM, 0), Err(TraceError::SegmentFull(_))));
}

#[test]
fn segment_boundary_chains_memory_and_timestamps() {
    let p = NebulaParams::test_profile();
    let mut mem = Memory::new(p, &rom_image(&p)).unwrap();
    let mut rng = Rng(16);

    let t1 = honest_segment(&mut mem, &mut rng);
    let t2 = honest_segment(&mut mem, &mut rng);

    // The model-level analogs of the F′ close checks (spec §6.3/§6.4):
    // FS(k) == IS(k+1) cell-for-cell, and the global timestamp carries.
    assert_eq!(t1.fs_cells, t2.is_cells);
    assert_eq!(t2.ts_in, t1.ts_out);
    assert_eq!((t1.seg_idx, t2.seg_idx), (0, 1));
    assert!(t1.balanced(&Rng(1).gammas()));
    assert!(t2.balanced(&Rng(2).gammas()));
}

#[test]
fn memory_reset_between_segments_is_detected() {
    let p = NebulaParams::test_profile();
    let mut mem = Memory::new(p, &rom_image(&p)).unwrap();
    let mut seg = mem.begin_segment().unwrap();
    seg.write(RAM, 0, 42).unwrap();
    let t1 = seg.finish().expect("segment close");

    // A cheating prover restarts from fresh memory for "segment 1".
    let mut fresh = Memory::new(p, &rom_image(&p)).unwrap();
    let t2_forged = honest_segment(&mut fresh, &mut Rng(17));

    // The boundary equality that D_is == D_mem enforces fails on cells.
    assert_ne!(t1.fs_cells, t2_forged.is_cells);
}

#[test]
fn cross_segment_reads_see_earlier_segments() {
    let p = NebulaParams::test_profile();
    let mut mem = Memory::new(p, &rom_image(&p)).unwrap();

    let mut seg = mem.begin_segment().unwrap();
    seg.write(RAM, 11, 77).unwrap();
    let t1 = seg.finish().expect("segment close");

    let mut seg = mem.begin_segment().unwrap();
    assert_eq!(seg.read(RAM, 11).unwrap(), 77);
    let t2 = seg.finish().expect("segment close");

    assert!(t1.balanced(&Rng(3).gammas()));
    assert!(t2.balanced(&Rng(4).gammas()));
}

#[test]
fn step_chunking_is_sequential_fill() {
    let p = NebulaParams::test_profile();
    let mut mem = Memory::new(p, &rom_image(&p)).unwrap();
    let mut seg = mem.begin_segment().unwrap();
    for _ in 0..20 {
        seg.read(RAM, 1).unwrap();
    }
    let trace = seg.finish().expect("segment close");

    assert_eq!(trace.step_ops(0).len(), p.b_ops);
    assert_eq!(trace.step_ops(1).len(), p.b_ops);
    assert_eq!(trace.step_ops(2).len(), 20 - 2 * p.b_ops);
    assert_eq!(trace.step_ops(3).len(), 0);
}
