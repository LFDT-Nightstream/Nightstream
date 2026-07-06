//! Layout round-trips and plan validation — spec §2, §3, §4.4.
//!
//! Covers: parameter validation (exact cover, packing bound, timestamp
//! capacity), lane encode/decode round-trips, L-ALIGN alignment, pad
//! canonicality (E7 shape), and the step public-input encoding.

use neo_fold_clean::frontends::nebula::layout::{
    CellRecord, LayoutError, MemOpRecord, NebulaParams, StepPublicInput, TS_BITS, X_BITS,
};
use neo_math::field::KExtensions;
use neo_math::{D, F, K};
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

    fn k(&mut self) -> K {
        K::from_coeffs([F::from_u64(self.next()), F::from_u64(self.next())])
    }
}

fn random_op(rng: &mut Rng, p: &NebulaParams) -> MemOpRecord {
    let seg = rng.next() & 1 == 1;
    let cells = if seg { p.ram_cells() } else { p.rom_cells() };
    MemOpRecord {
        is_write: seg && rng.next() & 1 == 1,
        seg,
        addr: rng.next() % cells,
        v_r: rng.next() as u32,
        v_w: rng.next() as u32,
        rt: rng.next() & ((1 << TS_BITS) - 1),
    }
}

#[test]
fn spec_profiles_match_their_derived_sizes() {
    let t = NebulaParams::test_profile();
    assert_eq!(t.total_cells(), 16 + 256);
    assert_eq!(t.steps_per_segment(), 34); // spec §2 test profile
    assert_eq!(t.op_bits(), 3 + 8 + 32 + 32 + 44);

    let v3 = NebulaParams::v3_targets();
    assert_eq!(v3.steps_per_segment(), 1_088); // spec §2: N = (R+M)/B_scan
    assert_eq!(v3.addr_bits(), 16);
    assert_eq!(v3.op_bits(), 127); // spec §3.2: OP_BITS
    assert_eq!(X_BITS, 1_400); // spec §4.4
}

#[test]
fn params_reject_spec_violations() {
    // Exact cover: R + M = 4 + 8 = 12 is not a multiple of B_scan = 8.
    assert!(NebulaParams::new(2, 3, 8, 8, 1).is_err());
    // Packing bound: bits(R + M) = 19 pushes TS_BITS + 19 = 63 > 62.
    assert!(NebulaParams::new(4, 18, 8, 8, 1).is_err());
    // Timestamp capacity: seg_max · N · B_ops must stay below 2^TS_BITS.
    assert!(NebulaParams::new(4, 8, 8, 8, 1 << 40).is_err());
    // Zero-sized blocks are meaningless.
    assert!(NebulaParams::new(4, 8, 0, 8, 1).is_err());
    // r > mu: RAM addresses would escape their bitness bound (spec §2,
    // external-review fix). R + M = 256 + 16 = 272 is a multiple of
    // B_scan = 8, so only the r ≤ mu rule rejects this.
    assert!(NebulaParams::new(8, 4, 8, 8, 1).is_err());
}

#[test]
fn lanes_are_ring_column_aligned() {
    // L-ALIGN (spec §5.1): committed lanes must span whole ring columns.
    for p in [NebulaParams::test_profile(), NebulaParams::v3_targets()] {
        assert_eq!(p.ops_lane_bits() % D, 0);
        assert_eq!(p.scan_lane_bits() % D, 0);
        assert!(p.ops_lane_bits() >= p.b_ops * p.op_bits());
        assert!(p.scan_lane_bits() >= p.b_scan * neo_fold_clean::frontends::nebula::layout::CELL_BITS);
    }
}

#[test]
fn ops_lane_round_trips_and_is_bits() {
    let p = NebulaParams::test_profile();
    let mut rng = Rng(1);
    let ops: Vec<_> = (0..5).map(|_| random_op(&mut rng, &p)).collect();
    let lane = p.encode_ops_lane(&ops).unwrap();
    assert_eq!(lane.len(), p.ops_lane_bits());
    assert!(lane.iter().all(|&b| b == F::ZERO || b == F::ONE));
    assert_eq!(p.decode_ops_lane(&lane).unwrap(), ops);
}

#[test]
fn ops_lane_rejects_pad_violations() {
    let p = NebulaParams::test_profile();
    let mut rng = Rng(2);
    let ops: Vec<_> = (0..3).map(|_| random_op(&mut rng, &p)).collect();
    let lane = p.encode_ops_lane(&ops).unwrap();

    // A nonzero field bit inside a pad slot breaks E7 canonicality.
    let mut tampered = lane.clone();
    tampered[3 * p.op_bits() + 1] = F::ONE; // is_write bit of pad slot 3
    assert!(matches!(
        p.decode_ops_lane(&tampered),
        Err(LayoutError::PadNotCanonical(_))
    ));

    // A "real" op after a pad slot breaks sequential fill.
    let mut gap = lane.clone();
    gap[4 * p.op_bits()] = F::ZERO; // clear pad bit of slot 4; slot 3 stays pad
    assert!(matches!(p.decode_ops_lane(&gap), Err(LayoutError::PadNotCanonical(_))));

    // A non-bit coordinate anywhere is rejected.
    let mut nonbit = lane;
    nonbit[0] = F::from_u64(2);
    assert!(matches!(p.decode_ops_lane(&nonbit), Err(LayoutError::NonBit(_))));

    // Too many ops for the lane.
    let many: Vec<_> = (0..p.b_ops + 1).map(|_| random_op(&mut rng, &p)).collect();
    assert!(matches!(p.encode_ops_lane(&many), Err(LayoutError::TooManyOps { .. })));
}

#[test]
fn scan_lane_round_trips() {
    let p = NebulaParams::test_profile();
    let mut rng = Rng(3);
    let cells: Vec<_> = (0..p.b_scan)
        .map(|_| CellRecord {
            v: rng.next() as u32,
            t: rng.next() & ((1 << TS_BITS) - 1),
        })
        .collect();
    let lane = p.encode_scan_lane(&cells).unwrap();
    assert_eq!(lane.len(), p.scan_lane_bits());
    assert_eq!(p.decode_scan_lane(&lane).unwrap(), cells);

    // Exact cover: a scan lane never takes a partial chunk.
    assert!(matches!(
        p.encode_scan_lane(&cells[1..]),
        Err(LayoutError::ScanLen { .. })
    ));
}

#[test]
fn step_public_input_round_trips() {
    let mut rng = Rng(4);
    let x = StepPublicInput {
        seg_idx: rng.next() & 0xFFFF,
        idx: rng.next() & 0xFFFF,
        ts_in: rng.next() & ((1 << TS_BITS) - 1),
        ts_out: rng.next() & ((1 << TS_BITS) - 1),
        gamma: [rng.k(), rng.k()],
        h_in: [rng.k(), rng.k(), rng.k(), rng.k()],
        h_out: [rng.k(), rng.k(), rng.k(), rng.k()],
    };
    let bits = x.encode().unwrap();
    assert_eq!(bits.len(), X_BITS);
    assert!(bits.iter().all(|&b| b == F::ZERO || b == F::ONE));
    assert_eq!(StepPublicInput::decode(&bits).unwrap(), x);

    // Out-of-width counters are rejected at encode time.
    let bad = StepPublicInput {
        ts_in: 1 << TS_BITS,
        ..x
    };
    assert!(matches!(bad.encode(), Err(LayoutError::FieldRange { .. })));
}
