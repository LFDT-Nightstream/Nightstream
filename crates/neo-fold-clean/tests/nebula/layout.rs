//! Layout round trips and plan validation.
//!
//! Covers: parameter validation (exact cover, packing bound, timestamp
//! capacity), lane encode/decode round-trips, L-ALIGN alignment, pad
//! canonicality (E7 shape), and the step public-input encoding.

use neo_fold_clean::frontends::nebula::layout::{
    CellRecord, LayoutError, MemOpRecord, MemSpace, NebulaParams, StackShape, StepPublicInput, TS_BITS, X_BASE_BITS,
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
    let ram = rng.next() & 1 == 1;
    let space = if ram { MemSpace::Ram } else { MemSpace::Rom };
    let cells = if ram { p.ram_cells() } else { p.rom_cells() };
    MemOpRecord {
        is_write: ram && rng.next() & 1 == 1,
        space,
        addr: rng.next() % cells,
        v_r: rng.next() as u32,
        v_w: rng.next() as u32,
        rt: rng.next() & ((1 << TS_BITS) - 1),
    }
}

#[test]
fn spec_profiles_match_their_derived_sizes() {
    let t = NebulaParams::test_profile();
    assert_eq!(t.scanned_cells(), 16 + 256);
    assert_eq!(t.steps_per_segment(), 34); // the plan test profile
    assert_eq!(t.op_bits(), 3 + 8 + 32 + 32 + 44);

    let v3 = NebulaParams::v3_targets();
    assert_eq!(v3.steps_per_segment(), 1_088); // the plan: N = (R+M)/B_scan
    assert_eq!(v3.addr_bits(), 16);
    assert_eq!(v3.op_bits(), 127); // OP_BITS at S = 0
    assert_eq!(X_BASE_BITS, 1_400);
    assert_eq!(StackShape::NONE.x_bits(), 1_400); // v3 degeneracy

    // v3.1 stacks: OP_BITS grows by S, x by 2·S·σ; the scan domain and
    // N are untouched because stacks are never scanned.
    let ts = t.with_stacks(2, 4).expect("test profile + stacks");
    assert_eq!(ts.op_bits(), t.op_bits() + 2);
    assert_eq!(ts.x_bits(), 1_400 + 2 * 2 * 4);
    assert_eq!(ts.scanned_cells(), t.scanned_cells());
    assert_eq!(ts.steps_per_segment(), t.steps_per_segment());
    assert_eq!(
        ts.global_index(MemSpace::Stack(1), 3).unwrap(),
        ts.scanned_cells() + 16 + 3,
        "stack namespaces linearize above RAM"
    );
}

#[test]
fn params_reject_spec_violations() {
    // Exact cover: R + M = 4 + 8 = 12 is not a multiple of B_scan = 8.
    assert!(NebulaParams::new(2, 3, 8, 8, 1).is_err());
    // Packing bound: bits(R + M) = 19 pushes TS_BITS + 19 = 63 > 62.
    assert!(NebulaParams::new(4, 18, 8, 8, 1).is_err());
    // Timestamp capacity: seg_max · N · B_ops must stay below 2^TS_BITS.
    assert!(NebulaParams::new(4, 8, 8, 8, 1 << 40).is_err());
    // The verifier relation treats seg_max as a strict chain bound, so zero
    // segments and bounds wider than the public segment counter are invalid.
    assert!(NebulaParams::new(0, 0, 1, 2, 0).is_err());
    assert!(NebulaParams::new(0, 0, 1, 2, (1 << 16) + 1).is_err());
    // Every in-segment index is carried in STEP_IDX_BITS public bits.
    assert!(NebulaParams::new(0, 16, 1, 1, 1).is_err());
    // Zero-sized blocks are meaningless.
    assert!(NebulaParams::new(4, 8, 0, 8, 1).is_err());
    // r > mu: RAM addresses would escape their bitness bound (the plan,
    // external-review fix). R + M = 256 + 16 = 272 is a multiple of
    // B_scan = 8, so only the r ≤ mu rule rejects this.
    assert!(NebulaParams::new(8, 4, 8, 8, 1).is_err());

    // v3.1 stack rules: σ within [1, μ], S within [1, MAX_STACKS], and
    // the packing bound now covers the stack namespaces.
    let t = NebulaParams::test_profile();
    assert!(t.with_stacks(2, 0).is_err());
    assert!(t.with_stacks(2, 9).is_err()); // σ > μ = 8
    assert!(t.with_stacks(0, 4).is_err());
    assert!(t.with_stacks(3, 4).is_err());
    // Packing bound with stacks: μ = 17 alone fits (44 + 18 = 62), but
    // two 2^17-cell stacks widen the address space to 19 bits — 63 > 62.
    let wide = NebulaParams::new(4, 17, 8, 8, 1).expect("R + M alone fits the packing bound");
    assert!(
        wide.with_stacks(2, 17).is_err(),
        "stacks must respect the packing bound"
    );
}

#[test]
fn lanes_are_ring_column_aligned() {
    // L-ALIGN (the lane layout): committed lanes must span whole ring columns.
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
        sp_in: [0; 2],
        sp_out: [0; 2],
    };
    let bits = x.encode(StackShape::NONE).unwrap();
    assert_eq!(bits.len(), X_BASE_BITS);
    assert!(bits.iter().all(|&b| b == F::ZERO || b == F::ONE));
    assert_eq!(StepPublicInput::decode(&bits, StackShape::NONE).unwrap(), x);

    // With stacks (v3.1): sp slots ride the tail and round-trip.
    let shape = StackShape { count: 2, sigma: 4 };
    let with_sp = StepPublicInput {
        sp_in: [3, 15],
        sp_out: [4, 14],
        ..x.clone()
    };
    let bits = with_sp.encode(shape).unwrap();
    assert_eq!(bits.len(), shape.x_bits());
    assert_eq!(StepPublicInput::decode(&bits, shape).unwrap(), with_sp);
    // An sp beyond σ bits is rejected at encode time, like every counter.
    let bad_sp = StepPublicInput {
        sp_in: [16, 0],
        ..with_sp.clone()
    };
    assert!(matches!(bad_sp.encode(shape), Err(LayoutError::FieldRange { .. })));

    // Out-of-width counters are rejected at encode time.
    let bad = StepPublicInput {
        ts_in: 1 << TS_BITS,
        ..x
    };
    assert!(matches!(
        bad.encode(StackShape::NONE),
        Err(LayoutError::FieldRange { .. })
    ));
}

/// v3.1 stack ops round-trip through the ops lane, and the decoder
/// polices the one-hot selector encoding.
#[test]
fn stack_ops_round_trip_and_selectors_are_one_hot() {
    let p = NebulaParams::test_profile()
        .with_stacks(2, 4)
        .expect("stacks");
    let ops = [
        MemOpRecord {
            is_write: true, // push 7 onto stack 0 at sp = 0
            space: MemSpace::Stack(0),
            addr: 0,
            v_r: 0,
            v_w: 7,
            rt: 0,
        },
        MemOpRecord {
            is_write: false, // pop it back
            space: MemSpace::Stack(0),
            addr: 0,
            v_r: 7,
            v_w: 7,
            rt: 1,
        },
        MemOpRecord {
            is_write: false, // an ordinary ROM read beside them
            space: MemSpace::Rom,
            addr: 3,
            v_r: 9,
            v_w: 9,
            rt: 0,
        },
    ];
    let lane = p.encode_ops_lane(&ops).unwrap();
    assert_eq!(p.decode_ops_lane(&lane).unwrap(), ops);

    // Setting a second selector bit on slot 0 (ram beside stk_0) breaks
    // the one-hot rule.
    let mut two = lane.clone();
    two[2] = F::ONE; // slot 0's ram bit (pad, is_write, ram, ...)
    assert!(matches!(
        p.decode_ops_lane(&two),
        Err(LayoutError::SelectorNotOneHot(0))
    ));

    // A stack op addressed outside its namespace is rejected at encode.
    let oob = MemOpRecord {
        addr: 16, // 2^σ = 16
        ..ops[0]
    };
    assert!(matches!(p.encode_ops_lane(&[oob]), Err(LayoutError::AddrRange { .. })));
    let missing = MemOpRecord {
        space: MemSpace::Stack(2), // plan has stacks 0 and 1
        ..ops[0]
    };
    assert!(matches!(
        p.encode_ops_lane(&[missing]),
        Err(LayoutError::StackIndex { .. })
    ));
}
