//! `S_mem` circuit tests — spec §4 rows, honest and adversarial, plus the
//! PR-3 cost gate (spec §10's 2× rule as an executable assertion).
//!
//! The load-bearing test is `full_segment_chains_and_matches_oracle`: every
//! step of a native segment satisfies the circuit, steps chain through
//! their public inputs, and the chained products equal the trace oracle's —
//! circuit semantics and native semantics cannot drift without this failing.

use neo_ccs::relations::check_ccs_rowwise_zero;
use neo_fold_clean::frontends::nebula::circuit::{SMemCircuit, StepData};
use neo_fold_clean::frontends::nebula::fingerprint::Gammas;
use neo_fold_clean::frontends::nebula::layout::{MemOpRecord, NebulaParams, H_RS};
use neo_fold_clean::frontends::nebula::trace::{Memory, SegmentTrace};
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

fn rom_image(p: &NebulaParams) -> Vec<u32> {
    (0..p.rom_cells() as u32)
        .map(|i| i.wrapping_mul(2654435761).wrapping_add(17))
        .collect()
}

/// Run a busy native segment (mixed RAM/ROM traffic) and return its trace.
fn native_segment(p: &NebulaParams, seed: u64) -> SegmentTrace {
    let mut mem = Memory::new(*p, &rom_image(p)).unwrap();
    let mut rng = Rng(seed);
    let mut seg = mem.begin_segment().unwrap();
    for _ in 0..60 {
        let addr = rng.next() % p.ram_cells();
        match rng.next() % 3 {
            0 => {
                seg.write(true, addr, rng.next() as u32).unwrap();
            }
            1 => {
                seg.read(true, addr).unwrap();
            }
            _ => {
                seg.read(false, rng.next() % p.rom_cells()).unwrap();
            }
        }
    }
    seg.finish()
}

/// Step `i`'s inputs from a segment trace plus the incoming carry.
fn step_data<'a>(trace: &'a SegmentTrace, i: usize, ts_in: u64, h_in: [K; 4]) -> StepData<'a> {
    let b_scan = trace.params().b_scan;
    StepData {
        seg_idx: trace.seg_idx,
        idx: i as u64,
        ts_in,
        h_in,
        ops: trace.step_ops(i),
        is_cells: &trace.is_cells[i * b_scan..(i + 1) * b_scan],
        fs_cells: &trace.fs_cells[i * b_scan..(i + 1) * b_scan],
    }
}

fn check(circuit: &SMemCircuit, z: &[F]) -> Result<(), neo_ccs::CcsError> {
    check_ccs_rowwise_zero(circuit.structure(), &z[..circuit.m_in()], &z[circuit.m_in()..])
}

#[test]
fn honest_step_satisfies() {
    let p = NebulaParams::test_profile();
    let circuit = SMemCircuit::new(p);
    let trace = native_segment(&p, 21);
    let gammas = Rng(1).gammas();

    let data = step_data(&trace, 0, trace.ts_in, [K::ONE; 4]);
    let (z, x) = circuit.witness(&gammas, &data).unwrap();
    check(&circuit, &z).unwrap();
    assert_eq!(x.ts_out, trace.ts_in + trace.step_ops(0).len() as u64);
}

#[test]
fn pad_only_step_satisfies() {
    // A step with zero ops: every op slot is a pad, products pass through.
    let p = NebulaParams::test_profile();
    let circuit = SMemCircuit::new(p);
    let trace = native_segment(&p, 22);
    let last = p.steps_per_segment() - 1; // ops are sequential-fill: empty
    let gammas = Rng(2).gammas();

    let data = step_data(&trace, last, trace.ts_out, [K::ONE; 4]);
    assert!(data.ops.is_empty());
    let (z, x) = circuit.witness(&gammas, &data).unwrap();
    check(&circuit, &z).unwrap();
    assert_eq!(x.ts_out, x.ts_in);
    assert_eq!(x.h_out[H_RS], K::ONE);
}

#[test]
fn full_segment_chains_and_matches_oracle() {
    let p = NebulaParams::test_profile();
    let circuit = SMemCircuit::new(p);
    let trace = native_segment(&p, 23);
    let gammas = Rng(3).gammas();

    let mut ts = trace.ts_in;
    let mut h = [K::ONE; 4];
    for i in 0..p.steps_per_segment() {
        let (z, x) = circuit
            .witness(&gammas, &step_data(&trace, i, ts, h))
            .unwrap();
        check(&circuit, &z).unwrap_or_else(|e| panic!("step {i}: {e:?}"));
        ts = x.ts_out;
        h = x.h_out;
    }
    assert_eq!(ts, trace.ts_out);
    assert_eq!(h, trace.products(&gammas));
    // And the segment close would accept: the Nebula balance holds.
    assert!(trace.balanced(&gammas));
}

#[test]
fn forged_ops_are_rejected_by_rows() {
    let p = NebulaParams::test_profile();
    let circuit = SMemCircuit::new(p);
    let trace = native_segment(&p, 24);
    let gammas = Rng(4).gammas();
    let honest = step_data(&trace, 0, trace.ts_in, [K::ONE; 4]);

    let forge = |mutate: &dyn Fn(&mut MemOpRecord)| {
        let mut ops = honest.ops.to_vec();
        mutate(&mut ops[2]);
        let (z, _) = circuit
            .witness(&gammas, &StepData { ops: &ops, ..honest })
            .unwrap();
        check(&circuit, &z)
    };

    // E5: write into ROM.
    assert!(forge(&|op| {
        op.seg = false;
        op.addr %= 16;
        op.is_write = true;
    })
    .is_err());
    // E6: ROM address past R.
    assert!(forge(&|op| {
        op.seg = false;
        op.addr = 16 + (op.addr % 16);
        op.is_write = false;
        op.v_w = op.v_r;
    })
    .is_err());
    // E3: a read that writes back a different value.
    assert!(forge(&|op| {
        op.is_write = false;
        op.v_w = op.v_r ^ 1;
    })
    .is_err());
    // E4: rt not strictly older than the write timestamp.
    assert!(forge(&|op| op.rt = 1 << 43).is_err());
}

#[test]
fn tampered_assignments_are_rejected_by_rows() {
    let p = NebulaParams::test_profile();
    let circuit = SMemCircuit::new(p);
    let trace = native_segment(&p, 25);
    let gammas = Rng(5).gammas();
    let data = step_data(&trace, 0, trace.ts_in, [K::ONE; 4]);
    let (z, _) = circuit.witness(&gammas, &data).unwrap();
    check(&circuit, &z).unwrap();

    // Boundary: claim a different outgoing RS product (flip one x bit).
    let mut wrong_h_out = z.clone();
    let h_out_col = 1 + neo_fold_clean::frontends::nebula::layout::x_offsets::H_OUT;
    wrong_h_out[h_out_col] = F::ONE - wrong_h_out[h_out_col];
    assert!(check(&circuit, &wrong_h_out).is_err());

    // Bitness: a non-bit witness coordinate.
    let mut non_bit = z.clone();
    non_bit[circuit.m_in() + 1] = F::from_u64(2);
    assert!(check(&circuit, &non_bit).is_err());

    // E7: turn a real op slot into a "pad" while its fields are nonzero.
    let mut fake_pad = z;
    let slot0 = circuit.op_slot_column(0);
    assert_eq!(fake_pad[slot0], F::ZERO);
    fake_pad[slot0] = F::ONE;
    assert!(check(&circuit, &fake_pad).is_err());
}

#[test]
fn cost_gate_within_spec_budget() {
    // Spec §10: `S_mem` ≈ 58k rows at v3 targets; off by 2× reopens the
    // spec. This test is that rule, executable.
    let v3 = SMemCircuit::new(NebulaParams::v3_targets());
    println!(
        "S_mem v3 targets: rows={} cols={} nnz={} (m_in={})",
        v3.rows(),
        v3.cols(),
        v3.nnz(),
        v3.m_in()
    );
    assert!(v3.rows() < 2 * 58_000, "rows {} blew the 2× spec budget", v3.rows());
    assert!(v3.rows() > 58_000 / 2, "rows {} — spec table is stale", v3.rows());

    let t = SMemCircuit::new(NebulaParams::test_profile());
    println!(
        "S_mem test profile: rows={} cols={} nnz={}",
        t.rows(),
        t.cols(),
        t.nnz()
    );
}
