//! Derived read stamps (scratch prototype): the lane carries `diff` in place
//! of `rt`, the read stamp is `wt − 1 − diff`, and row O4 and the `rt` bits
//! are gone. These tests show which rows reject the attacks that O4 and the
//! `rt` bits rejected before.

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use super::{Layout, Sym};
use crate::application::Variable;
use crate::nebula::{words, Expr, Geometry};

fn eval(expr: &Expr, words: &[F]) -> F {
    match expr {
        Expr::Var(variable) => words[variable.index()],
        Expr::Const(value) => *value,
        Expr::Add(left, right) => eval(left, words) + eval(right, words),
        Expr::Mul(left, right) => eval(left, words) * eval(right, words),
    }
}

fn set_bits(words: &mut [F], position: impl Fn(usize) -> usize, count: usize, value: u64) {
    for k in 0..count {
        words[position(k)] = F::from_u64((value >> k) & 1);
    }
}

/// `η1 = 3 + 5u`, `η2 = 11 + 13u`.
const ETA: [u64; 4] = [3, 5, 11, 13];
const TS: u64 = 9;
const WT: u64 = TS + 1;
const VALUE: u64 = 7;
const ADDR: u64 = 5;

/// 9-bit stamps, so a stamp above the write stamp fits the lane.
fn geometry() -> Geometry {
    let geometry = Geometry::new(4, 4, 2, 2, 64).expect("a valid geometry");
    assert_eq!(geometry.w_ts, 9);
    geometry
}

/// Slot 0 is an active RAM read of cell 5 (value 7) at carry `ts` 9, so its
/// write stamp is 10. Its lane stamp field holds `lane` (`diff` when read
/// stamps are derived, else `rt`, with `diff` in the auxiliary bits), and its
/// read fingerprint absorbs stamp `read`.
fn read_slot(layout: &Layout, lane: u64, diff: u64, read: F) -> Vec<F> {
    let geometry = geometry();
    let mut w = vec![F::ZERO; layout.count()];
    w[layout.op(0, 2)] = F::ONE;
    set_bits(&mut w, |k| layout.op(0, layout.op_addr(k)), geometry.addr_width(), ADDR);
    set_bits(&mut w, |k| layout.op(0, layout.op_vr(k)), 32, VALUE);
    set_bits(&mut w, |k| layout.op(0, layout.op_vw(k)), 32, VALUE);
    set_bits(&mut w, |k| layout.op(0, layout.op_rt(k)), geometry.w_ts, lane);
    set_bits(&mut w, |k| layout.op(0, layout.op_diff(k)), geometry.w_ts, diff);
    w[words::CARRY_IN + 2] = F::from_u64(TS);
    for (i, e) in ETA.iter().enumerate() {
        w[words::CARRY_OUT + 3 + i] = F::from_u64(*e);
    }
    w[layout.cnt(0)] = F::ONE;
    w[layout.cnt(1)] = F::ONE;
    w[layout.start(0, 0)] = F::ONE;
    w[layout.start(1, 0)] = F::ONE;
    let g = F::from_u64(ADDR + geometry.rom_size() as u64);
    for (pair, t) in [(0, read), (1, F::from_u64(WT))] {
        let packed = F::from_u64(VALUE) + F::from_u64(1 << 32) * t;
        let a0 = F::from_u64(ETA[0]) * packed + g - F::from_u64(ETA[2]);
        let a1 = F::from_u64(ETA[1]) * packed - F::from_u64(ETA[3]);
        for (k, word) in [a0, a1, a0, a1].into_iter().enumerate() {
            w[layout.ops_factor(0, pair, k)] = word;
        }
        w[layout.ops_products(0, 2 * pair)] = a0;
        w[layout.ops_products(0, 2 * pair + 1)] = a1;
    }
    w
}

/// The slot rows of slot 0 that do not hold.
fn failing(derived: bool, w: &[F]) -> Vec<usize> {
    let geometry = geometry();
    let layout = Layout::new(&geometry, 0, 0);
    let wires: Vec<Variable> = (0..layout.count()).map(Variable).collect();
    let sym = Sym::new(&geometry, layout, &wires, [F::ZERO; 12], false, derived);
    sym.slot_rows(0)
        .iter()
        .enumerate()
        .filter(|(_, row)| eval(row, w) != F::ZERO)
        .map(|(i, _)| i)
        .collect()
}

fn layout() -> Layout {
    Layout::new(&geometry(), 0, 0)
}

#[test]
fn an_honest_read_holds_with_the_derived_stamp() {
    let rt = 6;
    let diff = WT - 1 - rt;
    let w = read_slot(&layout(), diff, 0, F::from_u64(rt));
    assert!(failing(true, &w).is_empty(), "stamp wt − 1 − diff");
}

#[test]
fn no_lane_diff_lets_a_read_absorb_a_stamp_at_or_after_its_write() {
    let geometry = geometry();
    for read in [WT, WT + 1, WT + 5] {
        for diff in 0..1u64 << geometry.w_ts {
            let w = read_slot(&layout(), diff, 0, F::from_u64(read));
            assert!(
                !failing(true, &w).is_empty(),
                "the read fingerprint rows bind the stamp to wt − 1 − diff (read {read}, diff {diff})"
            );
        }
    }
}

#[test]
fn the_same_future_read_needed_row_o4_before() {
    let geometry = geometry();
    let read = WT + 1;
    let wrapped = WT.wrapping_sub(read + 1) & ((1 << geometry.w_ts) - 1);
    let w = read_slot(&layout(), read, wrapped, F::from_u64(read));
    let rows = failing(false, &w);
    assert_eq!(rows, [2], "only row O4 rejects a lane rt above wt with a wrapped diff");
}

/// For `W_ts ≤ 30`, a derived stamp `t` in `(−2^W_ts, 2^W_ts)` and a value
/// below `2^32` pack to the word of an `IS ∪ WS` tuple `(t', v')` (stamp in
/// `[0, 2^W_ts)`) only when `(t, v) = (t', v')`: both sides of
/// `v − v' = 2^32 (t' − t)` are below `q / 2` in size, so the congruence is an
/// integer equation and `|v − v'| < 2^32` makes both sides zero.
#[test]
fn a_derived_stamp_packs_to_a_legal_tuple_only_when_equal() {
    let w = 30i64;
    let field = |t: i64| {
        if t < 0 {
            -F::from_u64(t.unsigned_abs())
        } else {
            F::from_u64(t as u64)
        }
    };
    let packed = |t: i64, v: u64| F::from_u64(v) + F::from_u64(1 << 32) * field(t);
    let stamps = [
        -(1i64 << w) + 1,
        -(1 << 29),
        -2,
        -1,
        0,
        1,
        2,
        (1 << 29) + 3,
        (1 << w) - 1,
    ];
    let values = [0u64, 1, 2, 1 << 31, (1 << 32) - 2, (1 << 32) - 1];
    for &t in &stamps {
        for &v in &values {
            for &t2 in stamps.iter().filter(|t| **t >= 0) {
                for &v2 in &values {
                    if packed(t, v) == packed(t2, v2) {
                        assert_eq!((t, v), (t2, v2));
                    }
                }
            }
        }
    }
    let wide = 1i64 << 32;
    assert_eq!(
        packed(-wide, (1 << 32) - 1),
        packed(0, 0),
        "a stamp 2^32 below a legal one aliases it, so the width bound is needed"
    );
}
