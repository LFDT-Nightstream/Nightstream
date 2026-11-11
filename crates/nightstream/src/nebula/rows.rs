//! The polynomial rows of a memory invocation as Lean expression trees over
//! witness wires: the memory rows of `Lifecycle/Nebula/RowList.lean`
//! (`RowExprs.lean` for the symbols), and the `View` that an application's
//! rows read (`AppView.env` of `PortApp.lean`). The trees match Lean node for
//! node, because the lowering of a generic row depends on its shape. It does
//! not own the hash rows or any application's rows.

use std::collections::BTreeSet;

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use super::expr::{Expr, KExpr};
use super::geometry::Geometry;
use super::words::{self, Layout};
use crate::application::Variable;

/// The symbols of `RowExprs.lean` over the witness wires of one geometry.
pub(super) struct Sym<'a> {
    geometry: &'a Geometry,
    layout: Layout,
    wires: &'a [Variable],
    headers: [F; 12],
    /// Row O7: zero each field of an inactive slot.
    zero_idle: bool,
    derived: bool,
}

fn c(value: u64) -> Expr {
    Expr::constant(value)
}

fn power_of_two(exponent: usize) -> Expr {
    Expr::Const(F::TWO.exp_u64(exponent as u64))
}

/// `x · (x − 1)`.
pub(super) fn bit_row(x: Expr) -> Expr {
    x.clone() * (x - c(1))
}

/// `Expr.chunk`: `b₀ + 2·b₁ + 4·b₂ + …`, nested to the left.
pub(super) fn chunk(bits: Vec<Expr>) -> Expr {
    let mut bits = bits.into_iter();
    let Some(first) = bits.next() else {
        return c(0);
    };
    bits.enumerate()
        .fold(first, |acc, (k, bit)| acc + power_of_two(k + 1) * bit)
}

/// `Expr.pack`: chunks of 63 bits.
pub(super) fn pack(bits: Vec<Expr>) -> Vec<Expr> {
    bits.chunks(63).map(|bits| chunk(bits.to_vec())).collect()
}

impl<'a> Sym<'a> {
    pub(super) fn new(
        geometry: &'a Geometry,
        layout: Layout,
        wires: &'a [Variable],
        headers: [F; 12],
        zero_idle: bool,
        derived: bool,
    ) -> Self {
        Self {
            geometry,
            layout,
            wires,
            headers,
            zero_idle,
            derived,
        }
    }

    /// Witness word `k`, or `0` past the witness (`MemoryApp.wire`).
    pub(super) fn word(&self, k: usize) -> Expr {
        self.wires
            .get(k)
            .map_or_else(|| c(0), |wire| Expr::Var(*wire))
    }

    pub(super) fn c_in(&self, k: usize) -> Expr {
        if k < 39 {
            self.word(words::CARRY_IN + k)
        } else {
            c(0)
        }
    }

    pub(super) fn c_out(&self, k: usize) -> Expr {
        if k < 39 {
            self.word(words::CARRY_OUT + k)
        } else {
            c(0)
        }
    }

    fn is_open(&self) -> Expr {
        self.word(words::IS_OPEN)
    }

    fn is_close(&self) -> Expr {
        self.word(words::IS_CLOSE)
    }

    fn idx_eff(&self) -> Expr {
        self.word(words::IDX_EFF)
    }

    fn op(&self, j: usize, k: usize) -> Expr {
        self.word(self.layout.op(j, k))
    }

    fn pad(&self, j: usize) -> Expr {
        self.op(j, 0)
    }

    fn is_write(&self, j: usize) -> Expr {
        self.op(j, 1)
    }

    fn is_ram(&self, j: usize) -> Expr {
        self.op(j, 2)
    }

    fn bits(&self, j: usize, offset: fn(&Layout, usize) -> usize, count: usize) -> Vec<Expr> {
        (0..count)
            .map(|k| self.op(j, offset(&self.layout, k)))
            .collect()
    }

    fn addr_bits(&self, j: usize) -> Vec<Expr> {
        self.bits(j, Layout::op_addr, self.geometry.addr_width())
    }

    fn vr_bits(&self, j: usize) -> Vec<Expr> {
        self.bits(j, Layout::op_vr, 32)
    }

    fn vw_bits(&self, j: usize) -> Vec<Expr> {
        self.bits(j, Layout::op_vw, 32)
    }

    /// The lane's stamp field of slot `j`: `rt`, or `diff` when read
    /// stamps are derived.
    fn rt_bits(&self, j: usize) -> Vec<Expr> {
        self.bits(j, Layout::op_rt, self.geometry.w_ts)
    }

    /// The read stamp of slot `j`: the lane's `rt`, or `wt − 1 − diff`.
    fn read_stamp(&self, j: usize) -> Expr {
        if self.derived {
            self.wt(j) - c(1) - chunk(self.rt_bits(j))
        } else {
            chunk(self.rt_bits(j))
        }
    }

    fn diff_bits(&self, j: usize) -> Vec<Expr> {
        self.bits(j, Layout::op_diff, self.geometry.w_ts)
    }

    /// The lane bits of operation slot `j` (SPEC.md section 7.2).
    pub(super) fn op_lane(&self, j: usize) -> Vec<Expr> {
        let mut lane = vec![self.pad(j), self.is_write(j), self.is_ram(j)];
        lane.extend(self.addr_bits(j));
        lane.extend(self.vr_bits(j));
        lane.extend(self.vw_bits(j));
        lane.extend(self.rt_bits(j));
        lane
    }

    /// The lane bits of the scan slot at word `base`: value, then stamp.
    pub(super) fn scan_lane(&self, base: usize) -> Vec<Expr> {
        let mut lane = self.scan_value_bits(base);
        lane.extend(self.scan_stamp_bits(base));
        lane
    }

    fn scan_value_bits(&self, base: usize) -> Vec<Expr> {
        (0..32).map(|k| self.word(base + k)).collect()
    }

    fn scan_stamp_bits(&self, base: usize) -> Vec<Expr> {
        (0..self.geometry.w_ts)
            .map(|k| self.word(base + 32 + k))
            .collect()
    }

    pub(super) fn initial(&self, j: usize) -> usize {
        self.layout.initial(j, 0)
    }

    pub(super) fn final_(&self, j: usize) -> usize {
        self.layout.final_(j, 0)
    }

    fn ts_word(&self) -> Expr {
        chunk(
            (0..self.geometry.w_ts)
                .map(|k| self.word(self.layout.ts_bits(k)))
                .collect(),
        )
    }

    fn seg_word(&self) -> Expr {
        chunk(
            (0..self.geometry.seg_width())
                .map(|k| self.word(self.layout.seg_bits(k)))
                .collect(),
        )
    }

    /// The counter `cnt_j` of slot `j` (row O2).
    fn cnt(&self, j: usize) -> Expr {
        self.word(self.layout.cnt(j))
    }

    /// `cnt` before slot `k`: `0`, then the counter of slot `k − 1`.
    fn cnt_before(&self, k: usize) -> Expr {
        if k == 0 {
            c(0)
        } else {
            self.cnt(k - 1)
        }
    }

    fn wt(&self, j: usize) -> Expr {
        self.c_in(2) + self.cnt(j)
    }

    fn global_index(&self, j: usize) -> Expr {
        chunk(self.addr_bits(j)) + self.is_ram(j) * c(self.geometry.rom_size() as u64)
    }

    fn scan_index(&self, j: usize) -> Expr {
        self.idx_eff() * c(self.geometry.b_scan as u64) + c(j as u64)
    }

    /// Coefficient `c` of start product `s` (0 RS, 1 WS, 2 IS, 3 FS).
    fn start_word(&self, s: usize, k: usize) -> Expr {
        self.word(self.layout.start(s, k))
    }

    fn product_after(
        &self,
        start: usize,
        pair: usize,
        k: usize,
        count: usize,
        base: fn(&Layout, usize, usize) -> usize,
    ) -> KExpr {
        if k == 0 {
            KExpr::of(self.start_word(start + pair, 0), self.start_word(start + pair, 1))
        } else if k - 1 < count {
            KExpr::of(
                self.word(base(&self.layout, k - 1, 2 * pair % 4)),
                self.word(base(&self.layout, k - 1, (2 * pair + 1) % 4)),
            )
        } else {
            KExpr::zero()
        }
    }

    fn ops_product_after(&self, pair: usize, k: usize) -> KExpr {
        self.product_after(0, pair, k, self.geometry.b_ops, Layout::ops_products)
    }

    fn scan_product_after(&self, pair: usize, k: usize) -> KExpr {
        self.product_after(2, pair, k, self.geometry.b_scan, Layout::scan_products)
    }

    /// `Rows.fingerprintRows`: `(a 0, a 1) = f`. The word `v + 2^32 · t` is
    /// affine, so each row is `word − recipe` with one product and lowers to
    /// one R1CS row.
    fn fingerprint_rows(&self, a: &impl Fn(usize) -> Expr, t: Expr, g: Expr, v: Expr) -> Vec<Expr> {
        let packed = v + c(1 << 32) * t;
        vec![
            a(0) - (self.c_out(3) * packed.clone() + (g - self.c_out(5))),
            a(1) - (self.c_out(4) * packed - self.c_out(6)),
        ]
    }

    /// `Rows.productRows`: `next = prod · x` with the cross products
    /// `m = prod.c1 · x1` and `n = prod.c1 · x0`.
    fn product_rows(prod: KExpr, next: KExpr, x0: Expr, x1: Expr, m: Expr, n: Expr) -> Vec<Expr> {
        vec![
            m.clone() - prod.c1.clone() * x1.clone(),
            n.clone() - prod.c1 * x0.clone(),
            next.c0 - (prod.c0.clone() * x0 + c(7) * m),
            next.c1 - (prod.c0 * x1 + n),
        ]
    }

    /// `Rows.factorRows`: an ungated fingerprint step over four factor words.
    fn factor_rows(&self, a: impl Fn(usize) -> Expr, prod: KExpr, next: KExpr, t: Expr, g: Expr, v: Expr) -> Vec<Expr> {
        let mut rows = self.fingerprint_rows(&a, t, g, v);
        rows.extend(Self::product_rows(prod, next, a(0), a(1), a(2), a(3)));
        rows
    }

    /// `Rows.gatedRows`: a gated fingerprint step over six factor words.
    fn gated_rows(
        &self,
        a: impl Fn(usize) -> Expr,
        pad: Expr,
        prod: KExpr,
        next: KExpr,
        t: Expr,
        g: Expr,
        v: Expr,
    ) -> Vec<Expr> {
        let mut rows = self.fingerprint_rows(&a, t, g, v);
        rows.push(a(2) - ((c(1) - pad.clone()) * a(0) + pad.clone()));
        rows.push(a(3) - (c(1) - pad) * a(1));
        rows.extend(Self::product_rows(prod, next, a(2), a(3), a(4), a(5)));
        rows
    }

    /// `memoryRows`: bits, arm, slots, scans, close, in that order, each
    /// group with its name.
    pub(super) fn memory_row_groups(&self) -> [(&'static str, Vec<Expr>); 5] {
        [
            ("Bit rows", self.bit_rows()),
            ("Arm rows", self.arm_rows()),
            (
                "Operation slot rows",
                (0..self.geometry.b_ops)
                    .flat_map(|j| self.slot_rows(j))
                    .collect(),
            ),
            (
                "Scan slot rows",
                (0..self.geometry.b_scan)
                    .flat_map(|j| self.scan_rows(j))
                    .collect(),
            ),
            ("Close rows", self.close_rows()),
        ]
    }

    /// The words that the bit rows force to 0 or 1, in row order.
    fn bit_words(&self) -> Vec<Expr> {
        let geometry = self.geometry;
        let mut words: Vec<Expr> = (0..geometry.b_ops).flat_map(|j| self.op_lane(j)).collect();
        if !self.derived {
            words.extend((0..geometry.b_ops).flat_map(|j| self.diff_bits(j)));
        }
        words.extend((0..geometry.b_scan).flat_map(|j| self.scan_lane(self.initial(j))));
        words.extend((0..geometry.b_scan).flat_map(|j| self.scan_lane(self.final_(j))));
        words.extend((0..geometry.w_ts).map(|k| self.word(self.layout.ts_bits(k))));
        words.extend((0..geometry.seg_width()).map(|k| self.word(self.layout.seg_bits(k))));
        words.extend([self.is_open(), self.is_close()]);
        words
    }

    fn bit_rows(&self) -> Vec<Expr> {
        self.bit_words().into_iter().map(bit_row).collect()
    }

    fn arm_rows(&self) -> Vec<Expr> {
        let geometry = self.geometry;
        let (is_open, n) = (self.is_open(), c(geometry.n as u64));
        let mut rows = vec![
            (self.c_in(1) - n.clone()) * is_open.clone(),
            (self.c_in(1) - n) * self.word(words::OPEN_INVERSE) - (c(1) - is_open.clone()),
            is_open.clone() * (c(geometry.s_max as u64 - 1) - self.c_in(0) - self.seg_word()),
            self.idx_eff() - (c(1) - is_open.clone()) * self.c_in(1),
        ];
        rows.extend((0..12).map(|i| {
            self.word(words::SEEN_PREV + i)
                - (is_open.clone() * Expr::Const(self.headers[i]) + (c(1) - is_open.clone()) * self.c_in(23 + i))
        }));
        rows.extend((0..8).map(|i| {
            self.c_out(15 + i)
                - (is_open.clone() * self.word(words::PROPOSAL + i) + (c(1) - is_open.clone()) * self.c_in(15 + i))
        }));
        rows.extend((0..4).map(|i| {
            self.c_out(3 + i)
                - (is_open.clone() * self.word(words::ETA_FRESH + i) + (c(1) - is_open.clone()) * self.c_in(3 + i))
        }));
        for s in 0..4 {
            rows.push(self.start_word(s, 0) - ((c(1) - is_open.clone()) * self.c_in(7 + 2 * s) + is_open.clone()));
            rows.push(self.start_word(s, 1) - (c(1) - is_open.clone()) * self.c_in(8 + 2 * s));
        }
        rows
    }

    fn slot_rows(&self, j: usize) -> Vec<Expr> {
        let geometry = self.geometry;
        let (pad, is_write, is_ram) = (self.pad(j), self.is_write(j), self.is_ram(j));
        let (vr, vw) = (chunk(self.vr_bits(j)), chunk(self.vw_bits(j)));
        let mut rows = vec![
            self.cnt(j) - (self.cnt_before(j) + (c(1) - pad.clone())),
            (c(1) - is_write.clone()) * (vw.clone() - vr.clone()),
        ];
        if !self.derived {
            rows.push((c(1) - pad.clone()) * (self.wt(j) - chunk(self.rt_bits(j)) - c(1) - chunk(self.diff_bits(j))));
        }
        rows.push(is_write * (c(1) - is_ram.clone()));
        rows.extend(
            self.addr_bits(j)
                .into_iter()
                .enumerate()
                .filter(|(k, _)| geometry.r <= *k)
                .map(|(_, bit)| (c(1) - is_ram.clone()) * bit),
        );
        rows.extend(
            self.addr_bits(j)
                .into_iter()
                .enumerate()
                .filter(|(k, _)| geometry.mu <= *k)
                .map(|(_, bit)| is_ram.clone() * bit),
        );
        if self.zero_idle {
            rows.extend(self.op_lane(j).into_iter().skip(1).map(|x| pad.clone() * x));
        }
        for (pair, stamp, value) in [(0, self.read_stamp(j), vr), (1, self.wt(j), vw)] {
            rows.extend(self.gated_rows(
                |k| self.word(self.layout.ops_factor(j, pair, k)),
                pad.clone(),
                self.ops_product_after(pair, j),
                self.ops_product_after(pair, j + 1),
                stamp,
                self.global_index(j),
                value,
            ));
        }
        rows
    }

    fn scan_rows(&self, j: usize) -> Vec<Expr> {
        let mut rows = Vec::new();
        for (pair, base) in [(0, self.initial(j)), (1, self.final_(j))] {
            rows.extend(self.factor_rows(
                |k| self.word(self.layout.scan_factor(j, pair, k)),
                self.scan_product_after(pair, j),
                self.scan_product_after(pair, j + 1),
                chunk(self.scan_stamp_bits(base)),
                self.scan_index(j),
                chunk(self.scan_value_bits(base)),
            ));
        }
        rows
    }

    fn close_rows(&self) -> Vec<Expr> {
        let geometry = self.geometry;
        let is_close = self.is_close();
        let k_out = |a: usize| KExpr::of(self.c_out(a), self.c_out(a + 1));
        let mut rows = vec![
            self.c_out(2) - (self.c_in(2) + self.cnt_before(geometry.b_ops)),
            self.c_out(2) - self.ts_word(),
            self.c_out(1) - (self.idx_eff() + c(1)),
        ];
        rows.extend(k_out(7).equalities(self.ops_product_after(0, geometry.b_ops)));
        rows.extend(k_out(9).equalities(self.ops_product_after(1, geometry.b_ops)));
        rows.extend(k_out(11).equalities(self.scan_product_after(0, geometry.b_scan)));
        rows.extend(k_out(13).equalities(self.scan_product_after(1, geometry.b_scan)));
        let n = c(geometry.n as u64);
        rows.push((self.c_out(1) - n.clone()) * is_close.clone());
        rows.push((self.c_out(1) - n) * self.word(words::CLOSE_INVERSE) - (c(1) - is_close.clone()));
        rows.extend((0..4).map(|i| is_close.clone() * (self.c_out(23 + i) - self.c_out(15 + i))));
        rows.extend((0..4).map(|i| is_close.clone() * (self.c_out(27 + i) - self.c_in(35 + i))));
        rows.extend((0..4).map(|i| is_close.clone() * (self.c_out(31 + i) - self.c_out(19 + i))));
        rows.extend(
            KExpr::embed(is_close.clone())
                .mul(k_out(11).mul(k_out(9)).sub(k_out(7).mul(k_out(13))))
                .equalities(KExpr::zero()),
        );
        rows.push(self.c_out(0) - (self.c_in(0) + is_close.clone()));
        rows.extend((0..4).map(|i| {
            self.c_out(35 + i) - (is_close.clone() * self.c_out(19 + i) + (c(1) - is_close.clone()) * self.c_in(35 + i))
        }));
        rows
    }
}

/// The values that an application's rows read (Lean `AppView.env`): its input
/// and output state words, its private words, and each operation slot. Local
/// variable `k` is witness word `Layout::app_var(k)`, so a slot past `B_ops`
/// reads zero, as in Lean. The index sums saturate: an index past the witness
/// reads zero, as a Lean natural number does, and never wraps onto a word.
pub struct View<'a> {
    sym: &'a Sym<'a>,
}

impl<'a> View<'a> {
    pub(super) fn new(sym: &'a Sym<'a>) -> Self {
        Self { sym }
    }

    fn local(&self, k: usize) -> Expr {
        self.sym.word(self.sym.layout.app_var(k))
    }

    /// The variables of the view's words. An application row reads no
    /// other: not `η`, the carry, or any other memory word.
    pub(super) fn variables(&self) -> BTreeSet<Variable> {
        (0..self.sym.layout.slot_local(self.sym.geometry.b_ops, 0))
            .filter_map(|k| self.sym.wires.get(self.sym.layout.app_var(k)).copied())
            .collect()
    }

    fn slot(&self, j: usize, k: usize) -> Expr {
        self.local(self.sym.layout.slot_local(j, k))
    }

    fn slot_bits(&self, j: usize, offset: fn(&Layout, usize) -> usize, count: usize) -> Vec<Expr> {
        (0..count)
            .map(|k| self.slot(j, offset(&self.sym.layout, k)))
            .collect()
    }

    /// Input state word `i`.
    pub fn input(&self, i: usize) -> Expr {
        self.local(i)
    }

    /// Output state word `i`.
    pub fn output(&self, i: usize) -> Expr {
        self.local(self.sym.layout.width().saturating_add(i))
    }

    /// Private word `i`.
    pub fn aux(&self, i: usize) -> Expr {
        self.local((2 * self.sym.layout.width()).saturating_add(i))
    }

    /// The pad bit of operation slot `j`: one for an inactive slot.
    pub fn pad(&self, j: usize) -> Expr {
        self.slot(j, 0)
    }

    pub fn is_write(&self, j: usize) -> Expr {
        self.slot(j, 1)
    }

    pub fn is_ram(&self, j: usize) -> Expr {
        self.slot(j, 2)
    }

    pub(super) fn geometry(&self) -> &Geometry {
        self.sym.geometry
    }

    /// The address bits of slot `j`, least significant first.
    pub(super) fn addr_bits(&self, j: usize) -> Vec<Expr> {
        self.slot_bits(j, Layout::op_addr, self.sym.geometry.addr_width())
    }

    /// The address of slot `j` from its `μ` bits.
    pub fn addr(&self, j: usize) -> Expr {
        chunk(self.addr_bits(j))
    }

    /// The 32 bits of the value that slot `j` reads, least significant first.
    pub fn vr_bits(&self, j: usize) -> Vec<Expr> {
        self.slot_bits(j, Layout::op_vr, 32)
    }

    /// The value that slot `j` reads.
    pub fn vr(&self, j: usize) -> Expr {
        chunk(self.vr_bits(j))
    }

    /// The value that slot `j` writes.
    pub fn vw(&self, j: usize) -> Expr {
        chunk(self.slot_bits(j, Layout::op_vw, 32))
    }
}

#[cfg(test)]
#[path = "../../tests/nebula/idle_slots.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/nebula/derived_stamps.rs"]
mod derived_stamps;
