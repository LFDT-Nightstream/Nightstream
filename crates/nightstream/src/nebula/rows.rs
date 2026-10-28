//! The polynomial rows of a memory invocation as Lean expression trees over
//! witness wires: the memory rows of `Lifecycle/Nebula/RowList.lean`
//! (`RowExprs.lean` for the symbols) and the machine's rows of
//! `MachineApp.lean`. The trees match Lean node for node, because the lowering
//! of a generic row depends on its shape. It does not own the hash rows.

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use super::expr::{Expr, KExpr};
use super::plan::Plan;
use super::words::{self, Layout};
use crate::application::Variable;

/// The symbols of `RowExprs.lean` over the witness wires of one plan.
pub(super) struct Sym<'a> {
    plan: &'a Plan,
    layout: Layout,
    wires: &'a [Variable],
    headers: [F; 12],
}

fn c(value: u64) -> Expr {
    Expr::constant(value)
}

fn power_of_two(exponent: usize) -> Expr {
    Expr::Const(F::TWO.exp_u64(exponent as u64))
}

/// `x · (x − 1)`.
fn bit_row(x: Expr) -> Expr {
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

/// `Expr.sum`: `e₀ + (e₁ + (… + 0))`.
fn sum(terms: Vec<Expr>) -> Expr {
    terms.into_iter().rev().fold(c(0), |rest, e| e + rest)
}

fn fingerprint(eta1: KExpr, eta2: KExpr, eta1_sq: KExpr, t: Expr, g: Expr, v: Expr) -> KExpr {
    KExpr::embed(g)
        .add(eta1.mul(KExpr::embed(v)))
        .add(eta1_sq.mul(KExpr::embed(t)))
        .sub(eta2)
}

/// `gatedE pad f = pad + (1 − pad) · f`.
fn gated(pad: Expr, f: KExpr) -> KExpr {
    KExpr::embed(pad.clone()).add(KExpr::embed(c(1) - pad).mul(f))
}

impl<'a> Sym<'a> {
    pub(super) fn new(plan: &'a Plan, wires: &'a [Variable], headers: [F; 12]) -> Self {
        Self {
            plan,
            layout: Layout::new(plan),
            wires,
            headers,
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

    fn eta1(&self) -> KExpr {
        KExpr::of(self.c_out(3), self.c_out(4))
    }

    fn eta2(&self) -> KExpr {
        KExpr::of(self.c_out(5), self.c_out(6))
    }

    fn eta1_sq(&self) -> KExpr {
        KExpr::of(self.word(words::ETA1_SQUARE), self.word(words::ETA1_SQUARE + 1))
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
        self.bits(j, Layout::op_addr, self.plan.mu)
    }

    fn vr_bits(&self, j: usize) -> Vec<Expr> {
        self.bits(j, Layout::op_vr, 32)
    }

    fn vw_bits(&self, j: usize) -> Vec<Expr> {
        self.bits(j, Layout::op_vw, 32)
    }

    fn rt_bits(&self, j: usize) -> Vec<Expr> {
        self.bits(j, Layout::op_rt, self.plan.w_ts)
    }

    fn diff_bits(&self, j: usize) -> Vec<Expr> {
        self.bits(j, Layout::op_diff, self.plan.w_ts)
    }

    /// The lane bits of operation slot `j` (spec §6.3).
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
        (0..self.plan.w_ts)
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
            (0..self.plan.w_ts)
                .map(|k| self.word(self.layout.ts_bits(k)))
                .collect(),
        )
    }

    fn seg_word(&self) -> Expr {
        chunk(
            (0..self.plan.seg_width())
                .map(|k| self.word(self.layout.seg_bits(k)))
                .collect(),
        )
    }

    fn cnt_before(&self, k: usize) -> Expr {
        sum((0..self.plan.b_ops.min(k))
            .map(|j| c(1) - self.pad(j))
            .collect())
    }

    fn wt(&self, j: usize) -> Expr {
        self.c_in(2) + self.cnt_before(j + 1)
    }

    fn global_index(&self, j: usize) -> Expr {
        chunk(self.addr_bits(j)) + self.is_ram(j) * c(self.plan.rom_size() as u64)
    }

    fn scan_index(&self, j: usize) -> Expr {
        self.idx_eff() * c(self.plan.b_scan as u64) + c(j as u64)
    }

    fn start_product(&self, k: usize) -> KExpr {
        KExpr::embed(self.is_open())
            .add(KExpr::embed(c(1) - self.is_open()).mul(KExpr::of(self.c_in(k), self.c_in(k + 1))))
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
            self.start_product(start + 2 * pair)
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
        self.product_after(7, pair, k, self.plan.b_ops, Layout::ops_products)
    }

    fn scan_product_after(&self, pair: usize, k: usize) -> KExpr {
        self.product_after(11, pair, k, self.plan.b_scan, Layout::scan_products)
    }

    /// `memoryRows`: bits, arm, slots, scans, close, in that order.
    pub(super) fn memory_rows(&self) -> Vec<Expr> {
        let mut rows = self.bit_rows();
        rows.extend(self.arm_rows());
        for j in 0..self.plan.b_ops {
            rows.extend(self.slot_rows(j));
        }
        for j in 0..self.plan.b_scan {
            rows.extend(self.scan_rows(j));
        }
        rows.extend(self.close_rows());
        rows
    }

    fn bit_rows(&self) -> Vec<Expr> {
        let plan = self.plan;
        let mut rows: Vec<Expr> = (0..plan.b_ops)
            .flat_map(|j| self.op_lane(j))
            .map(bit_row)
            .collect();
        rows.extend((0..plan.b_ops).flat_map(|j| self.diff_bits(j)).map(bit_row));
        rows.extend(
            (0..plan.b_scan)
                .flat_map(|j| self.scan_lane(self.initial(j)))
                .map(bit_row),
        );
        rows.extend(
            (0..plan.b_scan)
                .flat_map(|j| self.scan_lane(self.final_(j)))
                .map(bit_row),
        );
        rows.extend((0..plan.w_ts).map(|k| bit_row(self.word(self.layout.ts_bits(k)))));
        rows.extend((0..plan.seg_width()).map(|k| bit_row(self.word(self.layout.seg_bits(k)))));
        rows.extend([bit_row(self.is_open()), bit_row(self.is_close())]);
        rows
    }

    fn arm_rows(&self) -> Vec<Expr> {
        let plan = self.plan;
        let (is_open, n) = (self.is_open(), c(plan.n as u64));
        let mut rows = vec![
            (self.c_in(1) - n.clone()) * is_open.clone(),
            (self.c_in(1) - n) * self.word(words::OPEN_INVERSE) - (c(1) - is_open.clone()),
            is_open.clone() * (c(plan.s_max as u64 - 1) - self.c_in(0) - self.seg_word()),
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
        rows.extend(self.eta1_sq().equalities(self.eta1().mul(self.eta1())));
        rows
    }

    fn slot_rows(&self, j: usize) -> Vec<Expr> {
        let plan = self.plan;
        let (pad, is_write, is_ram) = (self.pad(j), self.is_write(j), self.is_ram(j));
        let (vr, vw) = (chunk(self.vr_bits(j)), chunk(self.vw_bits(j)));
        let mut rows = vec![
            (c(1) - is_write.clone()) * (vw.clone() - vr.clone()),
            (c(1) - pad.clone()) * (self.wt(j) - chunk(self.rt_bits(j)) - c(1) - chunk(self.diff_bits(j))),
            is_write * (c(1) - is_ram.clone()),
        ];
        rows.extend(
            self.addr_bits(j)
                .into_iter()
                .enumerate()
                .filter(|(k, _)| plan.r <= *k)
                .map(|(_, bit)| (c(1) - is_ram.clone()) * bit),
        );
        rows.extend(self.op_lane(j).into_iter().skip(1).map(|x| pad.clone() * x));
        for (pair, stamp, value) in [(0, chunk(self.rt_bits(j)), vr), (1, self.wt(j), vw)] {
            let factor = gated(
                pad.clone(),
                fingerprint(
                    self.eta1(),
                    self.eta2(),
                    self.eta1_sq(),
                    stamp,
                    self.global_index(j),
                    value,
                ),
            );
            rows.extend(
                self.ops_product_after(pair, j + 1)
                    .equalities(self.ops_product_after(pair, j).mul(factor)),
            );
        }
        rows
    }

    fn scan_rows(&self, j: usize) -> Vec<Expr> {
        let mut rows = Vec::new();
        for (pair, base) in [(0, self.initial(j)), (1, self.final_(j))] {
            let factor = fingerprint(
                self.eta1(),
                self.eta2(),
                self.eta1_sq(),
                chunk(self.scan_stamp_bits(base)),
                self.scan_index(j),
                chunk(self.scan_value_bits(base)),
            );
            rows.extend(
                self.scan_product_after(pair, j + 1)
                    .equalities(self.scan_product_after(pair, j).mul(factor)),
            );
        }
        rows
    }

    fn close_rows(&self) -> Vec<Expr> {
        let plan = self.plan;
        let is_close = self.is_close();
        let k_out = |a: usize| KExpr::of(self.c_out(a), self.c_out(a + 1));
        let mut rows = vec![
            self.c_out(2) - (self.c_in(2) + self.cnt_before(plan.b_ops)),
            self.c_out(2) - self.ts_word(),
            self.c_out(1) - (self.idx_eff() + c(1)),
        ];
        rows.extend(k_out(7).equalities(self.ops_product_after(0, plan.b_ops)));
        rows.extend(k_out(9).equalities(self.ops_product_after(1, plan.b_ops)));
        rows.extend(k_out(11).equalities(self.scan_product_after(0, plan.b_scan)));
        rows.extend(k_out(13).equalities(self.scan_product_after(1, plan.b_scan)));
        let n = c(plan.n as u64);
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

    /// The machine's rows (`Machine.rows`) with each local variable at its
    /// witness word (`Words.appVar`).
    pub(super) fn machine_rows(&self) -> Vec<Expr> {
        let layout = &self.layout;
        let input = |i: usize| self.word(layout.app_in(i));
        let output = |i: usize| self.word(layout.app_out(i));
        let idle = self.word(layout.idle());
        let slot = |j: usize, k: usize| self.op(j, k);
        let addr = |j: usize| chunk(self.addr_bits(j));
        let vr = |j: usize| chunk(self.vr_bits(j));
        let vw = |j: usize| chunk(self.vw_bits(j));
        let instruction = self.vr_bits(0);
        let is_load = instruction[0].clone() * (c(1) - instruction[1].clone());
        let is_store = (c(1) - instruction[0].clone()) * instruction[1].clone();
        let is_loadi = instruction[0].clone() * instruction[1].clone();
        let argument = chunk(instruction[2..].to_vec());
        let touches = || is_load.clone() + is_store.clone();
        vec![
            bit_row(idle.clone()),
            slot(0, 0) - idle.clone(),
            slot(0, 1),
            slot(0, 2),
            addr(0) - (c(1) - idle) * input(0),
            slot(1, 0) - (c(1) - touches()),
            slot(1, 1) - is_store.clone(),
            slot(1, 2) - touches(),
            addr(1) - touches() * argument.clone(),
            vw(1) - (is_store.clone() * input(1) + is_load.clone() * vr(1)),
            output(0) - (input(0) + is_load.clone() + is_store + is_loadi.clone()),
            output(1)
                - (is_load.clone() * vr(1) + is_loadi.clone() * argument + (c(1) - is_load - is_loadi) * input(1)),
        ]
    }
}
