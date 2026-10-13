//! The witness word layout of one memory-application step. Mirrors
//! `Lifecycle/Nebula/WitnessWords.lean` index by index: the memory words
//! first, then the machine's words (`Lifecycle/Nebula/MachineApp.lean`): two
//! input state words, two output state words, and the idle flag.

use super::plan::Plan;

pub const CARRY_IN: usize = 0;
pub const CARRY_OUT: usize = 39;
pub const PROPOSAL: usize = 78;
pub const IS_OPEN: usize = 86;
pub const OPEN_INVERSE: usize = 87;
pub const IS_CLOSE: usize = 88;
pub const CLOSE_INVERSE: usize = 89;
pub const ETA_FRESH: usize = 90;
pub const ETA1_SQUARE: usize = 94;
pub const SEEN_PREV: usize = 96;
pub const IDX_EFF: usize = 108;
const OPS_START: usize = 109;

/// The machine's state words `(pc, acc)`.
pub(super) const APP_WIDTH: usize = 2;

/// Offsets that depend on the plan.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    op_words: usize,
    op_width: usize,
    mu: usize,
    scan_width: usize,
    initial_start: usize,
    final_start: usize,
    ts_bits: usize,
    seg_bits: usize,
    ops_products: usize,
    scan_products: usize,
    memory_count: usize,
}

impl Layout {
    pub fn new(plan: &Plan) -> Self {
        let op_width = plan.op_width();
        let op_words = op_width + plan.w_ts;
        let scan_width = plan.scan_width();
        let initial_start = OPS_START + plan.b_ops * op_words;
        let final_start = initial_start + plan.b_scan * scan_width;
        let ts_bits = final_start + plan.b_scan * scan_width;
        let seg_bits = ts_bits + plan.w_ts;
        let ops_products = seg_bits + plan.seg_width();
        let scan_products = ops_products + 4 * plan.b_ops;
        let memory_count = scan_products + 4 * plan.b_scan;
        Self {
            op_words,
            op_width,
            mu: plan.mu,
            scan_width,
            initial_start,
            final_start,
            ts_bits,
            seg_bits,
            ops_products,
            scan_products,
            memory_count,
        }
    }

    pub fn count(&self) -> usize {
        self.memory_count + 2 * APP_WIDTH + 1
    }

    /// Input state word `k` of the machine.
    pub fn app_in(&self, k: usize) -> usize {
        self.memory_count + k
    }

    /// Output state word `k` of the machine.
    pub fn app_out(&self, k: usize) -> usize {
        self.memory_count + APP_WIDTH + k
    }

    /// The machine's idle flag, its one private word.
    pub fn idle(&self) -> usize {
        self.memory_count + 2 * APP_WIDTH
    }

    /// Word `k` of operation slot `j`: pad, is_write, is_ram, addr, v_r, v_w, rt, diff.
    pub fn op(&self, j: usize, k: usize) -> usize {
        OPS_START + j * self.op_words + k
    }

    pub fn op_addr(&self, k: usize) -> usize {
        3 + k
    }

    pub fn op_vr(&self, k: usize) -> usize {
        3 + self.mu + k
    }

    pub fn op_vw(&self, k: usize) -> usize {
        3 + self.mu + 32 + k
    }

    pub fn op_rt(&self, k: usize) -> usize {
        3 + self.mu + 64 + k
    }

    pub fn op_diff(&self, k: usize) -> usize {
        self.op_width + k
    }

    /// Word `k` of IS slot `j`: value bits, then stamp bits.
    pub fn initial(&self, j: usize, k: usize) -> usize {
        self.initial_start + j * self.scan_width + k
    }

    /// Word `k` of FS slot `j`.
    pub fn final_(&self, j: usize, k: usize) -> usize {
        self.final_start + j * self.scan_width + k
    }

    pub fn ts_bits(&self, k: usize) -> usize {
        self.ts_bits + k
    }

    pub fn seg_bits(&self, k: usize) -> usize {
        self.seg_bits + k
    }

    pub fn ops_products(&self, j: usize, c: usize) -> usize {
        self.ops_products + 4 * j + c
    }

    pub fn scan_products(&self, j: usize, c: usize) -> usize {
        self.scan_products + 4 * j + c
    }
}
