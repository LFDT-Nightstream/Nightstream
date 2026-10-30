//! The witness word layout of one memory-application step. Mirrors
//! `Lifecycle/Nebula/WitnessWords.lean` index by index: the memory words
//! first, then the application's input state words, its output state words,
//! and its private words. `app_var` mirrors `Words.appVar` of `PortApp.lean`.

use super::geometry::Geometry;

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

/// Offsets that depend on the geometry and on the application's word counts.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    b_ops: usize,
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
    width: usize,
    aux: usize,
}

impl Layout {
    /// The layout of an application with `width` state words and `aux`
    /// private words.
    pub fn new(geometry: &Geometry, width: usize, aux: usize) -> Self {
        let op_width = geometry.op_width();
        let op_words = op_width + geometry.w_ts;
        let scan_width = geometry.scan_width();
        let initial_start = OPS_START + geometry.b_ops * op_words;
        let final_start = initial_start + geometry.b_scan * scan_width;
        let ts_bits = final_start + geometry.b_scan * scan_width;
        let seg_bits = ts_bits + geometry.w_ts;
        let ops_products = seg_bits + geometry.seg_width();
        let scan_products = ops_products + 4 * geometry.b_ops;
        let memory_count = scan_products + 4 * geometry.b_scan;
        Self {
            b_ops: geometry.b_ops,
            op_words,
            op_width,
            mu: geometry.mu,
            scan_width,
            initial_start,
            final_start,
            ts_bits,
            seg_bits,
            ops_products,
            scan_products,
            memory_count,
            width,
            aux,
        }
    }

    pub fn count(&self) -> usize {
        self.memory_count + 2 * self.width + self.aux
    }

    /// The number of application state words.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The memory's witness regions and their word counts, in layout order.
    pub fn memory_regions(&self) -> [(&'static str, usize); 6] {
        [
            ("Carry, proposal, and control words", OPS_START),
            ("Operation slot words", self.initial_start - OPS_START),
            ("Scan slot words (initial and final)", self.ts_bits - self.initial_start),
            ("Timestamp and segment bits", self.ops_products - self.ts_bits),
            ("Operation product words", self.scan_products - self.ops_products),
            ("Scan product words", self.memory_count - self.scan_products),
        ]
    }

    /// Input state word `k` of the application.
    pub fn app_in(&self, k: usize) -> usize {
        self.memory_count + k
    }

    /// Output state word `k` of the application.
    pub fn app_out(&self, k: usize) -> usize {
        self.memory_count + self.width + k
    }

    /// Private word `k` of the application.
    pub fn aux(&self, k: usize) -> usize {
        self.memory_count + 2 * self.width + k
    }

    /// The application's local variable of word `k` of operation slot `j`
    /// (Lean `AppView.slotStart + j · opWords + k`).
    pub fn slot_local(&self, j: usize, k: usize) -> usize {
        2 * self.width + self.aux + j * self.op_words + k
    }

    /// The witness word of the application's local variable `k`
    /// (`Words.appVar`): the application words in order, then the slot
    /// words. Past them it is a word past the witness, which reads zero.
    pub fn app_var(&self, k: usize) -> usize {
        let slot_start = self.slot_local(0, 0);
        if k < slot_start {
            return self.memory_count + k;
        }
        let (j, i) = ((k - slot_start) / self.op_words, (k - slot_start) % self.op_words);
        if j < self.b_ops {
            self.op(j, i)
        } else {
            self.count() + k
        }
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
