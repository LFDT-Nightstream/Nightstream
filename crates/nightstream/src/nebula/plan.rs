//! The verifier-owned memory plan of spec §4 and its derived widths.
//! Mirrors `Spec/Nebula/Plan.lean`; `FirstPlan.lean` and `SecondPlan.lean` own the
//! package plans.

/// Spec §4.1 `NebulaPlan`. Images hold 32-bit words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub r: usize,
    pub mu: usize,
    pub w_ts: usize,
    pub b_ops: usize,
    pub b_scan: usize,
    pub n: usize,
    pub s_max: usize,
    pub rom: Vec<u64>,
    pub ram: Vec<u64>,
}

impl Plan {
    /// The first memory-application package plan (`FirstPlan.lean`): ROM and
    /// RAM of four words, two ports, segments of two steps, at most four
    /// segments, 5-bit timestamps, and the ROM program
    /// `loadi 5; store 0; load 0; halt`.
    pub fn first() -> Self {
        Self {
            r: 2,
            mu: 2,
            w_ts: 5,
            b_ops: 2,
            b_scan: 4,
            n: 2,
            s_max: 4,
            rom: vec![23, 2, 1, 0],
            ram: vec![0; 4],
        }
    }

    /// The second package plan (`SecondPlan.lean`), for the spec §14 cases
    /// that the first plan cannot reach: ROM of four words and RAM of eight
    /// (`r < μ`), segments of one step, at most four segments, 4-bit
    /// timestamps, and the ROM program `load 0; store 1; load 1; store 0`.
    pub fn second() -> Self {
        let mut ram = vec![0; 8];
        ram[0] = 7;
        Self {
            r: 2,
            mu: 3,
            w_ts: 4,
            b_ops: 2,
            b_scan: 12,
            n: 1,
            s_max: 4,
            rom: vec![1, 6, 5, 2],
            ram,
        }
    }

    pub fn rom_size(&self) -> usize {
        1 << self.r
    }

    pub fn ram_size(&self) -> usize {
        1 << self.mu
    }

    pub fn cells(&self) -> usize {
        self.rom_size() + self.ram_size()
    }

    /// Bits of one operation slot (spec §6.1).
    pub fn op_width(&self) -> usize {
        3 + self.mu + 64 + self.w_ts
    }

    /// Bits of one scan slot (spec §6.2).
    pub fn scan_width(&self) -> usize {
        32 + self.w_ts
    }

    /// `W_seg`: enough bits for every value below `S_max` (Lean `Nat.log2 + 1`).
    pub fn seg_width(&self) -> usize {
        (usize::BITS - self.s_max.leading_zeros()) as usize
    }

    /// The image word of global cell `g` (spec §5).
    pub fn image(&self, g: usize) -> u64 {
        if g < self.rom_size() {
            self.rom[g]
        } else {
            self.ram[g - self.rom_size()]
        }
    }
}
