//! The memory geometry of spec §4, its validity, and its derived widths.
//! Mirrors `Geometry` in `Spec/Nebula/Plan.lean`; `FirstPlan.lean` and
//! `SecondPlan.lean` own the package geometries. A geometry has no image, so
//! the relation, the plan digest, and the witness layout read no program.

use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks as F;

/// Spec §4.1 without the images.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub r: usize,
    pub mu: usize,
    pub w_ts: usize,
    pub b_ops: usize,
    pub b_scan: usize,
    pub n: usize,
    pub s_max: usize,
}

impl Geometry {
    /// The first memory-application package (`FirstPlan.lean`): ROM and RAM
    /// of four words, two ports, segments of two steps, at most four
    /// segments, and 5-bit timestamps.
    pub(super) fn first() -> Self {
        Self {
            r: 2,
            mu: 2,
            w_ts: 5,
            b_ops: 2,
            b_scan: 4,
            n: 2,
            s_max: 4,
        }
    }

    /// The second package (`SecondPlan.lean`), for the spec §14 cases that
    /// the first geometry cannot reach: ROM of four words and RAM of eight
    /// (`r < μ`), segments of one step, at most four segments, and 4-bit
    /// timestamps.
    pub(super) fn second() -> Self {
        Self {
            r: 2,
            mu: 3,
            w_ts: 4,
            b_ops: 2,
            b_scan: 12,
            n: 1,
            s_max: 4,
        }
    }

    /// Lean `Geometry.Valid`: spec §4.2 rules 1–5 and positive counts. The
    /// memory theorems hold only for a valid geometry. `μ ≥ 64` breaks
    /// `R + M < q`, so the shifts below cannot overflow.
    pub fn is_valid(&self) -> bool {
        if self.mu >= 64 || self.r > self.mu || self.w_ts > 62 {
            return false;
        }
        let q = u128::from(F::ORDER_U64);
        let cells = (1u128 << self.r) + (1u128 << self.mu);
        let product = |values: &[usize]| {
            values
                .iter()
                .try_fold(1u128, |product, value| product.checked_mul(*value as u128))
        };
        product(&[self.n, self.b_scan]) == Some(cells)
            && product(&[self.s_max, self.n, self.b_ops]).is_some_and(|count| count < 1u128 << self.w_ts)
            && product(&[self.b_ops, self.op_width()]).is_some_and(|bits| bits < q)
            && product(&[self.b_scan, self.scan_width()]).is_some_and(|bits| bits < q)
            && cells < q
            && [self.b_ops, self.b_scan, self.n, self.s_max]
                .iter()
                .all(|count| *count > 0)
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

    /// `m_mem`: the most tuples on each side of one segment's memory check.
    pub fn max_tuples(&self) -> usize {
        self.cells() + self.n * self.b_ops
    }

    /// `-log2` of the memory fingerprint error `2·m_mem/q²` (security note
    /// §5). Spec §4.2 rule 6 requires at least 109.91 bits (Lean
    /// `Geometry.secure_iff`).
    pub fn memory_security_bits(&self) -> f64 {
        2.0 * (F::ORDER_U64 as f64).log2() - (2.0 * self.max_tuples() as f64).log2()
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
}
