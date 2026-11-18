//! SPEC.md section 7.1; `Geometry` in `Spec/Nebula/Plan.lean`.

use neo_math::F;
use p3_field::PrimeField64;

const MAX_SECURE_TUPLES: usize = 139_509;

/// SPEC.md section 7.1 without the images.
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
    /// `FirstPlan.lean`.
    #[cfg(test)]
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

    /// `SecondPlan.lean`, for SPEC.md section 9.
    #[cfg(test)]
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

    /// `ThirdPlan.lean`.
    #[cfg(test)]
    pub(super) fn third() -> Self {
        Self {
            r: 3,
            mu: 2,
            w_ts: 5,
            b_ops: 2,
            b_scan: 6,
            n: 2,
            s_max: 4,
        }
    }

    /// The geometry of a ROM of `2^r` cells and a RAM of `2^mu` cells, with
    /// `b_ops` operation slots in each step, segments of `n` steps, and at
    /// most `s_max` segments. `w_ts` is the fewest bits that hold
    /// `s_max · n · b_ops`, and `b_scan` is the cells over `n`. `None` when
    /// the result breaks SPEC.md section 7.1, rules 1 to 6 (`is_valid` and
    /// Lean `Geometry.Secure`).
    pub fn new(r: usize, mu: usize, b_ops: usize, n: usize, s_max: usize) -> Option<Self> {
        let space = |log2: usize| 1usize.checked_shl(u32::try_from(log2).ok()?);
        let cells = space(r)?.checked_add(space(mu)?)?;
        let stamps = s_max.checked_mul(n)?.checked_mul(b_ops)?;
        let geometry = Self {
            r,
            mu,
            w_ts: (usize::BITS - stamps.leading_zeros()) as usize,
            b_ops,
            b_scan: cells.checked_div(n)?,
            n,
            s_max,
        };
        (geometry.is_valid() && geometry.is_secure()).then_some(geometry)
    }

    /// Lean `Geometry.Valid`: SPEC.md section 7.1, rules 1 to 5 and positive counts. The
    /// memory theorems hold only for a valid geometry. `r ≥ 64` or `μ ≥ 64` breaks
    /// `R + M < q`, so the shifts below cannot overflow.
    pub fn is_valid(&self) -> bool {
        if self.r >= 64 || self.mu >= 64 || self.w_ts > 31 {
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

    /// Lean `Geometry.Secure`, `Geometry.secure_iff`; SPEC.md section 7.1, rule 6.
    pub(super) fn is_secure(&self) -> bool {
        self.max_tuples() <= MAX_SECURE_TUPLES
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

    /// `-log2` of the memory fingerprint error `2·m_mem/q²` (`SECURITY_MODEL.md`,
    /// adversary 5). SPEC.md section 7.1, rule 6 requires at least 109.91 bits (Lean
    /// `Geometry.secure_iff`).
    pub fn memory_security_bits(&self) -> f64 {
        2.0 * (F::ORDER_U64 as f64).log2() - (2.0 * self.max_tuples() as f64).log2()
    }

    /// The largest `k` with `S_max · 2 · m_mem / q² ≤ 2^−k` (`SECURITY_MODEL.md`,
    /// adversary 5). `Circuit::prover` and `Circuit::verifier` accept a
    /// minimum of at most this value.
    pub fn segments_security_bits(&self) -> u32 {
        let q = u128::from(F::ORDER_U64);
        (2 * self.max_tuples() as u128)
            .checked_mul(self.s_max as u128)
            .and_then(|error| (q * q).checked_div(error))
            .and_then(u128::checked_ilog2)
            .unwrap_or(0)
    }

    /// Lean `Geometry.addrWidth`.
    pub(crate) fn addr_width(&self) -> usize {
        self.r.max(self.mu)
    }

    /// SPEC.md section 7.2.
    pub(crate) fn op_width(&self) -> usize {
        3 + self.addr_width() + 64 + self.w_ts
    }

    /// SPEC.md section 7.2.
    pub(crate) fn scan_width(&self) -> usize {
        32 + self.w_ts
    }

    pub(crate) fn seg_width(&self) -> usize {
        (usize::BITS - self.s_max.leading_zeros()) as usize
    }
}
