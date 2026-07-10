//! Nebula plan parameters and bit-level lane/`x` layouts — spec §2, §3, §4.4.
//!
//! Owns: [`NebulaParams`] (validated plan constants and derived sizes) and
//! the record ↔ bit encodings shared by the native prover, the future
//! `S_mem` circuit builder, and tests: the ops lane, the IS/FS scan lanes,
//! and the step public input. Every committed coordinate produced here is a
//! bit (`{0, 1}` in `F`), as the engine's norm bound requires.
//!
//! Does not own: memory semantics ([`super::trace`]), fingerprint math
//! ([`super::fingerprint`]), or CCS rows (circuit builder, spec §13 step 2).
//!
//! ## Encoding contract (normative, spec §3)
//!
//! - Multi-bit fields are little-endian: bit `k` of a field holds
//!   `(value >> k) & 1`.
//! - Fields pack in their table order (spec §3.2/§3.3); slots pack
//!   consecutively; lane tails pad with zero bits to a multiple of
//!   `neo_math::D` (**L-ALIGN**, spec §5.1 — lanes must occupy whole ring
//!   columns of the embedding or the fold action stops commuting with lane
//!   slicing).
//! - Op pad slots are all-zero except the `pad` bit (E7 canonicality).
//! - Scan lanes have no pads: exact cover (`N · B_scan = R + M`, spec §2)
//!   makes every scan slot a real cell.

use neo_math::field::KExtensions;
use neo_math::{D, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

/// Cell value width in bits (one memory cell = one 32-bit word). Spec §2.
pub const VAL_BITS: usize = 32;
/// Timestamp width in bits. Spec §2.
pub const TS_BITS: usize = 44;
/// Segment-counter width in the step public input. Spec §4.4.
pub const SEG_IDX_BITS: usize = 16;
/// Step-counter width in the step public input. Spec §4.4.
pub const STEP_IDX_BITS: usize = 16;
/// Bits per `K` coefficient (canonical Goldilocks limb).
pub const K_LIMB_BITS: usize = 64;
/// Bits per `K` element (two limbs: real, then imaginary).
pub const K_BITS: usize = 2 * K_LIMB_BITS;
/// Bits per scan-lane slot: value then timestamp. Spec §3.3.
pub const CELL_BITS: usize = VAL_BITS + TS_BITS;
/// Bits of the stack-less step public input (spec §4.4, `= 1,400`); a
/// plan's full width is [`NebulaParams::x_bits`].
pub use crate::paper::construction2::nebula_lane::{StackShape, MAX_STACKS, X_BASE_BITS};

/// Bit offsets of each field inside the encoded step public input.
///
/// Must match the push order of [`StepPublicInput::encode`]; the circuit
/// builder reads γ and `h` words straight from these offsets, so any
/// reorder must touch both places (the round-trip test pins them).
pub mod x_offsets {
    use super::{K_BITS, SEG_IDX_BITS, STEP_IDX_BITS, TS_BITS};

    pub const SEG_IDX: usize = 0;
    pub const IDX: usize = SEG_IDX + SEG_IDX_BITS;
    pub const TS_IN: usize = IDX + STEP_IDX_BITS;
    pub const TS_OUT: usize = TS_IN + TS_BITS;
    /// γ1 then γ2, each as two 64-bit limbs (real, imaginary).
    pub const GAMMA: usize = TS_OUT + TS_BITS;
    /// Four incoming products, order per [`super::H_RS`].
    pub const H_IN: usize = GAMMA + 2 * K_BITS;
    /// Four outgoing products, order per [`super::H_RS`].
    pub const H_OUT: usize = H_IN + 4 * K_BITS;
    /// Stack-pointer slots (v3.1, appended): per stack `s`, `sp_in` at
    /// `SP + s·2σ` and `sp_out` at `SP + s·2σ + σ`, each σ bits.
    pub const SP: usize = H_OUT + 4 * K_BITS;
}

/// Order of the four running products wherever `[K; 4]` appears in this
/// module: `h[0] = h_rs`, `h[1] = h_ws`, `h[2] = h_is`, `h[3] = h_fs`.
pub const H_RS: usize = 0;
/// See [`H_RS`].
pub const H_WS: usize = 1;
/// See [`H_RS`].
pub const H_IS: usize = 2;
/// See [`H_RS`].
pub const H_FS: usize = 3;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LayoutError {
    #[error("nebula params: {0}")]
    Params(&'static str),
    #[error("ops lane holds at most {max} ops, got {got}")]
    TooManyOps { max: usize, got: usize },
    #[error("scan lane needs exactly {want} cells, got {got}")]
    ScanLen { want: usize, got: usize },
    #[error("{field} value {value} exceeds its {bits}-bit width")]
    FieldRange {
        field: &'static str,
        value: u64,
        bits: usize,
    },
    #[error("address {addr} out of range for namespace of {cells} cells")]
    AddrRange { addr: u64, cells: u64 },
    #[error("stack {got} does not exist (plan has {stacks} stacks)")]
    StackIndex { got: u8, stacks: usize },
    #[error("op slot {0} sets more than one namespace selector")]
    SelectorNotOneHot(usize),
    #[error("lane bit at index {0} is not 0/1")]
    NonBit(usize),
    #[error("pad slot {0} violates canonicality (nonzero field bits)")]
    PadNotCanonical(usize),
}

/// Validated plan constants and derived sizes. Spec §2.
///
/// Immutable once constructed; every size the circuit, prover, and tests
/// agree on is derived from here (single source of truth). The constructor
/// enforces the spec's plan-validity rules, including exact cover and the
/// fingerprint-packing bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaParams {
    /// Public-ROM cells = `2^r`, addresses `[0, 2^r)`.
    pub r: u32,
    /// RAM cells = `2^mu`, addresses `[0, 2^mu)`.
    pub mu: u32,
    /// Memory-op slots per step.
    pub b_ops: usize,
    /// Scan slots per step.
    pub b_scan: usize,
    /// Maximum segments per chain (bounds the global timestamp).
    pub seg_max: u64,
    /// `S`: segment-local stacks (spec §2, v3.1; `≤ MAX_STACKS`).
    pub num_stacks: usize,
    /// `σ`: stack-pointer width in bits (0 iff `num_stacks == 0`);
    /// capacity per stack is `2^σ − 1` cells.
    pub sigma: u32,
}

impl NebulaParams {
    /// Validate and construct a stack-less plan (v3 shape; add stacks
    /// with [`Self::with_stacks`]). Rules (spec §2):
    ///
    /// 1. exact cover: `B_scan` divides `R + M` (steps per segment
    ///    `N = (R + M) / B_scan`);
    /// 2. packing: `TS_BITS + bits(address space) ≤ 62` so
    ///    `packed(t, g) = t + 2^TS_BITS · g` cannot overflow Goldilocks;
    /// 3. timestamps: `seg_max · N · B_ops < 2^TS_BITS`;
    /// 4. `r ≤ μ`: the ops-lane `addr` is `max(r, μ)` bits and only ROM
    ///    addresses are range-gated (E6), so RAM bitness bounds
    ///    `addr < M` only when `μ = max(r, μ)`.
    pub fn new(r: u32, mu: u32, b_ops: usize, b_scan: usize, seg_max: u64) -> Result<Self, LayoutError> {
        Self::validated(Self {
            r,
            mu,
            b_ops,
            b_scan,
            seg_max,
            num_stacks: 0,
            sigma: 0,
        })
    }

    /// Add segment-local stacks (spec §2, v3.1): `num_stacks ≤ MAX_STACKS`
    /// namespaces of `2^σ − 1` cells each, `1 ≤ σ ≤ μ` — σ at most μ keeps
    /// the stack address inside the `addr` field's bitness.
    pub fn with_stacks(self, num_stacks: usize, sigma: u32) -> Result<Self, LayoutError> {
        if num_stacks == 0 || num_stacks > MAX_STACKS {
            return Err(LayoutError::Params("num_stacks must be in 1..=MAX_STACKS"));
        }
        if sigma == 0 || sigma > self.mu {
            return Err(LayoutError::Params("sigma must satisfy 1 ≤ σ ≤ μ (spec §2)"));
        }
        Self::validated(Self {
            num_stacks,
            sigma,
            ..self
        })
    }

    fn validated(p: Self) -> Result<Self, LayoutError> {
        if p.b_ops == 0 || p.b_scan == 0 {
            return Err(LayoutError::Params("b_ops and b_scan must be nonzero"));
        }
        if p.r >= 32 || p.mu >= 32 {
            return Err(LayoutError::Params(
                "r and mu must be < 32 (u32 cell values, u64 indices)",
            ));
        }
        if p.r > p.mu {
            return Err(LayoutError::Params(
                "r must be ≤ mu: RAM addresses are bounded by bitness alone (spec §2)",
            ));
        }
        if p.scanned_cells() % (p.b_scan as u64) != 0 {
            return Err(LayoutError::Params("exact cover: b_scan must divide R + M"));
        }
        if TS_BITS + p.address_space_bits() > 62 {
            return Err(LayoutError::Params(
                "packing bound: TS_BITS + bits(R + M + S·2^σ) must be ≤ 62",
            ));
        }
        let ts_capacity = (p.seg_max as u128) * (p.steps_per_segment() as u128) * (p.b_ops as u128);
        if ts_capacity >= 1u128 << TS_BITS {
            return Err(LayoutError::Params("seg_max · N · b_ops must stay below 2^TS_BITS"));
        }
        Ok(p)
    }

    /// Spec §2 test profile (`r = 4, μ = 8, B_ops = B_scan = 8` → `N = 34`).
    pub fn test_profile() -> Self {
        Self::new(4, 8, 8, 8, 1 << 10).expect("spec test profile is valid")
    }

    /// Spec §2 v3 targets (`r = 12, μ = 16, B = 64` → `N = 1,088`).
    pub fn v3_targets() -> Self {
        Self::new(12, 16, 64, 64, 1 << 16).expect("spec v3 targets are valid")
    }

    /// `R`: cells in the public-ROM namespace.
    pub fn rom_cells(&self) -> u64 {
        1 << self.r
    }

    /// `M`: cells in the RAM namespace.
    pub fn ram_cells(&self) -> u64 {
        1 << self.mu
    }

    /// `R + M`: the scanned cells — ROM then RAM, the scan/global-index
    /// prefix. Stacks live above and are never scanned (spec §3.1).
    pub fn scanned_cells(&self) -> u64 {
        self.rom_cells() + self.ram_cells()
    }

    /// Cells per stack namespace (`2^σ`; usable capacity `2^σ − 1`).
    pub fn stack_cells(&self) -> u64 {
        1 << self.sigma
    }

    /// The plan's stack geometry, as the verifier side carries it.
    pub fn stack_shape(&self) -> StackShape {
        StackShape {
            count: self.num_stacks,
            sigma: self.sigma as usize,
        }
    }

    /// Bits of the step public input under this plan (spec §4.4).
    pub fn x_bits(&self) -> usize {
        self.stack_shape().x_bits()
    }

    /// `N = (R + M) / B_scan`: steps per segment (exact cover, spec §2).
    pub fn steps_per_segment(&self) -> usize {
        (self.scanned_cells() / self.b_scan as u64) as usize
    }

    /// Op capacity of one segment (`N · B_ops`).
    pub fn ops_per_segment(&self) -> usize {
        self.steps_per_segment() * self.b_ops
    }

    /// Address field width: `max(r, μ)` bits (spec §3.2).
    pub fn addr_bits(&self) -> usize {
        self.r.max(self.mu) as usize
    }

    /// Bits needed for a global cell index over the full address space
    /// `R + M + S·2^σ` (the §2 packing bound's operand).
    pub fn address_space_bits(&self) -> usize {
        let top = self.scanned_cells() + self.num_stacks as u64 * self.stack_cells() - 1;
        (64 - top.leading_zeros()) as usize
    }

    /// `OP_BITS`: one ops-lane slot — `pad, is_write, ram, stk_0..,
    /// addr, v_r, v_w, rt` in that order (spec §3.2).
    pub fn op_bits(&self) -> usize {
        3 + self.num_stacks + self.addr_bits() + 2 * VAL_BITS + TS_BITS
    }

    /// Ops-lane width in bits, L-ALIGN padded to whole ring columns.
    pub fn ops_lane_bits(&self) -> usize {
        align_to_ring_columns(self.b_ops * self.op_bits())
    }

    /// One scan-lane width in bits (IS and FS each), L-ALIGN padded.
    pub fn scan_lane_bits(&self) -> usize {
        align_to_ring_columns(self.b_scan * CELL_BITS)
    }

    /// Global cell index (spec §3.1/§4.3): ROM at `[0, R)`, RAM at
    /// `[R, R + M)`, stack `s` at `[R + M + s·2^σ, ·)`. Injective onto
    /// `(namespace, addr)` because every namespace's addresses are
    /// range-bound below its span.
    pub fn global_index(&self, space: MemSpace, addr: u64) -> Result<u64, LayoutError> {
        let (cells, base) = match space {
            MemSpace::Rom => (self.rom_cells(), 0),
            MemSpace::Ram => (self.ram_cells(), self.rom_cells()),
            MemSpace::Stack(s) => {
                if s as usize >= self.num_stacks {
                    return Err(LayoutError::StackIndex {
                        got: s,
                        stacks: self.num_stacks,
                    });
                }
                (self.stack_cells(), self.scanned_cells() + s as u64 * self.stack_cells())
            }
        };
        if addr >= cells {
            return Err(LayoutError::AddrRange { addr, cells });
        }
        Ok(base + addr)
    }
}

/// Round a bit width up to a multiple of `d = 54` (L-ALIGN, spec §5.1).
pub fn align_to_ring_columns(bits: usize) -> usize {
    bits.div_ceil(D) * D
}

/// A memory namespace (spec §3.1): the two random-access spaces, plus
/// the segment-local stacks (v3.1). Encoded in the ops lane as one-hot
/// selector bits (`ram`, then `stk_s`; ROM = none set).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemSpace {
    /// Public ROM, addresses `[0, R)`; read-only.
    Rom,
    /// RAM, addresses `[0, M)`.
    Ram,
    /// Stack `s < S`; access only through push/pop at the stack pointer.
    Stack(u8),
}

/// One real memory operation, as the ops lane stores it (spec §3.2). Pad
/// slots are not represented — encoders append them, decoders drop them.
///
/// `rt` is the prover-supplied timestamp of the previous access to this
/// cell (for pops: the push time; for pushes: 0); the write timestamp is
/// *not* stored (it is `ts_in + cnt_j` by construction, spec §3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemOpRecord {
    /// `false` = read/pop, `true` = write/push.
    pub is_write: bool,
    /// The namespace this op touches.
    pub space: MemSpace,
    /// Address within the namespace (stacks: the E13-bound `sp` slot).
    pub addr: u64,
    /// Value read (writes: the old value; pushes: 0 per E14).
    pub v_r: u32,
    /// Value written back (reads/pops: equals `v_r`).
    pub v_w: u32,
    /// Timestamp of the previous access to this cell.
    pub rt: u64,
}

/// One memory cell as the scan lanes store it (spec §3.3). The address is
/// not stored: it is the slot's scan position (structural).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellRecord {
    /// Cell value.
    pub v: u32,
    /// Timestamp of the last write (0 if never written).
    pub t: u64,
}

/// Step public input — the carried slots of `x` (spec §4.4). The
/// canonical struct lives with its F′ consumer
/// (`paper::construction2::nebula_lane`); this module owns its bit
/// encoding: field order `seg_idx`, `idx`, `ts_in`, `ts_out`,
/// `gamma[0..2]`, `h_in[0..4]`, `h_out[0..4]`, then per stack
/// `sp_in`/`sp_out` (v3.1); product order per [`H_RS`].
pub use crate::paper::construction2::nebula_lane::NebulaStepX as StepPublicInput;

impl NebulaParams {
    /// Encode up to `B_ops` real ops into one ops lane; remaining slots are
    /// canonical pads (all-zero fields, `pad = 1`). Output length is
    /// [`Self::ops_lane_bits`], every element a bit.
    pub fn encode_ops_lane(&self, ops: &[MemOpRecord]) -> Result<Vec<F>, LayoutError> {
        if ops.len() > self.b_ops {
            return Err(LayoutError::TooManyOps {
                max: self.b_ops,
                got: ops.len(),
            });
        }
        let mut bits = BitSink::with_capacity(self.ops_lane_bits());
        for op in ops {
            // Range checks mirror the circuit's bitness exactly.
            self.global_index(op.space, op.addr)?; // validates addr for its namespace
            check_width("rt", op.rt, TS_BITS)?;
            bits.push_bit(false); // pad = 0
            bits.push_bit(op.is_write);
            bits.push_bit(op.space == MemSpace::Ram);
            for s in 0..self.num_stacks {
                bits.push_bit(op.space == MemSpace::Stack(s as u8));
            }
            bits.push_u64(op.addr, self.addr_bits());
            bits.push_u64(op.v_r as u64, VAL_BITS);
            bits.push_u64(op.v_w as u64, VAL_BITS);
            bits.push_u64(op.rt, TS_BITS);
        }
        for _ in ops.len()..self.b_ops {
            bits.push_bit(true); // pad = 1
            bits.push_zeros(self.op_bits() - 1);
        }
        Ok(bits.finish_aligned())
    }

    /// Decode an ops lane back to its real ops, validating bitness, pad
    /// canonicality (E7), and that pads only follow real ops.
    pub fn decode_ops_lane(&self, lane: &[F]) -> Result<Vec<MemOpRecord>, LayoutError> {
        if lane.len() != self.ops_lane_bits() {
            return Err(LayoutError::ScanLen {
                want: self.ops_lane_bits(),
                got: lane.len(),
            });
        }
        let mut src = BitSource::new(lane);
        let mut ops = Vec::new();
        for slot in 0..self.b_ops {
            let pad = src.read_bit()?;
            let is_write = src.read_bit()?;
            let ram = src.read_bit()?;
            let mut stack = None;
            for s in 0..self.num_stacks {
                if src.read_bit()? {
                    if ram || stack.is_some() {
                        return Err(LayoutError::SelectorNotOneHot(slot));
                    }
                    stack = Some(s as u8);
                }
            }
            let addr = src.read_u64(self.addr_bits())?;
            let v_r = src.read_u64(VAL_BITS)? as u32;
            let v_w = src.read_u64(VAL_BITS)? as u32;
            let rt = src.read_u64(TS_BITS)?;
            let space = match (ram, stack) {
                (true, _) => MemSpace::Ram,
                (false, Some(s)) => MemSpace::Stack(s),
                (false, None) => MemSpace::Rom,
            };
            if pad {
                if is_write || space != MemSpace::Rom || addr != 0 || v_r != 0 || v_w != 0 || rt != 0 {
                    return Err(LayoutError::PadNotCanonical(slot));
                }
            } else {
                if slot != ops.len() {
                    // A real op after a pad slot: sequential fill violated.
                    return Err(LayoutError::PadNotCanonical(slot));
                }
                ops.push(MemOpRecord {
                    is_write,
                    space,
                    addr,
                    v_r,
                    v_w,
                    rt,
                });
            }
        }
        src.expect_zero_padding()?;
        Ok(ops)
    }

    /// Encode exactly `B_scan` cells into one scan lane (IS or FS — same
    /// layout, spec §3.3). Output length is [`Self::scan_lane_bits`].
    pub fn encode_scan_lane(&self, cells: &[CellRecord]) -> Result<Vec<F>, LayoutError> {
        if cells.len() != self.b_scan {
            return Err(LayoutError::ScanLen {
                want: self.b_scan,
                got: cells.len(),
            });
        }
        let mut bits = BitSink::with_capacity(self.scan_lane_bits());
        for cell in cells {
            check_width("t", cell.t, TS_BITS)?;
            bits.push_u64(cell.v as u64, VAL_BITS);
            bits.push_u64(cell.t, TS_BITS);
        }
        Ok(bits.finish_aligned())
    }

    /// Decode one scan lane back to its `B_scan` cells.
    pub fn decode_scan_lane(&self, lane: &[F]) -> Result<Vec<CellRecord>, LayoutError> {
        if lane.len() != self.scan_lane_bits() {
            return Err(LayoutError::ScanLen {
                want: self.scan_lane_bits(),
                got: lane.len(),
            });
        }
        let mut src = BitSource::new(lane);
        let mut cells = Vec::with_capacity(self.b_scan);
        for _ in 0..self.b_scan {
            let v = src.read_u64(VAL_BITS)? as u32;
            let t = src.read_u64(TS_BITS)?;
            cells.push(CellRecord { v, t });
        }
        src.expect_zero_padding()?;
        Ok(cells)
    }
}

impl StepPublicInput {
    /// Encode to the `stacks.x_bits()` public-input bits (spec §4.4
    /// order; the trailing `sp` slots are the plan's, v3.1).
    pub fn encode(&self, stacks: StackShape) -> Result<Vec<F>, LayoutError> {
        check_width("seg_idx", self.seg_idx, SEG_IDX_BITS)?;
        check_width("idx", self.idx, STEP_IDX_BITS)?;
        check_width("ts_in", self.ts_in, TS_BITS)?;
        check_width("ts_out", self.ts_out, TS_BITS)?;
        let mut bits = BitSink::with_capacity(stacks.x_bits());
        bits.push_u64(self.seg_idx, SEG_IDX_BITS);
        bits.push_u64(self.idx, STEP_IDX_BITS);
        bits.push_u64(self.ts_in, TS_BITS);
        bits.push_u64(self.ts_out, TS_BITS);
        for k in self
            .gamma
            .iter()
            .chain(self.h_in.iter())
            .chain(self.h_out.iter())
        {
            let (c0, c1) = k.to_limbs_u64();
            bits.push_u64(c0, K_LIMB_BITS);
            bits.push_u64(c1, K_LIMB_BITS);
        }
        for s in 0..stacks.count {
            check_width("sp_in", self.sp_in[s], stacks.sigma)?;
            check_width("sp_out", self.sp_out[s], stacks.sigma)?;
            bits.push_u64(self.sp_in[s], stacks.sigma);
            bits.push_u64(self.sp_out[s], stacks.sigma);
        }
        Ok(bits.finish_exact())
    }

    /// Decode from `stacks.x_bits()` bits, validating bitness and counter
    /// widths.
    pub fn decode(bits: &[F], stacks: StackShape) -> Result<Self, LayoutError> {
        if bits.len() != stacks.x_bits() {
            return Err(LayoutError::ScanLen {
                want: stacks.x_bits(),
                got: bits.len(),
            });
        }
        let mut src = BitSource::new(bits);
        let seg_idx = src.read_u64(SEG_IDX_BITS)?;
        let idx = src.read_u64(STEP_IDX_BITS)?;
        let ts_in = src.read_u64(TS_BITS)?;
        let ts_out = src.read_u64(TS_BITS)?;
        let read_k = |src: &mut BitSource<'_>| -> Result<K, LayoutError> {
            let c0 = src.read_u64(K_LIMB_BITS)?;
            let c1 = src.read_u64(K_LIMB_BITS)?;
            Ok(K::from_coeffs([F::from_u64(c0), F::from_u64(c1)]))
        };
        let gamma = [read_k(&mut src)?, read_k(&mut src)?];
        let h = |src: &mut BitSource<'_>| -> Result<[K; 4], LayoutError> {
            Ok([read_k(src)?, read_k(src)?, read_k(src)?, read_k(src)?])
        };
        let h_in = h(&mut src)?;
        let h_out = h(&mut src)?;
        let mut sp_in = [0u64; MAX_STACKS];
        let mut sp_out = [0u64; MAX_STACKS];
        for s in 0..stacks.count {
            sp_in[s] = src.read_u64(stacks.sigma)?;
            sp_out[s] = src.read_u64(stacks.sigma)?;
        }
        Ok(Self {
            seg_idx,
            idx,
            ts_in,
            ts_out,
            gamma,
            h_in,
            h_out,
            sp_in,
            sp_out,
        })
    }
}

/// Encode the public suffix consumed by the following F' step:
/// `[step_x_bits || open || bits(D_pre)]`.
pub fn encode_delayed_f_prime_suffix(
    step: &StepPublicInput,
    stacks: StackShape,
    d_pre: Option<[[F; 4]; 3]>,
) -> Result<Vec<F>, LayoutError> {
    let mut out = step.encode(stacks)?;
    out.push(if d_pre.is_some() { F::ONE } else { F::ZERO });
    for digest in d_pre.unwrap_or([[F::ZERO; 4]; 3]) {
        for lane in digest {
            let value = lane.as_canonical_u64();
            for bit in 0..K_LIMB_BITS {
                out.push(F::from_u64((value >> bit) & 1));
            }
        }
    }
    debug_assert_eq!(
        out.len(),
        crate::paper::f_prime::nebula_lane_circuit::delayed_nebula_public_suffix_len(stacks)
    );
    Ok(out)
}

fn check_width(field: &'static str, value: u64, bits: usize) -> Result<(), LayoutError> {
    if bits < 64 && value >> bits != 0 {
        return Err(LayoutError::FieldRange { field, value, bits });
    }
    Ok(())
}

/// Little-endian bit writer producing `{0, 1}` field elements.
struct BitSink {
    bits: Vec<F>,
}

impl BitSink {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            bits: Vec::with_capacity(capacity),
        }
    }

    fn push_bit(&mut self, b: bool) {
        self.bits.push(if b { F::ONE } else { F::ZERO });
    }

    fn push_u64(&mut self, value: u64, nbits: usize) {
        for k in 0..nbits {
            self.push_bit((value >> k) & 1 == 1);
        }
    }

    fn push_zeros(&mut self, n: usize) {
        let len = self.bits.len();
        self.bits.resize(len + n, F::ZERO);
    }

    /// Finish, padding with zeros to whole ring columns (L-ALIGN).
    fn finish_aligned(mut self) -> Vec<F> {
        let target = align_to_ring_columns(self.bits.len());
        self.push_zeros(target - self.bits.len());
        self.bits
    }

    /// Finish without padding (the `x` region is not a committed lane).
    fn finish_exact(self) -> Vec<F> {
        self.bits
    }
}

/// Little-endian bit reader over `{0, 1}` field elements.
struct BitSource<'a> {
    bits: &'a [F],
    pos: usize,
}

impl<'a> BitSource<'a> {
    fn new(bits: &'a [F]) -> Self {
        Self { bits, pos: 0 }
    }

    fn read_bit(&mut self) -> Result<bool, LayoutError> {
        let f = self.bits[self.pos];
        let b = if f == F::ZERO {
            false
        } else if f == F::ONE {
            true
        } else {
            return Err(LayoutError::NonBit(self.pos));
        };
        self.pos += 1;
        Ok(b)
    }

    fn read_u64(&mut self, nbits: usize) -> Result<u64, LayoutError> {
        let mut value = 0u64;
        for k in 0..nbits {
            if self.read_bit()? {
                value |= 1 << k;
            }
        }
        Ok(value)
    }

    /// Require every remaining bit (the L-ALIGN tail) to be zero.
    fn expect_zero_padding(&mut self) -> Result<(), LayoutError> {
        while self.pos < self.bits.len() {
            if self.read_bit()? {
                return Err(LayoutError::PadNotCanonical(self.pos - 1));
            }
        }
        Ok(())
    }
}
