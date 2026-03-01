use neo_ccs::matrix::Mat;
use neo_ccs::relations::{CcsClaim, CcsWitness};
use neo_math::K;
use neo_reductions::error::PiCcsError;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::ops::Range;

use crate::mem_init::MemInit;
use crate::riscv::lookups::RiscvOpcode;

fn default_one_usize() -> usize {
    1
}

/// Time-domain column payload for shard-local proving.
///
/// `cpu_cols` and `mem_cols` are stored in column-major form over time:
/// `cols[col_id][j]` where `j in [0, t)`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TimeColumns<F> {
    pub t: usize,
    pub cpu_cols: Vec<Vec<F>>,
    pub mem_cols: Vec<Vec<F>>,
    /// Chunk-padding activity mask over time (`1` for real step rows, `0` for padded rows).
    /// This is not the semantic CPU trace `active` column.
    pub active_col: Vec<F>,
    /// Logical column ids in the same order as `cpu_cols || mem_cols`.
    pub col_ids: Vec<usize>,
}

impl<F> Default for TimeColumns<F> {
    fn default() -> Self {
        Self {
            t: 0,
            cpu_cols: Vec::new(),
            mem_cols: Vec::new(),
            active_col: Vec::new(),
            col_ids: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LutTableSpec {
    /// A "virtual" lookup table defined by the RISC-V opcode semantics over
    /// an interleaved operand address space.
    ///
    /// Addressing convention:
    /// - `n_side = 2`, `ell = 1`
    /// - `d = 2*xlen` (one bit per dimension)
    /// - Address bits are little-endian and correspond to `interleave_bits(rs1, rs2)`.
    RiscvOpcode { opcode: RiscvOpcode, xlen: usize },

    /// A packed-key (non-bit-addressed) variant of `RiscvOpcode`, intended for "no width bloat"
    /// Shout/ALU proofs that do **not** commit to `ell_addr=64` addr-bit columns.
    ///
    /// Current implementation status:
    /// - Supported: `xlen = 32` for selected RV32 ops, including:
    ///   - bitwise: `And | Andn | Or | Xor`
    ///   - arithmetic: `Add | Sub`
    ///   - compares: `Eq | Neq | Slt | Sltu`
    ///   - shifts: `Sll | Srl | Sra`
    ///   - RV32M: `Mul | Mulh | Mulhu | Mulhsu | Div | Divu | Rem | Remu`
    /// - Witness convention: the Shout lane's `addr_bits` slice is repurposed as packed columns.
    ///   The exact layout depends on `opcode`; the suffix columns are always `[has_lookup, val_u32]`.
    ///   Examples:
    ///   - `Add/Sub` (d=3): `[lhs_u32, rhs_u32, aux_bit]` (carry for `Add`, borrow for `Sub`)
    ///   - `Eq/Neq` (d=35): `[lhs_u32, rhs_u32, borrow_bit, diff_bits[0..32]]` where `val_u32` is the out bit
    ///   - `Mul` (d=34): `[lhs_u32, rhs_u32, hi_bits[0..32]]` where `val_u32` is the low 32 bits
    ///   - `Mulhu` (d=34): `[lhs_u32, rhs_u32, lo_bits[0..32]]` where `val_u32` is the high 32 bits
    ///   - `Sltu` (d=35): `[lhs_u32, rhs_u32, diff_u32, diff_bits[0..32]]` where `val_u32` is the out bit
    ///   - `Sll/Srl/Sra` (d=38): `[lhs_u32, shamt_bits[0..5], ...]`
    ///
    /// For packed-key instances, Route-A enforces correctness directly via time-domain constraints
    /// (claimed sum forced to 0); table MLE evaluation is not used.
    RiscvOpcodePacked { opcode: RiscvOpcode, xlen: usize },

    /// An "event table" packed-key variant of `RiscvOpcodePacked` for RV32.
    ///
    /// Instead of storing one Shout lane over time (one row per cycle), the witness stores only
    /// the executed lookup events (one row per lookup). Each event row carries:
    /// - a prefix of `time_bits` boolean columns encoding the original Route-A time index `t`
    ///   (little-endian), and
    /// - the same packed columns as `RiscvOpcodePacked` for `opcode`.
    ///
    /// The Route-A protocol then links the event table back to the CPU trace via a "scatter"
    /// check at a random time point `r_cycle` (Jolt-ish): roughly,
    ///   Σ_events hash(event)·χ_{r_cycle}(t_event) == Σ_t hash(trace[t])·χ_{r_cycle}(t).
    ///
    /// Notes:
    /// - This is RV32-only (`xlen = 32`), `n_side = 2`, `ell = 1`.
    /// - `time_bits` must match the Route-A `ell_n` used for the time domain.
    RiscvOpcodeEventTablePacked {
        opcode: RiscvOpcode,
        xlen: usize,
        time_bits: usize,
    },

    /// Implicit identity table over 32-bit addresses: `table[addr] = addr`.
    ///
    /// Addressing convention:
    /// - `n_side = 2`, `ell = 1`
    /// - `d = 32` (one bit per dimension)
    /// - Address bits are little-endian
    IdentityU32,
}

impl LutTableSpec {
    pub fn eval_table_mle(&self, r_addr: &[K]) -> Result<K, PiCcsError> {
        match self {
            LutTableSpec::RiscvOpcode { opcode, xlen } => {
                Ok(crate::riscv::lookups::evaluate_opcode_mle(*opcode, r_addr, *xlen))
            }
            LutTableSpec::RiscvOpcodePacked { .. } => Err(PiCcsError::InvalidInput(
                "RiscvOpcodePacked does not support eval_table_mle (not bit-addressed)".into(),
            )),
            LutTableSpec::RiscvOpcodeEventTablePacked { .. } => Err(PiCcsError::InvalidInput(
                "RiscvOpcodeEventTablePacked does not support eval_table_mle (not bit-addressed)".into(),
            )),
            LutTableSpec::IdentityU32 => {
                if r_addr.len() != 32 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "IdentityU32: expected r_addr.len()=32, got {}",
                        r_addr.len()
                    )));
                }
                Ok(crate::identity::eval_identity_mle_le(r_addr))
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemInstance<C, F> {
    /// Logical memory instance identifier (e.g. RISC-V `PROG_ID/REG_ID/RAM_ID`).
    ///
    /// This is used by higher-level protocols to link Twist instances to CPU trace columns
    /// without relying on a fixed instance ordering.
    pub mem_id: u32,
    pub comms: Vec<C>,
    pub k: usize,
    pub d: usize,
    pub n_side: usize,
    pub steps: usize,
    /// Number of access lanes per VM step for this Twist instance.
    ///
    /// Each lane carries the canonical Twist bus slice:
    /// `[ra_bits, wa_bits, has_read, has_write, wv, rv, inc]`.
    #[serde(default = "default_one_usize")]
    pub lanes: usize,
    /// Bits per address dimension: ell = ceil(log2(n_side))
    /// With index-bit addressing, we commit d*ell bit-columns instead of d*n_side one-hot columns.
    pub ell: usize,
    /// Public initial memory state for cells [0..k).
    pub init: MemInit<F>,
    /// Optional precomputed digest for transcript binding fast paths (prover-only optimization).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_digest: Option<[u8; 32]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemWitness<F> {
    pub mats: Vec<Mat<F>>,
}

#[derive(Clone, Debug)]
pub struct TwistWitnessLaneLayout {
    pub ell_addr: usize,
    pub ra_bits: Range<usize>,
    pub wa_bits: Range<usize>,
    pub has_read: usize,
    pub has_write: usize,
    pub wv: usize,
    pub rv: usize,
    pub inc_at_write_addr: usize,
}

#[derive(Clone, Debug)]
pub struct TwistWitnessLayout {
    pub lane_len: usize,
    pub lanes: Vec<TwistWitnessLaneLayout>,
}

impl TwistWitnessLayout {
    pub fn expected_len(&self) -> usize {
        self.lane_len
            .checked_mul(self.lanes.len())
            .expect("TwistWitnessLayout: lane_len*lanes overflow")
    }
}

impl<C, F> MemInstance<C, F> {
    pub fn twist_layout(&self) -> TwistWitnessLayout {
        let ell_addr = self.d * self.ell;
        let lane_len = 2 * ell_addr + 5;
        let lanes = self.lanes.max(1);

        let mut out = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let base = lane * lane_len;
            out.push(TwistWitnessLaneLayout {
                ell_addr,
                ra_bits: base..(base + ell_addr),
                wa_bits: (base + ell_addr)..(base + 2 * ell_addr),
                has_read: base + 2 * ell_addr,
                has_write: base + 2 * ell_addr + 1,
                wv: base + 2 * ell_addr + 2,
                rv: base + 2 * ell_addr + 3,
                inc_at_write_addr: base + 2 * ell_addr + 4,
            });
        }
        TwistWitnessLayout { lane_len, lanes: out }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LutInstance<C, F> {
    /// Logical shout table identifier.
    pub table_id: u32,
    pub comms: Vec<C>,
    pub k: usize,
    pub d: usize,
    pub n_side: usize,
    pub steps: usize,
    /// Number of lookup lanes per VM step for this Shout instance.
    ///
    /// Each lane carries the canonical Shout bus slice:
    /// `[addr_bits, has_lookup, val]`.
    #[serde(default = "default_one_usize")]
    pub lanes: usize,
    /// Bits per address dimension: ell = ceil(log2(n_side))
    pub ell: usize,
    /// Optional virtual table descriptor (when the full table is not materialized).
    #[serde(default)]
    pub table_spec: Option<LutTableSpec>,
    pub table: Vec<F>,
    /// Optional precomputed digest for transcript binding fast paths (prover-only optimization).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table_digest: Option<[u8; 32]>,
    /// Optional address-sharing group id for shared-bus column layout.
    #[serde(default)]
    pub addr_group: Option<u64>,
    /// Optional selector-sharing group id for shared-bus column layout.
    #[serde(default)]
    pub selector_group: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LutWitness<F> {
    pub mats: Vec<Mat<F>>,
}

#[derive(Clone, Debug)]
pub struct ShoutWitnessLayout {
    pub ell_addr: usize,
    pub addr_bits: Range<usize>,
    pub has_lookup: usize,
    pub val: usize,
}

impl ShoutWitnessLayout {
    pub fn expected_len(&self) -> usize {
        self.ell_addr + 2
    }
}

impl<C, F> LutInstance<C, F> {
    pub fn shout_layout(&self) -> ShoutWitnessLayout {
        let ell_addr = self.d * self.ell;
        ShoutWitnessLayout {
            ell_addr,
            addr_bits: 0..ell_addr,
            has_lookup: ell_addr,
            val: ell_addr + 1,
        }
    }
}

/// Per-step bundle that carries CPU + memory witnesses for a single folding step.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepWitnessBundle<Cmt, F, K> {
    pub mcs: (CcsClaim<Cmt, F>, CcsWitness<F>),
    pub lut_instances: Vec<(LutInstance<Cmt, F>, LutWitness<F>)>,
    pub mem_instances: Vec<(MemInstance<Cmt, F>, MemWitness<F>)>,
    #[serde(default)]
    pub time_columns: TimeColumns<F>,
    #[serde(skip)]
    pub _phantom: PhantomData<K>,
}

impl<Cmt, F, K> From<(CcsClaim<Cmt, F>, CcsWitness<F>)> for StepWitnessBundle<Cmt, F, K> {
    fn from(mcs: (CcsClaim<Cmt, F>, CcsWitness<F>)) -> Self {
        Self {
            mcs,
            lut_instances: Vec::new(),
            mem_instances: Vec::new(),
            time_columns: TimeColumns::default(),
            _phantom: PhantomData,
        }
    }
}

/// Per-step bundle that carries *only public instances* (no witnesses).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepInstanceBundle<Cmt, F, K> {
    pub mcs_inst: CcsClaim<Cmt, F>,
    pub lut_insts: Vec<LutInstance<Cmt, F>>,
    pub mem_insts: Vec<MemInstance<Cmt, F>>,
    // Time columns are verifier-local execution context and should not be serialized
    // as part of public step instances/proofs.
    #[serde(skip, default)]
    pub time_columns: TimeColumns<F>,
    #[serde(skip)]
    pub _phantom: PhantomData<K>,
}

impl<Cmt, F, K> From<CcsClaim<Cmt, F>> for StepInstanceBundle<Cmt, F, K> {
    fn from(mcs_inst: CcsClaim<Cmt, F>) -> Self {
        Self {
            mcs_inst,
            lut_insts: Vec::new(),
            mem_insts: Vec::new(),
            time_columns: TimeColumns::default(),
            _phantom: PhantomData,
        }
    }
}

impl<Cmt: Clone, F: Clone, K> From<&StepWitnessBundle<Cmt, F, K>> for StepInstanceBundle<Cmt, F, K> {
    fn from(step: &StepWitnessBundle<Cmt, F, K>) -> Self {
        Self {
            mcs_inst: step.mcs.0.clone(),
            lut_insts: step
                .lut_instances
                .iter()
                .map(|(inst, _)| inst.clone())
                .collect(),
            mem_insts: step
                .mem_instances
                .iter()
                .map(|(inst, _)| inst.clone())
                .collect(),
            // Canonical time columns are verifier-local context (serde-skip on instance),
            // and are required by Route-A named-opening verification.
            time_columns: step.time_columns.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<Cmt, F, K> From<StepWitnessBundle<Cmt, F, K>> for StepInstanceBundle<Cmt, F, K> {
    fn from(step: StepWitnessBundle<Cmt, F, K>) -> Self {
        let StepWitnessBundle {
            mcs: (mcs_inst, _mcs_wit),
            lut_instances,
            mem_instances,
            time_columns,
            _phantom: _,
        } = step;
        Self {
            mcs_inst,
            lut_insts: lut_instances.into_iter().map(|(inst, _)| inst).collect(),
            mem_insts: mem_instances.into_iter().map(|(inst, _)| inst).collect(),
            // Canonical time columns are verifier-local context (serde-skip on instance),
            // and are required by Route-A named-opening verification.
            time_columns,
            _phantom: PhantomData,
        }
    }
}
