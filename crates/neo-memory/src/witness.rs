use neo_ccs::matrix::Mat;
use neo_ccs::relations::{McsInstance, McsWitness};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::ops::Range;

use crate::mem_init::MemInit;
use crate::riscv::lookups::RiscvOpcode;

fn default_one_usize() -> usize {
    1
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemInstance<C, F> {
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
    #[serde(skip)]
    pub _phantom: PhantomData<F>,
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
    #[serde(skip)]
    pub _phantom: PhantomData<F>,
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
    pub mcs: (McsInstance<Cmt, F>, McsWitness<F>),
    pub lut_instances: Vec<(LutInstance<Cmt, F>, LutWitness<F>)>,
    pub mem_instances: Vec<(MemInstance<Cmt, F>, MemWitness<F>)>,
    #[serde(skip)]
    pub _phantom: PhantomData<K>,
}

impl<Cmt, F, K> From<(McsInstance<Cmt, F>, McsWitness<F>)> for StepWitnessBundle<Cmt, F, K> {
    fn from(mcs: (McsInstance<Cmt, F>, McsWitness<F>)) -> Self {
        Self {
            mcs,
            lut_instances: Vec::new(),
            mem_instances: Vec::new(),
            _phantom: PhantomData,
        }
    }
}

/// Per-step bundle that carries *only public instances* (no witnesses).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepInstanceBundle<Cmt, F, K> {
    pub mcs_inst: McsInstance<Cmt, F>,
    pub lut_insts: Vec<LutInstance<Cmt, F>>,
    pub mem_insts: Vec<MemInstance<Cmt, F>>,
    #[serde(skip)]
    pub _phantom: PhantomData<K>,
}

impl<Cmt, F, K> From<McsInstance<Cmt, F>> for StepInstanceBundle<Cmt, F, K> {
    fn from(mcs_inst: McsInstance<Cmt, F>) -> Self {
        Self {
            mcs_inst,
            lut_insts: Vec::new(),
            mem_insts: Vec::new(),
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
            _phantom: _,
        } = step;
        Self {
            mcs_inst,
            lut_insts: lut_instances.into_iter().map(|(inst, _)| inst).collect(),
            mem_insts: mem_instances.into_iter().map(|(inst, _)| inst).collect(),
            _phantom: PhantomData,
        }
    }
}
