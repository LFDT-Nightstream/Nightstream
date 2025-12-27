use neo_ccs::matrix::Mat;
use neo_ccs::relations::{McsInstance, McsWitness};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::ops::Range;

use crate::mem_init::MemInit;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemInstance<C, F> {
    pub comms: Vec<C>,
    pub k: usize,
    pub d: usize,
    pub n_side: usize,
    pub steps: usize,
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
pub struct TwistWitnessLayout {
    pub ell_addr: usize,
    pub ra_bits: Range<usize>,
    pub wa_bits: Range<usize>,
    pub has_read: usize,
    pub has_write: usize,
    pub wv: usize,
    pub rv: usize,
    pub inc_at_write_addr: usize,
}

impl TwistWitnessLayout {
    pub fn expected_len(&self) -> usize {
        2 * self.ell_addr + 5
    }

    pub fn val_lane_len(&self) -> usize {
        self.ell_addr + 2
    }
}

impl<C, F> MemInstance<C, F> {
    pub fn twist_layout(&self) -> TwistWitnessLayout {
        let ell_addr = self.d * self.ell;
        TwistWitnessLayout {
            ell_addr,
            ra_bits: 0..ell_addr,
            wa_bits: ell_addr..(2 * ell_addr),
            has_read: 2 * ell_addr,
            has_write: 2 * ell_addr + 1,
            wv: 2 * ell_addr + 2,
            rv: 2 * ell_addr + 3,
            inc_at_write_addr: 2 * ell_addr + 4,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LutInstance<C, F> {
    pub comms: Vec<C>,
    pub k: usize,
    pub d: usize,
    pub n_side: usize,
    pub steps: usize,
    /// Bits per address dimension: ell = ceil(log2(n_side))
    pub ell: usize,
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
