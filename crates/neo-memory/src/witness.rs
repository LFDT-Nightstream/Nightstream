use neo_ccs::matrix::Mat;
use neo_ccs::relations::{McsInstance, McsWitness};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

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
    /// Public initial memory values for cells [0..k).
    /// If you want zero-init, set this to vec![F::ZERO; k].
    pub init_vals: Vec<F>,
    #[serde(skip)]
    pub _phantom: PhantomData<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemWitness<F> {
    pub mats: Vec<Mat<F>>,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardWitnessBundle<Cmt, F, K> {
    pub mcss: Vec<(McsInstance<Cmt, F>, McsWitness<F>)>,
    pub lut_shard_instances: Vec<(LutInstance<Cmt, F>, LutWitness<F>)>,
    pub mem_shard_instances: Vec<(MemInstance<Cmt, F>, MemWitness<F>)>,
    #[serde(skip)]
    pub _phantom: PhantomData<K>,
}
