#![allow(non_snake_case)]

use std::marker::PhantomData;

use neo_ajtai::{decomp_b, Commitment as Cmt, DecompStyle};
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness, MeInstance};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{
    fold_shard_prove, fold_shard_verify, fold_shard_verify_and_finalize, ShardFoldOutputs, ShardProof,
};
use neo_fold::shard::CommitMixers;
use neo_fold::{finalize::ObligationFinalizer, PiCcsError};
use neo_math::{D, F, K};
use neo_memory::encode::{encode_lut_for_shout, encode_mem_for_twist};
use neo_memory::plain::{LutTable, PlainLutTrace, PlainMemLayout, PlainMemTrace};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

pub type Mixers = CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>;

#[derive(Clone, Copy, Default)]
pub struct HashCommit;

impl HashCommit {
    fn digest_mat(mat: &Mat<F>) -> u64 {
        // Simple FNV-1a style hash over dimensions + entries.
        let mut h: u64 = 0xcbf29ce484222325;
        h ^= mat.rows() as u64;
        h = h.wrapping_mul(0x100000001b3);
        h ^= mat.cols() as u64;
        h = h.wrapping_mul(0x100000001b3);
        for r in 0..mat.rows() {
            for c in 0..mat.cols() {
                h ^= mat[(r, c)].as_canonical_u64();
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }
}

impl SModuleHomomorphism<F, Cmt> for HashCommit {
    fn commit(&self, z: &Mat<F>) -> Cmt {
        let h = Self::digest_mat(z);
        let base = F::from_u64(h);
        let mut out = Cmt::zeros(z.rows(), 1);
        for i in 0..z.rows() {
            out.data[i] = base + F::from_u64(i as u64);
        }
        out
    }

    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows();
        let mut out = Mat::zero(rows, m_in, F::ZERO);
        for r in 0..rows {
            for c in 0..m_in.min(z.cols()) {
                out[(r, c)] = z[(r, c)];
            }
        }
        out
    }
}

pub fn default_mixers() -> Mixers {
    fn mix_rhos_commits(_rhos: &[Mat<F>], _cs: &[Cmt]) -> Cmt {
        Cmt::zeros(D, 1)
    }
    fn combine_b_pows(_cs: &[Cmt], _b: u32) -> Cmt {
        Cmt::zeros(D, 1)
    }
    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

fn create_identity_ccs(n: usize) -> CcsStructure<F> {
    let mat = Mat::identity(n);
    let f = SparsePoly::new(1, vec![]);
    CcsStructure::new(vec![mat], f).expect("CCS")
}

fn decompose_z_to_Z(params: &NeoParams, z: &[F]) -> Mat<F> {
    let d = D;
    let m = z.len();
    let digits = decomp_b(z, params.b, d, DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for c in 0..m {
        for r in 0..d {
            row_major[r * m + c] = digits[c * d + r];
        }
    }
    Mat::from_row_major(d, m, row_major)
}

fn create_mcs(
    params: &NeoParams,
    ccs: &CcsStructure<F>,
    l: &HashCommit,
    m_in: usize,
    tag: u64,
) -> (McsInstance<Cmt, F>, McsWitness<F>) {
    let m = ccs.m;
    let mut z: Vec<F> = vec![F::ZERO; m];
    if !z.is_empty() {
        z[0] = F::from_u64(tag);
    }
    if z.len() > 1 {
        z[1] = F::from_u64(tag.wrapping_add(1));
    }

    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();
    let Z = decompose_z_to_Z(params, &z);
    let c = l.commit(&Z);

    (McsInstance { c, x, m_in }, McsWitness { w, Z })
}

#[derive(Clone)]
pub struct ShardFixture {
    pub params: NeoParams,
    pub ccs: CcsStructure<F>,
    pub steps_witness: Vec<StepWitnessBundle<Cmt, F, K>>,
    pub steps_instance: Vec<StepInstanceBundle<Cmt, F, K>>,
    pub acc_init: Vec<MeInstance<Cmt, F, K>>,
    pub acc_wit_init: Vec<Mat<F>>,
    pub l: HashCommit,
    pub mixers: Mixers,
}

fn build_twist_shout_2step_fixture_inner(seed: u64, bad_lookup_step1: bool) -> ShardFixture {
    // Keep CCS tiny: n=m=4, with M0=I so Route A's identity-first assumption holds.
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = HashCommit::default();
    let mixers = default_mixers();

    // Empty initial accumulator (start from scratch).
    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    // Per-step MCS instances (vary tags so transcript-binding is strong).
    let (mcs0, mcs_wit0) = create_mcs(&params, &ccs, &l, 0, seed);
    let (mcs1, mcs_wit1) = create_mcs(&params, &ccs, &l, 0, seed.wrapping_add(10_000));

    // Memory: k=2, d=1, n_side=2 (minimal nontrivial).
    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
    let mem_init0 = MemInit::Zero;

    let write0 = F::from_u64(seed.wrapping_add(10));
    let write1 = write0 + F::ONE;

    let mem_trace0 = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ZERO],
        has_write: vec![F::ONE],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO],
        write_val: vec![write0],
        inc_at_write_addr: vec![write0],
    };
    let mem_init1 = MemInit::Sparse(vec![(0, write0)]);
    let mem_trace1 = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ONE],
        has_write: vec![F::ONE],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![write0],
        write_val: vec![write1],
        inc_at_write_addr: vec![write1 - write0],
    };

    // Shout table: k=2, d=1, n_side=2 (minimal nontrivial).
    let lut_table = LutTable {
        table_id: 0,
        k: 2,
        d: 1,
        n_side: 2,
        content: vec![F::from_u64(11), F::from_u64(22)],
    };
    let lut_trace0 = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![0],
        val: vec![lut_table.content[0]],
    };
    let lut_trace1_val = if bad_lookup_step1 {
        lut_table.content[1] + F::ONE
    } else {
        lut_table.content[1]
    };
    let lut_trace1 = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![1],
        val: vec![lut_trace1_val],
    };

    let commit_fn = |mat: &Mat<F>| l.commit(mat);
    let (mem_inst0, mem_wit0) = encode_mem_for_twist(
        &params,
        &mem_layout,
        &mem_init0,
        &mem_trace0,
        &commit_fn,
        Some(ccs.m),
        mcs0.m_in,
    );
    let (lut_inst0, lut_wit0) =
        encode_lut_for_shout(&params, &lut_table, &lut_trace0, &commit_fn, Some(ccs.m), mcs0.m_in);

    let (mem_inst1, mem_wit1) = encode_mem_for_twist(
        &params,
        &mem_layout,
        &mem_init1,
        &mem_trace1,
        &commit_fn,
        Some(ccs.m),
        mcs1.m_in,
    );
    let (lut_inst1, lut_wit1) =
        encode_lut_for_shout(&params, &lut_table, &lut_trace1, &commit_fn, Some(ccs.m), mcs1.m_in);

    let step0 = StepWitnessBundle {
        mcs: (mcs0, mcs_wit0),
        lut_instances: vec![(lut_inst0, lut_wit0)],
        mem_instances: vec![(mem_inst0, mem_wit0)],
        _phantom: PhantomData::<K>,
    };
    let step1 = StepWitnessBundle {
        mcs: (mcs1, mcs_wit1),
        lut_instances: vec![(lut_inst1, lut_wit1)],
        mem_instances: vec![(mem_inst1, mem_wit1)],
        _phantom: PhantomData::<K>,
    };

    let steps_witness = vec![step0, step1];
    let steps_instance = steps_witness.iter().map(StepInstanceBundle::from).collect();

    ShardFixture {
        params,
        ccs,
        steps_witness,
        steps_instance,
        acc_init,
        acc_wit_init,
        l,
        mixers,
    }
}

pub fn build_twist_shout_2step_fixture(seed: u64) -> ShardFixture {
    build_twist_shout_2step_fixture_inner(seed, false)
}

pub fn build_twist_shout_2step_fixture_bad_lookup(seed: u64) -> ShardFixture {
    build_twist_shout_2step_fixture_inner(seed, true)
}

pub fn prove(mode: FoldingMode, fx: &ShardFixture) -> ShardProof {
    let mut tr = Poseidon2Transcript::new(b"twist-shout/fixture");
    fold_shard_prove(
        mode,
        &mut tr,
        &fx.params,
        &fx.ccs,
        &fx.steps_witness,
        &fx.acc_init,
        &fx.acc_wit_init,
        &fx.l,
        fx.mixers,
    )
    .expect("prove should succeed")
}

pub fn verify(
    mode: FoldingMode,
    fx: &ShardFixture,
    proof: &ShardProof,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError> {
    let mut tr = Poseidon2Transcript::new(b"twist-shout/fixture");
    fold_shard_verify(
        mode,
        &mut tr,
        &fx.params,
        &fx.ccs,
        &fx.steps_instance,
        &fx.acc_init,
        proof,
        fx.mixers,
    )
}

pub fn verify_and_finalize<Fin>(
    mode: FoldingMode,
    fx: &ShardFixture,
    proof: &ShardProof,
    finalizer: &mut Fin,
) -> Result<(), PiCcsError>
where
    Fin: ObligationFinalizer<Cmt, F, K, Error = PiCcsError>,
{
    let mut tr = Poseidon2Transcript::new(b"twist-shout/fixture");
    fold_shard_verify_and_finalize(
        mode,
        &mut tr,
        &fx.params,
        &fx.ccs,
        &fx.steps_instance,
        &fx.acc_init,
        proof,
        fx.mixers,
        finalizer,
    )
}
