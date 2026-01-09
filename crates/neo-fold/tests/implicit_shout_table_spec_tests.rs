#![allow(non_snake_case)]

use std::marker::PhantomData;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness, MeInstance};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::CommitMixers;
use neo_fold::shard::{absorb_step_memory, fold_shard_prove, fold_shard_verify};
use neo_math::{D, F, K};
use neo_memory::riscv::lookups::{compute_op, interleave_bits, RiscvOpcode};
use neo_memory::witness::{LutInstance, LutTableSpec, LutWitness, StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

type Mixers = CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>;

const TEST_M: usize = 128;
const M_IN: usize = 0;

#[derive(Clone, Copy, Default)]
struct DummyCommit;

impl SModuleHomomorphism<F, Cmt> for DummyCommit {
    fn commit(&self, z: &Mat<F>) -> Cmt {
        Cmt::zeros(z.rows(), 1)
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

fn default_mixers() -> Mixers {
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
    // Constant polynomial: no active constraints (keeps shared-bus padding guardrails off in tests).
    let f = SparsePoly::new(1, vec![]);
    CcsStructure::new(vec![mat], f).expect("CCS")
}

fn create_mcs_from_z(
    params: &NeoParams,
    l: &DummyCommit,
    m_in: usize,
    z: Vec<F>,
) -> (McsInstance<Cmt, F>, McsWitness<F>) {
    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();
    let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(params, &z);
    let c = l.commit(&Z);

    (McsInstance { c, x, m_in }, McsWitness { w, Z })
}

fn write_shout_bus_row(
    z: &mut [F],
    bus_base: usize,
    chunk_size: usize,
    j: usize,
    inst: &LutInstance<Cmt, F>,
    addr: u64,
    val: F,
    has_lookup: F,
) {
    debug_assert_eq!(chunk_size, 1);
    debug_assert_eq!(j, 0);

    let mut col_id = 0usize;
    if has_lookup == F::ONE {
        let mut tmp = addr;
        for _dim in 0..inst.d {
            let comp = (tmp % (inst.n_side as u64)) as u64;
            tmp /= inst.n_side as u64;
            for bit in 0..inst.ell {
                z[bus_base + col_id * chunk_size + j] = if (comp >> bit) & 1 == 1 { F::ONE } else { F::ZERO };
                col_id += 1;
            }
        }
    } else {
        col_id += inst.d * inst.ell;
    }

    z[bus_base + col_id * chunk_size + j] = has_lookup;
    col_id += 1;
    z[bus_base + col_id * chunk_size + j] = if has_lookup == F::ONE { val } else { F::ZERO };
    col_id += 1;

    debug_assert_eq!(col_id, inst.d * inst.ell + 2);
}

#[test]
fn absorb_step_memory_binds_table_spec() {
    let dummy_mcs = McsInstance {
        c: Cmt::zeros(D, 1),
        x: vec![],
        m_in: 0,
    };

    let make_step = |opcode: RiscvOpcode| StepInstanceBundle::<Cmt, F, K> {
        mcs_inst: dummy_mcs.clone(),
        lut_insts: vec![LutInstance {
            comms: Vec::new(),
            k: 0,
            d: 64,
            n_side: 2,
            steps: 1,
            lanes: 1,
            ell: 1,
            table_spec: Some(LutTableSpec::RiscvOpcode { opcode, xlen: 32 }),
            table: vec![],
            _phantom: PhantomData,
        }],
        mem_insts: vec![],
        _phantom: PhantomData,
    };

    let mut tr_add = Poseidon2Transcript::new(b"bind-table-spec");
    absorb_step_memory(&mut tr_add, &make_step(RiscvOpcode::Add));
    let d_add = tr_add.digest32();

    let mut tr_sub = Poseidon2Transcript::new(b"bind-table-spec");
    absorb_step_memory(&mut tr_sub, &make_step(RiscvOpcode::Sub));
    let d_sub = tr_sub.digest32();

    assert_ne!(d_add, d_sub, "transcript must bind `table_spec` fields");
}

#[test]
fn route_a_shout_implicit_table_spec_verifies() {
    let ccs = create_identity_ccs(TEST_M);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.m).expect("params");
    params.k_rho = 16;

    let l = DummyCommit;
    let mixers = default_mixers();

    let opcode = RiscvOpcode::Add;
    let xlen = 32usize;
    let rs1 = 0x1234_5678u64;
    let rs2 = 0x9abc_def0u64;
    let addr = interleave_bits(rs1, rs2) as u64;
    let out = compute_op(opcode, rs1, rs2, xlen);

    let inst = LutInstance::<Cmt, F> {
        comms: Vec::new(),
        k: 0,
        d: 64,
        n_side: 2,
        steps: 1,
        lanes: 1,
        ell: 1,
        table_spec: Some(LutTableSpec::RiscvOpcode { opcode, xlen }),
        table: vec![],
        _phantom: PhantomData,
    };
    let wit = LutWitness { mats: Vec::new() };

    let bus_cols_total = inst.d * inst.ell + 2;
    let bus_base = ccs.m - bus_cols_total;
    let mut z = vec![F::ZERO; ccs.m];
    write_shout_bus_row(
        &mut z,
        bus_base,
        1,
        0,
        &inst,
        addr,
        F::from_u64(out),
        F::ONE,
    );

    let (mcs, mcs_wit) = create_mcs_from_z(&params, &l, M_IN, z);
    let step_bundle = StepWitnessBundle {
        mcs: (mcs, mcs_wit),
        lut_instances: vec![(inst, wit)],
        mem_instances: vec![],
        _phantom: PhantomData::<K>,
    };

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"implicit-shout-table-spec");
    let proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed for implicit RISC-V Shout table");

    let mut tr_verify = Poseidon2Transcript::new(b"implicit-shout-table-spec");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let _ = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    )
    .expect("verify should succeed for implicit RISC-V Shout table");

    // Changing the opcode in the public instance must break verification.
    let mut tr_verify_bad = Poseidon2Transcript::new(b"implicit-shout-table-spec");
    let mut steps_public_bad = [StepInstanceBundle::from(&step_bundle)];
    steps_public_bad[0].lut_insts[0].table_spec = Some(LutTableSpec::RiscvOpcode {
        opcode: RiscvOpcode::Sub,
        xlen,
    });
    let err = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify_bad,
        &params,
        &ccs,
        &steps_public_bad,
        &acc_init,
        &proof,
        mixers,
    )
    .expect_err("verification should fail under a different opcode");
    let _ = err;
}
