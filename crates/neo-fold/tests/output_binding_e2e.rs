#![allow(non_snake_case)]

use std::marker::PhantomData;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::matrix::Mat;
use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness, MeInstance};
use neo_ccs::traits::SModuleHomomorphism;
use neo_fold::output_binding::OutputBindingConfig;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{
    fold_shard_prove_with_output_binding, fold_shard_verify_with_output_binding, CommitMixers,
};
use neo_fold::PiCcsError;
use neo_math::{D, F, K};
use neo_memory::cpu::build_bus_layout_for_instances;
use neo_memory::cpu::constraints::{extend_ccs_with_shared_cpu_bus_constraints, TwistCpuBinding};
use neo_memory::output_check::ProgramIO;
use neo_memory::witness::{MemInstance, MemWitness, StepInstanceBundle, StepWitnessBundle};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

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

fn default_mixers() -> CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt> {
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

fn empty_identity_first_r1cs_ccs(n: usize) -> CcsStructure<F> {
    let i_n = Mat::identity(n);
    let a = Mat::zero(n, n, F::ZERO);
    let b = Mat::zero(n, n, F::ZERO);
    let c = Mat::zero(n, n, F::ZERO);

    // f(I, A, B, C) = A * B - C, with I unused.
    let f = SparsePoly::new(
        4,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![0, 1, 1, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 0, 1],
            },
        ],
    );
    CcsStructure::new(vec![i_n, a, b, c], f).expect("CCS")
}

fn create_mcs_from_z(params: &NeoParams, l: &DummyCommit, m_in: usize, z: Vec<F>) -> (McsInstance<Cmt, F>, McsWitness<F>) {
    let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(params, &z);
    let c = l.commit(&Z);
    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();
    (McsInstance { c, x, m_in }, McsWitness { w, Z })
}

#[test]
fn output_binding_e2e_wrong_claim_fails() -> Result<(), PiCcsError> {
    // R1CS base CCS with enough slack rows to inject shared-bus constraints.
    let n = 64usize;
    let base_ccs = empty_identity_first_r1cs_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;
    let l = DummyCommit::default();
    let mixers = default_mixers();

    // One Twist instance: 4-cell memory (num_bits=2), single write at addr=2 to value=7.
    let m_in = 1usize;
    let const_one_col = 0usize;

    let mem_inst = MemInstance::<Cmt, F> {
        comms: Vec::new(),
        k: 4,
        d: 1,
        n_side: 4,
        steps: 1,
        ell: 2,
        init: MemInit::Zero,
        _phantom: PhantomData,
    };
    let mem_wit = MemWitness { mats: Vec::new() };

    // Minimal CPU columns used by the injected constraints (all < bus_base).
    const COL_HAS_READ: usize = 1;
    const COL_HAS_WRITE: usize = 2;
    const COL_READ_ADDR: usize = 3;
    const COL_WRITE_ADDR: usize = 4;
    const COL_RV: usize = 5;
    const COL_WV: usize = 6;
    const COL_INC: usize = 7;

    let twist_cpu = vec![TwistCpuBinding {
        has_read: COL_HAS_READ,
        has_write: COL_HAS_WRITE,
        read_addr: COL_READ_ADDR,
        write_addr: COL_WRITE_ADDR,
        rv: COL_RV,
        wv: COL_WV,
        inc: Some(COL_INC),
    }];

    let ccs = extend_ccs_with_shared_cpu_bus_constraints(
        &base_ccs,
        m_in,
        const_one_col,
        &[], // no Shout instances
        &twist_cpu,
        &[], // no LUT instances
        &[mem_inst.clone()],
    )
    .map_err(|e| PiCcsError::InvalidInput(e))?;

    let bus = build_bus_layout_for_instances(
        ccs.m,
        m_in,
        1,
        core::iter::empty(),
        core::iter::once(mem_inst.d * mem_inst.ell),
    )
    .map_err(PiCcsError::InvalidInput)?;

    // Build CPU witness z = [x | w] including the shared bus tail.
    let mut z = vec![F::ZERO; ccs.m];
    z[0] = F::ONE; // public const-one

    // CPU semantic columns (must match bus via injected constraints).
    z[COL_HAS_READ] = F::ZERO;
    z[COL_HAS_WRITE] = F::ONE;
    z[COL_READ_ADDR] = F::ZERO;
    z[COL_WRITE_ADDR] = F::from_u64(2);
    z[COL_RV] = F::ZERO;
    z[COL_WV] = F::from_u64(7);
    z[COL_INC] = F::from_u64(7);

    // Bus tail (Twist only).
    let twist = &bus.twist_cols[0];
    // ra_bits = 0 when has_read=0
    for col_id in twist.ra_bits.clone() {
        z[bus.bus_cell(col_id, 0)] = F::ZERO;
    }
    // wa_bits for addr=2 (little-endian bits: [0]=0, [1]=1)
    let wa_bits = [F::ZERO, F::ONE];
    for (i, col_id) in twist.wa_bits.clone().enumerate() {
        z[bus.bus_cell(col_id, 0)] = wa_bits[i];
    }
    z[bus.bus_cell(twist.has_read, 0)] = F::ZERO;
    z[bus.bus_cell(twist.has_write, 0)] = F::ONE;
    z[bus.bus_cell(twist.wv, 0)] = F::from_u64(7);
    z[bus.bus_cell(twist.rv, 0)] = F::ZERO;
    z[bus.bus_cell(twist.inc, 0)] = F::from_u64(7);

    let (mcs_inst, mcs_wit) = create_mcs_from_z(&params, &l, m_in, z);

    let steps_witness: Vec<StepWitnessBundle<Cmt, F, K>> = vec![StepWitnessBundle {
        mcs: (mcs_inst, mcs_wit),
        lut_instances: vec![],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData,
    }];
    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let final_memory_state = vec![F::ZERO, F::ZERO, F::from_u64(7), F::ZERO];

    let ob_cfg_ok = OutputBindingConfig::new(2, ProgramIO::new().with_output(2, F::from_u64(7)));
    let ob_cfg_bad = OutputBindingConfig::new(2, ProgramIO::new().with_output(2, F::from_u64(8)));

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"output-binding-e2e");
    let proof = fold_shard_prove_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
        &ob_cfg_ok,
        &final_memory_state,
    )?;

    let mut tr_verify_ok = Poseidon2Transcript::new(b"output-binding-e2e");
    let _outputs_ok = fold_shard_verify_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_verify_ok,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &ob_cfg_ok,
    )?;

    let mut tr_verify_bad = Poseidon2Transcript::new(b"output-binding-e2e");
    let res = fold_shard_verify_with_output_binding(
        FoldingMode::PaperExact,
        &mut tr_verify_bad,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &ob_cfg_bad,
    );
    assert!(res.is_err(), "wrong output claim must fail verification");

    Ok(())
}
