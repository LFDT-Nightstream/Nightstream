#![allow(non_snake_case)]

use std::marker::PhantomData;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness, MeInstance};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::output_binding::OutputBindingConfig;
use neo_fold::shard::{fold_shard_prove_with_output_binding, fold_shard_verify_with_output_binding, CommitMixers};
use neo_fold::PiCcsError;
use neo_math::{D, F, K};
use neo_memory::encode::encode_mem_for_twist;
use neo_memory::output_check::ProgramIO;
use neo_memory::plain::{PlainMemLayout, PlainMemTrace};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
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

fn create_identity_ccs(n: usize) -> CcsStructure<F> {
    let mat = Mat::identity(n);
    let f = SparsePoly::new(1, vec![]);
    CcsStructure::new(vec![mat], f).expect("CCS")
}

#[test]
fn output_binding_e2e_wrong_claim_fails() -> Result<(), PiCcsError> {
    // Smallest square CCS that yields ell_n=2 (n_pad=4).
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;
    let l = DummyCommit::default();
    let mixers = default_mixers();

    // One-step Twist trace: write addr 2 := 7 in a 4-cell memory (num_bits=2).
    let mem_layout = PlainMemLayout { k: 4, d: 1, n_side: 4 };
    let mem_init = MemInit::Zero;
    let plain_mem = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ZERO],
        has_write: vec![F::ONE],
        read_addr: vec![0],
        write_addr: vec![2],
        read_val: vec![F::ZERO],
        write_val: vec![F::from_u64(7)],
        inc_at_write_addr: vec![F::from_u64(7)],
    };

    // CPU witness is trivial and unconstrained for this test.
    let z: Vec<F> = vec![F::ZERO; ccs.m];
    let Z = neo_memory::encode::ajtai_encode_vector(&params, &z);
    let c = l.commit(&Z);
    let mcs_inst = McsInstance { c, x: vec![], m_in: 0 };
    let mcs_wit = McsWitness { w: z, Z };

    let commit_fn = |mat: &Mat<F>| l.commit(mat);
    let (mem_inst, mem_wit) = encode_mem_for_twist(
        &params,
        &mem_layout,
        &mem_init,
        &plain_mem,
        &commit_fn,
        Some(ccs.m),
        mcs_inst.m_in,
    );

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

    let ob_cfg_bad = OutputBindingConfig::new(2, ProgramIO::new().with_output(2, F::from_u64(8)));
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
