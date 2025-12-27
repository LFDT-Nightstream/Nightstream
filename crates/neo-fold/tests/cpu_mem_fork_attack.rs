#![allow(non_snake_case)]

//! Regression test: CPU↔memory “forked trace” attack is rejected in shared-bus mode.
//!
//! The attack (in the pre-linkage design) is that the prover can satisfy:
//! - CPU constraints using one memory story, and
//! - Twist/Shout using a different (independently committed) memory story,
//! and the verifier accepts.
//!
//! In shared-bus mode, Twist/Shout consume bus openings derived from the CPU commitment, so an
//! independently committed Twist/Shout witness cannot “override” the CPU bus.

use std::marker::PhantomData;

use neo_ajtai::{decomp_b, Commitment as Cmt, DecompStyle};
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{r1cs_to_ccs, Mat};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{
    fold_shard_prove as fold_shard_prove_shared_cpu_bus,
    fold_shard_verify as fold_shard_verify_shared_cpu_bus,
    CommitMixers,
};
use neo_math::{D, F, K};
use neo_memory::encode::encode_mem_for_twist;
use neo_memory::plain::{PlainMemLayout, PlainMemTrace};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

#[derive(Clone, Copy, Default)]
struct HashCommit;

impl HashCommit {
    fn digest_mat(mat: &Mat<F>) -> u64 {
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

#[test]
fn cpu_memory_fork_attack_is_rejected_in_shared_bus_mode() {
    // CPU claims “loaded value” = 1 (public input), and constrains the **bus read-value** to equal it.
    // Twist memory metadata says memory is zero-initialized, so a read from addr=0 must return 0.
    //
    // We attempt the attack by providing:
    // - CPU bus rv = 1 (to satisfy CPU), and
    // - an independently encoded Twist witness with rv = 0 (which would satisfy Twist if it were used).
    //
    // In shared-bus mode, Twist ignores the independent witness and consumes the CPU bus, so the
    // prover/verifier must reject.

    // ---------------------------------------------------------------------
    // 1) CCS/R1CS: enforce (bus_rv - x0) * 1 = 0 and x0^2 = x0, with n=m=10.
    //
    // Witness layout (m_in=2):
    // - z[0] = 1 (public constant)
    // - z[1] = x0 (public)
    // - z[3..10) = CPU bus tail for one Twist instance, chunk_size=1:
    //   [ra_bit, wa_bit, has_read, has_write, wv, rv, inc]
    // So bus_rv is at z[8] (= m - 2).
    // ---------------------------------------------------------------------
    let n = 10usize;
    let m = 10usize;
    let m_in = 2usize;
    let bus_cols = 7usize;
    let bus_base = m - bus_cols; // chunk_size=1
    let bus_rv = bus_base + 5;

    let mut A = Mat::zero(n, m, F::ZERO);
    let mut B = Mat::zero(n, m, F::ZERO);
    let mut C = Mat::zero(n, m, F::ZERO);

    // Row 0: (bus_rv - x0) * 1 = 0
    A[(0, 1)] = -F::ONE;
    A[(0, bus_rv)] = F::ONE;
    B[(0, 0)] = F::ONE;

    // Row 1: x0^2 = x0 (booleanize x0)
    A[(1, 1)] = F::ONE;
    B[(1, 1)] = F::ONE;
    C[(1, 1)] = F::ONE;

    let ccs: CcsStructure<F> = r1cs_to_ccs(A, B, C);
    assert_eq!(ccs.n, n);
    assert_eq!(ccs.m, m);

    // ---------------------------------------------------------------------
    // 2) CPU witness: z[0]=1, x0=1, and bus encodes a read returning rv=1.
    // ---------------------------------------------------------------------
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    params.k_rho = 16;

    let l = HashCommit::default();
    let mixers = default_mixers();

    let mut z_cpu: Vec<F> = vec![F::ZERO; ccs.m];
    z_cpu[0] = F::ONE; // constant 1
    z_cpu[1] = F::ONE; // x0

    // Fill bus (chunk_size=1, one Twist instance).
    // [ra_bit, wa_bit, has_read, has_write, wv, rv, inc] at z[bus_base..bus_base+7).
    z_cpu[bus_base + 0] = F::ZERO; // ra_bit (addr 0)
    z_cpu[bus_base + 1] = F::ZERO; // wa_bit
    z_cpu[bus_base + 2] = F::ONE; // has_read
    z_cpu[bus_base + 3] = F::ZERO; // has_write
    z_cpu[bus_base + 4] = F::ZERO; // wv
    z_cpu[bus_base + 5] = F::ONE; // rv (CPU claims 1)
    z_cpu[bus_base + 6] = F::ZERO; // inc

    let Z_cpu = decompose_z_to_Z(&params, &z_cpu);
    let c_cpu = l.commit(&Z_cpu);

    let mcs = (
        McsInstance {
            c: c_cpu,
            x: z_cpu[..m_in].to_vec(),
            m_in,
        },
        McsWitness {
            w: z_cpu[m_in..].to_vec(),
            Z: Z_cpu,
        },
    );

    // ---------------------------------------------------------------------
    // 3) Twist memory witness: one read returning 0 from zero-init memory.
    //    (This would satisfy Twist if it were committed/checked independently.)
    // ---------------------------------------------------------------------
    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
    let mem_init = MemInit::Zero;

    let mem_trace = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ONE],
        has_write: vec![F::ZERO],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO], // <-- memory says read value is 0
        write_val: vec![F::ZERO],
        inc_at_write_addr: vec![F::ZERO],
    };

    let commit_fn = |mat: &Mat<F>| l.commit(mat);
    let (mem_inst, mem_wit) = encode_mem_for_twist(
        &params,
        &mem_layout,
        &mem_init,
        &mem_trace,
        &commit_fn,
        ccs.m,
        m_in,
    );

    let steps_witness = vec![StepWitnessBundle {
        mcs,
        lut_instances: vec![],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData::<K>,
    }];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    // ---------------------------------------------------------------------
    // 4) Prove + verify (shared-bus mode).
    // ---------------------------------------------------------------------
    let mut tr = Poseidon2Transcript::new(b"cpu-mem-fork-attack");
    let prove_res = fold_shard_prove_shared_cpu_bus(
        FoldingMode::Optimized,
        &mut tr,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    );

    let mut tr_v = Poseidon2Transcript::new(b"cpu-mem-fork-attack");
    match prove_res {
        Err(_) => {
            // Expected: Twist's addr-pre sumcheck cannot be satisfied because the CPU bus claims rv=1
            // but memory is zero-init, so the shared-bus memory argument rejects.
        }
        Ok(proof) => {
            let verify_res = fold_shard_verify_shared_cpu_bus(
                FoldingMode::Optimized,
                &mut tr_v,
                &params,
                &ccs,
                &steps_instance,
                &[],
                &proof,
                mixers,
            );
            assert!(
                verify_res.is_err(),
                "verifier must reject forked CPU rv=1 vs Twist zero-init read rv=0 in shared-bus mode"
            );
        }
    }
}
