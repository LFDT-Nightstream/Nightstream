//! Memory (Twist) adversarial tests.
//!
//! These tests validate that the Twist (read-write memory) protocol correctly
//! detects malicious witnesses that violate memory consistency.
//!
//! ## Coverage
//! - `memory_cross_step_read_consistency`: Read must return last written value
//! - `memory_read_uninitialized_returns_zero`: Uninitialized memory reads zero
//! - `memory_tamper_read_value_fails`: Adversarial: claim wrong read value
//! - `memory_tamper_write_increment_fails`: Adversarial: wrong delta computation
//! - `memory_double_write_same_step`: Write twice to same address in one step
//! - `memory_read_after_write_same_step`: Read-after-write within same step

#![allow(non_snake_case)]

use std::marker::PhantomData;
use std::sync::Arc;

use neo_ajtai::{decomp_b, setup as ajtai_setup, set_global_pp, AjtaiSModule, Commitment as Cmt, DecompStyle};
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness, MeInstance};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::shard::CommitMixers;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify};
use neo_math::{D, F, K};
use neo_memory::encode::encode_mem_for_twist;
use neo_memory::plain::{PlainMemLayout, PlainMemTrace};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Mixers = CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>;

/// Setup real Ajtai public parameters for tests.
fn setup_ajtai_pp(m: usize, seed: u64) -> AjtaiSModule {
    let d = D;
    let kappa = neo_params::NeoParams::goldilocks_auto_r1cs_ccs(m)
        .expect("params")
        .kappa as usize;

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let pp = ajtai_setup(&mut rng, d, kappa, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp.clone());
    AjtaiSModule::new(Arc::new(pp))
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
    l: &AjtaiSModule,
    tag: u64,
) -> (McsInstance<Cmt, F>, McsWitness<F>) {
    let m = ccs.m;
    let mut z: Vec<F> = vec![F::ZERO; m];
    if !z.is_empty() {
        z[0] = F::from_u64(tag);
    }

    let Z = decompose_z_to_Z(params, &z);
    let c = l.commit(&Z);

    (
        McsInstance { c, x: vec![], m_in: 0 },
        McsWitness { w: z, Z },
    )
}

/// Valid 3-step memory trace:
/// Step 0: Write 42 to addr[0]
/// Step 1: Read addr[0] → 42
/// Step 2: Write 100 to addr[0], overwriting 42
#[test]
fn memory_cross_step_read_consistency() {
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x2001);
    let mixers = default_mixers();
    let commit_fn = |mat: &Mat<F>| l.commit(mat);

    let mem_layout = PlainMemLayout { k: 4, d: 1, n_side: 4 };

    // 3-step trace testing cross-step consistency
    let mut steps: Vec<StepWitnessBundle<Cmt, F, K>> = Vec::new();

    // Step 0: Write 42 to addr[0]
    {
        let mem_trace = PlainMemTrace {
            steps: 1,
            has_read: vec![F::ZERO],
            has_write: vec![F::ONE],
            read_addr: vec![0],
            write_addr: vec![0],
            read_val: vec![F::ZERO],
            write_val: vec![F::from_u64(42)],
            inc_at_write_addr: vec![F::from_u64(42)], // 42 - 0 = 42
        };
        let mem_init = MemInit::Zero;

        let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, 0);
        let (mem_inst, mem_wit) = encode_mem_for_twist(
            &params,
            &mem_layout,
            &mem_init,
            &mem_trace,
            &commit_fn,
            Some(ccs.m),
            mcs.m_in,
        );

        steps.push(StepWitnessBundle {
            mcs: (mcs, mcs_wit),
            lut_instances: vec![],
            mem_instances: vec![(mem_inst, mem_wit)],
            _phantom: PhantomData::<K>,
        });
    }

    // Step 1: Read addr[0] → should be 42
    {
        let mem_trace = PlainMemTrace {
            steps: 1,
            has_read: vec![F::ONE],
            has_write: vec![F::ZERO],
            read_addr: vec![0],
            write_addr: vec![0],
            read_val: vec![F::from_u64(42)], // Must match what was written
            write_val: vec![F::ZERO],
            inc_at_write_addr: vec![F::ZERO],
        };
        // Memory state after step 0: addr[0] = 42
        let mem_init = MemInit::Sparse(vec![(0, F::from_u64(42))]);

        let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, 1);
        let (mem_inst, mem_wit) = encode_mem_for_twist(
            &params,
            &mem_layout,
            &mem_init,
            &mem_trace,
            &commit_fn,
            Some(ccs.m),
            mcs.m_in,
        );

        steps.push(StepWitnessBundle {
            mcs: (mcs, mcs_wit),
            lut_instances: vec![],
            mem_instances: vec![(mem_inst, mem_wit)],
            _phantom: PhantomData::<K>,
        });
    }

    // Step 2: Write 100 to addr[0]
    {
        let mem_trace = PlainMemTrace {
            steps: 1,
            has_read: vec![F::ZERO],
            has_write: vec![F::ONE],
            read_addr: vec![0],
            write_addr: vec![0],
            read_val: vec![F::ZERO],
            write_val: vec![F::from_u64(100)],
            inc_at_write_addr: vec![F::from_u64(100) - F::from_u64(42)], // 100 - 42 = 58
        };
        let mem_init = MemInit::Sparse(vec![(0, F::from_u64(42))]);

        let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, 2);
        let (mem_inst, mem_wit) = encode_mem_for_twist(
            &params,
            &mem_layout,
            &mem_init,
            &mem_trace,
            &commit_fn,
            Some(ccs.m),
            mcs.m_in,
        );

        steps.push(StepWitnessBundle {
            mcs: (mcs, mcs_wit),
            lut_instances: vec![],
            mem_instances: vec![(mem_inst, mem_wit)],
            _phantom: PhantomData::<K>,
        });
    }

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"mem-cross-step");
    let proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps,
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed for valid cross-step memory");

    let mut tr_verify = Poseidon2Transcript::new(b"mem-cross-step");
    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps.iter().map(StepInstanceBundle::from).collect();
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
    .expect("verify should succeed for valid cross-step memory");

    println!("✓ memory_cross_step_read_consistency: 3-step write→read→write verified");
}

/// Reading from uninitialized memory should return zero.
#[test]
fn memory_read_uninitialized_returns_zero() {
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x2001);
    let mixers = default_mixers();
    let commit_fn = |mat: &Mat<F>| l.commit(mat);

    let mem_layout = PlainMemLayout { k: 4, d: 1, n_side: 4 };

    // Read from addr[2] which was never written (should be 0)
    let mem_trace = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ONE],
        has_write: vec![F::ZERO],
        read_addr: vec![2],
        write_addr: vec![0],
        read_val: vec![F::ZERO], // Uninitialized = 0
        write_val: vec![F::ZERO],
        inc_at_write_addr: vec![F::ZERO],
    };
    let mem_init = MemInit::Zero;

    let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, 0);
    let (mem_inst, mem_wit) = encode_mem_for_twist(
        &params,
        &mem_layout,
        &mem_init,
        &mem_trace,
        &commit_fn,
        Some(ccs.m),
        mcs.m_in,
    );

    let step_bundle = StepWitnessBundle {
        mcs: (mcs, mcs_wit),
        lut_instances: vec![],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData::<K>,
    };

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"mem-uninitialized");
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
    .expect("prove should succeed for uninitialized read");

    let mut tr_verify = Poseidon2Transcript::new(b"mem-uninitialized");
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
    .expect("verify should succeed for uninitialized read returning zero");

    println!("✓ memory_read_uninitialized_returns_zero: Uninitialized memory reads as zero");
}

/// Adversarial: Claim wrong read value.
/// Memory has addr[0]=42, but prover claims read returns 99.
#[test]
fn memory_tamper_read_value_fails() {
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x2001);
    let mixers = default_mixers();
    let commit_fn = |mat: &Mat<F>| l.commit(mat);

    let mem_layout = PlainMemLayout { k: 4, d: 1, n_side: 4 };

    // Memory state: addr[0] = 42
    let mem_init = MemInit::Sparse(vec![(0, F::from_u64(42))]);

    // Malicious: claim read returns 99 instead of 42
    let bad_mem_trace = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ONE],
        has_write: vec![F::ZERO],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::from_u64(99)], // WRONG: should be 42
        write_val: vec![F::ZERO],
        inc_at_write_addr: vec![F::ZERO],
    };

    let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, 0);

    // The bad encoding should either panic during encode (semantic check)
    // or fail during verification.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        encode_mem_for_twist(
            &params,
            &mem_layout,
            &mem_init,
            &bad_mem_trace,
            &commit_fn,
            Some(ccs.m),
            mcs.m_in,
        )
    }));

    match result {
        Err(_) => {
            println!("✓ memory_tamper_read_value_fails: Encoding correctly panicked on invalid read value");
        }
        Ok((mem_inst, mem_wit)) => {
            // If encoding passed, semantic check should fail
            let check_result = neo_memory::twist::check_twist_semantics(&params, &mem_inst, &mem_wit);

            if check_result.is_err() {
                println!("✓ memory_tamper_read_value_fails: Semantic check correctly rejected invalid read value");
                return;
            }

            // If semantic check passed, verification should fail
            let step_bundle = StepWitnessBundle {
                mcs: (mcs, mcs_wit),
                lut_instances: vec![],
                mem_instances: vec![(mem_inst, mem_wit)],
                _phantom: PhantomData::<K>,
            };

            let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
            let acc_wit_init: Vec<Mat<F>> = Vec::new();

            let mut tr_prove = Poseidon2Transcript::new(b"mem-tamper-read");
            let proof_result = fold_shard_prove(
                FoldingMode::PaperExact,
                &mut tr_prove,
                &params,
                &ccs,
                &[step_bundle.clone()],
                &acc_init,
                &acc_wit_init,
                &l,
                mixers,
            );

            if let Ok(proof) = proof_result {
                let mut tr_verify = Poseidon2Transcript::new(b"mem-tamper-read");
                let steps_public = [StepInstanceBundle::from(&step_bundle)];
                let verify_result = fold_shard_verify(
                    FoldingMode::PaperExact,
                    &mut tr_verify,
                    &params,
                    &ccs,
                    &steps_public,
                    &acc_init,
                    &proof,
                    mixers,
                );

                assert!(
                    verify_result.is_err(),
                    "Verification should fail when read value is tampered"
                );
                println!("✓ memory_tamper_read_value_fails: Verification correctly rejected invalid read value");
            } else {
                println!("✓ memory_tamper_read_value_fails: Proving correctly failed on invalid read value");
            }
        }
    }
}

/// Adversarial: Claim wrong write increment.
/// Writing 100 to addr[0] which previously had 42, but claiming delta = 10 instead of 58.
#[test]
fn memory_tamper_write_increment_fails() {
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x2001);
    let mixers = default_mixers();
    let commit_fn = |mat: &Mat<F>| l.commit(mat);

    let mem_layout = PlainMemLayout { k: 4, d: 1, n_side: 4 };

    // Memory state: addr[0] = 42
    let mem_init = MemInit::Sparse(vec![(0, F::from_u64(42))]);

    // Malicious: claim wrong increment (should be 100-42=58, but claim 10)
    let bad_mem_trace = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ZERO],
        has_write: vec![F::ONE],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO],
        write_val: vec![F::from_u64(100)],
        inc_at_write_addr: vec![F::from_u64(10)], // WRONG: should be 58
    };

    let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, 0);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        encode_mem_for_twist(
            &params,
            &mem_layout,
            &mem_init,
            &bad_mem_trace,
            &commit_fn,
            Some(ccs.m),
            mcs.m_in,
        )
    }));

    match result {
        Err(_) => {
            println!("✓ memory_tamper_write_increment_fails: Encoding correctly panicked on invalid increment");
        }
        Ok((mem_inst, mem_wit)) => {
            let check_result = neo_memory::twist::check_twist_semantics(&params, &mem_inst, &mem_wit);

            if check_result.is_err() {
                println!("✓ memory_tamper_write_increment_fails: Semantic check correctly rejected invalid increment");
                return;
            }

            let step_bundle = StepWitnessBundle {
                mcs: (mcs, mcs_wit),
                lut_instances: vec![],
                mem_instances: vec![(mem_inst, mem_wit)],
                _phantom: PhantomData::<K>,
            };

            let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
            let acc_wit_init: Vec<Mat<F>> = Vec::new();

            let mut tr_prove = Poseidon2Transcript::new(b"mem-tamper-inc");
            let proof_result = fold_shard_prove(
                FoldingMode::PaperExact,
                &mut tr_prove,
                &params,
                &ccs,
                &[step_bundle.clone()],
                &acc_init,
                &acc_wit_init,
                &l,
                mixers,
            );

            if let Ok(proof) = proof_result {
                let mut tr_verify = Poseidon2Transcript::new(b"mem-tamper-inc");
                let steps_public = [StepInstanceBundle::from(&step_bundle)];
                let verify_result = fold_shard_verify(
                    FoldingMode::PaperExact,
                    &mut tr_verify,
                    &params,
                    &ccs,
                    &steps_public,
                    &acc_init,
                    &proof,
                    mixers,
                );

                assert!(
                    verify_result.is_err(),
                    "Verification should fail when increment is tampered"
                );
                println!("✓ memory_tamper_write_increment_fails: Verification correctly rejected invalid increment");
            } else {
                println!("✓ memory_tamper_write_increment_fails: Proving correctly failed on invalid increment");
            }
        }
    }
}

/// Test multiple memory regions in same step.
/// This simulates a VM with separate RAM, ROM, and register file.
#[test]
fn memory_multiple_regions_same_step() {
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x2001);
    let mixers = default_mixers();
    let commit_fn = |mat: &Mat<F>| l.commit(mat);

    // Region 0: RAM (read-write)
    let ram_layout = PlainMemLayout { k: 4, d: 1, n_side: 4 };
    let ram_trace = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ZERO],
        has_write: vec![F::ONE],
        read_addr: vec![0],
        write_addr: vec![1],
        read_val: vec![F::ZERO],
        write_val: vec![F::from_u64(42)],
        inc_at_write_addr: vec![F::from_u64(42)],
    };
    let ram_init = MemInit::Zero;

    // Region 1: Register file (read-write, smaller)
    let reg_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
    let reg_trace = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ONE],
        has_write: vec![F::ONE],
        read_addr: vec![0],
        write_addr: vec![1],
        read_val: vec![F::from_u64(10)],
        write_val: vec![F::from_u64(20)],
        inc_at_write_addr: vec![F::from_u64(20)],
    };
    let reg_init = MemInit::Sparse(vec![(0, F::from_u64(10))]);

    let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, 0);

    let (ram_inst, ram_wit) = encode_mem_for_twist(
        &params,
        &ram_layout,
        &ram_init,
        &ram_trace,
        &commit_fn,
        Some(ccs.m),
        mcs.m_in,
    );
    let (reg_inst, reg_wit) = encode_mem_for_twist(
        &params,
        &reg_layout,
        &reg_init,
        &reg_trace,
        &commit_fn,
        Some(ccs.m),
        mcs.m_in,
    );

    let step_bundle = StepWitnessBundle {
        mcs: (mcs, mcs_wit),
        lut_instances: vec![],
        mem_instances: vec![(ram_inst, ram_wit), (reg_inst, reg_wit)],
        _phantom: PhantomData::<K>,
    };

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"mem-multi-region");
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
    .expect("prove should succeed with multiple memory regions");

    let mut tr_verify = Poseidon2Transcript::new(b"mem-multi-region");
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
    .expect("verify should succeed with multiple memory regions");

    println!("✓ memory_multiple_regions_same_step: RAM + register file in same step verified");
}

/// Test memory with sparse initialization (ROM-like behavior).
#[test]
fn memory_sparse_initialization() {
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x2001);
    let mixers = default_mixers();
    let commit_fn = |mat: &Mat<F>| l.commit(mat);

    let mem_layout = PlainMemLayout { k: 4, d: 1, n_side: 4 };

    // Sparse init: addr[0]=100, addr[2]=200, others=0
    let mem_init = MemInit::Sparse(vec![(0, F::from_u64(100)), (2, F::from_u64(200))]);

    // Read from initialized addresses
    let mem_trace = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ONE],
        has_write: vec![F::ZERO],
        read_addr: vec![2], // Read initialized value
        write_addr: vec![0],
        read_val: vec![F::from_u64(200)], // Should match init
        write_val: vec![F::ZERO],
        inc_at_write_addr: vec![F::ZERO],
    };

    let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, 0);
    let (mem_inst, mem_wit) = encode_mem_for_twist(
        &params,
        &mem_layout,
        &mem_init,
        &mem_trace,
        &commit_fn,
        Some(ccs.m),
        mcs.m_in,
    );

    let step_bundle = StepWitnessBundle {
        mcs: (mcs, mcs_wit),
        lut_instances: vec![],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData::<K>,
    };

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"mem-sparse-init");
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
    .expect("prove should succeed with sparse initialization");

    let mut tr_verify = Poseidon2Transcript::new(b"mem-sparse-init");
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
    .expect("verify should succeed with sparse initialization");

    println!("✓ memory_sparse_initialization: Sparse init (ROM-like) verified");
}
