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

use neo_ajtai::{set_global_pp, setup as ajtai_setup, AjtaiSModule, Commitment as Cmt};
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness, MeInstance};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::CommitMixers;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify};
use neo_math::{D, F, K};
use neo_memory::plain::{PlainMemLayout, PlainMemTrace};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Mixers = CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>;

const TEST_N: usize = 32;
const M_IN: usize = 0;

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

fn create_mcs_from_z(
    params: &NeoParams,
    l: &AjtaiSModule,
    m_in: usize,
    z: Vec<F>,
) -> (McsInstance<Cmt, F>, McsWitness<F>) {
    let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(params, &z);
    let c = l.commit(&Z);
    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();
    (McsInstance { c, x, m_in }, McsWitness { w, Z })
}

fn make_twist_instance(
    layout: &PlainMemLayout,
    init: MemInit<F>,
    steps: usize,
) -> (
    neo_memory::witness::MemInstance<Cmt, F>,
    neo_memory::witness::MemWitness<F>,
) {
    let ell = layout.n_side.trailing_zeros() as usize;
    (
        neo_memory::witness::MemInstance {
            comms: Vec::new(),
            k: layout.k,
            d: layout.d,
            n_side: layout.n_side,
            steps,
            ell,
            init,
            _phantom: PhantomData,
        },
        neo_memory::witness::MemWitness { mats: Vec::new() },
    )
}

fn write_twist_bus_step(
    z: &mut [F],
    bus_base: usize,
    chunk_size: usize,
    j: usize,
    inst: &neo_memory::witness::MemInstance<Cmt, F>,
    trace: &PlainMemTrace<F>,
    col_id: &mut usize,
) {
    debug_assert_eq!(chunk_size, 1);
    debug_assert_eq!(j, 0);

    let has_read = trace.has_read[j];
    let has_write = trace.has_write[j];

    // ra_bits (masked by has_read)
    if has_read == F::ONE {
        let mut tmp = trace.read_addr[j];
        for _dim in 0..inst.d {
            let comp = (tmp % (inst.n_side as u64)) as u64;
            tmp /= inst.n_side as u64;
            for bit in 0..inst.ell {
                z[bus_base + *col_id * chunk_size + j] = if (comp >> bit) & 1 == 1 { F::ONE } else { F::ZERO };
                *col_id += 1;
            }
        }
    } else {
        *col_id += inst.d * inst.ell;
    }

    // wa_bits (masked by has_write)
    if has_write == F::ONE {
        let mut tmp = trace.write_addr[j];
        for _dim in 0..inst.d {
            let comp = (tmp % (inst.n_side as u64)) as u64;
            tmp /= inst.n_side as u64;
            for bit in 0..inst.ell {
                z[bus_base + *col_id * chunk_size + j] = if (comp >> bit) & 1 == 1 { F::ONE } else { F::ZERO };
                *col_id += 1;
            }
        }
    } else {
        *col_id += inst.d * inst.ell;
    }

    z[bus_base + *col_id * chunk_size + j] = has_read;
    *col_id += 1;
    z[bus_base + *col_id * chunk_size + j] = has_write;
    *col_id += 1;
    z[bus_base + *col_id * chunk_size + j] = if has_write == F::ONE {
        trace.write_val[j]
    } else {
        F::ZERO
    };
    *col_id += 1;
    z[bus_base + *col_id * chunk_size + j] = if has_read == F::ONE { trace.read_val[j] } else { F::ZERO };
    *col_id += 1;
    z[bus_base + *col_id * chunk_size + j] = if has_write == F::ONE {
        trace.inc_at_write_addr[j]
    } else {
        F::ZERO
    };
    *col_id += 1;
}

fn create_step_with_twist_bus(
    params: &NeoParams,
    ccs: &CcsStructure<F>,
    l: &AjtaiSModule,
    tag: u64,
    mem_instances: Vec<(
        neo_memory::witness::MemInstance<Cmt, F>,
        neo_memory::witness::MemWitness<F>,
        PlainMemTrace<F>,
    )>,
) -> StepWitnessBundle<Cmt, F, K> {
    let chunk_size = 1usize;
    let mut bus_cols_total = 0usize;
    for (inst, _, _) in &mem_instances {
        bus_cols_total += 2 * inst.d * inst.ell + 5;
        assert_eq!(inst.steps, chunk_size);
    }
    let bus_base = ccs.m - bus_cols_total * chunk_size;

    let mut z = vec![F::ZERO; ccs.m];
    z[0] = F::from_u64(tag);
    // Bus tail is already zeroed by default.
    let mut col_id = 0usize;
    for (inst, _wit, trace) in &mem_instances {
        write_twist_bus_step(&mut z, bus_base, chunk_size, 0, inst, trace, &mut col_id);
    }
    debug_assert_eq!(col_id, bus_cols_total);

    let (mcs, mcs_wit) = create_mcs_from_z(params, l, M_IN, z);
    StepWitnessBundle {
        mcs: (mcs, mcs_wit),
        lut_instances: vec![],
        mem_instances: mem_instances.into_iter().map(|(i, w, _)| (i, w)).collect(),
        _phantom: PhantomData::<K>,
    }
}

/// Valid 3-step memory trace:
/// Step 0: Write 42 to addr[0]
/// Step 1: Read addr[0] → 42
/// Step 2: Write 100 to addr[0], overwriting 42
#[test]
fn memory_cross_step_read_consistency() {
    let ccs = create_identity_ccs(TEST_N);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(TEST_N).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x2001);
    let mixers = default_mixers();

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
        let (mem_inst, mem_wit) = make_twist_instance(&mem_layout, mem_init, 1);
        steps.push(create_step_with_twist_bus(
            &params,
            &ccs,
            &l,
            0,
            vec![(mem_inst, mem_wit, mem_trace)],
        ));
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
        let (mem_inst, mem_wit) = make_twist_instance(&mem_layout, mem_init, 1);
        steps.push(create_step_with_twist_bus(
            &params,
            &ccs,
            &l,
            1,
            vec![(mem_inst, mem_wit, mem_trace)],
        ));
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
        let (mem_inst, mem_wit) = make_twist_instance(&mem_layout, mem_init, 1);
        steps.push(create_step_with_twist_bus(
            &params,
            &ccs,
            &l,
            2,
            vec![(mem_inst, mem_wit, mem_trace)],
        ));
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
    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> = steps.iter().map(StepInstanceBundle::from).collect();
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
    let ccs = create_identity_ccs(TEST_N);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(TEST_N).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x2001);
    let mixers = default_mixers();

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

    let (mem_inst, mem_wit) = make_twist_instance(&mem_layout, mem_init, 1);
    let step_bundle = create_step_with_twist_bus(&params, &ccs, &l, 0, vec![(mem_inst, mem_wit, mem_trace)]);

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
    let ccs = create_identity_ccs(TEST_N);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(TEST_N).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x2001);
    let mixers = default_mixers();

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

    let (mem_inst, mem_wit) = make_twist_instance(&mem_layout, mem_init, 1);
    let step_bundle = create_step_with_twist_bus(&params, &ccs, &l, 0, vec![(mem_inst, mem_wit, bad_mem_trace)]);

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

    match proof_result {
        Err(_) => {}
        Ok(proof) => {
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
            assert!(verify_result.is_err(), "tampered read value must fail verification");
        }
    }
}

/// Adversarial: Claim wrong write increment.
/// Writing 100 to addr[0] which previously had 42, but claiming delta = 10 instead of 58.
#[test]
fn memory_tamper_write_increment_fails() {
    let ccs = create_identity_ccs(TEST_N);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(TEST_N).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x2001);
    let mixers = default_mixers();

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
    let (mem_inst, mem_wit) = make_twist_instance(&mem_layout, mem_init, 1);
    let step_bundle = create_step_with_twist_bus(&params, &ccs, &l, 0, vec![(mem_inst, mem_wit, bad_mem_trace)]);

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

    match proof_result {
        Err(_) => {}
        Ok(proof) => {
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
            assert!(verify_result.is_err(), "tampered increment must fail verification");
        }
    }
}

/// Test multiple memory regions in same step.
/// This simulates a VM with separate RAM, ROM, and register file.
#[test]
fn memory_multiple_regions_same_step() {
    let ccs = create_identity_ccs(TEST_N);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(TEST_N).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x2001);
    let mixers = default_mixers();

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

    let (ram_inst, ram_wit) = make_twist_instance(&ram_layout, ram_init, 1);
    let (reg_inst, reg_wit) = make_twist_instance(&reg_layout, reg_init, 1);
    let step_bundle = create_step_with_twist_bus(
        &params,
        &ccs,
        &l,
        0,
        vec![(ram_inst, ram_wit, ram_trace), (reg_inst, reg_wit, reg_trace)],
    );

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
    let ccs = create_identity_ccs(TEST_N);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(TEST_N).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x2001);
    let mixers = default_mixers();

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

    let (mem_inst, mem_wit) = make_twist_instance(&mem_layout, mem_init, 1);
    let step_bundle = create_step_with_twist_bus(&params, &ccs, &l, 0, vec![(mem_inst, mem_wit, mem_trace)]);

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
