//! Multi-table Shout (lookup) tests.
//!
//! These tests validate the folding pipeline when multiple lookup tables are used
//! simultaneously, which is essential for real VMs that need:
//! - Bytecode table (instruction decoding)
//! - Range check tables (8-bit, 16-bit ranges)
//! - Bitwise operation tables (AND, XOR, etc.)
//!
//! ## Coverage
//! - `multi_table_shout_two_tables`: Basic test with two independent tables
//! - `multi_table_shout_three_tables_interleaved`: Three tables with interleaved lookups
//! - `multi_table_wrong_table_value_fails`: Adversarial: wrong value for a table
//! - `multi_table_optional_lookups`: `has_lookup=0` rows are ignored

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
use neo_memory::plain::{LutTable, PlainLutTrace};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
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
    let _ = set_global_pp(pp.clone()); // Ignore if already set
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

fn make_shout_instance(
    table: &LutTable<F>,
    steps: usize,
) -> (
    neo_memory::witness::LutInstance<Cmt, F>,
    neo_memory::witness::LutWitness<F>,
) {
    let ell = table.n_side.trailing_zeros() as usize;
    (
        neo_memory::witness::LutInstance {
            comms: Vec::new(),
            k: table.k,
            d: table.d,
            n_side: table.n_side,
            steps,
            ell,
            table_spec: None,
            table: table.content.clone(),
            _phantom: PhantomData,
        },
        neo_memory::witness::LutWitness { mats: Vec::new() },
    )
}

fn write_shout_bus_step(
    z: &mut [F],
    bus_base: usize,
    chunk_size: usize,
    j: usize,
    inst: &neo_memory::witness::LutInstance<Cmt, F>,
    trace: &PlainLutTrace<F>,
    col_id: &mut usize,
) {
    debug_assert_eq!(chunk_size, 1);
    debug_assert_eq!(j, 0);

    let has_lookup = trace.has_lookup[j];
    if has_lookup == F::ONE {
        let mut tmp = trace.addr[j];
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
    z[bus_base + *col_id * chunk_size + j] = has_lookup;
    *col_id += 1;
    z[bus_base + *col_id * chunk_size + j] = if has_lookup == F::ONE { trace.val[j] } else { F::ZERO };
    *col_id += 1;
}

fn create_step_with_shout_bus(
    params: &NeoParams,
    ccs: &CcsStructure<F>,
    l: &AjtaiSModule,
    tag: u64,
    luts: Vec<(&LutTable<F>, PlainLutTrace<F>)>,
) -> StepWitnessBundle<Cmt, F, K> {
    let chunk_size = 1usize;
    let mut bus_cols_total = 0usize;
    let mut lut_instances = Vec::with_capacity(luts.len());
    let mut traces = Vec::with_capacity(luts.len());

    for (table, trace) in luts {
        let steps = trace.has_lookup.len();
        assert_eq!(steps, chunk_size);
        assert_eq!(trace.addr.len(), chunk_size);
        assert_eq!(trace.val.len(), chunk_size);
        let (inst, wit) = make_shout_instance(table, steps);
        bus_cols_total += inst.d * inst.ell + 2;
        lut_instances.push((inst, wit));
        traces.push(trace);
    }

    let bus_base = ccs.m - bus_cols_total * chunk_size;
    let mut z = vec![F::ZERO; ccs.m];
    z[0] = F::from_u64(tag);

    let mut col_id = 0usize;
    for ((inst, _wit), trace) in lut_instances.iter().zip(traces.iter()) {
        write_shout_bus_step(&mut z, bus_base, chunk_size, 0, inst, trace, &mut col_id);
    }
    debug_assert_eq!(col_id, bus_cols_total);

    let (mcs, mcs_wit) = create_mcs_from_z(params, l, M_IN, z);
    StepWitnessBundle {
        mcs: (mcs, mcs_wit),
        lut_instances,
        mem_instances: vec![],
        _phantom: PhantomData::<K>,
    }
}

/// Two lookup tables used in the same step:
/// - Table 0 (opcode): [ADD=1, MUL=2, SUB=3, DIV=4]
/// - Table 1 (range8): [0, 1, 2, ..., 15] (small range for test)
#[test]
fn multi_table_shout_two_tables() {
    let ccs = create_identity_ccs(TEST_N);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.m).expect("params");
    params.k_rho = 16;

    // Setup real Ajtai commitments
    let l = setup_ajtai_pp(ccs.m, 0x1001);
    let mixers = default_mixers();

    // Table 0: Opcode table (4 entries)
    let opcode_table = LutTable {
        table_id: 0,
        k: 4,
        d: 1,
        n_side: 4,
        content: vec![
            F::from_u64(1), // ADD
            F::from_u64(2), // MUL
            F::from_u64(3), // SUB
            F::from_u64(4), // DIV
        ],
    };

    // Table 1: Range check table (4 entries for simplicity)
    let range_table = LutTable {
        table_id: 1,
        k: 4,
        d: 1,
        n_side: 4,
        content: vec![F::from_u64(0), F::from_u64(1), F::from_u64(2), F::from_u64(3)],
    };

    // Step 0: Lookup opcode[1]=MUL=2, range[2]=2
    let opcode_trace0 = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![1],
        val: vec![opcode_table.content[1]], // MUL = 2
    };
    let range_trace0 = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![2],
        val: vec![range_table.content[2]], // 2
    };
    let step_bundle = create_step_with_shout_bus(
        &params,
        &ccs,
        &l,
        42,
        vec![(&opcode_table, opcode_trace0), (&range_table, range_trace0)],
    );

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"multi-table-two");
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
    .expect("prove should succeed with two tables");

    let mut tr_verify = Poseidon2Transcript::new(b"multi-table-two");
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
    .expect("verify should succeed with two tables");

    println!("✓ multi_table_shout_two_tables: Two independent tables verified successfully");
}

/// Three lookup tables with interleaved lookups across multiple steps:
/// - Table 0: Opcodes
/// - Table 1: Range check (8-bit)
/// - Table 2: Bitwise AND results for nibbles
#[test]
fn multi_table_shout_three_tables_interleaved() {
    let ccs = create_identity_ccs(TEST_N);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.m).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x1002);
    let mixers = default_mixers();

    // Table 0: Opcodes
    let opcode_table = LutTable {
        table_id: 0,
        k: 4,
        d: 1,
        n_side: 4,
        content: vec![F::from_u64(10), F::from_u64(20), F::from_u64(30), F::from_u64(40)],
    };

    // Table 1: Small range [0..4)
    let range_table = LutTable {
        table_id: 1,
        k: 4,
        d: 1,
        n_side: 4,
        content: vec![F::ZERO, F::ONE, F::from_u64(2), F::from_u64(3)],
    };

    // Table 2: Some precomputed values (e.g., AND results)
    let bitwise_table = LutTable {
        table_id: 2,
        k: 4,
        d: 1,
        n_side: 4,
        content: vec![F::from_u64(100), F::from_u64(101), F::from_u64(102), F::from_u64(103)],
    };

    // Create 2 steps with different lookup patterns
    let mut steps: Vec<StepWitnessBundle<Cmt, F, K>> = Vec::new();

    for step_idx in 0..2u64 {
        // Step 0: lookup opcode[0], range[1], bitwise[2]
        // Step 1: lookup opcode[3], range[0], bitwise[1]
        let opcode_addr = if step_idx == 0 { 0 } else { 3 };
        let range_addr = if step_idx == 0 { 1 } else { 0 };
        let bitwise_addr = if step_idx == 0 { 2 } else { 1 };

        let opcode_trace = PlainLutTrace {
            has_lookup: vec![F::ONE],
            addr: vec![opcode_addr],
            val: vec![opcode_table.content[opcode_addr as usize]],
        };
        let range_trace = PlainLutTrace {
            has_lookup: vec![F::ONE],
            addr: vec![range_addr],
            val: vec![range_table.content[range_addr as usize]],
        };
        let bitwise_trace = PlainLutTrace {
            has_lookup: vec![F::ONE],
            addr: vec![bitwise_addr],
            val: vec![bitwise_table.content[bitwise_addr as usize]],
        };

        steps.push(create_step_with_shout_bus(
            &params,
            &ccs,
            &l,
            step_idx * 1000,
            vec![
                (&opcode_table, opcode_trace),
                (&range_table, range_trace),
                (&bitwise_table, bitwise_trace),
            ],
        ));
    }

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"multi-table-three");
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
    .expect("prove should succeed with three tables");

    let mut tr_verify = Poseidon2Transcript::new(b"multi-table-three");
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
    .expect("verify should succeed with three tables");

    println!("✓ multi_table_shout_three_tables_interleaved: Three tables with 2 steps verified");
}

/// Adversarial test: Claim a value from the wrong table.
/// Prover claims table[0][1] = X, but uses the value from table[1][1] instead.
#[test]
fn multi_table_wrong_table_value_fails() {
    let ccs = create_identity_ccs(TEST_N);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.m).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x1003);
    let mixers = default_mixers();

    // Table 0: [100, 200, 300, 400]
    let table0 = LutTable {
        table_id: 0,
        k: 4,
        d: 1,
        n_side: 4,
        content: vec![F::from_u64(100), F::from_u64(200), F::from_u64(300), F::from_u64(400)],
    };

    // Table 1: [1, 2, 3, 4]
    let table1 = LutTable {
        table_id: 1,
        k: 4,
        d: 1,
        n_side: 4,
        content: vec![F::ONE, F::from_u64(2), F::from_u64(3), F::from_u64(4)],
    };

    // Malicious: Lookup table0[1] but claim value from table1[1] = 2
    let bad_trace0 = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![1],
        val: vec![table1.content[1]], // WRONG: should be table0.content[1] = 200
    };

    // Valid lookup from table1
    let good_trace1 = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![1],
        val: vec![table1.content[1]], // Correct: 2
    };
    let step_bundle = create_step_with_shout_bus(
        &params,
        &ccs,
        &l,
        999,
        vec![(&table0, bad_trace0), (&table1, good_trace1)],
    );

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"multi-table-wrong");
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
        Err(_) => {
            println!("✓ multi_table_wrong_table_value_fails: Proving correctly failed on invalid lookup");
        }
        Ok(proof) => {
            let mut tr_verify = Poseidon2Transcript::new(b"multi-table-wrong");
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
                "Verification should fail when lookup uses wrong table's value"
            );
            println!("✓ multi_table_wrong_table_value_fails: Verification correctly rejected invalid lookup");
        }
    }
}

/// Test: Optional lookups (has_lookup=0) should not affect table consistency.
#[test]
fn multi_table_optional_lookups() {
    let ccs = create_identity_ccs(TEST_N);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.m).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x1004);
    let mixers = default_mixers();

    let table0 = LutTable {
        table_id: 0,
        k: 4,
        d: 1,
        n_side: 4,
        content: vec![F::from_u64(10), F::from_u64(20), F::from_u64(30), F::from_u64(40)],
    };

    let table1 = LutTable {
        table_id: 1,
        k: 4,
        d: 1,
        n_side: 4,
        content: vec![F::from_u64(1), F::from_u64(2), F::from_u64(3), F::from_u64(4)],
    };

    // Step 0: lookup from table0 only, skip table1
    let trace0_t0 = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![2],
        val: vec![table0.content[2]], // 30
    };
    let trace0_t1 = PlainLutTrace {
        has_lookup: vec![F::ZERO], // No lookup
        addr: vec![0],
        val: vec![F::ZERO],
    };

    // Step 1: lookup from table1 only, skip table0
    let trace1_t0 = PlainLutTrace {
        has_lookup: vec![F::ZERO], // No lookup
        addr: vec![0],
        val: vec![F::ZERO],
    };
    let trace1_t1 = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![3],
        val: vec![table1.content[3]], // 4
    };

    let mut steps: Vec<StepWitnessBundle<Cmt, F, K>> = Vec::new();

    for (step_idx, (t0_trace, t1_trace)) in [(trace0_t0, trace0_t1), (trace1_t0, trace1_t1)]
        .into_iter()
        .enumerate()
    {
        steps.push(create_step_with_shout_bus(
            &params,
            &ccs,
            &l,
            step_idx as u64,
            vec![(&table0, t0_trace), (&table1, t1_trace)],
        ));
    }

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"multi-table-optional");
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
    .expect("prove should succeed with optional lookups");

    let mut tr_verify = Poseidon2Transcript::new(b"multi-table-optional");
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
    .expect("verify should succeed with optional lookups");

    println!("✓ multi_table_optional_lookups: Optional (skipped) lookups verified correctly");
}
