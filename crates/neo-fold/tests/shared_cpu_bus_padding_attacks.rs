//! Comprehensive tests for "padding row constraints" vulnerabilities.
//!
//! These tests validate the constraints identified by the architecture review:
//!
//! > For Twist:
//! > - For each `ra_bit_i`: `(1 - has_read) · ra_bit_i = 0`
//! > - `rv`: `(1 - has_read) · rv = 0`
//! > - For each `wa_bit_i`: `(1 - has_write) · wa_bit_i = 0`
//! > - `wv`: `(1 - has_write) · wv = 0`
//! > - `inc_at_write_addr`: `(1 - has_write) · inc_at_write_addr = 0`
//! >
//! > For Shout:
//! > - For each `addr_bit_i`: `(1 - has_lookup) · addr_bit_i = 0`
//! > - `val`: `(1 - has_lookup) · val = 0`
//!
//! ## Expected Behavior
//!
//! - **Before fix (vulnerable)**: Tests **FAIL** (red) → vulnerability exists
//! - **After fix**: Tests **PASS** (green) → system is secure

#![allow(non_snake_case)]

use std::marker::PhantomData;
use std::sync::Arc;

use neo_ajtai::{s_lincomb, s_mul, setup as ajtai_setup, AjtaiSModule, Commitment as Cmt};
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{
    fold_shard_prove as fold_shard_prove_shared_cpu_bus, fold_shard_verify as fold_shard_verify_shared_cpu_bus,
    CommitMixers,
};
use neo_math::ring::Rq as RqEl;
use neo_math::{D, F, K};
use neo_memory::cpu::build_bus_layout_for_instances;
use neo_memory::cpu::constraints::{CpuColumnLayout, CpuConstraintBuilder};
use neo_memory::plain::{LutTable, PlainLutTrace, PlainMemLayout, PlainMemTrace};
use neo_memory::witness::{LutInstance, LutWitness, MemInstance, MemWitness, StepInstanceBundle, StepWitnessBundle};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

// ============================================================================
// Test Infrastructure (same as comprehensive attacks)
// ============================================================================

fn setup_ajtai_committer(params: &NeoParams, m: usize) -> AjtaiSModule {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, m).expect("Ajtai setup should succeed");
    AjtaiSModule::new(Arc::new(pp))
}

fn rot_matrix_to_rq(mat: &Mat<F>) -> RqEl {
    use neo_math::ring::cf_inv;

    debug_assert_eq!(mat.rows(), D);
    debug_assert_eq!(mat.cols(), D);

    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}

fn default_mixers() -> CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt> {
    fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Cmt]) -> Cmt {
        assert!(!cs.is_empty(), "mix_rhos_commits: empty commitments");
        if cs.len() == 1 {
            return cs[0].clone();
        }
        let rq_els: Vec<RqEl> = rhos.iter().map(rot_matrix_to_rq).collect();
        s_lincomb(&rq_els, cs).expect("s_lincomb should succeed")
    }
    fn combine_b_pows(cs: &[Cmt], b: u32) -> Cmt {
        assert!(!cs.is_empty(), "combine_b_pows: empty commitments");
        let mut acc = cs[0].clone();
        let mut pow = F::from_u64(b as u64);
        for i in 1..cs.len() {
            let rq_pow = RqEl::from_field_scalar(pow);
            let term = s_mul(&rq_pow, &cs[i]);
            acc.add_inplace(&term);
            pow *= F::from_u64(b as u64);
        }
        acc
    }
    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

fn metadata_only_mem_instance(
    layout: &PlainMemLayout,
    init: MemInit<F>,
    steps: usize,
) -> (MemInstance<Cmt, F>, MemWitness<F>) {
    let ell = layout.n_side.trailing_zeros() as usize;
    (
        MemInstance {
            comms: Vec::new(),
            k: layout.k,
            d: layout.d,
            n_side: layout.n_side,
            steps,
            ell,
            init,
            _phantom: PhantomData,
        },
        MemWitness { mats: Vec::new() },
    )
}

fn metadata_only_lut_instance(table: &LutTable<F>, steps: usize) -> (LutInstance<Cmt, F>, LutWitness<F>) {
    let ell = table.n_side.trailing_zeros() as usize;
    (
        LutInstance {
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
        LutWitness { mats: Vec::new() },
    )
}

const TWIST_BUS_COLS: usize = 7;
const SHOUT_BUS_COLS: usize = 3;

/// Witness column layout for our test CPU.
///
/// Layout: [const_one, is_load, is_store, rd_value, rs2_value, effective_addr, is_lookup, lookup_key, lookup_out, ...bus...]
const COL_CONST_ONE: usize = 0;
const COL_IS_LOAD: usize = 1;
const COL_IS_STORE: usize = 2;
const COL_RD_VALUE: usize = 3;
const COL_RS2_VALUE: usize = 4;
const COL_EFFECTIVE_ADDR: usize = 5;
const COL_IS_LOOKUP: usize = 6;
const COL_LOOKUP_KEY: usize = 7;
const COL_LOOKUP_OUT: usize = 8;

fn create_cpu_layout() -> CpuColumnLayout {
    CpuColumnLayout {
        is_load: COL_IS_LOAD,
        is_store: COL_IS_STORE,
        effective_addr: COL_EFFECTIVE_ADDR,
        rd_write_value: COL_RD_VALUE,
        rs2_value: COL_RS2_VALUE,
        is_lookup: COL_IS_LOOKUP,
        lookup_key: COL_LOOKUP_KEY,
        lookup_output: COL_LOOKUP_OUT,
    }
}

/// Create a CCS that enforces the required Twist padding constraints.
fn create_ccs_referencing_all_twist_bus_cols(n: usize, m: usize, m_in: usize, bus_base: usize) -> CcsStructure<F> {
    let cpu_layout = create_cpu_layout();
    let bus = build_bus_layout_for_instances(m, m_in, 1, [], [1]).expect("bus layout");
    assert_eq!(bus.bus_base, bus_base, "test assumes canonical bus_base");

    let mut builder = CpuConstraintBuilder::<F>::new(n, m, COL_CONST_ONE);
    builder.add_twist_instance(&bus, &bus.twist_cols[0], &cpu_layout);

    builder
        .build()
        .expect("should build CCS with Twist constraints")
}

/// Create a CCS referencing all required Shout + Twist bus columns
fn create_ccs_referencing_all_shout_twist_bus_cols(
    n: usize,
    m: usize,
    m_in: usize,
    bus_base: usize,
    shout_cols: usize,
) -> CcsStructure<F> {
    assert_eq!(shout_cols, SHOUT_BUS_COLS, "test assumes fixed shout_cols");

    let cpu_layout = create_cpu_layout();
    let bus = build_bus_layout_for_instances(m, m_in, 1, [1], [1]).expect("bus layout");
    assert_eq!(bus.bus_base, bus_base, "test assumes canonical bus_base");

    let mut builder = CpuConstraintBuilder::<F>::new(n, m, COL_CONST_ONE);
    builder.add_shout_instance(&bus, &bus.shout_cols[0], &cpu_layout);
    builder.add_twist_instance(&bus, &bus.twist_cols[0], &cpu_layout);

    builder
        .build()
        .expect("should build CCS with Shout+Twist constraints")
}

// ============================================================================
// Padding Attack Tests: Twist
// ============================================================================

/// Test: has_write=0 but wv≠0 should be rejected
///
/// Constraint needed: `(1 - has_write) · wv = 0`
#[test]
fn has_write_flag_mismatch_wv_nonzero_should_be_rejected() {
    let n = 16usize;
    let m = 16usize;
    let m_in = 1usize;
    let bus_cols = TWIST_BUS_COLS;
    let bus_base = m - bus_cols;

    let ccs = create_ccs_referencing_all_twist_bus_cols(n, m, m_in, bus_base);

    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    params.k_rho = 16;
    let l = setup_ajtai_committer(&params, m);
    let mixers = default_mixers();

    let mem_init = MemInit::Zero;
    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };

    // ATTACK: has_write=0 but wv=99 (non-zero)
    let mut z = vec![F::ZERO; m];
    z[0] = F::ONE;
    z[bus_base + 0] = F::ZERO; // ra_bit
    z[bus_base + 1] = F::ZERO; // wa_bit
    z[bus_base + 2] = F::ZERO; // has_read = 0
    z[bus_base + 3] = F::ZERO; // has_write = 0
    z[bus_base + 4] = F::from_u64(99); // wv = 99 (SHOULD BE 0)
    z[bus_base + 5] = F::ZERO; // rv
    z[bus_base + 6] = F::ZERO; // inc

    let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z);
    let c = l.commit(&Z);

    let mcs = (
        McsInstance {
            c,
            x: z[..m_in].to_vec(),
            m_in,
        },
        McsWitness {
            w: z[m_in..].to_vec(),
            Z,
        },
    );

    let mem_trace = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ZERO],
        has_write: vec![F::ZERO],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO],
        write_val: vec![F::from_u64(99)], // Matching the attack
        inc_at_write_addr: vec![F::ZERO],
    };

    let (mem_inst, mem_wit) = metadata_only_mem_instance(&mem_layout, mem_init, mem_trace.steps);

    let steps_witness = vec![StepWitnessBundle {
        mcs,
        lut_instances: vec![],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData::<K>,
    }];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr = Poseidon2Transcript::new(b"has-write-wv-mismatch");
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

    match prove_res {
        Err(e) => {
            println!("✓ SECURE: Prover rejected has_write/wv mismatch: {:?}", e);
        }
        Ok(proof) => {
            let mut tr_v = Poseidon2Transcript::new(b"has-write-wv-mismatch");
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
                "\n\
                ╔══════════════════════════════════════════════════════════════════╗\n\
                ║                    CRITICAL VULNERABILITY                        ║\n\
                ╠══════════════════════════════════════════════════════════════════╣\n\
                ║ HAS_WRITE FLAG MISMATCH ATTACK SUCCEEDED!                        ║\n\
                ║                                                                  ║\n\
                ║ has_write=0 but wv=99 (non-zero).                                ║\n\
                ║                                                                  ║\n\
                ║ FIX: Add constraint: (1 - has_write) · wv = 0                    ║\n\
                ╚══════════════════════════════════════════════════════════════════╝"
            );
        }
    }
}

/// Test: has_write=0 but inc_at_write_addr≠0 should be rejected
///
/// Constraint needed: `(1 - has_write) · inc_at_write_addr = 0`
#[test]
fn has_write_flag_mismatch_inc_nonzero_should_be_rejected() {
    let n = 16usize;
    let m = 16usize;
    let m_in = 1usize;
    let bus_cols = TWIST_BUS_COLS;
    let bus_base = m - bus_cols;

    let ccs = create_ccs_referencing_all_twist_bus_cols(n, m, m_in, bus_base);

    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    params.k_rho = 16;
    let l = setup_ajtai_committer(&params, m);
    let mixers = default_mixers();

    let mem_init = MemInit::Zero;
    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };

    // ATTACK: has_write=0 but inc=50 (non-zero)
    let mut z = vec![F::ZERO; m];
    z[0] = F::ONE;
    z[bus_base + 0] = F::ZERO; // ra_bit
    z[bus_base + 1] = F::ZERO; // wa_bit
    z[bus_base + 2] = F::ZERO; // has_read = 0
    z[bus_base + 3] = F::ZERO; // has_write = 0
    z[bus_base + 4] = F::ZERO; // wv = 0
    z[bus_base + 5] = F::ZERO; // rv
    z[bus_base + 6] = F::from_u64(50); // inc = 50 (SHOULD BE 0)

    let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z);
    let c = l.commit(&Z);

    let mcs = (
        McsInstance {
            c,
            x: z[..m_in].to_vec(),
            m_in,
        },
        McsWitness {
            w: z[m_in..].to_vec(),
            Z,
        },
    );

    let mem_trace = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ZERO],
        has_write: vec![F::ZERO],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO],
        write_val: vec![F::ZERO],
        inc_at_write_addr: vec![F::from_u64(50)], // Attack
    };

    let (mem_inst, mem_wit) = metadata_only_mem_instance(&mem_layout, mem_init, mem_trace.steps);

    let steps_witness = vec![StepWitnessBundle {
        mcs,
        lut_instances: vec![],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData::<K>,
    }];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr = Poseidon2Transcript::new(b"has-write-inc-mismatch");
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

    match prove_res {
        Err(e) => {
            println!("✓ SECURE: Prover rejected has_write/inc mismatch: {:?}", e);
        }
        Ok(proof) => {
            let mut tr_v = Poseidon2Transcript::new(b"has-write-inc-mismatch");
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
                "\n\
                ╔══════════════════════════════════════════════════════════════════╗\n\
                ║                    CRITICAL VULNERABILITY                        ║\n\
                ╠══════════════════════════════════════════════════════════════════╣\n\
                ║ HAS_WRITE/INC MISMATCH ATTACK SUCCEEDED!                         ║\n\
                ║                                                                  ║\n\
                ║ has_write=0 but inc_at_write_addr=50 (non-zero).                 ║\n\
                ║                                                                  ║\n\
                ║ FIX: Add constraint: (1 - has_write) · inc_at_write_addr = 0    ║\n\
                ╚══════════════════════════════════════════════════════════════════╝"
            );
        }
    }
}

/// Test: has_read=0 but ra_bits≠0 should be rejected
///
/// Constraint needed: `(1 - has_read) · ra_bit = 0` for each bit
#[test]
fn has_read_flag_mismatch_ra_bits_nonzero_should_be_rejected() {
    let n = 16usize;
    let m = 16usize;
    let m_in = 1usize;
    let bus_cols = TWIST_BUS_COLS;
    let bus_base = m - bus_cols;

    let ccs = create_ccs_referencing_all_twist_bus_cols(n, m, m_in, bus_base);

    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    params.k_rho = 16;
    let l = setup_ajtai_committer(&params, m);
    let mixers = default_mixers();

    let mem_init = MemInit::Zero;
    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };

    // ATTACK: has_read=0 but ra_bit=1 (non-zero address bits when no read)
    let mut z = vec![F::ZERO; m];
    z[0] = F::ONE;
    z[bus_base + 0] = F::ONE; // ra_bit = 1 (SHOULD BE 0 when has_read=0)
    z[bus_base + 1] = F::ZERO; // wa_bit
    z[bus_base + 2] = F::ZERO; // has_read = 0
    z[bus_base + 3] = F::ZERO; // has_write = 0
    z[bus_base + 4] = F::ZERO; // wv
    z[bus_base + 5] = F::ZERO; // rv
    z[bus_base + 6] = F::ZERO; // inc

    let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z);
    let c = l.commit(&Z);

    let mcs = (
        McsInstance {
            c,
            x: z[..m_in].to_vec(),
            m_in,
        },
        McsWitness {
            w: z[m_in..].to_vec(),
            Z,
        },
    );

    let mem_trace = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ZERO],
        has_write: vec![F::ZERO],
        read_addr: vec![1], // Non-zero address even though no read
        write_addr: vec![0],
        read_val: vec![F::ZERO],
        write_val: vec![F::ZERO],
        inc_at_write_addr: vec![F::ZERO],
    };

    let (mem_inst, mem_wit) = metadata_only_mem_instance(&mem_layout, mem_init, mem_trace.steps);

    let steps_witness = vec![StepWitnessBundle {
        mcs,
        lut_instances: vec![],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData::<K>,
    }];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr = Poseidon2Transcript::new(b"has-read-ra-bits-mismatch");
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

    match prove_res {
        Err(e) => {
            println!("✓ SECURE: Prover rejected has_read/ra_bits mismatch: {:?}", e);
        }
        Ok(proof) => {
            let mut tr_v = Poseidon2Transcript::new(b"has-read-ra-bits-mismatch");
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
                "\n\
                ╔══════════════════════════════════════════════════════════════════╗\n\
                ║                    CRITICAL VULNERABILITY                        ║\n\
                ╠══════════════════════════════════════════════════════════════════╣\n\
                ║ RA_BITS PADDING ATTACK SUCCEEDED!                                ║\n\
                ║                                                                  ║\n\
                ║ has_read=0 but ra_bit=1 (non-zero address bits).                 ║\n\
                ║                                                                  ║\n\
                ║ FIX: Add constraint: (1 - has_read) · ra_bit_i = 0               ║\n\
                ╚══════════════════════════════════════════════════════════════════╝"
            );
        }
    }
}

/// Test: has_write=0 but wa_bits≠0 should be rejected
///
/// Constraint needed: `(1 - has_write) · wa_bit = 0` for each bit
#[test]
fn has_write_flag_mismatch_wa_bits_nonzero_should_be_rejected() {
    let n = 16usize;
    let m = 16usize;
    let m_in = 1usize;
    let bus_cols = TWIST_BUS_COLS;
    let bus_base = m - bus_cols;

    let ccs = create_ccs_referencing_all_twist_bus_cols(n, m, m_in, bus_base);

    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    params.k_rho = 16;
    let l = setup_ajtai_committer(&params, m);
    let mixers = default_mixers();

    let mem_init = MemInit::Zero;
    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };

    // ATTACK: has_write=0 but wa_bit=1 (non-zero address bits when no write)
    let mut z = vec![F::ZERO; m];
    z[0] = F::ONE;
    z[bus_base + 0] = F::ZERO; // ra_bit
    z[bus_base + 1] = F::ONE; // wa_bit = 1 (SHOULD BE 0 when has_write=0)
    z[bus_base + 2] = F::ZERO; // has_read = 0
    z[bus_base + 3] = F::ZERO; // has_write = 0
    z[bus_base + 4] = F::ZERO; // wv
    z[bus_base + 5] = F::ZERO; // rv
    z[bus_base + 6] = F::ZERO; // inc

    let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z);
    let c = l.commit(&Z);

    let mcs = (
        McsInstance {
            c,
            x: z[..m_in].to_vec(),
            m_in,
        },
        McsWitness {
            w: z[m_in..].to_vec(),
            Z,
        },
    );

    let mem_trace = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ZERO],
        has_write: vec![F::ZERO],
        read_addr: vec![0],
        write_addr: vec![1], // Non-zero address even though no write
        read_val: vec![F::ZERO],
        write_val: vec![F::ZERO],
        inc_at_write_addr: vec![F::ZERO],
    };

    let (mem_inst, mem_wit) = metadata_only_mem_instance(&mem_layout, mem_init, mem_trace.steps);

    let steps_witness = vec![StepWitnessBundle {
        mcs,
        lut_instances: vec![],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData::<K>,
    }];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr = Poseidon2Transcript::new(b"has-write-wa-bits-mismatch");
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

    match prove_res {
        Err(e) => {
            println!("✓ SECURE: Prover rejected has_write/wa_bits mismatch: {:?}", e);
        }
        Ok(proof) => {
            let mut tr_v = Poseidon2Transcript::new(b"has-write-wa-bits-mismatch");
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
                "\n\
                ╔══════════════════════════════════════════════════════════════════╗\n\
                ║                    CRITICAL VULNERABILITY                        ║\n\
                ╠══════════════════════════════════════════════════════════════════╣\n\
                ║ WA_BITS PADDING ATTACK SUCCEEDED!                                ║\n\
                ║                                                                  ║\n\
                ║ has_write=0 but wa_bit=1 (non-zero address bits).                ║\n\
                ║                                                                  ║\n\
                ║ FIX: Add constraint: (1 - has_write) · wa_bit_i = 0              ║\n\
                ╚══════════════════════════════════════════════════════════════════╝"
            );
        }
    }
}

// ============================================================================
// Padding Attack Tests: Shout (Lookup)
// ============================================================================

/// Test: has_lookup=0 but val≠0 should be rejected
///
/// Constraint needed: `(1 - has_lookup) · val = 0`
#[test]
fn has_lookup_flag_mismatch_val_nonzero_should_be_rejected() {
    // Must be large enough to hold all injected Shout+Twist constraints (incl. bitness checks).
    let n = 22usize;
    let m = 22usize;
    let m_in = 1usize;

    let shout_cols = SHOUT_BUS_COLS;
    let total_bus_cols = shout_cols + TWIST_BUS_COLS;
    let bus_base = m - total_bus_cols;

    let ccs = create_ccs_referencing_all_shout_twist_bus_cols(n, m, m_in, bus_base, shout_cols);

    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    params.k_rho = 16;
    let l = setup_ajtai_committer(&params, m);
    let mixers = default_mixers();

    let lut_table = LutTable {
        table_id: 0,
        k: 2,
        d: 1,
        n_side: 2,
        content: vec![F::from_u64(100), F::from_u64(200)],
    };

    // ATTACK: has_lookup=0 but val=999 (non-zero)
    let mut z = vec![F::ZERO; m];
    z[0] = F::ONE;

    // Shout bus
    z[bus_base + 0] = F::ZERO; // addr_bit = 0
    z[bus_base + 1] = F::ZERO; // has_lookup = 0
    z[bus_base + 2] = F::from_u64(999); // val = 999 (SHOULD BE 0)

    // Twist bus (inactive)
    let twist_base = bus_base + shout_cols;
    z[twist_base + 2] = F::ZERO; // has_read = 0
    z[twist_base + 3] = F::ZERO; // has_write = 0

    let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z);
    let c = l.commit(&Z);

    let mcs = (
        McsInstance {
            c,
            x: z[..m_in].to_vec(),
            m_in,
        },
        McsWitness {
            w: z[m_in..].to_vec(),
            Z,
        },
    );

    let lut_trace = PlainLutTrace {
        has_lookup: vec![F::ZERO],
        addr: vec![0],
        val: vec![F::from_u64(999)], // Non-zero despite no lookup
    };

    let (lut_inst, lut_wit) = metadata_only_lut_instance(&lut_table, lut_trace.has_lookup.len());

    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
    let mem_init = MemInit::Zero;
    let mem_trace = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ZERO],
        has_write: vec![F::ZERO],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO],
        write_val: vec![F::ZERO],
        inc_at_write_addr: vec![F::ZERO],
    };
    let (mem_inst, mem_wit) = metadata_only_mem_instance(&mem_layout, mem_init, mem_trace.steps);

    let steps_witness = vec![StepWitnessBundle {
        mcs,
        lut_instances: vec![(lut_inst, lut_wit)],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData::<K>,
    }];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr = Poseidon2Transcript::new(b"has-lookup-val-mismatch");
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

    match prove_res {
        Err(e) => {
            println!("✓ SECURE: Prover rejected has_lookup/val mismatch: {:?}", e);
        }
        Ok(proof) => {
            let mut tr_v = Poseidon2Transcript::new(b"has-lookup-val-mismatch");
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
                "\n\
                ╔══════════════════════════════════════════════════════════════════╗\n\
                ║                    CRITICAL VULNERABILITY                        ║\n\
                ╠══════════════════════════════════════════════════════════════════╣\n\
                ║ HAS_LOOKUP/VAL MISMATCH ATTACK SUCCEEDED!                        ║\n\
                ║                                                                  ║\n\
                ║ has_lookup=0 but val=999 (non-zero).                             ║\n\
                ║                                                                  ║\n\
                ║ FIX: Add constraint: (1 - has_lookup) · val = 0                  ║\n\
                ╚══════════════════════════════════════════════════════════════════╝"
            );
        }
    }
}

/// Test: has_lookup=0 but addr_bits≠0 should be rejected
///
/// Constraint needed: `(1 - has_lookup) · addr_bit = 0` for each bit
#[test]
fn has_lookup_flag_mismatch_addr_bits_nonzero_should_be_rejected() {
    // Must be large enough to hold all injected Shout+Twist constraints (incl. bitness checks).
    let n = 22usize;
    let m = 22usize;
    let m_in = 1usize;

    let shout_cols = SHOUT_BUS_COLS;
    let total_bus_cols = shout_cols + TWIST_BUS_COLS;
    let bus_base = m - total_bus_cols;

    let ccs = create_ccs_referencing_all_shout_twist_bus_cols(n, m, m_in, bus_base, shout_cols);

    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    params.k_rho = 16;
    let l = setup_ajtai_committer(&params, m);
    let mixers = default_mixers();

    let lut_table = LutTable {
        table_id: 0,
        k: 2,
        d: 1,
        n_side: 2,
        content: vec![F::from_u64(100), F::from_u64(200)],
    };

    // ATTACK: has_lookup=0 but addr_bit=1 (non-zero address when no lookup)
    let mut z = vec![F::ZERO; m];
    z[0] = F::ONE;

    // Shout bus
    z[bus_base + 0] = F::ONE; // addr_bit = 1 (SHOULD BE 0)
    z[bus_base + 1] = F::ZERO; // has_lookup = 0
    z[bus_base + 2] = F::ZERO; // val = 0

    // Twist bus (inactive)
    let twist_base = bus_base + shout_cols;
    z[twist_base + 2] = F::ZERO; // has_read = 0
    z[twist_base + 3] = F::ZERO; // has_write = 0

    let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z);
    let c = l.commit(&Z);

    let mcs = (
        McsInstance {
            c,
            x: z[..m_in].to_vec(),
            m_in,
        },
        McsWitness {
            w: z[m_in..].to_vec(),
            Z,
        },
    );

    let lut_trace = PlainLutTrace {
        has_lookup: vec![F::ZERO],
        addr: vec![1], // Non-zero address despite no lookup
        val: vec![F::ZERO],
    };

    let (lut_inst, lut_wit) = metadata_only_lut_instance(&lut_table, lut_trace.has_lookup.len());

    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
    let mem_init = MemInit::Zero;
    let mem_trace = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ZERO],
        has_write: vec![F::ZERO],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO],
        write_val: vec![F::ZERO],
        inc_at_write_addr: vec![F::ZERO],
    };
    let (mem_inst, mem_wit) = metadata_only_mem_instance(&mem_layout, mem_init, mem_trace.steps);

    let steps_witness = vec![StepWitnessBundle {
        mcs,
        lut_instances: vec![(lut_inst, lut_wit)],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData::<K>,
    }];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr = Poseidon2Transcript::new(b"has-lookup-addr-bits-mismatch");
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

    match prove_res {
        Err(e) => {
            println!("✓ SECURE: Prover rejected has_lookup/addr_bits mismatch: {:?}", e);
        }
        Ok(proof) => {
            let mut tr_v = Poseidon2Transcript::new(b"has-lookup-addr-bits-mismatch");
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
                "\n\
                ╔══════════════════════════════════════════════════════════════════╗\n\
                ║                    CRITICAL VULNERABILITY                        ║\n\
                ╠══════════════════════════════════════════════════════════════════╣\n\
                ║ ADDR_BITS PADDING ATTACK SUCCEEDED!                              ║\n\
                ║                                                                  ║\n\
                ║ has_lookup=0 but addr_bit=1 (non-zero address bits).             ║\n\
                ║                                                                  ║\n\
                ║ FIX: Add constraint: (1 - has_lookup) · addr_bit_i = 0           ║\n\
                ╚══════════════════════════════════════════════════════════════════╝"
            );
        }
    }
}
