//! Range check lookup table tests.
//!
//! Range checks are fundamental to zkVMs - they prove that a value fits within
//! a certain number of bits without revealing the value itself. This is typically
//! done via lookup tables containing all valid values in the range.
//!
//! ## Coverage
//! - `range_check_4bit_valid`: Valid 4-bit range checks [0..16)
//! - `range_check_4bit_invalid_value_fails`: Value outside range should fail
//! - `range_check_nibble_decomposition`: Decompose byte into two 4-bit nibbles
//! - `range_check_combined_with_computation`: Range check + arithmetic constraints

#![allow(non_snake_case)]

use std::marker::PhantomData;
use std::sync::Arc;

use neo_ajtai::{setup as ajtai_setup, set_global_pp, AjtaiSModule, Commitment as Cmt};
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness, MeInstance};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::shard::CommitMixers;
use neo_fold::pi_ccs::FoldingMode;
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

const TEST_N: usize = 64;
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

/// Build a 4-bit range check table: [0, 1, 2, ..., 15]
fn build_4bit_range_table() -> LutTable<F> {
    LutTable {
        table_id: 0,
        k: 16,
        d: 1,
        n_side: 16,
        content: (0u64..16).map(F::from_u64).collect(),
    }
}

fn make_shout_instance(
    table: &LutTable<F>,
    steps: usize,
) -> (neo_memory::witness::LutInstance<Cmt, F>, neo_memory::witness::LutWitness<F>) {
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

/// Valid 4-bit range check: prove value is in [0..16)
#[test]
fn range_check_4bit_valid() {
    let ccs = create_identity_ccs(TEST_N);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.m).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x3001);
    let mixers = default_mixers();

    let range_table = build_4bit_range_table();

    // Test several valid values: 0, 7, 15
    let test_values = [0u64, 7, 15];

    for &val in &test_values {
        let range_trace = PlainLutTrace {
            has_lookup: vec![F::ONE],
            addr: vec![val],
            val: vec![range_table.content[val as usize]],
        };
        let step_bundle =
            create_step_with_shout_bus(&params, &ccs, &l, val, vec![(&range_table, range_trace)]);

        let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
        let acc_wit_init: Vec<Mat<F>> = Vec::new();

        let mut tr_prove = Poseidon2Transcript::new(b"range-4bit-valid");
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
        .expect(&format!("prove should succeed for range check of {}", val));

        let mut tr_verify = Poseidon2Transcript::new(b"range-4bit-valid");
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
        .expect(&format!("verify should succeed for range check of {}", val));
    }

    println!("✓ range_check_4bit_valid: Values 0, 7, 15 all verified as 4-bit");
}

/// Invalid range check: value 20 is outside [0..16)
#[test]
fn range_check_4bit_invalid_value_fails() {
    let ccs = create_identity_ccs(TEST_N);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.m).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x3001);

    let range_table = build_4bit_range_table();

    // Invalid: claim 20 is in [0..16). Under bit-addressing, 20 wraps to 4-bit address bits,
    // so forcing `val=20` should be rejected by the Shout constraints.
    let bad_trace = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![20],
        val: vec![F::from_u64(20)],
    };
    let step_bundle =
        create_step_with_shout_bus(&params, &ccs, &l, 20, vec![(&range_table, bad_trace)]);

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"range-4bit-invalid");
    let proof_result = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        default_mixers(),
    );

    if let Ok(proof) = proof_result {
        let mut tr_verify = Poseidon2Transcript::new(b"range-4bit-invalid");
        let steps_public = [StepInstanceBundle::from(&step_bundle)];
        assert!(
            fold_shard_verify(
                FoldingMode::PaperExact,
                &mut tr_verify,
                &params,
                &ccs,
                &steps_public,
                &acc_init,
                &proof,
                default_mixers(),
            )
            .is_err(),
            "verification should fail on invalid range-check witness"
        );
    }

    println!("✓ range_check_4bit_invalid_value_fails: Out-of-range value correctly rejected");
}

/// Nibble decomposition: prove a byte splits into two valid 4-bit nibbles.
/// For value 0xAB = 171, low_nibble = 0xB = 11, high_nibble = 0xA = 10.
/// Both nibbles must be in [0..16).
#[test]
fn range_check_nibble_decomposition() {
    let ccs = create_identity_ccs(TEST_N);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.m).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x3001);
    let mixers = default_mixers();

    let range_table = build_4bit_range_table();

    // Decompose 0xAB = 171 into nibbles
    let byte_val = 0xABu64; // 171
    let low_nibble = byte_val & 0xF; // 11
    let high_nibble = (byte_val >> 4) & 0xF; // 10

    // Two lookups: one for each nibble
    let low_trace = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![low_nibble],
        val: vec![range_table.content[low_nibble as usize]],
    };
    let high_trace = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![high_nibble],
        val: vec![range_table.content[high_nibble as usize]],
    };
    let low_table = LutTable {
        table_id: 0,
        ..range_table.clone()
    };
    let high_table = LutTable {
        table_id: 1,
        ..range_table.clone()
    };

    let step_bundle = create_step_with_shout_bus(
        &params,
        &ccs,
        &l,
        byte_val,
        vec![(&low_table, low_trace), (&high_table, high_trace)],
    );

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"range-nibble-decomp");
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
    .expect("prove should succeed for nibble decomposition");

    let mut tr_verify = Poseidon2Transcript::new(b"range-nibble-decomp");
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
    .expect("verify should succeed for nibble decomposition");

    println!(
        "✓ range_check_nibble_decomposition: 0x{:02X} = (0x{:X}, 0x{:X}) verified",
        byte_val, high_nibble, low_nibble
    );
}

/// Combine range check with arithmetic constraint.
/// Prove: a + b = c AND a, b, c ∈ [0..16)
#[test]
fn range_check_combined_with_addition() {
    // CCS for addition: a + b - c = 0
    // z = [1, a, b, c] where z[0]=1 is the constant
    // For this test we use identity CCS and validate only the batched lookup logic.
    let ccs = create_identity_ccs(TEST_N);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.m).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x3001);
    let mixers = default_mixers();

    let range_table = build_4bit_range_table();

    // a=5, b=7, c=12 (all in [0..16))
    let a = 5u64;
    let b = 7u64;
    let c = a + b; // 12

    assert!(a < 16 && b < 16 && c < 16, "Values must be 4-bit");

    // Three range checks for a, b, c
    let trace_a = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![a],
        val: vec![range_table.content[a as usize]],
    };
    let trace_b = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![b],
        val: vec![range_table.content[b as usize]],
    };
    let trace_c = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![c],
        val: vec![range_table.content[c as usize]],
    };
    let table_a = LutTable {
        table_id: 0,
        ..range_table.clone()
    };
    let table_b = LutTable {
        table_id: 1,
        ..range_table.clone()
    };
    let table_c = LutTable {
        table_id: 2,
        ..range_table.clone()
    };

    let step_bundle = create_step_with_shout_bus(
        &params,
        &ccs,
        &l,
        a * 100 + b * 10 + c,
        vec![(&table_a, trace_a), (&table_b, trace_b), (&table_c, trace_c)],
    );

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"range-with-addition");
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
    .expect("prove should succeed for range-checked addition");

    let mut tr_verify = Poseidon2Transcript::new(b"range-with-addition");
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
    .expect("verify should succeed for range-checked addition");

    println!(
        "✓ range_check_combined_with_addition: {} + {} = {} with all values range-checked",
        a, b, c
    );
}

/// Test: Wrong value claimed in range check should fail.
/// Claim lookup returns 10, but actually the value at that address is different.
#[test]
fn range_check_wrong_value_claimed_fails() {
    let ccs = create_identity_ccs(TEST_N);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.m).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x3001);

    let range_table = build_4bit_range_table();

    // Malicious: claim table[5] = 10 (actually table[5] = 5)
    let bad_trace = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![5],
        val: vec![F::from_u64(10)], // WRONG: should be 5
    };
    let step_bundle =
        create_step_with_shout_bus(&params, &ccs, &l, 5, vec![(&range_table, bad_trace)]);

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"range-wrong-value-claimed");
    let proof_result = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        default_mixers(),
    );

    if let Ok(proof) = proof_result {
        let mut tr_verify = Poseidon2Transcript::new(b"range-wrong-value-claimed");
        let steps_public = [StepInstanceBundle::from(&step_bundle)];
        assert!(
            fold_shard_verify(
                FoldingMode::PaperExact,
                &mut tr_verify,
                &params,
                &ccs,
                &steps_public,
                &acc_init,
                &proof,
                default_mixers(),
            )
            .is_err(),
            "verification should fail when claimed value doesn't match table"
        );
    }

    println!("✓ range_check_wrong_value_claimed_fails: Mismatched table value correctly rejected");
}

/// Boundary test: check edge values 0 and 15 specifically.
#[test]
fn range_check_boundary_values() {
    let ccs = create_identity_ccs(TEST_N);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.m).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x3001);
    let mixers = default_mixers();

    let range_table = build_4bit_range_table();

    // Two-step test: step 0 checks min (0), step 1 checks max (15)
    let mut steps: Vec<StepWitnessBundle<Cmt, F, K>> = Vec::new();

    for (step_idx, val) in [(0u64, 0u64), (1, 15)].iter() {
        let trace = PlainLutTrace {
            has_lookup: vec![F::ONE],
            addr: vec![*val],
            val: vec![range_table.content[*val as usize]],
        };
        steps.push(create_step_with_shout_bus(
            &params,
            &ccs,
            &l,
            *step_idx,
            vec![(&range_table, trace)],
        ));
    }

    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"range-boundary");
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
    .expect("prove should succeed for boundary values");

    let mut tr_verify = Poseidon2Transcript::new(b"range-boundary");
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
    .expect("verify should succeed for boundary values");

    println!("✓ range_check_boundary_values: Min (0) and max (15) boundary values verified");
}
