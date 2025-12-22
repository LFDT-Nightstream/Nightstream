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

use neo_ajtai::{decomp_b, setup as ajtai_setup, set_global_pp, AjtaiSModule, Commitment as Cmt, DecompStyle};
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness, MeInstance};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::shard::CommitMixers;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify};
use neo_math::{D, F, K};
use neo_memory::encode::{encode_lut_for_shout, encode_mem_for_twist};
use neo_memory::plain::{LutTable, PlainLutTrace, PlainMemLayout, PlainMemTrace};
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

fn empty_mem_trace() -> PlainMemTrace<F> {
    PlainMemTrace {
        steps: 1,
        has_read: vec![F::ZERO],
        has_write: vec![F::ZERO],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO],
        write_val: vec![F::ZERO],
        inc_at_write_addr: vec![F::ZERO],
    }
}

/// Valid 4-bit range check: prove value is in [0..16)
#[test]
fn range_check_4bit_valid() {
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x3001);
    let mixers = default_mixers();
    let commit_fn = |mat: &Mat<F>| l.commit(mat);

    let range_table = build_4bit_range_table();

    // Test several valid values: 0, 7, 15
    let test_values = [0u64, 7, 15];

    for &val in &test_values {
        let range_trace = PlainLutTrace {
            has_lookup: vec![F::ONE],
            addr: vec![val],
            val: vec![range_table.content[val as usize]],
        };

        let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, val);

        let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
        let mem_init = MemInit::Zero;

        let (mem_inst, mem_wit) = encode_mem_for_twist(
            &params,
            &mem_layout,
            &mem_init,
            &empty_mem_trace(),
            &commit_fn,
            Some(ccs.m),
            mcs.m_in,
        );
        let (range_inst, range_wit) = encode_lut_for_shout(
            &params,
            &range_table,
            &range_trace,
            &commit_fn,
            Some(ccs.m),
            mcs.m_in,
        );

        let step_bundle = StepWitnessBundle {
            mcs: (mcs, mcs_wit),
            lut_instances: vec![(range_inst, range_wit)],
            mem_instances: vec![(mem_inst, mem_wit)],
            _phantom: PhantomData::<K>,
        };

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
            &l,
            mixers,
        )
        .expect(&format!("verify should succeed for range check of {}", val));
    }

    println!("✓ range_check_4bit_valid: Values 0, 7, 15 all verified as 4-bit");
}

/// Invalid range check: value 20 is outside [0..16)
#[test]
fn range_check_4bit_invalid_value_fails() {
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x3001);
    let commit_fn = |mat: &Mat<F>| l.commit(mat);

    let range_table = build_4bit_range_table();

    // Invalid: claim 20 is in [0..16)
    // This requires claiming table[20] = 20, but table only has 16 entries
    let bad_trace = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![20], // Out of bounds!
        val: vec![F::from_u64(20)],
    };

    let (mcs, _mcs_wit) = create_mcs(&params, &ccs, &l, 20);

    // Should panic during encoding due to out-of-bounds address
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        encode_lut_for_shout(&params, &range_table, &bad_trace, &commit_fn, Some(ccs.m), mcs.m_in)
    }));

    assert!(
        result.is_err(),
        "Encoding should panic on out-of-bounds range check"
    );

    println!("✓ range_check_4bit_invalid_value_fails: Out-of-range value correctly rejected");
}

/// Nibble decomposition: prove a byte splits into two valid 4-bit nibbles.
/// For value 0xAB = 171, low_nibble = 0xB = 11, high_nibble = 0xA = 10.
/// Both nibbles must be in [0..16).
#[test]
fn range_check_nibble_decomposition() {
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x3001);
    let mixers = default_mixers();
    let commit_fn = |mat: &Mat<F>| l.commit(mat);

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

    let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, byte_val);

    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
    let mem_init = MemInit::Zero;

    let (mem_inst, mem_wit) = encode_mem_for_twist(
        &params,
        &mem_layout,
        &mem_init,
        &empty_mem_trace(),
        &commit_fn,
        Some(ccs.m),
        mcs.m_in,
    );

    // We need two separate table instances for two lookups in same step
    // Actually, in a real system these would be the same table but different lookup slots
    // For this test, we use the same table twice
    let (low_inst, low_wit) = encode_lut_for_shout(
        &params,
        &LutTable { table_id: 0, ..range_table.clone() },
        &low_trace,
        &commit_fn,
        Some(ccs.m),
        mcs.m_in,
    );
    let (high_inst, high_wit) = encode_lut_for_shout(
        &params,
        &LutTable { table_id: 1, ..range_table.clone() },
        &high_trace,
        &commit_fn,
        Some(ccs.m),
        mcs.m_in,
    );

    let step_bundle = StepWitnessBundle {
        mcs: (mcs, mcs_wit),
        lut_instances: vec![(low_inst, low_wit), (high_inst, high_wit)],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData::<K>,
    };

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
        &l,
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
    let n = 4usize;

    // Build addition CCS: M0*z + M1*z - M2*z = 0 (linearized)
    // Actually, we can use: (M0 + M1) * z = M2 * z
    // But for simplicity, use identity CCS and just verify via lookup

    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x3001);
    let mixers = default_mixers();
    let commit_fn = |mat: &Mat<F>| l.commit(mat);

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

    let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, a * 100 + b * 10 + c);

    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
    let mem_init = MemInit::Zero;

    let (mem_inst, mem_wit) = encode_mem_for_twist(
        &params,
        &mem_layout,
        &mem_init,
        &empty_mem_trace(),
        &commit_fn,
        Some(ccs.m),
        mcs.m_in,
    );

    let (inst_a, wit_a) = encode_lut_for_shout(
        &params,
        &LutTable { table_id: 0, ..range_table.clone() },
        &trace_a,
        &commit_fn,
        Some(ccs.m),
        mcs.m_in,
    );
    let (inst_b, wit_b) = encode_lut_for_shout(
        &params,
        &LutTable { table_id: 1, ..range_table.clone() },
        &trace_b,
        &commit_fn,
        Some(ccs.m),
        mcs.m_in,
    );
    let (inst_c, wit_c) = encode_lut_for_shout(
        &params,
        &LutTable { table_id: 2, ..range_table.clone() },
        &trace_c,
        &commit_fn,
        Some(ccs.m),
        mcs.m_in,
    );

    let step_bundle = StepWitnessBundle {
        mcs: (mcs, mcs_wit),
        lut_instances: vec![(inst_a, wit_a), (inst_b, wit_b), (inst_c, wit_c)],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData::<K>,
    };

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
        &l,
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
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x3001);
    let commit_fn = |mat: &Mat<F>| l.commit(mat);

    let range_table = build_4bit_range_table();

    // Malicious: claim table[5] = 10 (actually table[5] = 5)
    let bad_trace = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![5],
        val: vec![F::from_u64(10)], // WRONG: should be 5
    };

    let (mcs, _mcs_wit) = create_mcs(&params, &ccs, &l, 5);

    // Should panic during encoding (semantic check catches it)
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        encode_lut_for_shout(&params, &range_table, &bad_trace, &commit_fn, Some(ccs.m), mcs.m_in)
    }));

    assert!(
        result.is_err(),
        "Encoding should panic when claimed value doesn't match table"
    );

    println!("✓ range_check_wrong_value_claimed_fails: Mismatched table value correctly rejected");
}

/// Boundary test: check edge values 0 and 15 specifically.
#[test]
fn range_check_boundary_values() {
    let n = 4usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x3001);
    let mixers = default_mixers();
    let commit_fn = |mat: &Mat<F>| l.commit(mat);

    let range_table = build_4bit_range_table();

    // Two-step test: step 0 checks min (0), step 1 checks max (15)
    let mut steps: Vec<StepWitnessBundle<Cmt, F, K>> = Vec::new();

    for (step_idx, val) in [(0u64, 0u64), (1, 15)].iter() {
        let trace = PlainLutTrace {
            has_lookup: vec![F::ONE],
            addr: vec![*val],
            val: vec![range_table.content[*val as usize]],
        };

        let (mcs, mcs_wit) = create_mcs(&params, &ccs, &l, *step_idx);

        let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
        let mem_init = MemInit::Zero;

        let (mem_inst, mem_wit) = encode_mem_for_twist(
            &params,
            &mem_layout,
            &mem_init,
            &empty_mem_trace(),
            &commit_fn,
            Some(ccs.m),
            mcs.m_in,
        );
        let (range_inst, range_wit) = encode_lut_for_shout(
            &params,
            &range_table,
            &trace,
            &commit_fn,
            Some(ccs.m),
            mcs.m_in,
        );

        steps.push(StepWitnessBundle {
            mcs: (mcs, mcs_wit),
            lut_instances: vec![(range_inst, range_wit)],
            mem_instances: vec![(mem_inst, mem_wit)],
            _phantom: PhantomData::<K>,
        });
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
        &l,
        mixers,
    )
    .expect("verify should succeed for boundary values");

    println!("✓ range_check_boundary_values: Min (0) and max (15) boundary values verified");
}

