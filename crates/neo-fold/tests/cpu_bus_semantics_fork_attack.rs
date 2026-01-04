//! Adversarial regression test: CPU/memory fork via shadow columns.
//!
//! ## Attack Description
//!
//! This test catches the "dead witness" class of bug where the CPU's **semantic** columns for
//! memory/lookup are not constrained to equal the **shared CPU bus** columns that Twist/Shout
//! consume.
//!
//! The attack shape:
//! 1. Build a CPU witness with both "shadow" columns (used internally by CPU constraints) and
//!    "bus" columns (the shared bus region consumed by Twist/Shout).
//! 2. Keep the bus region consistent with the real memory trace (so Twist/Shout pass).
//! 3. Tamper the CPU's internal "shadow" columns to an inconsistent value.
//! 4. Prove + verify.
//!
//! ## Expected Behavior
//!
//! - **Before fix (vulnerable)**: The malicious proof **verifies** → test **fails**, confirming
//!   the vulnerability exists.
//! - **After fix**: Proof generation or verification **fails** → test **passes**.
//!
//! ## Security Invariant Being Tested
//!
//! CPU constraints must treat shared bus variables as the **source of truth** for memory/lookup
//! events. If shadow columns exist for engineering reasons, constraints must enforce:
//!   `shadow_mem == mem_bus` (per event/row)
//!
//! This prevents the "fork" attack where:
//! - CPU proves semantics for world A (shadow values),
//! - sidecars prove consistency for world B (bus values),
//! - verifier incorrectly accepts if CPU never references bus.

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
use neo_memory::plain::{PlainMemLayout, PlainMemTrace};
use neo_memory::witness::{LutInstance, LutWitness, MemInstance, MemWitness, StepInstanceBundle, StepWitnessBundle};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

// ============================================================================
// Test Infrastructure
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

// ============================================================================
// Test: CPU/Memory Semantic Fork Attack
// ============================================================================

/// Test that CPU semantic shadow columns must be constrained to equal bus columns.
///
/// ## Witness Layout (m=16)
///
/// ```text
/// z = [ 1 | x0 | shadow_rv | ... padding ... | bus_cols (7) ]
///       0    1        2                         bus_base=9
/// ```
///
/// Bus layout for Twist (d=1, ell=1, chunk_size=1):
/// - z[9]  = ra_bit (read address bit)
/// - z[10] = wa_bit (write address bit)
/// - z[11] = has_read
/// - z[12] = has_write
/// - z[13] = wv (write value)
/// - z[14] = rv (read value on bus)
/// - z[15] = inc (increment at write addr)
///
/// ## Attack
///
/// 1. CPU constraint (row 0): `shadow_rv == x0` (semantic: register gets memory value)
/// 2. Bus rv (z[14]) is set to correct memory value (0 from zero-init memory)
/// 3. Shadow rv (z[2]) is tampered to 1
/// 4. x0 is set to 1 (to satisfy CPU constraint with tampered shadow)
///
/// If there's NO constraint enforcing `shadow_rv == bus_rv`, the attack succeeds:
/// - CPU sees shadow_rv=1, x0=1 → constraint satisfied
/// - Twist sees bus_rv=0 from zero-init → memory check satisfied
/// - Verifier accepts → VULNERABILITY
///
/// If the system is secure, either:
/// - CCS must include constraints that bind shadow_rv to bus_rv, OR
/// - The verifier must detect the mismatch through other means
#[test]
fn cpu_semantic_shadow_fork_attack_should_be_rejected() {
    // -------------------------------------------------------------------------
    // 1) CCS Layout with Shadow Column
    //
    // We create a CCS where:
    // - z[2] is a "shadow" read value used by CPU semantics
    // - z[14] (bus_rv) is the actual bus column for the read value
    //
    // CPU constraint (row 0): (shadow_rv - x0) * 1 = 0
    // This models: "register x0 gets the value from a memory load"
    //
    // VULNERABILITY CASE: No constraint links shadow_rv to bus_rv
    // -------------------------------------------------------------------------
    let n = 16usize;
    let m = 16usize;
    let m_in = 2usize; // z[0]=1 (const), z[1]=x0 (public)
    let bus_cols = 7usize; // Twist bus: ra_bit, wa_bit, has_read, has_write, wv, rv, inc
    let bus_base = m - bus_cols; // = 9
    let _bus_rv = bus_base + 5; // = 14 (read value on bus) - shown for documentation
    let shadow_rv = 2usize; // Shadow read value column (internal to CPU)

    // Create CCS with CPU semantic constraint:
    // Row 0: (shadow_rv - x0) * 1 = 0  ⟹  shadow_rv == x0
    let mut A = Mat::zero(n, m, F::ZERO);
    let mut B = Mat::zero(n, m, F::ZERO);
    let mut C = Mat::zero(n, m, F::ZERO);

    // Row 0: (z[shadow_rv] - z[1]) * z[0] = 0
    // This is: shadow_rv == x0 (when z[0]=1)
    A[(0, shadow_rv)] = F::ONE;
    A[(0, 1)] = -F::ONE;
    B[(0, 0)] = F::ONE;
    // C[(0, _)] = 0 (already)

    // Row 1: Booleanize x0: x0^2 = x0
    A[(1, 1)] = F::ONE;
    B[(1, 1)] = F::ONE;
    C[(1, 1)] = F::ONE;

    // CRITICAL: In a VULNERABLE system, there is NO constraint linking shadow_rv to bus_rv.
    // In a SECURE system, we would add: Row 2: (shadow_rv - bus_rv) * 1 = 0
    //
    // We intentionally DO NOT add this constraint to test if the system is vulnerable.

    let ccs: CcsStructure<F> = neo_ccs::r1cs_to_ccs(A, B, C);
    assert_eq!(ccs.n, n);
    assert_eq!(ccs.m, m);

    // -------------------------------------------------------------------------
    // 2) Build CPU Witness with Forked Values
    //
    // Attack setup:
    // - Memory is zero-initialized, so a read from addr 0 returns 0
    // - We set bus_rv = 0 (correct value for Twist)
    // - We set shadow_rv = 1 (tampered value)
    // - We set x0 = 1 (to satisfy CPU constraint: shadow_rv == x0)
    //
    // This creates a fork: CPU semantics see rv=1, Twist sees rv=0
    // -------------------------------------------------------------------------
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_committer(&params, m);
    let mixers = default_mixers();

    let mut z_cpu: Vec<F> = vec![F::ZERO; ccs.m];
    z_cpu[0] = F::ONE; // constant 1
    z_cpu[1] = F::ONE; // x0 = 1 (tampered: should be 0 if linked to real memory)
    z_cpu[shadow_rv] = F::ONE; // shadow_rv = 1 (TAMPERED: real value is 0)

    // Fill bus with CORRECT values (to satisfy Twist):
    // [ra_bit, wa_bit, has_read, has_write, wv, rv, inc] at z[bus_base..bus_base+7)
    z_cpu[bus_base + 0] = F::ZERO; // ra_bit (addr 0, bit 0)
    z_cpu[bus_base + 1] = F::ZERO; // wa_bit
    z_cpu[bus_base + 2] = F::ONE; // has_read = 1
    z_cpu[bus_base + 3] = F::ZERO; // has_write = 0
    z_cpu[bus_base + 4] = F::ZERO; // wv = 0
    z_cpu[bus_base + 5] = F::ZERO; // rv = 0 (CORRECT value from zero-init memory)
    z_cpu[bus_base + 6] = F::ZERO; // inc = 0

    let Z_cpu = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z_cpu);
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

    // -------------------------------------------------------------------------
    // 3) Twist Memory Instance
    //
    // Memory is zero-initialized. The read at addr 0 should return 0.
    // This is consistent with bus_rv=0, but inconsistent with shadow_rv=1.
    // -------------------------------------------------------------------------
    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
    let mem_init = MemInit::Zero;

    let mem_trace = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ONE],
        has_write: vec![F::ZERO],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO], // Correct value from zero-init memory
        write_val: vec![F::ZERO],
        inc_at_write_addr: vec![F::ZERO],
    };

    let mem_inst = MemInstance::<Cmt, F> {
        comms: Vec::new(),
        k: mem_layout.k,
        d: mem_layout.d,
        n_side: mem_layout.n_side,
        steps: mem_trace.steps,
        ell: mem_layout.n_side.trailing_zeros() as usize,
        init: mem_init,
        _phantom: PhantomData,
    };
    let mem_wit = MemWitness { mats: Vec::new() };

    let steps_witness = vec![StepWitnessBundle {
        mcs,
        lut_instances: vec![],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData::<K>,
    }];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    // -------------------------------------------------------------------------
    // 4) Prove + Verify
    // -------------------------------------------------------------------------
    let mut tr = Poseidon2Transcript::new(b"cpu-semantic-fork-attack");
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

    // -------------------------------------------------------------------------
    // 5) Security Assertion
    //
    // The system MUST reject this proof because:
    // - CPU semantics claim rv=1 (shadow_rv)
    // - Memory/bus claims rv=0 (bus_rv)
    //
    // If the verifier accepts, there's a CRITICAL VULNERABILITY where CPU
    // semantic columns can diverge from the shared bus without detection.
    // -------------------------------------------------------------------------
    match prove_res {
        Err(e) => {
            // Prover rejected the inconsistent witness - system is secure
            println!("✓ SECURE: Prover rejected CPU semantic fork attack: {:?}", e);
        }
        Ok(proof) => {
            let mut tr_v = Poseidon2Transcript::new(b"cpu-semantic-fork-attack");
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
                ║ CPU SEMANTIC FORK ATTACK SUCCEEDED!                              ║\n\
                ║                                                                  ║\n\
                ║ The verifier accepted a proof where:                             ║\n\
                ║   - CPU shadow column (shadow_rv) = 1                            ║\n\
                ║   - Shared bus column (bus_rv) = 0                               ║\n\
                ║                                                                  ║\n\
                ║ This means CPU constraints are NOT linked to the shared bus.     ║\n\
                ║ An attacker can prove arbitrary CPU semantics while Twist/Shout  ║\n\
                ║ verify a completely different memory/lookup story.               ║\n\
                ║                                                                  ║\n\
                ║ FIX: Add constraints enforcing shadow_rv == bus_rv for all       ║\n\
                ║ memory/lookup events, or eliminate shadow columns entirely.      ║\n\
                ╚══════════════════════════════════════════════════════════════════╝"
            );

            println!(
                "✓ SECURE: Verifier rejected CPU semantic fork attack: {:?}",
                verify_res.unwrap_err()
            );
        }
    }
}

/// Test variant: Splice bus from a valid trace into a fake CPU witness.
///
/// This is the purest "fork" demonstration:
/// - Build `cpu_wit_fake` with CPU-internal values that differ from memory reality
/// - Use the real bus values (correct memory operations)
/// - If no linkage exists, both CPU and Twist verify their separate worlds
#[test]
fn cpu_semantic_fork_splice_attack_should_be_rejected() {
    // Same setup as above, but we make it explicit that we're splicing
    // a "real" bus into a "fake" CPU witness.

    let n = 16usize;
    let m = 16usize;
    let m_in = 2usize;
    let bus_cols = 7usize;
    let bus_base = m - bus_cols;
    let shadow_rv = 2usize;
    let shadow_wv = 3usize; // Also add a shadow write value

    // CPU constraints:
    // Row 0: shadow_rv == x0 (load result goes to x0)
    // Row 1: x0^2 = x0 (booleanize)
    // No linkage to bus_rv!
    let mut A = Mat::zero(n, m, F::ZERO);
    let mut B = Mat::zero(n, m, F::ZERO);
    let mut C = Mat::zero(n, m, F::ZERO);

    A[(0, shadow_rv)] = F::ONE;
    A[(0, 1)] = -F::ONE;
    B[(0, 0)] = F::ONE;

    A[(1, 1)] = F::ONE;
    B[(1, 1)] = F::ONE;
    C[(1, 1)] = F::ONE;

    let ccs: CcsStructure<F> = neo_ccs::r1cs_to_ccs(A, B, C);

    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    params.k_rho = 16;
    let l = setup_ajtai_committer(&params, m);
    let mixers = default_mixers();

    // =========================================================================
    // Build the "REAL" trace: Write 42 to addr 0, then read it back
    // =========================================================================
    let real_write_val = F::from_u64(42);
    let real_read_val = F::from_u64(42); // After write, read returns 42

    // =========================================================================
    // Build "FAKE" CPU witness: Claims to have read value 99
    // =========================================================================
    let fake_read_val = F::from_u64(99);

    let mut z_fake: Vec<F> = vec![F::ZERO; ccs.m];
    z_fake[0] = F::ONE;
    z_fake[1] = fake_read_val; // x0 = 99 (FAKE)
    z_fake[shadow_rv] = fake_read_val; // shadow_rv = 99 (FAKE)
    z_fake[shadow_wv] = real_write_val; // Consistent write value

    // Bus: Splice in the REAL values
    z_fake[bus_base + 0] = F::ZERO; // ra_bit
    z_fake[bus_base + 1] = F::ZERO; // wa_bit
    z_fake[bus_base + 2] = F::ONE; // has_read
    z_fake[bus_base + 3] = F::ZERO; // has_write (we'll just do a read in this step)
    z_fake[bus_base + 4] = F::ZERO; // wv
    z_fake[bus_base + 5] = real_read_val; // rv = 42 (REAL from previous write)
    z_fake[bus_base + 6] = F::ZERO; // inc

    // Wait, for this to work with Twist, we need proper memory state.
    // Let's use sparse init with the pre-written value.
    let mem_init = MemInit::Sparse(vec![(0, real_write_val)]);

    let Z_fake = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z_fake);
    let c_fake = l.commit(&Z_fake);

    let mcs = (
        McsInstance {
            c: c_fake,
            x: z_fake[..m_in].to_vec(),
            m_in,
        },
        McsWitness {
            w: z_fake[m_in..].to_vec(),
            Z: Z_fake,
        },
    );

    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
    let mem_trace = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ONE],
        has_write: vec![F::ZERO],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![real_read_val], // 42 - matches bus and init
        write_val: vec![F::ZERO],
        inc_at_write_addr: vec![F::ZERO],
    };

    let mem_inst = MemInstance::<Cmt, F> {
        comms: Vec::new(),
        k: mem_layout.k,
        d: mem_layout.d,
        n_side: mem_layout.n_side,
        steps: mem_trace.steps,
        ell: mem_layout.n_side.trailing_zeros() as usize,
        init: mem_init,
        _phantom: PhantomData,
    };
    let mem_wit = MemWitness { mats: Vec::new() };

    let steps_witness = vec![StepWitnessBundle {
        mcs,
        lut_instances: vec![],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData::<K>,
    }];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr = Poseidon2Transcript::new(b"cpu-splice-fork-attack");
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
            println!("✓ SECURE: Prover rejected splice fork attack: {:?}", e);
        }
        Ok(proof) => {
            let mut tr_v = Poseidon2Transcript::new(b"cpu-splice-fork-attack");
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
                ║ SPLICE FORK ATTACK SUCCEEDED!                                    ║\n\
                ║                                                                  ║\n\
                ║ The verifier accepted a proof where:                             ║\n\
                ║   - CPU claims load result = 99 (fake)                           ║\n\
                ║   - Memory bus shows load result = 42 (real)                     ║\n\
                ║                                                                  ║\n\
                ║ The attacker spliced a real bus into a fake CPU witness.         ║\n\
                ║ CPU semantics are completely decoupled from memory reality.      ║\n\
                ╚══════════════════════════════════════════════════════════════════╝"
            );

            println!(
                "✓ SECURE: Verifier rejected splice fork attack: {:?}",
                verify_res.unwrap_err()
            );
        }
    }
}

/// Test that lookup (Shout) shadow columns are also properly constrained.
///
/// This tests the same vulnerability pattern but for lookup tables instead of memory.
#[test]
fn cpu_lookup_shadow_fork_attack_should_be_rejected() {
    use neo_memory::plain::{LutTable, PlainLutTrace};

    let n = 20usize;
    let m = 20usize;
    let m_in = 2usize;

    // Shout bus layout for d=1, ell=1 (n_side=2):
    // addr_bits (1) + has_lookup (1) + val (1) = 3 columns
    let shout_bus_cols = 3usize;
    // Twist bus for a minimal memory instance (d=1, ell=1):
    // ra_bits (1) + wa_bits (1) + has_read (1) + has_write (1) + wv (1) + rv (1) + inc (1) = 7
    let twist_bus_cols = 7usize;
    let total_bus_cols = shout_bus_cols + twist_bus_cols;
    let bus_base = m - total_bus_cols;

    // Shout bus positions (appear BEFORE Twist in bus layout):
    let shout_addr_bit = bus_base + 0;
    let shout_has_lookup = bus_base + 1;
    let shout_val = bus_base + 2;

    // Shadow column for lookup result
    let shadow_lookup_val = 2usize;

    // CPU constraint: shadow_lookup_val == x0
    let mut A = Mat::zero(n, m, F::ZERO);
    let mut B = Mat::zero(n, m, F::ZERO);
    let mut C = Mat::zero(n, m, F::ZERO);

    A[(0, shadow_lookup_val)] = F::ONE;
    A[(0, 1)] = -F::ONE;
    B[(0, 0)] = F::ONE;

    A[(1, 1)] = F::ONE;
    B[(1, 1)] = F::ONE;
    C[(1, 1)] = F::ONE;

    // NO constraint linking shadow_lookup_val to shout_val!

    let ccs: CcsStructure<F> = neo_ccs::r1cs_to_ccs(A, B, C);

    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    params.k_rho = 16;
    let l = setup_ajtai_committer(&params, m);
    let mixers = default_mixers();

    // LUT table: table[0] = 100, table[1] = 200
    let real_table_val_at_1 = F::from_u64(200);
    let fake_lookup_result = F::from_u64(999); // Attacker claims lookup returned 999

    let mut z_cpu: Vec<F> = vec![F::ZERO; ccs.m];
    z_cpu[0] = F::ONE;
    z_cpu[1] = fake_lookup_result; // x0 = 999 (FAKE)
    z_cpu[shadow_lookup_val] = fake_lookup_result; // shadow = 999 (FAKE)

    // Shout bus: real lookup at addr 1 returning 200
    z_cpu[shout_addr_bit] = F::ONE; // addr = 1
    z_cpu[shout_has_lookup] = F::ONE;
    z_cpu[shout_val] = real_table_val_at_1; // val = 200 (REAL)

    // Twist bus: minimal valid config (no memory operations)
    let twist_bus_start = bus_base + shout_bus_cols;
    z_cpu[twist_bus_start + 0] = F::ZERO; // ra_bit
    z_cpu[twist_bus_start + 1] = F::ZERO; // wa_bit
    z_cpu[twist_bus_start + 2] = F::ZERO; // has_read = 0
    z_cpu[twist_bus_start + 3] = F::ZERO; // has_write = 0
    z_cpu[twist_bus_start + 4] = F::ZERO; // wv
    z_cpu[twist_bus_start + 5] = F::ZERO; // rv
    z_cpu[twist_bus_start + 6] = F::ZERO; // inc

    let Z_cpu = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z_cpu);
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

    // LUT instance
    let lut_table = LutTable {
        table_id: 0,
        k: 2,
        d: 1,
        n_side: 2,
        content: vec![F::from_u64(100), F::from_u64(200)],
    };
    let lut_trace = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![1],
        val: vec![real_table_val_at_1],
    };

    let lut_inst = LutInstance::<Cmt, F> {
        comms: Vec::new(),
        k: lut_table.k,
        d: lut_table.d,
        n_side: lut_table.n_side,
        steps: lut_trace.has_lookup.len(),
        ell: lut_table.n_side.trailing_zeros() as usize,
        table_spec: None,
        table: lut_table.content.clone(),
        _phantom: PhantomData,
    };
    let lut_wit = LutWitness { mats: Vec::new() };

    // Memory instance (inactive)
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
    let mem_inst = MemInstance::<Cmt, F> {
        comms: Vec::new(),
        k: mem_layout.k,
        d: mem_layout.d,
        n_side: mem_layout.n_side,
        steps: mem_trace.steps,
        ell: mem_layout.n_side.trailing_zeros() as usize,
        init: mem_init,
        _phantom: PhantomData,
    };
    let mem_wit = MemWitness { mats: Vec::new() };

    let steps_witness = vec![StepWitnessBundle {
        mcs,
        lut_instances: vec![(lut_inst, lut_wit)],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData::<K>,
    }];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr = Poseidon2Transcript::new(b"cpu-lookup-fork-attack");
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
            println!("✓ SECURE: Prover rejected lookup fork attack: {:?}", e);
        }
        Ok(proof) => {
            let mut tr_v = Poseidon2Transcript::new(b"cpu-lookup-fork-attack");
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
                ║ LOOKUP SHADOW FORK ATTACK SUCCEEDED!                             ║\n\
                ║                                                                  ║\n\
                ║ The verifier accepted a proof where:                             ║\n\
                ║   - CPU claims lookup result = 999 (fake)                        ║\n\
                ║   - Shout bus shows lookup result = 200 (real)                   ║\n\
                ║                                                                  ║\n\
                ║ Lookup constraints are decoupled from the shared bus.            ║\n\
                ╚══════════════════════════════════════════════════════════════════╝"
            );

            println!(
                "✓ SECURE: Verifier rejected lookup fork attack: {:?}",
                verify_res.unwrap_err()
            );
        }
    }
}
