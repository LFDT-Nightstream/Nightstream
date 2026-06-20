//! Bellpepper SHA-256 → R1CS-F' → real IVC recursion via `R1csChainBuilder`.
//!
//! One broad test pins the load-bearing IVC contract: two distinct
//! SHA-256 proofs (different preimages, same R1CS shape) are chained
//! through R1CS-F' so step 1 embeds and verifies the NIFS fold authority
//! produced by step 0. Mirrors the Fibonacci recursive test
//! ([`fibonacci_chain_builder_appends_recursive_step_under_tiny_params`])
//! in shape and assertion depth; replaces the Fibonacci app-step
//! transition with one Bellpepper-synthesized SHA-256 per step.
//!
//! Runs under a test-only smaller `Params` profile (`kappa = 2,
//! m = 2^15, lambda = 40`) so prove + extend fit under the 5-minute
//! test cap. The Goldilocks ring + Π_RLC / Π_DEC algebraic identities
//! are unchanged; only the Ajtai-SIS security parameter is reduced,
//! which is the right knob for an algebraic-correctness fixture.
//!
//! Why R1CS-F', not direct-CCS: this is real IVC recursion — each
//! recursive step's encoded image embeds and verifies the prior NIFS
//! fold authority, threads chain state (`z_i`, `acc_digest`,
//! `public_trace`, counters), and commits to the per-step SHA digest
//! through the carried semantic-state digest absorbed by `state_x_out`.
//! Direct-CCS aggregates K independent CCS
//! claims into one accumulator without per-step state advance or
//! cross-step verification — a different protocol contract.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use ::bellpepper::gadgets::boolean::{AllocatedBit, Boolean};
use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError};
use ff::Field;
use neo_ccs::Mat;
use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::bellpepper::{synthesize_to_ccs, BellpepperGoldilocks};
use neo_fold_clean::frontends::direct_ccs::ajtai as direct_ajtai;
use neo_fold_clean::frontends::f_prime::image::{FPrimeImageLayout, NifsCeClaimShape, NifsPayloadShape};
use neo_fold_clean::frontends::f_prime::recursive_plan::{
    build_recursive_step_image_config, build_semantic_state_preimage_fields, state_x_out_preimage_sources,
    AccumulatorPlanOptions, RecursiveStepImagePlan, StateXOutPlanOptions,
};
use neo_fold_clean::frontends::r1cs_f_prime::{self, R1csChainBuilder, R1csCompilerError};
use neo_fold_clean::lifecycle;
use neo_fold_clean::paper::construction2::{ProofState, FINAL_FOLD_TRANSCRIPT_LABEL};
use neo_fold_clean::paper::digest::{pi_ccs_instance_digest_parent_authority, structure_digest, AccumulatorHandle};
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::relations::{CcsClaim, CcsWitness, CeClaim, Structure};
use neo_math::F;
use neo_params::{goldilocks_paper_b2, NeoParams};
use neo_reductions::optimized_engine::{
    optimized_replay_trace_with_cache_instance_digest_and_me_input_handle_and_perf, OptimizedStructureCache,
};
use neo_transcript::{Poseidon2Transcript, Transcript as _};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// One phase of the IVC test, timed and logged via `--nocapture`.
fn phase<R>(label: &str, f: impl FnOnce() -> R) -> R {
    let t = Instant::now();
    let out = f();
    eprintln!("[sha-ivc] {label:<32} {:>7.2}s", t.elapsed().as_secs_f64());
    out
}

fn expect_f_prime_non_replay_unsupported(err: lifecycle::Error, chunk_count: u64) {
    assert!(
        matches!(err, lifecycle::Error::FPrimeNonReplayUnsupported { chunk_count: got } if got == chunk_count),
        "expected FPrimeNonReplayUnsupported({chunk_count}), got {err:?}"
    );
}

const SHA256_AJTAI_SEED: u64 = 0x5348_4132_3556_5345;
const TERMINAL_REPLAY_FIXTURE_VERSION: u32 = 1;

/// Local perf fixture for replaying the terminal Π_CCS prover input.
///
/// This is a developer cache, not protocol authority. The file is written
/// under `target/perf-fixtures`, validated on load by recomputing the
/// structure digest and shape, and regenerated if stale.
#[derive(Serialize, Deserialize)]
struct TerminalOptimizedProveFixture {
    header: TerminalOptimizedProveFixtureHeader,
    params: NeoParams,
    structure: Structure,
    mcs_list: Vec<CcsClaim>,
    mcs_witnesses: Vec<CcsWitness>,
    me_inputs: Vec<CeClaim>,
    me_witnesses: Vec<Mat<F>>,
    public_instance_digest: [F; 4],
    me_input_accumulator_handle: [F; 4],
}

#[derive(Serialize, Deserialize)]
struct TerminalOptimizedProveFixtureHeader {
    version: u32,
    ajtai_seed: u64,
    structure_n: usize,
    structure_m: usize,
    structure_t: usize,
    structure_digest: [F; 4],
    params_kappa: u32,
    params_d: u32,
    params_m: u64,
    params_lambda: u32,
    mcs_count: usize,
    me_count: usize,
}

#[derive(Clone, Debug)]
struct Sha256Circuit {
    preimage: Vec<u8>,
}

impl Circuit<BellpepperGoldilocks> for Sha256Circuit {
    fn synthesize<CS: ConstraintSystem<BellpepperGoldilocks>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let bit_values = ::bellpepper::gadgets::multipack::bytes_to_bits(&self.preimage)
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>();
        let preimage_bits = bit_values
            .into_iter()
            .enumerate()
            .map(|(idx, bit)| AllocatedBit::alloc(cs.namespace(|| format!("preimage_bit_{idx}")), bit))
            .map(|bit| bit.map(Boolean::from))
            .collect::<Result<Vec<_>, _>>()?;

        let hash_bits = ::bellpepper::gadgets::sha256::sha256(cs.namespace(|| "sha256"), &preimage_bits)?;
        for (bit_idx, bit) in hash_bits.iter().enumerate() {
            let value = bit
                .get_value()
                .ok_or(SynthesisError::AssignmentMissing)
                .map(|bit| {
                    if bit {
                        BellpepperGoldilocks::ONE
                    } else {
                        BellpepperGoldilocks::ZERO
                    }
                })?;
            let input = cs.alloc_input(|| format!("hash_out_bit_{bit_idx}"), || Ok(value))?;
            cs.enforce(
                || format!("hash_out_bit_match_{bit_idx}"),
                |_| bit.lc(CS::one(), BellpepperGoldilocks::ONE),
                |lc| lc + CS::one(),
                |lc| lc + input,
            );
        }
        Ok(())
    }
}

fn expected_sha256_public_inputs(preimage: &[u8]) -> Vec<F> {
    let digest = Sha256::digest(preimage);
    let digest_bits = ::bellpepper::gadgets::multipack::bytes_to_bits(&digest);
    let mut out = Vec::with_capacity(1 + digest_bits.len());
    out.push(F::ONE);
    out.extend(
        digest_bits
            .into_iter()
            .map(|bit| if bit { F::ONE } else { F::ZERO }),
    );
    out
}

/// Test-only `Params` profile. Reuses the production Goldilocks ring
/// (Q, ETA, D, B_BASE, K_RHO, T, EXTENSION_DEGREE); only `kappa`, `m`,
/// `lambda` are shrunk so the lifecycle fits under the 5-minute cap.
/// Every Π_RLC / Π_DEC algebraic identity holds bit-for-bit at this
/// profile — only the Ajtai-SIS security parameter is reduced.
fn sha256_tiny_neo_params() -> NeoParams {
    NeoParams::new(
        goldilocks_paper_b2::Q,
        goldilocks_paper_b2::ETA as u32,
        goldilocks_paper_b2::D as u32,
        /* kappa  */ 2,
        /* m      */ 1u64 << 15,
        goldilocks_paper_b2::B_BASE,
        goldilocks_paper_b2::K_RHO,
        goldilocks_paper_b2::T,
        goldilocks_paper_b2::EXTENSION_DEGREE,
        /* lambda */ 40,
    )
    .expect("tiny NeoParams must satisfy the Π_RLC guard")
}

fn sha256_tiny_params() -> Params {
    Params::test_only_from_neo_params(sha256_tiny_neo_params())
}

/// Legacy SHA plan with the empirically-discovered CE shape under
/// [`sha256_tiny_params`] for the full-64-bit app-private layout.
///
/// `c_data_entries = KAPPA * D = 108` and `child_count = K_RHO = 14`
/// are params-derived. The typed app-private helper below derives its
/// row/column challenge lengths from the generated F' structure, since
/// those lengths move when app variables occupy fewer than 64 bits.
fn sha256_lifecycle_plan_with_ce_shape(
    m: usize,
    m_in: usize,
    c_data_entries: usize,
    child_count: u64,
    r_len: usize,
    s_col_len: usize,
) -> RecursiveStepImagePlan {
    let limbs = m * POSEIDON2_GOLDILOCKS_BITS + 1;
    let ce_shape = NifsCeClaimShape {
        c_data_entries,
        x_rows: 54,
        x_active_cols: 5,
        r_len,
        y_ring_inner_lens: vec![64; 8],
        y_zcol_len: 64,
        s_col_len,
    };
    let probe_plan = RecursiveStepImagePlan {
        limbs,
        app_private_var_widths: Vec::new(),
        app_private_widths_are_range_constraints: false,
        boundary_bits: 4 * POSEIDON2_GOLDILOCKS_BITS,
        kmul_count: 0,
        ring_action_pair_count: 0,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
        ),
        sponge_transcript_permutes: 0,
        nifs_payload_shapes: vec![NifsPayloadShape::CeClaim(ce_shape)],
        accumulator: Some(AccumulatorPlanOptions {
            ce_claim_payload_index: 0,
            c_data_entries,
            child_count,
            unified: true,
        }),
        state_x_out: None,
    };
    let probe_layout = FPrimeImageLayout::new(build_recursive_step_image_config(&probe_plan));
    let public_x_out_lane_bit_starts: [usize; 4] =
        std::array::from_fn(|i| probe_layout.boundary.offset + i * POSEIDON2_GOLDILOCKS_BITS);
    let mut plan = probe_plan;
    plan.state_x_out = Some(StateXOutPlanOptions {
        pc: 1,
        public_x_out_lane_bit_starts,
        app_public_input_var_indices: Vec::new(),
        app_public_input_bit_var_indices: (0..m_in).collect(),
        semantic_state_in_var_indices: Vec::new(),
        semantic_state_out_var_indices: Vec::new(),
        initial_semantic_state_digest_anchor: None,
    });
    plan
}

fn sha256_tiny_lifecycle_plan(m: usize, m_in: usize) -> RecursiveStepImagePlan {
    const TINY_C_DATA_ENTRIES: usize = 108;
    const TINY_CHILD_COUNT: u64 = 14;
    const TINY_R_LEN: usize = 16;
    const TINY_S_COL_LEN: usize = 22;

    sha256_lifecycle_plan_with_ce_shape(
        m,
        m_in,
        TINY_C_DATA_ENTRIES,
        TINY_CHILD_COUNT,
        TINY_R_LEN,
        TINY_S_COL_LEN,
    )
}

fn challenge_len_for_domain(size: usize) -> usize {
    size.next_power_of_two().max(2).trailing_zeros() as usize
}

fn sha256_tiny_lifecycle_plan_for_r1cs(r1cs: &r1cs_f_prime::SparseR1cs) -> RecursiveStepImagePlan {
    let shape = r1cs_f_prime::R1csShape::from(r1cs);
    let mut widths = shape.conservative_app_private_var_widths();
    for index in 0..shape.m_in() {
        widths[index] = 1;
    }
    let typed_bits: usize = widths.iter().sum();
    let mut plan = sha256_tiny_lifecycle_plan(shape.m(), shape.m_in());
    plan.limbs = typed_bits + 1;
    plan.app_private_var_widths = widths;
    let layout = FPrimeImageLayout::new(build_recursive_step_image_config(&plan));
    let (structure, _) = r1cs_f_prime::build_r1cs_f_prime_structure(layout, &shape);
    let acc = plan
        .accumulator
        .as_ref()
        .expect("tiny SHA plan has accumulator");
    let NifsPayloadShape::CeClaim(ce_shape) = &mut plan.nifs_payload_shapes[acc.ce_claim_payload_index] else {
        panic!("tiny SHA accumulator payload must be a CE claim");
    };
    ce_shape.r_len = challenge_len_for_domain(structure.ccs.n);
    ce_shape.s_col_len = challenge_len_for_domain(structure.ccs.m);
    plan
}

fn sha256_production_core_lifecycle_plan_for_r1cs(
    r1cs: &r1cs_f_prime::SparseR1cs,
) -> (
    RecursiveStepImagePlan,
    neo_fold_clean::frontends::f_prime::structure::FPrimeStructure,
) {
    sha256_lifecycle_plan_for_r1cs_with_params(r1cs, &Params::production())
}

fn sha256_lifecycle_plan_for_r1cs_with_params(
    r1cs: &r1cs_f_prime::SparseR1cs,
    params: &Params,
) -> (
    RecursiveStepImagePlan,
    neo_fold_clean::frontends::f_prime::structure::FPrimeStructure,
) {
    let shape = r1cs_f_prime::R1csShape::from(r1cs);
    let mut widths = shape.conservative_app_private_var_widths();
    for index in 0..shape.m_in() {
        widths[index] = 1;
    }
    let typed_bits: usize = widths.iter().sum();
    let c_data_entries = params.kappa() as usize * params.d() as usize;
    let child_count = params.k_rho() as u64;
    let mut r_len = 1;
    let mut s_col_len = 1;

    for _ in 0..8 {
        let mut plan =
            sha256_lifecycle_plan_with_ce_shape(shape.m(), shape.m_in(), c_data_entries, child_count, r_len, s_col_len);
        plan.limbs = typed_bits + 1;
        plan.app_private_var_widths = widths.clone();
        let layout = FPrimeImageLayout::new(build_recursive_step_image_config(&plan));
        let (structure, _) = r1cs_f_prime::build_r1cs_f_prime_structure(layout, &shape);
        let next_r_len = challenge_len_for_domain(structure.ccs.n);
        let next_s_col_len = challenge_len_for_domain(structure.ccs.m);
        if next_r_len == r_len && next_s_col_len == s_col_len {
            return (plan, structure);
        }
        r_len = next_r_len;
        s_col_len = next_s_col_len;
    }

    panic!("SHA-256 production-core R1CS-F' CE shape did not converge")
}

fn sha256_b3_probe_params() -> Params {
    Params::test_only_from_neo_params(
        NeoParams::new(
            goldilocks_paper_b2::Q,
            goldilocks_paper_b2::ETA as u32,
            goldilocks_paper_b2::D as u32,
            goldilocks_paper_b2::KAPPA,
            goldilocks_paper_b2::M,
            /* b */ 3,
            /* k_rho */ 8,
            goldilocks_paper_b2::T,
            goldilocks_paper_b2::EXTENSION_DEGREE,
            /* lambda */ 107,
        )
        .expect("b=3 probe params must satisfy the Π_RLC guard"),
    )
}

fn sha256_b2_k12_probe_params() -> Params {
    Params::test_only_from_neo_params(
        NeoParams::new(
            goldilocks_paper_b2::Q,
            goldilocks_paper_b2::ETA as u32,
            goldilocks_paper_b2::D as u32,
            goldilocks_paper_b2::KAPPA,
            goldilocks_paper_b2::M,
            /* b */ 2,
            /* k_rho */ 12,
            goldilocks_paper_b2::T,
            goldilocks_paper_b2::EXTENSION_DEGREE,
            /* lambda */ 107,
        )
        .expect("b=2,k_rho=12 probe params must satisfy the Π_RLC guard"),
    )
}

#[test]
fn sha256_semantic_state_packs_public_input_bits_static_layout() {
    let artifact = synthesize_to_ccs(Sha256Circuit {
        preimage: b"abc".to_vec(),
    })
    .expect("synthesize SHA-256(abc)");
    assert_eq!(
        artifact.shape.inputs, 257,
        "SHA fixture exposes one constant input plus 256 digest bits"
    );

    let m = artifact.shape.inputs + artifact.shape.aux;
    let plan = sha256_tiny_lifecycle_plan(m, artifact.shape.inputs);
    let config = build_recursive_step_image_config(&plan);
    assert_eq!(
        config.poseidon_one_shot_preimage_lens.len(),
        2,
        "tiny SHA canonical plan carries semantic-output and state_x_out one-shot traces"
    );

    let semantic_prefix_lanes = build_semantic_state_preimage_fields(&[]).len();
    let chain_prefix_lanes = state_x_out_preimage_sources(1).len();
    let packed_public_lanes = artifact.shape.inputs.div_ceil(POSEIDON2_GOLDILOCKS_BITS);
    let full_public_lanes = artifact.shape.inputs;
    let packed_semantic_lanes = config.poseidon_one_shot_preimage_lens[0];
    let state_x_out_lanes = config.poseidon_one_shot_preimage_lens[1];

    assert_eq!(
        packed_semantic_lanes,
        semantic_prefix_lanes + packed_public_lanes,
        "SHA public inputs should be packed into 64-bit semantic-state lanes"
    );
    assert_eq!(
        state_x_out_lanes, chain_prefix_lanes,
        "state_x_out should absorb the semantic digest, not append SHA public bits directly"
    );
    assert!(
        packed_semantic_lanes < semantic_prefix_lanes + full_public_lanes,
        "packed layout must be strictly narrower than one hash lane per public bit"
    );
    eprintln!(
        "[sha-layout] semantic lanes: packed={} vs old_full={} (semantic_prefix={}, state_x_out={}, public_bits={})",
        packed_semantic_lanes,
        semantic_prefix_lanes + full_public_lanes,
        semantic_prefix_lanes,
        state_x_out_lanes,
        artifact.shape.inputs
    );
}

#[test]
fn sha256_r1cs_shape_has_explicit_boolean_variables_for_typed_layout() {
    let artifact = synthesize_to_ccs(Sha256Circuit {
        preimage: b"abc".to_vec(),
    })
    .expect("synthesize SHA-256(abc)");

    let r1cs_shape = r1cs_f_prime::R1csShape::from(&artifact.sparse_r1cs);
    let widths = r1cs_shape.conservative_app_private_var_widths();
    let boolean_count = widths.iter().filter(|&&width| width == 1).count();
    let m = artifact.shape.inputs + artifact.shape.aux;
    let current_app_private_bits = m * POSEIDON2_GOLDILOCKS_BITS;
    let typed_app_private_bits: usize = widths.iter().sum();
    let mut non_boolean_bitlen_buckets = [0usize; 9];
    for (index, &width) in widths.iter().enumerate() {
        if width == 1 {
            continue;
        }
        let value = artifact.assignment[index].as_canonical_u64();
        let bit_len = if value == 0 {
            0
        } else {
            64 - value.leading_zeros() as usize
        };
        let bucket = match bit_len {
            0 => 0,
            1..=8 => 1,
            9..=16 => 2,
            17..=24 => 3,
            25..=32 => 4,
            33..=40 => 5,
            41..=48 => 6,
            49..=56 => 7,
            _ => 8,
        };
        non_boolean_bitlen_buckets[bucket] += 1;
    }

    assert!(
        boolean_count >= 6_000,
        "expected thousands of explicit Boolean variables, got {boolean_count}/{m}"
    );
    assert!(
        typed_app_private_bits < current_app_private_bits,
        "typed Boolean layout estimate must be narrower than canonical 64-bit lanes"
    );
    eprintln!(
        "[sha-layout] Boolean R1CS vars: {}/{}; app_private bits current={} typed_estimate={}",
        boolean_count, m, current_app_private_bits, typed_app_private_bits
    );
    eprintln!(
        "[sha-layout] non-Boolean assignment bit-length buckets: zero={}, 1-8={}, 9-16={}, 17-24={}, 25-32={}, 33-40={}, 41-48={}, 49-56={}, 57-64={}",
        non_boolean_bitlen_buckets[0],
        non_boolean_bitlen_buckets[1],
        non_boolean_bitlen_buckets[2],
        non_boolean_bitlen_buckets[3],
        non_boolean_bitlen_buckets[4],
        non_boolean_bitlen_buckets[5],
        non_boolean_bitlen_buckets[6],
        non_boolean_bitlen_buckets[7],
        non_boolean_bitlen_buckets[8],
    );
}

#[test]
fn sha256_typed_app_private_layout_static_width_cut() {
    let artifact = synthesize_to_ccs(Sha256Circuit {
        preimage: b"abc".to_vec(),
    })
    .expect("synthesize SHA-256(abc)");

    let m = artifact.shape.inputs + artifact.shape.aux;
    let legacy_plan = sha256_tiny_lifecycle_plan(m, artifact.shape.inputs);
    let typed_plan = sha256_tiny_lifecycle_plan_for_r1cs(&artifact.sparse_r1cs);
    let typed_bits: usize = typed_plan.app_private_var_widths.iter().sum();

    assert_eq!(legacy_plan.limbs, m * POSEIDON2_GOLDILOCKS_BITS + 1);
    assert_eq!(typed_plan.limbs, typed_bits + 1);
    assert!(
        typed_plan.limbs < legacy_plan.limbs,
        "typed app-private plan must reduce committed source coordinates"
    );

    let typed_layout = FPrimeImageLayout::new(build_recursive_step_image_config(&typed_plan));
    assert_eq!(typed_layout.app_private.bits, typed_bits);
    let (typed_structure, _) = r1cs_f_prime::build_r1cs_f_prime_structure(typed_layout, &artifact.sparse_r1cs);
    let acc = typed_plan
        .accumulator
        .as_ref()
        .expect("tiny SHA plan has accumulator");
    let NifsPayloadShape::CeClaim(ce_shape) = &typed_plan.nifs_payload_shapes[acc.ce_claim_payload_index] else {
        panic!("tiny SHA accumulator payload must be a CE claim");
    };
    assert_eq!(ce_shape.r_len, challenge_len_for_domain(typed_structure.ccs.n));
    assert_eq!(ce_shape.s_col_len, challenge_len_for_domain(typed_structure.ccs.m));
    eprintln!(
        "[sha-layout] typed app_private bits: current={} typed={} (vars={}); F' n={} m={} r_len={} s_col_len={}",
        legacy_plan.limbs - 1,
        typed_bits,
        m,
        typed_structure.ccs.n,
        typed_structure.ccs.m,
        ce_shape.r_len,
        ce_shape.s_col_len
    );
}

#[test]
fn sha256_production_core_r1cs_f_prime_static_shape_budget() {
    let artifact = synthesize_to_ccs(Sha256Circuit {
        preimage: b"abc".to_vec(),
    })
    .expect("synthesize SHA-256(abc)");

    let tiny_plan = sha256_tiny_lifecycle_plan_for_r1cs(&artifact.sparse_r1cs);
    let (prod_plan, prod_structure) = sha256_production_core_lifecycle_plan_for_r1cs(&artifact.sparse_r1cs);
    let prod_params = Params::for_ccs_shape(
        prod_structure.ccs.n,
        prod_structure.ccs.t(),
        prod_structure.ccs.max_degree(),
    )
    .expect("production-core params for SHA-256 F' shape");
    let prod_layout = FPrimeImageLayout::new(build_recursive_step_image_config(&prod_plan));
    let typed_bits: usize = prod_plan.app_private_var_widths.iter().sum();
    let acc = prod_plan
        .accumulator
        .as_ref()
        .expect("production-core SHA plan has accumulator");
    let NifsPayloadShape::CeClaim(ce_shape) = &prod_plan.nifs_payload_shapes[acc.ce_claim_payload_index] else {
        panic!("production-core SHA accumulator payload must be a CE claim");
    };

    assert!(
        prod_params.has_production_core(),
        "shape-specific params must keep the Appendix B.2 Goldilocks core"
    );
    assert_eq!(
        prod_params.kappa(),
        goldilocks_paper_b2::KAPPA,
        "production-core SHA shape must use production κ",
    );
    assert_eq!(
        acc.c_data_entries,
        goldilocks_paper_b2::KAPPA as usize * goldilocks_paper_b2::D,
        "terminal CE parent commitment width should scale with production κ",
    );
    assert_eq!(acc.child_count, goldilocks_paper_b2::K_RHO as u64);
    assert_eq!(prod_plan.limbs, typed_bits + 1);
    assert_eq!(
        prod_plan.limbs, tiny_plan.limbs,
        "typed app witness width is app-shape-owned; production κ should grow CE/NIFS authority, not app bits",
    );
    assert_eq!(ce_shape.r_len, challenge_len_for_domain(prod_structure.ccs.n));
    assert_eq!(ce_shape.s_col_len, challenge_len_for_domain(prod_structure.ccs.m));

    let state_bits = prod_layout.state_in.bits + prod_layout.state_out.bits + prod_layout.chunk_digest.bits;
    let control_bits = prod_layout.boundary.bits + prod_layout.app_private.bits + prod_layout.is_base.bits;
    let non_poseidon_bits = prod_layout.end - prod_layout.poseidon.bits;
    let app_private_pct = prod_layout.app_private.bits as f64 * 100.0 / prod_layout.end as f64;
    let poseidon_pct = prod_layout.poseidon.bits as f64 * 100.0 / prod_layout.end as f64;

    eprintln!(
        "[sha-prod-shape] params: kappa={}, lambda={}, m={}, b={}, k_rho={}, T={}",
        prod_params.kappa(),
        prod_params.lambda(),
        prod_params.m(),
        prod_params.b(),
        prod_params.k_rho(),
        prod_params.T(),
    );
    eprintln!(
        "[sha-prod-shape] app: constraints={}, inputs={}, aux={}, typed_bits={}",
        artifact.shape.constraints, artifact.shape.inputs, artifact.shape.aux, typed_bits,
    );
    eprintln!(
        "[sha-prod-shape] structure: n={}, m={}, t={}, image_width={}, poseidon_bits={}",
        prod_structure.ccs.n,
        prod_structure.ccs.m,
        prod_structure.ccs.t(),
        prod_layout.end,
        prod_layout.poseidon.bits,
    );
    eprintln!("[sha-prod-shape] layout breakdown:");
    eprintln!(
        "[sha-prod-shape]   app/private/control bits  {} ({:.2}%)",
        control_bits, app_private_pct
    );
    eprintln!("[sha-prod-shape]   state/chunk bits         {}", state_bits);
    eprintln!(
        "[sha-prod-shape]   NIFS payload bits        {}",
        prod_layout.nifs_payloads.bits
    );
    eprintln!("[sha-prod-shape]   kmul trace bits          {}", prod_layout.kmul.bits);
    eprintln!(
        "[sha-prod-shape]   ring-action trace bits   {}",
        prod_layout.ring_action.bits
    );
    eprintln!(
        "[sha-prod-shape]   Poseidon trace bits      {} ({:.2}%)",
        prod_layout.poseidon.bits, poseidon_pct
    );
    eprintln!("[sha-prod-shape]   non-Poseidon subtotal    {}", non_poseidon_bits);
    eprintln!(
        "[sha-prod-shape] CE shape: c_data={}, child_count={}, r_len={}, s_col_len={}, y_ring={}x{}",
        acc.c_data_entries,
        acc.child_count,
        ce_shape.r_len,
        ce_shape.s_col_len,
        ce_shape.y_ring_inner_lens.len(),
        ce_shape
            .y_ring_inner_lens
            .first()
            .copied()
            .unwrap_or_default(),
    );
}

impl TerminalOptimizedProveFixture {
    fn validate(&self) -> Result<(), String> {
        if self.header.version != TERMINAL_REPLAY_FIXTURE_VERSION {
            return Err(format!(
                "fixture version {} != {}",
                self.header.version, TERMINAL_REPLAY_FIXTURE_VERSION
            ));
        }
        let digest = structure_digest(&self.structure);
        if digest != self.header.structure_digest {
            return Err("structure digest mismatch".into());
        }
        let shape = (self.structure.n, self.structure.m, self.structure.t());
        let expected_shape = (
            self.header.structure_n,
            self.header.structure_m,
            self.header.structure_t,
        );
        if shape != expected_shape {
            return Err(format!(
                "structure shape {shape:?} != fixture header {expected_shape:?}"
            ));
        }
        if self.params.kappa != self.header.params_kappa
            || self.params.d != self.header.params_d
            || self.params.m != self.header.params_m
            || self.params.lambda != self.header.params_lambda
        {
            return Err("params fingerprint mismatch".into());
        }
        if self.mcs_list.len() != self.header.mcs_count || self.mcs_witnesses.len() != self.header.mcs_count {
            return Err("MCS count mismatch".into());
        }
        if self.me_inputs.len() != self.header.me_count || self.me_witnesses.len() != self.header.me_count {
            return Err("ME count mismatch".into());
        }
        Ok(())
    }
}

fn terminal_replay_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/perf-fixtures/neo-fold-clean/sha256-r1cs-fprime-terminal-optimized-prove-v1.bincode")
}

fn load_or_build_terminal_replay_fixture() -> TerminalOptimizedProveFixture {
    let path = terminal_replay_fixture_path();
    if let Ok(bytes) = fs::read(&path) {
        match bincode::deserialize::<TerminalOptimizedProveFixture>(&bytes) {
            Ok(fixture) => match fixture.validate() {
                Ok(()) => {
                    eprintln!("[sha-ivc] loaded replay fixture: {}", path.display());
                    return fixture;
                }
                Err(err) => eprintln!("[sha-ivc] stale replay fixture ({}): {err}", path.display()),
            },
            Err(err) => eprintln!("[sha-ivc] unreadable replay fixture ({}): {err}", path.display()),
        }
    }

    let fixture = build_terminal_replay_fixture();
    fixture.validate().expect("fresh replay fixture validates");
    let bytes = bincode::serialize(&fixture).expect("serialize terminal replay fixture");
    fs::create_dir_all(path.parent().expect("fixture path has parent")).expect("create fixture directory");
    fs::write(&path, bytes).expect("write terminal replay fixture");
    eprintln!("[sha-ivc] wrote replay fixture: {}", path.display());
    fixture
}

fn build_terminal_replay_fixture() -> TerminalOptimizedProveFixture {
    let artifact_a = phase("fixture synth A", || {
        synthesize_to_ccs(Sha256Circuit {
            preimage: b"abc".to_vec(),
        })
        .expect("synthesize SHA-256(abc)")
    });
    let plan = sha256_tiny_lifecycle_plan_for_r1cs(&artifact_a.sparse_r1cs);
    let params = sha256_tiny_neo_params();
    let prep = phase("fixture preprocess", || {
        r1cs_f_prime::preprocess_sparse_seeded_with_params(
            &artifact_a.sparse_r1cs,
            &plan,
            Params::test_only_from_neo_params(params.clone()),
            SHA256_AJTAI_SEED,
        )
        .expect("SHA-256 R1CS-F' tiny-params preprocess")
    });

    let mut chain = R1csChainBuilder::new(&prep).expect("start chain");
    let compiled_a = phase("fixture step 0 append", || {
        chain
            .append_assignment(artifact_a.assignment.clone())
            .expect("base step appends")
    });
    let artifact_b = phase("fixture synth B", || {
        synthesize_to_ccs(Sha256Circuit {
            preimage: b"xyz".to_vec(),
        })
        .expect("synthesize SHA-256(xyz)")
    });
    let compiled_b = phase("fixture step 1 append", || {
        chain
            .append_assignment(artifact_b.assignment.clone())
            .expect("recursive step appends")
    });
    drop(compiled_a);
    drop(compiled_b);

    let audit = chain.audit().expect("audit after recursive append");
    let ProofState::Active { running, latest } = &audit.proof.state.proof else {
        panic!("two-step SHA chain must be active before finalization");
    };
    let parent = running
        .parent_authority
        .as_ref()
        .expect("non-empty running accumulator carries parent authority");
    let mcs_list: Vec<CcsClaim> = latest
        .instances
        .iter()
        .map(|inst| inst.claim.clone())
        .collect();
    let mcs_witnesses: Vec<CcsWitness> = latest
        .instances
        .iter()
        .map(|inst| inst.witness.clone())
        .collect();
    let public_instance_digest = pi_ccs_instance_digest_parent_authority(&mcs_list, running.claims.len(), Some(parent));
    let me_input_accumulator_handle =
        AccumulatorHandle::from_running_parts(&running.claims, Some(parent)).digest_fields();
    let structure = prep.prep.structure().clone();
    let structure_digest = structure_digest(&structure);
    TerminalOptimizedProveFixture {
        header: TerminalOptimizedProveFixtureHeader {
            version: TERMINAL_REPLAY_FIXTURE_VERSION,
            ajtai_seed: SHA256_AJTAI_SEED,
            structure_n: structure.n,
            structure_m: structure.m,
            structure_t: structure.t(),
            structure_digest,
            params_kappa: params.kappa,
            params_d: params.d,
            params_m: params.m,
            params_lambda: params.lambda,
            mcs_count: mcs_list.len(),
            me_count: running.claims.len(),
        },
        params,
        structure,
        mcs_list,
        mcs_witnesses,
        me_inputs: running.claims.clone(),
        me_witnesses: running.witnesses.clone(),
        public_instance_digest,
        me_input_accumulator_handle,
    }
}

#[test]
#[ignore = "perf-only replay fixture; run manually to profile terminal optimized_prove without rebuilding the full SHA chain"]
fn sha256_terminal_optimized_prove_replay_perf() {
    let fixture = load_or_build_terminal_replay_fixture();

    let params = Params::test_only_from_neo_params(fixture.params.clone());
    let log = direct_ajtai::setup_seeded(&params, &fixture.structure, fixture.header.ajtai_seed);
    let cache = phase("replay optimized cache build", || {
        OptimizedStructureCache::build(&fixture.structure).expect("build optimized replay cache")
    });
    let mut tr = Poseidon2Transcript::new(FINAL_FOLD_TRANSCRIPT_LABEL);
    let (_terminal, _proof) = phase("replay terminal optimized_prove", || {
        optimized_replay_trace_with_cache_instance_digest_and_me_input_handle_and_perf(
            &mut tr,
            &fixture.params,
            &fixture.structure,
            &fixture.mcs_list,
            &fixture.mcs_witnesses,
            &fixture.me_inputs,
            &fixture.me_witnesses,
            fixture.public_instance_digest,
            fixture.me_input_accumulator_handle,
            &log,
            &cache,
        )
        .expect("terminal optimized replay accepts fixture")
    });
}

#[test]
fn sha256_bellpepper_ivc_chain_two_steps() {
    let total = Instant::now();

    // 1. Synthesize the first SHA-256 ("abc"). Cross-check the gadget
    //    produced the real digest before sinking it into the chain.
    let artifact_a = phase("synth A (Bellpepper SHA)", || {
        synthesize_to_ccs(Sha256Circuit {
            preimage: b"abc".to_vec(),
        })
        .expect("synthesize SHA-256(abc)")
    });
    eprintln!(
        "[sha-ivc]   shape: constraints={}, inputs={}, aux={}",
        artifact_a.shape.constraints, artifact_a.shape.inputs, artifact_a.shape.aux,
    );
    assert_eq!(
        artifact_a.public_inputs(),
        expected_sha256_public_inputs(b"abc"),
        "Bellpepper SHA-256(abc) must expose the real digest bits"
    );

    // 2. Preprocess R1CS-F' under tiny params. The verifier pins
    //    artifact_a's sparse-CCS shape as the per-step `F'_j`; both
    //    chain steps must satisfy this same shape.
    //
    //    Builds: the F' structure (semantic Boolean shell rows + R1CS
    //    recompose rows), the optimized engine cache (sparse + SuperNeo
    //    eval tables + matrix digest), structure_digest, vk, and the
    //    Ajtai PP. Single largest one-time cost in the test.
    let plan = sha256_tiny_lifecycle_plan_for_r1cs(&artifact_a.sparse_r1cs);
    let prep = phase("preprocess R1CS-F'", || {
        r1cs_f_prime::preprocess_sparse_seeded_with_params(
            &artifact_a.sparse_r1cs,
            &plan,
            sha256_tiny_params(),
            SHA256_AJTAI_SEED,
        )
        .expect("SHA-256 R1CS-F' tiny-params preprocess")
    });
    eprintln!(
        "[sha-ivc]   structure.n={}, structure.m={}, plan.limbs={}",
        prep.prep.structure().n,
        prep.prep.structure().m,
        plan.limbs,
    );

    // 3. Step 0 (base): SHA-256("abc"). `R1csChainBuilder` owns
    //    compile → lifecycle::prove for this step. Internally:
    //      - compile_step reuses the cached `prep.structure` Arc
    //        (built once at preprocess time), runs R1CS satisfaction,
    //        encodes the image, runs a satisfaction self-check against
    //        the cached structure.
    //      - lifecycle::prove runs Π_CCS sumcheck + Π_RLC + Π_DEC.
    let mut chain = R1csChainBuilder::new(&prep).expect("start chain");
    let compiled_a = phase("step 0 append (base)", || {
        chain
            .append_assignment(artifact_a.assignment.clone())
            .expect("base step (SHA-256 abc) appends")
    });
    assert!(
        compiled_a.encoded.image.decode_is_base(),
        "step 0 must take the base branch"
    );

    // 4. Second SHA-256 with a different preimage of the same byte
    //    length. Same R1CS shape (matrices are preimage-independent),
    //    different digest.
    let artifact_b = phase("synth B (Bellpepper SHA)", || {
        synthesize_to_ccs(Sha256Circuit {
            preimage: b"xyz".to_vec(),
        })
        .expect("synthesize SHA-256(xyz)")
    });
    assert_eq!(
        artifact_b.shape, artifact_a.shape,
        "same-length preimages must produce the same R1CS shape"
    );
    assert_ne!(
        artifact_b.public_inputs(),
        artifact_a.public_inputs(),
        "different preimages must produce different SHA-256 digests"
    );

    // 5. Step 1 (recursive): SHA-256("xyz"). Heaviest single phase —
    //    inside `append_assignment` the builder does THREE things:
    //      a. `prepare_next_fold`: extend a CLONED audit with step 0's
    //         latest to derive the NIFS proof + post_running that the
    //         recursive compile needs. This is a full Π_CCS+RLC+DEC
    //         prover pass on the cloned audit. The post-fold audit is
    //         **stashed** and reused at deposit time (see (c)).
    //      b. `compile_step`: reuse the cached `prep.structure` Arc,
    //         run NIFS.V on the prior fold under the per-step F'
    //         transcript, encode the image with the prior fold
    //         authority embedded in the NIFS payload, satisfaction
    //         self-check against the cached structure.
    //      c. deposit: swap the freshly compiled recursive instance into
    //         the stashed post-fold audit's `latest` — **no** second
    //         prover pass (the fold from (a) is reused; the chunk digest
    //         is shape+count-only so it is identical for the real
    //         instance). So step 1's wall time ≈ 1 × (prove cost) +
    //         1 × (compile + NIFS.V), not 2× prove.
    let compiled_b = phase("step 1 append (recursive)", || {
        chain
            .append_assignment(artifact_b.assignment.clone())
            .expect("recursive step (SHA-256 xyz) appends")
    });
    assert!(
        !compiled_b.encoded.image.decode_is_base(),
        "step 1 must take the recursive branch"
    );

    // 6. HyperNova fixed-`F'_j` invariant: base and recursive R1CS-F'
    //    compiles share one verifier-owned structure (`prep.plan`),
    //    so their encoded steps share one `structure_digest`. This is
    //    the load-bearing IVC property.
    assert!(
        std::sync::Arc::ptr_eq(&compiled_a.encoded.structure, &compiled_b.encoded.structure),
        "base and recursive SHA-256 R1CS-F' compiles must share one verifier-owned structure \
         (HyperNova \u{00A7}6.3 Construction 2 fixed-`F'_j` invariant)"
    );

    // 7. Each step's semantic-state digest absorbs its app public input
    //    (the one constant plus 256 SHA digest bits, packed into 64-bit
    //    lanes), and `state_x_out` absorbs that semantic digest. Different
    //    preimages therefore yield different `public_output_digest`s.
    //    The verifier-visible chain output really commits to which SHA
    //    was proven, not just "some SHA was proven".
    assert_ne!(
        compiled_a.public_output_digest, compiled_b.public_output_digest,
        "state_x_out must depend on each step's SHA-256 digest"
    );

    // 8. Builder advanced the lifecycle once per appended step.
    //    audit.steps is the per-step `StepProof` trail the audit-form
    //    verifier replays.
    assert_eq!(
        chain
            .audit()
            .expect("audit after recursive append")
            .steps
            .len(),
        2,
        "builder must extend the lifecycle once per compiled SHA-256 step"
    );
    assert_eq!(
        chain.context().chain_state.step_count,
        2,
        "builder must thread chain state across base and recursive appends"
    );

    // 9. Drop the per-step compiled outputs (no longer needed by the
    //    terminal verifier path). Each `EncodedFPrimeStep` carries an
    //    `Arc<FPrimeStructure>` cloned from `prep.structure`, so the
    //    structure itself is freed only when `prep` is dropped below;
    //    dropping a compiled step frees the per-step image + witness
    //    (a few MiB each) and decrements the Arc.
    let drop_compiled = Instant::now();
    drop(compiled_a);
    drop(compiled_b);
    eprintln!(
        "[sha-ivc] {:<32} {:>7.2}s",
        "drop compiled steps",
        drop_compiled.elapsed().as_secs_f64()
    );

    // 10. Finalize with audit. The audit verifier is the accepting
    //     surface for this two-chunk stateful chain; terminal-only
    //     `verify_uncompressed` must reject until the compressed decider
    //     proves the cross-step recursive-link rows.
    let audit = phase("chain.finish_with_audit()", || {
        chain
            .finish_with_audit()
            .expect("finish SHA-256 R1CS-F' chain with audit")
    });
    phase("verify_uncompressed_audit", || {
        lifecycle::verify_uncompressed_audit(&prep.prep, &audit).expect("audit verifier accepts SHA-256 R1CS-F' chain")
    });
    phase("verify_uncompressed (expected reject)", || {
        let err = lifecycle::verify_uncompressed(&prep.prep, &audit.proof)
            .expect_err("terminal-only verifier must reject multi-chunk F' SHA chain");
        expect_f_prime_non_replay_unsupported(err, 2);
    });

    // 11. Drop the remaining heavy allocations explicitly so the wall
    //     time is attributed to a labeled phase.
    let drop_rest = Instant::now();
    drop(audit);
    drop(prep);
    drop(artifact_a);
    drop(artifact_b);
    eprintln!(
        "[sha-ivc] {:<32} {:>7.2}s",
        "drop prep + proof + artifacts",
        drop_rest.elapsed().as_secs_f64()
    );

    eprintln!(
        "[sha-ivc] {:<32} {:>7.2}s",
        "TOTAL (incl. drops)",
        total.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore = "production-core perf snapshot; run manually because it is heavier than the tiny algebraic-correctness SHA fixture"]
fn sha256_production_core_bellpepper_ivc_chain_two_steps_perf_snapshot() {
    let total = Instant::now();
    let artifact_a = phase("prod synth A (Bellpepper SHA)", || {
        synthesize_to_ccs(Sha256Circuit {
            preimage: b"abc".to_vec(),
        })
        .expect("synthesize SHA-256(abc)")
    });
    let (plan, structure_probe) = sha256_production_core_lifecycle_plan_for_r1cs(&artifact_a.sparse_r1cs);
    let params = Params::for_ccs_shape(
        structure_probe.ccs.n,
        structure_probe.ccs.t(),
        structure_probe.ccs.max_degree(),
    )
    .expect("production-core params");
    eprintln!(
        "[sha-prod-ivc] params: kappa={}, lambda={}, m={}, b={}, k_rho={}, T={}",
        params.kappa(),
        params.lambda(),
        params.m(),
        params.b(),
        params.k_rho(),
        params.T(),
    );

    let prep = phase("prod preprocess R1CS-F'", || {
        r1cs_f_prime::preprocess_sparse_seeded_with_params(&artifact_a.sparse_r1cs, &plan, params, SHA256_AJTAI_SEED)
            .expect("production-core SHA-256 R1CS-F' preprocess")
    });
    eprintln!(
        "[sha-prod-ivc]   structure.n={}, structure.m={}, plan.limbs={}",
        prep.prep.structure().n,
        prep.prep.structure().m,
        plan.limbs,
    );

    let mut chain = R1csChainBuilder::new(&prep).expect("start chain");
    let compiled_a = phase("prod step 0 append (base)", || {
        chain
            .append_assignment(artifact_a.assignment.clone())
            .expect("base step appends")
    });
    let artifact_b = phase("prod synth B (Bellpepper SHA)", || {
        synthesize_to_ccs(Sha256Circuit {
            preimage: b"xyz".to_vec(),
        })
        .expect("synthesize SHA-256(xyz)")
    });
    assert_eq!(
        artifact_b.shape, artifact_a.shape,
        "same-length preimages must produce the same R1CS shape"
    );
    let compiled_b = phase("prod step 1 append (recursive)", || {
        chain
            .append_assignment(artifact_b.assignment.clone())
            .expect("recursive step appends")
    });
    assert!(std::sync::Arc::ptr_eq(
        &compiled_a.encoded.structure,
        &compiled_b.encoded.structure
    ));
    drop(compiled_a);
    drop(compiled_b);

    let audit = phase("prod chain.finish_with_audit()", || {
        chain
            .finish_with_audit()
            .expect("finish production-core SHA chain with audit")
    });
    phase("prod verify_uncompressed_audit", || {
        lifecycle::verify_uncompressed_audit(&prep.prep, &audit)
            .expect("audit verifier accepts production-core SHA chain")
    });
    phase("prod verify_uncompressed (expected reject)", || {
        let err = lifecycle::verify_uncompressed(&prep.prep, &audit.proof)
            .expect_err("terminal-only verifier must reject production-core multi-chunk F' SHA chain");
        expect_f_prime_non_replay_unsupported(err, 2);
    });

    drop(audit);
    drop(prep);
    drop(artifact_a);
    drop(artifact_b);
    eprintln!(
        "[sha-prod-ivc] {:<32} {:>7.2}s",
        "TOTAL (incl. drops)",
        total.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore = "production-core serial-folding perf snapshot; run manually to measure repeated recursive appends"]
fn sha256_production_core_bellpepper_ivc_chain_five_steps_perf_snapshot() {
    const STEPS: usize = 5;

    let total = Instant::now();
    let reference = synthesize_to_ccs(Sha256Circuit {
        preimage: nth_preimage(0),
    })
    .expect("reference SHA synth");
    let (plan, structure_probe) = sha256_production_core_lifecycle_plan_for_r1cs(&reference.sparse_r1cs);
    let params = Params::for_ccs_shape(
        structure_probe.ccs.n,
        structure_probe.ccs.t(),
        structure_probe.ccs.max_degree(),
    )
    .expect("production-core params");
    let prep = phase("prod-5 preprocess R1CS-F'", || {
        r1cs_f_prime::preprocess_sparse_seeded_with_params(&reference.sparse_r1cs, &plan, params, SHA256_AJTAI_SEED)
            .expect("production-core SHA-256 R1CS-F' preprocess")
    });

    let mut chain = R1csChainBuilder::new(&prep).expect("start chain");
    let mut recursive_append_s = 0.0;
    let mut recursive_count = 0usize;

    for step in 0..STEPS {
        let assignment = if step == 0 {
            reference.assignment.clone()
        } else {
            let artifact = synthesize_to_ccs(Sha256Circuit {
                preimage: nth_preimage(step),
            })
            .expect("same-shape SHA synth");
            assert_eq!(
                artifact.shape, reference.shape,
                "same-length SHA preimages must keep the same R1CS shape"
            );
            artifact.assignment
        };

        let start = Instant::now();
        let compiled = chain
            .append_assignment(assignment)
            .expect("append SHA step");
        let elapsed_s = start.elapsed().as_secs_f64();
        if step == 0 {
            assert!(compiled.encoded.image.decode_is_base());
            eprintln!("[sha-prod-5] step {step} base append        {elapsed_s:>7.2}s");
        } else {
            assert!(!compiled.encoded.image.decode_is_base());
            recursive_append_s += elapsed_s;
            recursive_count += 1;
            eprintln!("[sha-prod-5] step {step} recursive append   {elapsed_s:>7.2}s");
        }
        drop(compiled);
    }

    assert_eq!(
        chain
            .audit()
            .expect("audit after five production-core SHA appends")
            .steps
            .len(),
        STEPS
    );

    let audit = phase("prod-5 chain.finish_with_audit()", || {
        chain
            .finish_with_audit()
            .expect("finish production-core five-step SHA chain with audit")
    });
    phase("prod-5 verify_uncompressed_audit", || {
        lifecycle::verify_uncompressed_audit(&prep.prep, &audit)
            .expect("audit verifier accepts production-core five-step SHA chain")
    });
    phase("prod-5 verify_uncompressed (expected reject)", || {
        let err = lifecycle::verify_uncompressed(&prep.prep, &audit.proof)
            .expect_err("terminal-only verifier must reject production-core five-step F' SHA chain");
        expect_f_prime_non_replay_unsupported(err, STEPS as u64);
    });
    eprintln!(
        "[sha-prod-5] recursive append avg       {:>7.2}s/op",
        recursive_append_s / recursive_count as f64
    );
    eprintln!(
        "[sha-prod-5] TOTAL (incl. drops)        {:>7.2}s",
        total.elapsed().as_secs_f64()
    );
}

fn run_sha256_param_probe_bellpepper_ivc_chain_five_steps(label: &str, params: Params) {
    const STEPS: usize = 5;

    let total = Instant::now();
    let reference = synthesize_to_ccs(Sha256Circuit {
        preimage: nth_preimage(0),
    })
    .expect("reference SHA synth");
    let (plan, structure_probe) = sha256_lifecycle_plan_for_r1cs_with_params(&reference.sparse_r1cs, &params);
    let acc = plan
        .accumulator
        .as_ref()
        .expect("parameter probe plan has accumulator");
    assert_eq!(acc.child_count, params.k_rho() as u64);
    eprintln!(
        "[{label}] params: kappa={}, lambda={}, m={}, b={}, k_rho={}, T={}",
        params.kappa(),
        params.lambda(),
        params.m(),
        params.b(),
        params.k_rho(),
        params.T(),
    );
    eprintln!(
        "[{label}] probe structure: n={}, m={}, plan.limbs={}, child_count={}",
        structure_probe.ccs.n, structure_probe.ccs.m, plan.limbs, acc.child_count,
    );

    let prep = phase(&format!("{label} preprocess R1CS-F'"), || {
        r1cs_f_prime::preprocess_sparse_seeded_with_params(
            &reference.sparse_r1cs,
            &plan,
            params.clone(),
            SHA256_AJTAI_SEED,
        )
        .expect("parameter-probe SHA-256 R1CS-F' preprocess")
    });

    let mut chain = R1csChainBuilder::new(&prep).expect("start parameter-probe chain");
    let mut recursive_append_s = 0.0;
    let mut recursive_count = 0usize;

    for step in 0..STEPS {
        let assignment = if step == 0 {
            reference.assignment.clone()
        } else {
            let artifact = synthesize_to_ccs(Sha256Circuit {
                preimage: nth_preimage(step),
            })
            .expect("same-shape SHA synth");
            assert_eq!(
                artifact.shape, reference.shape,
                "same-length SHA preimages must keep the same R1CS shape"
            );
            artifact.assignment
        };

        let start = Instant::now();
        let compiled = chain
            .append_assignment(assignment)
            .expect("append parameter-probe SHA step");
        let elapsed_s = start.elapsed().as_secs_f64();
        if step == 0 {
            assert!(compiled.encoded.image.decode_is_base());
            eprintln!("[{label}] step {step} base append        {elapsed_s:>7.2}s");
        } else {
            assert!(!compiled.encoded.image.decode_is_base());
            recursive_append_s += elapsed_s;
            recursive_count += 1;
            eprintln!("[{label}] step {step} recursive append   {elapsed_s:>7.2}s");
        }
        drop(compiled);
    }

    assert_eq!(
        chain
            .audit()
            .expect("audit after five parameter-probe SHA appends")
            .steps
            .len(),
        STEPS
    );

    let audit = phase(&format!("{label} chain.finish_with_audit()"), || {
        chain
            .finish_with_audit()
            .expect("finish parameter-probe five-step SHA chain with audit")
    });
    phase(&format!("{label} verify_uncompressed_audit"), || {
        lifecycle::verify_uncompressed_audit(&prep.prep, &audit)
            .expect("audit verifier accepts parameter-probe five-step SHA chain")
    });
    phase(&format!("{label} verify_uncompressed (expected reject)"), || {
        let err = lifecycle::verify_uncompressed(&prep.prep, &audit.proof)
            .expect_err("terminal-only verifier must reject parameter-probe five-step F' SHA chain");
        expect_f_prime_non_replay_unsupported(err, STEPS as u64);
    });
    eprintln!(
        "[{label}] recursive append avg        {:>7.2}s/op",
        recursive_append_s / recursive_count as f64
    );
    eprintln!(
        "[{label}] TOTAL (incl. drops)         {:>7.2}s",
        total.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore = "b=3 parameter probe; run manually to compare against the b=2 production-core perf snapshot"]
fn sha256_b3_probe_bellpepper_ivc_chain_five_steps_perf_snapshot() {
    run_sha256_param_probe_bellpepper_ivc_chain_five_steps("sha-b3-5", sha256_b3_probe_params());
}

#[test]
#[ignore = "b=2,k_rho=12 parameter probe; run manually to compare against production k_rho=14"]
fn sha256_b2_k12_probe_bellpepper_ivc_chain_five_steps_perf_snapshot() {
    run_sha256_param_probe_bellpepper_ivc_chain_five_steps("sha-b2-k12-5", sha256_b2_k12_probe_params());
}

/// Distinct 3-byte preimage for index `i` (same length ⇒ same R1CS
/// shape, different digest ⇒ genuinely different SHA instance).
fn nth_preimage(i: usize) -> Vec<u8> {
    vec![
        b'a' + (i % 26) as u8,
        b'a' + ((i / 26) % 26) as u8,
        b'a' + ((i / 676) % 26) as u8,
    ]
}

/// SHA app-public output is currently serial-only: one chunk carries one
/// outgoing semantic-state digest. K=4 must reject until an aggregate
/// public-output digest design lands.
#[test]
fn sha256_bellpepper_ivc_chain_rejects_k4_chunks() {
    const K: usize = 4;

    let ref_artifact = synthesize_to_ccs(Sha256Circuit {
        preimage: nth_preimage(0),
    })
    .expect("reference synth");
    let plan = sha256_tiny_lifecycle_plan_for_r1cs(&ref_artifact.sparse_r1cs);
    let prep = phase("preprocess R1CS-F'", || {
        r1cs_f_prime::preprocess_sparse_seeded_with_params(
            &ref_artifact.sparse_r1cs,
            &plan,
            sha256_tiny_params(),
            SHA256_AJTAI_SEED,
        )
        .expect("preprocess")
    });

    let mk_batch = |chunk: usize| -> Vec<Vec<F>> {
        (0..K)
            .map(|i| {
                synthesize_to_ccs(Sha256Circuit {
                    preimage: nth_preimage(chunk * K + i),
                })
                .expect("synth")
                .assignment
            })
            .collect()
    };

    let mut chain = R1csChainBuilder::new(&prep).expect("chain");

    let err = match chain.append_assignments(mk_batch(0)) {
        Ok(_) => panic!("K=4 SHA app-public semantic chunk must reject"),
        Err(err) => err,
    };
    assert!(
        matches!(err, r1cs_f_prime::Error::Compiler(R1csCompilerError::StatefulChunkMustBeSerial { got }) if got == K),
        "expected StatefulChunkMustBeSerial(K=4), got {err:?}"
    );
}

/// Even K=2 is rejected for the same reason. This keeps the old
/// benchmark entry point from accidentally suggesting that batched SHA
/// app-public output is production-supported.
#[test]
fn sha256_bellpepper_ivc_chain_rejects_k2_chunks() {
    let ref_artifact = synthesize_to_ccs(Sha256Circuit {
        preimage: nth_preimage(0),
    })
    .expect("reference synth");
    let plan = sha256_tiny_lifecycle_plan_for_r1cs(&ref_artifact.sparse_r1cs);
    let prep = r1cs_f_prime::preprocess_sparse_seeded_with_params(
        &ref_artifact.sparse_r1cs,
        &plan,
        sha256_tiny_params(),
        SHA256_AJTAI_SEED,
    )
    .expect("preprocess");
    let mut chain = R1csChainBuilder::new(&prep).expect("chain");
    let batch = (0..2)
        .map(|i| {
            synthesize_to_ccs(Sha256Circuit {
                preimage: nth_preimage(i),
            })
            .expect("synth")
            .assignment
        })
        .collect();

    let err = match chain.append_assignments(batch) {
        Ok(_) => panic!("K=2 SHA app-public semantic chunk must reject"),
        Err(err) => err,
    };
    assert!(
        matches!(err, r1cs_f_prime::Error::Compiler(R1csCompilerError::StatefulChunkMustBeSerial { got }) if got == 2),
        "expected StatefulChunkMustBeSerial(K=2), got {err:?}"
    );
}
