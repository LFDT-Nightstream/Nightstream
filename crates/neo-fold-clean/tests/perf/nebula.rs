//! Perf snapshot — spec §13 step 8: record the §10 cost-budget actuals.
//! R7 is an active production-budget gate; the remaining snapshots are
//! ignored. Off-by-2× on any §10 line reopens the spec.
//!
//! ```text
//! cargo test -p neo-fold-clean --release --test perf_nebula -- \
//!     nebula_v3_targets_folded_f_prime_production_preflight -- --exact --nocapture
//! ```

use std::time::Instant;

use neo_ccs::CcsMatrix;
use neo_fold_clean::config;
use neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimeRelation;
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::nebula::prove::prove_segment;
use neo_fold_clean::frontends::nebula::trace::Memory;
use neo_fold_clean::lifecycle::{self, preprocess, verify_uncompressed_audit};
use neo_fold_clean::paper::f_prime::nebula_lane_circuit::delayed_nebula_public_suffix_len;
use neo_fold_clean::paper::f_prime::r1cs::F_PRIME_PUBLIC_INPUT_LEN;
use neo_fold_clean::paper::reductions::accumulator_sis_circuit::PROTOCOL_BINDING_KAPPA;
use neo_fold_clean::paper::relations::superneo_public_x_cols;
use neo_math::D;

#[path = "../support/mod.rs"]
mod support;

const LANE_KAPPA: usize = 18;
const PRODUCTION_F_PRIME_COMMITTED_COORDINATE_TARGET: usize = 25_000_000;

/// R7 production preflight: Appendix B.2 SuperNeo parameters over the maximum
/// normative v3.1 memory geometry. Width is audited before allocating the
/// fixed point, so an oversized design fails without constructing huge CCS
/// matrices.
#[test]
fn nebula_v3_targets_folded_f_prime_production_preflight() {
    let memory_params = NebulaParams::v3_targets()
        .with_stacks(2, 12)
        .expect("v3 targets plus maximum stack geometry");
    let params = neo_fold_clean::Params::for_ccs_shape_with(
        PRODUCTION_F_PRIME_COMMITTED_COORDINATE_TARGET,
        13,
        8,
        config::MIN_EFFECTIVE_LAMBDA,
        config::EXTENSION_SAFETY_MARGIN_BITS,
    )
    .expect("production coordinate upper bound must fit the supported extension policy");
    let rom = vec![0u32; memory_params.rom_cells() as usize];

    let started = Instant::now();
    let plan =
        NebulaPlan::new(memory_params, rom, [0xC7; 32], params.kappa() as usize).expect("production Nebula plan");
    let plan_time = started.elapsed();

    let started = Instant::now();
    let audit = NebulaFPrimeRelation::audit_field_shapes(&params, plan.circuit().structure(), &plan)
        .expect("production field-arm census");
    let census_time = started.elapsed();
    println!("== R7 authoritative folded F' production preflight ==");
    println!(
        "params                    κ={} k_rho={} T={} λ={}",
        params.kappa(),
        params.k_rho(),
        params.T(),
        params.lambda()
    );
    println!(
        "memory                    r={} μ={} B_ops={} B_scan={} N={} stacks={} σ={}",
        memory_params.r,
        memory_params.mu,
        memory_params.b_ops,
        memory_params.b_scan,
        memory_params.steps_per_segment(),
        memory_params.num_stacks,
        memory_params.sigma
    );
    println!(
        "S_mem                     rows={} cols={} nnz={} m_in={}",
        plan.circuit().rows(),
        plan.circuit().cols(),
        plan.circuit().nnz(),
        plan.circuit().m_in()
    );
    println!("field arms                {audit:?}");
    println!("plan / census             {plan_time:.2?} / {census_time:.2?}");
    assert_eq!(params.kappa(), 18, "Appendix B.2 κ");
    assert_eq!(params.k_rho(), 14, "Appendix B.2 k_rho");
    assert_eq!(params.T(), 216, "Appendix B.2 T");
    let fold_inputs = 1 + params.k_rho() as usize;
    let folded_t = plan.circuit().structure().t();
    let fresh_m_in = F_PRIME_PUBLIC_INPUT_LEN + delayed_nebula_public_suffix_len(plan.config().stacks);
    let active_x_columns = superneo_public_x_cols(fresh_m_in);
    let projection_identities = 4 * params.kappa() as usize + active_x_columns + 2 * folded_t + 2;
    let projection_pairs = fold_inputs * projection_identities;
    println!(
        "projection census          n={fold_inputs} t={folded_t} a_X={active_x_columns} J={projection_identities} P={projection_pairs}"
    );
    assert_eq!(projection_identities, 150, "maximum-geometry Lemma 5 identity census");
    assert_eq!(projection_pairs, 2_250, "maximum-geometry Lemma 5 pair census");

    let started = Instant::now();
    let width_audit = NebulaFPrimeRelation::audit_low_norm_width(&params, plan.circuit().structure(), &plan)
        .expect("production selective-width census");
    let width_audit_time = started.elapsed();
    println!("== selective low-norm width ==");
    println!(
        "shared prefix              {} (public {}, selectors {}, padding {}, app-private {})",
        width_audit.branch_start,
        width_audit.public_coordinates,
        width_audit.selector_coordinates,
        width_audit.alignment_padding,
        width_audit.shared_private_coordinates,
    );
    for (name, arm) in ["base", "bootstrap", "steady"]
        .into_iter()
        .zip(&width_audit.arms)
    {
        println!(
            "{name:<10} branch         {:>10} + derived {:>10} = {:>10} (unit {}, balanced {}, binary {}, eliminated {})",
            arm.branch_coordinates,
            arm.derived_coordinates,
            arm.total_branch_coordinates,
            arm.unit_columns,
            arm.balanced_columns,
            arm.binary_columns,
            arm.eliminated_columns,
        );
        println!(
            "{name:<10} trace values   poseidon {:>10} / poly-eval {:>8} / product {:>8} (internal {:>8})",
            arm.traces.poseidon2_coordinates,
            arm.traces.polynomial_evaluation_coordinates,
            arm.traces.product_sum_coordinates,
            arm.traces.product_sum_internal_coordinates,
        );
    }
    let steady = width_audit.arms.last().expect("steady arm");
    println!("steady row-family touches (nested ranges overlap):");
    for family in &steady.row_families {
        if family.name.ends_with(".total") {
            continue;
        }
        println!(
            "  {:<32} coords {:>10} (unit {:>8}, balanced {:>8}, binary {:>4}, p2 {:>4}/{:>9})",
            family.name,
            family.coordinates_before_aliases,
            family.unit_columns,
            family.balanced_columns,
            family.binary_columns,
            family.poseidon2_permutations,
            family.poseidon2_coordinates,
        );
    }
    println!("total selective width     {}", width_audit.total_coordinates);
    println!(
        "zero-derived lower bound  {}",
        width_audit.total_coordinates - steady.derived_coordinates
    );
    println!("width census              {width_audit_time:.2?}");
    assert_eq!(
        width_audit.total_coordinates, 15_730_104,
        "production selective-width census drifted"
    );
    assert!(
        width_audit.total_coordinates <= PRODUCTION_F_PRIME_COMMITTED_COORDINATE_TARGET,
        "production selective lowering uses {} committed coordinates, above the {}-coordinate test target",
        width_audit.total_coordinates,
        PRODUCTION_F_PRIME_COMMITTED_COORDINATE_TARGET,
    );

    let started = Instant::now();
    let relation = NebulaFPrimeRelation::compile_fixed_point_with_coordinate_limit(
        &params,
        &plan,
        PRODUCTION_F_PRIME_COMMITTED_COORDINATE_TARGET,
    )
    .expect("production authoritative relation must stabilize within the test target");
    let fixed_point_time = started.elapsed();
    let structure = relation.structure();
    let mut stored_sparse_nnz = 0usize;
    let mut seeded_blocks = 0usize;
    let mut seeded_coefficient_slots = 0u128;
    let mut long_binding_blocks = 0usize;
    let mut digest_compression_blocks = 0usize;
    let mut max_long_binding_words = 0usize;
    let mut max_long_binding_cols = 0usize;
    let mut max_digest_compression_words = 0usize;
    let mut max_digest_compression_cols = 0usize;
    for matrix in &structure.matrices {
        match matrix {
            CcsMatrix::Identity { n } => stored_sparse_nnz += n,
            CcsMatrix::Csc(csc) => stored_sparse_nnz += csc.vals.len(),
            CcsMatrix::CscWithSeededPhi81 { csc, blocks, .. } => {
                stored_sparse_nnz += csc.vals.len();
                seeded_blocks += blocks.len();
                for block in blocks {
                    assert_eq!(
                        block.word_width(),
                        41,
                        "R2 binding maps must reuse the 41 authoritative trits"
                    );
                    match block.kappa() {
                        PROTOCOL_BINDING_KAPPA => {
                            long_binding_blocks += 1;
                            max_long_binding_words = max_long_binding_words.max(block.word_starts().len());
                            max_long_binding_cols = max_long_binding_cols.max(block.message_cols());
                        }
                        1 => {
                            digest_compression_blocks += 1;
                            max_digest_compression_words = max_digest_compression_words.max(block.word_starts().len());
                            max_digest_compression_cols = max_digest_compression_cols.max(block.message_cols());
                        }
                        rank => panic!("unexpected compact-binding rank {rank}"),
                    }
                    seeded_coefficient_slots +=
                        (D as u128) * (D as u128) * (block.kappa() as u128) * (block.message_cols() as u128);
                }
            }
        }
    }
    println!(
        "fixed point                rows={} cols={} matrices={} degree={} sparse_nnz={} seeded_blocks={} seeded_slots={} long_blocks={} long_max={}/{} short_blocks={} short_max={}/{} time={fixed_point_time:.2?}",
        structure.n,
        structure.m,
        structure.t(),
        structure.max_degree(),
        stored_sparse_nnz,
        seeded_blocks,
        seeded_coefficient_slots,
        long_binding_blocks,
        max_long_binding_words,
        max_long_binding_cols,
        digest_compression_blocks,
        max_digest_compression_words,
        max_digest_compression_cols,
    );
    assert!(
        structure.n < structure.m,
        "SplitNc must preserve the smaller semantic-row domain"
    );
    assert!(
        !structure.matrices[0].is_identity(),
        "SplitNc must not carry the obsolete NC identity matrix"
    );
    assert_eq!(structure.t(), 13, "production verifier matrix count drifted");
    assert_eq!(seeded_blocks, 72, "production compact-binding block count drifted");
    assert_eq!(
        seeded_coefficient_slots, 480_533_472,
        "production compact-binding slot count drifted"
    );
    assert_eq!(long_binding_blocks, 36, "production rank-2 binding block count drifted");
    assert_eq!(
        max_long_binding_words, 27_233,
        "maximum rank-2 binding preimage width drifted"
    );
    assert_eq!(
        max_long_binding_cols, 20_677,
        "maximum rank-2 binding message dimension drifted"
    );
    assert_eq!(
        digest_compression_blocks, 36,
        "short digest-compression block count drifted"
    );
    assert_eq!(
        max_digest_compression_words, 108,
        "maximum short-map preimage width drifted"
    );
    assert_eq!(
        max_digest_compression_cols, 82,
        "maximum short-map message dimension drifted"
    );
    let extension = params
        .validate_ccs_shape(structure.n.max(structure.m), structure.t(), structure.max_degree())
        .expect("production parameters must cover the actual fixed relation");
    assert!(
        extension.slack_bits >= config::EXTENSION_SAFETY_MARGIN_BITS as i32,
        "production relation has {} extension-safety bits, expected at least {}",
        extension.slack_bits,
        config::EXTENSION_SAFETY_MARGIN_BITS,
    );

    let budget = plan.error_budget();
    assert_eq!(budget.end_to_end_target_bits, config::NEBULA_END_TO_END_SECURITY_BITS);
    assert_eq!(budget.max_fs_query_bits, config::NEBULA_MAX_FS_QUERY_BITS);
    let d4_factor = params
        .ccs_soundness_factor(structure.n.max(structure.m), structure.t(), structure.max_degree())
        .expect("exact SuperNeo D.4 soundness factor");
    assert_eq!(d4_factor, 1_336_848, "SuperNeo D.4 production numerator drifted");
    let q_h = 2f64.powi(budget.max_fs_query_bits as i32);
    let n_seg = memory_params.seg_max as f64;
    let n_f = n_seg * memory_params.steps_per_segment() as f64;
    let log2_k = 2.0 * (params.q() as f64).log2();
    let pipeline_bits = log2_k - (q_h * n_f * d4_factor as f64).log2();
    let projection_bits = log2_k - (q_h * n_f * projection_pairs as f64 * (2 * D - 2) as f64).log2();
    let fingerprint_bits = log2_k - (q_h * n_seg * budget.m_seg as f64).log2();
    let challenge_set_bits = D as f64 * 5f64.log2();
    let mixing_bits = challenge_set_bits - (q_h * n_f * fold_inputs as f64).log2();
    // Conservative rounded floors from the pinned estimator run: the main
    // Appendix-B.2 maps exceed 100 bits, and the union of the five rank-2
    // maps plus their short rank-1 compression map exceeds 160 bits.
    let term_bits = [
        pipeline_bits,
        projection_bits,
        fingerprint_bits,
        mixing_bits,
        100.0,
        160.0,
        128.0,
    ];
    let end_to_end_bits = -(term_bits.iter().map(|bits| 2f64.powf(-bits)).sum::<f64>()).log2();
    println!(
        "security budget            q_H<=2^{} D.4={} pipe={pipeline_bits:.2} projection={projection_bits:.2} fingerprint={fingerprint_bits:.2} mixing={mixing_bits:.2} total={end_to_end_bits:.2} target={}",
        budget.max_fs_query_bits,
        d4_factor,
        budget.end_to_end_target_bits,
    );
    assert!(
        end_to_end_bits >= budget.end_to_end_target_bits as f64,
        "maximum-chain security is {end_to_end_bits:.2} bits, below the declared {}-bit target",
        budget.end_to_end_target_bits,
    );
    assert!(
        (65.3..65.4).contains(&end_to_end_bits),
        "maximum-chain union-bound accounting drifted: {end_to_end_bits:.4} bits"
    );
    assert_eq!(
        (structure.n, structure.m),
        (2_819_360, 15_612_210),
        "production rectangular verifier fixed point drifted"
    );
    assert!(
        structure.m <= PRODUCTION_F_PRIME_COMMITTED_COORDINATE_TARGET,
        "production fixed relation uses {} committed coordinates, above the {}-coordinate test target",
        structure.m,
        PRODUCTION_F_PRIME_COMMITTED_COORDINATE_TARGET,
    );
}

/// §10 structural actuals at the v3 targets (`r = 12, μ = 16, B = 64,
/// N = 1,088`): `S_mem` rows / columns / nnz, plan compile time (includes
/// the full `D_init` sweep of 1,088 lane commitments), and the evaluated
/// §9 budget. Structure-only — no folding.
#[test]
#[ignore]
fn nebula_v3_targets_structure_snapshot() {
    let params = NebulaParams::v3_targets();
    let t = Instant::now();
    let rom = vec![0u32; params.rom_cells() as usize];
    let plan = NebulaPlan::new(params, rom, [0xBE; 32], LANE_KAPPA).expect("v3 plan");
    let compile = t.elapsed();

    let c = plan.circuit();
    let budget = plan.error_budget();
    // §10 "per-op amortized rows": one step's rows divided by its op
    // slots (the scan cost is inside the step, so the ratio is per-step).
    let ops_amortized = c.rows() as f64 / params.b_ops as f64;
    println!("== Nebula §10 actuals, v3 targets ==");
    println!("S_mem rows                {:>10}   (§10 budget ≈ 58k)", c.rows());
    println!(
        "S_mem witness columns     {:>10}   (committed coordinates per step)",
        c.cols()
    );
    println!("S_mem nnz                 {:>10}", c.nnz());
    println!("steps per segment N       {:>10}", params.steps_per_segment());
    println!(
        "rows per op (amortized)   {:>10.1}   (§10 budget ≈ 875 at N_ops = R+M)",
        ops_amortized
    );
    println!(
        "§9 budget per FS attempt  m_seg = {} → 2^{:.1}",
        budget.m_seg, budget.log2_bound_per_attempt
    );
    println!("plan compile (incl. D_init sweep) {:>8.2?}", compile);

    // §10 discipline: off-by-2× on the headline row count reopens the spec.
    assert!(
        c.rows() < 2 * 58_000,
        "S_mem rows {} exceed 2× the §10 budget",
        c.rows()
    );
}

/// §10 stacks-delta actuals (v3.1, `S = 2, σ = 12` on the v3 targets):
/// row/column/nnz growth of `S_mem` and the widened `x`. Budget line:
/// ≈ +2.5k rows (≈ +4%), `x` +48 bits — off-by-2× reopens the spec.
#[test]
#[ignore]
fn nebula_v3_targets_stacks_delta_snapshot() {
    use neo_fold_clean::frontends::nebula::circuit::SMemCircuit;

    let base = NebulaParams::v3_targets();
    let stacked = base.with_stacks(2, 12).expect("v3 targets + stacks");
    let c0 = SMemCircuit::new(base);
    let c1 = SMemCircuit::new(stacked);

    let d_rows = c1.rows() - c0.rows();
    println!("== Nebula §10 stacks delta, v3 targets + (S = 2, σ = 12) ==");
    println!(
        "S_mem rows                {:>10}   (+{} vs S = 0, {:+.1}%)",
        c1.rows(),
        d_rows,
        100.0 * d_rows as f64 / c0.rows() as f64
    );
    println!(
        "S_mem witness columns     {:>10}   (+{})",
        c1.cols(),
        c1.cols() - c0.cols()
    );
    println!(
        "S_mem nnz                 {:>10}   (+{})",
        c1.nnz(),
        c1.nnz() - c0.nnz()
    );
    println!(
        "x bits                    {:>10}   (+{})",
        stacked.x_bits(),
        stacked.x_bits() - base.x_bits()
    );

    assert!(
        d_rows < 2 * 2_500,
        "stack rows delta {d_rows} exceeds 2× the §10 budget"
    );
    assert_eq!(stacked.x_bits() - base.x_bits(), 48);
}

/// End-to-end timing at the spec §2 test profile (`r = 4, μ = 8,
/// B_ops = B_scan = 8, N = 34`): one full segment — native pass, lane
/// commits, γ, 34 witnesses, 34 folds — plus finalization and audit
/// verification.
#[test]
#[ignore]
fn nebula_test_profile_segment_snapshot() {
    let params = NebulaParams::test_profile();
    let rom: Vec<u32> = (0..params.rom_cells() as u32).map(|i| i * 3 + 1).collect();
    let plan = NebulaPlan::new(params, rom, [0xBF; 32], LANE_KAPPA).expect("test-profile plan");

    let structure = plan.circuit().structure().clone();
    let engine_params = config::r1cs_params(structure.n, structure.m).expect("engine params");
    support::install_ajtai_module(&engine_params, &structure);
    let t = Instant::now();
    let prep = preprocess(engine_params, structure, Some(plan.circuit().m_in()))
        .expect("preprocessing")
        .with_nebula(plan.config());
    let t_preprocess = t.elapsed();

    let mut memory = Memory::new(params, plan.rom_image()).expect("memory");
    let mut run = memory.begin_segment().expect("segment");
    for i in 0..(params.steps_per_segment() * params.b_ops) as u64 / 2 {
        let addr = i % params.ram_cells();
        run.write(true, addr, (i + 1) as u32).expect("write");
        assert_eq!(run.read(true, addr).expect("read"), (i + 1) as u32);
    }
    let trace = run.finish().expect("segment close");

    let t = Instant::now();
    let audit = lifecycle::prove(&prep, Vec::<Vec<_>>::new()).expect("base");
    let audit = prove_segment(&prep, &plan, audit, &trace).expect("segment");
    let t_prove = t.elapsed();

    let t = Instant::now();
    let audit = lifecycle::finish_uncompressed_with_audit(&prep, audit).expect("finalize");
    let t_finalize = t.elapsed();

    let t = Instant::now();
    verify_uncompressed_audit(&prep, &audit).expect("verify");
    let t_verify = t.elapsed();

    let n = params.steps_per_segment();
    println!("== Nebula segment timing, §2 test profile (N = {n}) ==");
    println!(
        "F' steps (chunks)        {:>8}   (batched folding: ⌈N / max_fresh⌉ — SuperNeo multi-folding, Theorem 1's K ≤ 61 arity)",
        audit.steps.len()
    );
    println!(
        "S_mem rows / cols        {:>8} / {:>8}",
        plan.circuit().rows(),
        plan.circuit().cols()
    );
    println!("preprocess               {:>10.2?}", t_preprocess);
    println!(
        "prove segment            {:>10.2?}   ({:.2?} per step incl. fold)",
        t_prove,
        t_prove / n as u32
    );
    println!("finalize                 {:>10.2?}", t_finalize);
    println!("audit verify             {:>10.2?}", t_verify);
}
