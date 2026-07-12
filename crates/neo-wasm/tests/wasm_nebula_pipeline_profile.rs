//! Structural and timing census for one folded WASM + Nebula proof.
//!
//! ```text
//! cargo test -p neo-wasm --release --test wasm_nebula_pipeline_profile \
//!   --features perf-timers -- --ignored --nocapture
//! ```

mod common;

use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use neo_ccs::{CcsMatrix, CcsStructure};
#[cfg(feature = "perf-timers")]
use neo_fold_clean::config;
#[cfg(feature = "perf-timers")]
use neo_fold_clean::frontends::nebula::f_prime::{NebulaFPrimeChainBuilder, ROAD_A_COMMITTED_BIT_BUDGET};
use neo_fold_clean::frontends::r1cs_f_prime::R1csShape;
use neo_fold_clean::paper::construction2::ProofState;
use neo_fold_clean::paper::params::Params;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

const PROFILE_WAT: &str = r#"
(module
  (memory 1 1)
  (func (export "main") (result i32)
    i32.const 0
    i32.const 255
    i32.store8
    i32.const 0
    i32.load8_u
    i32.const 6
    i32.const 7
    i32.mul
    i32.add))
"#;

#[derive(Clone, Copy, Debug, Default)]
struct StorageStats {
    explicit_nnz: usize,
    seeded_blocks: usize,
    seeded_slots: u128,
    geometric_runs: usize,
    geometric_slots: usize,
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn matrix_stats(matrix: &CcsMatrix<F>) -> StorageStats {
    match matrix {
        CcsMatrix::Identity { n } => StorageStats {
            explicit_nnz: *n,
            ..StorageStats::default()
        },
        CcsMatrix::Csc(csc) => StorageStats {
            explicit_nnz: csc.vals.len(),
            ..StorageStats::default()
        },
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => StorageStats {
            explicit_nnz: csc.vals.len(),
            seeded_blocks: blocks.len(),
            seeded_slots: blocks
                .iter()
                .map(|block| (D as u128) * (D as u128) * (block.kappa() as u128) * (block.message_cols() as u128))
                .sum(),
            geometric_runs: geometric_runs.len(),
            geometric_slots: geometric_runs.iter().map(|run| run.len()).sum(),
        },
    }
}

fn structure_stats(structure: &CcsStructure<F>) -> StorageStats {
    structure
        .matrices
        .iter()
        .map(matrix_stats)
        .fold(StorageStats::default(), |sum, item| StorageStats {
            explicit_nnz: sum.explicit_nnz + item.explicit_nnz,
            seeded_blocks: sum.seeded_blocks + item.seeded_blocks,
            seeded_slots: sum.seeded_slots + item.seeded_slots,
            geometric_runs: sum.geometric_runs + item.geometric_runs,
            geometric_slots: sum.geometric_slots + item.geometric_slots,
        })
}

fn r1cs_nnz(shape: &R1csShape) -> usize {
    match shape {
        R1csShape::Dense(r1cs) => [&r1cs.a, &r1cs.b, &r1cs.c]
            .into_iter()
            .map(|matrix| {
                matrix
                    .as_slice()
                    .iter()
                    .filter(|&&value| value != F::ZERO)
                    .count()
            })
            .sum(),
        R1csShape::Sparse(r1cs) => [&r1cs.a, &r1cs.b, &r1cs.c]
            .into_iter()
            .map(|matrix| matrix_stats(matrix).explicit_nnz)
            .sum(),
    }
}

fn test_params() -> Params {
    let raw = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        1 << 14,
        neo_params::goldilocks_paper_b2::B_BASE,
        neo_params::goldilocks_paper_b2::K_RHO,
        neo_params::goldilocks_paper_b2::T,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        20,
    )
    .expect("profile SuperNeo parameters");
    Params::test_only_from_neo_params(raw)
}

#[test]
#[ignore = "end-to-end profiling census; run explicitly with --nocapture"]
fn wasm_nebula_pipeline_profile() {
    let wall_started = Instant::now();

    let started = Instant::now();
    let wasm = wat::parse_str(PROFILE_WAT).expect("valid profile WAT");
    let parse_elapsed = started.elapsed();

    let started = Instant::now();
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let artifacts_elapsed = started.elapsed();

    let started = Instant::now();
    let run = neo_wasm::collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    let execute_elapsed = started.elapsed();
    assert_eq!(run.results.as_slice(), &["297".to_string()]);

    let started = Instant::now();
    let trace = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("normalized trace");
    let normalize_elapsed = started.elapsed();

    let started = Instant::now();
    let _witnesses = common::sanity_check_trace(&trace, &artifacts, &run.initial_locals);
    common::ccs_check_trace(&trace);
    let trace_check_elapsed = started.elapsed();

    let profile = neo_wasm::WasmNebulaProfile::test_profile();
    let batch_size = profile.batch_size();
    let steps_per_segment = profile.memory().steps_per_segment();
    let wasm_rows_per_segment = steps_per_segment * batch_size;
    let segment_count = trace.len().div_ceil(wasm_rows_per_segment);
    let folded_steps = segment_count * steps_per_segment;
    let padded_wasm_rows = folded_steps * batch_size;
    let unbatched_folded_steps = trace.len().div_ceil(steps_per_segment) * steps_per_segment;
    assert!(
        folded_steps * 2 <= unbatched_folded_steps,
        "profile batch must remove at least half of the unbatched folds"
    );

    let mut opcode_counts = BTreeMap::<String, usize>::new();
    for row in &trace {
        *opcode_counts
            .entry(format!("{:?}", row.opcode))
            .or_default() += 1;
    }

    let entry_pc = common::single_function_entry_pc(&artifacts);
    let params = test_params();
    let started = Instant::now();
    let prep = neo_wasm::nebula::preprocess_seeded_reduced_memory_test_only(
        params.clone(),
        profile,
        &artifacts,
        &run.initial_locals,
        entry_pc,
        0x57a5_7001,
    )
    .expect("WASM Nebula preprocessing");
    let preprocess_elapsed = started.elapsed();

    let relation = prep.inner().relation();
    let structure = relation.structure();
    let application = relation.application().expect("WASM application relation");
    let app_shape = application.shape();
    let plan = prep.inner().plan();
    let s_mem = plan.circuit();
    let width = relation.low_norm_width_audit();
    let arms = relation.field_arm_shapes();
    let dims = prep
        .inner()
        .prep
        .nifs_v_circuit_config()
        .expect("SplitNc dimensions");
    let final_storage = structure_stats(structure);

    assert_ne!(structure.n, structure.m, "profile must exercise rectangular SplitNc");
    assert_ne!(dims.pi_ccs.ell_n, dims.pi_ccs.ell_m);
    assert!(width.total_coordinates <= structure.m);
    assert!(
        structure.m - width.total_coordinates < D,
        "only D-alignment may trail the width audit"
    );

    let core = neo_wasm::WasmVmSpec::default();
    let core_structure = &core.core_ccs_spec().structure;
    let core_storage = structure_stats(core_structure);
    let app_nnz = r1cs_nnz(app_shape);

    println!("\n== WASM + Nebula folded-proof pipeline census ==");
    println!(
        "profile                  reduced timing profile: kappa={} m={} | production core b={} k_rho={} T={} s={}",
        params.kappa(),
        params.m(),
        params.b(),
        params.k_rho(),
        params.T(),
        params.extension_degree(),
    );
    println!(
        "program                  bytes={} trace={} padded={} batch={} folds={}/{}-unbatched segments={} folds/segment={} opcodes={opcode_counts:?}",
        wasm.len(),
        trace.len(),
        padded_wasm_rows,
        batch_size,
        folded_steps,
        unbatched_folded_steps,
        segment_count,
        steps_per_segment,
    );
    println!("\n-- structural waterfall --");
    println!("stage                         rows        columns      public       matrices degree       explicit-nnz");
    println!(
        "WASM core               {:>10} {:>14} {:>11} {:>14} {:>6} {:>18}",
        core_structure.n,
        core_structure.m,
        core.core_ccs_spec().m_in,
        core_structure.t(),
        core_structure.max_degree(),
        core_storage.explicit_nnz,
    );
    println!(
        "WASM + lookup R1CS      {:>10} {:>14} {:>11} {:>14} {:>6} {:>18}",
        app_shape.n(),
        app_shape.m(),
        app_shape.m_in(),
        3,
        2,
        app_nnz,
    );
    println!(
        "S_mem                   {:>10} {:>14} {:>11} {:>14} {:>6} {:>18}",
        s_mem.rows(),
        s_mem.cols(),
        s_mem.m_in(),
        s_mem.structure().t(),
        s_mem.structure().max_degree(),
        s_mem.nnz(),
    );
    for (name, arm) in ["F' base field", "F' bootstrap field", "F' recursive field"]
        .into_iter()
        .zip(arms)
    {
        println!(
            "{name:<24} {:>10} {:>14} {:>11} {:>14} {:>6} {:>18}",
            arm.rows, arm.columns, arm.public_columns, 3, 2, "-",
        );
        println!("  Poseidon2 permutations: {}", arm.poseidon2_permutations);
    }
    println!(
        "selective CCS           {:>10} {:>14} {:>11} {:>14} {:>6} {:>18}",
        structure.n,
        structure.m,
        relation.public_input_len(),
        structure.t(),
        structure.max_degree(),
        final_storage.explicit_nnz,
    );
    println!(
        "SplitNc dimensions       ell_n={} ell_m={} ell_d={} d_sc={} | row pad={} column pad={}",
        dims.pi_ccs.ell_n,
        dims.pi_ccs.ell_m,
        dims.pi_ccs.ell_d,
        dims.pi_ccs.d_sc,
        (1usize << dims.pi_ccs.ell_n) - structure.n,
        (1usize << dims.pi_ccs.ell_m) - structure.m,
    );
    println!(
        "batched app overhead     rows=+{} columns=+{} lookup_aux={}/step, {}/batch",
        app_shape.n().saturating_sub(batch_size * core_structure.n),
        app_shape.m().saturating_sub(batch_size * core_structure.m),
        prep.lookup_auxiliary_columns_per_instruction(),
        prep.total_lookup_auxiliary_columns(),
    );
    println!(
        "memory layout            regions={} ports={} R={} M={} B_ops={} B_scan={} N={}",
        application.memory().regions().len(),
        application.memory().port_count(),
        profile.memory().rom_cells(),
        profile.memory().ram_cells(),
        profile.memory().b_ops,
        profile.memory().b_scan,
        steps_per_segment,
    );

    println!("\n-- selective committed-width attribution --");
    println!(
        "shared prefix            constant={} public={} selectors={} alignment={} shared_private={} branch_start={} total_before_D_pad={} D_pad={}",
        width.constant_coordinate,
        width.public_coordinates,
        width.selector_coordinates,
        width.alignment_padding,
        width.shared_private_coordinates,
        width.branch_start,
        width.total_coordinates,
        structure.m - width.total_coordinates,
    );
    for (name, arm) in ["base", "bootstrap", "recursive"]
        .into_iter()
        .zip(&width.arms)
    {
        println!(
            "{name:<10} source_fields={:>8} eliminated={:>8} unit={:>8} balanced={:>8} binary={:>8} aliases={:>8} branch_coords={:>10} derived={:>10} total={:>10}",
            arm.branch_source_columns,
            arm.eliminated_columns,
            arm.unit_columns,
            arm.balanced_columns,
            arm.binary_columns,
            arm.decomposition_aliases + arm.equality_aliases,
            arm.branch_coordinates,
            arm.derived_coordinates,
            arm.total_branch_coordinates,
        );
    }

    println!("\n-- recursive F' family touches (inclusive; nested families overlap) --");
    let mut families = width.arms[2].row_families.iter().collect::<Vec<_>>();
    families.sort_by_key(|family| Reverse(family.coordinates_before_aliases + family.poseidon2_coordinates));
    println!(
        "family                                    rows  source-coords  unit  balanced binary  p2-perms    p2-coords"
    );
    for family in families {
        println!(
            "{:<39} {:>8} {:>13} {:>6} {:>9} {:>6} {:>9} {:>12}",
            family.name,
            family.inclusive_rows,
            family.coordinates_before_aliases,
            family.unit_columns,
            family.balanced_columns,
            family.binary_columns,
            family.poseidon2_permutations,
            family.poseidon2_coordinates,
        );
    }

    println!("\n-- final CCS matrix storage --");
    println!("matrix  explicit-nnz  seeded-blocks  virtual-seeded-slots  geometric-runs  virtual-run-slots");
    for (index, matrix) in structure.matrices.iter().enumerate() {
        let stats = matrix_stats(matrix);
        println!(
            "M{index:<5} {:>12} {:>14} {:>21} {:>15} {:>18}",
            stats.explicit_nnz, stats.seeded_blocks, stats.seeded_slots, stats.geometric_runs, stats.geometric_slots,
        );
    }
    println!(
        "TOTAL  {:>12} {:>14} {:>21} {:>15} {:>18}",
        final_storage.explicit_nnz,
        final_storage.seeded_blocks,
        final_storage.seeded_slots,
        final_storage.geometric_runs,
        final_storage.geometric_slots,
    );

    let started = Instant::now();
    let proof = neo_wasm::prove(&prep, &trace).expect("folded WASM proof");
    let prove_elapsed = started.elapsed();

    let started = Instant::now();
    neo_wasm::verify(&prep, &proof, common::final_state(&trace)).expect("terminal-only verification");
    let verify_elapsed = started.elapsed();

    let terminal = proof.inner();
    let (running_claims, latest_instances) = match &terminal.state.proof {
        ProofState::Initial => (0, 0),
        ProofState::Active { running, latest } => (running.claims.len(), latest.instances.len()),
    };
    assert!(
        terminal.final_fold.is_some(),
        "Nebula must consume its delayed terminal claim"
    );
    assert_eq!(terminal.state.step_count as usize, folded_steps);
    assert_eq!(terminal.state.chunk_count as usize, folded_steps);

    println!("\n-- end-to-end timing --");
    println!("parse WAT                  {:>10.2}ms", ms(parse_elapsed));
    println!("program artifacts          {:>10.2}ms", ms(artifacts_elapsed));
    println!("wasmtime execution         {:>10.2}ms", ms(execute_elapsed));
    println!("trace normalization        {:>10.2}ms", ms(normalize_elapsed));
    println!("native + row checks        {:>10.2}ms", ms(trace_check_elapsed));
    println!("fixed-point preprocess     {:>10.2}ms", ms(preprocess_elapsed));
    println!("folded prove               {:>10.2}ms", ms(prove_elapsed));
    println!("terminal verify            {:>10.2}ms", ms(verify_elapsed));
    println!("wall total                 {:>10.2}ms", ms(wall_started.elapsed()));
    println!(
        "proof state               chunks={} steps={} running={} latest={} final_fold={}",
        terminal.state.chunk_count,
        terminal.state.step_count,
        running_claims,
        latest_instances,
        terminal.final_fold.is_some(),
    );
    println!(
        "PROFILE_JSON={{\"trace_steps\":{},\"padded_wasm_steps\":{},\"batch_size\":{},\"folded_steps\":{},\"unbatched_folded_steps\":{},\"segments\":{},\"kappa\":{},\"parameter_m\":{},\"k_rho\":{},\"rows\":{},\"columns\":{},\"matrices\":{},\"ell_n\":{},\"ell_m\":{},\"explicit_nnz\":{},\"seeded_blocks\":{},\"geometric_runs\":{},\"geometric_slots\":{},\"preprocess_ms\":{:.3},\"prove_ms\":{:.3},\"verify_ms\":{:.3},\"total_ms\":{:.3}}}",
        trace.len(),
        padded_wasm_rows,
        batch_size,
        folded_steps,
        unbatched_folded_steps,
        segment_count,
        params.kappa(),
        params.m(),
        params.k_rho(),
        structure.n,
        structure.m,
        structure.t(),
        dims.pi_ccs.ell_n,
        dims.pi_ccs.ell_m,
        final_storage.explicit_nnz,
        final_storage.seeded_blocks,
        final_storage.geometric_runs,
        final_storage.geometric_slots,
        ms(preprocess_elapsed),
        ms(prove_elapsed),
        ms(verify_elapsed),
        ms(wall_started.elapsed()),
    );
}

/// Builds the exact production relation and executes the first occurrence of
/// every F' branch. The prefix is intentionally left open: a complete
/// production memory segment contains 1,088 folds and is a separate endurance
/// benchmark, not a prerequisite for measuring one real steady-state fold.
#[test]
#[cfg(feature = "perf-timers")]
#[ignore = "production kappa=18 fixed point plus three real folds; run explicitly"]
fn wasm_nebula_production_prefix_profile() {
    const PREFIX_FOLDS: usize = 3;

    let wall_started = Instant::now();
    let wasm = wat::parse_str(PROFILE_WAT).expect("valid profile WAT");
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = neo_wasm::collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    assert_eq!(run.results.as_slice(), &["297".to_string()]);
    let trace = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("normalized trace");
    let _witnesses = common::sanity_check_trace(&trace, &artifacts, &run.initial_locals);
    common::ccs_check_trace(&trace);

    let profile = neo_wasm::WasmNebulaProfile::production();
    let params = Params::for_ccs_shape_with(
        ROAD_A_COMMITTED_BIT_BUDGET,
        13,
        8,
        config::MIN_EFFECTIVE_LAMBDA,
        config::EXTENSION_SAFETY_MARGIN_BITS,
    )
    .expect("production WASM parameters");
    assert_eq!(params.kappa(), 18);
    assert_eq!(params.k_rho(), 14);
    assert_eq!(params.b(), 2);
    assert_eq!(params.big_b(), 1 << 14);
    assert_eq!(params.T(), 216);
    assert_eq!(profile.batch_size(), 3);
    assert_eq!(profile.memory().rom_cells(), 4_096);
    assert_eq!(profile.memory().ram_cells(), 65_536);
    assert_eq!(profile.memory().steps_per_segment(), 1_088);

    let entry_pc = common::single_function_entry_pc(&artifacts);
    let started = Instant::now();
    let prep = neo_wasm::nebula::preprocess_seeded_unbounded_profile(
        params.clone(),
        profile,
        &artifacts,
        &run.initial_locals,
        entry_pc,
        0x57a5_7018,
    )
    .expect("unbounded production WASM + Nebula preprocessing");
    let preprocess_elapsed = started.elapsed();

    let relation = prep.inner().relation();
    let structure = relation.structure();
    let width = relation.low_norm_width_audit();
    let arms = relation.field_arm_shapes();
    let storage = structure_stats(structure);
    let dims = prep
        .inner()
        .prep
        .nifs_v_circuit_config()
        .expect("SplitNc dimensions");
    assert_ne!(structure.n, structure.m, "production SplitNc must remain rectangular");
    assert_eq!(structure.t(), 13);
    assert_eq!(structure.max_degree(), 8);
    assert_eq!(width.total_coordinates.div_ceil(D) * D, structure.m);

    println!("\n== WASM + Nebula exact production prefix profile ==");
    println!(
        "parameters               kappa={} k_rho={} b={} B={} T={} lambda={} s={}",
        params.kappa(),
        params.k_rho(),
        params.b(),
        params.big_b(),
        params.T(),
        params.lambda(),
        params.extension_degree(),
    );
    println!(
        "memory geometry          R={} M={} B_ops={} B_scan={} N={} batch={}",
        profile.memory().rom_cells(),
        profile.memory().ram_cells(),
        profile.memory().b_ops,
        profile.memory().b_scan,
        profile.memory().steps_per_segment(),
        profile.batch_size(),
    );
    println!(
        "relation                 rows={} columns={} public={} matrices={} degree={} budget={} over_budget={}",
        structure.n,
        structure.m,
        relation.public_input_len(),
        structure.t(),
        structure.max_degree(),
        ROAD_A_COMMITTED_BIT_BUDGET,
        structure.m.saturating_sub(ROAD_A_COMMITTED_BIT_BUDGET),
    );
    println!(
        "SplitNc                  ell_n={} ell_m={} ell_d={} d_sc={} row_pad={} column_pad={}",
        dims.pi_ccs.ell_n,
        dims.pi_ccs.ell_m,
        dims.pi_ccs.ell_d,
        dims.pi_ccs.d_sc,
        (1usize << dims.pi_ccs.ell_n) - structure.n,
        (1usize << dims.pi_ccs.ell_m) - structure.m,
    );
    println!(
        "matrix storage           explicit_nnz={} seeded_blocks={} virtual_seeded_slots={} geometric_runs={} virtual_run_slots={}",
        storage.explicit_nnz,
        storage.seeded_blocks,
        storage.seeded_slots,
        storage.geometric_runs,
        storage.geometric_slots,
    );
    println!(
        "shared width             prefix={} branch_start={} audited={} D_pad={}",
        width.constant_coordinate
            + width.public_coordinates
            + width.selector_coordinates
            + width.alignment_padding
            + width.shared_private_coordinates,
        width.branch_start,
        width.total_coordinates,
        structure.m - width.total_coordinates,
    );
    for (name, arm) in ["base", "bootstrap", "recursive"]
        .into_iter()
        .zip(&width.arms)
    {
        println!(
            "{name:<24} source_fields={} eliminated={} unit={} balanced={} binary={} aliases={} branch_coords={} derived={} total={}",
            arm.branch_source_columns,
            arm.eliminated_columns,
            arm.unit_columns,
            arm.balanced_columns,
            arm.binary_columns,
            arm.decomposition_aliases + arm.equality_aliases,
            arm.branch_coordinates,
            arm.derived_coordinates,
            arm.total_branch_coordinates,
        );
    }
    for (name, arm) in ["base field", "bootstrap field", "recursive field"]
        .into_iter()
        .zip(arms)
    {
        println!(
            "{name:<24} rows={} columns={} public={} poseidon2={}",
            arm.rows, arm.columns, arm.public_columns, arm.poseidon2_permutations,
        );
    }

    let mut families = width.arms[2].row_families.iter().collect::<Vec<_>>();
    families.sort_by_key(|family| Reverse(family.coordinates_before_aliases + family.poseidon2_coordinates));
    println!("\n-- recursive F' family touches (inclusive; overlaps expected) --");
    println!(
        "family                                    rows  source-coords  unit  balanced binary  p2-perms    p2-coords"
    );
    for family in families {
        println!(
            "{:<39} {:>8} {:>13} {:>6} {:>9} {:>6} {:>9} {:>12}",
            family.name,
            family.inclusive_rows,
            family.coordinates_before_aliases,
            family.unit_columns,
            family.balanced_columns,
            family.binary_columns,
            family.poseidon2_permutations,
            family.poseidon2_coordinates,
        );
    }

    let started = Instant::now();
    let segment = neo_wasm::nebula::build_application_segment_for_profile(&prep, &trace)
        .expect("full production application segment");
    let segment_build_elapsed = started.elapsed();

    let started = Instant::now();
    let mut chain = NebulaFPrimeChainBuilder::new(prep.inner());
    chain
        .append_application_prefix_for_profile(&segment, PREFIX_FOLDS)
        .expect("base, bootstrap, and steady production folds");
    let prefix_elapsed = started.elapsed();
    let audit = chain.into_audit().expect("nonempty production prefix");
    assert_eq!(audit.proof.state.step_count as usize, PREFIX_FOLDS);
    assert_eq!(audit.proof.state.chunk_count as usize, PREFIX_FOLDS);
    assert_eq!(audit.steps.len(), PREFIX_FOLDS);
    assert!(audit.proof.final_fold.is_none());

    let segment_steps = profile.memory().steps_per_segment();
    let naive_segment_projection = prefix_elapsed.mul_f64(segment_steps as f64 / PREFIX_FOLDS as f64);
    println!("\n-- exact production timing --");
    println!("fixed-point preprocess     {:>12.2}ms", ms(preprocess_elapsed));
    println!("segment witness build      {:>12.2}ms", ms(segment_build_elapsed));
    println!("three-fold prefix          {:>12.2}ms", ms(prefix_elapsed));
    println!(
        "mean observed fold        {:>12.2}ms",
        ms(prefix_elapsed) / PREFIX_FOLDS as f64
    );
    println!(
        "naive 1,088-fold projection {:>10.2}ms ({:.2}h)",
        ms(naive_segment_projection),
        naive_segment_projection.as_secs_f64() / 3_600.0,
    );
    println!("wall total                 {:>12.2}ms", ms(wall_started.elapsed()));
    println!(
        "PROFILE_PRODUCTION_JSON={{\"trace_steps\":{},\"padded_wasm_steps\":{},\"batch_size\":{},\"prefix_folds\":{},\"segment_folds\":{},\"kappa\":{},\"k_rho\":{},\"lambda\":{},\"rows\":{},\"columns\":{},\"matrices\":{},\"ell_n\":{},\"ell_m\":{},\"explicit_nnz\":{},\"seeded_blocks\":{},\"geometric_runs\":{},\"preprocess_ms\":{:.3},\"segment_build_ms\":{:.3},\"prefix_ms\":{:.3},\"naive_segment_ms\":{:.3},\"wall_ms\":{:.3}}}",
        trace.len(),
        segment_steps * profile.batch_size(),
        profile.batch_size(),
        PREFIX_FOLDS,
        segment_steps,
        params.kappa(),
        params.k_rho(),
        params.lambda(),
        structure.n,
        structure.m,
        structure.t(),
        dims.pi_ccs.ell_n,
        dims.pi_ccs.ell_m,
        storage.explicit_nnz,
        storage.seeded_blocks,
        storage.geometric_runs,
        ms(preprocess_elapsed),
        ms(segment_build_elapsed),
        ms(prefix_elapsed),
        ms(naive_segment_projection),
        ms(wall_started.elapsed()),
    );
}
