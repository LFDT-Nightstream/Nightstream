//! Structural and timing census for one folded WASM + Nebula proof.
//!
//! ```text
//! cargo test -p neo-wasm --release --test wasm_nebula_pipeline_profile \
//!   --features perf-timers -- --ignored --nocapture
//! ```

#[cfg(feature = "perf-timers")]
#[path = "wasm_nebula_pipeline_profile/centered_domain_artifact.rs"]
mod centered_domain_artifact;
mod common;
#[cfg(feature = "perf-timers")]
#[path = "wasm_nebula_pipeline_profile/radix_four.rs"]
mod radix_four;

#[cfg(feature = "perf-timers")]
use std::cmp::Reverse;
#[cfg(feature = "perf-timers")]
use std::collections::BTreeMap;
#[cfg(feature = "perf-timers")]
use std::fs::File;
#[cfg(feature = "perf-timers")]
use std::io::{BufReader, BufWriter, Write};
use std::time::{Duration, Instant};

use neo_ccs::{CcsMatrix, CcsStructure};
#[cfg(feature = "perf-timers")]
use neo_fold_clean::config;
#[cfg(feature = "perf-timers")]
use neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimeChainBuilder;
use neo_fold_clean::frontends::r1cs_f_prime::R1csShape;
#[cfg(feature = "perf-timers")]
use neo_fold_clean::paper::construction2::ProofState;
use neo_fold_clean::paper::params::Params;
use neo_math::{D, F};
#[cfg(all(feature = "perf-timers", feature = "metal", target_vendor = "apple"))]
use neo_prover_metal::MetalNifsProver;
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

const PROFILE_WAT_VARIANT: &str = r#"
(module
  (memory 0 1)
  (func (export "main") (result i32)
    i32.const 42))
"#;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct StorageStats {
    explicit_nnz: usize,
    seeded_blocks: usize,
    seeded_slots: u128,
    geometric_runs: usize,
    geometric_slots: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RelationStructureStats {
    application_constraints: usize,
    application_columns: usize,
    application_nnz: usize,
    s_mem_constraints: usize,
    s_mem_assignment_bits: usize,
    s_mem_public_bits: usize,
    s_mem_private_bits: usize,
    s_mem_nnz: usize,
    logical_ports: usize,
    routed_slots: usize,
    b_ops: usize,
    final_constraints: usize,
    final_columns: usize,
    final_committed_coordinates: usize,
    final_explicit_nnz: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReusableRelationReceipt {
    signature: (usize, usize, usize, u32),
    structure_digest: [F; 4],
    pi_ccs_header_bundle: [F; 4],
    ajtai_pp_digest: [F; 4],
    initial_semantic_state_digest: [F; 4],
    program_binding_digest: [F; 4],
    plan_digest: [F; 4],
    d_init: [F; 4],
}

#[cfg(feature = "perf-timers")]
fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(feature = "perf-timers")]
fn performance_profile(batch_size: usize) -> neo_wasm::WasmNebulaProfile {
    // These limits define this benchmark fixture. The library does not select
    // application limits or an instruction batch for callers.
    let rom = neo_wasm::WasmNebulaRomLimits::new(64, 2, 2, 2, 2, 2).expect("WASM performance fixture ROM limits");
    let limits = neo_wasm::nebula::WasmNebulaLimits::new(4096, 256, 256, 16, 32768, 64, 8, 256, rom)
        .expect("WASM performance fixture limits");
    neo_wasm::WasmNebulaProfile::production(limits, batch_size).expect("WASM performance fixture profile")
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
        CcsMatrix::VerifierArtifact { .. } => StorageStats::default(),
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
        // This one-step timing schedule requires the 2^25 joint domain.
        1 << 25,
        neo_params::goldilocks_paper_b2::B_BASE,
        neo_params::goldilocks_paper_b2::K_RHO,
        neo_params::goldilocks_paper_b2::T,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        20,
    )
    .expect("profile SuperNeo parameters");
    Params::test_only_from_neo_params(raw)
}

fn reusable_relation_receipt(prep: &neo_wasm::WasmNebulaPreprocessing) -> ReusableRelationReceipt {
    let structure = prep.inner().relation().structure();
    ReusableRelationReceipt {
        signature: (structure.n, structure.m, structure.t(), structure.max_degree()),
        structure_digest: *prep.inner().prep.structure_digest(),
        pi_ccs_header_bundle: prep.inner().prep.pi_ccs_header_bundle(),
        ajtai_pp_digest: prep.inner().prep.ajtai_pp_digest(),
        initial_semantic_state_digest: prep
            .inner()
            .relation()
            .nebula_config()
            .initial_semantic_state_digest,
        program_binding_digest: prep
            .inner()
            .relation()
            .nebula_config()
            .program_binding_digest(),
        plan_digest: prep.inner().plan().plan_digest(),
        d_init: prep.inner().plan().d_init(),
    }
}

fn collect_relation_structure_census(
    profile: neo_wasm::WasmNebulaProfile,
    seed: u64,
    label: &str,
) -> RelationStructureStats {
    let wasm = wat::parse_str(PROFILE_WAT).expect("valid profile WAT");
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = neo_wasm::collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    let entry_pc = common::single_function_entry_pc(&artifacts);
    let prep = neo_wasm::nebula::preprocess_seeded_reduced_memory_test_only(
        test_params(),
        profile,
        &artifacts,
        &run.initial_locals,
        entry_pc,
        seed,
    )
    .expect("WASM Nebula structural preprocessing");

    let relation = prep.inner().relation();
    let structure = relation.structure();
    let application = relation.application().expect("WASM application relation");
    let app_shape = application.shape();
    let s_mem = prep.inner().plan().circuit();
    let width = relation
        .low_norm_width_audit()
        .expect("live relation has a width audit");
    let storage = structure_stats(structure);
    let final_committed_coordinates = structure.m - width.constant_coordinate;
    let s_mem_assignment_bits = s_mem.cols() - 1;
    let stats = RelationStructureStats {
        application_constraints: app_shape.n(),
        application_columns: app_shape.m(),
        application_nnz: r1cs_nnz(app_shape),
        s_mem_constraints: s_mem.rows(),
        s_mem_assignment_bits,
        s_mem_public_bits: s_mem.m_in() - 1,
        s_mem_private_bits: s_mem.cols() - s_mem.m_in(),
        s_mem_nnz: s_mem.nnz(),
        logical_ports: application.memory().logical_port_count(),
        routed_slots: application.memory().slot_count(),
        b_ops: profile.memory().b_ops,
        final_constraints: structure.n,
        final_columns: structure.m,
        final_committed_coordinates,
        final_explicit_nnz: storage.explicit_nnz,
    };

    println!("== WASM + Nebula structural census ({label}) ==");
    println!(
        "application R1CS         constraints={} columns={} nnz={}",
        stats.application_constraints, stats.application_columns, stats.application_nnz,
    );
    println!(
        "S_mem                   constraints={} assignment_bits={} public_bits={} private_bits={} nnz={}",
        stats.s_mem_constraints,
        stats.s_mem_assignment_bits,
        stats.s_mem_public_bits,
        stats.s_mem_private_bits,
        stats.s_mem_nnz,
    );
    println!(
        "memory routing           logical_ports={} routed_slots={} B_ops={}",
        stats.logical_ports, stats.routed_slots, stats.b_ops,
    );
    println!(
        "final selective CCS      constraints={} columns={} committed_coordinates={} explicit_nnz={}",
        stats.final_constraints, stats.final_columns, stats.final_committed_coordinates, stats.final_explicit_nnz,
    );
    println!(
        "compact matrix storage   seeded_blocks={} virtual_seeded_slots={} geometric_runs={} virtual_run_slots={}",
        storage.seeded_blocks, storage.seeded_slots, storage.geometric_runs, storage.geometric_slots,
    );
    println!("memory-related recursive-arm families (inclusive; ranges may overlap):");
    for family in width.arms[1].row_families.iter().filter(|family| {
        family.name.starts_with("nebula.application.s_mem")
            || family.name.starts_with("nebula.application.memory_ports")
    }) {
        println!(
            "  {:<42} rows={} source_coordinates={} poseidon2_coordinates={}",
            family.name, family.inclusive_rows, family.coordinates_before_aliases, family.poseidon2_coordinates,
        );
    }

    assert_eq!(stats.logical_ports, 76 * profile.batch_size());
    assert!(stats.routed_slots <= stats.b_ops);
    assert_eq!(width.total_coordinates.div_ceil(D) * D, structure.m);
    assert_eq!(
        stats.s_mem_assignment_bits,
        stats.s_mem_public_bits + stats.s_mem_private_bits
    );

    stats
}

#[test]
#[ignore = "full F-prime structural census; builds preprocessing but does not prove"]
fn wasm_nebula_relation_structure_census() {
    let stats = collect_relation_structure_census(
        neo_wasm::WasmNebulaProfile::test_profile(),
        0x57a5_7019,
        "reduced test profile, compact geometry",
    );

    assert_eq!(stats.routed_slots, stats.b_ops);
    assert_eq!(
        (
            stats.application_constraints,
            stats.application_columns,
            stats.application_nnz,
        ),
        (51_329, 23_505, 211_854),
        "application R1CS structure changed; review the structural census",
    );
    assert_eq!(
        (
            stats.s_mem_constraints,
            stats.s_mem_assignment_bits,
            stats.s_mem_public_bits,
            stats.s_mem_private_bits,
            stats.s_mem_nnz,
        ),
        (449_870, 446_283, 1_403, 444_880, 2_824_724),
        "reduced-profile S_mem structure changed; review the memory-overhead census",
    );
    assert!(
        stats.final_constraints < 36_874_004,
        "21-slot routing should use fewer final constraints than the previous 58-slot route",
    );
    assert!(
        stats.final_committed_coordinates < 29_662_631,
        "21-slot routing should commit fewer coordinates than the previous 58-slot route",
    );
}

#[test]
#[ignore = "compiles one full F-prime profile and times two cheap program bindings"]
fn same_profile_programs_reuse_relation_but_not_program_binding() {
    let profile = neo_wasm::WasmNebulaProfile::test_profile();
    let first_wasm = wat::parse_str(PROFILE_WAT).expect("valid first profile WAT");
    let first_artifacts = neo_wasm::extract_wasm_program_artifacts(&first_wasm).expect("first program artifacts");
    let first_run = neo_wasm::collect_wasmtime_steps(&first_wasm, "main", &[]).expect("first wasmtime trace");
    let first_entry_pc = common::single_function_entry_pc(&first_artifacts);
    let second_wasm = wat::parse_str(PROFILE_WAT_VARIANT).expect("valid second profile WAT");
    let second_artifacts = neo_wasm::extract_wasm_program_artifacts(&second_wasm).expect("second program artifacts");
    let second_run = neo_wasm::collect_wasmtime_steps(&second_wasm, "main", &[]).expect("second wasmtime trace");
    let second_entry_pc = common::single_function_entry_pc(&second_artifacts);

    let compile_started = Instant::now();
    let prepared = neo_wasm::nebula::prepare_profile_seeded_reduced_memory_test_only(
        test_params(),
        profile,
        &first_artifacts,
        &first_run.initial_locals,
        first_entry_pc,
        0x57a5_7019,
    )
    .expect("prepared WASM Nebula profile");
    let compile_elapsed = compile_started.elapsed();

    let first_bind_started = Instant::now();
    let first_prep = prepared
        .bind_program_reduced_memory_test_only(&first_artifacts, &first_run.initial_locals, first_entry_pc)
        .expect("first program binding");
    let first_bind_elapsed = first_bind_started.elapsed();
    let second_bind_started = Instant::now();
    let second_prep = prepared
        .bind_program_reduced_memory_test_only(&second_artifacts, &second_run.initial_locals, second_entry_pc)
        .expect("second program binding");
    let second_bind_elapsed = second_bind_started.elapsed();

    assert!(
        first_prep
            .inner()
            .shares_prepared_storage_with(second_prep.inner()),
        "program bindings must share relation matrices and evaluator storage",
    );
    assert!(
        first_bind_elapsed < Duration::from_millis(500),
        "first program binding took {first_bind_elapsed:?}",
    );
    assert!(
        second_bind_elapsed < Duration::from_millis(500),
        "second program binding took {second_bind_elapsed:?}",
    );

    let first = reusable_relation_receipt(&first_prep);
    let second = reusable_relation_receipt(&second_prep);

    assert_eq!(first.signature, second.signature);
    assert_eq!(first.structure_digest, second.structure_digest);
    assert_eq!(first.pi_ccs_header_bundle, second.pi_ccs_header_bundle);
    assert_eq!(first.ajtai_pp_digest, second.ajtai_pp_digest);
    assert_ne!(
        first.initial_semantic_state_digest,
        second.initial_semantic_state_digest
    );
    assert_ne!(first.plan_digest, second.plan_digest);
    assert_ne!(first.d_init, second.d_init);
    assert_ne!(first.program_binding_digest, second.program_binding_digest);
    println!(
        "PREPARED_PROFILE_JSON={{\"compile_ms\":{:.3},\"first_bind_ms\":{:.3},\"second_bind_ms\":{:.3}}}",
        compile_elapsed.as_secs_f64() * 1_000.0,
        first_bind_elapsed.as_secs_f64() * 1_000.0,
        second_bind_elapsed.as_secs_f64() * 1_000.0,
    );

    #[cfg(all(feature = "perf-timers", feature = "metal", target_vendor = "apple"))]
    {
        let trace = neo_wasm::traces_from_wasmtime_steps(&first_run.steps).expect("first normalized trace");
        let final_state = trace.last().expect("nonempty first trace").state_after;
        let mut prover = MetalNifsProver::new().expect("Metal prover");
        prover
            .prepare_static(
                &first_prep.inner().prep.log,
                first_prep.inner().relation().structure(),
                first_prep.inner().prep.optimized_cache(),
                first_prep
                    .inner()
                    .prep
                    .nebula()
                    .map(|config| &config.scheme),
            )
            .expect("prepare shared Metal profile");
        let proof =
            neo_wasm::nebula::prove_with_nifs_adapter(&first_prep, &mut prover, &trace).expect("first program proof");
        neo_wasm::nebula::verify_with_witness_opening_backend(&first_prep, &proof, final_state, &mut prover)
            .expect("first program verification");
        assert!(
            neo_wasm::nebula::verify_with_witness_opening_backend(&second_prep, &proof, final_state, &mut prover,)
                .is_err(),
            "a proof bound to the first program must fail under the second program",
        );
    }
}

#[test]
#[cfg(all(feature = "perf-timers", feature = "metal", target_vendor = "apple"))]
#[ignore = "full cached Metal folded-proof profile; run explicitly"]
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

    // One application step owns the complete ten-instruction trace and the
    // complete reduced-memory scan. The proof still performs its required
    // delayed terminal NIFS fold.
    let memory = neo_fold_clean::frontends::nebula::layout::NebulaParams::new(11, 11, 64, 2048, 16)
        .expect("one-step reduced Nebula scan");
    let profile = neo_wasm::WasmNebulaProfile::test_profile_with_schedule(memory, trace.len());
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
    let width = relation
        .low_norm_width_audit()
        .expect("live relation has a width audit");
    let arms = relation.field_arm_shapes();
    let padded_rows = structure.n.max(structure.m).next_power_of_two();
    let ell = padded_rows.ilog2();
    let final_storage = structure_stats(structure);

    assert_ne!(structure.n, structure.m, "profile must exercise a rectangular relation");
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
        "padded one-SumCheck      ell={} | row pad={} column pad={}",
        ell,
        padded_rows - structure.n,
        padded_rows - structure.m,
    );
    println!(
        "batched app overhead     rows=+{} columns=+{} lookup_aux={}/step, {}/batch",
        app_shape.n().saturating_sub(batch_size * core_structure.n),
        app_shape.m().saturating_sub(batch_size * core_structure.m),
        prep.lookup_auxiliary_columns_per_instruction(),
        prep.total_lookup_auxiliary_columns(),
    );
    println!(
        "memory layout            regions={} logical_ports={} physical_slots={} R={} M={} B_ops={} B_scan={} N={}",
        application.memory().regions().len(),
        application.memory().logical_port_count(),
        application.memory().slot_count(),
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
    for (name, arm) in ["base", "recursive"].into_iter().zip(&width.arms) {
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
    let mut families = width.arms[1].row_families.iter().collect::<Vec<_>>();
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

    let mut prover = MetalNifsProver::new().expect("Metal prover");
    let metal_device = prover
        .session()
        .device_info()
        .expect("Metal device information");
    let started = Instant::now();
    prover
        .prepare_static(
            &prep.inner().prep.log,
            structure,
            prep.inner().prep.optimized_cache(),
            prep.inner().prep.nebula().map(|config| &config.scheme),
        )
        .expect("prepare static Metal proof state");
    let metal_prepare_elapsed = started.elapsed();
    prover.session().reset_activity();

    let started = Instant::now();
    let proof =
        neo_wasm::nebula::prove_with_nifs_adapter(&prep, &mut prover, &trace).expect("folded WASM proof on Metal");
    let prove_elapsed = started.elapsed();
    let proof_metal_activity = prover.session().activity();
    assert!(
        proof_metal_activity.dispatches > 0,
        "profile proof must dispatch Metal kernels"
    );
    assert!(
        proof_metal_activity.host_waits > 0,
        "profile proof must wait for Metal results"
    );

    let started = Instant::now();
    neo_wasm::nebula::verify_with_witness_opening_backend(&prep, &proof, common::final_state(&trace), &mut prover)
        .expect("terminal-only verification with Metal openings");
    let verify_elapsed = started.elapsed();
    let metal_activity = prover.session().activity();
    assert!(
        metal_activity.dispatches > proof_metal_activity.dispatches,
        "profile verification must dispatch Metal opening kernels"
    );

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
    println!("Metal static preparation   {:>10.2}ms", ms(metal_prepare_elapsed));
    println!("folded prove               {:>10.2}ms", ms(prove_elapsed));
    println!("terminal verify            {:>10.2}ms", ms(verify_elapsed));
    println!("wall total                 {:>10.2}ms", ms(wall_started.elapsed()));
    println!(
        "Metal backend             device={:?} dispatches={} waits={} uploaded={} downloaded={}",
        metal_device.name,
        metal_activity.dispatches,
        metal_activity.host_waits,
        metal_activity.uploaded_bytes,
        metal_activity.downloaded_bytes,
    );
    println!(
        "proof state               chunks={} steps={} running={} latest={} final_fold={}",
        terminal.state.chunk_count,
        terminal.state.step_count,
        running_claims,
        latest_instances,
        terminal.final_fold.is_some(),
    );
    println!(
        "PROFILE_JSON={{\"trace_steps\":{},\"padded_wasm_steps\":{},\"batch_size\":{},\"folded_steps\":{},\"unbatched_folded_steps\":{},\"segments\":{},\"kappa\":{},\"parameter_m\":{},\"k_rho\":{},\"rows\":{},\"columns\":{},\"matrices\":{},\"ell\":{},\"explicit_nnz\":{},\"seeded_blocks\":{},\"geometric_runs\":{},\"geometric_slots\":{},\"preprocess_ms\":{:.3},\"metal_prepare_ms\":{:.3},\"prove_ms\":{:.3},\"verify_ms\":{:.3},\"total_ms\":{:.3}}}",
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
        ell,
        final_storage.explicit_nnz,
        final_storage.seeded_blocks,
        final_storage.geometric_runs,
        final_storage.geometric_slots,
        ms(preprocess_elapsed),
        ms(metal_prepare_elapsed),
        ms(prove_elapsed),
        ms(verify_elapsed),
        ms(wall_started.elapsed()),
    );
}

#[test]
#[cfg(feature = "perf-timers")]
#[ignore = "builds, writes, and reloads the full compact evaluator cache"]
fn wasm_nebula_compact_cache_artifact_profile() {
    let wasm = wat::parse_str(PROFILE_WAT).expect("valid profile WAT");
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = neo_wasm::collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    let trace = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("normalized trace");
    let second_wasm = wat::parse_str(PROFILE_WAT_VARIANT).expect("valid second profile WAT");
    let second_artifacts = neo_wasm::extract_wasm_program_artifacts(&second_wasm).expect("second program artifacts");
    let second_run = neo_wasm::collect_wasmtime_steps(&second_wasm, "main", &[]).expect("second wasmtime trace");
    let memory = neo_fold_clean::frontends::nebula::layout::NebulaParams::new(11, 11, 64, 2048, 16)
        .expect("one-step reduced Nebula scan");
    let profile = neo_wasm::WasmNebulaProfile::test_profile_with_schedule(memory, trace.len());
    let entry_pc = common::single_function_entry_pc(&artifacts);

    let started = Instant::now();
    let prep = neo_wasm::nebula::preprocess_seeded_reduced_memory_test_only(
        test_params(),
        profile,
        &artifacts,
        &run.initial_locals,
        entry_pc,
        0x57a5_7001,
    )
    .expect("WASM Nebula preprocessing");
    let preprocess_elapsed = started.elapsed();
    let structure = prep.inner().relation().structure();
    let cache = prep.inner().prep.optimized_cache().superneo();
    let matrix_digest = prep.inner().prep.pi_ccs_header_bundle();

    let cache_artifact_path =
        std::env::temp_dir().join(format!("nightstream-superneo-cache-{}.bin", std::process::id()));
    let started = Instant::now();
    let file = File::create(&cache_artifact_path).expect("create compact cache artifact");
    let mut writer = BufWriter::new(file);
    let cache_receipt = cache
        .write_artifact(&mut writer, matrix_digest)
        .expect("write compact cache artifact");
    writer.flush().expect("flush compact cache artifact");
    drop(writer);
    let cache_write_elapsed = started.elapsed();

    let encoder_artifact_path =
        std::env::temp_dir().join(format!("nightstream-fprime-encoder-{}.bin", std::process::id()));
    let started = Instant::now();
    let file = File::create(&encoder_artifact_path).expect("create encoder artifact");
    let mut writer = BufWriter::new(file);
    let encoder_receipt = prep
        .inner()
        .write_encoder_artifact(&mut writer)
        .expect("write encoder artifact");
    writer.flush().expect("flush encoder artifact");
    drop(writer);
    let encoder_write_elapsed = started.elapsed();

    let limits = neo_reductions::superneo_eval::SuperneoCacheArtifactLimits::new(
        cache_receipt.artifact_bytes(),
        structure.n,
        structure.m.div_ceil(D) * D,
        structure.t(),
    );
    let encoder_limits = neo_fold_clean::frontends::r1cs_f_prime::LowNormEncoderArtifactLimits::new(
        encoder_receipt.encoder().artifact_bytes(),
        structure.n,
        structure.m,
        structure.t(),
        2,
        encoder_receipt
            .encoder()
            .arm_field_counts()
            .iter()
            .copied()
            .max()
            .expect("encoder arm fields") as usize,
        encoder_receipt
            .encoder()
            .arm_derived_counts()
            .iter()
            .copied()
            .max()
            .expect("encoder derived counts") as usize,
    );
    let started = Instant::now();
    let ((loaded, cache_load_elapsed), (loaded_encoder, encoder_load_elapsed)) = std::thread::scope(|scope| {
        let cache = scope.spawn(|| {
            let started = Instant::now();
            let file = File::open(&cache_artifact_path).expect("open compact cache artifact");
            let loaded = neo_reductions::superneo_eval::SuperneoEvalCache::read_verified_artifact(
                BufReader::new(file),
                &cache_receipt,
                limits,
            )
            .expect("load compact cache artifact");
            (loaded, started.elapsed())
        });
        let encoder = scope.spawn(|| {
            let started = Instant::now();
            let file = File::open(&encoder_artifact_path).expect("open encoder artifact");
            let loaded = neo_fold_clean::frontends::nebula::f_prime::VerifiedNebulaFPrimeEncoderArtifact::read(
                BufReader::new(file),
                &encoder_receipt,
                encoder_limits,
            )
            .expect("load encoder artifact");
            (loaded, started.elapsed())
        });
        (
            cache.join().expect("cache artifact loader"),
            encoder.join().expect("encoder artifact loader"),
        )
    });
    let parallel_load_elapsed = started.elapsed();
    assert_eq!(loaded.cache().matrix_caches().len(), structure.t());
    std::fs::remove_file(&cache_artifact_path).expect("remove compact cache artifact");
    std::fs::remove_file(&encoder_artifact_path).expect("remove encoder artifact");

    let started = Instant::now();
    let prepared = neo_wasm::nebula::prepare_profile_seeded_reduced_memory_with_artifacts_test_only(
        test_params(),
        profile,
        &artifacts,
        &run.initial_locals,
        entry_pc,
        0x57a5_7001,
        loaded,
        loaded_encoder,
    )
    .expect("prepare WASM Nebula profile from persistent artifacts");
    let artifact_prepare_elapsed = started.elapsed();
    assert!(
        prepared
            .inner()
            .relation()
            .structure()
            .is_verifier_artifact_header(),
        "artifact-backed profile must not rebuild raw matrices",
    );
    let started = Instant::now();
    let rebound = prepared
        .bind_program_reduced_memory_test_only(&artifacts, &run.initial_locals, entry_pc)
        .expect("bind program to artifact-backed profile");
    let bind_elapsed = started.elapsed();
    let second_entry_pc = common::single_function_entry_pc(&second_artifacts);
    let started = Instant::now();
    let second_rebound = prepared
        .bind_program_reduced_memory_test_only(&second_artifacts, &second_run.initial_locals, second_entry_pc)
        .expect("bind second program to artifact-backed profile");
    let second_bind_elapsed = started.elapsed();
    assert!(
        rebound
            .inner()
            .shares_prepared_storage_with(second_rebound.inner()),
        "artifact-backed program bindings must share relation and cache storage",
    );
    assert_eq!(
        rebound.inner().prep.pi_ccs_header_bundle(),
        prep.inner().prep.pi_ccs_header_bundle(),
        "artifact-backed profile matrix authority",
    );
    assert_eq!(
        rebound.inner().relation().structure().n,
        structure.n,
        "artifact-backed relation rows",
    );
    assert_eq!(
        rebound.inner().relation().structure().m,
        structure.m,
        "artifact-backed relation columns",
    );

    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    {
        let mut prover = MetalNifsProver::new().expect("Metal prover");
        let started = Instant::now();
        prover
            .prepare_static(
                &rebound.inner().prep.log,
                rebound.inner().relation().structure(),
                rebound.inner().prep.optimized_cache(),
                rebound.inner().prep.nebula().map(|config| &config.scheme),
            )
            .expect("prepare artifact-backed Metal proof state");
        let metal_prepare_elapsed = started.elapsed();
        prover.session().reset_activity();

        let started = Instant::now();
        let proof = neo_wasm::nebula::prove_with_nifs_adapter(&rebound, &mut prover, &trace)
            .expect("artifact-backed Metal proof");
        let prove_elapsed = started.elapsed();
        assert!(prover.session().activity().dispatches > 0);

        let started = Instant::now();
        neo_wasm::nebula::verify_with_witness_opening_backend(
            &rebound,
            &proof,
            common::final_state(&trace),
            &mut prover,
        )
        .expect("artifact-backed Metal verification");
        let verify_elapsed = started.elapsed();
        assert!(
            neo_wasm::nebula::verify_with_witness_opening_backend(
                &second_rebound,
                &proof,
                common::final_state(&trace),
                &mut prover,
            )
            .is_err(),
            "an artifact-backed proof must fail under another program binding",
        );
        println!(
            "CACHE_ARTIFACT_METAL_JSON={{\"metal_prepare_ms\":{:.3},\"prove_ms\":{:.3},\"verify_ms\":{:.3}}}",
            ms(metal_prepare_elapsed),
            ms(prove_elapsed),
            ms(verify_elapsed),
        );
    }

    println!(
        "CACHE_ARTIFACT_JSON={{\"rows\":{},\"columns\":{},\"matrices\":{},\"cache_bytes\":{},\"cache_mib\":{:.3},\"encoder_bytes\":{},\"encoder_mib\":{:.3},\"preprocess_ms\":{:.3},\"cache_write_ms\":{:.3},\"cache_load_ms\":{:.3},\"encoder_write_ms\":{:.3},\"encoder_load_ms\":{:.3},\"parallel_load_ms\":{:.3},\"artifact_prepare_ms\":{:.3},\"cold_profile_ms\":{:.3},\"bind_ms\":{:.3},\"second_bind_ms\":{:.3}}}",
        structure.n,
        structure.m,
        structure.t(),
        cache_receipt.artifact_bytes(),
        cache_receipt.artifact_bytes() as f64 / (1u64 << 20) as f64,
        encoder_receipt.encoder().artifact_bytes(),
        encoder_receipt.encoder().artifact_bytes() as f64 / (1u64 << 20) as f64,
        ms(preprocess_elapsed),
        ms(cache_write_elapsed),
        ms(cache_load_elapsed),
        ms(encoder_write_elapsed),
        ms(encoder_load_elapsed),
        ms(parallel_load_elapsed),
        ms(artifact_prepare_elapsed),
        ms(parallel_load_elapsed + artifact_prepare_elapsed),
        ms(bind_elapsed),
        ms(second_bind_elapsed),
    );
}

#[test]
#[cfg(all(feature = "perf-timers", feature = "metal", target_vendor = "apple"))]
#[ignore = "full Metal proof covering base, bootstrap, steady recursion, and terminal finalization"]
fn wasm_nebula_all_branch_metal_profile() {
    let wall_started = Instant::now();
    let wasm = wat::parse_str(PROFILE_WAT).expect("valid profile WAT");
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = neo_wasm::collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    let trace = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("normalized trace");

    // Four application folds force the deterministic branch schedule:
    // base, bootstrap-recursive, steady-recursive, steady-recursive.
    let memory = neo_fold_clean::frontends::nebula::layout::NebulaParams::new(11, 11, 64, 1024, 16)
        .expect("four-fold reduced Nebula scan");
    let profile = neo_wasm::WasmNebulaProfile::test_profile_with_schedule(memory, trace.len());
    assert_eq!(profile.memory().steps_per_segment(), 4);

    let entry_pc = common::single_function_entry_pc(&artifacts);
    let started = Instant::now();
    let prep = neo_wasm::nebula::preprocess_seeded_reduced_memory_test_only(
        test_params(),
        profile,
        &artifacts,
        &run.initial_locals,
        entry_pc,
        0x57a5_7021,
    )
    .expect("all-branch WASM Nebula preprocessing");
    let preprocess_elapsed = started.elapsed();

    let mut prover = MetalNifsProver::new().expect("Metal prover");
    let started = Instant::now();
    prover
        .prepare_static(
            &prep.inner().prep.log,
            prep.inner().relation().structure(),
            prep.inner().prep.optimized_cache(),
            prep.inner().prep.nebula().map(|config| &config.scheme),
        )
        .expect("prepare static Metal proof state");
    let metal_prepare_elapsed = started.elapsed();
    prover.session().reset_activity();

    let started = Instant::now();
    let proof = neo_wasm::nebula::prove_with_nifs_adapter(&prep, &mut prover, &trace).expect("all-branch Metal proof");
    let prove_elapsed = started.elapsed();
    assert_eq!(proof.inner().state.step_count, 4);
    assert_eq!(proof.inner().state.chunk_count, 4);
    assert!(proof.inner().final_fold.is_some());
    assert!(prover.session().activity().dispatches > 0);

    let started = Instant::now();
    neo_wasm::nebula::verify_with_witness_opening_backend(&prep, &proof, common::final_state(&trace), &mut prover)
        .expect("all-branch Metal verification");
    let verify_elapsed = started.elapsed();

    println!(
        "ALL_BRANCH_PROFILE_JSON={{\"folds\":4,\"preprocess_ms\":{:.3},\"metal_prepare_ms\":{:.3},\"prove_ms\":{:.3},\"verify_ms\":{:.3},\"total_ms\":{:.3}}}",
        ms(preprocess_elapsed),
        ms(metal_prepare_elapsed),
        ms(prove_elapsed),
        ms(verify_elapsed),
        ms(wall_started.elapsed()),
    );
}

/// Builds the production-parameter relation without enforcing the fixed-shape width
/// budget and executes the first occurrence of every F' branch. The prefix is
/// intentionally left open: a complete production memory segment contains
/// 1,088 folds and is a separate endurance benchmark.
#[test]
#[cfg(feature = "perf-timers")]
#[ignore = "production kappa=18 fixed point plus three real folds; run explicitly"]
fn wasm_nebula_production_prefix_profile() {
    production_prefix_profile(performance_profile(3));
}

#[test]
#[cfg(feature = "perf-timers")]
#[ignore = "production kappa=18 batch-4 fixed point plus three real folds; run explicitly"]
fn wasm_nebula_production_batch_4_prefix_profile() {
    production_prefix_profile(performance_profile(4));
}

#[test]
#[cfg(feature = "perf-timers")]
#[ignore = "production kappa=18 batch-5 fixed point plus three real folds; run explicitly"]
fn wasm_nebula_production_batch_5_prefix_profile() {
    production_prefix_profile(performance_profile(5));
}

#[cfg(feature = "perf-timers")]
fn production_prefix_profile(profile: neo_wasm::WasmNebulaProfile) {
    const PREFIX_FOLDS: usize = 3;

    let wall_started = Instant::now();
    let wasm = wat::parse_str(PROFILE_WAT).expect("valid profile WAT");
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = neo_wasm::collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    assert_eq!(run.results.as_slice(), &["297".to_string()]);
    let trace = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("normalized trace");
    let _witnesses = common::sanity_check_trace(&trace, &artifacts, &run.initial_locals);
    common::ccs_check_trace(&trace);

    // The current relation pads both CCS axes to 2^25. This selects soundness
    // parameters for that domain. It is not a coordinate limit.
    let params = config::ccs_params(1 << 25, 1 << 25, 13, 8).expect("production WASM parameters");
    assert_eq!(params.kappa(), 18);
    assert_eq!(params.k_rho(), 14);
    assert_eq!(params.b(), 2);
    assert_eq!(params.big_b(), 1 << 14);
    assert_eq!(params.T(), 216);
    assert_eq!(profile.memory().rom_cells(), 4_096);
    assert_eq!(profile.memory().ram_cells(), 65_536);
    assert_eq!(profile.memory().steps_per_segment(), 1_088);

    let entry_pc = common::single_function_entry_pc(&artifacts);
    let started = Instant::now();
    let prep = neo_wasm::nebula::preprocess_seeded(
        params.clone(),
        profile,
        &artifacts,
        &run.initial_locals,
        entry_pc,
        0x57a5_7018,
    )
    .expect("production WASM + Nebula preprocessing");
    let preprocess_elapsed = started.elapsed();

    let relation = prep.inner().relation();
    let structure = relation.structure();
    let width = relation
        .low_norm_width_audit()
        .expect("live relation has a width audit");
    let arms = relation.field_arm_shapes();
    let storage = structure_stats(structure);
    let padded_rows = structure.n.max(structure.m).next_power_of_two();
    let ell = padded_rows.ilog2();
    assert_ne!(structure.n, structure.m, "production relation must remain rectangular");
    assert_eq!(structure.t(), 13);
    assert_eq!(structure.max_degree(), 8);
    assert_eq!(width.total_coordinates.div_ceil(D) * D, structure.m);

    println!("\n== WASM + Nebula production-parameter prefix profile ==");
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
        "relation                 rows={} columns={} public={} matrices={} degree={}",
        structure.n,
        structure.m,
        relation.public_input_len(),
        structure.t(),
        structure.max_degree(),
    );
    println!(
        "padded one-SumCheck      ell={} row_pad={} column_pad={}",
        ell,
        padded_rows - structure.n,
        padded_rows - structure.m,
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
    for (name, arm) in ["base", "recursive"].into_iter().zip(&width.arms) {
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

    let stages = &width.arms[1].physical_stages;
    assert!(
        !stages.is_empty(),
        "production recursive F' must have complete physical-stage provenance"
    );
    assert_eq!(
        stages
            .iter()
            .map(|stage| stage.allocated_coordinates)
            .sum::<usize>(),
        width.arms[1].branch_coordinates,
        "exclusive stages must own the complete recursive branch"
    );
    let mut stage_totals = BTreeMap::<&str, [usize; 5]>::new();
    let mut phase_totals = BTreeMap::<&str, usize>::new();
    for stage in stages {
        let totals = stage_totals.entry(stage.path).or_default();
        totals[0] += 1;
        totals[1] += stage.source_column_count;
        totals[2] += stage.retained_coordinates_before_aliases;
        totals[3] += stage.decomposition_alias_coordinates + stage.equality_alias_coordinates;
        totals[4] += stage.allocated_coordinates;
        let phase = if stage.path.starts_with("nifs.pi_ccs") {
            "PiCCS"
        } else if stage.path.starts_with("nifs.pi_rlc") {
            "PiRLC"
        } else if stage.path.starts_with("nifs.pi_dec")
            || stage.path.starts_with("nifs.running_parent_pi_dec")
            || stage.path.starts_with("nifs.point_binding")
        {
            "PiDEC and point binding"
        } else if stage.path.starts_with("fprime.recursive.step.accumulator") {
            "accumulator authority"
        } else if stage.path.starts_with("fprime.recursive.step.nebula") {
            "delayed Nebula transition"
        } else if stage.path.starts_with("fprime.recursive.finalize") {
            "application and final links"
        } else {
            "outer F' state and transcript"
        };
        *phase_totals.entry(phase).or_default() += stage.allocated_coordinates;
    }
    assert_eq!(
        phase_totals.values().sum::<usize>(),
        width.arms[1].branch_coordinates,
        "exclusive phase groups must own the complete recursive branch"
    );
    let mut phase_totals = phase_totals.into_iter().collect::<Vec<_>>();
    phase_totals.sort_by_key(|(_, coordinates)| Reverse(*coordinates));
    println!("\n-- recursive F' exclusive phase width --");
    for (phase, coordinates) in phase_totals {
        println!("{phase:<32} {coordinates:>10}");
    }
    let mut stage_totals = stage_totals.into_iter().collect::<Vec<_>>();
    stage_totals.sort_by_key(|(_, totals)| Reverse(totals[4]));
    println!("\n-- recursive F' exclusive stage width --");
    println!("stage                                                   occurrences  source-cols  before-alias  aliases  allocated");
    for (path, totals) in stage_totals
        .into_iter()
        .filter(|(_, totals)| totals[4] != 0)
    {
        println!(
            "{path:<55} {:>11} {:>11} {:>13} {:>8} {:>10}",
            totals[0], totals[1], totals[2], totals[3], totals[4],
        );
    }

    let mut families = width.arms[1].row_families.iter().collect::<Vec<_>>();
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
    println!("\n-- unbounded production-parameter timing --");
    println!("fixed-point preprocess     {:>12.2}ms", ms(preprocess_elapsed));
    println!("segment witness build      {:>12.2}ms", ms(segment_build_elapsed));
    println!("three-fold prefix*         {:>12.2}ms", ms(prefix_elapsed));
    println!(
        "mean prefix wall / fold*  {:>12.2}ms",
        ms(prefix_elapsed) / PREFIX_FOLDS as f64
    );
    println!(
        "* includes one full {segment_steps}-step lane precommit; per-step timers are the branch-latency measurements"
    );
    println!(
        "naive projection (overcounts precommit) {:>10.2}ms ({:.2}h)",
        ms(naive_segment_projection),
        naive_segment_projection.as_secs_f64() / 3_600.0,
    );
    println!("wall total                 {:>12.2}ms", ms(wall_started.elapsed()));
    println!(
        "PROFILE_PRODUCTION_JSON={{\"trace_steps\":{},\"padded_wasm_steps\":{},\"batch_size\":{},\"prefix_folds\":{},\"segment_folds\":{},\"kappa\":{},\"k_rho\":{},\"lambda\":{},\"rows\":{},\"columns\":{},\"matrices\":{},\"ell\":{},\"explicit_nnz\":{},\"seeded_blocks\":{},\"geometric_runs\":{},\"preprocess_ms\":{:.3},\"segment_build_ms\":{:.3},\"prefix_ms\":{:.3},\"naive_segment_ms\":{:.3},\"wall_ms\":{:.3}}}",
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
        ell,
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
