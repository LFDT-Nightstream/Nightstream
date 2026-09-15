//! Demo gate: import/export events and linear-memory accesses cross multiple
//! recursive WASM/Nebula steps, followed by terminal verification on Metal.

#![cfg(all(feature = "metal", feature = "perf-timers", target_vendor = "apple"))]

mod common;

use std::time::Instant;

use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::paper::params::Params;
use neo_math::F;
use neo_prover_metal::MetalNifsProver;
use neo_wasm::host_event_bindings::{EventSequenceBuilder, HostEventBindingsBuilder, TurnInputs};
use p3_field::PrimeCharacteristicRing;

const COMPONENT: &str = r#"
(component
  (type $double-type (func (param "value" s32) (result s32)))
  (import "double" (func $double (type $double-type)))
  (core module $m
    (import "" "double" (func $double (param i32) (result i32)))
    (memory 1 1)
    (func (export "run") (param i32) (result i32)
      i32.const 0
      local.get 0
      call $double
      i32.store
      i32.const 0
      i32.load))
  (core func $lowered (canon lower (func $double)))
  (core instance $host (export "double" (func $lowered)))
  (core instance $i (instantiate $m (with "" (instance $host))))
  (alias core export $i "run" (core func $run))
  (func (export "run") (type $double-type) (canon lift (core func $run))))
"#;

#[test]
#[ignore = "full recursive Metal demo; run explicitly with the 300-second test cap"]
fn wasm_nebula_metal_import_export_demo() {
    let wall = Instant::now();
    let bytes = wat::parse_str(COMPONENT).expect("component");
    let run = neo_wasm::collect_wasmtime_component_run_with_linker_and_args(
        &bytes,
        "run",
        &[wasmtime::component::Val::S32(21)],
        |linker| {
            linker
                .root()
                .func_wrap("double", |_store, (value,): (i32,)| Ok((value * 2,)))
                .map_err(|error| neo_wasm::WasmBuildError::Trace(error.to_string()))
        },
    )
    .expect("component execution");
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&bytes).expect("artifacts");
    let import_fref = run
        .steps
        .iter()
        .find(|row| row.opcode_decoded == Some(neo_wasm::WasmOpcode::Call) && !row.target_function_is_guest)
        .and_then(|row| row.function_ref)
        .expect("host call");
    let export_fref = run
        .steps
        .iter()
        .find_map(|row| row.current_function_ref)
        .expect("export");
    let mut builder = HostEventBindingsBuilder::new(&run.program_tables);
    builder
        .import(
            import_fref,
            EventSequenceBuilder::op(2)
                .arg_i32(0)
                .unwrap()
                .result()
                .unwrap()
                .finish()
                .unwrap(),
        )
        .unwrap();
    builder
        .export(
            export_fref,
            EventSequenceBuilder::op(1)
                .input_local_i32(0, 0)
                .unwrap()
                .finish()
                .unwrap(),
            EventSequenceBuilder::op(3)
                .output_i32()
                .unwrap()
                .finish()
                .unwrap(),
        )
        .unwrap();
    let bindings = builder.finish().expect("bindings");
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(
        &run.steps,
        &run.program_tables,
        &bindings,
        &[TurnInputs { entry: vec![21] }],
        Default::default(),
    )
    .expect("event-bound trace");
    common::ccs_check_trace(&trace);
    common::check_native_event_hashes(&trace).expect("event hashes");
    let final_state = common::final_state(&trace);
    assert_eq!(final_state.output.value_lo, 42);
    assert!(trace
        .iter()
        .any(|row| row.opcode == neo_wasm::WasmOpcode::I32Store));
    assert!(trace
        .iter()
        .any(|row| row.opcode == neo_wasm::WasmOpcode::I32Load));
    let events = [
        [1, 21, 0, 0, 0, 0, 0, 0],
        [2, 21, 42, 0, 0, 0, 0, 0],
        [3, 42, 0, 0, 0, 0, 0, 0],
    ]
    .map(|block| block.map(F::from_u64));
    let expected_chain = neo_wasm::comm_chain::fold_event_blocks(Default::default(), &events).canonical_u64();
    assert_eq!(final_state.comm_chain, expected_chain);

    // Four application steps cover both bootstrap and ordinary recursive F'.
    // The 64-word linear-memory fixture uses only address zero; this profile
    // does not model all addresses in a WASM page and is not production setup.
    let memory = NebulaParams::new(12, 12, 64, 2048, 16).unwrap();
    let batch = trace.len().div_ceil(4);
    let profile = neo_wasm::WasmNebulaProfile::test_profile_with_schedule(memory, batch);
    assert_eq!(profile.memory().steps_per_segment(), 4);
    let raw = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        1 << 25,
        neo_params::goldilocks_paper_b2::B_BASE,
        neo_params::goldilocks_paper_b2::K_RHO,
        neo_params::goldilocks_paper_b2::T,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        20,
    )
    .unwrap();
    println!(
        "DEMO trace_rows={} batch={} application_steps=4 events=3 output=42",
        trace.len(),
        batch
    );
    let started = Instant::now();
    let prep = neo_wasm::nebula::preprocess_seeded_host_events_reduced_memory_test_only(
        Params::test_only_from_neo_params(raw),
        profile,
        &artifacts,
        common::entry_pc_for_function_ref(&artifacts, u64::from(export_fref)),
        &bindings,
        export_fref,
        0x57a5_de00,
        Default::default(),
    )
    .expect("WASM + Nebula + recursive NIFS preprocessing");
    let preprocess = started.elapsed();
    println!(
        "DEMO preprocessing_s={:.3} rows={} columns={}",
        preprocess.as_secs_f64(),
        prep.inner().relation().structure().n,
        prep.inner().relation().structure().m
    );
    let mut metal = MetalNifsProver::new().expect("Metal required; no CPU fallback");
    let started = Instant::now();
    metal
        .prepare_static(
            &prep.inner().prep.log,
            prep.inner().relation().structure(),
            prep.inner().prep.optimized_cache(),
            prep.inner().prep.nebula().map(|config| &config.scheme),
        )
        .expect("Metal static preparation");
    let prepare = started.elapsed();
    metal.session().reset_activity();
    let started = Instant::now();
    let proof = neo_wasm::nebula::prove_with_nifs_adapter(&prep, &mut metal, &trace).expect("recursive Metal proof");
    let prove = started.elapsed();
    let activity = metal.session().activity();
    assert!(activity.dispatches > 0 && activity.host_waits > 0);
    assert!(
        proof.inner().state.step_count >= 4,
        "ordinary recursive F' must execute"
    );
    assert!(
        proof.inner().final_fold.is_some(),
        "the delayed terminal fold must execute"
    );
    let started = Instant::now();
    neo_wasm::nebula::verify_with_witness_opening_backend(&prep, &proof, final_state, &mut metal)
        .expect("terminal proof verification");
    let verify = started.elapsed();
    assert!(metal.session().activity().dispatches > activity.dispatches);
    println!("DEMO PASS preprocess_s={:.3} metal_prepare_s={:.3} prove_s={:.3} verify_s={:.3} total_s={:.3} steps={} metal_dispatches={}", preprocess.as_secs_f64(), prepare.as_secs_f64(), prove.as_secs_f64(), verify.as_secs_f64(), wall.elapsed().as_secs_f64(), proof.inner().state.step_count, metal.session().activity().dispatches);

    let mut false_claim = final_state;
    false_claim.output.value_lo = 43;
    assert!(neo_wasm::nebula::verify_with_witness_opening_backend(&prep, &proof, false_claim, &mut metal).is_err());
    let mut false_events = events;
    false_events[1][2] += F::ONE;
    false_claim = final_state;
    false_claim.comm_chain = neo_wasm::comm_chain::fold_event_blocks(Default::default(), &false_events).canonical_u64();
    assert!(neo_wasm::nebula::verify_with_witness_opening_backend(&prep, &proof, false_claim, &mut metal).is_err());
    println!("DEMO rejected changed output and changed import transcript");
}
