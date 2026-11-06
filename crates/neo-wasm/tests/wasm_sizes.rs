//! The fixed WASM sizes (`neo_wasm::nebula::WasmSize`,
//! `decisions/wasm-fixed-sizes.md`). Each size is a secure geometry whose
//! regions fit its spaces. A prover takes the smallest size that holds a run,
//! and the run steps natively through the segments of that size.

mod common;

use neo_wasm::host_event_bindings::HostEventBindings;
use neo_wasm::memory_semantics::preload_host_event_tables;
use neo_wasm::nebula::{WasmApplicationError, WasmLimits, WasmSize};
use neo_wasm::{preload_from_program_artifacts, top_level_initial_state, WasmMemoryPreload, WasmStepState, WasmVmStep};
use nightstream::nebula::{Application, Run};

/// Enzo's iterative Fibonacci program (`tests/wasm_fibonacci.rs`) with two
/// loop rounds.
const FIB: &str = r#"(module
  (func (export "main") (result i32)
    (local i32 i32 i32)
    i32.const 2
    local.set 0
    i32.const 1
    local.set 1
    i32.const 0
    local.set 2
    (loop $L
      local.get 1
      local.get 2
      i32.add
      local.get 2
      local.set 1
      local.set 2
      local.get 0
      i32.const 1
      i32.sub
      local.tee 0
      br_if $L
    )
    local.get 2))"#;

/// A program of `count` `nop`s, so that its program counters need the bits
/// of `count`.
fn nops(count: usize) -> String {
    format!(
        r#"(module (func (export "main") (result i32) {} i32.const 7))"#,
        "nop ".repeat(count)
    )
}

/// The preload, the entry state and the trace of `main`, with the host-event
/// tables of an export that has no import.
fn run_of(wat: &str) -> (WasmMemoryPreload, WasmStepState, Vec<WasmVmStep>) {
    let checked = common::checked_main(wat);
    let tables = &checked.artifacts.tables;
    let (export_fref, entry_pc) = tables
        .function_entries
        .iter()
        .copied()
        .max_by_key(|(_, pc)| *pc)
        .expect("a function");
    let mut preload = preload_from_program_artifacts(&checked.artifacts);
    preload_host_event_tables(
        &mut preload,
        &HostEventBindings::import_free(u32::try_from(export_fref).unwrap()),
    );
    (preload, top_level_initial_state(tables, entry_pc), checked.trace)
}

#[test]
fn each_size_is_a_secure_geometry_whose_regions_fit_its_spaces() {
    let mut smaller: Option<WasmLimits> = None;
    for size in WasmSize::ALL {
        let geometry = size.geometry();
        assert!(geometry.is_valid(), "{size:?}");
        assert!(
            geometry.max_tuples() <= 139_509,
            "{size:?} keeps SPEC.md section 7.1, rule 6"
        );
        let (_, used) = size.limits().regions();
        assert!(
            used[0] <= 1 << geometry.r && used[1] <= 1 << geometry.mu,
            "{size:?} uses {used:?}"
        );
        assert_eq!(size.application().unwrap().port_count(), geometry.b_ops);
        assert_eq!(size.max_rows(), 16_384, "{size:?}");
        if let Some(smaller) = smaller {
            assert!(size.limits().holds(&smaller), "{size:?} holds the smaller sizes");
        }
        smaller = Some(size.limits());
    }
}

/// The limits that the six test transactions of Starstream `159bedc` need
/// (`starstream-proving-runtime/tests`, all coroutine instances of each), and
/// the rows of their longest WASM run.
#[test]
fn the_starstream_test_transactions_fit_small_or_medium() {
    let need = |pc, function, event, linear_memory, global| WasmLimits {
        pc,
        choice: 1,
        function,
        event,
        slot: 3,
        stack: 5,
        linear_memory,
        global,
        ..WasmLimits::default()
    };
    for (name, limits, rows, size) in [
        ("minimal_method_call", need(7, 5, 3, 0, 0), 363, WasmSize::Small),
        ("storage_counter", need(7, 5, 3, 4, 0), 387, WasmSize::Small),
        ("multi_utxo", need(8, 5, 3, 4, 0), 482, WasmSize::Small),
        ("effect_reentry", need(5, 4, 2, 0, 0), 328, WasmSize::Small),
        ("score", need(8, 6, 3, 4, 0), 903, WasmSize::Medium),
        ("distinct_utxo_types", need(8, 6, 3, 5, 4), 646, WasmSize::Medium),
    ] {
        assert_eq!(WasmSize::smallest(&limits, rows), Some(size), "{name}");
    }
}

/// The memory security of each size is below 114 bits, so a prover that asks
/// for 114 gets the named security error before it reads its circuit file.
#[test]
fn a_size_refuses_more_security_than_its_memory_gives() {
    use nightstream::error::ParamsError;
    use nightstream::nebula::Prover;
    for (size, available) in [(WasmSize::Small, 111), (WasmSize::Medium, 110), (WasmSize::Large, 109)] {
        let result = Prover::load(
            "no-circuit-file",
            size.geometry(),
            size.application().unwrap(),
            nightstream::Engine::Optimized,
            114,
        );
        let Err(nightstream::Error::Parameters(error)) = result else {
            panic!("{size:?} accepts 114 bits");
        };
        let error: ParamsError = error;
        assert!(
            matches!(error, ParamsError::InsufficientStatisticalSecurity { required: 114, available: a } if a == available),
            "{size:?}: {error}"
        );
    }
}

/// Programs of more instructions need larger sizes, and one of 1,100
/// instructions fits none. A run of more rows than every size holds fits
/// none either.
#[test]
fn a_prover_takes_the_smallest_size_that_holds_the_run() {
    for (count, size) in [
        (10, Some(WasmSize::Small)),
        (300, Some(WasmSize::Medium)),
        (600, Some(WasmSize::Large)),
        (1_100, None),
    ] {
        let (preload, _, trace) = run_of(&nops(count));
        match (WasmSize::for_run(&preload, &trace), size) {
            (Ok(found), Some(size)) => assert_eq!(found, size, "{count} nops"),
            (Err(WasmApplicationError::TooLarge(limits)), None) => assert_eq!(limits.pc, 11),
            (other, _) => panic!("{count} nops: {other:?}"),
        }
    }
    assert_eq!(
        WasmSize::smallest(&WasmLimits::default(), 16_384),
        Some(WasmSize::Small)
    );
    assert_eq!(WasmSize::smallest(&WasmLimits::default(), 16_385), None);
}

/// Fibonacci steps natively through Small in one segment. The 600 `nop`s
/// step through Large in three segments of 256 rows. Each run ends in the
/// state after its last trace row.
#[test]
fn a_run_steps_natively_through_the_segments_of_its_size() {
    for (wat, size, segments) in [(FIB.to_string(), WasmSize::Small, 1), (nops(600), WasmSize::Large, 3)] {
        let (preload, initial, trace) = run_of(&wat);
        assert_eq!(WasmSize::for_run(&preload, &trace).unwrap(), size);
        let app = size.application().unwrap();
        let program = app.program(&preload).unwrap();
        let mut run = Run::new(app, &program, initial);
        let inputs = size.segments(&trace).unwrap();
        assert_eq!(inputs.len(), segments, "{size:?}");
        for segment in &inputs {
            run.segment(segment).unwrap();
        }
        assert_eq!(*run.state(), trace.last().unwrap().state_after);
    }
}

/// Fibonacci proved on Metal in Small, one segment of 64 steps of 8 rows. The
/// proof verifies after its bytes round-trip, and a last state with another
/// `pc` does not verify.
#[test]
#[ignore = "Metal proof at a fixed size, about 10 minutes; build with --features nightstream/metal and run it alone, only with approval for a run longer than 300 seconds."]
fn a_small_proof_verifies_on_metal() {
    use nightstream::nebula::{Circuit, VerifyError};
    use std::time::Instant;
    let started = Instant::now();
    let (preload, initial, trace) = run_of(FIB);
    let size = WasmSize::for_run(&preload, &trace).unwrap();
    assert_eq!(size, WasmSize::Small);
    let app = size.application().unwrap();
    let program = app.program(&preload).unwrap();
    let circuit = Circuit::compile(size.geometry(), app).unwrap();
    eprintln!("compile elapsed={:?}", started.elapsed());
    let engine = nightstream::Engine::Metal;
    let prover = circuit.prover(engine, 109).unwrap();
    let verifier = circuit.verifier(engine, 109).unwrap();
    eprintln!("setup elapsed={:?}", started.elapsed());

    let proving = Instant::now();
    let mut run = prover.start(&program, initial).unwrap();
    for segment in size.segments(&trace).unwrap() {
        run.segment(&segment).unwrap();
    }
    let last = *run.state();
    let proof = run.finish().unwrap();
    eprintln!("prove 64 steps elapsed={:?}", proving.elapsed());

    let bytes = prover.encode_proof(&proof).unwrap();
    let verifying = Instant::now();
    let proof = verifier.decode_proof(&bytes).unwrap();
    let verified = verifier.verify(&program, &initial, &last, &proof).unwrap();
    eprintln!("verify elapsed={:?}, proof bytes={}", verifying.elapsed(), bytes.len());
    assert_eq!(verified.steps, 64);
    let mut other = last;
    other.pc += 1;
    assert!(matches!(
        verifier.verify(&program, &initial, &other, &proof),
        Err(VerifyError::Proof(_))
    ));
    eprintln!("total elapsed={:?}", started.elapsed());
}
