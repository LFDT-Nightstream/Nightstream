mod common;

use std::num::NonZeroUsize;

use neo_math::F;
use neo_wasm::host_event_bindings::HostEventBindings;
use neo_wasm::memory_semantics::preload_host_event_tables;
use neo_wasm::nebula::{plan_ordered_slots, step_inputs, WasmApplication, WasmLimits, WasmStep};
use neo_wasm::{
    collect_wasmtime_steps, extract_wasm_program_artifacts, preload_from_program_artifacts, top_level_initial_state,
    traces_from_wasmtime_steps, WasmMemoryPreload, WasmProgramArtifacts, WasmStepState, WasmVmStep,
};
use nightstream::nebula::{Circuit, Geometry};

fn wasmtime_main(wasm: &[u8]) -> Result<Vec<String>, String> {
    let mut config = wasmtime::Config::new();
    config.wasm_reference_types(true);
    config.wasm_function_references(true);
    let engine = wasmtime::Engine::new(&config).unwrap();
    let module = wasmtime::Module::new(&engine, wasm).map_err(|e| format!("compile: {e}"))?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).map_err(|e| format!("instantiate: {e}"))?;
    let main = instance.get_func(&mut store, "main").ok_or("no main")?;
    let mut results = vec![wasmtime::Val::I32(0); main.ty(&store).results().len()];
    main.call(&mut store, &[], &mut results)
        .map_err(|e| format!("trap: {e}"))?;
    Ok(results
        .iter()
        .map(|value| match value {
            wasmtime::Val::I32(v) => v.to_string(),
            other => format!("{other:?}"),
        })
        .collect())
}

fn preload_and_initial(artifacts: &WasmProgramArtifacts) -> (WasmMemoryPreload, WasmStepState) {
    let (export_fref, entry_pc) = artifacts
        .tables
        .function_entries
        .iter()
        .copied()
        .max_by_key(|(_, pc)| *pc)
        .unwrap();
    let mut preload = preload_from_program_artifacts(artifacts);
    preload_host_event_tables(
        &mut preload,
        &HostEventBindings::import_free(u32::try_from(export_fref).unwrap()),
        artifacts,
    );
    (preload, top_level_initial_state(&artifacts.tables, entry_pc).unwrap())
}

fn honest_trace(wasm: &[u8]) -> (Vec<WasmVmStep>, WasmProgramArtifacts) {
    let run = collect_wasmtime_steps(wasm, &HostEventBindings::default(), "main", &[]).unwrap();
    let trace = traces_from_wasmtime_steps(&run.steps).unwrap();
    common::sanity_check_trace(&trace, run.artifacts());
    common::ccs_check_trace(&trace);
    (trace, run.artifacts().clone())
}

fn bits(value: u64) -> usize {
    (u64::BITS - value.leading_zeros()) as usize
}

fn prove_and_verify_for(
    trace: &[WasmVmStep],
    artifacts: &WasmProgramArtifacts,
    others: &[(WasmMemoryPreload, WasmStepState)],
) -> Vec<bool> {
    let n = trace.len().next_power_of_two();
    let inputs: Vec<WasmStep> = step_inputs(trace, n, NonZeroUsize::MIN).unwrap();
    let rows = inputs
        .iter()
        .map(|input| input.z.clone())
        .collect::<Vec<Vec<F>>>();
    let (preload, initial) = preload_and_initial(artifacts);
    let last = trace.last().unwrap().state_after;
    let preloads = std::iter::once(&preload)
        .chain(others.iter().map(|(preload, _)| preload))
        .collect::<Vec<_>>();
    let (regions, used) = WasmLimits::of(&preloads, &rows).unwrap().regions().unwrap();
    let log2 = used.map(|cells| bits(cells - 1));
    let slots = plan_ordered_slots(NonZeroUsize::MIN).unwrap();
    let r = log2[0].max(n.trailing_zeros() as usize);
    let geometry = Geometry::new(r, log2[1].max(r), slots.len(), n, 1).unwrap();
    let app = WasmApplication::with_slots(&geometry, regions, NonZeroUsize::MIN, &slots).unwrap();
    let engine = nightstream::Engine::Optimized;
    let circuit = Circuit::compile(geometry, app.clone()).unwrap();
    let prover = circuit.prover(engine, 109).unwrap();
    let verifier = circuit.verifier(engine, 109).unwrap();
    let program = app.program(&preload).unwrap();
    let mut run = prover.start(&program, initial).unwrap();
    run.segment(&inputs).unwrap();
    let proof = run.finish().unwrap();
    verifier
        .verify(&program, &initial, &last, &proof)
        .expect("the honest proof verifies");
    others
        .iter()
        .map(|(preload, initial)| {
            let program = app.program(preload).unwrap();
            verifier.verify(&program, initial, &last, &proof).is_ok()
        })
        .collect()
}

/// WebAssembly type equivalence: `$q` is the type `$p` when `callee_param`
/// is `(ref null $y)`, because `$y` is the type `$x`.
fn equivalent_types(callee_param: &str) -> Vec<u8> {
    wat::parse_str(format!(
        r#"(module
            (type $x (func))
            (type $y (func))
            (type $p (func (param (ref null $x)) (result i32)))
            (type $q (func (param {callee_param}) (result i32)))
            (table 1 funcref)
            (elem (i32.const 0) func $callee)
            (elem declare func $nop)
            (func $nop (type $x))
            (func $callee (type $q) (param {callee_param}) (result i32) i32.const 42)
            (func (export "main") (result i32)
                ref.func $nop
                i32.const 0
                call_indirect (type $p)))"#
    ))
    .unwrap()
}

/// A type that refers to itself: `$b` is the type `$a`, and `$c`, which
/// refers to `$a` from outside its own group, is not.
fn self_referring_types(callee_type: &str) -> Vec<u8> {
    wat::parse_str(format!(
        r#"(module
            (type $a (func (param (ref null $a)) (result i32)))
            (type $b (func (param (ref null $b)) (result i32)))
            (type $c (func (param (ref null $a)) (result i32)))
            (table 1 funcref)
            (elem (i32.const 0) func $callee)
            (elem declare func $arg)
            (func $arg (type $a) i32.const 0)
            (func $callee (type {callee_type}) i32.const 42)
            (func (export "main") (result i32)
                ref.func $arg
                i32.const 0
                call_indirect (type $a)))"#
    ))
    .unwrap()
}

#[test]
fn equivalent_call_indirect_types_are_bound_by_the_public_verifier() {
    let equivalent = equivalent_types("(ref null $y)");
    let different = equivalent_types("externref");
    assert_eq!(wasmtime_main(&equivalent).unwrap(), ["42"]);
    assert!(wasmtime_main(&different).unwrap_err().starts_with("trap"));

    let (trace, artifacts) = honest_trace(&equivalent);
    assert_eq!(trace.last().unwrap().state_after.output.value_lo, 42);

    let (trap, trap_artifacts) = honest_trace(&different);
    let last = trap.last().unwrap().state_after;
    assert!(last.is_terminal() && last.trapped && !last.output.enabled);
    let other = preload_and_initial(&extract_wasm_program_artifacts(&equivalent).unwrap());
    assert_eq!(
        prove_and_verify_for(&trap, &trap_artifacts, &[other]),
        [false],
        "the public verifier accepted a trap for a call_indirect whose type equals the callee type; Wasmtime returns 42"
    );
    prove_and_verify_for(&trace, &artifacts, &[]);
}

#[test]
fn self_referring_call_indirect_types_match_wasmtime() {
    let same = self_referring_types("$b");
    assert_eq!(wasmtime_main(&same).unwrap(), ["42"]);
    let (trace, _) = honest_trace(&same);
    assert_eq!(trace.last().unwrap().state_after.output.value_lo, 42);

    let other = self_referring_types("$c");
    assert!(wasmtime_main(&other).unwrap_err().starts_with("trap"));
    let (trace, _) = honest_trace(&other);
    let last = trace.last().unwrap().state_after;
    assert!(last.is_terminal() && last.trapped && !last.output.enabled);
}
