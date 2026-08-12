use neo_wasm::{
    build_debug_function_id_map, build_pc_rom_from_binary, collect_wasmtime_steps, extract_wasm_program_artifacts,
    opcode_code, traces_from_wasmtime_steps, traces_from_wasmtime_wasm_bytes, StackValueAccess, WasmOpcode,
    WasmPcEdgeKind, WasmTraceSink, WasmtimeTraceHandler, WasmtimeTraceState, WasmtimeTraceStep,
};
use std::collections::{HashMap, HashSet};
use wasmparser::{Parser, Payload};
use wasmtime::{Config, Engine, Linker, Module, Store, Val};

struct EmbedderStoreData {
    trace: WasmtimeTraceState,
    host_counter: u32,
}

impl WasmTraceSink for EmbedderStoreData {
    // Single traced instance: ignore the index and route to the one trace.
    fn wasm_trace_state(&self, _instance_index: u32) -> Option<&WasmtimeTraceState> {
        Some(&self.trace)
    }

    fn wasm_trace_state_mut(&mut self, _instance_index: u32) -> Option<&mut WasmtimeTraceState> {
        Some(&mut self.trace)
    }

    fn record_untraced_instance(&mut self, _instance_index: u32) {}
}

fn sample_steps() -> Vec<WasmtimeTraceStep> {
    vec![
        WasmtimeTraceStep {
            step: 0,
            frame_depth: 0,
            function: "DefinedFuncIndex(0)".to_string(),
            function_index: Some(0),
            pc: Some(49),
            opcode: Some("I32Const { value: 7 }".to_string()),
            opcode_decoded: Some(WasmOpcode::I32Const),
            immediate_i32: Some(7),
            current_function_ref: Some(1),
            pc_edge_kind: Some(WasmPcEdgeKind::Static),
            ..Default::default()
        },
        WasmtimeTraceStep {
            step: 1,
            frame_depth: 0,
            function: "DefinedFuncIndex(0)".to_string(),
            function_index: Some(0),
            pc: Some(51),
            opcode: Some("I32Const { value: 9 }".to_string()),
            opcode_decoded: Some(WasmOpcode::I32Const),
            immediate_i32: Some(9),
            operand_stack: vec!["7".to_string()],
            operand_stack_words: vec![7],
            operand_stack_words_hi: vec![0],
            pc_edge_kind: Some(WasmPcEdgeKind::Static),
            current_function_ref: Some(1),
            ..Default::default()
        },
        WasmtimeTraceStep {
            step: 2,
            frame_depth: 0,
            function: "DefinedFuncIndex(0)".to_string(),
            function_index: Some(0),
            pc: Some(53),
            opcode: Some("I32Add".to_string()),
            opcode_decoded: Some(WasmOpcode::I32Add),
            operand_stack: vec!["7".to_string(), "9".to_string()],
            operand_stack_words: vec![7, 9],
            operand_stack_words_hi: vec![0, 0],
            pc_edge_kind: Some(WasmPcEdgeKind::Static),
            current_function_ref: Some(1),
            ..Default::default()
        },
        WasmtimeTraceStep {
            step: 3,
            frame_depth: 0,
            function: "DefinedFuncIndex(0)".to_string(),
            function_index: Some(0),
            pc: Some(55),
            opcode: Some("End".to_string()),
            opcode_decoded: Some(WasmOpcode::End),
            operand_stack: vec!["16".to_string()],
            operand_stack_words: vec![16],
            operand_stack_words_hi: vec![0],
            pc_edge_kind: Some(WasmPcEdgeKind::ReturnLike),
            current_function_ref: Some(1),
            ..Default::default()
        },
    ]
}

#[test]
fn wasmtime_trace_handler_records_into_embedder_store_data() {
    let wasm = wat::parse_str(
        r#"(module
            (func (export "run") (result i32)
                i32.const 7
                i32.const 9
                i32.add)
        )"#,
    )
    .expect("wat");
    let artifacts = extract_wasm_program_artifacts(&wasm).expect("program artifacts");

    let mut config = Config::new();
    config.guest_debug(true);
    config.wasm_reference_types(true);
    config.wasm_function_references(true);

    let engine = Engine::new(&config).expect("engine");
    let module = Module::from_binary(&engine, &wasm).expect("module");
    let mut store = Store::new(
        &engine,
        EmbedderStoreData {
            trace: WasmtimeTraceState::from_program_artifacts(&artifacts),
            host_counter: 7,
        },
    );
    store.set_debug_handler(WasmtimeTraceHandler::<EmbedderStoreData>::new());
    store
        .edit_breakpoints()
        .expect("guest debug enabled")
        .single_step(true)
        .expect("single-step mode");

    let linker = Linker::new(&engine);
    let instance = futures::executor::block_on(linker.instantiate_async(&mut store, &module)).expect("instantiate");
    let instance_index = instance.debug_index_in_store();
    let func_ref_ids = build_debug_function_id_map(&instance, &mut store).expect("funcref map");
    store
        .data_mut()
        .wasm_trace_state_mut(instance_index)
        .expect("registered trace")
        .set_func_ref_ids(func_ref_ids);

    let func = instance.get_func(&mut store, "run").expect("exported func");
    let mut results = vec![Val::I32(0)];
    futures::executor::block_on(func.call_async(&mut store, &[], &mut results)).expect("call");

    assert_eq!(store.data().host_counter, 7);
    let steps = store.data().trace.steps();
    assert!(
        steps
            .iter()
            .any(|step| step.opcode_decoded == Some(WasmOpcode::I32Add)),
        "expected traced i32.add row, got {steps:?}"
    );
}

/// Embedder store data keyed by `Instance::debug_index_in_store()`.
struct MultiInstanceSink {
    traces: HashMap<u32, WasmtimeTraceState>,
    untraced: HashSet<u32>,
}

impl WasmTraceSink for MultiInstanceSink {
    fn wasm_trace_state(&self, instance_index: u32) -> Option<&WasmtimeTraceState> {
        self.traces.get(&instance_index)
    }
    fn wasm_trace_state_mut(&mut self, instance_index: u32) -> Option<&mut WasmtimeTraceState> {
        self.traces.get_mut(&instance_index)
    }
    fn record_untraced_instance(&mut self, instance_index: u32) {
        self.untraced.insert(instance_index);
    }
}

/// The `ref.func` row pushes the funcref; the following `drop` row captures it on
/// its pre-state operand stack. Returns the normalized funcref id seen there.
fn funcref_seen_at_drop(trace: &WasmtimeTraceState) -> Option<u32> {
    trace
        .steps()
        .iter()
        .find(|step| step.opcode_decoded == Some(WasmOpcode::Drop))
        .and_then(|step| step.operand_stack_words.last().copied())
}

fn split_after_first_end(steps: &[WasmtimeTraceStep]) -> Option<(&[WasmtimeTraceStep], &[WasmtimeTraceStep])> {
    let end = steps
        .iter()
        .position(|step| step.opcode_decoded == Some(WasmOpcode::End))?;
    Some(steps.split_at(end + 1))
}

/// Cross-instance imports must route rows and normalize funcrefs in each
/// instance's own module-local namespace.
#[test]
fn wasmtime_trace_routes_per_instance_with_per_instance_funcref_ids() {
    // `shared` is defined function index 1 (after `unused`) -> A funcref id 2.
    let wasm_a = wat::parse_str(
        r#"(module
            (func $unused (result i32) i32.const 1)
            (func (export "shared") (result i32)
                i32.const 3
                i32.const 4
                i32.add)
            (elem declare func 1)
            (func (export "run_a") (result i32)
                ref.func 1
                drop
                i32.const 0))
        "#,
    )
    .expect("wat a");

    // B imports A's `shared` as function index 0 -> B funcref id 1.
    let wasm_b = wat::parse_str(
        r#"(module
            (import "a" "shared" (func $s (result i32)))
            (elem declare func 0)
            (func $double (param i32) (result i32)
                local.get 0
                i32.const 2
                i32.mul)
            (func (export "run_b") (result i32)
                ref.func 0
                drop
                call $s
                call $double))
        "#,
    )
    .expect("wat b");

    let mut config = Config::new();
    config.guest_debug(true);
    config.wasm_reference_types(true);
    config.wasm_function_references(true);
    let engine = Engine::new(&config).expect("engine");

    let module_a = Module::from_binary(&engine, &wasm_a).expect("module a");
    let module_b = Module::from_binary(&engine, &wasm_b).expect("module b");

    let mut store = Store::new(
        &engine,
        MultiInstanceSink {
            traces: HashMap::new(),
            untraced: HashSet::new(),
        },
    );
    store.set_debug_handler(WasmtimeTraceHandler::<MultiInstanceSink>::new());
    store
        .edit_breakpoints()
        .expect("guest debug enabled")
        .single_step(true)
        .expect("single-step mode");

    // Register A with its own lowering state.
    let linker_a = Linker::new(&engine);
    let instance_a = futures::executor::block_on(linker_a.instantiate_async(&mut store, &module_a)).expect("inst a");
    let idx_a = instance_a.debug_index_in_store();
    let map_a = build_debug_function_id_map(&instance_a, &mut store).expect("funcref map a");
    let mut trace_a =
        WasmtimeTraceState::from_program_artifacts(&extract_wasm_program_artifacts(&wasm_a).expect("art a"));
    trace_a.set_func_ref_ids(map_a);
    store.data_mut().traces.insert(idx_a, trace_a);

    // Register B with A's `shared` wired in as its import.
    let shared = instance_a
        .get_func(&mut store, "shared")
        .expect("shared export");
    let mut linker_b = Linker::new(&engine);
    linker_b
        .define(&store, "a", "shared", shared)
        .expect("define import");
    let instance_b = futures::executor::block_on(linker_b.instantiate_async(&mut store, &module_b)).expect("inst b");
    let idx_b = instance_b.debug_index_in_store();
    let map_b = build_debug_function_id_map(&instance_b, &mut store).expect("funcref map b");
    let mut trace_b =
        WasmtimeTraceState::from_program_artifacts(&extract_wasm_program_artifacts(&wasm_b).expect("art b"));
    trace_b.set_func_ref_ids(map_b);
    store.data_mut().traces.insert(idx_b, trace_b);

    assert_ne!(idx_a, idx_b, "instances must have distinct debug indices");

    let run_a = instance_a.get_func(&mut store, "run_a").expect("run_a");
    futures::executor::block_on(run_a.call_async(&mut store, &[], &mut [Val::I32(0)])).expect("call run_a");
    let run_b = instance_b.get_func(&mut store, "run_b").expect("run_b");
    futures::executor::block_on(run_b.call_async(&mut store, &[], &mut [Val::I32(0)])).expect("call run_b");

    assert!(
        store.data().untraced.is_empty(),
        "unexpected untraced instances: {:?}",
        store.data().untraced
    );

    let trace_a = &store.data().traces[&idx_a];
    let trace_b = &store.data().traces[&idx_b];
    assert!(
        trace_a.steps().iter().any(|s| s.function_index.is_some()),
        "A captured no wasm frames"
    );
    assert!(
        trace_b.steps().iter().any(|s| s.function_index.is_some()),
        "B captured no wasm frames"
    );

    // Only executing frames are captured.
    assert!(
        trace_a.steps().iter().all(|s| s.frame_depth == 0),
        "A trace has non-innermost frames"
    );
    assert!(
        trace_b.steps().iter().all(|s| s.frame_depth == 0),
        "B trace has non-innermost frames"
    );

    // Distinctive ops stay in the instance that executed them.
    let a_ops: Vec<_> = trace_a
        .steps()
        .iter()
        .filter_map(|s| s.opcode_decoded)
        .collect();
    let b_ops: Vec<_> = trace_b
        .steps()
        .iter()
        .filter_map(|s| s.opcode_decoded)
        .collect();
    assert!(
        a_ops.contains(&WasmOpcode::I32Add),
        "A trace missing its own i32.add: {a_ops:?}"
    );
    assert!(
        !a_ops.contains(&WasmOpcode::I32Mul),
        "A trace contaminated with B's i32.mul: {a_ops:?}"
    );
    assert!(
        b_ops.contains(&WasmOpcode::I32Mul),
        "B trace missing its own i32.mul: {b_ops:?}"
    );
    assert!(
        !b_ops.contains(&WasmOpcode::I32Add),
        "B trace contaminated with A's i32.add: {b_ops:?}"
    );

    // The same raw function normalizes to each instance's own id.
    assert_eq!(
        funcref_seen_at_drop(trace_a),
        Some(2),
        "module A funcref id (own module-local)"
    );
    assert_eq!(
        funcref_seen_at_drop(trace_b),
        Some(1),
        "module B funcref id (own module-local)"
    );

    let (a_run_steps, a_shared_steps) = split_after_first_end(trace_a.steps()).expect("A run_a segment");
    let a_run_trace = traces_from_wasmtime_steps(a_run_steps).expect("normalize A run_a");
    let a_shared_trace = traces_from_wasmtime_steps(a_shared_steps).expect("normalize A shared");
    assert!(!a_run_trace.is_empty());
    assert!(!a_shared_trace.is_empty());
    // B calls across the instance boundary: from B's perspective the shared
    // function is a host import, so proving it needs an import template.
    let b_err = traces_from_wasmtime_steps(trace_b.steps()).expect_err("cross-instance import needs a template");
    assert!(b_err
        .to_string()
        .contains("no grammar template for host import"));
}

#[test]
fn wasmtime_steps_normalize_to_wasm_ir() {
    let trace = traces_from_wasmtime_steps(&sample_steps()).expect("normalize");
    assert_eq!(trace.len(), 4);

    assert_eq!(trace[0].opcode, WasmOpcode::I32Const);
    assert_eq!(trace[0].info.code, opcode_code(WasmOpcode::I32Const));
    assert_eq!(trace[0].stack_write0, Some(StackValueAccess::new(0, 7)));

    assert_eq!(trace[2].opcode, WasmOpcode::I32Add);
    assert_eq!(trace[2].stack_read0, Some(StackValueAccess::new(0, 7)));
    assert_eq!(trace[2].stack_read1, Some(StackValueAccess::new(2, 9)));
    assert_eq!(trace[2].stack_write0, Some(StackValueAccess::new(0, 16)));

    assert_eq!(trace[3].opcode, WasmOpcode::End);
    assert!(trace[3].state_after.halted);
}

#[test]
fn wasmtime_runtime_trace_normalizes_supported_rows() {
    let wasm = wat::parse_str(
        r#"(module
            (func (export "run") (result i32)
                i32.const 7
                i32.const 9
                i32.add)
        )"#,
    )
    .expect("wat");

    let run = collect_wasmtime_steps(&wasm, "run", &[]).expect("trace run");
    assert_eq!(run.results.as_slice(), &["16".to_string()]);
    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize wasmtime trace");
    let opcodes = trace.iter().map(|row| row.opcode).collect::<Vec<_>>();
    assert_eq!(
        opcodes,
        vec![
            WasmOpcode::I32Const,
            WasmOpcode::I32Const,
            WasmOpcode::I32Add,
            WasmOpcode::End,
        ]
    );
    assert_eq!(trace[2].stack_write0.unwrap().value_lo, 16);
    assert!(run
        .steps
        .iter()
        .any(|step| step.opcode.as_deref() == Some("I32Add")));
}

#[test]
fn wasmtime_trace_normalizes_local_get_and_set() {
    // A no-argument function that stores constants into locals then reads them back.
    let wasm = wat::parse_str(
        r#"(module
            (func (export "run") (result i32)
                (local i32 i32)
                i32.const 7
                local.set 0
                i32.const 9
                local.set 1
                local.get 0
                local.get 1
                i32.add)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let opcodes: Vec<_> = trace.iter().map(|r| r.opcode).collect();
    assert!(
        opcodes.contains(&WasmOpcode::LocalGet),
        "expected local.get in trace: {opcodes:?}"
    );
    assert!(
        opcodes.contains(&WasmOpcode::LocalSet),
        "expected local.set in trace: {opcodes:?}"
    );

    // All local.get rows must have a local_index and local_read_value.
    for row in trace.iter().filter(|r| r.opcode == WasmOpcode::LocalGet) {
        assert!(row.local_index.is_some(), "local.get missing local_index");
        assert!(row.local_read_value.is_some(), "local.get missing local_read_value");
        // The pushed value must match the local's pre-step value.
        assert_eq!(
            row.stack_write0.map(|w| w.value_lo),
            row.local_read_value,
            "local.get write != local_read_value"
        );
    }
    // All local.set rows must have a local_index and local_write_value.
    for row in trace.iter().filter(|r| r.opcode == WasmOpcode::LocalSet) {
        assert!(row.local_index.is_some(), "local.set missing local_index");
        assert!(row.local_write_value.is_some(), "local.set missing local_write_value");
        // The consumed stack value must match what is stored.
        assert_eq!(
            row.stack_read0.map(|r| r.value_lo),
            row.local_write_value,
            "local.set read0 != local_write_value"
        );
    }
}

#[test]
fn wasmtime_trace_normalizes_global_get_and_set() {
    let wasm = wat::parse_str(
        r#"(module
            (global (mut i32) (i32.const 7))
            (func (export "run") (result i32)
                global.get 0
                drop
                i32.const 9
                global.set 0
                global.get 0)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let get_rows = trace
        .iter()
        .filter(|r| r.opcode == WasmOpcode::GlobalGet)
        .collect::<Vec<_>>();
    let set_row = trace
        .iter()
        .find(|r| r.opcode == WasmOpcode::GlobalSet)
        .expect("global.set row");

    assert_eq!(get_rows.len(), 2, "expected two global.get rows");
    assert_eq!(get_rows[0].global_index, Some(0));
    assert_eq!(get_rows[0].global_read_value, Some(7));
    assert_eq!(get_rows[0].stack_write0.map(|w| w.value_lo), Some(7));
    assert_eq!(set_row.global_index, Some(0));
    assert_eq!(set_row.global_write_value, Some(9));
    assert_eq!(set_row.stack_read0.map(|r| r.value_lo), Some(9));
    assert_eq!(get_rows[1].global_read_value, Some(9));
    assert_eq!(get_rows[1].stack_write0.map(|w| w.value_lo), Some(9));
}

#[test]
fn float_global_initializer_is_rejected_at_parse() {
    let wasm = wat::parse_str(
        r#"(module
            (global f64 (f64.const 3.5))
            (func (export "run") (result i32)
                i32.const 1)
        )"#,
    )
    .expect("wat");

    let err = extract_wasm_program_artifacts(&wasm).expect_err("float global initializer must be rejected");
    assert!(
        err.to_string().contains("float initializer"),
        "expected float-initializer rejection, got: {err}",
    );
}

#[test]
fn function_type_arity_limit_is_checked_without_truncation() {
    let params_255 = vec!["i32"; 255].join(" ");
    let params_256 = vec!["i32"; 256].join(" ");
    let results_255 = vec!["i32"; 255].join(" ");
    let results_256 = vec!["i32"; 256].join(" ");

    let accepted_params = wat::parse_str(format!("(module (type (func (param {params_255}))))")).expect("wat");
    extract_wasm_program_artifacts(&accepted_params).expect("255 parameters are supported");
    let rejected_params = wat::parse_str(format!("(module (type (func (param {params_256}))))")).expect("wat");
    let err = extract_wasm_program_artifacts(&rejected_params).expect_err("256 parameters must be rejected");
    assert!(
        err.to_string()
            .contains("256 parameters; neo-wasm supports at most 255"),
        "unexpected parameter-limit error: {err}",
    );

    let accepted_results = wat::parse_str(format!("(module (type (func (result {results_255}))))")).expect("wat");
    extract_wasm_program_artifacts(&accepted_results).expect("255 results are supported");
    let rejected_results = wat::parse_str(format!("(module (type (func (result {results_256}))))")).expect("wat");
    let err = extract_wasm_program_artifacts(&rejected_results).expect_err("256 results must be rejected");
    assert!(
        err.to_string()
            .contains("256 results; neo-wasm supports at most 255"),
        "unexpected result-limit error: {err}",
    );
}

#[test]
fn wasmtime_trace_normalizes_memory_size_and_grow_rows() {
    let wasm = wat::parse_str(
        r#"(module
            (memory 1 3)
            (func (export "run") (result i32)
                memory.size
                drop
                i32.const 1
                memory.grow
                drop
                memory.size)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let size_rows = trace
        .iter()
        .filter(|r| r.opcode == WasmOpcode::MemorySize)
        .collect::<Vec<_>>();
    let grow_row = trace
        .iter()
        .find(|r| r.opcode == WasmOpcode::MemoryGrow)
        .expect("memory.grow row");

    assert_eq!(size_rows.len(), 2, "expected two memory.size rows");
    assert_eq!(size_rows[0].state_before.memory_pages, Some(1));
    assert_eq!(size_rows[0].state_after.memory_pages, Some(1));
    assert_eq!(size_rows[0].stack_write0.map(|w| w.value_lo), Some(1));
    assert_eq!(grow_row.state_before.memory_pages, Some(1));
    assert_eq!(grow_row.state_after.memory_pages, Some(2));
    assert_eq!(grow_row.stack_read0.map(|r| r.value_lo), Some(1));
    assert_eq!(grow_row.stack_write0.map(|w| w.value_lo), Some(1));
    assert_eq!(size_rows[1].state_before.memory_pages, Some(2));
    assert_eq!(size_rows[1].stack_write0.map(|w| w.value_lo), Some(2));
}

#[test]
fn wasmtime_trace_normalizes_table_size_rows() {
    let wasm = wat::parse_str(
        r#"(module
            (table 4 funcref)
            (func (export "run") (result i32)
                table.size 0)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let row = trace
        .iter()
        .find(|r| r.opcode == WasmOpcode::TableSize)
        .expect("table.size row");

    assert_eq!(row.table_id, Some(0));
    assert_eq!(row.table_size, Some(4));
    assert_eq!(row.stack_write0.map(|w| w.value_lo), Some(4));
}

/// `table.get` OOB raises the same `TableOutOfBounds` trap as `call_indirect`
/// OOB, but only the latter is modeled. The adapter must keep it a loud error
/// rather than presenting it as a clean terminal run with no results.
#[test]
fn table_get_out_of_bounds_stays_a_loud_error() {
    let wasm = wat::parse_str(
        r#"(module
            (table 1 funcref)
            (func (export "run") (result i32)
                i32.const 5
                table.get 0
                ref.is_null))"#,
    )
    .expect("valid WAT");
    let err = collect_wasmtime_steps(&wasm, "run", &[]).expect_err("table.get OOB must be a hard error");
    assert!(
        format!("{err:?}").contains("failed to execute"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn wasmtime_trace_normalizes_funcref_table_rows() {
    let wasm = wat::parse_str(
        r#"(module
            (type (func))
            (func $f)
            (elem declare funcref (ref.func $f))
            (table 1 funcref)
            (func (export "run") (result i32)
                i32.const 0
                ref.func $f
                table.set 0
                i32.const 0
                table.get 0
                drop
                i32.const 1)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let ref_func = trace
        .iter()
        .find(|r| r.opcode == WasmOpcode::RefFunc)
        .expect("ref.func row");
    let table_set = trace
        .iter()
        .find(|r| r.opcode == WasmOpcode::TableSet)
        .expect("table.set row");
    let table_get = trace
        .iter()
        .find(|r| r.opcode == WasmOpcode::TableGet)
        .expect("table.get row");

    assert_eq!(ref_func.stack_write0.map(|w| w.value_lo), Some(1));
    assert_eq!(ref_func.function_type_id, Some(1));
    assert_eq!(table_set.table_id, Some(0));
    assert_eq!(table_set.table_index, Some(0));
    assert_eq!(table_set.table_value, Some(1));
    assert_eq!(table_set.function_type_id, Some(1));
    assert_eq!(table_get.table_id, Some(0));
    assert_eq!(table_get.table_index, Some(0));
    assert_eq!(table_get.table_value, Some(1));
    assert_eq!(table_get.function_type_id, Some(1));
}

#[test]
fn wasmtime_trace_normalizes_call_indirect_row() {
    let wasm = wat::parse_str(
        r#"(module
            (type $t (func (result i32)))
            (func $f (type $t) (result i32)
                i32.const 7)
            (table 1 funcref)
            (elem (i32.const 0) func $f)
            (func (export "run") (result i32)
                i32.const 0
                call_indirect (type $t))
        )"#,
    )
    .expect("wat");

    let artifacts = extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = collect_wasmtime_steps(&wasm, "run", &[]).expect("trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let row = trace
        .iter()
        .find(|r| r.opcode == WasmOpcode::CallIndirect)
        .expect("call_indirect row");

    assert_eq!(row.table_id, Some(0));
    assert_eq!(row.table_index, Some(0));
    assert_eq!(row.table_value, Some(1));
    assert_eq!(row.function_type_id, Some(1));
    assert_eq!(row.call_indirect_type_index, Some(0));
    assert_eq!(row.expected_type_id, Some(1));
    assert_eq!(row.stack_reads_override, Some(1));
    assert_eq!(
        artifacts
            .tables
            .function_entries
            .iter()
            .find(|(f, _)| *f == 1)
            .map(|(_, pc)| *pc),
        Some(row.state_after.pc)
    );
}

#[test]
fn wasmtime_trace_normalizes_basic_i64_rows() {
    let wasm = wat::parse_str(
        r#"(module
            (func (export "run") (result i32)
                i64.const 4294967295
                i64.const 1
                i64.add
                drop
                i64.const 6
                i64.const 7
                i64.mul
                drop
                i64.const 0x00ff00ff00ff00ff
                i64.const 0x0f0f0f0f0f0f0f0f
                i64.and
                drop
                i64.const 0x00ff00ff00ff00ff
                i64.const 0x0f0f0f0f0f0f0f0f
                i64.or
                drop
                i64.const 0x00ff00ff00ff00ff
                i64.const 0x0f0f0f0f0f0f0f0f
                i64.xor
                drop
                i64.const 4294967296
                i64.const 4294967296
                i64.sub
                i64.eqz)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let add = trace
        .iter()
        .find(|r| r.opcode == WasmOpcode::I64Add)
        .expect("i64.add row");
    let sub = trace
        .iter()
        .find(|r| r.opcode == WasmOpcode::I64Sub)
        .expect("i64.sub row");
    let mul = trace
        .iter()
        .find(|r| r.opcode == WasmOpcode::I64Mul)
        .expect("i64.mul row");
    let and = trace
        .iter()
        .find(|r| r.opcode == WasmOpcode::I64And)
        .expect("i64.and row");
    let or = trace
        .iter()
        .find(|r| r.opcode == WasmOpcode::I64Or)
        .expect("i64.or row");
    let xor = trace
        .iter()
        .find(|r| r.opcode == WasmOpcode::I64Xor)
        .expect("i64.xor row");
    let eqz = trace
        .iter()
        .find(|r| r.opcode == WasmOpcode::I64Eqz)
        .expect("i64.eqz row");

    assert!(add.wide_values_enabled);
    assert_eq!(add.stack_read0.map(|w| w.value_lo), Some(0xffff_ffff));
    assert_eq!(add.stack_read0.and_then(|lane| lane.value_hi), Some(0));
    assert_eq!(add.stack_read1.map(|w| w.value_lo), Some(1));
    assert_eq!(add.stack_write0.map(|w| w.value_lo), Some(0));
    assert_eq!(add.stack_write0.and_then(|lane| lane.value_hi), Some(1));

    assert!(sub.wide_values_enabled);
    assert_eq!(sub.stack_read0.map(|w| w.value_lo), Some(0));
    assert_eq!(sub.stack_read0.and_then(|lane| lane.value_hi), Some(1));
    assert_eq!(sub.stack_read1.map(|w| w.value_lo), Some(0));
    assert_eq!(sub.stack_read1.and_then(|lane| lane.value_hi), Some(1));
    assert_eq!(sub.stack_write0.map(|w| w.value_lo), Some(0));
    assert_eq!(sub.stack_write0.and_then(|lane| lane.value_hi), Some(0));

    assert!(mul.wide_values_enabled);
    assert_eq!(mul.stack_read0.map(|w| w.value_lo), Some(6));
    assert_eq!(mul.stack_read1.map(|w| w.value_lo), Some(7));
    assert_eq!(mul.stack_write0.map(|w| w.value_lo), Some(42));
    assert_eq!(mul.stack_write0.and_then(|lane| lane.value_hi), Some(0));

    assert!(and.wide_values_enabled);
    assert_eq!(and.stack_write0.map(|w| w.value_lo), Some(0x000f000f));
    assert_eq!(and.stack_write0.and_then(|lane| lane.value_hi), Some(0x000f000f));

    assert!(or.wide_values_enabled);
    assert_eq!(or.stack_write0.map(|w| w.value_lo), Some(0x0fff0fff));
    assert_eq!(or.stack_write0.and_then(|lane| lane.value_hi), Some(0x0fff0fff));

    assert!(xor.wide_values_enabled);
    assert_eq!(xor.stack_write0.map(|w| w.value_lo), Some(0x0ff00ff0));
    assert_eq!(xor.stack_write0.and_then(|lane| lane.value_hi), Some(0x0ff00ff0));

    assert!(eqz.wide_values_enabled);
    assert_eq!(eqz.stack_read0.map(|w| w.value_lo), Some(0));
    assert_eq!(eqz.stack_read0.and_then(|lane| lane.value_hi), Some(0));
    assert_eq!(eqz.stack_write0.map(|w| w.value_lo), Some(1));
    assert_eq!(eqz.stack_write0.and_then(|lane| lane.value_hi), Some(0));
}

#[test]
fn wasmtime_trace_normalizes_aligned_i64_memory_rows() {
    let wasm = wat::parse_str(
        r#"(module
            (memory 1)
            (func (export "run") (result i32)
                i32.const 8
                i64.const 0x1122334455667788
                i64.store
                i32.const 8
                i64.load
                i64.const 0x1122334455667788
                i64.sub
                i64.eqz)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let store = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I64Store)
        .expect("i64.store row");
    let load = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I64Load)
        .expect("i64.load row");

    assert!(store.wide_values_enabled);
    assert_eq!(store.stack_read0.map(|w| w.value_lo), Some(8));
    assert_eq!(store.stack_read1.map(|w| w.value_lo), Some(0x5566_7788));
    assert_eq!(store.stack_read1.and_then(|lane| lane.value_hi), Some(0x1122_3344));
    let store_mem = store.linear_memory.expect("store memory");
    assert_eq!(store_mem.width_bytes, 8);
    assert_eq!(store_mem.byte_offset, 0);
    assert_eq!(store_mem.lane0.value_after, 0x5566_7788);
    assert_eq!(store_mem.lane1.expect("store lane1").value_after, 0x1122_3344);

    assert!(load.wide_values_enabled);
    assert_eq!(load.stack_read0.map(|w| w.value_lo), Some(8));
    assert_eq!(load.stack_write0.map(|w| w.value_lo), Some(0x5566_7788));
    assert_eq!(load.stack_write0.and_then(|lane| lane.value_hi), Some(0x1122_3344));
    let load_mem = load.linear_memory.expect("load memory");
    assert_eq!(load_mem.width_bytes, 8);
    assert_eq!(load_mem.byte_offset, 0);
    assert_eq!(load_mem.lane0.value_before, 0x5566_7788);
    assert_eq!(load_mem.lane1.expect("load lane1").value_before, 0x1122_3344);
}

#[test]
fn wasmtime_trace_normalizes_unaligned_i64_memory_rows() {
    let wasm = wat::parse_str(
        r#"(module
            (memory 1)
            (func (export "run") (result i32)
                i32.const 9
                i64.const 0x1122334455667788
                i64.store
                i32.const 9
                i64.load
                i64.const 0x1122334455667788
                i64.sub
                i64.eqz)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let store = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I64Store)
        .expect("i64.store row");
    let load = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I64Load)
        .expect("i64.load row");

    let store_mem = store.linear_memory.expect("store memory");
    assert_eq!(store_mem.width_bytes, 8);
    assert_eq!(store_mem.byte_offset, 1);
    assert_eq!(store_mem.lane0.value_after, 0x6677_8800);
    assert_eq!(store_mem.lane1.expect("store lane1").value_after, 0x2233_4455);
    assert_eq!(store_mem.lane2.expect("store lane2").value_after, 0x0000_0011);

    let load_mem = load.linear_memory.expect("load memory");
    assert_eq!(load_mem.width_bytes, 8);
    assert_eq!(load_mem.byte_offset, 1);
    assert_eq!(load_mem.lane0.value_before, 0x6677_8800);
    assert_eq!(load_mem.lane1.expect("load lane1").value_before, 0x2233_4455);
    assert_eq!(load_mem.lane2.expect("load lane2").value_before, 0x0000_0011);
    assert_eq!(load.stack_write0.map(|w| w.value_lo), Some(0x5566_7788));
    assert_eq!(load.stack_write0.and_then(|lane| lane.value_hi), Some(0x1122_3344));
}

#[test]
fn wasmtime_trace_normalizes_shift_div_rem_rows() {
    let wasm = wat::parse_str(
        r#"(module
            (func (export "run") (result i32)
                i32.const 3
                i32.const 4
                i32.shl
                drop
                i32.const 128
                i32.const 3
                i32.shr_u
                drop
                i32.const -128
                i32.const 3
                i32.shr_s
                drop
                i32.const 22
                i32.const 5
                i32.div_u
                drop
                i32.const -22
                i32.const 5
                i32.div_s
                drop
                i32.const 22
                i32.const 5
                i32.rem_u
                drop
                i32.const -22
                i32.const 5
                i32.rem_s
                drop
                i32.const 123)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let opcodes = trace.iter().map(|r| r.opcode).collect::<Vec<_>>();
    for op in [
        WasmOpcode::I32Shl,
        WasmOpcode::I32ShrU,
        WasmOpcode::I32ShrS,
        WasmOpcode::I32DivU,
        WasmOpcode::I32DivS,
        WasmOpcode::I32RemU,
        WasmOpcode::I32RemS,
    ] {
        assert!(opcodes.contains(&op), "missing {op:?} in trace: {opcodes:?}");
    }
    let expected = [
        (WasmOpcode::I32Shl, 48),
        (WasmOpcode::I32ShrU, 16),
        (WasmOpcode::I32ShrS, 0xffff_fff0),
        (WasmOpcode::I32DivU, 4),
        (WasmOpcode::I32DivS, 0xffff_fffc),
        (WasmOpcode::I32RemU, 2),
        (WasmOpcode::I32RemS, 0xffff_fffe),
    ];
    for (opcode, output) in expected {
        let row = trace
            .iter()
            .find(|r| r.opcode == opcode)
            .expect("opcode row");
        assert_eq!(
            row.stack_write0.map(|w| w.value_lo),
            Some(output),
            "wrong output for {opcode:?}"
        );
    }
}

#[test]
fn wasmtime_trace_normalizes_compare_unary_and_rotate_rows() {
    let wasm = wat::parse_str(
        r#"(module
            (func (export "run") (result i32)
                i32.const 16
                i32.clz
                drop
                i32.const 24
                i32.ctz
                drop
                i32.const 9
                i32.const 3
                i32.gt_s
                drop
                i32.const 9
                i32.const 3
                i32.gt_u
                drop
                i32.const 3
                i32.const 9
                i32.le_s
                drop
                i32.const 3
                i32.const 9
                i32.le_u
                drop
                i32.const 9
                i32.const 3
                i32.ge_s
                drop
                i32.const 9
                i32.const 3
                i32.ge_u
                drop
                i32.const 305419896
                i32.const 8
                i32.rotl
                drop
                i32.const 305419896
                i32.const 8
                i32.rotr
                drop
                i32.const 123)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let opcodes = trace.iter().map(|r| r.opcode).collect::<Vec<_>>();
    for op in [
        WasmOpcode::I32Clz,
        WasmOpcode::I32Ctz,
        WasmOpcode::I32GtS,
        WasmOpcode::I32GtU,
        WasmOpcode::I32LeS,
        WasmOpcode::I32LeU,
        WasmOpcode::I32GeS,
        WasmOpcode::I32GeU,
        WasmOpcode::I32Rotl,
        WasmOpcode::I32Rotr,
    ] {
        assert!(opcodes.contains(&op), "missing {op:?} in trace: {opcodes:?}");
    }
    let expected = [
        (WasmOpcode::I32Clz, 27),
        (WasmOpcode::I32Ctz, 3),
        (WasmOpcode::I32GtS, 1),
        (WasmOpcode::I32GtU, 1),
        (WasmOpcode::I32LeS, 1),
        (WasmOpcode::I32LeU, 1),
        (WasmOpcode::I32GeS, 1),
        (WasmOpcode::I32GeU, 1),
        (WasmOpcode::I32Rotl, 0x3456_7812),
        (WasmOpcode::I32Rotr, 0x7812_3456),
    ];
    for (opcode, output) in expected {
        let row = trace
            .iter()
            .find(|r| r.opcode == opcode)
            .expect("opcode row");
        assert_eq!(
            row.stack_write0.map(|w| w.value_lo),
            Some(output),
            "wrong output for {opcode:?}"
        );
    }
}

#[test]
fn wasmtime_trace_normalizes_br_table_rows() {
    // One module per selector value: the selector is a baked-in constant
    // (plain-mode locals start all-zero, so it cannot be an entry param).
    let wat_for = |selector: i32| {
        format!(
            r#"(module
            (func (export "run") (result i32)
                (block $default
                    (block $case1
                        (block $case0
                            i32.const {selector}
                            br_table $case0 $case1 $default
                        )
                        i32.const 10
                        return
                    )
                    i32.const 20
                    return
                )
                i32.const 30))
        "#
        )
    };

    for (param, expected_value, expected_choice) in [(0, 10, 1_u32), (1, 20, 2_u32), (5, 30, 0_u32)] {
        let wasm = wat::parse_str(wat_for(param)).expect("wat");
        let artifacts = extract_wasm_program_artifacts(&wasm).expect("program artifacts");
        let run = collect_wasmtime_steps(&wasm, "run", &[]).expect("trace run");
        let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
        let row = trace
            .iter()
            .find(|row| row.opcode == WasmOpcode::BrTable)
            .expect("br_table row");
        assert_eq!(
            row.control_choice, expected_choice,
            "wrong control choice for param={param}"
        );
        let edges_from_row = artifacts
            .tables
            .pc_rom
            .iter()
            .filter(|(pc, _, _)| *pc == row.state_before.pc)
            .count();
        assert_eq!(edges_from_row, 3, "expected three br_table edges in pc rom");
        assert_eq!(
            run.results.as_slice(),
            &[expected_value.to_string()],
            "wrong final value for param={param}"
        );
    }
}

#[test]
fn wasmtime_trace_keeps_structured_control_rows() {
    let wasm = wat::parse_str(
        r#"(module
            (func (export "run") (result i32)
                block
                    i32.const 1
                    if
                        i32.const 7
                        drop
                    else
                        i32.const 9
                        drop
                    end
                end
                i32.const 5)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize wasmtime trace");
    let opcodes = trace.iter().map(|row| row.opcode).collect::<Vec<_>>();
    assert_eq!(
        opcodes,
        vec![
            WasmOpcode::Block,
            WasmOpcode::I32Const,
            WasmOpcode::If,
            WasmOpcode::I32Const,
            WasmOpcode::Drop,
            WasmOpcode::Else,
            WasmOpcode::End,
            WasmOpcode::I32Const,
            WasmOpcode::End,
        ]
    );
}

#[test]
fn wasmtime_trace_keeps_nop_and_br_rows() {
    let wasm = wat::parse_str(
        r#"(module
            (func (export "run") (result i32)
                block
                    nop
                    br 0
                    i32.const 9
                    drop
                end
                i32.const 5)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize wasmtime trace");
    let opcodes = trace.iter().map(|row| row.opcode).collect::<Vec<_>>();
    assert_eq!(
        opcodes,
        vec![
            WasmOpcode::Block,
            WasmOpcode::Nop,
            WasmOpcode::Br,
            WasmOpcode::I32Const,
            WasmOpcode::End,
        ]
    );
}

#[test]
fn wasmtime_trace_normalizes_byte_memory_rows() {
    let wasm = wat::parse_str(
        r#"(module
            (memory 1)
            (func (export "run") (result i32)
                i32.const 0
                i32.const 511
                i32.store8
                i32.const 0
                i32.load8_u)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize wasmtime trace");
    let opcodes = trace.iter().map(|row| row.opcode).collect::<Vec<_>>();
    assert!(opcodes.contains(&WasmOpcode::I32Store8));
    assert!(opcodes.contains(&WasmOpcode::I32Load8U));
    let load = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Load8U)
        .expect("load8_u row");
    assert_eq!(load.stack_write0.expect("load result").value_lo, 255);
}

#[test]
fn wasmtime_trace_normalizes_halfword_memory_rows() {
    let wasm = wat::parse_str(
        r#"(module
            (memory 1)
            (func (export "run") (result i32)
                i32.const 3
                i32.const 4660
                i32.store16
                i32.const 3
                i32.load16_u)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize wasmtime trace");
    let opcodes = trace.iter().map(|row| row.opcode).collect::<Vec<_>>();
    assert!(opcodes.contains(&WasmOpcode::I32Store16));
    assert!(opcodes.contains(&WasmOpcode::I32Load16U));
    let load = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Load16U)
        .expect("load16_u row");
    assert_eq!(load.stack_write0.expect("load result").value_lo, 4660);
    assert!(load.linear_memory.expect("linear memory").lane1.is_some());
}

#[test]
fn wasmtime_trace_normalizes_signed_subword_load_rows() {
    let wasm = wat::parse_str(
        r#"(module
            (memory 1)
            (func (export "run") (result i32)
                i32.const 0
                i32.const 129
                i32.store8
                i32.const 0
                i32.load8_s
                drop
                i32.const 3
                i32.const 32769
                i32.store16
                i32.const 3
                i32.load16_s)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize wasmtime trace");
    let load8 = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Load8S)
        .expect("load8_s row");
    let load16 = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Load16S)
        .expect("load16_s row");
    assert_eq!(load8.stack_write0.expect("load8 result").value_lo, (-127i32) as u32);
    assert_eq!(load16.stack_write0.expect("load16 result").value_lo, (-32767i32) as u32);
    assert!(load16.linear_memory.expect("linear memory").lane1.is_some());
}

#[test]
fn pc_rom_keeps_both_if_edges_and_else_fallthrough() {
    let wasm = wat::parse_str(
        r#"(module
            (func (export "run") (result i32)
                i32.const 1
                if
                    i32.const 7
                    drop
                else
                    i32.const 9
                    drop
                end
                i32.const 5)
        )"#,
    )
    .expect("wat");

    let rom = build_pc_rom_from_binary(&wasm).expect("pc rom");

    let mut if_pc = None;
    let mut if_then_pc = None;
    let mut else_pc = None;
    let mut else_body_pc = None;
    let mut end_pc = None;
    let mut after_end_pc = None;

    for payload in Parser::new(0).parse_all(&wasm) {
        let payload = payload.expect("payload");
        let Payload::CodeSectionEntry(body) = payload else {
            continue;
        };
        let mut reader = body.get_operators_reader().expect("operators");
        while !reader.eof() {
            let pc_before = reader.original_position() as u64;
            let operator = reader.read().expect("operator");
            let pc_after = reader.original_position() as u64;
            match operator {
                wasmparser::Operator::If { .. } => {
                    if_pc = Some(pc_before);
                    if_then_pc = Some(pc_after);
                }
                wasmparser::Operator::Else => {
                    else_pc = Some(pc_before);
                    else_body_pc = Some(pc_after);
                }
                wasmparser::Operator::End => {
                    end_pc = Some(pc_before);
                    after_end_pc = Some(pc_after);
                    break;
                }
                _ => {}
            }
        }
    }

    let if_pc = if_pc.expect("if pc");
    let if_then_pc = if_then_pc.expect("if then pc");
    let else_pc = else_pc.expect("else pc");
    let else_body_pc = else_body_pc.expect("else body pc");
    let end_pc = end_pc.expect("end pc");
    let after_end_pc = after_end_pc.expect("after end pc");

    assert!(rom.contains(&(if_pc, 1, if_then_pc)), "if true edge missing: {rom:?}");
    assert!(
        rom.contains(&(if_pc, 0, else_body_pc)),
        "if false edge missing: {rom:?}"
    );
    assert!(
        rom.contains(&(else_pc, 0, after_end_pc)),
        "else fallthrough edge missing: {rom:?}"
    );
    assert!(rom.contains(&(end_pc, 0, after_end_pc)), "end edge missing: {rom:?}");
}
