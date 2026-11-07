use super::*;
use neo_wasm::WasmOpcode;
use nightstream::nebula::Circuit;

const GROW: &str = r#"(module
    (memory (export "mem") 1 3)
    (func (export "try_grow") (result i32)
        i32.const 1
        memory.grow)
    (func (export "main") (result i32)
        i32.const 1
        memory.grow
        drop
        i32.const 7))"#;

#[test]
fn legal_memory_grow_allocation_failure_has_a_public_proof() {
    let wasm = wat::parse_str(GROW).unwrap();
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    let limits = wasmtime::StoreLimitsBuilder::new()
        .memory_size(65_536)
        .build();
    let mut store = wasmtime::Store::new(&engine, limits);
    store.limiter(|limits| limits);
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    let try_grow = instance
        .get_typed_func::<(), i32>(&mut store, "try_grow")
        .unwrap();
    assert_eq!(try_grow.call(&mut store, ()).unwrap(), -1);
    let main = instance
        .get_typed_func::<(), i32>(&mut store, "main")
        .unwrap();
    assert_eq!(main.call(&mut store, ()).unwrap(), 7);
    assert_eq!(instance.get_memory(&mut store, "mem").unwrap().size(&store), 1);

    let mut checked = common::checked_main(GROW);
    let grow_index = checked
        .trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::MemoryGrow)
        .unwrap();
    assert_eq!(checked.trace[grow_index].state_before.memory_pages, Some(1));
    assert_eq!(checked.trace[grow_index].state_after.memory_pages, Some(2));
    checked.trace[grow_index].state_after.memory_pages = Some(1);
    checked.trace[grow_index]
        .stack_write0
        .as_mut()
        .unwrap()
        .value_lo = u32::MAX;

    let mut changed_drop = false;
    for row in &mut checked.trace[grow_index + 1..] {
        row.state_before.memory_pages = Some(1);
        row.state_after.memory_pages = Some(1);
        if !changed_drop && row.opcode == WasmOpcode::Drop {
            row.stack_read0.as_mut().unwrap().value_lo = u32::MAX;
            changed_drop = true;
        }
    }
    assert!(changed_drop);
    assert!(checked.trace.last().unwrap().state_after.is_terminal());
    assert_eq!(checked.trace.last().unwrap().state_after.output.value_lo, 7);

    let relation = neo_wasm::build_wasm_relation().unwrap();
    let ccs = relation.r1cs().structure();
    let public_words = relation.r1cs().public_input_count();
    let mut grow_error = None;
    for (index, row) in checked.trace.iter().enumerate() {
        let witness = neo_wasm::witness_builder::build_witness_vector(row);
        let result = neo_ccs::check_ccs_rowwise_zero(ccs, &witness[..public_words], &witness[public_words..]);
        if index == grow_index {
            grow_error = result.err().map(|error| error.to_string());
        } else {
            assert!(
                result.is_ok(),
                "the resource-limited trace has an unrelated invalid row at {index}: {result:?}"
            );
        }
    }

    let (export_fref, entry_pc) = checked
        .artifacts
        .tables
        .function_entries
        .iter()
        .copied()
        .max_by_key(|(_, pc)| *pc)
        .unwrap();
    let mut preload = preload_from_program_artifacts(&checked.artifacts);
    preload_host_event_tables(
        &mut preload,
        &HostEventBindings::import_free(u32::try_from(export_fref).unwrap()),
    );
    let n = checked.trace.len().next_power_of_two();
    let per_step = NonZeroUsize::MIN;
    let inputs = step_inputs(&checked.trace, n, per_step).unwrap();
    let traced = Traced {
        states: inputs.iter().map(|row| row.next).collect(),
        rows: inputs.iter().map(|row| row.z.clone()).collect(),
        inputs,
        real: checked.trace.len(),
        preload,
        initial: top_level_initial_state(&checked.artifacts.tables, entry_pc),
        last: checked.trace.last().unwrap().state_after,
    };
    let setup = Setup::of(vec![traced], n, per_step);
    let (app, geometry) = setup.complete();
    let traced = &setup.programs[0];
    let program = app.program(&traced.preload).unwrap();
    let circuit = Circuit::compile(geometry, app).unwrap();
    let prover = circuit.prover(nightstream::Engine::Optimized, 109).unwrap();
    let verifier = circuit
        .verifier(nightstream::Engine::Optimized, 109)
        .unwrap();
    let mut run = prover.start(&program, traced.initial).unwrap();
    let result = run.segment(&traced.inputs);
    assert!(
        result.is_ok(),
        "the public prover rejected a legal resource-limited memory.grow failure: {result:?}; grow CCS: {grow_error:?}"
    );
    let proof = run.finish().unwrap();
    assert!(
        verifier
            .verify(&program, &traced.initial, &traced.last, &proof)
            .is_ok(),
        "the public verifier rejected a legal resource-limited memory.grow failure"
    );
}
