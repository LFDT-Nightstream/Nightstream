use super::*;
use nightstream::nebula::Circuit;

const LARGE_GET: &str =
    r#"(module (table 6 funcref) (func (export "main") (result i32) i32.const 5 table.get 0 drop i32.const 7))"#;
const SMALL_GET: &str =
    r#"(module (table 1 funcref) (func (export "main") (result i32) i32.const 5 table.get 0 drop i32.const 7))"#;
const SMALL_SET: &str = r#"(module (func $f) (elem declare funcref (ref.func $f)) (table 1 funcref) (func (export "main") (result i32) i32.const 5 ref.func $f table.set 0 i32.const 7))"#;

#[test]
fn table_access_past_declared_size_is_rejected_by_public_verifier() {
    let cases = [
        ("table.get", LARGE_GET, SMALL_GET),
        (
            "table.set",
            r#"(module (func $f) (elem declare funcref (ref.func $f)) (table 6 funcref) (func (export "main") (result i32) i32.const 5 ref.func $f table.set 0 i32.const 7))"#,
            SMALL_SET,
        ),
    ];
    let mut accepted = Vec::new();

    for (name, large_wat, small_wat) in cases {
        let small = wat::parse_str(small_wat).unwrap();
        let engine = wasmtime::Engine::default();
        let module = wasmtime::Module::new(&engine, &small).unwrap();
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
        let main = instance
            .get_typed_func::<(), i32>(&mut store, "main")
            .unwrap();
        assert!(main.call(&mut store, ()).is_err(), "{name} must trap at index 5");

        let artifacts = neo_wasm::extract_wasm_program_artifacts(&small).unwrap();
        let (export_fref, entry_pc) = artifacts
            .tables
            .function_entries
            .iter()
            .copied()
            .max_by_key(|(_, pc)| *pc)
            .unwrap();
        let setup = setup_of(large_wat, NonZeroUsize::MIN, false);
        let (app, geometry) = setup.complete();
        let traced = &setup.programs[0];
        assert_eq!(top_level_initial_state(&artifacts.tables, entry_pc), traced.initial);
        assert!(traced.last.is_terminal());
        assert!(traced.last.output.enabled);
        assert!(!traced.last.trapped);
        assert_eq!(traced.last.output.value_lo, 7);

        let mut small_preload = preload_from_program_artifacts(&artifacts);
        preload_host_event_tables(
            &mut small_preload,
            &HostEventBindings::import_free(u32::try_from(export_fref).unwrap()),
        );
        let small_program = app.program(&small_preload).unwrap();
        let circuit = Circuit::compile(geometry, app.clone()).unwrap();
        let prover = circuit.prover(nightstream::Engine::Optimized, 109).unwrap();
        let verifier = circuit
            .verifier(nightstream::Engine::Optimized, 109)
            .unwrap();

        let verified = prover
            .start(&small_program, traced.initial)
            .ok()
            .and_then(|mut run| {
                run.segment(&traced.inputs).ok()?;
                run.finish().ok()
            })
            .is_some_and(|proof| {
                verifier
                    .verify(&small_program, &traced.initial, &traced.last, &proof)
                    .is_ok()
            });
        if verified {
            accepted.push(name);
        }
    }

    assert!(
        accepted.is_empty(),
        "the public verifier accepted a normal return for out-of-bounds table accesses: {accepted:?}"
    );
}

#[test]
fn honest_out_of_bounds_table_traps_have_public_proofs() {
    for (name, wasm) in [("table.get", SMALL_GET), ("table.set", SMALL_SET)] {
        let setup = setup_of(wasm, NonZeroUsize::MIN, false);
        let (app, geometry) = setup.complete();
        let traced = &setup.programs[0];
        assert!(traced.last.is_terminal() && traced.last.trapped, "{name}");
        assert!(!traced.last.output.enabled, "{name}");

        let program = app.program(&traced.preload).unwrap();
        let circuit = Circuit::compile(geometry, app).unwrap();
        let prover = circuit.prover(nightstream::Engine::Optimized, 109).unwrap();
        let verifier = circuit
            .verifier(nightstream::Engine::Optimized, 109)
            .unwrap();
        let mut run = prover.start(&program, traced.initial).unwrap();
        run.segment(&traced.inputs)
            .unwrap_or_else(|error| panic!("the honest {name} trap has no proof: {error:?}"));
        let proof = run.finish().unwrap();
        verifier
            .verify(&program, &traced.initial, &traced.last, &proof)
            .unwrap_or_else(|error| panic!("the honest {name} trap does not verify: {error:?}"));
    }
}

#[test]
#[ignore = "full fixed-size WASM proof; run separately with the 900-second cap"]
fn fixed_small_size_rejects_table_get_past_declared_size() {
    let small = wat::parse_str(SMALL_GET).unwrap();
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &small).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    let main = instance
        .get_typed_func::<(), i32>(&mut store, "main")
        .unwrap();
    assert!(main.call(&mut store, ()).is_err());

    let checked = common::checked_main(LARGE_GET);
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&small).unwrap();
    let (export_fref, entry_pc) = artifacts
        .tables
        .function_entries
        .iter()
        .copied()
        .max_by_key(|(_, pc)| *pc)
        .unwrap();
    let initial = top_level_initial_state(&artifacts.tables, entry_pc);
    let large_entry_pc = checked.artifacts.tables.function_entries[0].1;
    assert_eq!(
        initial,
        top_level_initial_state(&checked.artifacts.tables, large_entry_pc)
    );
    let last = checked.trace.last().unwrap().state_after;
    assert!(last.is_terminal() && last.output.enabled && !last.trapped);
    assert_eq!(last.output.value_lo, 7);

    let mut preload = preload_from_program_artifacts(&artifacts);
    preload_host_event_tables(
        &mut preload,
        &HostEventBindings::import_free(u32::try_from(export_fref).unwrap()),
    );
    let size = neo_wasm::nebula::WasmSize::Small;
    let app = size.application().unwrap();
    let program = app.program(&preload).unwrap();
    let circuit = Circuit::compile(size.geometry(), app).unwrap();
    let prover = circuit.prover(nightstream::Engine::Metal, 109).unwrap();
    let verifier = circuit.verifier(nightstream::Engine::Metal, 109).unwrap();
    let segments = size.segments(&checked.trace).unwrap();
    let accepted = prover
        .start(&program, initial)
        .ok()
        .and_then(|mut run| {
            for segment in &segments {
                run.segment(segment).ok()?;
            }
            run.finish().ok()
        })
        .is_some_and(|proof| verifier.verify(&program, &initial, &last, &proof).is_ok());
    assert!(
        !accepted,
        "the fixed Small WASM verifier accepted a normal return after an out-of-bounds table.get"
    );
}
