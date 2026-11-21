mod common;

#[path = "wasm_generated_modules/module.rs"]
mod module;
#[path = "wasm_generated_modules/types.rs"]
mod types;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use neo_wasm::host_event_bindings::HostEventBindings;
use neo_wasm::{collect_wasmtime_steps, extract_wasm_program_artifacts, traces_from_wasmtime_steps, WasmProgramError};
use proptest::collection::vec;
use proptest::prelude::any;
use proptest::test_runner::{Config, RngAlgorithm, TestCaseError, TestError, TestRng, TestRunner};
use types::Choices;

const SEED: [u8; 32] = [0x5e; 32];
const CHOICES: usize = 4096;

fn runner(cases: u32) -> TestRunner {
    let config = Config {
        cases,
        failure_persistence: None,
        max_shrink_iters: 2000,
        max_shrink_time: 120_000,
        ..Config::default()
    };
    TestRunner::new_with_rng(config, TestRng::from_seed(RngAlgorithm::ChaCha, &SEED))
}

fn engine() -> wasmtime::Engine {
    let mut config = wasmtime::Config::new();
    config.wasm_reference_types(true);
    config.wasm_function_references(true);
    wasmtime::Engine::new(&config).unwrap()
}

#[derive(Clone, Debug, PartialEq)]
enum Value {
    I32(i32),
    I64(i64),
    Other(String),
}

type Run = Result<Vec<Value>, String>;

struct Instance {
    store: wasmtime::Store<()>,
    instance: wasmtime::Instance,
}

fn instantiate(engine: &wasmtime::Engine, wasm: &[u8]) -> Result<Instance, TestCaseError> {
    let module = wasmtime::Module::new(engine, wasm)
        .map_err(|err| TestCaseError::fail(format!("the generator made a module that Wasmtime refuses: {err:?}")))?;
    let mut store = wasmtime::Store::new(engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[])
        .map_err(|err| TestCaseError::fail(format!("Wasmtime cannot instantiate the module: {err:?}")))?;
    Ok(Instance { store, instance })
}

impl Instance {
    fn run_main(&mut self) -> Result<Run, TestCaseError> {
        let main = self.instance.get_func(&mut self.store, "main").unwrap();
        let mut results = vec![wasmtime::Val::I32(0); main.ty(&self.store).results().len()];
        match main.call(&mut self.store, &[], &mut results) {
            Ok(()) => Ok(Ok(results
                .iter()
                .map(|value| match value {
                    wasmtime::Val::I32(v) => Value::I32(*v),
                    wasmtime::Val::I64(v) => Value::I64(*v),
                    other => Value::Other(format!("{other:?}")),
                })
                .collect())),
            Err(err) => match err.downcast_ref::<wasmtime::Trap>() {
                Some(trap) => Ok(Err(format!("{trap:?}"))),
                None => Err(TestCaseError::fail(format!("Wasmtime failed without a trap: {err:?}"))),
            },
        }
    }

    fn slot_types(&mut self) -> Vec<wasmtime::FuncType> {
        let table = self.instance.get_table(&mut self.store, "types").unwrap();
        (0..table.size(&self.store))
            .map(|slot| match table.get(&mut self.store, slot) {
                Some(wasmtime::Ref::Func(Some(func))) => func.ty(&self.store),
                other => panic!("slot {slot} holds {other:?}"),
            })
            .collect()
    }
}

#[derive(Debug)]
enum Outcome {
    Refused(String),
    Returned,
    Trapped(String),
}

fn differential(wasm: &[u8]) -> Result<Outcome, TestCaseError> {
    let expected = instantiate(&engine(), wasm)?.run_main()?;
    match extract_wasm_program_artifacts(wasm) {
        Ok(_) => {}
        Err(WasmProgramError::Unsupported(reason)) => return Ok(Outcome::Refused(reason)),
        Err(WasmProgramError::UnsupportedOpcode { operator, .. }) => return Ok(Outcome::Refused(operator)),
        Err(WasmProgramError::Invalid(reason)) => {
            return Err(TestCaseError::fail(format!(
                "the parser calls a module that Wasmtime runs invalid: {reason}"
            )))
        }
    }
    let run = collect_wasmtime_steps(wasm, &HostEventBindings::default(), "main", &[])
        .map_err(|err| TestCaseError::fail(format!("capture failed; Wasmtime gives {expected:?}: {err}")))?;
    let trace = traces_from_wasmtime_steps(&run.steps)
        .map_err(|err| TestCaseError::fail(format!("normalization failed; Wasmtime gives {expected:?}: {err}")))?;
    common::sanity_check_trace(&trace, run.artifacts());
    common::ccs_check_trace(&trace);
    let last = common::final_state(&trace);
    let output = last.output;
    proptest::prop_assert!(last.is_terminal(), "the trace ends in {last:?}");
    let values = match expected {
        Err(trap) => {
            proptest::prop_assert!(
                last.trapped && !output.enabled,
                "Wasmtime traps with {trap}, the trace ends with trapped={} and output {output:?}",
                last.trapped
            );
            return Ok(Outcome::Trapped(trap));
        }
        Ok(values) => values,
    };
    proptest::prop_assert!(!last.trapped, "Wasmtime returns {values:?}, the trace traps");
    let traced = (output.enabled, output.value_lo, output.value_hi);
    match values.as_slice() {
        [] => proptest::prop_assert!(
            !output.enabled,
            "Wasmtime returns nothing, the trace outputs {output:?}"
        ),
        [Value::I32(v)] => proptest::prop_assert_eq!(traced, (true, *v as u32, 0), "Wasmtime returns {}", v),
        [Value::I64(v)] => {
            proptest::prop_assert_eq!(traced, (true, *v as u32, (*v >> 32) as u32), "Wasmtime returns {}", v)
        }
        other => proptest::prop_assert!(false, "the parser accepts an export with results {other:?}"),
    }
    Ok(Outcome::Returned)
}

fn with_source<T>(source: &str, check: impl FnOnce() -> Result<T, TestCaseError>) -> Result<T, TestCaseError> {
    match catch_unwind(AssertUnwindSafe(check)) {
        Ok(Err(TestCaseError::Fail(reason))) => Err(TestCaseError::fail(format!("{reason}\n{source}"))),
        Ok(result) => result,
        Err(panic) => {
            let message = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            Err(TestCaseError::fail(format!("panic: {message}\n{source}")))
        }
    }
}

fn run_cases(cases: u32, test: impl Fn(&[u32]) -> Result<(), TestCaseError>) {
    let result = runner(cases).run(&vec(any::<u32>(), CHOICES), |choices| test(&choices));
    match result {
        Ok(()) => {}
        Err(TestError::Fail(reason, _)) => panic!("{reason}"),
        Err(TestError::Abort(reason)) => panic!("aborted: {reason}"),
    }
}

#[derive(Debug, Default)]
struct Counts(BTreeMap<String, u32>);

impl Counts {
    fn add(&mut self, key: impl Into<String>) {
        *self.0.entry(key.into()).or_default() += 1;
    }

    fn get(&self, key: &str) -> u32 {
        self.0.get(key).copied().unwrap_or(0)
    }
}

const GENERATED_CASES: u32 = 768;

#[test]
fn generated_modules_trace_as_wasmtime_runs_them() {
    let counts = RefCell::new(Counts::default());
    run_cases(GENERATED_CASES, |choices| {
        let generated = module::module(&mut Choices::new(choices));
        let wasm = wat::parse_str(&generated.wat).map_err(|err| TestCaseError::fail(format!("{err}")))?;
        let outcome = with_source(&generated.wat, || differential(&wasm))?;
        let mut counts = counts.borrow_mut();
        match outcome {
            Outcome::Refused(reason) => {
                counts.add(format!("refused: {reason}"));
                for feature in generated.features {
                    counts.add(format!("refused {feature}"));
                }
            }
            outcome => {
                if let Outcome::Trapped(trap) = &outcome {
                    counts.add("Trapped");
                    counts.add(format!("trap: {trap}"));
                } else {
                    counts.add("Returned");
                }
                for feature in generated.features {
                    counts.add(feature);
                }
            }
        }
        Ok(())
    });
    let counts = counts.into_inner();
    eprintln!("{counts:#?}");
    let checked = counts.get("Returned") + counts.get("Trapped");
    assert!(checked >= GENERATED_CASES / 2, "{counts:#?}");
    assert!(counts.get("Returned") >= GENERATED_CASES / 4, "{counts:#?}");
    for feature in module::FEATURES {
        assert!(counts.get(feature) >= 4, "{feature}: {counts:#?}");
    }
    assert!(counts.get("refused multi_result_main") >= 4, "{counts:#?}");
}

fn first_difference(a: &str, b: &str) -> String {
    a.lines()
        .zip(b.lines())
        .find(|(a, b)| a != b)
        .map(|(a, b)| format!("{a} / {b}"))
        .unwrap_or_default()
}

fn type_mutation(wats: &[String; 2], counts: &mut Counts) -> Result<(), TestCaseError> {
    let wasm = wats
        .each_ref()
        .map(|wat| wat::parse_str(wat).expect("generated WAT parses"));
    let engine = engine();
    let mut instances = [instantiate(&engine, &wasm[0])?, instantiate(&engine, &wasm[1])?];
    let types = instances.each_mut().map(Instance::slot_types);
    let equal_types = types[0]
        .iter()
        .zip(&types[1])
        .all(|(a, b)| wasmtime::FuncType::eq(a, b));
    let runs = [instances[0].run_main()?, instances[1].run_main()?];
    let artifacts = match wasm
        .each_ref()
        .map(|wasm| extract_wasm_program_artifacts(wasm))
    {
        [Ok(a), Ok(b)] => [a, b].map(|artifacts| format!("{:#?}", artifacts.tables)),
        other => {
            counts.add(format!("refused: {other:?}"));
            return Ok(());
        }
    };
    let same_artifacts = artifacts[0] == artifacts[1];
    if equal_types {
        counts.add("equal types");
        if wats[0] != wats[1] {
            counts.add("equal types, different declarations");
        }
        proptest::prop_assert_eq!(&runs[0], &runs[1], "Wasmtime gives equal types and different runs");
        proptest::prop_assert!(
            same_artifacts,
            "Wasmtime gives equal types, the artifacts differ: {}",
            first_difference(&artifacts[0], &artifacts[1])
        );
    }
    if !equal_types && runs[0] == runs[1] {
        counts.add("different types, equal runs");
    }
    if runs[0] != runs[1] {
        counts.add("different runs");
        proptest::prop_assert!(
            !same_artifacts,
            "Wasmtime runs {:?} and {:?}, the artifacts are equal",
            runs[0],
            runs[1]
        );
    }
    for wasm in &wasm {
        if let Outcome::Refused(reason) = differential(wasm)? {
            counts.add(format!("refused: {reason}"));
        }
    }
    Ok(())
}

const MUTATION_CASES: u32 = 1024;

#[test]
fn type_declaration_mutations_change_artifacts_as_wasmtime_changes_types() {
    let counts = RefCell::new(Counts::default());
    run_cases(MUTATION_CASES, |choices| {
        let wats = types::type_mutation_pair(&mut Choices::new(choices));
        let source = format!("first:\n{}\nsecond:\n{}", wats[0], wats[1]);
        with_source(&source, || type_mutation(&wats, &mut counts.borrow_mut()))
    });
    let counts = counts.into_inner();
    eprintln!("{counts:#?}");
    assert!(
        counts.get("equal types, different declarations") >= MUTATION_CASES / 16,
        "{counts:#?}"
    );
    assert!(counts.get("different runs") >= MUTATION_CASES / 8, "{counts:#?}");
}
