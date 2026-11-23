#[path = "callbacks.rs"]
mod callbacks;
#[path = "nested.rs"]
mod nested;

use neo_wasm::host_event_bindings::{EventBlock, ExportTemplate, HostEventBindings, Limb, SlotBinding};
use neo_wasm::{WasmtimeTraceHandler, WasmtimeTraceRegistry};
use wasmtime::{Config, Engine, Module, Store};

use super::{add_template, common, counter_component_wat, run_counter_turns};

fn event(tag: u32, values: impl IntoIterator<Item = SlotBinding>) -> EventBlock {
    let mut slots = [SlotBinding::Const(0); neo_wasm::comm_chain::COMM_CHAIN_EVENT_ARGS];
    for (index, value) in values.into_iter().enumerate() {
        slots[index] = value;
    }
    EventBlock::op(tag.into(), slots)
}

fn export_bindings(exports: &[(u32, u8)]) -> HostEventBindings {
    let mut bindings = HostEventBindings::default();
    for &(fref, params) in exports {
        let inputs = (0..params).map(|local| SlotBinding::InputLocal {
            input: local,
            local,
            limb: Limb::Lo,
        });
        bindings.exports.insert(
            fref,
            ExportTemplate {
                entry: vec![event(fref * 10, inputs)],
                exit: vec![event(fref * 10 + 1, [SlotBinding::OutputElem { limb: Limb::Lo }])],
                entry_input_count: params,
            },
        );
    }
    bindings
}

fn traced_module(wat: &str, bindings: HostEventBindings) -> (Store<WasmtimeTraceRegistry>, Module) {
    let bytes = wat::parse_str(wat).unwrap();
    let mut config = Config::new();
    config.guest_debug(true);
    let engine = Engine::new(&config).unwrap();
    let mut registry = WasmtimeTraceRegistry::default();
    registry.register_module(&bytes, bindings).unwrap();
    let module = Module::new(&engine, &bytes).unwrap();
    let mut store = Store::new(&engine, registry);
    store.set_debug_handler(WasmtimeTraceHandler::new());
    store.edit_breakpoints().unwrap().single_step(true).unwrap();
    (store, module)
}

#[test]
fn swapped_additions_bind_order() {
    let component = wat::parse_str(counter_component_wat()).unwrap();
    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(1, add_template());
    let states = [[7, 35], [35, 7]].map(|inputs| {
        let run = run_counter_turns(&component, &bindings, inputs);
        let trace =
            neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, &run.artifacts, Default::default())
                .expect("normalize the two additions");
        common::final_state(&trace)
    });
    let [first, swapped] = states;
    assert_eq!(first.output.value_lo, 42);
    assert!(first.output.enabled);
    assert_eq!(first.output, swapped.output);
    assert_ne!(first.comm_chain, swapped.comm_chain);
}
