//! Store-local module discovery and per-instance capture. Module selection is
//! embedder policy; discovered artifacts are witness data, not verifier authority.

use super::{
    capture_frame, entry_inputs, parse, runtime_read, LoweringTables, WasmProgramArtifacts, WasmtimeTraceStep,
};
use crate::host_event_bindings::HostEventBindings;
use crate::{WasmBuildError, WasmOpcode};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use wasmtime::{FrameHandle, StoreContextMut};

/// Exposes the store-local registry to the debug handler.
pub trait WasmTraceSink {
    fn wasm_trace_registry(&self) -> &WasmtimeTraceRegistry;
    fn wasm_trace_registry_mut(&mut self) -> &mut WasmtimeTraceRegistry;
}

/// Configure selected core modules before instantiation, then install the debug
/// handler. Artifacts and instance function identities are discovered on the
/// first instruction, including instructions executed by a Wasm start section.
///
/// Unconfigured modules are ignored. Registering one after it has executed is
/// rejected: its missing prefix cannot be recovered. An explicitly registered
/// module with empty bindings is still traced. Use one registry per store.
/// Capturing a start function does not establish proof semantics for instantiation.
#[derive(Debug, Default)]
pub struct WasmtimeTraceRegistry {
    bindings: BTreeMap<Vec<u8>, HostEventBindings>,
    modules: BTreeMap<u64, Option<Arc<WasmProgramArtifacts>>>,
    ignored_modules: BTreeSet<Vec<u8>>,
    ignored_instances: BTreeSet<u32>,
    instances: BTreeMap<u32, WasmtimeTraceState>,
    pub(super) error: Option<WasmBuildError>,
}

impl WasmTraceSink for WasmtimeTraceRegistry {
    fn wasm_trace_registry(&self) -> &Self {
        self
    }
    fn wasm_trace_registry_mut(&mut self) -> &mut Self {
        self
    }
}

impl WasmtimeTraceRegistry {
    /// Select exact core-module bytes and their bindings. No program tables are
    /// accepted or parsed here. Bindings are validated against the live module
    /// when it first executes, before its first row is captured.
    /// Repeated registration is allowed only with identical bindings.
    pub fn register_module(&mut self, wasm: &[u8], bindings: HostEventBindings) -> Result<(), WasmBuildError> {
        self.check_errors()?;
        if !wasm.starts_with(b"\0asm\x01\0\0\0") {
            return Err(WasmBuildError::Trace(
                "trace registration requires core Wasm bytes".into(),
            ));
        }
        if self.ignored_modules.contains(wasm) {
            return Err(WasmBuildError::Trace(
                "cannot register a module after untraced execution".into(),
            ));
        }
        if let Some(existing) = self.bindings.get(wasm) {
            if existing != &bindings {
                return Err(WasmBuildError::Trace(
                    "module already registered with different bindings".into(),
                ));
            }
        } else {
            self.bindings.insert(wasm.to_vec(), bindings);
        }
        Ok(())
    }

    /// Surface a retained discovery or frame-capture failure. Such failures
    /// poison this registry; partial traces must not be used for normalization.
    pub fn check_errors(&self) -> Result<(), WasmBuildError> {
        match &self.error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    pub fn instance(&self, index: u32) -> Result<&WasmtimeTraceState, WasmBuildError> {
        self.check_errors()?;
        self.instances
            .get(&index)
            .ok_or_else(|| WasmBuildError::Trace(format!("no captured trace for instance {index}")))
    }

    pub fn instances(&self) -> Result<impl Iterator<Item = (u32, &WasmtimeTraceState)>, WasmBuildError> {
        self.check_errors()?;
        Ok(self.instances.iter().map(|(&index, state)| (index, state)))
    }

    /// Convenience for executions expected to contain one traced instance.
    /// Never concatenates traces from different instances or silently accepts
    /// an execution in which none of the configured modules ran.
    pub fn single_instance(&self) -> Result<&WasmtimeTraceState, WasmBuildError> {
        self.check_errors()?;
        if self.instances.len() != 1 {
            return Err(WasmBuildError::Trace(format!(
                "expected one captured instance, found {}",
                self.instances.len()
            )));
        }
        Ok(self.instances.values().next().unwrap())
    }

    pub(super) fn into_run(self, results: Vec<String>) -> Result<super::WasmtimeTraceRun, WasmBuildError> {
        self.single_instance()?;
        let state = self.instances.into_values().next().unwrap();
        Ok(super::WasmtimeTraceRun {
            artifacts: state.tables.artifacts.clone(),
            results,
            steps: state.steps,
        })
    }

    /// Attach advice to the latest host-call row of the specified instance.
    /// Keep the caller's instance index across calls into other instances so
    /// reentry cannot redirect advice to the callee's trace.
    pub fn record_call_inputs(&mut self, instance_index: u32, words: &[u64]) -> Result<(), WasmBuildError> {
        self.check_errors()?;
        let state = self
            .instances
            .get_mut(&instance_index)
            .ok_or_else(|| WasmBuildError::Trace(format!("no captured trace for instance {instance_index}")))?;
        state.record_call_inputs(words)
    }
}

/// A capture and the immutable module configuration actually used to produce it.
#[derive(Debug)]
pub struct WasmtimeTraceState {
    next_step: u64,
    steps: Vec<WasmtimeTraceStep>,
    tables: Arc<LoweringTables>,
}

impl WasmtimeTraceState {
    pub fn artifacts(&self) -> &WasmProgramArtifacts {
        &self.tables.artifacts
    }
    pub fn steps(&self) -> &[WasmtimeTraceStep] {
        &self.steps
    }
    /// Record per-call host-event input words for the in-flight host call
    /// (for example, ref ids or caller identities). Call from
    /// inside a host-function implementation (`store.data_mut()`): the debug
    /// hook captures each instruction before it executes, so the latest
    /// captured step is the host-call row being serviced and the batch
    /// attaches to it — no call-order bookkeeping. Repeated calls append.
    fn record_call_inputs(&mut self, words: &[u64]) -> Result<(), WasmBuildError> {
        let row = self.steps.last_mut().ok_or_else(|| {
            WasmBuildError::Trace("record_call_inputs: no captured step; not inside a traced host call".to_string())
        })?;
        let is_host_call = matches!(row.opcode_decoded, Some(WasmOpcode::Call | WasmOpcode::CallIndirect))
            && !row.target_function_is_guest;
        if !is_host_call {
            return Err(WasmBuildError::Trace(format!(
                "record_call_inputs: latest captured step (cycle {}, opcode {:?}) is not a host-call row",
                row.step, row.opcode
            )));
        }
        row.host_call_inputs.extend_from_slice(words);
        Ok(())
    }
}

pub(super) fn capture_step<T: WasmTraceSink + 'static>(
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, T>,
) -> Result<(), WasmBuildError> {
    let instance = frame
        .instance(&mut *store)
        .map_err(|err| WasmBuildError::Trace(format!("failed to inspect frame instance: {err}")))?;
    let index = instance.debug_index_in_store();
    if store
        .data()
        .wasm_trace_registry()
        .ignored_instances
        .contains(&index)
    {
        return Ok(());
    }
    if !store
        .data()
        .wasm_trace_registry()
        .instances
        .contains_key(&index)
    {
        let module = frame
            .module(&mut *store)
            .map_err(|err| WasmBuildError::Trace(format!("failed to inspect frame module: {err}")))?
            .ok_or_else(|| WasmBuildError::Trace("executing frame has no core module".into()))?
            .clone();
        let engine_index = module.debug_index_in_engine();
        if !store
            .data()
            .wasm_trace_registry()
            .modules
            .contains_key(&engine_index)
        {
            let bytes = module
                .debug_bytecode()
                .ok_or_else(|| WasmBuildError::Trace("debug bytecode unavailable for executing module".into()))?;
            let registry = store.data_mut().wasm_trace_registry_mut();
            let artifacts = if let Some(bindings) = registry.bindings.get(bytes) {
                let mut artifacts = parse::parse_wasm_artifacts(bytes)?;
                bindings.validate_against_program(&artifacts.tables)?;
                artifacts.host_event_bindings = bindings.clone();
                Some(Arc::new(artifacts))
            } else {
                registry.ignored_modules.insert(bytes.to_vec());
                None
            };
            registry.modules.insert(engine_index, artifacts);
        }
        let registry = store.data_mut().wasm_trace_registry_mut();
        let Some(artifacts) = registry.modules[&engine_index].clone() else {
            registry.ignored_instances.insert(index);
            return Ok(());
        };
        let func_ref_ids = runtime_read::build_debug_function_id_map(&instance, &mut *store)?;
        store.data_mut().wasm_trace_registry_mut().instances.insert(
            index,
            WasmtimeTraceState {
                next_step: 0,
                steps: Vec::new(),
                tables: Arc::new(LoweringTables {
                    artifacts,
                    func_ref_ids,
                }),
            },
        );
    }
    let registry = store.data().wasm_trace_registry();
    let state = &registry.instances[&index];
    let tables = state.tables.clone();
    let step = state.next_step;
    let mut row = capture_frame(step, frame, store, &tables)?;
    entry_inputs::capture_entry_memory(&mut row, frame, store, &tables);
    let registry = store.data_mut().wasm_trace_registry_mut();
    let state = registry.instances.get_mut(&index).unwrap();
    state.next_step += 1;
    state.steps.push(row);
    Ok(())
}
