//! Store-local module discovery and per-instance capture. Module selection is
//! embedder policy; discovered artifacts are witness data, not verifier authority.

use super::{
    capture_frame, entry_inputs, import_inputs::PendingImport, memory_inputs::capture_bytes, parse, runtime_read,
    LoweringTables, WasmProgramArtifacts, WasmtimeTraceStep,
};
use crate::host_event_bindings::HostEventBindings;
use crate::WasmBuildError;
use std::cmp::Ordering;
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
/// Core-module registration has no canonical ABI metadata: it cannot identify
/// post-return cleanup or bind its arguments to a preceding component result.
/// The component collectors reject post-return; this registry alone does not
/// validate those component-level relationships.
/// Embedders must not normalize an instance whose captured rows include a failed
/// invocation, including a guest callback whose failure the host caught before
/// returning successfully: those rows remain in the trace and are not flagged.
#[derive(Debug, Default)]
pub struct WasmtimeTraceRegistry {
    bindings: BTreeMap<Vec<u8>, HostEventBindings>,
    modules: BTreeMap<u64, Option<Arc<WasmProgramArtifacts>>>,
    ignored_modules: BTreeSet<Vec<u8>>,
    ignored_instances: BTreeSet<u32>,
    instances: BTreeMap<u32, WasmtimeTraceState>,
    // Activation depth of the latest host tail, until the next captured row.
    // An unwind below it means the tail's host call failed.
    last_host_tail_depth: Option<usize>,
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

    /// An error unwinds the current host-to-guest activation. Outer suspended
    /// calls may still return normally if their host catches this error.
    pub(super) fn discard_unwound_imports(&mut self, activation_depth: usize) {
        if self
            .last_host_tail_depth
            .is_some_and(|depth| activation_depth < depth)
        {
            self.error = Some(WasmBuildError::Trace(
                "terminal host tail did not complete successfully".into(),
            ));
        }
        for state in self.instances.values_mut() {
            state
                .pending_imports
                .retain(|pending| pending.activation_depth < activation_depth);
        }
        // Leave the failed calls' memory captures absent so replay rejects them.
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
}

/// A capture and the immutable module configuration actually used to produce it.
#[derive(Debug)]
pub struct WasmtimeTraceState {
    next_step: u64,
    pending_imports: Vec<PendingImport>,
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
}

pub(super) fn capture_step<T: WasmTraceSink + 'static>(
    frame: &FrameHandle,
    activation_depth: usize,
    store: &mut StoreContextMut<'_, T>,
) -> Result<(), WasmBuildError> {
    // Any later guest row ends failure tracking. A later unwind usually belongs to
    // that execution; during reentry it may still be the tail's host call, which
    // is deliberately no longer distinguished.
    store
        .data_mut()
        .wasm_trace_registry_mut()
        .last_host_tail_depth = None;
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
                pending_imports: Vec::new(),
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
    if super::host_tail::is_direct_host_tail(&row) {
        if frame
            .parent(&mut *store)
            .map_err(|err| WasmBuildError::Trace(err.to_string()))?
            .is_some()
        {
            return Err(WasmBuildError::Unsupported(
                "host tail calls currently must return directly to the host".into(),
            ));
        }
        if row
            .function_ref
            .and_then(|fref| tables.artifacts.host_event_bindings.imports.get(&fref))
            .is_some_and(|template| template.input_count != 0)
        {
            return Err(WasmBuildError::Unsupported(
                "host tail calls with declared memory writes are not supported".into(),
            ));
        }
    }
    // Nested activations may execute arbitrary guest code before this host call
    // returns. Only its original activation can supply the output buffer.
    let pending = {
        let state = store
            .data_mut()
            .wasm_trace_registry_mut()
            .instances
            .get_mut(&index)
            .unwrap();
        if let Some(pending) = state.pending_imports.last() {
            match pending.activation_depth.cmp(&activation_depth) {
                // Nested reentry: the original host call is still suspended.
                Ordering::Less => None,
                // Possible return: validate the caller and continuation below.
                Ordering::Equal => state.pending_imports.pop(),
                // A deeper pending call survived an unwind without cleanup.
                Ordering::Greater => {
                    return Err(WasmBuildError::Trace(format!(
                        "host call at step {} did not resume at its expected continuation",
                        pending.row
                    )));
                }
            }
        } else {
            None
        }
    };
    if let Some(pending) = pending {
        if Some(pending.function_index) != row.function_index || Some(pending.return_pc) != row.pc.map(u64::from) {
            return Err(WasmBuildError::Trace(format!(
                "host call at step {} did not resume at its expected continuation",
                pending.row
            )));
        }
        let captured = pending
            .writes
            .and_then(|writes| {
                if pending.memory_pages != row.memory_pages_before {
                    return Err(WasmBuildError::Trace(
                        "host memory growth across an import is unsupported".into(),
                    ));
                }
                capture_bytes(&writes, frame, store)
            })
            .map_err(|err| format!("host call at step {}: {err}", pending.row));
        store
            .data_mut()
            .wasm_trace_registry_mut()
            .instances
            .get_mut(&index)
            .unwrap()
            .steps[pending.row]
            .host_call_memory = Some(captured);
    }
    let pending = PendingImport::from_call(&row, &tables, activation_depth)?;
    entry_inputs::capture_entry_memory(&mut row, frame, store, &tables);
    let registry = store.data_mut().wasm_trace_registry_mut();
    let state = registry.instances.get_mut(&index).unwrap();
    if let Some(pending) = pending {
        state.pending_imports.push(pending);
    }
    state.next_step += 1;
    if super::host_tail::is_direct_host_tail(&row) {
        registry.last_host_tail_depth = Some(activation_depth);
    }
    state.steps.push(row);
    Ok(())
}
