use neo_application::{plan_ordered_memory_slots, MemoryActivationFacts, MemoryCatalog, MemorySlotPlan};

use crate::{
    memory_activation::{activation_supports, gate_exclusions},
    WasmMemoryId,
};

/// Plan slots using the activation facts enforced by the WASM relation.
/// The consumer must enforce the slot bindings.
pub fn plan_ordered_wasm_memory_slots(catalog: &MemoryCatalog<WasmMemoryId>) -> MemorySlotPlan {
    plan_ordered_memory_slots(
        catalog,
        &MemoryActivationFacts {
            supports: activation_supports(),
            gate_exclusions: gate_exclusions(),
        },
    )
}
