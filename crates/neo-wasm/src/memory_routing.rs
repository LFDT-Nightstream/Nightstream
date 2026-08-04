//! Compiles logical WASM memory declarations into verifier-owned Nebula slots.

use std::collections::{BTreeMap, BTreeSet};

use neo_fold_clean::frontends::nebula::application::{MemoryOpSlot, MemoryPort, MemoryPortActivation, MemoryPortKind};

use crate::isa::{opcode_code, opcode_info_from_code, WasmMemoryAccessKind, WasmOpcode};
use crate::layout::{
    selector_col, COL_CALL_INDIRECT_IS_NOT_TRAP, COL_CALL_STACK_POP_PRESENT, COL_CALL_STACK_PUSH_PRESENT,
    COL_CI_HOST_CALL, COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, COL_GATHER_ACTIVE, COL_GATHER_LOCAL_WRITE,
    COL_GATHER_LOCAL_WRITE_LO, COL_GRAMMAR_EXIT_LATCH, COL_GRAMMAR_HOST_CALL, COL_GUEST_ENTRY_ACTIVE,
    COL_HOST_ARGS_ACTIVE_BEFORE, COL_HOST_RESULT_ACTIVE, COL_IS_PROGRAM_ROW, COL_LINEAR_MEM_LANE0_LOAD_ACTIVE,
    COL_LINEAR_MEM_LANE0_STORE_ACTIVE, COL_LINEAR_MEM_LANE1_LOAD_ACTIVE, COL_LINEAR_MEM_LANE1_STORE_ACTIVE,
    COL_LINEAR_MEM_LANE2_LOAD_ACTIVE, COL_LINEAR_MEM_LANE2_STORE_ACTIVE, COL_LOCAL_WRITE_ENABLED, COL_OUTPUT_CAPTURED,
    COL_PADDING_ACTIVE, COL_PARAM_INIT_ACTIVE_BEFORE, COL_PC_FREF_ACTIVE, COL_PC_ROM_ACTIVE, COL_STACK_READ0_ACTIVE,
    COL_STACK_READ1_ACTIVE, COL_STACK_READ2_ACTIVE, COL_STACK_WRITE0_ACTIVE, COL_STACK_WRITE0_HI_ACTIVE,
    COL_TABLE_READ_ENABLED, COL_TABLE_SIZE_READ_ENABLED, COL_TAIL_ENTER_ACTIVE, COL_TURN_BOUNDARY, SELECTOR_COLS,
};
use crate::relation_layout::{WasmMemoryActivation, WasmMemoryColumnKind, WasmRelationLayout};

/// Auxiliary row-kind columns that are pairwise disjoint with each other and
/// with every opcode selector under the CCS row-kind and opcode one-hots.
const AUXILIARY_ROW_SUPPORT_ATOMS: [usize; 6] = [
    COL_PARAM_INIT_ACTIVE_BEFORE,
    COL_TAIL_ENTER_ACTIVE,
    COL_HOST_ARGS_ACTIVE_BEFORE,
    COL_HOST_RESULT_ACTIVE,
    COL_GATHER_ACTIVE,
    COL_TURN_BOUNDARY,
];

fn activation_supports() -> BTreeMap<usize, BTreeSet<usize>> {
    let mut supports = SELECTOR_COLS
        .into_iter()
        .chain(AUXILIARY_ROW_SUPPORT_ATOMS)
        .map(|atom| (atom, BTreeSet::from([atom])))
        .collect();
    insert_derived_activation_support(
        &mut supports,
        "local writes",
        COL_LOCAL_WRITE_ENABLED,
        [
            selector_col(WasmOpcode::LocalSet).unwrap(),
            selector_col(WasmOpcode::LocalTee).unwrap(),
        ],
    );
    insert_derived_activation_support(
        &mut supports,
        "table reads",
        COL_TABLE_READ_ENABLED,
        [
            selector_col(WasmOpcode::TableGet).unwrap(),
            selector_col(WasmOpcode::CallIndirect).unwrap(),
            selector_col(WasmOpcode::ReturnCallIndirect).unwrap(),
        ],
    );
    insert_derived_activation_support(
        &mut supports,
        "table-size reads",
        COL_TABLE_SIZE_READ_ENABLED,
        [
            selector_col(WasmOpcode::TableSize).unwrap(),
            selector_col(WasmOpcode::CallIndirect).unwrap(),
            selector_col(WasmOpcode::ReturnCallIndirect).unwrap(),
        ],
    );

    let load_selectors = memory_access_selectors(WasmMemoryAccessKind::Load);
    let store_selectors = memory_access_selectors(WasmMemoryAccessKind::Store);
    for (name, gate, selectors) in [
        (
            "linear-memory lane 0 loads",
            COL_LINEAR_MEM_LANE0_LOAD_ACTIVE,
            load_selectors.as_slice(),
        ),
        (
            "linear-memory lane 1 loads",
            COL_LINEAR_MEM_LANE1_LOAD_ACTIVE,
            load_selectors.as_slice(),
        ),
        (
            "linear-memory lane 2 loads",
            COL_LINEAR_MEM_LANE2_LOAD_ACTIVE,
            load_selectors.as_slice(),
        ),
        (
            "linear-memory lane 0 stores",
            COL_LINEAR_MEM_LANE0_STORE_ACTIVE,
            store_selectors.as_slice(),
        ),
        (
            "linear-memory lane 1 stores",
            COL_LINEAR_MEM_LANE1_STORE_ACTIVE,
            store_selectors.as_slice(),
        ),
        (
            "linear-memory lane 2 stores",
            COL_LINEAR_MEM_LANE2_STORE_ACTIVE,
            store_selectors.as_slice(),
        ),
    ] {
        insert_derived_activation_support(&mut supports, name, gate, selectors.iter().copied());
    }

    let indirect_call_selectors = opcode_selectors([WasmOpcode::CallIndirect, WasmOpcode::ReturnCallIndirect]);
    for (name, gate) in [
        ("indirect-call type lookups", COL_FUNCTION_CALL_TYPE_LOOKUP_GATE),
        ("successful indirect calls", COL_CALL_INDIRECT_IS_NOT_TRAP),
        ("indirect host calls", COL_CI_HOST_CALL),
    ] {
        insert_derived_activation_support(&mut supports, name, gate, indirect_call_selectors.iter().copied());
    }

    insert_derived_activation_support(
        &mut supports,
        "call-stack pushes",
        COL_CALL_STACK_PUSH_PRESENT,
        opcode_selectors([WasmOpcode::Call, WasmOpcode::CallIndirect]),
    );
    insert_derived_activation_support(
        &mut supports,
        "call-stack pops",
        COL_CALL_STACK_POP_PRESENT,
        opcode_selectors([WasmOpcode::Return, WasmOpcode::End]),
    );
    insert_derived_activation_support(
        &mut supports,
        "captured outputs",
        COL_OUTPUT_CAPTURED,
        opcode_selectors([WasmOpcode::Return, WasmOpcode::End]),
    );

    let call_selectors = opcode_selectors([
        WasmOpcode::Call,
        WasmOpcode::CallIndirect,
        WasmOpcode::ReturnCall,
        WasmOpcode::ReturnCallIndirect,
    ]);
    for (name, gate) in [
        ("guest entries", COL_GUEST_ENTRY_ACTIVE),
        ("grammar host calls", COL_GRAMMAR_HOST_CALL),
    ] {
        insert_derived_activation_support(&mut supports, name, gate, call_selectors.iter().copied());
    }

    insert_derived_activation_support(
        &mut supports,
        "static pc-ROM reads",
        COL_PC_ROM_ACTIVE,
        opcode_selectors(WasmOpcode::supported().into_iter().filter(|opcode| {
            !matches!(
                opcode,
                WasmOpcode::CallIndirect
                    | WasmOpcode::ReturnCallIndirect
                    | WasmOpcode::Return
                    | WasmOpcode::Unreachable
            )
        })),
    );

    let program_selectors = SELECTOR_COLS.to_vec();
    insert_derived_activation_support(
        &mut supports,
        "program rows",
        COL_IS_PROGRAM_ROW,
        program_selectors.iter().copied(),
    );
    insert_derived_activation_support(
        &mut supports,
        "grammar exit latches",
        COL_GRAMMAR_EXIT_LATCH,
        opcode_selectors([WasmOpcode::Return, WasmOpcode::End]),
    );

    let read0_selectors = opcode_selectors(WasmOpcode::supported().into_iter().filter(|opcode| {
        opcode_info_from_code(opcode_code(*opcode)).stack_reads >= 1
            || matches!(opcode, WasmOpcode::CallIndirect | WasmOpcode::ReturnCallIndirect)
    }));
    insert_derived_activation_support(
        &mut supports,
        "stack read lane 0",
        COL_STACK_READ0_ACTIVE,
        read0_selectors.iter().copied().chain([
            COL_PARAM_INIT_ACTIVE_BEFORE,
            COL_HOST_ARGS_ACTIVE_BEFORE,
            COL_GATHER_ACTIVE,
        ]),
    );
    for (name, gate, minimum_reads) in [
        ("stack read lane 1", COL_STACK_READ1_ACTIVE, 2),
        ("stack read lane 2", COL_STACK_READ2_ACTIVE, 3),
    ] {
        insert_derived_activation_support(
            &mut supports,
            name,
            gate,
            opcode_selectors(
                WasmOpcode::supported()
                    .into_iter()
                    .filter(|opcode| opcode_info_from_code(opcode_code(*opcode)).stack_reads >= minimum_reads),
            ),
        );
    }

    let write0_support = opcode_selectors(
        WasmOpcode::supported()
            .into_iter()
            .filter(|opcode| opcode_info_from_code(opcode_code(*opcode)).stack_writes >= 1),
    )
    .into_iter()
    .chain([COL_HOST_RESULT_ACTIVE, COL_GATHER_ACTIVE])
    .collect::<Vec<_>>();
    for (name, gate) in [
        ("stack write lane 0", COL_STACK_WRITE0_ACTIVE),
        ("stack write high limb", COL_STACK_WRITE0_HI_ACTIVE),
    ] {
        insert_derived_activation_support(&mut supports, name, gate, write0_support.iter().copied());
    }

    for (name, gate) in [
        ("gather local writes", COL_GATHER_LOCAL_WRITE),
        ("gather local low-limb writes", COL_GATHER_LOCAL_WRITE_LO),
    ] {
        insert_derived_activation_support(&mut supports, name, gate, [COL_GATHER_ACTIVE]);
    }

    insert_derived_activation_support(
        &mut supports,
        "pc-to-function reads",
        COL_PC_FREF_ACTIVE,
        program_selectors.iter().copied().chain([
            COL_PARAM_INIT_ACTIVE_BEFORE,
            COL_TAIL_ENTER_ACTIVE,
            COL_HOST_ARGS_ACTIVE_BEFORE,
            COL_HOST_RESULT_ACTIVE,
        ]),
    );
    supports
}

pub(crate) struct MemoryActivationSupport {
    pub(crate) gate: usize,
    pub(crate) atoms: BTreeSet<usize>,
}

/// Derived support claims only; opcode selectors and auxiliary row-kind
/// atoms are already their own singleton supports.
pub(crate) fn derived_activation_supports() -> Vec<MemoryActivationSupport> {
    activation_supports()
        .into_iter()
        .filter(|(gate, _)| !SELECTOR_COLS.contains(gate) && !AUXILIARY_ROW_SUPPORT_ATOMS.contains(gate))
        .map(|(gate, atoms)| MemoryActivationSupport { gate, atoms })
        .collect()
}

fn opcode_selectors(opcodes: impl IntoIterator<Item = WasmOpcode>) -> Vec<usize> {
    opcodes
        .into_iter()
        .map(|opcode| selector_col(opcode).expect("supported opcode selector"))
        .collect()
}

fn memory_access_selectors(kind: WasmMemoryAccessKind) -> Vec<usize> {
    opcode_selectors(WasmOpcode::supported().into_iter().filter(|opcode| {
        opcode
            .memory_access_info()
            .is_some_and(|access| access.kind == kind)
    }))
}

/// Record the verifier-owned implication `gate = 1 => one of support atoms = 1`.
/// These support bounds come from the WASM CCS, not sampled traces.
fn insert_derived_activation_support(
    supports: &mut BTreeMap<usize, BTreeSet<usize>>,
    name: &'static str,
    gate: usize,
    possible_atoms: impl IntoIterator<Item = usize>,
) {
    assert!(
        !SELECTOR_COLS.contains(&gate) && !AUXILIARY_ROW_SUPPORT_ATOMS.contains(&gate),
        "activation support {name:?} uses support atom {gate} as a derived gate",
    );
    let mut unique = BTreeSet::new();
    for atom in possible_atoms {
        assert!(
            SELECTOR_COLS.contains(&atom) || AUXILIARY_ROW_SUPPORT_ATOMS.contains(&atom),
            "activation support {name:?} contains non-atomic column {atom}",
        );
        assert!(
            unique.insert(atom),
            "activation support {name:?} repeats support atom {atom}",
        );
    }
    assert!(
        !unique.is_empty(),
        "activation support {name:?} needs at least one possible atom",
    );
    assert!(
        supports.insert(gate, unique).is_none(),
        "activation support {name:?} repeats gate column {gate}",
    )
}

pub(crate) fn build_batched_memory_slots(
    relation: &WasmRelationLayout,
    batch_size: usize,
    single_step_columns: usize,
) -> Vec<MemoryOpSlot> {
    let single_step = build_single_step_memory_slots(relation);
    let mut batched = Vec::with_capacity(single_step.len() * batch_size);
    for block in 0..batch_size {
        let offset = block * single_step_columns;
        batched.extend(single_step.iter().map(|slot| offset_slot(slot, offset)));
    }
    batched
}

pub(crate) fn build_single_step_memory_slots(relation: &WasmRelationLayout) -> Vec<MemoryOpSlot> {
    let activation_supports = activation_supports();
    let mut singleton_slots = Vec::new();
    let mut shared_slots: Vec<Vec<MemoryPort>> = Vec::new();

    for (region, memory) in relation.auxiliary.memories.iter().enumerate() {
        for column in &memory.columns {
            let port = MemoryPort::new(
                region,
                column
                    .address_columns
                    .iter()
                    .map(|column| column.0)
                    .collect(),
                column.value_column.0,
                match column.kind {
                    WasmMemoryColumnKind::Read => MemoryPortKind::Read,
                    WasmMemoryColumnKind::Write { value_before_column } => MemoryPortKind::Write {
                        value_before_column: value_before_column.map(|column| column.0),
                    },
                },
                match column.activation {
                    WasmMemoryActivation::Always => MemoryPortActivation::UnlessColumn(COL_PADDING_ACTIVE),
                    WasmMemoryActivation::BooleanGate(column) => MemoryPortActivation::Column(column.0),
                },
            );

            let Some(activation) = activation_column(port.activation()) else {
                singleton_slots.push(MemoryOpSlot::new(vec![port]));
                continue;
            };
            if !activation_supports.contains_key(&activation) {
                singleton_slots.push(MemoryOpSlot::new(vec![port]));
                continue;
            }

            let target = shared_slots.iter().position(|slot| {
                slot.iter().all(|candidate| {
                    let candidate_activation = activation_column(candidate.activation())
                        .expect("shared slots contain only column-activated ports");
                    activations_are_disjoint(activation, candidate_activation, &activation_supports)
                })
            });
            match target {
                Some(index) => shared_slots[index].push(port),
                None => shared_slots.push(vec![port]),
            }
        }
    }

    singleton_slots.extend(shared_slots.into_iter().map(MemoryOpSlot::new));
    singleton_slots
}

fn activation_column(activation: MemoryPortActivation) -> Option<usize> {
    match activation {
        MemoryPortActivation::Column(column) => Some(column),
        _ => None,
    }
}

fn activations_are_disjoint(left: usize, right: usize, activation_supports: &BTreeMap<usize, BTreeSet<usize>>) -> bool {
    let (Some(left), Some(right)) = (activation_supports.get(&left), activation_supports.get(&right)) else {
        return false;
    };
    left.is_disjoint(right)
}

fn offset_slot(slot: &MemoryOpSlot, offset: usize) -> MemoryOpSlot {
    MemoryOpSlot::new(
        slot.candidates()
            .iter()
            .map(|port| {
                MemoryPort::new(
                    port.region(),
                    port.address_columns()
                        .iter()
                        .map(|column| column + offset)
                        .collect(),
                    port.value_column() + offset,
                    match port.kind() {
                        MemoryPortKind::Read => MemoryPortKind::Read,
                        MemoryPortKind::Write { value_before_column } => MemoryPortKind::Write {
                            value_before_column: value_before_column.map(|column| column + offset),
                        },
                    },
                    match port.activation() {
                        MemoryPortActivation::Always => MemoryPortActivation::Always,
                        MemoryPortActivation::Column(column) => MemoryPortActivation::Column(column + offset),
                        MemoryPortActivation::UnlessColumn(column) => {
                            MemoryPortActivation::UnlessColumn(column + offset)
                        }
                    },
                )
            })
            .collect(),
    )
}

#[cfg(test)]
#[path = "../tests/nebula/memory_routing.rs"]
mod tests;
