//! Range checks that the memory slots make (SPEC.md section 7.6, rules 1 to
//! 4). When port `p` is active, the binding row `a_p · (x − field) = 0` makes
//! its column `x` equal a slot field of at most 32 bits, which row O1 keeps in
//! bits. The row `(1 − g) · x = 0` of `PORT_COVERS`, where `g` is a sum of
//! activation bits of ports that bind `x`, makes `x` zero when `g = 0`. When
//! `g ≠ 0`, `x ≠ 0` gives `g = 1`, so exactly one of those ports is active.
//! Thus `x < 2^32` on every row, as its 33 range-check rows gave, with one
//! row and no bit column. An honest row has `x = 0` when `g = 0`, and at most
//! one gate of `g` on when `x ≠ 0`.
//!
//! `PLUS_ONE` columns are `lo + 1` by an ungated row of `ccs.rs`, where `lo`
//! is in `PORT_COVERS` with the same gates (so `x = 1` when no port is active,
//! and a slot field otherwise), or `lo` is twice a sum of two values below
//! `2^31` and so is even and below `2^32`. Not in Lean or Rocq.

use neo_application::MemoryPortActivation;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::layout::*;
use crate::relation_layout::WasmRelationLayout;
use crate::tagged_r1cs_builder::{WasmConstraintScope, WasmTaggedR1csBuilder};

/// Each column whose range check its ports make, and the activation columns
/// of its gate `g`.
pub const PORT_COVERS: &[(usize, &[usize])] = &[
    (COL_OPCODE_CODE, &[COL_IS_PROGRAM_ROW]),
    (COL_PC_EDGE_KIND, &[COL_IS_PROGRAM_ROW]),
    (COL_CONTROL_CHOICE, &[COL_PC_ROM_ACTIVE, COL_HOST_EVENT_EXIT_LATCH]),
    (COL_EXIT_SLIDE_DISTANCE, &[COL_HOST_EVENT_EXIT_LATCH]),
    (
        COL_CALL_STACK_RETURN_PC_VALUE,
        &[COL_CALL_STACK_PUSH_PRESENT, COL_CALL_STACK_POP_PRESENT],
    ),
    (
        COL_CALL_STACK_CALLER_FBP_VALUE,
        &[COL_CALL_STACK_PUSH_PRESENT, COL_CALL_STACK_POP_PRESENT],
    ),
    (
        COL_CALL_STACK_CALLER_SP_BASE_VALUE,
        &[COL_CALL_STACK_PUSH_PRESENT, COL_CALL_STACK_POP_PRESENT],
    ),
    (
        COL_CALL_STACK_ADDR,
        &[COL_CALL_STACK_PUSH_PRESENT, COL_CALL_STACK_POP_PRESENT],
    ),
    (
        COL_PC_ROM_CALL_RETURN_CHOICE,
        &[COL_CALL_STACK_PUSH_PRESENT, COL_CI_HOST_CALL],
    ),
    (
        COL_ENTERED_FUNCTION_NUM_LOCALS,
        &[COL_GUEST_ENTRY_ACTIVE, COL_TURN_BOUNDARY],
    ),
    (COL_GLOBAL_INDEX, &[COL_PROGRAM_GLOBAL_INDEX_ACTIVE]),
    (COL_GLOBAL_VALUE, &[COL_SEL_GLOBAL_GET, COL_SEL_GLOBAL_SET]),
    (COL_GLOBAL_VALUE_HI, &[COL_SEL_GLOBAL_GET, COL_SEL_GLOBAL_SET]),
    (COL_TABLE_ID, &[COL_PROGRAM_TABLE_ID_ACTIVE]),
    (COL_TABLE_SIZE, &[COL_TABLE_SIZE_READ_ENABLED]),
    (COL_EXPECTED_TYPE_ID, &[COL_PROGRAM_CALL_INDIRECT_IMMEDIATES_ACTIVE]),
    (
        COL_STACK_READ_ADDR_LO[0],
        &[COL_STACK_READ_ACTIVE[0], COL_OUTPUT_CAPTURED],
    ),
    (COL_STACK_READ_ADDR_LO[1], &[COL_STACK_READ_ACTIVE[1]]),
    (COL_STACK_READ_ADDR_LO[2], &[COL_STACK_READ_ACTIVE[2]]),
    (COL_LINEAR_MEM_IMM_OFFSET, &[COL_LINEAR_MEM_USE_LANE0]),
    (COL_BR_TABLE_LEN, &[COL_SEL_BR_TABLE]),
    (COL_HOST_CONTEXT_ADDR, &[COL_HOST_SUSPEND, COL_HOST_CONTEXT_READ]),
    (COL_HOST_CONTEXT[0], &[COL_HOST_SUSPEND, COL_HOST_CONTEXT_READ]),
    (COL_HOST_CONTEXT[1], &[COL_HOST_SUSPEND, COL_HOST_CONTEXT_READ]),
    (COL_HOST_CONTEXT[2], &[COL_HOST_SUSPEND, COL_HOST_CONTEXT_READ]),
    (COL_HOST_CONTEXT[3], &[COL_HOST_SUSPEND, COL_HOST_CONTEXT_READ]),
    (COL_HOST_CONTEXT[4], &[COL_HOST_SUSPEND, COL_HOST_CONTEXT_READ]),
    (COL_HOST_CONTEXT[5], &[COL_HOST_SUSPEND, COL_HOST_CONTEXT_READ]),
    (COL_HOST_CONTEXT[6], &[COL_HOST_SUSPEND, COL_HOST_CONTEXT_READ]),
    (COL_HOST_CONTEXT[7], &[COL_HOST_SUSPEND, COL_HOST_CONTEXT_READ]),
    (
        COL_HOST_IMPORT_SCHEDULE_COUNTS,
        &[COL_HOST_CALL_ACTIVE, COL_HOST_RESUME],
    ),
];

/// Columns that are `lo + 1` (module docs), with their `lo`.
pub const PLUS_ONE: &[(usize, usize)] = &[
    (COL_STACK_READ_ADDR_HI[0], COL_STACK_READ_ADDR_LO[0]),
    (COL_STACK_READ_ADDR_HI[1], COL_STACK_READ_ADDR_LO[1]),
    (COL_STACK_READ_ADDR_HI[2], COL_STACK_READ_ADDR_LO[2]),
    (COL_LOCAL_ADDR_HI, COL_LOCAL_ADDR_LO),
];

/// The gate rows of `PORT_COVERS`. Panics when a gate is not the `When`
/// activation of a port that binds its column, since the row would then
/// prove no range.
pub(crate) fn push_port_cover_rows(builder: &mut WasmTaggedR1csBuilder<'_>, layout: &WasmRelationLayout) {
    let binds = |column: usize, gate: usize| {
        layout.auxiliary.memory.entries().iter().any(|memory| {
            memory.ports.iter().any(|port| {
                port.activation == MemoryPortActivation::When(gate)
                    && (port.value_column == column || port.address_columns.contains(&column))
            })
        })
    };
    builder.with_tag(
        neo_application::ConstraintTag::new(
            std::borrow::Cow::Borrowed("port range cover"),
            WasmConstraintScope::Always,
        ),
        |builder| {
            for &(column, gates) in PORT_COVERS {
                assert!(
                    gates.iter().all(|&gate| binds(column, gate)),
                    "each gate of column {column} activates a port that binds it"
                );
                let open = std::iter::once((COL_ONE, F::ONE)).chain(gates.iter().map(|&gate| (gate, -F::ONE)));
                builder.push_row(open, [(column, F::ONE)], []);
            }
        },
    );
}
