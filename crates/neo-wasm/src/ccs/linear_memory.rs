//! Owns the per-step linear-memory CCS rows.
//!
//! Each wasm load/store opcode normalizes its effective byte address into
//! `(lane0_addr, byte_offset, offset_is[0..4])`, then selects which
//! per-lane bytes flow to/from the operand stack based on access width
//! (byte / half / full / double) and alignment. The functions below
//! emit those rows, gated by the relevant opcode/width selectors so
//! exactly one branch is active per row.

use super::super::gadgets::{push_gated_linear_zero, push_u32_le_bytes_decomp};
use super::super::isa::{WasmMemoryAccessKind, WasmMemoryExtension, WasmOpcode};
use super::super::layout::{selector_col, COL_ONE};
use super::super::lookup_binding_builder::{Column, LinearMemoryColumns, OperandStackColumns, SignExtensionColumns};
use super::super::tagged_r1cs_builder::WasmTaggedR1csBuilder;
use super::{f_u64, idx, linear_memory_ops, opcode_tag, shared};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

type R1csBuilder = WasmTaggedR1csBuilder;

fn op_selector(op: WasmOpcode) -> usize {
    selector_col(op).expect("wasm opcode selector")
}

fn op_selectors(ops: &[WasmOpcode]) -> impl Iterator<Item = (usize, F)> + '_ {
    ops.iter().copied().map(|op| (op_selector(op), F::ONE))
}

fn memory_ops_by_kind(kind: WasmMemoryAccessKind) -> Vec<WasmOpcode> {
    linear_memory_ops()
        .into_iter()
        .filter(|op| {
            op.memory_access_info()
                .is_some_and(|access| access.kind == kind)
        })
        .collect()
}

fn memory_ops_by_width(width_bytes: u8) -> Vec<WasmOpcode> {
    linear_memory_ops()
        .into_iter()
        .filter(|op| {
            op.memory_access_info()
                .is_some_and(|access| access.width_bytes == width_bytes)
        })
        .collect()
}

fn load_ops_by_result_extension(result_bits: u8, extension: WasmMemoryExtension) -> Vec<WasmOpcode> {
    linear_memory_ops()
        .into_iter()
        .filter(|op| {
            op.memory_access_info().is_some_and(|access| {
                access.kind == WasmMemoryAccessKind::Load
                    && access.result_bits == result_bits
                    && access.extension == extension
            })
        })
        .collect()
}

pub(super) fn push_linear_memory_constraints(
    b: &mut R1csBuilder,
    stack: &OperandStackColumns,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    // Derived from `WasmOpcode::uses_linear_memory` so it stays in sync
    // with the single declaration on `WasmOpcodeInfo`. Adding a new memory
    // opcode here is a no-op once `uses_linear_memory` returns true for it.
    let linear_memory_selectors: Vec<usize> = linear_memory_ops().into_iter().map(op_selector).collect();
    push_address_normalization(b, stack, linear_memory, &linear_memory_selectors);
    push_width_selector_constraints(b, linear_memory);
    push_width_opcode_bindings(b, linear_memory);
    push_lane_usage_constraints(b, linear_memory);
    push_lane_direction_gates(b, linear_memory);
    push_lane_adjacency_constraints(b, linear_memory);
    push_access_byte_bindings(b, stack, linear_memory, sign_extension);
    push_byte_preservation_constraints(b, linear_memory);

    push_load_routing_constraints(b, stack, linear_memory, sign_extension);
    push_i64_load_extension_constraints(b, stack, linear_memory, sign_extension);
    push_store_routing_constraints(b, stack, linear_memory, sign_extension);
}

/// On store rows, preserve each touched lane byte outside the store's write
/// window by constraining `lane_byte == lane_byte_before`.
///
/// The same width/offset selectors also fire on loads, where this only shapes
/// unused `_before` scratch; loads emit only Read tuples in the memory spec.
/// To keep the CCS small, cases are inverted by byte slot: each row gates a
/// byte equality by the sum of all cases that preserve that slot.
fn push_byte_preservation_constraints(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    let lanes = [
        (linear_memory.lane0_bytes, linear_memory.lane0_bytes_before),
        (linear_memory.lane1_bytes, linear_memory.lane1_bytes_before),
        (linear_memory.lane2_bytes, linear_memory.lane2_bytes_before),
    ];
    let lane_byte_width = linear_memory.lane0_bytes.len();

    let cases: &[(Column, usize, usize)] = &[
        (linear_memory.byte_width_offset_is[0], 0, 1),
        (linear_memory.byte_width_offset_is[1], 1, 1),
        (linear_memory.byte_width_offset_is[2], 2, 1),
        (linear_memory.byte_width_offset_is[3], 3, 1),
        (linear_memory.half_width_offset_is[0], 0, 2),
        (linear_memory.half_width_offset_is[1], 1, 2),
        (linear_memory.half_width_offset_is[2], 2, 2),
        (linear_memory.half_width_offset_is[3], 3, 2),
        // full-width at offset 0 writes all 4 bytes of lane0, no preservation needed.
        (linear_memory.full_width_offset_is[1], 1, 4),
        (linear_memory.full_width_offset_is[2], 2, 4),
        (linear_memory.full_width_offset_is[3], 3, 4),
        // double-width offset_is is split by direction (load vs store); cover both.
        (linear_memory.i64_load_offset_is[1], 1, 8),
        (linear_memory.i64_load_offset_is[2], 2, 8),
        (linear_memory.i64_load_offset_is[3], 3, 8),
        (linear_memory.i64_store_offset_is[1], 1, 8),
        (linear_memory.i64_store_offset_is[2], 2, 8),
        (linear_memory.i64_store_offset_is[3], 3, 8),
    ];
    let cases_meta: Vec<(
        // offset selector
        Column,
        // tuples of (lane, offset_in_lane) of slots/bytes written to by this offset selector
        std::collections::BTreeSet<(usize, usize)>,
        // projection of the first coordinate of the above (so only the lanes affected by this offset selector)
        std::collections::BTreeSet<usize>,
    )> = cases
        .iter()
        .map(|&(case_sel, offset, width)| {
            let slots_written: std::collections::BTreeSet<(usize, usize)> = (0..width)
                .map(|i| ((offset + i) / lane_byte_width, (offset + i) % lane_byte_width))
                .collect();
            let lanes_written: std::collections::BTreeSet<usize> = slots_written.iter().map(|&(l, _)| l).collect();
            (case_sel, slots_written, lanes_written)
        })
        .collect();
    b.with_tag(shared("linear memory byte preservation", &linear_memory_ops()), |b| {
        for (lane, (bytes, bytes_before)) in lanes.into_iter().enumerate() {
            for (j, (byte, byte_before)) in bytes.into_iter().zip(bytes_before).enumerate() {
                let selectors_preserving: Vec<Column> = cases_meta
                    .iter()
                    .filter(|(_, slots_written, lanes_written)| {
                        lanes_written.contains(&lane) && !slots_written.contains(&(lane, j))
                    })
                    .map(|(case_sel, _, _)| *case_sel)
                    .collect();
                if selectors_preserving.is_empty() {
                    // No case preserves this slot. lane1[0] and lane2[0] are
                    // always written whenever their lanes are touched.
                    continue;
                }
                b.push_row(
                    selectors_preserving.into_iter().map(|c| (idx(c), F::ONE)),
                    [(idx(byte), F::ONE), (idx(byte_before), -F::ONE)],
                    [],
                );
            }
        }
    });
}

fn push_address_normalization(
    b: &mut R1csBuilder,
    stack: &OperandStackColumns,
    linear_memory: &LinearMemoryColumns,
    linear_memory_selectors: &[usize],
) {
    b.with_tag(
        shared("linear memory address normalization", &linear_memory_ops()),
        |b| {
            // offset_is is one-hot
            b.push_row(
                linear_memory_selectors
                    .iter()
                    .map(|&selector| (selector, F::ONE)),
                [
                    (idx(linear_memory.offset_is[0]), F::ONE),
                    (idx(linear_memory.offset_is[1]), F::ONE),
                    (idx(linear_memory.offset_is[2]), F::ONE),
                    (idx(linear_memory.offset_is[3]), F::ONE),
                    (COL_ONE, -F::ONE),
                ],
                [],
            );
            // byte_offset is in 0..4 and decomposed into offset_is
            b.push_row(
                linear_memory_selectors
                    .iter()
                    .map(|&selector| (selector, F::ONE)),
                [
                    (idx(linear_memory.byte_offset), F::ONE),
                    (idx(linear_memory.offset_is[1]), -F::ONE),
                    (idx(linear_memory.offset_is[2]), -f_u64(2)),
                    (idx(linear_memory.offset_is[3]), -f_u64(3)),
                ],
                [],
            );
            // byte_offset contains the remainder mod 4 of the actual address
            b.push_row(
                linear_memory_selectors
                    .iter()
                    .map(|&selector| (selector, F::ONE)),
                [
                    (idx(linear_memory.lane0_addr), f_u64(4)),
                    (idx(linear_memory.byte_offset), F::ONE),
                    (idx(stack.read0_value), -F::ONE),
                    (idx(linear_memory.imm_offset), -F::ONE),
                ],
                [],
            );

            push_u32_le_bytes_decomp(
                b,
                linear_memory_selectors.iter().copied(),
                idx(linear_memory.lane0_value),
                linear_memory.lane0_bytes.map(idx),
            );
            push_u32_le_bytes_decomp(
                b,
                linear_memory_selectors.iter().copied(),
                idx(linear_memory.lane1_value),
                linear_memory.lane1_bytes.map(idx),
            );
            push_u32_le_bytes_decomp(
                b,
                [op_selector(WasmOpcode::I64Load), op_selector(WasmOpcode::I64Store)],
                idx(linear_memory.lane2_value),
                linear_memory.lane2_bytes.map(idx),
            );
            // Same byte_decomp on the `_before` columns so per-byte subword
            // preservation constraints can reference individual bytes of the
            // prior lane state. Gated identically: `_before` only matters when
            // the lane is actually accessed.
            push_u32_le_bytes_decomp(
                b,
                linear_memory_selectors.iter().copied(),
                idx(linear_memory.lane0_value_before),
                linear_memory.lane0_bytes_before.map(idx),
            );
            push_u32_le_bytes_decomp(
                b,
                linear_memory_selectors.iter().copied(),
                idx(linear_memory.lane1_value_before),
                linear_memory.lane1_bytes_before.map(idx),
            );
            push_u32_le_bytes_decomp(
                b,
                [op_selector(WasmOpcode::I64Load), op_selector(WasmOpcode::I64Store)],
                idx(linear_memory.lane2_value_before),
                linear_memory.lane2_bytes_before.map(idx),
            );
        },
    );
}

fn push_width_selector_constraints(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    b.with_tag(shared("linear memory width selectors", &linear_memory_ops()), |b| {
        // Under each gate, both offset families are one-hot, so the weighted 1..4
        // fingerprint is injective and can replace four per-case equalities.
        for (width_flag, width_offset_is) in [
            (linear_memory.is_byte_width, linear_memory.byte_width_offset_is),
            (linear_memory.is_half_width, linear_memory.half_width_offset_is),
            (linear_memory.is_full_width, linear_memory.full_width_offset_is),
            (linear_memory.is_double_width, linear_memory.double_width_offset_is),
        ] {
            push_width_offset_family_constraints(b, width_flag, width_offset_is, linear_memory.offset_is);
        }
        b.push_linear_zero(
            [
                (idx(linear_memory.i64_load_offset_is[0]), F::ONE),
                (idx(linear_memory.i64_load_offset_is[1]), F::ONE),
                (idx(linear_memory.i64_load_offset_is[2]), F::ONE),
                (idx(linear_memory.i64_load_offset_is[3]), F::ONE),
                (op_selector(WasmOpcode::I64Load), -F::ONE),
            ]
            .into_iter(),
        );
        push_gated_linear_zero(
            b,
            op_selector(WasmOpcode::I64Load),
            [
                (idx(linear_memory.i64_load_offset_is[0]), F::ONE),
                (idx(linear_memory.i64_load_offset_is[1]), f_u64(2)),
                (idx(linear_memory.i64_load_offset_is[2]), f_u64(3)),
                (idx(linear_memory.i64_load_offset_is[3]), f_u64(4)),
                (idx(linear_memory.double_width_offset_is[0]), -F::ONE),
                (idx(linear_memory.double_width_offset_is[1]), -f_u64(2)),
                (idx(linear_memory.double_width_offset_is[2]), -f_u64(3)),
                (idx(linear_memory.double_width_offset_is[3]), -f_u64(4)),
            ],
        );
        b.push_linear_zero(
            [
                (idx(linear_memory.i64_store_offset_is[0]), F::ONE),
                (idx(linear_memory.i64_store_offset_is[1]), F::ONE),
                (idx(linear_memory.i64_store_offset_is[2]), F::ONE),
                (idx(linear_memory.i64_store_offset_is[3]), F::ONE),
                (op_selector(WasmOpcode::I64Store), -F::ONE),
            ]
            .into_iter(),
        );
        push_gated_linear_zero(
            b,
            op_selector(WasmOpcode::I64Store),
            [
                (idx(linear_memory.i64_store_offset_is[0]), F::ONE),
                (idx(linear_memory.i64_store_offset_is[1]), f_u64(2)),
                (idx(linear_memory.i64_store_offset_is[2]), f_u64(3)),
                (idx(linear_memory.i64_store_offset_is[3]), f_u64(4)),
                (idx(linear_memory.double_width_offset_is[0]), -F::ONE),
                (idx(linear_memory.double_width_offset_is[1]), -f_u64(2)),
                (idx(linear_memory.double_width_offset_is[2]), -f_u64(3)),
                (idx(linear_memory.double_width_offset_is[3]), -f_u64(4)),
            ],
        );
    });
}

fn push_width_offset_family_constraints(
    b: &mut R1csBuilder,
    width_flag: Column,
    width_offset_is: [Column; 4],
    offset_is: [Column; 4],
) {
    b.push_linear_zero(
        width_offset_is
            .into_iter()
            .map(|col| (idx(col), F::ONE))
            .chain([(idx(width_flag), -F::ONE)]),
    );
    push_gated_linear_zero(
        b,
        idx(width_flag),
        [
            (idx(width_offset_is[0]), F::ONE),
            (idx(width_offset_is[1]), f_u64(2)),
            (idx(width_offset_is[2]), f_u64(3)),
            (idx(width_offset_is[3]), f_u64(4)),
            (idx(offset_is[0]), -F::ONE),
            (idx(offset_is[1]), -f_u64(2)),
            (idx(offset_is[2]), -f_u64(3)),
            (idx(offset_is[3]), -f_u64(4)),
        ],
    );
}

fn push_width_opcode_bindings(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    // Pin each width family to the opcodes that use it. Combined with the
    // unconditional `is_X_width = sum(X_width_offset_is)` above, this forces
    // the correct width-offset family active per opcode. Without it a prover
    // could zero the whole width family (is_X_width = 0, all X_width_offset_is
    // = 0), which would vacuously satisfy the byte-routing gates — those are
    // gated by `X_width_offset_is[k]`, not by the opcode selector — and so
    // bypass the access-byte ↔ lane-byte binding entirely.
    b.with_tag(
        shared("linear memory width opcode binding", &linear_memory_ops()),
        |b| {
            for (width_flag, width_bytes) in [
                (linear_memory.is_byte_width, 1u8),
                (linear_memory.is_half_width, 2),
                (linear_memory.is_full_width, 4),
                (linear_memory.is_double_width, 8),
            ] {
                let terms = std::iter::once((idx(width_flag), F::ONE)).chain(
                    memory_ops_by_width(width_bytes)
                        .into_iter()
                        .map(|op| (op_selector(op), -F::ONE)),
                );
                b.push_linear_zero(terms);
            }
        },
    );
}

fn push_lane_usage_constraints(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    b.with_tag(shared("linear memory lane usage", &linear_memory_ops()), |b| {
        let half_width_ops = memory_ops_by_width(2);
        let full_width_ops = memory_ops_by_width(4);
        let double_width_ops = memory_ops_by_width(8);
        b.push_row(
            op_selectors(&half_width_ops),
            [
                (idx(linear_memory.use_lane1), F::ONE),
                (idx(linear_memory.half_width_offset_is[3]), -F::ONE),
            ],
            [],
        );
        b.push_row(
            // Every full-width (4-byte) access — i32.load/store, i64.store32,
            // and i64.load32_{u,s} — crosses into lane1 at offsets 1/2/3, so
            // they must all force use_lane1 there. Omitting the i64 load32 ops
            // would let an unaligned load satisfy the byte shuffle from
            // unconstrained lane1 bytes without activating the lane1 access.
            op_selectors(&full_width_ops),
            [
                (idx(linear_memory.use_lane1), F::ONE),
                (idx(linear_memory.full_width_offset_is[1]), -F::ONE),
                (idx(linear_memory.full_width_offset_is[2]), -F::ONE),
                (idx(linear_memory.full_width_offset_is[3]), -F::ONE),
            ],
            [],
        );
        b.push_row(
            op_selectors(&double_width_ops),
            [(idx(linear_memory.use_lane1), F::ONE), (COL_ONE, -F::ONE)],
            [],
        );
        b.push_row(
            op_selectors(&double_width_ops),
            [
                (idx(linear_memory.use_lane2), F::ONE),
                (idx(linear_memory.double_width_offset_is[1]), -F::ONE),
                (idx(linear_memory.double_width_offset_is[2]), -F::ONE),
                (idx(linear_memory.double_width_offset_is[3]), -F::ONE),
            ],
            [],
        );
    });
}

/// Bind `laneN_load_active = use_laneN · Σ load_selectors` and likewise for
/// store, via one R1CS quadratic per lane × direction. These are the gate
/// columns the `linear_memory` `WasmMemorySpec` uses to fire its Read
/// (load) and Write+RMW (store) entries — keeping load rows from writing
/// to the cells log.
fn push_lane_direction_gates(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    let load_ops = memory_ops_by_kind(WasmMemoryAccessKind::Load);
    let store_ops = memory_ops_by_kind(WasmMemoryAccessKind::Store);
    b.with_tag(
        shared("linear memory lane direction gates", &linear_memory_ops()),
        |b| {
            for (use_lane_col, load_active_col, store_active_col) in [
                (
                    linear_memory.use_lane0,
                    linear_memory.lane0_load_active,
                    linear_memory.lane0_store_active,
                ),
                (
                    linear_memory.use_lane1,
                    linear_memory.lane1_load_active,
                    linear_memory.lane1_store_active,
                ),
                (
                    linear_memory.use_lane2,
                    linear_memory.lane2_load_active,
                    linear_memory.lane2_store_active,
                ),
            ] {
                // `(use_laneN) · (Σ load selectors) = laneN_load_active`.
                b.push_row(
                    [(idx(use_lane_col), F::ONE)],
                    load_ops.iter().map(|&op| (op_selector(op), F::ONE)),
                    [(idx(load_active_col), F::ONE)],
                );
                // `(use_laneN) · (Σ store selectors) = laneN_store_active`.
                b.push_row(
                    [(idx(use_lane_col), F::ONE)],
                    store_ops.iter().map(|&op| (op_selector(op), F::ONE)),
                    [(idx(store_active_col), F::ONE)],
                );
            }
        },
    );
}

fn push_lane_adjacency_constraints(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    b.with_tag(shared("linear memory lane adjacency", &linear_memory_ops()), |b| {
        push_gated_linear_zero(
            b,
            idx(linear_memory.use_lane1),
            [
                (idx(linear_memory.lane1_addr), F::ONE),
                (idx(linear_memory.lane0_addr), -F::ONE),
                (COL_ONE, -F::ONE),
            ],
        );
        push_gated_linear_zero(
            b,
            idx(linear_memory.use_lane2),
            [
                (idx(linear_memory.lane2_addr), F::ONE),
                (idx(linear_memory.lane1_addr), -F::ONE),
                (COL_ONE, -F::ONE),
            ],
        );
    });
}

fn push_access_byte_bindings(
    b: &mut R1csBuilder,
    stack: &OperandStackColumns,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    // Bind the shared low-byte scratch to the active value's low 32 bits and
    // `access_bytes_hi` to the high 32 bits (i64 only). The byte views are
    // direction-agnostic at the byte-shuffle layer (gated by offset, not
    // opcode), so the direction-specific binding lives here: write0 on loads,
    // read1 on stores; the i64 ops additionally bind the hi half.
    let load_low_byte_ops = memory_ops_by_kind(WasmMemoryAccessKind::Load);
    b.with_tag(shared("linear memory low bytes (loads)", &load_low_byte_ops), |b| {
        push_u32_le_bytes_decomp(
            b,
            load_low_byte_ops.iter().copied().map(op_selector),
            idx(stack.write0_value),
            sign_extension.bytes.map(idx),
        );
    });
    let store_low_byte_ops = memory_ops_by_kind(WasmMemoryAccessKind::Store);
    b.with_tag(shared("linear memory low bytes (stores)", &store_low_byte_ops), |b| {
        push_u32_le_bytes_decomp(
            b,
            store_low_byte_ops.iter().copied().map(op_selector),
            idx(stack.read1_value),
            sign_extension.bytes.map(idx),
        );
    });
    b.with_tag(
        shared(
            "linear memory access bytes hi (i64)",
            &[WasmOpcode::I64Load, WasmOpcode::I64Store],
        ),
        |b| {
            push_u32_le_bytes_decomp(
                b,
                [op_selector(WasmOpcode::I64Load)],
                idx(stack.write0_value_hi),
                linear_memory.access_bytes_hi.map(idx),
            );
            push_u32_le_bytes_decomp(
                b,
                [op_selector(WasmOpcode::I64Store)],
                idx(stack.read1_value_hi),
                linear_memory.access_bytes_hi.map(idx),
            );
        },
    );
}

fn push_load_routing_constraints(
    b: &mut R1csBuilder,
    stack: &OperandStackColumns,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    b.with_tag(
        opcode_tag("linear memory load32 byte routing", WasmOpcode::I32Load),
        |b| {
            push_linear_memory_load32_byte_selection(b, linear_memory, sign_extension);
        },
    );
    b.with_tag(
        opcode_tag("linear memory load8_s routing", WasmOpcode::I32Load8S),
        |b| {
            push_linear_memory_load8_s_constraints(b, linear_memory, sign_extension);
        },
    );
    b.with_tag(
        opcode_tag("linear memory load8_u routing", WasmOpcode::I32Load8U),
        |b| {
            push_linear_memory_load8_u_constraints(b, linear_memory, sign_extension);
        },
    );
    b.with_tag(
        opcode_tag("linear memory load16_s routing", WasmOpcode::I32Load16S),
        |b| {
            push_linear_memory_load16_s_constraints(b, linear_memory, sign_extension);
        },
    );
    b.with_tag(
        opcode_tag("linear memory load16_u routing", WasmOpcode::I32Load16U),
        |b| {
            push_linear_memory_load16_u_constraints(b, linear_memory, sign_extension);
        },
    );
    // i64.loadN_u: load N bytes into write0_value (lo limb) with the same
    // subword / full-width byte machinery as the i32 unsigned loads, then
    // zero-extend by pinning the output hi limb to 0. i64.load32_u rides the
    // full-width offset gates that already route i32.load, so it needs no
    // dedicated byte-selection row — only the hi-limb pin below.
    b.with_tag(
        opcode_tag("linear memory i64.load8_u routing", WasmOpcode::I64Load8U),
        |b| {
            push_linear_memory_load_subword_constraints(
                b,
                op_selector(WasmOpcode::I64Load8U),
                1,
                linear_memory,
                sign_extension,
            );
        },
    );
    b.with_tag(
        opcode_tag("linear memory i64.load16_u routing", WasmOpcode::I64Load16U),
        |b| {
            push_linear_memory_load_subword_constraints(
                b,
                op_selector(WasmOpcode::I64Load16U),
                2,
                linear_memory,
                sign_extension,
            );
        },
    );
    b.with_tag(opcode_tag("linear memory load64 routing", WasmOpcode::I64Load), |b| {
        push_linear_memory_load64_constraints(b, stack, linear_memory, sign_extension);
    });
}

fn push_i64_load_extension_constraints(
    b: &mut R1csBuilder,
    stack: &OperandStackColumns,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    let i64_unsigned_load_ops = load_ops_by_result_extension(64, WasmMemoryExtension::Zero);
    b.with_tag(
        shared("linear memory i64 unsigned load high zero", &i64_unsigned_load_ops),
        |b| {
            b.push_row(
                op_selectors(&i64_unsigned_load_ops),
                [(idx(stack.write0_value_hi), F::ONE)],
                [],
            );
        },
    );
    // i64.loadN_s: the lo limb sign-extends exactly like the i32 signed loads
    // (load8_s / load16_s reuse the subword sign-extension helper; load32_s
    // keeps the full word routed by the full-width offset gates and only
    // extracts the sign bit from its top byte). The hi limb is then filled
    // with the replicated sign bit: 0xFFFF_FFFF iff negative, else 0.
    b.with_tag(
        opcode_tag("linear memory i64.load8_s routing", WasmOpcode::I64Load8S),
        |b| {
            push_linear_memory_load_signed_subword_constraints(
                b,
                op_selector(WasmOpcode::I64Load8S),
                1,
                idx(sign_extension.bytes[0]),
                linear_memory,
                sign_extension,
            );
        },
    );
    b.with_tag(
        opcode_tag("linear memory i64.load16_s routing", WasmOpcode::I64Load16S),
        |b| {
            push_linear_memory_load_signed_subword_constraints(
                b,
                op_selector(WasmOpcode::I64Load16S),
                2,
                idx(sign_extension.bytes[1]),
                linear_memory,
                sign_extension,
            );
        },
    );
    b.with_tag(
        opcode_tag("linear memory i64.load32_s sign extract", WasmOpcode::I64Load32S),
        |b| {
            // Lo limb (the full word) is routed by the full-width offset gates;
            // here we only split the top byte into low7 + 128 * sign bit.
            push_gated_linear_zero(
                b,
                op_selector(WasmOpcode::I64Load32S),
                [
                    (idx(sign_extension.bytes[3]), F::ONE),
                    (idx(sign_extension.low7), -F::ONE),
                    (idx(sign_extension.bit), -f_u64(128)),
                ],
            );
        },
    );
    let i64_signed_load_ops = load_ops_by_result_extension(64, WasmMemoryExtension::Sign);
    b.with_tag(
        shared("linear memory i64 signed load high fill", &i64_signed_load_ops),
        |b| {
            // write0_value_hi = sign bit * 0xFFFF_FFFF (all-ones iff the
            // sign bit is set, else 0).
            b.push_row(
                op_selectors(&i64_signed_load_ops),
                [
                    (idx(stack.write0_value_hi), F::ONE),
                    (idx(sign_extension.bit), -f_u64(0xFFFF_FFFF)),
                ],
                [],
            );
        },
    );
}

fn push_store_routing_constraints(
    b: &mut R1csBuilder,
    stack: &OperandStackColumns,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    b.with_tag(
        opcode_tag("linear memory store32 byte routing", WasmOpcode::I32Store),
        |b| {
            push_linear_memory_store32_byte_selection(b, linear_memory, sign_extension);
        },
    );
    b.with_tag(opcode_tag("linear memory store8 routing", WasmOpcode::I32Store8), |b| {
        push_linear_memory_store8_constraints(b, linear_memory, sign_extension);
    });
    b.with_tag(
        opcode_tag("linear memory store16 routing", WasmOpcode::I32Store16),
        |b| {
            push_linear_memory_store16_constraints(b, linear_memory, sign_extension);
        },
    );
    b.with_tag(opcode_tag("linear memory store64 routing", WasmOpcode::I64Store), |b| {
        push_linear_memory_store64_constraints(b, stack, linear_memory, sign_extension);
    });
    // i64.storeN ops truncate the lo limb to N bytes and reuse the same
    // byte-selection / subword machinery as i32.storeN. The hi limb is read
    // from the stack (wide_values_enabled is on) but never written to memory.
    // i64.store32 piggybacks on the full-width offset gates that already
    // route i32.store, so it needs no extra rows here.
    b.with_tag(
        opcode_tag("linear memory i64.store8 routing", WasmOpcode::I64Store8),
        |b| {
            push_linear_memory_store_subword_constraints(
                b,
                op_selector(WasmOpcode::I64Store8),
                1,
                linear_memory,
                sign_extension,
            );
        },
    );
    b.with_tag(
        opcode_tag("linear memory i64.store16 routing", WasmOpcode::I64Store16),
        |b| {
            push_linear_memory_store_subword_constraints(
                b,
                op_selector(WasmOpcode::I64Store16),
                2,
                linear_memory,
                sign_extension,
            );
        },
    );
}

fn push_linear_memory_load32_byte_selection(
    b: &mut R1csBuilder,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    for (selector, lane_bytes) in [
        (idx(linear_memory.full_width_offset_is[0]), linear_memory.lane0_bytes),
        (
            idx(linear_memory.full_width_offset_is[1]),
            [
                linear_memory.lane0_bytes[1],
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
            ],
        ),
        (
            idx(linear_memory.full_width_offset_is[2]),
            [
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
            ],
        ),
        (
            idx(linear_memory.full_width_offset_is[3]),
            [
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
                linear_memory.lane1_bytes[2],
            ],
        ),
    ] {
        push_matching_byte_constraints(b, selector, sign_extension.bytes, lane_bytes);
    }
}

fn push_linear_memory_store32_byte_selection(
    b: &mut R1csBuilder,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    for (selector, access_bytes, lane_bytes) in [
        (
            idx(linear_memory.full_width_offset_is[0]),
            sign_extension.bytes,
            linear_memory.lane0_bytes,
        ),
        (
            idx(linear_memory.full_width_offset_is[1]),
            sign_extension.bytes,
            [
                linear_memory.lane0_bytes[1],
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
            ],
        ),
        (
            idx(linear_memory.full_width_offset_is[2]),
            sign_extension.bytes,
            [
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
            ],
        ),
        (
            idx(linear_memory.full_width_offset_is[3]),
            sign_extension.bytes,
            [
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
                linear_memory.lane1_bytes[2],
            ],
        ),
    ] {
        push_matching_byte_constraints(b, selector, access_bytes, lane_bytes);
    }
}

fn push_linear_memory_load8_u_constraints(
    b: &mut R1csBuilder,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    push_linear_memory_load_subword_constraints(
        b,
        selector_col(WasmOpcode::I32Load8U).unwrap(),
        1,
        linear_memory,
        sign_extension,
    );
}

fn push_linear_memory_load8_s_constraints(
    b: &mut R1csBuilder,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    push_linear_memory_load_signed_subword_constraints(
        b,
        selector_col(WasmOpcode::I32Load8S).unwrap(),
        1,
        idx(sign_extension.bytes[0]),
        linear_memory,
        sign_extension,
    );
}

fn push_linear_memory_store8_constraints(
    b: &mut R1csBuilder,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    push_linear_memory_store_subword_constraints(
        b,
        selector_col(WasmOpcode::I32Store8).unwrap(),
        1,
        linear_memory,
        sign_extension,
    );
}

fn push_linear_memory_load16_s_constraints(
    b: &mut R1csBuilder,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    push_linear_memory_load_signed_subword_constraints(
        b,
        selector_col(WasmOpcode::I32Load16S).unwrap(),
        2,
        idx(sign_extension.bytes[1]),
        linear_memory,
        sign_extension,
    );
}

fn push_linear_memory_load16_u_constraints(
    b: &mut R1csBuilder,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    push_linear_memory_load_subword_constraints(
        b,
        selector_col(WasmOpcode::I32Load16U).unwrap(),
        2,
        linear_memory,
        sign_extension,
    );
}

fn push_linear_memory_load64_constraints(
    b: &mut R1csBuilder,
    stack: &OperandStackColumns,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    // Aligned (offset == 0): the 8 loaded bytes are exactly `lane0` (lo
    // limb) and `lane1` (hi limb), so a word-level binding skips the
    // byte-by-byte match.
    push_gated_linear_zero(
        b,
        idx(linear_memory.i64_load_offset_is[0]),
        [
            (idx(stack.write0_value), F::ONE),
            (idx(linear_memory.lane0_value), -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        idx(linear_memory.i64_load_offset_is[0]),
        [
            (idx(stack.write0_value_hi), F::ONE),
            (idx(linear_memory.lane1_value), -F::ONE),
        ],
    );
    // Unaligned (offset ∈ {1,2,3}): the 8 loaded bytes straddle 2 or 3
    // lanes; match the shared low-byte scratch / `access_bytes_hi` against
    // the shuffled lane bytes. The byte-decomp above binds those to
    // `stack.write0_value{,_hi}`.
    for (case_selector, low_bytes, high_bytes) in [
        (
            idx(linear_memory.i64_load_offset_is[1]),
            [
                linear_memory.lane0_bytes[1],
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
            ],
            [
                linear_memory.lane1_bytes[1],
                linear_memory.lane1_bytes[2],
                linear_memory.lane1_bytes[3],
                linear_memory.lane2_bytes[0],
            ],
        ),
        (
            idx(linear_memory.i64_load_offset_is[2]),
            [
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
            ],
            [
                linear_memory.lane1_bytes[2],
                linear_memory.lane1_bytes[3],
                linear_memory.lane2_bytes[0],
                linear_memory.lane2_bytes[1],
            ],
        ),
        (
            idx(linear_memory.i64_load_offset_is[3]),
            [
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
                linear_memory.lane1_bytes[2],
            ],
            [
                linear_memory.lane1_bytes[3],
                linear_memory.lane2_bytes[0],
                linear_memory.lane2_bytes[1],
                linear_memory.lane2_bytes[2],
            ],
        ),
    ] {
        for (byte, lane_byte) in sign_extension.bytes.into_iter().zip(low_bytes) {
            push_gated_linear_zero(b, case_selector, [(idx(byte), F::ONE), (idx(lane_byte), -F::ONE)]);
        }
        for (byte, lane_byte) in linear_memory.access_bytes_hi.into_iter().zip(high_bytes) {
            push_gated_linear_zero(b, case_selector, [(idx(byte), F::ONE), (idx(lane_byte), -F::ONE)]);
        }
    }
}

fn push_linear_memory_store64_constraints(
    b: &mut R1csBuilder,
    stack: &OperandStackColumns,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    push_gated_linear_zero(
        b,
        idx(linear_memory.i64_store_offset_is[0]),
        [
            (idx(stack.read1_value), F::ONE),
            (idx(linear_memory.lane0_value), -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        idx(linear_memory.i64_store_offset_is[0]),
        [
            (idx(stack.read1_value_hi), F::ONE),
            (idx(linear_memory.lane1_value), -F::ONE),
        ],
    );
    for (case_selector, target_bytes) in [
        (
            idx(linear_memory.i64_store_offset_is[1]),
            [
                linear_memory.lane0_bytes[1],
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
                linear_memory.lane1_bytes[2],
                linear_memory.lane1_bytes[3],
                linear_memory.lane2_bytes[0],
            ],
        ),
        (
            idx(linear_memory.i64_store_offset_is[2]),
            [
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
                linear_memory.lane1_bytes[2],
                linear_memory.lane1_bytes[3],
                linear_memory.lane2_bytes[0],
                linear_memory.lane2_bytes[1],
            ],
        ),
        (
            idx(linear_memory.i64_store_offset_is[3]),
            [
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
                linear_memory.lane1_bytes[2],
                linear_memory.lane1_bytes[3],
                linear_memory.lane2_bytes[0],
                linear_memory.lane2_bytes[1],
                linear_memory.lane2_bytes[2],
            ],
        ),
    ] {
        for (byte, target_byte) in sign_extension
            .bytes
            .into_iter()
            .chain(linear_memory.access_bytes_hi)
            .zip(target_bytes)
        {
            push_gated_linear_zero(b, case_selector, [(idx(byte), F::ONE), (idx(target_byte), -F::ONE)]);
        }
    }
}

fn push_linear_memory_store16_constraints(
    b: &mut R1csBuilder,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    push_linear_memory_store_subword_constraints(
        b,
        selector_col(WasmOpcode::I32Store16).unwrap(),
        2,
        linear_memory,
        sign_extension,
    );
}

fn push_linear_memory_load_subword_constraints(
    b: &mut R1csBuilder,
    selector: usize,
    width_bytes: usize,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    for byte in &sign_extension.bytes[width_bytes..] {
        push_gated_linear_zero(b, selector, [(idx(*byte), F::ONE)]);
    }
    push_linear_memory_subword_byte_constraints(b, selector, width_bytes, linear_memory, sign_extension);
}

fn push_linear_memory_load_signed_subword_constraints(
    b: &mut R1csBuilder,
    selector: usize,
    width_bytes: usize,
    sign_source_byte: usize,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    push_linear_memory_subword_byte_constraints(b, selector, width_bytes, linear_memory, sign_extension);
    push_gated_linear_zero(
        b,
        selector,
        [
            (sign_source_byte, F::ONE),
            (idx(sign_extension.low7), -F::ONE),
            (idx(sign_extension.bit), -f_u64(128)),
        ],
    );
    for byte in &sign_extension.bytes[width_bytes..] {
        push_gated_linear_zero(
            b,
            selector,
            [(idx(*byte), F::ONE), (idx(sign_extension.bit), -f_u64(255))],
        );
    }
}

fn push_linear_memory_store_subword_constraints(
    b: &mut R1csBuilder,
    selector: usize,
    width_bytes: usize,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    push_linear_memory_subword_byte_constraints(b, selector, width_bytes, linear_memory, sign_extension);
}

fn push_linear_memory_subword_byte_constraints(
    b: &mut R1csBuilder,
    selector: usize,
    width_bytes: usize,
    linear_memory: &LinearMemoryColumns,
    sign_extension: &SignExtensionColumns,
) {
    if width_bytes == 1 {
        push_gated_linear_zero(b, selector, [(idx(linear_memory.use_lane1), F::ONE)]);
    } else {
        push_gated_linear_zero(
            b,
            selector,
            [
                (idx(linear_memory.use_lane1), F::ONE),
                (idx(linear_memory.half_width_offset_is[3]), -F::ONE),
            ],
        );
    }
    for (case_selector, lane_bytes) in [
        (
            if width_bytes == 1 {
                idx(linear_memory.byte_width_offset_is[0])
            } else {
                idx(linear_memory.half_width_offset_is[0])
            },
            [
                linear_memory.lane0_bytes[0],
                linear_memory.lane0_bytes[1],
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
            ],
        ),
        (
            if width_bytes == 1 {
                idx(linear_memory.byte_width_offset_is[1])
            } else {
                idx(linear_memory.half_width_offset_is[1])
            },
            [
                linear_memory.lane0_bytes[1],
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
            ],
        ),
        (
            if width_bytes == 1 {
                idx(linear_memory.byte_width_offset_is[2])
            } else {
                idx(linear_memory.half_width_offset_is[2])
            },
            [
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
            ],
        ),
        (
            if width_bytes == 1 {
                idx(linear_memory.byte_width_offset_is[3])
            } else {
                idx(linear_memory.half_width_offset_is[3])
            },
            [
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
                linear_memory.lane1_bytes[2],
            ],
        ),
    ] {
        for (access_byte, lane_byte) in sign_extension.bytes[..width_bytes]
            .iter()
            .zip(lane_bytes[..width_bytes].iter())
        {
            push_gated_linear_zero(
                b,
                case_selector,
                [(idx(*access_byte), F::ONE), (idx(*lane_byte), -F::ONE)],
            );
        }
    }
}

fn push_matching_byte_constraints(b: &mut R1csBuilder, selector: usize, lhs: [Column; 4], rhs: [Column; 4]) {
    for (lhs_col, rhs_col) in lhs.into_iter().zip(rhs) {
        push_gated_linear_zero(b, selector, [(idx(lhs_col), F::ONE), (idx(rhs_col), -F::ONE)]);
    }
}
