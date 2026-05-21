//! Owns the per-step linear-memory CCS rows.
//!
//! Each wasm load/store opcode normalizes its effective byte address into
//! `(lane0_addr, byte_offset, offset_is[0..4])`, then selects which
//! per-lane bytes flow to/from the operand stack based on access width
//! (byte / half / full / double) and alignment. The functions below
//! emit those rows, gated by the relevant opcode/width selectors so
//! exactly one branch is active per row.

use super::super::gadgets::{push_gated_linear_zero, push_u32_le_bytes_decomp};
use super::super::isa::WasmOpcode;
use super::super::layout::{selector_col, COL_ONE};
use super::super::lookup_binding_builder::{Column, LinearMemoryColumns, OperandStackColumns};
use super::super::tagged_r1cs_builder::WasmTaggedR1csBuilder;
use super::{f_u64, idx, opcode_tag, shared, LINEAR_MEMORY_OPS};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

type R1csBuilder = WasmTaggedR1csBuilder;

pub(super) fn push_linear_memory_constraints(
    b: &mut R1csBuilder,
    stack: &OperandStackColumns,
    linear_memory: &LinearMemoryColumns,
) {
    let load_selector = selector_col(WasmOpcode::I32Load).unwrap();
    let i64_load_selector = selector_col(WasmOpcode::I64Load).unwrap();
    let load8s_selector = selector_col(WasmOpcode::I32Load8S).unwrap();
    let load8_selector = selector_col(WasmOpcode::I32Load8U).unwrap();
    let load16s_selector = selector_col(WasmOpcode::I32Load16S).unwrap();
    let load16_selector = selector_col(WasmOpcode::I32Load16U).unwrap();
    let store_selector = selector_col(WasmOpcode::I32Store).unwrap();
    let i64_store_selector = selector_col(WasmOpcode::I64Store).unwrap();
    let store8_selector = selector_col(WasmOpcode::I32Store8).unwrap();
    let store16_selector = selector_col(WasmOpcode::I32Store16).unwrap();
    let linear_memory_selectors = [
        load_selector,
        i64_load_selector,
        load8s_selector,
        load8_selector,
        load16s_selector,
        load16_selector,
        store_selector,
        i64_store_selector,
        store8_selector,
        store16_selector,
    ];
    b.with_tag(shared("linear memory address normalization", LINEAR_MEMORY_OPS), |b| {
        b.push_row(
            linear_memory_selectors
                .into_iter()
                .map(|selector| (selector, F::ONE)),
            [
                (idx(linear_memory.lane0_addr), f_u64(4)),
                (idx(linear_memory.byte_offset), F::ONE),
                (idx(stack.read0_value), -F::ONE),
                (idx(linear_memory.imm_offset), -F::ONE),
            ],
            [],
        );
        b.push_row(
            linear_memory_selectors
                .into_iter()
                .map(|selector| (selector, F::ONE)),
            [
                (idx(linear_memory.offset_is[0]), F::ONE),
                (idx(linear_memory.offset_is[1]), F::ONE),
                (idx(linear_memory.offset_is[2]), F::ONE),
                (idx(linear_memory.offset_is[3]), F::ONE),
                (COL_ONE, -F::ONE),
            ],
            [],
        );
        b.push_row(
            linear_memory_selectors
                .into_iter()
                .map(|selector| (selector, F::ONE)),
            [
                (idx(linear_memory.byte_offset), F::ONE),
                (idx(linear_memory.offset_is[1]), -F::ONE),
                (idx(linear_memory.offset_is[2]), -f_u64(2)),
                (idx(linear_memory.offset_is[3]), -f_u64(3)),
            ],
            [],
        );
        for selector in linear_memory_selectors {
            push_u32_le_bytes_decomp(
                b,
                [selector],
                idx(linear_memory.lane0_value),
                linear_memory.lane0_bytes.map(idx),
            );
            push_u32_le_bytes_decomp(
                b,
                [selector],
                idx(linear_memory.lane1_value),
                linear_memory.lane1_bytes.map(idx),
            );
        }
        for selector in [i64_load_selector, i64_store_selector] {
            push_u32_le_bytes_decomp(
                b,
                [selector],
                idx(linear_memory.lane2_value),
                linear_memory.lane2_bytes.map(idx),
            );
        }
    });

    b.with_tag(shared("linear memory width selectors", LINEAR_MEMORY_OPS), |b| {
        // Under each gate, both offset families are one-hot, so the weighted 1..4
        // fingerprint is injective and can replace four per-case equalities.
        b.push_linear_zero(
            [
                (idx(linear_memory.byte_width_offset_is[0]), F::ONE),
                (idx(linear_memory.byte_width_offset_is[1]), F::ONE),
                (idx(linear_memory.byte_width_offset_is[2]), F::ONE),
                (idx(linear_memory.byte_width_offset_is[3]), F::ONE),
                (idx(linear_memory.is_byte_width), -F::ONE),
            ]
            .into_iter(),
        );
        push_gated_linear_zero(
            b,
            idx(linear_memory.is_byte_width),
            [
                (idx(linear_memory.byte_width_offset_is[0]), F::ONE),
                (idx(linear_memory.byte_width_offset_is[1]), f_u64(2)),
                (idx(linear_memory.byte_width_offset_is[2]), f_u64(3)),
                (idx(linear_memory.byte_width_offset_is[3]), f_u64(4)),
                (idx(linear_memory.offset_is[0]), -F::ONE),
                (idx(linear_memory.offset_is[1]), -f_u64(2)),
                (idx(linear_memory.offset_is[2]), -f_u64(3)),
                (idx(linear_memory.offset_is[3]), -f_u64(4)),
            ],
        );

        b.push_linear_zero(
            [
                (idx(linear_memory.half_width_offset_is[0]), F::ONE),
                (idx(linear_memory.half_width_offset_is[1]), F::ONE),
                (idx(linear_memory.half_width_offset_is[2]), F::ONE),
                (idx(linear_memory.half_width_offset_is[3]), F::ONE),
                (idx(linear_memory.is_half_width), -F::ONE),
            ]
            .into_iter(),
        );
        push_gated_linear_zero(
            b,
            idx(linear_memory.is_half_width),
            [
                (idx(linear_memory.half_width_offset_is[0]), F::ONE),
                (idx(linear_memory.half_width_offset_is[1]), f_u64(2)),
                (idx(linear_memory.half_width_offset_is[2]), f_u64(3)),
                (idx(linear_memory.half_width_offset_is[3]), f_u64(4)),
                (idx(linear_memory.offset_is[0]), -F::ONE),
                (idx(linear_memory.offset_is[1]), -f_u64(2)),
                (idx(linear_memory.offset_is[2]), -f_u64(3)),
                (idx(linear_memory.offset_is[3]), -f_u64(4)),
            ],
        );

        b.push_linear_zero(
            [
                (idx(linear_memory.full_width_offset_is[0]), F::ONE),
                (idx(linear_memory.full_width_offset_is[1]), F::ONE),
                (idx(linear_memory.full_width_offset_is[2]), F::ONE),
                (idx(linear_memory.full_width_offset_is[3]), F::ONE),
                (idx(linear_memory.is_full_width), -F::ONE),
            ]
            .into_iter(),
        );
        push_gated_linear_zero(
            b,
            idx(linear_memory.is_full_width),
            [
                (idx(linear_memory.full_width_offset_is[0]), F::ONE),
                (idx(linear_memory.full_width_offset_is[1]), f_u64(2)),
                (idx(linear_memory.full_width_offset_is[2]), f_u64(3)),
                (idx(linear_memory.full_width_offset_is[3]), f_u64(4)),
                (idx(linear_memory.offset_is[0]), -F::ONE),
                (idx(linear_memory.offset_is[1]), -f_u64(2)),
                (idx(linear_memory.offset_is[2]), -f_u64(3)),
                (idx(linear_memory.offset_is[3]), -f_u64(4)),
            ],
        );
        b.push_linear_zero(
            [
                (idx(linear_memory.double_width_offset_is[0]), F::ONE),
                (idx(linear_memory.double_width_offset_is[1]), F::ONE),
                (idx(linear_memory.double_width_offset_is[2]), F::ONE),
                (idx(linear_memory.double_width_offset_is[3]), F::ONE),
                (idx(linear_memory.is_double_width), -F::ONE),
            ]
            .into_iter(),
        );
        push_gated_linear_zero(
            b,
            idx(linear_memory.is_double_width),
            [
                (idx(linear_memory.double_width_offset_is[0]), F::ONE),
                (idx(linear_memory.double_width_offset_is[1]), f_u64(2)),
                (idx(linear_memory.double_width_offset_is[2]), f_u64(3)),
                (idx(linear_memory.double_width_offset_is[3]), f_u64(4)),
                (idx(linear_memory.offset_is[0]), -F::ONE),
                (idx(linear_memory.offset_is[1]), -f_u64(2)),
                (idx(linear_memory.offset_is[2]), -f_u64(3)),
                (idx(linear_memory.offset_is[3]), -f_u64(4)),
            ],
        );
        b.push_linear_zero(
            [
                (idx(linear_memory.i64_load_offset_is[0]), F::ONE),
                (idx(linear_memory.i64_load_offset_is[1]), F::ONE),
                (idx(linear_memory.i64_load_offset_is[2]), F::ONE),
                (idx(linear_memory.i64_load_offset_is[3]), F::ONE),
                (i64_load_selector, -F::ONE),
            ]
            .into_iter(),
        );
        push_gated_linear_zero(
            b,
            i64_load_selector,
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
                (i64_store_selector, -F::ONE),
            ]
            .into_iter(),
        );
        push_gated_linear_zero(
            b,
            i64_store_selector,
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

    b.with_tag(shared("linear memory lane usage", LINEAR_MEMORY_OPS), |b| {
        b.push_row(
            [load16_selector, store16_selector]
                .into_iter()
                .map(|selector| (selector, F::ONE)),
            [
                (idx(linear_memory.use_lane1), F::ONE),
                (idx(linear_memory.half_width_offset_is[3]), -F::ONE),
            ],
            [],
        );
        b.push_row(
            [load_selector, store_selector]
                .into_iter()
                .map(|selector| (selector, F::ONE)),
            [
                (idx(linear_memory.use_lane1), F::ONE),
                (idx(linear_memory.full_width_offset_is[1]), -F::ONE),
                (idx(linear_memory.full_width_offset_is[2]), -F::ONE),
                (idx(linear_memory.full_width_offset_is[3]), -F::ONE),
            ],
            [],
        );
        b.push_row(
            [i64_load_selector, i64_store_selector]
                .into_iter()
                .map(|selector| (selector, F::ONE)),
            [(idx(linear_memory.is_double_width), F::ONE), (COL_ONE, -F::ONE)],
            [],
        );
        b.push_row(
            [i64_load_selector, i64_store_selector]
                .into_iter()
                .map(|selector| (selector, F::ONE)),
            [(idx(linear_memory.use_lane1), F::ONE), (COL_ONE, -F::ONE)],
            [],
        );
        b.push_row(
            [i64_load_selector, i64_store_selector]
                .into_iter()
                .map(|selector| (selector, F::ONE)),
            [
                (idx(linear_memory.use_lane2), F::ONE),
                (idx(linear_memory.double_width_offset_is[1]), -F::ONE),
                (idx(linear_memory.double_width_offset_is[2]), -F::ONE),
                (idx(linear_memory.double_width_offset_is[3]), -F::ONE),
            ],
            [],
        );
    });

    b.with_tag(shared("linear memory lane adjacency", LINEAR_MEMORY_OPS), |b| {
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

    // Bind `access_bytes_lo` to the active value's low 32 bits and
    // `access_bytes_hi` to the high 32 bits (i64 only). `access_bytes`
    // is direction-agnostic at the byte-shuffle layer (gated by
    // offset, not opcode), so the direction-specific binding lives
    // here: write0 on loads, read1 on stores; the i64 ops additionally
    // bind the hi half.
    let linear_memory_load_access_byte_ops = [
        WasmOpcode::I32Load,
        WasmOpcode::I32Load8S,
        WasmOpcode::I32Load8U,
        WasmOpcode::I32Load16S,
        WasmOpcode::I32Load16U,
        WasmOpcode::I64Load,
    ];
    b.with_tag(
        shared(
            "linear memory access bytes lo (loads)",
            &linear_memory_load_access_byte_ops,
        ),
        |b| {
            push_u32_le_bytes_decomp(
                b,
                linear_memory_load_access_byte_ops
                    .into_iter()
                    .map(|op| selector_col(op).expect("linear memory load access bytes selector")),
                idx(stack.write0_value),
                linear_memory.access_bytes_lo.map(idx),
            );
        },
    );
    let linear_memory_store_access_byte_ops = [
        WasmOpcode::I32Store,
        WasmOpcode::I32Store8,
        WasmOpcode::I32Store16,
        WasmOpcode::I64Store,
    ];
    b.with_tag(
        shared(
            "linear memory access bytes lo (stores)",
            &linear_memory_store_access_byte_ops,
        ),
        |b| {
            push_u32_le_bytes_decomp(
                b,
                linear_memory_store_access_byte_ops
                    .into_iter()
                    .map(|op| selector_col(op).expect("linear memory store access bytes selector")),
                idx(stack.read1_value),
                linear_memory.access_bytes_lo.map(idx),
            );
        },
    );
    b.with_tag(
        shared(
            "linear memory access bytes hi (i64)",
            &[WasmOpcode::I64Load, WasmOpcode::I64Store],
        ),
        |b| {
            push_u32_le_bytes_decomp(
                b,
                [i64_load_selector],
                idx(stack.write0_value_hi),
                linear_memory.access_bytes_hi.map(idx),
            );
            push_u32_le_bytes_decomp(
                b,
                [i64_store_selector],
                idx(stack.read1_value_hi),
                linear_memory.access_bytes_hi.map(idx),
            );
        },
    );

    b.with_tag(
        opcode_tag("linear memory load32 byte routing", WasmOpcode::I32Load),
        |b| {
            push_linear_memory_load32_byte_selection(b, linear_memory);
        },
    );
    b.with_tag(
        opcode_tag("linear memory load8_s routing", WasmOpcode::I32Load8S),
        |b| {
            push_linear_memory_load8_s_constraints(b, linear_memory);
        },
    );
    b.with_tag(
        opcode_tag("linear memory load8_u routing", WasmOpcode::I32Load8U),
        |b| {
            push_linear_memory_load8_u_constraints(b, linear_memory);
        },
    );
    b.with_tag(
        opcode_tag("linear memory load16_s routing", WasmOpcode::I32Load16S),
        |b| {
            push_linear_memory_load16_s_constraints(b, linear_memory);
        },
    );
    b.with_tag(
        opcode_tag("linear memory load16_u routing", WasmOpcode::I32Load16U),
        |b| {
            push_linear_memory_load16_u_constraints(b, linear_memory);
        },
    );
    b.with_tag(
        opcode_tag("linear memory store32 byte routing", WasmOpcode::I32Store),
        |b| {
            push_linear_memory_store32_byte_selection(b, linear_memory);
        },
    );
    b.with_tag(opcode_tag("linear memory store8 routing", WasmOpcode::I32Store8), |b| {
        push_linear_memory_store8_constraints(b, linear_memory);
    });
    b.with_tag(
        opcode_tag("linear memory store16 routing", WasmOpcode::I32Store16),
        |b| {
            push_linear_memory_store16_constraints(b, linear_memory);
        },
    );
    b.with_tag(opcode_tag("linear memory load64 routing", WasmOpcode::I64Load), |b| {
        push_linear_memory_load64_constraints(b, stack, linear_memory);
    });
    b.with_tag(opcode_tag("linear memory store64 routing", WasmOpcode::I64Store), |b| {
        push_linear_memory_store64_constraints(b, stack, linear_memory);
    });
}

fn push_linear_memory_load32_byte_selection(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
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
        push_matching_byte_constraints(b, selector, linear_memory.access_bytes_lo, lane_bytes);
    }
}

fn push_linear_memory_store32_byte_selection(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    for (selector, access_bytes, lane_bytes) in [
        (
            idx(linear_memory.full_width_offset_is[0]),
            linear_memory.access_bytes_lo,
            linear_memory.lane0_bytes,
        ),
        (
            idx(linear_memory.full_width_offset_is[1]),
            linear_memory.access_bytes_lo,
            [
                linear_memory.lane0_bytes[1],
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
            ],
        ),
        (
            idx(linear_memory.full_width_offset_is[2]),
            linear_memory.access_bytes_lo,
            [
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
            ],
        ),
        (
            idx(linear_memory.full_width_offset_is[3]),
            linear_memory.access_bytes_lo,
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

fn push_linear_memory_load8_u_constraints(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    push_linear_memory_load_subword_constraints(b, selector_col(WasmOpcode::I32Load8U).unwrap(), 1, linear_memory);
}

fn push_linear_memory_load8_s_constraints(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    push_linear_memory_load_signed_subword_constraints(
        b,
        selector_col(WasmOpcode::I32Load8S).unwrap(),
        1,
        idx(linear_memory.access_bytes_lo[0]),
        linear_memory,
    );
}

fn push_linear_memory_store8_constraints(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    push_linear_memory_store_subword_constraints(b, selector_col(WasmOpcode::I32Store8).unwrap(), 1, linear_memory);
}

fn push_linear_memory_load16_s_constraints(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    push_linear_memory_load_signed_subword_constraints(
        b,
        selector_col(WasmOpcode::I32Load16S).unwrap(),
        2,
        idx(linear_memory.access_bytes_lo[1]),
        linear_memory,
    );
}

fn push_linear_memory_load16_u_constraints(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    push_linear_memory_load_subword_constraints(b, selector_col(WasmOpcode::I32Load16U).unwrap(), 2, linear_memory);
}

fn push_linear_memory_load64_constraints(
    b: &mut R1csBuilder,
    stack: &OperandStackColumns,
    linear_memory: &LinearMemoryColumns,
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
    // lanes; match `access_bytes_lo` / `access_bytes_hi` against the
    // shuffled lane bytes. The byte-decomp above binds those to
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
        for (byte, lane_byte) in linear_memory.access_bytes_lo.into_iter().zip(low_bytes) {
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
        for (byte, target_byte) in linear_memory
            .access_bytes_lo
            .into_iter()
            .chain(linear_memory.access_bytes_hi)
            .zip(target_bytes)
        {
            push_gated_linear_zero(b, case_selector, [(idx(byte), F::ONE), (idx(target_byte), -F::ONE)]);
        }
    }
}

fn push_linear_memory_store16_constraints(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    push_linear_memory_store_subword_constraints(b, selector_col(WasmOpcode::I32Store16).unwrap(), 2, linear_memory);
}

fn push_linear_memory_load_subword_constraints(
    b: &mut R1csBuilder,
    selector: usize,
    width_bytes: usize,
    linear_memory: &LinearMemoryColumns,
) {
    for byte in &linear_memory.access_bytes_lo[width_bytes..] {
        push_gated_linear_zero(b, selector, [(idx(*byte), F::ONE)]);
    }
    push_linear_memory_subword_byte_constraints(b, selector, width_bytes, linear_memory);
}

fn push_linear_memory_load_signed_subword_constraints(
    b: &mut R1csBuilder,
    selector: usize,
    width_bytes: usize,
    sign_source_byte: usize,
    linear_memory: &LinearMemoryColumns,
) {
    push_linear_memory_subword_byte_constraints(b, selector, width_bytes, linear_memory);
    push_gated_linear_zero(
        b,
        selector,
        [
            (sign_source_byte, F::ONE),
            (idx(linear_memory.sign_ext_low7), -F::ONE),
            (idx(linear_memory.sign_ext_bit), -f_u64(128)),
        ],
    );
    for byte in &linear_memory.access_bytes_lo[width_bytes..] {
        push_gated_linear_zero(
            b,
            selector,
            [(idx(*byte), F::ONE), (idx(linear_memory.sign_ext_bit), -f_u64(255))],
        );
    }
}

fn push_linear_memory_store_subword_constraints(
    b: &mut R1csBuilder,
    selector: usize,
    width_bytes: usize,
    linear_memory: &LinearMemoryColumns,
) {
    push_linear_memory_subword_byte_constraints(b, selector, width_bytes, linear_memory);
}

fn push_linear_memory_subword_byte_constraints(
    b: &mut R1csBuilder,
    selector: usize,
    width_bytes: usize,
    linear_memory: &LinearMemoryColumns,
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
        for (access_byte, lane_byte) in linear_memory.access_bytes_lo[..width_bytes]
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
