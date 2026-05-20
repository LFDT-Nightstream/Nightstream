//! IMPORTANT: this doesn't prove anything, it's just a sanity checker for now

use super::isa::WasmShoutOpcode;
use super::layout::{ColumnWidth, COLUMN_SPECS};
use super::lookup_binding_builder::{WasmLookupBindingLayout, WasmLookupFamilyKind, WasmLookupFamilySpec};
use neo_math::F;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeField64;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupSemantics {
    pub predicate: LookupPredicate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LookupPredicate {
    Eq(LookupExpr, LookupExpr),
    And(Vec<LookupPredicate>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LookupExpr {
    Const(u64),
    Slot(usize),
    Apply(LookupBuiltin, Vec<LookupExpr>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookupBuiltin {
    LinearMemoryInBounds,
    ComposeU64,
    Low32,
    High32,
    I32Clz,
    I32Ctz,
    I32Eqz,
    I64Eqz,
    I32Eq,
    I32Ne,
    I32LtS,
    I32LtU,
    I32GtS,
    I32GtU,
    I32LeS,
    I32LeU,
    I32GeS,
    I32GeU,
    I32And,
    I32Or,
    I32Xor,
    I32Mul,
    I64And,
    I64Or,
    I64Xor,
    I64Mul,
    I32Shl,
    I32ShrU,
    I32ShrS,
    I32Rotl,
    I32Rotr,
    I32DivU,
    I32DivS,
    I32RemU,
    I32RemS,
}

pub fn semantics_for_lookup_family(family: &WasmLookupFamilySpec) -> LookupSemantics {
    match family.kind {
        WasmLookupFamilyKind::Shout(op) => shout_semantics(op),
        WasmLookupFamilyKind::LinearMemoryBounds => LookupSemantics {
            predicate: LookupPredicate::Eq(
                apply(
                    LookupBuiltin::LinearMemoryInBounds,
                    vec![slot(0), slot(1), slot(2), slot(3)],
                ),
                LookupExpr::Const(1),
            ),
        },
    }
}

pub fn append_lookup_semantics_digest(tr: &mut Poseidon2Transcript, semantics: &LookupSemantics) {
    append_predicate_digest(tr, &semantics.predicate);
}

pub fn sanity_check_lookup_row(layout: &WasmLookupBindingLayout, witness: &[F]) -> Result<(), String> {
    if witness.len() != layout.witness_width {
        return Err(format!(
            "lookup sanity check expected witness width {}, got {}",
            layout.witness_width,
            witness.len()
        ));
    }
    // Per-row check that every byte column carries a value in [0, 256). This
    // replaces the old `byte_u8` lookup family — `ColumnWidth::Byte` on the
    // column spec is now the single source of truth for byte-shaped columns.
    for spec in COLUMN_SPECS.iter().filter(|s| s.width == ColumnWidth::Byte) {
        let value = witness[spec.index].as_canonical_u64();
        if value > 0xff {
            return Err(format!(
                "column `{}` declared ColumnWidth::Byte but witness value is {value} (> 255)",
                spec.name
            ));
        }
    }
    let families = layout
        .lookup_families
        .iter()
        .map(|family| (family.name, family))
        .collect::<BTreeMap<_, _>>();
    for binding in &layout.lookup_bindings {
        let family = families
            .get(binding.family)
            .ok_or_else(|| format!("lookup sanity check missing family `{}`", binding.family))?;
        let slot_values = binding
            .columns
            .iter()
            .map(|column| witness[column.0].as_canonical_u64())
            .collect::<Vec<_>>();
        if !binding_is_active(binding, family, witness, &slot_values) {
            continue;
        }
        let semantics = &family.semantics;
        evaluate_predicate(&semantics.predicate, &slot_values).map_err(|err| {
            format!(
                "lookup sanity check failed for binding `{}` in family `{}`: {err}",
                binding.name, binding.family
            )
        })?;
    }
    Ok(())
}

fn binding_is_active(
    binding: &super::lookup_binding_builder::WasmLookupBindingSpec,
    family: &WasmLookupFamilySpec,
    witness: &[F],
    slot_values: &[u64],
) -> bool {
    let gate_active = binding
        .gate
        .map(|gate| witness[gate.0].as_canonical_u64() != 0)
        .unwrap_or(true);
    if !gate_active {
        return false;
    }
    match family.kind {
        WasmLookupFamilyKind::Shout(op) => slot_values.first().copied() == Some(op.to_shout_id() as u64),
        _ => true,
    }
}

fn shout_semantics(op: WasmShoutOpcode) -> LookupSemantics {
    let predicate = match op {
        WasmShoutOpcode::I32Clz => predicate_with_i32_unary(op, LookupBuiltin::I32Clz),
        WasmShoutOpcode::I32Ctz => predicate_with_i32_unary(op, LookupBuiltin::I32Ctz),
        WasmShoutOpcode::I32Eqz => predicate_with_i32_unary(op, LookupBuiltin::I32Eqz),
        WasmShoutOpcode::I64Eqz => LookupPredicate::And(vec![
            LookupPredicate::Eq(slot(0), LookupExpr::Const(op.to_shout_id() as u64)),
            LookupPredicate::Eq(
                slot(3),
                apply(LookupBuiltin::I64Eqz, vec![compose_u64(slot(1), slot(2))]),
            ),
        ]),
        WasmShoutOpcode::I32Eq => predicate_with_i32_binary(op, LookupBuiltin::I32Eq),
        WasmShoutOpcode::I32Ne => predicate_with_i32_binary(op, LookupBuiltin::I32Ne),
        WasmShoutOpcode::I32LtS => predicate_with_i32_binary(op, LookupBuiltin::I32LtS),
        WasmShoutOpcode::I32LtU => predicate_with_i32_binary(op, LookupBuiltin::I32LtU),
        WasmShoutOpcode::I32GtS => predicate_with_i32_binary(op, LookupBuiltin::I32GtS),
        WasmShoutOpcode::I32GtU => predicate_with_i32_binary(op, LookupBuiltin::I32GtU),
        WasmShoutOpcode::I32LeS => predicate_with_i32_binary(op, LookupBuiltin::I32LeS),
        WasmShoutOpcode::I32LeU => predicate_with_i32_binary(op, LookupBuiltin::I32LeU),
        WasmShoutOpcode::I32GeS => predicate_with_i32_binary(op, LookupBuiltin::I32GeS),
        WasmShoutOpcode::I32GeU => predicate_with_i32_binary(op, LookupBuiltin::I32GeU),
        WasmShoutOpcode::I32And => predicate_with_i32_binary(op, LookupBuiltin::I32And),
        WasmShoutOpcode::I32Or => predicate_with_i32_binary(op, LookupBuiltin::I32Or),
        WasmShoutOpcode::I32Xor => predicate_with_i32_binary(op, LookupBuiltin::I32Xor),
        WasmShoutOpcode::I32Mul => predicate_with_i32_binary(op, LookupBuiltin::I32Mul),
        WasmShoutOpcode::I64And => predicate_with_i64_binary(op, LookupBuiltin::I64And),
        WasmShoutOpcode::I64Or => predicate_with_i64_binary(op, LookupBuiltin::I64Or),
        WasmShoutOpcode::I64Xor => predicate_with_i64_binary(op, LookupBuiltin::I64Xor),
        WasmShoutOpcode::I64Mul => predicate_with_i64_binary(op, LookupBuiltin::I64Mul),
        WasmShoutOpcode::I32Shl => predicate_with_i32_binary(op, LookupBuiltin::I32Shl),
        WasmShoutOpcode::I32ShrU => predicate_with_i32_binary(op, LookupBuiltin::I32ShrU),
        WasmShoutOpcode::I32ShrS => predicate_with_i32_binary(op, LookupBuiltin::I32ShrS),
        WasmShoutOpcode::I32Rotl => predicate_with_i32_binary(op, LookupBuiltin::I32Rotl),
        WasmShoutOpcode::I32Rotr => predicate_with_i32_binary(op, LookupBuiltin::I32Rotr),
        WasmShoutOpcode::I32DivU => predicate_with_i32_binary(op, LookupBuiltin::I32DivU),
        WasmShoutOpcode::I32DivS => predicate_with_i32_binary(op, LookupBuiltin::I32DivS),
        WasmShoutOpcode::I32RemU => predicate_with_i32_binary(op, LookupBuiltin::I32RemU),
        WasmShoutOpcode::I32RemS => predicate_with_i32_binary(op, LookupBuiltin::I32RemS),
    };
    LookupSemantics { predicate }
}

fn predicate_with_i32_unary(op: WasmShoutOpcode, builtin: LookupBuiltin) -> LookupPredicate {
    LookupPredicate::And(vec![
        LookupPredicate::Eq(slot(0), LookupExpr::Const(op.to_shout_id() as u64)),
        LookupPredicate::Eq(slot(3), apply(builtin, vec![slot(1)])),
    ])
}

fn predicate_with_i32_binary(op: WasmShoutOpcode, builtin: LookupBuiltin) -> LookupPredicate {
    LookupPredicate::And(vec![
        LookupPredicate::Eq(slot(0), LookupExpr::Const(op.to_shout_id() as u64)),
        LookupPredicate::Eq(slot(3), apply(builtin, vec![slot(1), slot(2)])),
    ])
}

fn predicate_with_i64_binary(op: WasmShoutOpcode, builtin: LookupBuiltin) -> LookupPredicate {
    let result = apply(
        builtin,
        vec![compose_u64(slot(1), slot(2)), compose_u64(slot(3), slot(4))],
    );
    LookupPredicate::And(vec![
        LookupPredicate::Eq(slot(0), LookupExpr::Const(op.to_shout_id() as u64)),
        LookupPredicate::Eq(slot(5), apply(LookupBuiltin::Low32, vec![result.clone()])),
        LookupPredicate::Eq(slot(6), apply(LookupBuiltin::High32, vec![result])),
    ])
}

fn slot(index: usize) -> LookupExpr {
    LookupExpr::Slot(index)
}

fn apply(builtin: LookupBuiltin, args: Vec<LookupExpr>) -> LookupExpr {
    LookupExpr::Apply(builtin, args)
}

fn compose_u64(lo: LookupExpr, hi: LookupExpr) -> LookupExpr {
    apply(LookupBuiltin::ComposeU64, vec![lo, hi])
}

fn evaluate_predicate(predicate: &LookupPredicate, slots: &[u64]) -> Result<(), String> {
    match predicate {
        LookupPredicate::Eq(lhs, rhs) => {
            let lhs_value = evaluate_expr(lhs, slots)?;
            let rhs_value = evaluate_expr(rhs, slots)?;
            if lhs_value == rhs_value {
                Ok(())
            } else {
                Err(format!("expected equality, got {lhs_value} != {rhs_value}"))
            }
        }
        LookupPredicate::And(predicates) => {
            for predicate in predicates {
                evaluate_predicate(predicate, slots)?;
            }
            Ok(())
        }
    }
}

fn evaluate_expr(expr: &LookupExpr, slots: &[u64]) -> Result<u64, String> {
    match expr {
        LookupExpr::Const(value) => Ok(*value),
        LookupExpr::Slot(index) => slots
            .get(*index)
            .copied()
            .ok_or_else(|| format!("missing slot {index}")),
        LookupExpr::Apply(builtin, args) => {
            let values = args
                .iter()
                .map(|arg| evaluate_expr(arg, slots))
                .collect::<Result<Vec<_>, _>>()?;
            evaluate_builtin(*builtin, &values)
        }
    }
}

fn evaluate_builtin(builtin: LookupBuiltin, values: &[u64]) -> Result<u64, String> {
    Ok(match builtin {
        LookupBuiltin::LinearMemoryInBounds => {
            require_arity(builtin, values, 4)?;
            let pages_before = values[0];
            let lane0_addr = values[1];
            let use_lane1 = values[2] != 0;
            let use_lane2 = values[3] != 0;
            let touched_words = if use_lane2 {
                3
            } else if use_lane1 {
                2
            } else {
                1
            };
            let words_per_page = 65_536_u64 / 4_u64;
            let total_words = pages_before.saturating_mul(words_per_page);
            u64::from(
                lane0_addr
                    .checked_add(touched_words - 1)
                    .map(|last_word| last_word < total_words)
                    .unwrap_or(false),
            )
        }
        LookupBuiltin::ComposeU64 => {
            require_arity(builtin, values, 2)?;
            trunc_u32(values[0]) as u64 | ((trunc_u32(values[1]) as u64) << 32)
        }
        LookupBuiltin::Low32 => {
            require_arity(builtin, values, 1)?;
            trunc_u32(values[0]) as u64
        }
        LookupBuiltin::High32 => {
            require_arity(builtin, values, 1)?;
            (values[0] >> 32) & 0xffff_ffff
        }
        LookupBuiltin::I32Clz => {
            require_arity(builtin, values, 1)?;
            trunc_u32(values[0]).leading_zeros() as u64
        }
        LookupBuiltin::I32Ctz => {
            require_arity(builtin, values, 1)?;
            trunc_u32(values[0]).trailing_zeros() as u64
        }
        LookupBuiltin::I32Eqz => {
            require_arity(builtin, values, 1)?;
            u64::from(trunc_u32(values[0]) == 0)
        }
        LookupBuiltin::I64Eqz => {
            require_arity(builtin, values, 1)?;
            u64::from(values[0] == 0)
        }
        LookupBuiltin::I32Eq => compare_u32(values, |lhs, rhs| lhs == rhs)?,
        LookupBuiltin::I32Ne => compare_u32(values, |lhs, rhs| lhs != rhs)?,
        LookupBuiltin::I32LtS => compare_i32(values, |lhs, rhs| lhs < rhs)?,
        LookupBuiltin::I32LtU => compare_u32(values, |lhs, rhs| lhs < rhs)?,
        LookupBuiltin::I32GtS => compare_i32(values, |lhs, rhs| lhs > rhs)?,
        LookupBuiltin::I32GtU => compare_u32(values, |lhs, rhs| lhs > rhs)?,
        LookupBuiltin::I32LeS => compare_i32(values, |lhs, rhs| lhs <= rhs)?,
        LookupBuiltin::I32LeU => compare_u32(values, |lhs, rhs| lhs <= rhs)?,
        LookupBuiltin::I32GeS => compare_i32(values, |lhs, rhs| lhs >= rhs)?,
        LookupBuiltin::I32GeU => compare_u32(values, |lhs, rhs| lhs >= rhs)?,
        LookupBuiltin::I32And => binary_u32(values, |lhs, rhs| lhs & rhs)?,
        LookupBuiltin::I32Or => binary_u32(values, |lhs, rhs| lhs | rhs)?,
        LookupBuiltin::I32Xor => binary_u32(values, |lhs, rhs| lhs ^ rhs)?,
        LookupBuiltin::I32Mul => binary_u32(values, |lhs, rhs| lhs.wrapping_mul(rhs))?,
        LookupBuiltin::I64And => binary_u64(values, |lhs, rhs| lhs & rhs)?,
        LookupBuiltin::I64Or => binary_u64(values, |lhs, rhs| lhs | rhs)?,
        LookupBuiltin::I64Xor => binary_u64(values, |lhs, rhs| lhs ^ rhs)?,
        LookupBuiltin::I64Mul => binary_u64(values, |lhs, rhs| lhs.wrapping_mul(rhs))?,
        LookupBuiltin::I32Shl => binary_u32(values, |lhs, rhs| lhs.wrapping_shl(rhs & 31))?,
        LookupBuiltin::I32ShrU => binary_u32(values, |lhs, rhs| lhs.wrapping_shr(rhs & 31))?,
        LookupBuiltin::I32ShrS => {
            require_arity(builtin, values, 2)?;
            i32v(values[0]).wrapping_shr(trunc_u32(values[1]) & 31) as u32 as u64
        }
        LookupBuiltin::I32Rotl => binary_u32(values, |lhs, rhs| lhs.rotate_left(rhs & 31))?,
        LookupBuiltin::I32Rotr => binary_u32(values, |lhs, rhs| lhs.rotate_right(rhs & 31))?,
        LookupBuiltin::I32DivU => {
            require_arity(builtin, values, 2)?;
            let rhs = trunc_u32(values[1]);
            if rhs == 0 {
                return Err("i32.div_u with divisor 0".to_string());
            }
            (trunc_u32(values[0]) / rhs) as u64
        }
        LookupBuiltin::I32DivS => {
            require_arity(builtin, values, 2)?;
            let lhs = i32v(values[0]);
            let rhs = i32v(values[1]);
            if rhs == 0 {
                return Err("i32.div_s with divisor 0".to_string());
            }
            if lhs == i32::MIN && rhs == -1 {
                return Err("i32.div_s overflow".to_string());
            }
            (lhs / rhs) as u32 as u64
        }
        LookupBuiltin::I32RemU => {
            require_arity(builtin, values, 2)?;
            let rhs = trunc_u32(values[1]);
            if rhs == 0 {
                return Err("i32.rem_u with divisor 0".to_string());
            }
            (trunc_u32(values[0]) % rhs) as u64
        }
        LookupBuiltin::I32RemS => {
            require_arity(builtin, values, 2)?;
            let lhs = i32v(values[0]);
            let rhs = i32v(values[1]);
            if rhs == 0 {
                return Err("i32.rem_s with divisor 0".to_string());
            }
            if lhs == i32::MIN && rhs == -1 {
                0
            } else {
                (lhs % rhs) as u32 as u64
            }
        }
    })
}

fn compare_u32(values: &[u64], f: impl FnOnce(u32, u32) -> bool) -> Result<u64, String> {
    require_arity_name(values, 2, "u32 compare")?;
    Ok(u64::from(f(trunc_u32(values[0]), trunc_u32(values[1]))))
}

fn compare_i32(values: &[u64], f: impl FnOnce(i32, i32) -> bool) -> Result<u64, String> {
    require_arity_name(values, 2, "i32 compare")?;
    Ok(u64::from(f(i32v(values[0]), i32v(values[1]))))
}

fn binary_u32(values: &[u64], f: impl FnOnce(u32, u32) -> u32) -> Result<u64, String> {
    require_arity_name(values, 2, "u32 binary op")?;
    Ok(f(trunc_u32(values[0]), trunc_u32(values[1])) as u64)
}

fn binary_u64(values: &[u64], f: impl FnOnce(u64, u64) -> u64) -> Result<u64, String> {
    require_arity_name(values, 2, "u64 binary op")?;
    Ok(f(values[0], values[1]))
}

fn require_arity(builtin: LookupBuiltin, values: &[u64], expected: usize) -> Result<(), String> {
    require_arity_name(values, expected, builtin.name())
}

fn require_arity_name(values: &[u64], expected: usize, name: &str) -> Result<(), String> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(format!("{name} expected {expected} args, got {}", values.len()))
    }
}

fn trunc_u32(value: u64) -> u32 {
    value as u32
}

fn i32v(value: u64) -> i32 {
    value as u32 as i32
}

impl LookupBuiltin {
    fn name(self) -> &'static str {
        match self {
            LookupBuiltin::LinearMemoryInBounds => "linear_memory_in_bounds",
            LookupBuiltin::ComposeU64 => "compose_u64",
            LookupBuiltin::Low32 => "low32",
            LookupBuiltin::High32 => "high32",
            LookupBuiltin::I32Clz => "i32_clz",
            LookupBuiltin::I32Ctz => "i32_ctz",
            LookupBuiltin::I32Eqz => "i32_eqz",
            LookupBuiltin::I64Eqz => "i64_eqz",
            LookupBuiltin::I32Eq => "i32_eq",
            LookupBuiltin::I32Ne => "i32_ne",
            LookupBuiltin::I32LtS => "i32_lt_s",
            LookupBuiltin::I32LtU => "i32_lt_u",
            LookupBuiltin::I32GtS => "i32_gt_s",
            LookupBuiltin::I32GtU => "i32_gt_u",
            LookupBuiltin::I32LeS => "i32_le_s",
            LookupBuiltin::I32LeU => "i32_le_u",
            LookupBuiltin::I32GeS => "i32_ge_s",
            LookupBuiltin::I32GeU => "i32_ge_u",
            LookupBuiltin::I32And => "i32_and",
            LookupBuiltin::I32Or => "i32_or",
            LookupBuiltin::I32Xor => "i32_xor",
            LookupBuiltin::I32Mul => "i32_mul",
            LookupBuiltin::I64And => "i64_and",
            LookupBuiltin::I64Or => "i64_or",
            LookupBuiltin::I64Xor => "i64_xor",
            LookupBuiltin::I64Mul => "i64_mul",
            LookupBuiltin::I32Shl => "i32_shl",
            LookupBuiltin::I32ShrU => "i32_shr_u",
            LookupBuiltin::I32ShrS => "i32_shr_s",
            LookupBuiltin::I32Rotl => "i32_rotl",
            LookupBuiltin::I32Rotr => "i32_rotr",
            LookupBuiltin::I32DivU => "i32_div_u",
            LookupBuiltin::I32DivS => "i32_div_s",
            LookupBuiltin::I32RemU => "i32_rem_u",
            LookupBuiltin::I32RemS => "i32_rem_s",
        }
    }

    fn digest_id(self) -> u64 {
        match self {
            LookupBuiltin::LinearMemoryInBounds => 1,
            LookupBuiltin::ComposeU64 => 2,
            LookupBuiltin::Low32 => 3,
            LookupBuiltin::High32 => 4,
            LookupBuiltin::I32Clz => 5,
            LookupBuiltin::I32Ctz => 6,
            LookupBuiltin::I32Eqz => 7,
            LookupBuiltin::I64Eqz => 8,
            LookupBuiltin::I32Eq => 9,
            LookupBuiltin::I32Ne => 10,
            LookupBuiltin::I32LtS => 11,
            LookupBuiltin::I32LtU => 12,
            LookupBuiltin::I32GtS => 13,
            LookupBuiltin::I32GtU => 14,
            LookupBuiltin::I32LeS => 15,
            LookupBuiltin::I32LeU => 16,
            LookupBuiltin::I32GeS => 17,
            LookupBuiltin::I32GeU => 18,
            LookupBuiltin::I32And => 19,
            LookupBuiltin::I32Or => 20,
            LookupBuiltin::I32Xor => 21,
            LookupBuiltin::I32Mul => 22,
            LookupBuiltin::I64And => 23,
            LookupBuiltin::I64Or => 24,
            LookupBuiltin::I64Xor => 25,
            LookupBuiltin::I64Mul => 26,
            LookupBuiltin::I32Shl => 27,
            LookupBuiltin::I32ShrU => 28,
            LookupBuiltin::I32ShrS => 29,
            LookupBuiltin::I32Rotl => 30,
            LookupBuiltin::I32Rotr => 31,
            LookupBuiltin::I32DivU => 32,
            LookupBuiltin::I32DivS => 33,
            LookupBuiltin::I32RemU => 34,
            LookupBuiltin::I32RemS => 35,
        }
    }
}

fn append_predicate_digest(tr: &mut Poseidon2Transcript, predicate: &LookupPredicate) {
    match predicate {
        LookupPredicate::Eq(lhs, rhs) => {
            tr.append_u64s(b"wasm/lookup_semantics/predicate/tag", &[0]);
            append_expr_digest(tr, lhs);
            append_expr_digest(tr, rhs);
        }
        LookupPredicate::And(predicates) => {
            tr.append_u64s(b"wasm/lookup_semantics/predicate/tag", &[1, predicates.len() as u64]);
            for predicate in predicates {
                append_predicate_digest(tr, predicate);
            }
        }
    }
}

fn append_expr_digest(tr: &mut Poseidon2Transcript, expr: &LookupExpr) {
    match expr {
        LookupExpr::Const(value) => {
            tr.append_u64s(b"wasm/lookup_semantics/expr/tag", &[0, *value]);
        }
        LookupExpr::Slot(index) => {
            tr.append_u64s(b"wasm/lookup_semantics/expr/tag", &[1, *index as u64]);
        }
        LookupExpr::Apply(builtin, args) => {
            tr.append_u64s(
                b"wasm/lookup_semantics/expr/tag",
                &[2, builtin.digest_id(), args.len() as u64],
            );
            for arg in args {
                append_expr_digest(tr, arg);
            }
        }
    }
}
