//! Native operation evaluation for assignment and tests; constraints establish correctness.

use super::{WasmOperationBindingSpec, WasmOperationFamilySpec};
use crate::isa::{WasmOpcode, WasmOperation};
use crate::layout::ColumnWidth;
use crate::relation_layout::WasmAuxiliaryRelations;
use crate::witness_layout::declared_witness_column_families;
use neo_math::F;
use p3_field::PrimeField64;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationSemantics {
    pub predicate: OperationPredicate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OperationPredicate {
    Eq(OperationExpr, OperationExpr),
    And(Vec<OperationPredicate>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OperationExpr {
    Const(u64),
    Slot(usize),
    Apply(OperationBuiltin, Vec<OperationExpr>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperationBuiltin {
    ComposeU64,
    Low32,
    High32,
    I32Clz,
    I32Ctz,
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
    I32Popcnt,
    I64LtS,
    I64LtU,
    I64GtS,
    I64GtU,
    I64LeS,
    I64LeU,
    I64GeS,
    I64GeU,
    I64Shl,
    I64ShrS,
    I64ShrU,
    I64Rotl,
    I64Rotr,
    I64DivS,
    I64DivU,
    I64RemS,
    I64RemU,
    I64Clz,
    I64Ctz,
    I64Popcnt,
    I32RemU,
    I32RemS,
}

pub fn sanity_check_operation_row(auxiliary: &WasmAuxiliaryRelations, witness: &[F]) -> Result<(), String> {
    let expected = crate::RANGE_CHECKED_WITNESS_WIDTH;
    if witness.len() != expected {
        return Err(format!(
            "operation sanity check expected witness width {}, got {}",
            expected,
            witness.len()
        ));
    }
    // Per-row check that every byte column carries a value in [0, 256). This
    // replaces the old `byte_u8` lookup family — `ColumnWidth::Byte` on the
    // column family is now the single source of truth for byte-shaped columns.
    for family in declared_witness_column_families().filter(|family| family.width == ColumnWidth::Byte) {
        for column in family.start..family.end() {
            let value = witness[column].as_canonical_u64();
            if value > 0xff {
                return Err(format!(
                    "column `{}` declared ColumnWidth::Byte but witness value is {value} (> 255)",
                    family.name
                ));
            }
        }
    }
    let families = auxiliary
        .operation_families
        .iter()
        .map(|family| (family.name, family))
        .collect::<BTreeMap<_, _>>();
    for binding in &auxiliary.operation_bindings {
        let family = families
            .get(binding.family)
            .ok_or_else(|| format!("operation sanity check missing family `{}`", binding.family))?;
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
                "operation sanity check failed for binding `{}` in family `{}`: {err}",
                binding.name, binding.family
            )
        })?;
    }
    Ok(())
}

fn binding_is_active(
    binding: &WasmOperationBindingSpec,
    family: &WasmOperationFamilySpec,
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
    slot_values.first().copied() == Some(family.operation.operation_id() as u64)
}

pub fn semantics_for_operation(op: WasmOperation) -> OperationSemantics {
    let predicate = match op {
        WasmOperation::I32Clz => predicate_with_i32_unary(op, OperationBuiltin::I32Clz),
        WasmOperation::I32Ctz => predicate_with_i32_unary(op, OperationBuiltin::I32Ctz),
        WasmOperation::I32LtS => predicate_with_i32_binary(op, OperationBuiltin::I32LtS),
        WasmOperation::I32LtU => predicate_with_i32_binary(op, OperationBuiltin::I32LtU),
        WasmOperation::I32GtS => predicate_with_i32_binary(op, OperationBuiltin::I32GtS),
        WasmOperation::I32GtU => predicate_with_i32_binary(op, OperationBuiltin::I32GtU),
        WasmOperation::I32LeS => predicate_with_i32_binary(op, OperationBuiltin::I32LeS),
        WasmOperation::I32LeU => predicate_with_i32_binary(op, OperationBuiltin::I32LeU),
        WasmOperation::I32GeS => predicate_with_i32_binary(op, OperationBuiltin::I32GeS),
        WasmOperation::I32GeU => predicate_with_i32_binary(op, OperationBuiltin::I32GeU),
        WasmOperation::I32And => predicate_with_i32_binary(op, OperationBuiltin::I32And),
        WasmOperation::I32Or => predicate_with_i32_binary(op, OperationBuiltin::I32Or),
        WasmOperation::I32Xor => predicate_with_i32_binary(op, OperationBuiltin::I32Xor),
        WasmOperation::I32Mul => predicate_with_i32_binary(op, OperationBuiltin::I32Mul),
        WasmOperation::I64And => predicate_with_i64_binary(op, OperationBuiltin::I64And),
        WasmOperation::I64Or => predicate_with_i64_binary(op, OperationBuiltin::I64Or),
        WasmOperation::I64Xor => predicate_with_i64_binary(op, OperationBuiltin::I64Xor),
        WasmOperation::I64Mul => predicate_with_i64_binary(op, OperationBuiltin::I64Mul),
        WasmOperation::I32Shl => predicate_with_i32_binary(op, OperationBuiltin::I32Shl),
        WasmOperation::I32ShrU => predicate_with_i32_binary(op, OperationBuiltin::I32ShrU),
        WasmOperation::I32ShrS => predicate_with_i32_binary(op, OperationBuiltin::I32ShrS),
        WasmOperation::I32Rotl => predicate_with_i32_binary(op, OperationBuiltin::I32Rotl),
        WasmOperation::I32Rotr => predicate_with_i32_binary(op, OperationBuiltin::I32Rotr),
        WasmOperation::I32DivU => predicate_with_i32_binary(op, OperationBuiltin::I32DivU),
        WasmOperation::I32DivS => predicate_with_i32_binary(op, OperationBuiltin::I32DivS),
        WasmOperation::I32RemU => predicate_with_i32_binary(op, OperationBuiltin::I32RemU),
        WasmOperation::I32RemS => predicate_with_i32_binary(op, OperationBuiltin::I32RemS),
        WasmOperation::I32Popcnt => predicate_with_i32_unary(op, OperationBuiltin::I32Popcnt),
        WasmOperation::I64LtS => predicate_with_i64_binary(op, OperationBuiltin::I64LtS),
        WasmOperation::I64LtU => predicate_with_i64_binary(op, OperationBuiltin::I64LtU),
        WasmOperation::I64GtS => predicate_with_i64_binary(op, OperationBuiltin::I64GtS),
        WasmOperation::I64GtU => predicate_with_i64_binary(op, OperationBuiltin::I64GtU),
        WasmOperation::I64LeS => predicate_with_i64_binary(op, OperationBuiltin::I64LeS),
        WasmOperation::I64LeU => predicate_with_i64_binary(op, OperationBuiltin::I64LeU),
        WasmOperation::I64GeS => predicate_with_i64_binary(op, OperationBuiltin::I64GeS),
        WasmOperation::I64GeU => predicate_with_i64_binary(op, OperationBuiltin::I64GeU),
        WasmOperation::I64Shl => predicate_with_i64_binary(op, OperationBuiltin::I64Shl),
        WasmOperation::I64ShrS => predicate_with_i64_binary(op, OperationBuiltin::I64ShrS),
        WasmOperation::I64ShrU => predicate_with_i64_binary(op, OperationBuiltin::I64ShrU),
        WasmOperation::I64Rotl => predicate_with_i64_binary(op, OperationBuiltin::I64Rotl),
        WasmOperation::I64Rotr => predicate_with_i64_binary(op, OperationBuiltin::I64Rotr),
        WasmOperation::I64DivS => predicate_with_i64_binary(op, OperationBuiltin::I64DivS),
        WasmOperation::I64DivU => predicate_with_i64_binary(op, OperationBuiltin::I64DivU),
        WasmOperation::I64RemS => predicate_with_i64_binary(op, OperationBuiltin::I64RemS),
        WasmOperation::I64RemU => predicate_with_i64_binary(op, OperationBuiltin::I64RemU),
        WasmOperation::I64Clz => predicate_with_i64_unary(op, OperationBuiltin::I64Clz),
        WasmOperation::I64Ctz => predicate_with_i64_unary(op, OperationBuiltin::I64Ctz),
        WasmOperation::I64Popcnt => predicate_with_i64_unary(op, OperationBuiltin::I64Popcnt),
    };
    OperationSemantics { predicate }
}

fn predicate_with_i32_unary(op: WasmOperation, builtin: OperationBuiltin) -> OperationPredicate {
    OperationPredicate::And(vec![
        OperationPredicate::Eq(slot(0), OperationExpr::Const(op.operation_id() as u64)),
        OperationPredicate::Eq(slot(3), apply(builtin, vec![slot(1)])),
    ])
}

fn predicate_with_i32_binary(op: WasmOperation, builtin: OperationBuiltin) -> OperationPredicate {
    OperationPredicate::And(vec![
        OperationPredicate::Eq(slot(0), OperationExpr::Const(op.operation_id() as u64)),
        OperationPredicate::Eq(slot(3), apply(builtin, vec![slot(1), slot(2)])),
    ])
}

fn predicate_with_i64_binary(op: WasmOperation, builtin: OperationBuiltin) -> OperationPredicate {
    let result = apply(
        builtin,
        vec![compose_u64(slot(1), slot(2)), compose_u64(slot(3), slot(4))],
    );
    OperationPredicate::And(vec![
        OperationPredicate::Eq(slot(0), OperationExpr::Const(op.operation_id() as u64)),
        OperationPredicate::Eq(slot(5), apply(OperationBuiltin::Low32, vec![result.clone()])),
        OperationPredicate::Eq(slot(6), apply(OperationBuiltin::High32, vec![result])),
    ])
}

fn predicate_with_i64_unary(op: WasmOperation, builtin: OperationBuiltin) -> OperationPredicate {
    let result = apply(builtin, vec![compose_u64(slot(1), slot(2))]);
    OperationPredicate::And(vec![
        OperationPredicate::Eq(slot(0), OperationExpr::Const(op.operation_id() as u64)),
        OperationPredicate::Eq(slot(3), apply(OperationBuiltin::Low32, vec![result.clone()])),
        OperationPredicate::Eq(slot(4), apply(OperationBuiltin::High32, vec![result])),
    ])
}

fn slot(index: usize) -> OperationExpr {
    OperationExpr::Slot(index)
}

fn apply(builtin: OperationBuiltin, args: Vec<OperationExpr>) -> OperationExpr {
    OperationExpr::Apply(builtin, args)
}

fn compose_u64(lo: OperationExpr, hi: OperationExpr) -> OperationExpr {
    apply(OperationBuiltin::ComposeU64, vec![lo, hi])
}

fn evaluate_predicate(predicate: &OperationPredicate, slots: &[u64]) -> Result<(), String> {
    match predicate {
        OperationPredicate::Eq(lhs, rhs) => {
            let lhs_value = evaluate_expr(lhs, slots)?;
            let rhs_value = evaluate_expr(rhs, slots)?;
            if lhs_value == rhs_value {
                Ok(())
            } else {
                Err(format!("expected equality, got {lhs_value} != {rhs_value}"))
            }
        }
        OperationPredicate::And(predicates) => {
            for predicate in predicates {
                evaluate_predicate(predicate, slots)?;
            }
            Ok(())
        }
    }
}

fn evaluate_expr(expr: &OperationExpr, slots: &[u64]) -> Result<u64, String> {
    match expr {
        OperationExpr::Const(value) => Ok(*value),
        OperationExpr::Slot(index) => slots
            .get(*index)
            .copied()
            .ok_or_else(|| format!("missing slot {index}")),
        OperationExpr::Apply(builtin, args) => {
            let values = args
                .iter()
                .map(|arg| evaluate_expr(arg, slots))
                .collect::<Result<Vec<_>, _>>()?;
            evaluate_builtin(*builtin, &values)
        }
    }
}

fn evaluate_builtin(builtin: OperationBuiltin, values: &[u64]) -> Result<u64, String> {
    Ok(match builtin {
        OperationBuiltin::ComposeU64 => {
            require_arity(builtin, values, 2)?;
            trunc_u32(values[0]) as u64 | ((trunc_u32(values[1]) as u64) << 32)
        }
        OperationBuiltin::Low32 => {
            require_arity(builtin, values, 1)?;
            trunc_u32(values[0]) as u64
        }
        OperationBuiltin::High32 => {
            require_arity(builtin, values, 1)?;
            (values[0] >> 32) & 0xffff_ffff
        }
        OperationBuiltin::I32Clz => {
            require_arity(builtin, values, 1)?;
            trunc_u32(values[0]).leading_zeros() as u64
        }
        OperationBuiltin::I32Ctz => {
            require_arity(builtin, values, 1)?;
            trunc_u32(values[0]).trailing_zeros() as u64
        }
        OperationBuiltin::I32LtS => compare_i32(values, |lhs, rhs| lhs < rhs)?,
        OperationBuiltin::I32LtU => compare_u32(values, |lhs, rhs| lhs < rhs)?,
        OperationBuiltin::I32GtS => compare_i32(values, |lhs, rhs| lhs > rhs)?,
        OperationBuiltin::I32GtU => compare_u32(values, |lhs, rhs| lhs > rhs)?,
        OperationBuiltin::I32LeS => compare_i32(values, |lhs, rhs| lhs <= rhs)?,
        OperationBuiltin::I32LeU => compare_u32(values, |lhs, rhs| lhs <= rhs)?,
        OperationBuiltin::I32GeS => compare_i32(values, |lhs, rhs| lhs >= rhs)?,
        OperationBuiltin::I32GeU => compare_u32(values, |lhs, rhs| lhs >= rhs)?,
        OperationBuiltin::I32And => binary_u32(values, |lhs, rhs| lhs & rhs)?,
        OperationBuiltin::I32Or => binary_u32(values, |lhs, rhs| lhs | rhs)?,
        OperationBuiltin::I32Xor => binary_u32(values, |lhs, rhs| lhs ^ rhs)?,
        OperationBuiltin::I32Mul => binary_u32(values, |lhs, rhs| lhs.wrapping_mul(rhs))?,
        OperationBuiltin::I64And => binary_u64(values, |lhs, rhs| lhs & rhs)?,
        OperationBuiltin::I64Or => binary_u64(values, |lhs, rhs| lhs | rhs)?,
        OperationBuiltin::I64Xor => binary_u64(values, |lhs, rhs| lhs ^ rhs)?,
        OperationBuiltin::I64Mul => binary_u64(values, |lhs, rhs| lhs.wrapping_mul(rhs))?,
        OperationBuiltin::I32Shl => binary_u32(values, |lhs, rhs| lhs.wrapping_shl(rhs & 31))?,
        OperationBuiltin::I32ShrU => binary_u32(values, |lhs, rhs| lhs.wrapping_shr(rhs & 31))?,
        OperationBuiltin::I32ShrS => {
            require_arity(builtin, values, 2)?;
            i32v(values[0]).wrapping_shr(trunc_u32(values[1]) & 31) as u32 as u64
        }
        OperationBuiltin::I32Rotl => binary_u32(values, |lhs, rhs| lhs.rotate_left(rhs & 31))?,
        OperationBuiltin::I32Rotr => binary_u32(values, |lhs, rhs| lhs.rotate_right(rhs & 31))?,
        OperationBuiltin::I32DivU => {
            require_arity(builtin, values, 2)?;
            let rhs = trunc_u32(values[1]);
            if rhs == 0 {
                return Err("i32.div_u with divisor 0".to_string());
            }
            (trunc_u32(values[0]) / rhs) as u64
        }
        OperationBuiltin::I32DivS => {
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
        OperationBuiltin::I32RemU => {
            require_arity(builtin, values, 2)?;
            let rhs = trunc_u32(values[1]);
            if rhs == 0 {
                return Err("i32.rem_u with divisor 0".to_string());
            }
            (trunc_u32(values[0]) % rhs) as u64
        }
        OperationBuiltin::I32RemS => {
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
        OperationBuiltin::I32Popcnt => {
            require_arity(builtin, values, 1)?;
            u64::from(trunc_u32(values[0]).count_ones())
        }
        OperationBuiltin::I64LtS => compare_i64(values, |lhs, rhs| lhs < rhs)?,
        OperationBuiltin::I64LtU => compare_u64(values, |lhs, rhs| lhs < rhs)?,
        OperationBuiltin::I64GtS => compare_i64(values, |lhs, rhs| lhs > rhs)?,
        OperationBuiltin::I64GtU => compare_u64(values, |lhs, rhs| lhs > rhs)?,
        OperationBuiltin::I64LeS => compare_i64(values, |lhs, rhs| lhs <= rhs)?,
        OperationBuiltin::I64LeU => compare_u64(values, |lhs, rhs| lhs <= rhs)?,
        OperationBuiltin::I64GeS => compare_i64(values, |lhs, rhs| lhs >= rhs)?,
        OperationBuiltin::I64GeU => compare_u64(values, |lhs, rhs| lhs >= rhs)?,
        OperationBuiltin::I64Shl => binary_u64(values, |lhs, rhs| lhs.wrapping_shl(rhs as u32 & 63))?,
        OperationBuiltin::I64ShrU => binary_u64(values, |lhs, rhs| lhs.wrapping_shr(rhs as u32 & 63))?,
        OperationBuiltin::I64ShrS => {
            require_arity(builtin, values, 2)?;
            (values[0] as i64).wrapping_shr(values[1] as u32 & 63) as u64
        }
        OperationBuiltin::I64Rotl => binary_u64(values, |lhs, rhs| lhs.rotate_left(rhs as u32 & 63))?,
        OperationBuiltin::I64Rotr => binary_u64(values, |lhs, rhs| lhs.rotate_right(rhs as u32 & 63))?,
        OperationBuiltin::I64DivU => {
            require_arity(builtin, values, 2)?;
            if values[1] == 0 {
                return Err("i64.div_u with divisor 0".to_string());
            }
            values[0] / values[1]
        }
        OperationBuiltin::I64DivS => {
            require_arity(builtin, values, 2)?;
            let lhs = values[0] as i64;
            let rhs = values[1] as i64;
            if rhs == 0 {
                return Err("i64.div_s with divisor 0".to_string());
            }
            if lhs == i64::MIN && rhs == -1 {
                return Err("i64.div_s overflow".to_string());
            }
            (lhs / rhs) as u64
        }
        OperationBuiltin::I64RemU => {
            require_arity(builtin, values, 2)?;
            if values[1] == 0 {
                return Err("i64.rem_u with divisor 0".to_string());
            }
            values[0] % values[1]
        }
        OperationBuiltin::I64RemS => {
            require_arity(builtin, values, 2)?;
            let lhs = values[0] as i64;
            let rhs = values[1] as i64;
            if rhs == 0 {
                return Err("i64.rem_s with divisor 0".to_string());
            }
            if lhs == i64::MIN && rhs == -1 {
                0
            } else {
                (lhs % rhs) as u64
            }
        }
        OperationBuiltin::I64Clz => {
            require_arity(builtin, values, 1)?;
            u64::from(values[0].leading_zeros())
        }
        OperationBuiltin::I64Ctz => {
            require_arity(builtin, values, 1)?;
            u64::from(values[0].trailing_zeros())
        }
        OperationBuiltin::I64Popcnt => {
            require_arity(builtin, values, 1)?;
            u64::from(values[0].count_ones())
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

fn compare_u64(values: &[u64], f: impl FnOnce(u64, u64) -> bool) -> Result<u64, String> {
    require_arity_name(values, 2, "u64 compare")?;
    Ok(u64::from(f(values[0], values[1])))
}

fn compare_i64(values: &[u64], f: impl FnOnce(i64, i64) -> bool) -> Result<u64, String> {
    require_arity_name(values, 2, "i64 compare")?;
    Ok(u64::from(f(values[0] as i64, values[1] as i64)))
}

fn binary_u32(values: &[u64], f: impl FnOnce(u32, u32) -> u32) -> Result<u64, String> {
    require_arity_name(values, 2, "u32 binary op")?;
    Ok(f(trunc_u32(values[0]), trunc_u32(values[1])) as u64)
}

fn binary_u64(values: &[u64], f: impl FnOnce(u64, u64) -> u64) -> Result<u64, String> {
    require_arity_name(values, 2, "u64 binary op")?;
    Ok(f(values[0], values[1]))
}

fn require_arity(builtin: OperationBuiltin, values: &[u64], expected: usize) -> Result<(), String> {
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

impl OperationBuiltin {
    fn name(self) -> &'static str {
        match self {
            OperationBuiltin::ComposeU64 => "compose_u64",
            OperationBuiltin::Low32 => "low32",
            OperationBuiltin::High32 => "high32",
            OperationBuiltin::I32Clz => "i32_clz",
            OperationBuiltin::I32Ctz => "i32_ctz",
            OperationBuiltin::I32LtS => "i32_lt_s",
            OperationBuiltin::I32LtU => "i32_lt_u",
            OperationBuiltin::I32GtS => "i32_gt_s",
            OperationBuiltin::I32GtU => "i32_gt_u",
            OperationBuiltin::I32LeS => "i32_le_s",
            OperationBuiltin::I32LeU => "i32_le_u",
            OperationBuiltin::I32GeS => "i32_ge_s",
            OperationBuiltin::I32GeU => "i32_ge_u",
            OperationBuiltin::I32And => "i32_and",
            OperationBuiltin::I32Or => "i32_or",
            OperationBuiltin::I32Xor => "i32_xor",
            OperationBuiltin::I32Mul => "i32_mul",
            OperationBuiltin::I64And => "i64_and",
            OperationBuiltin::I64Or => "i64_or",
            OperationBuiltin::I64Xor => "i64_xor",
            OperationBuiltin::I64Mul => "i64_mul",
            OperationBuiltin::I32Shl => "i32_shl",
            OperationBuiltin::I32ShrU => "i32_shr_u",
            OperationBuiltin::I32ShrS => "i32_shr_s",
            OperationBuiltin::I32Rotl => "i32_rotl",
            OperationBuiltin::I32Rotr => "i32_rotr",
            OperationBuiltin::I32DivU => "i32_div_u",
            OperationBuiltin::I32DivS => "i32_div_s",
            OperationBuiltin::I32RemU => "i32_rem_u",
            OperationBuiltin::I32RemS => "i32_rem_s",
            OperationBuiltin::I32Popcnt => "i32_popcnt",
            OperationBuiltin::I64LtS => "i64_lt_s",
            OperationBuiltin::I64LtU => "i64_lt_u",
            OperationBuiltin::I64GtS => "i64_gt_s",
            OperationBuiltin::I64GtU => "i64_gt_u",
            OperationBuiltin::I64LeS => "i64_le_s",
            OperationBuiltin::I64LeU => "i64_le_u",
            OperationBuiltin::I64GeS => "i64_ge_s",
            OperationBuiltin::I64GeU => "i64_ge_u",
            OperationBuiltin::I64Shl => "i64_shl",
            OperationBuiltin::I64ShrS => "i64_shr_s",
            OperationBuiltin::I64ShrU => "i64_shr_u",
            OperationBuiltin::I64Rotl => "i64_rotl",
            OperationBuiltin::I64Rotr => "i64_rotr",
            OperationBuiltin::I64DivS => "i64_div_s",
            OperationBuiltin::I64DivU => "i64_div_u",
            OperationBuiltin::I64RemS => "i64_rem_s",
            OperationBuiltin::I64RemU => "i64_rem_u",
            OperationBuiltin::I64Clz => "i64_clz",
            OperationBuiltin::I64Ctz => "i64_ctz",
            OperationBuiltin::I64Popcnt => "i64_popcnt",
        }
    }
}

pub(crate) fn division_values(
    dividend: u64,
    divisor: u64,
    operation: Option<WasmOpcode>,
    wide: bool,
) -> Result<(u64, u64), String> {
    let Some(operation) = operation else {
        return Ok((0, 0));
    };
    if divisor == 0 {
        return Err("active division has a zero divisor".into());
    }
    if wide {
        if matches!(operation, WasmOpcode::I64DivS | WasmOpcode::I64RemS) {
            let left = dividend as i64;
            let right = divisor as i64;
            if operation == WasmOpcode::I64RemS && left == i64::MIN && right == -1 {
                return Ok((0, 0));
            }
            let quotient = left
                .checked_div(right)
                .ok_or_else(|| "active signed i64 division overflowed".to_owned())?;
            let remainder = left
                .checked_rem(right)
                .ok_or_else(|| "active signed i64 remainder overflowed".to_owned())?;
            Ok((quotient as u64, remainder as u64))
        } else {
            Ok((dividend / divisor, dividend % divisor))
        }
    } else if matches!(operation, WasmOpcode::I32DivS | WasmOpcode::I32RemS) {
        let left = dividend as u32 as i32;
        let right = divisor as u32 as i32;
        if operation == WasmOpcode::I32RemS && left == i32::MIN && right == -1 {
            return Ok((0, 0));
        }
        let quotient = left
            .checked_div(right)
            .ok_or_else(|| "active signed i32 division overflowed".to_owned())?;
        let remainder = left
            .checked_rem(right)
            .ok_or_else(|| "active signed i32 remainder overflowed".to_owned())?;
        Ok((quotient as u32 as u64, remainder as u32 as u64))
    } else {
        let left = dividend as u32;
        let right = divisor as u32;
        Ok((u64::from(left / right), u64::from(left % right)))
    }
}
