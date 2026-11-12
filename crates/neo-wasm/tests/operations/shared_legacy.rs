use super::bitwise::enforce_bitwise;
use super::mul_div::{enforce_div_rem, enforce_mul};
use super::shift::enforce_shifts;
use super::*;
use p3_field::PrimeField64;

const LEGACY_SECTIONS: &[Section] = {
    use WasmOpcode::*;
    &[
        Section {
            name: "bitwise",
            opcodes: || vec![I32And, I32Or, I32Xor, I64And, I64Or, I64Xor],
            constrain: enforce_bitwise,
        },
        Section {
            name: "compare",
            opcodes: || {
                compare::comparison_operations(false)
                    .iter()
                    .chain(compare::comparison_operations(true).iter())
                    .map(|entry| entry.0)
                    .collect()
            },
            constrain: enforce_comparisons,
        },
        Section {
            name: "shift",
            opcodes: || {
                vec![
                    I32Shl, I32ShrU, I32ShrS, I32Rotl, I32Rotr, I64Shl, I64ShrU, I64ShrS, I64Rotl, I64Rotr,
                ]
            },
            constrain: enforce_shifts,
        },
        Section {
            name: "mul",
            opcodes: || vec![I32Mul, I64Mul],
            constrain: enforce_mul,
        },
        Section {
            name: "zcount32",
            opcodes: || vec![I32Clz, I32Ctz],
            constrain: |b, i| enforce_zero_counts(b, &i.left32, &i.output32, false),
        },
        Section {
            name: "zcount64",
            opcodes: || vec![I64Clz, I64Ctz],
            constrain: |b, i| enforce_zero_counts(b, &i.left64, &i.output64, true),
        },
        Section {
            name: "popcnt32",
            opcodes: || vec![I32Popcnt],
            constrain: |b, i| enforce_popcount(b, &i.left32, &i.output32, I32Popcnt),
        },
        Section {
            name: "popcnt64",
            opcodes: || vec![I64Popcnt],
            constrain: |b, i| enforce_popcount(b, &i.left64, &i.output64, I64Popcnt),
        },
        Section {
            name: "divrem32",
            opcodes: || vec![I32DivU, I32DivS, I32RemU, I32RemS],
            constrain: |b, i| enforce_div_rem(b, &i.left32, &i.right32, &i.output32, false),
        },
        Section {
            name: "divrem64",
            opcodes: || vec![I64DivU, I64DivS, I64RemU, I64RemS],
            constrain: |b, i| enforce_div_rem(b, &i.left64, &i.right64, &i.output64, true),
        },
    ]
};

fn reference() -> OperationCircuit {
    let base = crate::build_wasm_relation().unwrap();
    let base_rows = base.r1cs().catalog().len();
    let inputs = Inputs::new().unwrap();
    let mut builder = CircuitBuilder::new(&base);
    let mut groups = Vec::new();
    for section in LEGACY_SECTIONS {
        let (column_start, row_start) = builder.position();
        builder.section(section.name, (section.opcodes)());
        (section.constrain)(&mut builder, &inputs).unwrap();
        let (column_end, row_end) = builder.position();
        groups.push(OperationGroup {
            name: section.name,
            columns: column_start..column_end,
            rows: row_start..row_end,
        });
    }
    let (relation, assignment) = builder.finish().unwrap();
    OperationCircuit {
        relation,
        assignment,
        base_rows,
        groups,
    }
}

fn accepted(circuit: &OperationCircuit, base: &[F]) -> bool {
    let mut witness = base.to_vec();
    witness.resize(circuit.relation().r1cs().column_count(), F::ZERO);
    if circuit.assign(&mut witness).is_err() {
        return false;
    }
    satisfied(circuit, &witness)
}

fn satisfied(circuit: &OperationCircuit, witness: &[F]) -> bool {
    circuit.operation_rows().all(|index| {
        let row = circuit.relation().r1cs().catalog().rows()[index].row();
        evaluate(row.a_terms(), witness) * evaluate(row.b_terms(), witness) == evaluate(row.c_terms(), witness)
    })
}

fn refresh_products(circuit: &OperationCircuit, witness: &mut [F]) {
    for index in circuit.operation_rows() {
        let row = circuit.relation().r1cs().catalog().rows()[index].row();
        let [(column, coefficient)] = row.c_terms() else {
            continue;
        };
        let family = circuit
            .relation()
            .columns()
            .family_for_column(*column)
            .unwrap();
        if family.role == "product" {
            witness[*column] = evaluate(row.a_terms(), witness) * evaluate(row.b_terms(), witness) / *coefficient;
        }
    }
}

fn refresh_ranges(circuit: &OperationCircuit, witness: &mut [F]) {
    let mut base = witness[..circuit.base_columns()].to_vec();
    crate::write_range_check_bits(&mut base);
    witness[..circuit.base_columns()].copy_from_slice(&base);
}

#[test]
fn shared_and_original_operation_relations_agree_on_dynamic_dispatch() {
    let old = reference();
    let new = operation_circuit().unwrap();
    assert_eq!(old.operation_rows().len(), 2213);
    assert!(new.operation_rows().len() < old.operation_rows().len());
    let mut body = String::new();
    for ty in ["i32", "i64"] {
        for op in [
            "and", "or", "xor", "mul", "shl", "shr_u", "shr_s", "rotl", "rotr", "div_u", "div_s", "rem_u", "rem_s",
            "lt_s", "lt_u", "gt_s", "gt_u", "le_s", "le_u", "ge_s", "ge_u",
        ] {
            body.push_str(&format!("{ty}.const -17 {ty}.const 5 {ty}.{op} drop\n"));
        }
        for op in ["clz", "ctz", "popcnt"] {
            body.push_str(&format!("{ty}.const -17 {ty}.{op} drop\n"));
        }
    }
    let source = format!("(module (func (export \"main\") (result i32) {body} i32.const 7))");
    let wasm = wat::parse_str(source).unwrap();
    let trace = crate::traces_from_wasmtime_wasm_bytes(&wasm, "main").unwrap();
    let mut seen = std::collections::HashSet::new();
    for step in &trace {
        let base = crate::build_witness_vector(step).unwrap();
        assert!(accepted(&old, &base), "original {:?}", step.opcode);
        assert!(accepted(new, &base), "shared {:?}", step.opcode);
        if step.info.operation.is_none() {
            continue;
        }
        seen.insert(step.opcode);
        for bit in [0, 1, 15, 31] {
            let mut wrong = base.clone();
            let column = crate::layout::COL_STACK_WRITE0_VALUE_LO;
            wrong[column] = F::from_u64(wrong[column].as_canonical_u64() ^ (1 << bit));
            crate::write_range_check_bits(&mut wrong);
            assert!(!accepted(&old, &wrong), "original accepts {:?} bit {bit}", step.opcode);
            assert!(!accepted(new, &wrong), "shared accepts {:?} bit {bit}", step.opcode);
        }
    }
    assert_eq!(seen.len(), 48);
    eprintln!(
        "operation_rows original={} shared={} complete={}",
        old.operation_rows().len(),
        new.operation_rows().len(),
        new.relation().r1cs().catalog().len()
    );
}

#[test]
fn shared_and_original_operation_relations_accept_the_same_traps() {
    let old = reference();
    let new = operation_circuit().unwrap();
    for ty in ["i32", "i64"] {
        let minimum = if ty == "i32" { i32::MIN as i64 } else { i64::MIN };
        for (a, b, op, trap) in [
            (7, 0, "div_u", true),
            (7, 0, "rem_u", true),
            (7, 0, "div_s", true),
            (7, 0, "rem_s", true),
            (minimum, -1, "div_s", true),
            (minimum, -1, "rem_s", false),
        ] {
            let source =
                format!("(module (func (export \"main\") (result {ty}) {ty}.const {a} {ty}.const {b} {ty}.{op}))");
            let wasm = wat::parse_str(source).unwrap();
            let trace = crate::traces_from_wasmtime_wasm_bytes(&wasm, "main").unwrap();
            assert_eq!(trace.last().unwrap().state_after.trapped, trap);
            for step in &trace {
                let base = crate::build_witness_vector(step).unwrap();
                assert!(accepted(&old, &base), "original {ty}.{op} {a} {b}");
                assert!(accepted(new, &base), "shared {ty}.{op} {a} {b}");
            }
        }
    }
}

#[test]
fn shared_advice_rejects_forgery_after_dependent_products_are_recomputed() {
    use crate::layout::COL_STACK_WRITE0_VALUE_LO;
    let circuit = operation_circuit().unwrap();
    let advice_columns = circuit
        .relation()
        .columns()
        .families()
        .iter()
        .filter(|family| {
            matches!(
                family.role,
                "integer_word"
                    | "absolute_carry"
                    | "remainder_delta"
                    | "remainder_borrow"
                    | "integer_carry"
                    | "integer_exponent"
            )
        })
        .flat_map(|family| family.start..family.end())
        .collect::<Vec<_>>();
    assert!(!advice_columns.is_empty());
    let mut body = String::new();
    for ty in ["i32", "i64"] {
        for op in [
            "and", "or", "xor", "mul", "shl", "shr_u", "shr_s", "rotl", "rotr", "div_u", "div_s", "rem_u", "rem_s",
        ] {
            body.push_str(&format!("{ty}.const -17 {ty}.const 5 {ty}.{op} drop\n"));
        }
    }
    for ty in ["i32", "i64"] {
        for op in ["clz", "ctz"] {
            for value in [0, 1, 16, -17] {
                body.push_str(&format!("{ty}.const {value} {ty}.{op} drop\n"));
            }
        }
    }
    let programs = [
        format!("(module (func (export \"main\") (result i32) {body} i32.const 7))"),
        "(module (func (export \"main\") (result i64) i64.const -9223372036854775808 i64.const -1 i64.div_s))".into(),
        "(module (func (export \"main\") (result i32) i32.const 7 i32.const 0 i32.div_u))".into(),
    ];
    let mut checked = 0;
    let mut traps = 0;
    for program in programs {
        let wasm = wat::parse_str(program).unwrap();
        let trace = crate::traces_from_wasmtime_wasm_bytes(&wasm, "main").unwrap();
        for step in trace {
            if step.info.operation.is_none() && step.opcode != WasmOpcode::I32Const {
                continue;
            }
            let mut honest = crate::build_witness_vector(&step).unwrap();
            honest.resize(circuit.relation().r1cs().column_count(), F::ZERO);
            circuit.assign(&mut honest).unwrap();
            assert!(satisfied(circuit, &honest));
            for &column in &advice_columns {
                let mut wrong = honest.clone();
                wrong[column] = F::ONE - wrong[column];
                refresh_products(circuit, &mut wrong);
                assert!(
                    !satisfied(circuit, &wrong),
                    "{:?} accepted shared advice {column}",
                    step.opcode
                );
                checked += 1;
            }
            if step.info.operation.is_some() && !step.state_after.trapped {
                for bit in 0..32 {
                    let mut wrong = honest.clone();
                    let value = wrong[COL_STACK_WRITE0_VALUE_LO].as_canonical_u64() ^ (1 << bit);
                    wrong[COL_STACK_WRITE0_VALUE_LO] = F::from_u64(value);
                    refresh_ranges(circuit, &mut wrong);
                    refresh_products(circuit, &mut wrong);
                    assert!(
                        !satisfied(circuit, &wrong),
                        "{:?} accepted low result bit {bit}",
                        step.opcode
                    );
                    checked += 1;
                }
            }
            traps += usize::from(step.state_after.trapped);
        }
    }
    assert_eq!(traps, 2);
    eprintln!("shared_advice_coordinated_forgeries_rejected\t{checked}\ttraps\t{traps}");
}

fn set_bits(witness: &mut [F], start: usize, width: usize, value: u64) {
    for bit in 0..width {
        witness[start + bit] = F::from_u64((value >> bit) & 1);
    }
}

#[test]
fn coordinated_integer_roles_reject_noncanonical_results() {
    let circuit = operation_circuit().unwrap();
    let families = circuit.relation().columns().families();
    let word_starts = families
        .iter()
        .filter(|f| f.role == "integer_word")
        .map(|f| f.start)
        .collect::<Vec<_>>();
    let carry_starts = families
        .iter()
        .filter(|f| f.role == "integer_carry")
        .map(|f| f.start)
        .collect::<Vec<_>>();
    assert_eq!(word_starts.len(), 9);
    assert_eq!(carry_starts.len(), 5);
    let exponent = families
        .iter()
        .find(|f| f.role == "integer_exponent")
        .unwrap()
        .start;
    let delta = families
        .iter()
        .find(|f| f.role == "remainder_delta")
        .unwrap()
        .start;
    let borrow = families
        .iter()
        .find(|f| f.role == "remainder_borrow")
        .unwrap()
        .start;
    let cases = [
        ("i64.const 101 i64.const 7 i64.div_u", 13, [13, 7, 10, 101, 0], 0),
        (
            "i64.const 1099511627776 i64.clz",
            22,
            [1 << 40, 1 << 22, 0, 1 << 62, 0],
            22,
        ),
        ("i64.const 16 i64.ctz", 3, [8, 2, 0, 16, 0], 3),
    ];
    for (body, result, roles, amount) in cases {
        let wasm = wat::parse_str(format!("(module (func (export \"main\") (result i64) {body}))")).unwrap();
        let trace = crate::traces_from_wasmtime_wasm_bytes(&wasm, "main").unwrap();
        let step = trace
            .iter()
            .find(|step| step.info.operation.is_some())
            .unwrap();
        let mut wrong = crate::build_witness_vector(step).unwrap();
        wrong.resize(circuit.relation().r1cs().column_count(), F::ZERO);
        circuit.assign(&mut wrong).unwrap();
        assert!(satisfied(circuit, &wrong));
        for index in [0, 3] {
            set_bits(&mut wrong, word_starts[index], 64, result);
        }
        for (index, value) in roles.into_iter().enumerate() {
            set_bits(&mut wrong, word_starts[index + 4], 64, value);
        }
        set_bits(&mut wrong, exponent, 6, amount);
        if step.opcode == WasmOpcode::I64DivU {
            let difference = roles[1].wrapping_sub(roles[2]).wrapping_sub(1);
            set_bits(&mut wrong, delta, 64, difference);
            let value = (roles[2] as u32 as u64 + difference as u32 as u64 + 1) >> 32;
            set_bits(&mut wrong, borrow, 1, value);
        }
        let [x, y, u, _, _] = roles;
        let mask = (1u64 << 22) - 1;
        let mut carry = 0;
        for column in 0..5 {
            let mut total = carry + if column < 3 { (u >> (22 * column)) & mask } else { 0 };
            for i in 0..3 {
                if column >= i && column - i < 3 {
                    total += ((x >> (22 * i)) & mask) * ((y >> (22 * (column - i))) & mask);
                }
            }
            carry = total >> 22;
            set_bits(&mut wrong, carry_starts[column], 24, carry);
        }
        wrong[crate::layout::COL_STACK_WRITE0_VALUE_LO] = F::from_u64(result);
        refresh_ranges(circuit, &mut wrong);
        refresh_products(circuit, &mut wrong);
        assert!(
            !satisfied(circuit, &wrong),
            "accepted coordinated {:?} forgery",
            step.opcode
        );
    }
}
