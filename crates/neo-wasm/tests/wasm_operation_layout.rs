mod common;
use neo_math::F;
use neo_wasm::operations::r1cs::operation_circuit;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

#[test]
fn complete_catalog_preserves_base_and_names_every_auxiliary() {
    let base = neo_wasm::build_wasm_relation().unwrap();
    let circuit = operation_circuit().unwrap();
    let relation = circuit.relation();
    assert_eq!(circuit.base_columns(), base.r1cs().column_count());
    assert_eq!(
        &relation.r1cs().catalog().rows()[..circuit.operation_rows().start],
        base.r1cs().catalog().rows()
    );
    assert_eq!(
        relation.r1cs().catalog().gadget_occurrences(),
        base.r1cs().catalog().gadget_occurrences()
    );
    let mut next_column = circuit.base_columns();
    let mut next_row = circuit.operation_rows().start;
    for group in circuit.groups() {
        assert_eq!(group.columns.start, next_column);
        assert_eq!(group.rows.start, next_row);
        for column in group.columns.clone() {
            let family = relation.columns().family_for_column(column).unwrap();
            assert!(family
                .name
                .starts_with(&format!("operations.{}.", group.name)));
        }
        for row in &relation.r1cs().catalog().rows()[group.rows.clone()] {
            assert_eq!(row.tag().label(), group.name);
        }
        next_column = group.columns.end;
        next_row = group.rows.end;
    }
    assert_eq!(next_column, relation.r1cs().column_count());
    assert_eq!(next_row, relation.r1cs().catalog().len());
}

#[test]
fn assignment_overwrites_advice_without_changing_base_or_shape() {
    let checked = common::checked_main(
        r#"(module (func (export "main") (result i32)
        i64.const -9223372036854775808 i64.const -1 i64.rem_s drop
        i64.const -17 i64.const 9 i64.div_s drop
        i32.const 65535 i32.const 65535 i32.mul drop
        i32.const -17 i32.const 63 i32.rotr drop
        i64.const 0 i64.clz drop
        i32.const 0))"#,
    );
    let circuit = operation_circuit().unwrap();
    let widths = neo_application::range_checked_variable_widths(circuit.relation().columns());
    for step in &checked.trace {
        let base = neo_wasm::build_witness_vector(step).unwrap();
        let mut witness = base.clone();
        witness.resize(circuit.relation().r1cs().column_count(), F::from_u64(999));
        circuit.assign(&mut witness).unwrap();
        assert_eq!(&witness[..base.len()], base);
        circuit.relation().r1cs().check_rows(&witness).unwrap();
        for (&value, &width) in witness.iter().zip(&widths) {
            assert!(width == 64 || value.as_canonical_u64() >> width == 0);
        }
        let mut clean = base;
        clean.resize(witness.len(), F::ZERO);
        circuit.assign(&mut clean).unwrap();
        assert_eq!(witness, clean);
    }
}
