use std::collections::BTreeMap;

use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use nightstream::application::{poseidon2_hash_chain_v1, Affine, ApplicationCircuit};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde_json::{json, Value};

fn reference() -> Value {
    serde_json::from_slice(include_bytes!("fixtures/poseidon2-application-reference.json")).unwrap()
}

fn execution() -> ([Goldilocks; 4], [Goldilocks; 4], [Goldilocks; 4]) {
    let value: Value = serde_json::from_slice(include_bytes!("fixtures/poseidon2-application-execution.json")).unwrap();
    let words = |index: usize| std::array::from_fn(|lane| field(&value[index][lane]));
    (words(0), words(1), words(2))
}

fn index(value: &Value) -> usize {
    usize::try_from(value.as_u64().unwrap()).unwrap()
}

fn field(value: &Value) -> Goldilocks {
    let word = value.as_u64().unwrap();
    assert!(word < 0xffff_ffff_0000_0001);
    Goldilocks::from_u64(word)
}

fn column_map(circuit: &ApplicationCircuit, reference: &Value) -> Vec<usize> {
    let mut columns = vec![usize::MAX; circuit.variable_count()];
    for (variable, column) in circuit
        .input_state()
        .iter()
        .zip(reference[2].as_array().unwrap())
    {
        columns[variable.index()] = index(column);
    }
    for (variable, column) in circuit
        .private_inputs()
        .iter()
        .zip(reference[3].as_array().unwrap())
    {
        columns[variable.index()] = index(column);
    }
    for (variable, column) in circuit
        .output_state()
        .iter()
        .zip(reference[4].as_array().unwrap())
    {
        columns[variable.index()] = index(column);
    }
    for (offset, variable) in circuit.generated_range().enumerate() {
        columns[variable] = index(&reference[5]) + offset;
    }
    assert!(!columns.contains(&usize::MAX));
    columns
}

fn affine(value: &Affine, columns: &[usize]) -> Value {
    let mut terms = BTreeMap::new();
    for (variable, coefficient) in value.terms() {
        *terms
            .entry(columns[variable.index()])
            .or_insert(Goldilocks::ZERO) += *coefficient;
    }
    let terms: Vec<_> = terms
        .into_iter()
        .filter(|(_, value)| *value != Goldilocks::ZERO)
        .map(|(column, value)| json!([column, value.as_canonical_u64()]))
        .collect();
    json!([value.constant_term().as_canonical_u64(), terms])
}

fn normalized(value: &Value) -> Value {
    let mut terms = BTreeMap::new();
    for term in value[1].as_array().unwrap() {
        *terms.entry(index(&term[0])).or_insert(Goldilocks::ZERO) += field(&term[1]);
    }
    let terms: Vec<_> = terms
        .into_iter()
        .filter(|(_, value)| *value != Goldilocks::ZERO)
        .map(|(column, value)| json!([column, value.as_canonical_u64()]))
        .collect();
    json!([field(&value[0]).as_canonical_u64(), terms])
}

fn actual_rows(circuit: &ApplicationCircuit, reference: &Value) -> Vec<Value> {
    let columns = column_map(circuit, reference);
    circuit
        .rows()
        .iter()
        .enumerate()
        .map(|(offset, row)| {
            json!([
                index(&reference[7]) + offset,
                affine(row.a(), &columns),
                affine(row.b(), &columns),
                affine(row.c(), &columns)
            ])
        })
        .collect()
}

fn reference_rows(reference: &Value) -> Vec<Value> {
    assert!(reference[14].as_array().unwrap().is_empty());
    reference[15]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| json!([row[0], normalized(&row[1]), normalized(&row[2]), normalized(&row[3])]))
        .collect()
}

fn expression(value: &Value, values: &BTreeMap<usize, Goldilocks>) -> Goldilocks {
    match index(&value[0]) {
        0 => values[&index(&value[1])],
        1 => field(&value[1]),
        2 => expression(&value[1], values) + expression(&value[2], values),
        3 => expression(&value[1], values) * expression(&value[2], values),
        _ => panic!("unsupported saved Lean witness expression"),
    }
}

#[test]
fn every_poseidon2_application_row_matches_the_saved_lean_reference() {
    let circuit = poseidon2_hash_chain_v1().unwrap();
    let reference = reference();
    assert_eq!(index(&reference[0]), 1);
    assert_eq!(circuit.private_input_count(), index(&reference[1]));
    assert_eq!(circuit.generated_range().len(), index(&reference[6]));
    assert_eq!(circuit.rows().len(), index(&reference[8]));
    for slot in 9..13 {
        assert!(reference[slot].as_array().unwrap().is_empty());
    }
    let actual = actual_rows(&circuit, &reference);
    let expected = reference_rows(&reference);
    for (row, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(actual, expected, "application row {row}");
    }
    assert_eq!(actual.len(), expected.len());

    let mut changed_constraint = expected.clone();
    changed_constraint[0][1][0] = json!((field(&changed_constraint[0][1][0]) + Goldilocks::ONE).as_canonical_u64());
    assert_ne!(
        actual, changed_constraint,
        "the comparison must detect a changed constraint"
    );
}

#[test]
fn every_poseidon2_witness_value_matches_the_saved_lean_program() {
    let circuit = poseidon2_hash_chain_v1().unwrap();
    let reference = reference();
    let (input, message, output) = execution();
    let witness = circuit.execute(input, &message).unwrap();
    assert_eq!(witness.output_state(), output);

    let mut preimage: Vec<_> = b"Nightstream/Stage1/Poseidon2HashChain/v1"
        .iter()
        .map(|byte| Goldilocks::from_u64(u64::from(*byte)))
        .collect();
    preimage.extend(input);
    preimage.extend(message);
    assert_eq!(witness.output_state(), poseidon2_hash(&preimage));

    let columns = column_map(&circuit, &reference);
    let mut expected = BTreeMap::new();
    for (variable, column) in columns
        .iter()
        .enumerate()
        .take(circuit.generated_range().start)
    {
        expected.insert(*column, witness.values()[variable]);
    }
    for batch in reference[13].as_array().unwrap() {
        assert!(batch[2].as_array().unwrap().is_empty());
        for (offset, recipe) in batch[1].as_array().unwrap().iter().enumerate() {
            let value = expression(recipe, &expected);
            assert!(expected.insert(index(&batch[0]) + offset, value).is_none());
        }
    }
    for variable in circuit.generated_range() {
        assert_eq!(
            witness.values()[variable],
            expected[&columns[variable]],
            "application witness variable {variable}"
        );
    }
}

#[test]
fn poseidon2_constraints_reject_changed_inputs_outputs_and_intermediates() {
    let circuit = poseidon2_hash_chain_v1().unwrap();
    let (input, message, _) = execution();
    let witness = circuit.execute(input, &message).unwrap();
    for variable in [
        circuit.input_state()[0].index(),
        circuit.private_inputs()[0].index(),
        circuit.output_state()[0].index(),
        circuit.generated_range().start,
    ] {
        let mut changed = witness.values().to_vec();
        changed[variable] += Goldilocks::ONE;
        assert!(
            circuit.check(&changed).is_err(),
            "changed application variable {variable}"
        );
    }
}
