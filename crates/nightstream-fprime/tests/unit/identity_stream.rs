use super::*;
use crate::application_records::{
    ApplicationForm, ApplicationRecipeNode, ApplicationRecordsWriter, ApplicationRowHeader, ApplicationTerm,
};
use crate::package::native_application::PreparedApplication;
use serde_json::json;
use std::{collections::BTreeMap, sync::Arc};

#[test]
fn streamed_identity_matches_the_original_numeric_array_preimage() {
    let value = json!([0, [7, u64::MAX], []]);
    let mut words = IDENTITY_DOMAIN.to_vec();
    words.extend([
        1,
        3,
        0,
        0, // outer array
        0,
        0,
        0,
        0, // zero
        1,
        2,
        0,
        0, // nested array
        0,
        7,
        0,
        0, // seven
        0,
        0xffff_ffff,
        0xffff_ffff,
        0, // both limbs of u64::MAX
        1,
        0,
        0,
        0, // empty array
    ]);
    let fields: Vec<_> = words.into_iter().map(Goldilocks::from_u64).collect();
    assert_eq!(
        relation_identifier(&value).unwrap(),
        poseidon2::poseidon2_hash(&fields).map(|value| value.as_canonical_u64())
    );
}

#[test]
fn streamed_identity_rejects_non_natural_atoms_and_non_array_nodes() {
    for value in [json!(-1), json!(0.5), json!(null), json!(true), json!("7"), json!({})] {
        assert!(relation_identifier(&json!([0, value])).is_err());
    }
}

#[test]
fn streamed_component_hash_matches_slice_hash_at_rate_boundaries() {
    for length in 0..=2 * poseidon2::RATE + 1 {
        let words: Vec<_> = (0..length)
            .map(|index| Goldilocks::ORDER_U64 - 1 - index as u64)
            .collect();
        let mut framed = bytes_as_words(VERIFIER_CONTEXT_COMPONENT_DOMAIN);
        framed.push(2);
        append_framed(&mut framed, &words).unwrap();
        let fields: Vec<_> = framed.into_iter().map(Goldilocks::from_u64).collect();
        let expected = poseidon2::poseidon2_hash(&fields).map(|value| value.as_canonical_u64());
        let application = ApplicationIdentity::from_words(&words).unwrap();
        assert_eq!(application.digest, expected);
        assert_eq!(application.word_count, words.len());
        let fields: Vec<_> = words.iter().copied().map(Goldilocks::from_u64).collect();
        assert_eq!(
            poseidon_words(&words),
            poseidon2::poseidon2_hash(&fields).map(|value| value.as_canonical_u64())
        );
    }
    assert!(ApplicationIdentity::from_words(&[Goldilocks::ORDER_U64]).is_err());
}

fn independent_preimage(value: &Value, words: &mut Vec<u64>) {
    match value {
        Value::Number(value) => {
            let value = value.as_u64().unwrap();
            words.extend([0, value & 0xffff_ffff, value >> 32, 0]);
        }
        Value::Array(values) => {
            let length = values.len() as u64;
            words.extend([1, length & 0xffff_ffff, length >> 32, 0]);
            for value in values {
                independent_preimage(value, words);
            }
        }
        _ => panic!("fixture must use numeric arrays"),
    }
}

fn fixture_records(original: &Value) -> (Value, PreparedApplication) {
    let application = original[3].as_array().unwrap();
    let row_start = application[7].as_u64().unwrap() as usize;
    let row_count = application[8].as_u64().unwrap() as usize;
    let private_start = application[5].as_u64().unwrap() as usize;
    let private_count = application[6].as_u64().unwrap() as usize;
    let mut columns = Vec::new();
    for field in [2, 3, 4] {
        columns.extend(
            application[field]
                .as_array()
                .unwrap()
                .iter()
                .map(|column| column.as_u64().unwrap() as usize),
        );
    }
    columns.extend(private_start..private_start + private_count);
    let inverse: BTreeMap<_, _> = columns
        .into_iter()
        .enumerate()
        .map(|(local, physical)| (physical, local))
        .collect();
    let rows = application[15].as_array().unwrap();
    let mut writer = ApplicationRecordsWriter::new().unwrap();
    for (local, row) in rows.iter().enumerate() {
        assert_eq!(row[0].as_u64().unwrap() as usize, row_start + local);
        let header = ApplicationRowHeader {
            constants: std::array::from_fn(|form| row[form + 1][0].as_u64().unwrap()),
            term_counts: std::array::from_fn(|form| row[form + 1][1].as_array().unwrap().len()),
        };
        let mut terms = Vec::new();
        for (form, kind) in [ApplicationForm::A, ApplicationForm::B, ApplicationForm::C]
            .into_iter()
            .enumerate()
        {
            for term in row[form + 1][1].as_array().unwrap() {
                terms.push(Ok(ApplicationTerm {
                    form: kind,
                    variable: inverse[&(term[0].as_u64().unwrap() as usize)],
                    coefficient: term[1].as_u64().unwrap(),
                }));
            }
        }
        writer.append_row(header, terms).unwrap();
    }
    let batches = application[13].as_array().unwrap();
    if let Some(batch) = batches.first() {
        assert_eq!(batches.len(), 1);
        assert_eq!(batch[0].as_u64().unwrap() as usize, private_start);
        assert!(batch[2].as_array().unwrap().is_empty());
        for (recipe, value) in batch[1].as_array().unwrap().iter().enumerate() {
            let target = json!([0, [[private_start + recipe, 1]]]);
            let generated_row = rows.iter().position(|row| row[3] == target).unwrap();
            let mut nodes = Vec::new();
            recipe_nodes(value, &inverse, &mut nodes);
            writer
                .append_recipe(generated_row, nodes.into_iter().map(Ok))
                .unwrap();
        }
    }
    let records = Arc::new(writer.finish().unwrap());
    let mut metadata = original[3].clone();
    metadata[13] = json!([]);
    metadata[15] = json!([]);
    let prepared = PreparedApplication::from_metadata(&metadata, records).unwrap();
    let mut fixed = original.clone();
    fixed[3] = metadata;
    let fixed_batches = fixed[1][10].as_array_mut().unwrap();
    assert!(fixed_batches.ends_with(batches));
    fixed_batches.truncate(fixed_batches.len() - batches.len());
    fixed[1][12].as_array_mut().unwrap().retain(|row| {
        let index = row[0].as_u64().unwrap() as usize;
        !(row_start..row_start + row_count).contains(&index)
    });
    (fixed, prepared)
}

fn recipe_nodes(value: &Value, inverse: &BTreeMap<usize, usize>, nodes: &mut Vec<ApplicationRecipeNode>) {
    let fields = value.as_array().unwrap();
    match fields[0].as_u64().unwrap() {
        0 => nodes.push(ApplicationRecipeNode::Variable(
            inverse[&(fields[1].as_u64().unwrap() as usize)],
        )),
        1 => nodes.push(ApplicationRecipeNode::Constant(fields[1].as_u64().unwrap())),
        tag @ (2 | 3) => {
            nodes.push(if tag == 2 {
                ApplicationRecipeNode::Add
            } else {
                ApplicationRecipeNode::Multiply
            });
            recipe_nodes(&fields[1], inverse, nodes);
            recipe_nodes(&fields[2], inverse, nodes);
        }
        _ => panic!("fixture recipe tag"),
    }
}

fn assert_native_preimage(original: &Value, fixed: &Value, application: &PreparedApplication) {
    let mut expected = Vec::new();
    independent_preimage(original, &mut expected);
    let mut offset = 0;
    native::visit_native_preimage_words(fixed, application, &mut |words| {
        let end = offset + words.len();
        assert_eq!(
            words,
            &expected[offset..end],
            "native envelope preimage at word {offset}"
        );
        offset = end;
        Ok(())
    })
    .unwrap();
    assert_eq!(offset, expected.len());
    expected.clear();
    independent_preimage(&original[3], &mut expected);
    offset = 0;
    visit_native_application_words(application, &mut |words| {
        let end = offset + words.len();
        assert_eq!(
            words,
            &expected[offset..end],
            "native application preimage at word {offset}"
        );
        offset = end;
        Ok(())
    })
    .unwrap();
    assert_eq!(offset, expected.len());
    let identity = native_application_identity(application).unwrap();
    assert_eq!(identity, ApplicationIdentity::from_words(&expected).unwrap());
}

#[test]
fn native_records_preserve_complete_frozen_envelope_preimage_and_binding() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../nightstream/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json");
    let original: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let (fixed, application) = fixture_records(&original);
    assert_native_preimage(&original, &fixed, &application);
    let structural = native_relation_identifier(&fixed, &application).unwrap();
    assert_eq!(structural, POSEIDON2_HASH_CHAIN_V1_STRUCTURAL_IDENTIFIER);
    let identity = native_application_identity(&application).unwrap();
    let relation = value_preimage_words(&original[1][4]).unwrap();
    let binding = stage1_verifier_binding(
        structural,
        original[1][4][1].as_u64().unwrap() as usize,
        &relation,
        &identity,
    )
    .unwrap();
    assert_eq!(binding.package_identity(), POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY);
    assert_eq!(
        binding.verification_key_digest(),
        POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST
    );
    assert_eq!(binding.verifier_context().descriptor_words().len(), 480);
    assert_eq!(binding.verification_key_words().len(), 520);
}

#[test]
fn native_records_preserve_duplicate_zero_terms_and_recipe_tree_order() {
    let row = json!([
        11,
        [0, [[3, 1], [2, 1], [3, Goldilocks::ORDER_U64 - 1], [2, 0]]],
        [1, []],
        [0, [[10, 1]]]
    ]);
    let tail = json!([2, [3, [1, Goldilocks::ORDER_U64 - 1], [0, 3]], [3, [1, 0], [0, 2]]]);
    let left = json!([2, [2, [0, 3], [0, 2]], tail]);
    let right = json!([2, [0, 3], [2, [0, 2], tail]]);
    let mut identities = Vec::new();
    for recipe in [left, right] {
        let batch = json!([10, [recipe], []]);
        let application = json!([
            1,
            0,
            [2, 3, 4, 5],
            [],
            [6, 7, 8, 9],
            10,
            1,
            11,
            1,
            [],
            [],
            [],
            [],
            [batch],
            [],
            [row]
        ]);
        let prefix = json!([5, [0, []], [1, []], [0, []]]);
        let suffix = json!([12, [0, []], [1, []], [0, []]]);
        let source = json!([
            8,
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [batch],
            [],
            [prefix, row, suffix],
            []
        ]);
        let original = json!([6, source, [], application, [], [12, 1], 0]);
        let (fixed, application) = fixture_records(&original);
        assert_native_preimage(&original, &fixed, &application);
        identities.push(native_relation_identifier(&fixed, &application).unwrap());
    }
    assert_ne!(identities[0], identities[1]);
}

#[test]
fn native_records_without_recipes_do_not_insert_a_batch() {
    let row = json!([11, [0, [[2, 1]]], [1, []], [0, [[6, 1]]]]);
    let application = json!([
        1,
        0,
        [2, 3, 4, 5],
        [],
        [6, 7, 8, 9],
        10,
        0,
        11,
        1,
        [],
        [],
        [],
        [],
        [],
        [],
        [row]
    ]);
    let source = json!([8, [], [], [], [], [], [], [], [], [], [], [], [row], []]);
    let original = json!([6, source, [], application, [], [12, 1], 0]);
    let (fixed, application) = fixture_records(&original);
    assert_native_preimage(&original, &fixed, &application);
}
