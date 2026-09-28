use super::*;
use crate::application_records::{
    ApplicationForm, ApplicationRecipeNode, ApplicationRecordsWriter, ApplicationRowHeader, ApplicationTerm,
};
use crate::package::load_prepared_application_records;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::OnceLock;

const FIXED_START: usize = MAGIC.len() + (1 + 4 + 4 + 1) * size_of::<u64>();

fn fixture() -> &'static (LoadedPerApplicationPackage, Vec<u8>) {
    static FIXTURE: OnceLock<(LoadedPerApplicationPackage, Vec<u8>)> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../nightstream/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json");
        let mut fixed: Value = serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();
        let application = fixed[3].as_array().unwrap();
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
        let batches = application[13].as_array().unwrap();
        let batch_count = batches.len();
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
        if let Some(batch) = batches.first() {
            assert_eq!(batch_count, 1);
            assert_eq!(batch[0].as_u64().unwrap() as usize, private_start);
            assert!(batch[2].as_array().unwrap().is_empty());
            for (recipe, value) in batch[1].as_array().unwrap().iter().enumerate() {
                let target = json!([0, [[private_start + recipe, 1]]]);
                let row = rows.iter().position(|row| row[3] == target).unwrap();
                let mut nodes = Vec::new();
                recipe_nodes(value, &inverse, &mut nodes);
                writer
                    .append_recipe(row, nodes.into_iter().map(Ok))
                    .unwrap();
            }
        }
        fixed[3][13] = json!([]);
        fixed[3][15] = json!([]);
        let fixed_batches = fixed[1][10].as_array_mut().unwrap();
        fixed_batches.truncate(fixed_batches.len() - batch_count);
        fixed[1][12].as_array_mut().unwrap().retain(|row| {
            let index = row[0].as_u64().unwrap() as usize;
            !(row_start..row_start + row_count).contains(&index)
        });
        let package = load_prepared_application_records(fixed, Arc::new(writer.finish().unwrap())).unwrap();
        let mut encoded = Vec::new();
        package.write_prepared(&mut encoded).unwrap();
        (package, encoded)
    })
}

fn recipe_nodes(value: &Value, inverse: &BTreeMap<usize, usize>, output: &mut Vec<ApplicationRecipeNode>) {
    let fields = value.as_array().unwrap();
    match fields[0].as_u64().unwrap() {
        0 => output.push(ApplicationRecipeNode::Variable(
            inverse[&(fields[1].as_u64().unwrap() as usize)],
        )),
        1 => output.push(ApplicationRecipeNode::Constant(fields[1].as_u64().unwrap())),
        tag @ (2 | 3) => {
            output.push(if tag == 2 {
                ApplicationRecipeNode::Add
            } else {
                ApplicationRecipeNode::Multiply
            });
            recipe_nodes(&fields[1], inverse, output);
            recipe_nodes(&fields[2], inverse, output);
        }
        _ => panic!("fixture recipe tag"),
    }
}

fn with_fixed_mutation(encoded: &[u8], mutate: impl FnOnce(&mut Value)) -> Vec<u8> {
    let length = u64::from_le_bytes(encoded[MAGIC.len()..MAGIC.len() + 8].try_into().unwrap()) as usize;
    let mut fixed: Value = serde_json::from_slice(&encoded[FIXED_START..FIXED_START + length]).unwrap();
    mutate(&mut fixed);
    let fixed = serde_json::to_vec(&fixed).unwrap();
    let mut output = encoded[..FIXED_START].to_vec();
    output[MAGIC.len()..MAGIC.len() + 8].copy_from_slice(&(fixed.len() as u64).to_le_bytes());
    output.extend(fixed);
    output.extend_from_slice(&encoded[FIXED_START + length..]);
    output
}

#[test]
fn prepared_round_trip_preserves_frozen_binding_and_exact_record_stream() {
    let (original, encoded) = fixture();
    let loaded = load_compiled_application_package(encoded.as_slice()).unwrap();
    assert_eq!(loaded.structural_identifier(), original.structural_identifier());
    assert_eq!(loaded.application(), original.application());
    assert_eq!(loaded.row_count(), original.row_count());
    assert_eq!(loaded.logical_column_count(), original.logical_column_count());
    let binding = loaded.production_verifier_binding().unwrap();
    assert_eq!(binding, original.production_verifier_binding().unwrap());
    assert_eq!(
        binding.package_identity(),
        crate::POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY
    );
    assert_eq!(
        binding.verification_key_digest(),
        crate::POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST
    );
    let mut saved = Vec::new();
    loaded.write_prepared(&mut saved).unwrap();
    assert_eq!(&saved, encoded);
    let before = original.application_records().unwrap();
    let after = loaded.application_records().unwrap();
    assert!(!Arc::ptr_eq(before, after), "load seals a separate immutable owner");
    if before.recipe_count() != 0 {
        assert_eq!(
            before
                .evaluate_recipe(0, |variable| Ok(variable as u64))
                .unwrap(),
            after
                .evaluate_recipe(0, |variable| Ok(variable as u64))
                .unwrap()
        );
    }
}

#[test]
fn prepared_loading_keeps_cached_identity_components_non_authoritative() {
    let (original, encoded) = fixture();
    let mut changed = encoded.clone();
    let start = MAGIC.len() + size_of::<u64>();
    let replacement = (original.structural_identifier()[0] + 1) % GOLDILOCKS_MODULUS;
    changed[start..start + 8].copy_from_slice(&replacement.to_le_bytes());
    let loaded = load_compiled_application_package(changed.as_slice()).unwrap();
    assert_eq!(loaded.structural_identifier()[0], replacement);
    assert_ne!(
        loaded.production_verifier_binding().unwrap(),
        original.production_verifier_binding().unwrap()
    );
}

#[test]
fn prepared_loading_rejects_bad_framing_fields_and_cached_word_count() {
    let (_, encoded) = fixture();
    let mut bad = encoded.clone();
    bad[0] ^= 1;
    assert!(load_compiled_application_package(bad.as_slice()).is_err());
    assert!(load_compiled_application_package(&encoded[..encoded.len() - 1]).is_err());
    let mut bad = encoded.clone();
    bad.push(0);
    assert!(matches!(
        load_compiled_application_package(bad.as_slice()),
        Err(PackageError::Invalid("prepared package has trailing bytes"))
    ));
    let mut bad = encoded.clone();
    let start = MAGIC.len() + size_of::<u64>();
    bad[start..start + 8].copy_from_slice(&GOLDILOCKS_MODULUS.to_le_bytes());
    assert!(matches!(
        load_compiled_application_package(bad.as_slice()),
        Err(PackageError::NonCanonicalField { .. })
    ));
    let mut bad = encoded.clone();
    let start = FIXED_START - size_of::<u64>();
    let count = u64::from_le_bytes(bad[start..start + 8].try_into().unwrap());
    bad[start..start + 8].copy_from_slice(&(count + 1).to_le_bytes());
    assert!(matches!(
        load_compiled_application_package(bad.as_slice()),
        Err(PackageError::Invalid("prepared application identity word count"))
    ));
}

#[test]
fn prepared_source_byte_bound_is_checked_before_copy_or_body_decode() {
    let mut encoded = MAGIC.to_vec();
    encoded.extend_from_slice(&(MAX_FIXED_SOURCE_BYTES + 1).to_le_bytes());
    assert!(matches!(
        load_compiled_application_package(encoded.as_slice()),
        Err(PackageError::Invalid(
            "prepared fixed envelope exceeds compiler byte bound"
        ))
    ));
}

#[test]
fn prepared_loading_retains_profile_layout_and_matrix_validation() {
    let (_, encoded) = fixture();
    for changed in [
        with_fixed_mutation(encoded, |fixed| fixed[1][0] = json!(0)),
        with_fixed_mutation(encoded, |fixed| fixed[1][1][1] = json!(4)),
        with_fixed_mutation(encoded, |fixed| fixed[3][6] = json!(fixed[3][6].as_u64().unwrap() + 1)),
        with_fixed_mutation(encoded, |fixed| fixed[2] = json!([])),
    ] {
        assert!(load_compiled_application_package(changed.as_slice()).is_err());
    }
}
