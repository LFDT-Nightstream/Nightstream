use super::*;
use crate::application::{poseidon2_hash_chain_v1, Affine, ApplicationBuilder};
use nightstream_fprime::{ApplicationForm, ApplicationRecipeNode};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use serde_json::json;
use std::ops::ControlFlow;

fn manifest_bytes() -> &'static [u8] {
    include_bytes!("../../artifacts/shared-verifier-v1.json")
}

// These small conformance fixtures materialize every word for comparison. The
// production assembler passes the sealed records directly to the native loader.
fn materialized_plan(application: &ApplicationCircuit, manifest: &Manifest) -> wire::ApplicationPlan {
    let mut plan = application::plan(application, manifest).unwrap();
    let mut columns = plan.input_columns.clone();
    columns.extend_from_slice(&plan.witness_columns);
    columns.extend_from_slice(&plan.output_columns);
    columns.extend(plan.private_start..plan.private_start + plan.private_count);
    let records = application.records();
    for row in 0..records.row_count() {
        let header = records.row_header(row).unwrap();
        let mut forms = header.constants.map(|constant| wire::Combination {
            constant,
            terms: Vec::new(),
        });
        assert_eq!(
            records
                .visit_terms(row, |term| {
                    let form = match term.form {
                        ApplicationForm::A => 0,
                        ApplicationForm::B => 1,
                        ApplicationForm::C => 2,
                    };
                    forms[form]
                        .terms
                        .push((columns[term.variable], term.coefficient));
                    Ok(ControlFlow::Continue(()))
                })
                .unwrap(),
            ControlFlow::Continue(())
        );
        let [a, b, c] = forms;
        plan.rows.push(wire::Row {
            index: plan.row_start + row,
            a,
            b,
            c,
        });
    }
    if records.recipe_count() != 0 {
        let mut recipes = Vec::new();
        for recipe in 0..records.recipe_count() {
            let mut nodes = Vec::new();
            assert_eq!(
                records
                    .visit_recipe_nodes(recipe, |node| {
                        nodes.push(node);
                        Ok(ControlFlow::Continue(()))
                    })
                    .unwrap(),
                ControlFlow::Continue(())
            );
            let mut nodes = nodes.into_iter();
            recipes.push(recipe_value(&mut nodes, &columns));
            assert!(nodes.next().is_none());
        }
        plan.batches.push(wire::Batch {
            start: plan.private_start,
            recipes,
            hints: Vec::new(),
        });
    }
    plan
}

fn recipe_value(nodes: &mut impl Iterator<Item = ApplicationRecipeNode>, columns: &[usize]) -> Value {
    match nodes.next().expect("complete stored recipe") {
        ApplicationRecipeNode::Variable(variable) => json!([0, columns[variable]]),
        ApplicationRecipeNode::Constant(value) => json!([1, value]),
        ApplicationRecipeNode::Add => json!([2, recipe_value(nodes, columns), recipe_value(nodes, columns)]),
        ApplicationRecipeNode::Multiply => json!([3, recipe_value(nodes, columns), recipe_value(nodes, columns)]),
    }
}

fn materialized_assembly(fixed: Value, application: &ApplicationCircuit, manifest: &Manifest) -> Value {
    let mut value: wire::Envelope = serde_json::from_value(fixed).unwrap();
    let plan = materialized_plan(application, manifest);
    let split = value
        .source
        .rows
        .partition_point(|row| row.index < plan.row_start);
    value
        .source
        .rows
        .splice(split..split, plan.rows.iter().cloned());
    value.source.batches.extend(plan.batches.iter().cloned());
    value.application = plan;
    serde_json::to_value(value).unwrap()
}

#[test]
fn rust_poseidon_plan_preserves_every_raw_row_recipe_and_identity_byte() {
    let manifest = Manifest::parse(manifest_bytes()).unwrap();
    let application = poseidon2_hash_chain_v1().unwrap();
    let plan = materialized_plan(&application, &manifest);
    let expected = include_bytes!("../fixtures/poseidon2-application-reference.json");
    let mut actual = serde_json::to_vec(&plan).unwrap();
    actual.push(b'\n');
    // Syntax, duplicate sparse terms and recipe operation order are identity inputs.
    assert_eq!(actual.len(), expected.len(), "raw application plan byte length");
    let first_difference = actual
        .iter()
        .zip(expected.iter())
        .position(|(actual, expected)| actual != expected);
    assert_eq!(first_difference, None, "first different raw application plan byte");
}

#[test]
fn independent_poseidon_assembly_equals_the_complete_reference_value() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json");
    let bytes = std::fs::read(path).unwrap();
    let expected: Value = serde_json::from_slice(&bytes).unwrap();
    let reference: wire::Envelope = serde_json::from_slice(&bytes).unwrap();
    let manifest = Manifest::parse(manifest_bytes()).unwrap();
    let application = poseidon2_hash_chain_v1().unwrap();
    let fixed = assemble(reference, &manifest, &application).unwrap();
    let actual = materialized_assembly(fixed, &application, &manifest);

    // Compare the entire numeric-array value: physical constraints and witness
    // recipes, every matrix block, assignment transport, layout and terminal data.
    // No identity digest or selected row interval substitutes for this equality.
    assert!(
        actual == expected,
        "complete assembled value differs from the selected reference"
    );
}

#[test]
fn assembled_fixed_source_reaches_the_compiler_node_bound() {
    fn nodes(value: &Value) -> usize {
        match value {
            Value::Array(values) => 1 + values.iter().map(nodes).sum::<usize>(),
            Value::Number(number) if number.as_u64().is_some() => 1,
            _ => panic!("fixed compiler output must contain only arrays and u64 values"),
        }
    }

    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json");
    let bytes = std::fs::read(path).unwrap();
    let manifest = Manifest::parse(manifest_bytes()).unwrap();
    // The selected key permits W+L=7,701. The largest fixed envelope uses
    // W=7,700 and L=1, because both nonempty private segments add array nodes.
    // Count actual assembler output independently of the loader's bound.
    for (witness, has_local, expected_nodes) in [(7_700, true, 32_045_229), (0, false, 32_037_521)] {
        let mut builder = ApplicationBuilder::new(witness).unwrap();
        if has_local {
            builder.affine(Affine::constant(Goldilocks::ZERO)).unwrap();
        }
        let input = builder.input_state();
        let application = builder.finish(input.map(Affine::from)).unwrap();
        if has_local {
            let counts = Counts::of(&application);
            assert_eq!((counts.witness, counts.local, counts.rows), (7_700, 1, 5));
            let key_width = neo_ajtai::nightstream_fprime_setup::PRODUCTION_CARRIER_WIDTH;
            assert!(manifest.geometry.logical_width.eval(counts).unwrap() <= key_width);
            assert!(
                manifest
                    .geometry
                    .logical_width
                    .eval(Counts {
                        witness: counts.witness + 1,
                        ..counts
                    })
                    .unwrap()
                    > key_width
            );
        }
        // Reuse the input bytes, while retaining only one assembled Value at a time.
        let reference: wire::Envelope = serde_json::from_slice(&bytes).unwrap();
        let fixed = assemble(reference, &manifest, &application).unwrap();
        assert_eq!(
            nodes(&fixed),
            expected_nodes,
            "W={witness}, L={}",
            usize::from(has_local)
        );
    }
}

#[test]
fn preparation_rejects_a_changed_reference_even_when_assembly_repairs_it() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json");
    let bytes = std::fs::read(path).unwrap();
    let expected: Value = serde_json::from_slice(&bytes).unwrap();
    let mut reference: wire::Envelope = serde_json::from_slice(&bytes).unwrap();
    drop(bytes);
    // This part of the reference is replaced by the caller's application.
    // A correct candidate therefore cannot authorize this altered blueprint.
    let constant = &mut reference.application.rows[0].a.constant;
    *constant = u64::from(*constant == 0);
    let mut changed = serde_json::to_vec(&reference).unwrap();
    changed.push(b'\n');
    let manifest = Manifest::parse(manifest_bytes()).unwrap();
    let application = poseidon2_hash_chain_v1().unwrap();
    let fixed = assemble(reference, &manifest, &application).unwrap();
    let candidate = materialized_assembly(fixed, &application, &manifest);
    assert!(
        candidate == expected,
        "the assembled circuit is still the pinned circuit"
    );
    drop(candidate);
    drop(expected);
    assert!(matches!(
        prepare(&changed, &application),
        Err(AssemblyError::Package(PackageError::ExpectedIdentityMismatch { .. }))
    ));
}

#[test]
fn rust_addition_plan_uses_declared_ports_and_causal_recipes() {
    let manifest = Manifest::parse(manifest_bytes()).unwrap();
    let mut builder = ApplicationBuilder::new(4).unwrap();
    let input = builder.input_state();
    let private = builder.private_inputs().to_vec();
    let mut output = std::array::from_fn(|_| Affine::constant(Goldilocks::ZERO));
    for lane in 0..4 {
        output[lane] = builder
            .affine(Affine::from(input[lane]) + Affine::from(private[lane]))
            .unwrap()
            .into();
    }
    let circuit = builder.finish(output).unwrap();
    let plan = materialized_plan(&circuit, &manifest);
    assert_eq!(plan.private_count, 4);
    assert_eq!(plan.row_count, 8);
    for lane in 0..4 {
        assert_eq!(
            plan.batches[0].recipes[lane],
            json!([2, [0, plan.input_columns[lane]], [0, plan.witness_columns[lane]]])
        );
        assert_eq!(plan.rows[lane].c.terms, vec![(plan.private_start + lane, 1)]);
    }
    manifest.check_dimensions(Counts::of(&circuit)).unwrap();
}

#[test]
fn manifest_rejects_missing_children_changed_roles_profile_and_dimensions() {
    let original: Value = serde_json::from_slice(manifest_bytes()).unwrap();
    let mut missing = original.clone();
    missing["children"].as_array_mut().unwrap().remove(0);
    let mut reordered = original.clone();
    reordered["children"].as_array_mut().unwrap().swap(0, 1);
    let mut role = original.clone();
    role["ports"][0]["role"] = json!("public_input");
    let mut profile = original.clone();
    profile["profile"][2] = json!(18);
    let mut width = original.clone();
    width["geometry"]["logical_width"] = json!([usize::MAX, 1, 1, 0]);
    let mut public = original;
    public["recursive_public"]["digest_port"] = json!("prior_public_input");
    for value in [missing, reordered, role, profile, width, public] {
        assert!(Manifest::parse(&serde_json::to_vec(&value).unwrap()).is_err());
    }
    let manifest = Manifest::parse(manifest_bytes()).unwrap();
    assert!(manifest
        .check_dimensions(Counts {
            witness: usize::MAX,
            local: 0,
            rows: 0
        })
        .is_err());
}
