use nightstream::application::{Affine, ApplicationBuilder, ApplicationError};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

#[test]
fn shared_affine_operations_do_not_expand_the_saved_records() {
    let mut builder = ApplicationBuilder::new(0).unwrap();
    let input = builder.input_state();
    let mut doubled = Affine::from(input[0]);
    let mut shared = Affine::from(input[1]);
    for _ in 0..16 {
        doubled = doubled.clone() + doubled;
        shared = (shared.clone() + Affine::constant(Goldilocks::ONE)) + shared;
    }
    let a = builder.affine(doubled).unwrap();
    let b = builder.affine(shared).unwrap();
    let circuit = builder
        .finish([a.into(), b.into(), input[2].into(), input[3].into()])
        .unwrap();
    assert_eq!(circuit.records().row_header(0).unwrap().term_counts, [1, 0, 1]);
    assert_eq!(circuit.records().row_header(1).unwrap().term_counts, [1, 0, 1]);
    let witness = circuit.execute([Goldilocks::ONE; 4], &[]).unwrap();
    assert_eq!(
        witness.output_state(),
        [65_536, 131_071, 1, 1].map(Goldilocks::from_u64)
    );
}

#[test]
fn rust_defined_addition_uses_the_same_application_interface() {
    let mut builder = ApplicationBuilder::new(4).unwrap();
    let input = builder.input_state();
    let private = builder.private_inputs().to_vec();
    let mut outputs = std::array::from_fn(|_| Affine::constant(Goldilocks::ZERO));
    for lane in 0..4 {
        outputs[lane] = builder
            .affine(Affine::from(input[lane]) + Affine::from(private[lane]))
            .unwrap()
            .into();
    }
    let circuit = builder.finish(outputs).unwrap();
    let input = [1, 2, 3, 4].map(Goldilocks::from_u64);
    let private = [5, 6, 7, 8].map(Goldilocks::from_u64);
    let witness = circuit.execute(input, &private).unwrap();
    assert_eq!(witness.output_state(), [6, 8, 10, 12].map(Goldilocks::from_u64));
    circuit.check(witness.values()).unwrap();
    let mut changed = witness.values().to_vec();
    changed[circuit.output_state()[2].index()] += Goldilocks::ONE;
    assert!(matches!(
        circuit.check(&changed),
        Err(ApplicationError::UnsatisfiedRow(_))
    ));
    assert!(matches!(
        circuit.execute(input, &private[..3]),
        Err(ApplicationError::PrivateInputCount { .. })
    ));
}

#[test]
fn builder_rejects_overflow_unallocated_variables_and_output_dependencies() {
    assert!(matches!(
        ApplicationBuilder::new(usize::MAX),
        Err(ApplicationError::DimensionOverflow)
    ));
    let mut other = ApplicationBuilder::new(0).unwrap();
    let out_of_scope = other.affine(Affine::constant(Goldilocks::ONE)).unwrap();
    let mut builder = ApplicationBuilder::new(0).unwrap();
    assert!(matches!(
        builder.affine(out_of_scope.into()),
        Err(ApplicationError::VariableOutOfScope(index)) if index == out_of_scope.index()
    ));
    assert!(matches!(
        builder.affine(Affine::from(out_of_scope) * Goldilocks::ZERO),
        Err(ApplicationError::VariableOutOfScope(index)) if index == out_of_scope.index()
    ));
    assert!(matches!(
        builder.affine(Affine::from(out_of_scope) - Affine::from(out_of_scope)),
        Err(ApplicationError::VariableOutOfScope(index)) if index == out_of_scope.index()
    ));
    let output = builder.output_state()[0];
    assert!(matches!(
        builder.affine(output.into()),
        Err(ApplicationError::OutputDependency(index)) if index == output.index()
    ));
    assert!(matches!(
        builder.affine(Affine::from(output) * Goldilocks::ZERO),
        Err(ApplicationError::OutputDependency(index)) if index == output.index()
    ));
    assert!(matches!(
        builder.affine(Affine::from(output) - Affine::from(output)),
        Err(ApplicationError::OutputDependency(index)) if index == output.index()
    ));
}

#[test]
fn sealed_rows_keep_raw_terms_and_both_output_lowerings() {
    use nightstream_fprime::{ApplicationForm, ApplicationRecipeNode};
    use p3_field::PrimeField64;
    use std::{ops::ControlFlow, sync::Arc};

    let mut builder = ApplicationBuilder::new(0).unwrap();
    let input = builder.input_state();
    let generated = builder
        .affine(Affine::from(input[0]) + Affine::from(input[1]) + Affine::from(input[0]) * Goldilocks::ZERO)
        .unwrap();
    let circuit = builder
        .finish([
            Affine::from(input[0]) + Affine::from(input[1]) - Affine::from(input[1]),
            Affine::from(input[1]) * Goldilocks::from_u64(2) + Affine::constant(Goldilocks::from_u64(3)),
            Affine::from(input[2]) * Goldilocks::ZERO + Affine::constant(Goldilocks::from_u64(7)),
            generated.into(),
        ])
        .unwrap();
    let clone = circuit.clone();
    assert!(Arc::ptr_eq(circuit.records(), clone.records()));
    assert_eq!(circuit.row_count(), 5);
    assert_eq!(circuit.records().row_header(0).unwrap().term_counts, [3, 0, 1]);
    let mut terms = Vec::new();
    assert!(circuit
        .records()
        .visit_terms(0, |term| {
            if term.form == ApplicationForm::A {
                terms.push((term.variable, term.coefficient));
            }
            Ok(ControlFlow::Continue(()))
        })
        .unwrap()
        .is_continue());
    assert_eq!(terms, [(0, 1), (1, 1), (0, 0)]);
    let mut nodes = Vec::new();
    assert!(circuit
        .records()
        .visit_recipe_nodes(0, |node| {
            nodes.push(node);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap()
        .is_continue());
    use ApplicationRecipeNode::*;
    assert_eq!(
        nodes,
        [Add, Add, Variable(0), Variable(1), Multiply, Constant(0), Variable(0)]
    );
    let witness = clone
        .execute([2, 3, 5, 7].map(Goldilocks::from_u64), &[])
        .unwrap();
    assert_eq!(witness.output_state(), [2, 9, 7, 5].map(Goldilocks::from_u64));
    assert_eq!(
        circuit
            .records()
            .evaluate_recipe(0, |index| Ok(witness.values()[index].as_canonical_u64()))
            .unwrap(),
        5
    );
    circuit.check(witness.values()).unwrap();
}

#[test]
fn assertions_restrict_the_generated_witness() {
    let mut builder = ApplicationBuilder::new(0).unwrap();
    let input = builder.input_state();
    builder
        .assert_equal(input[0].into(), Affine::constant(Goldilocks::ONE))
        .unwrap();
    let circuit = builder.finish(input.map(Affine::from)).unwrap();
    circuit.execute([Goldilocks::ONE; 4], &[]).unwrap();
    assert!(matches!(
        circuit.execute([Goldilocks::ZERO; 4], &[]),
        Err(ApplicationError::UnsatisfiedRow(0))
    ));
}
