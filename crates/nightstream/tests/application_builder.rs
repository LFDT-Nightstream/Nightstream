use nightstream::application::{Affine, ApplicationBuilder, ApplicationError};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

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
    assert_eq!(
        builder.affine(out_of_scope.into()),
        Err(ApplicationError::VariableOutOfScope(out_of_scope.index()))
    );
    assert_eq!(
        builder.affine(Affine::from(out_of_scope) * Goldilocks::ZERO),
        Err(ApplicationError::VariableOutOfScope(out_of_scope.index()))
    );
    assert_eq!(
        builder.affine(Affine::from(out_of_scope) - Affine::from(out_of_scope)),
        Err(ApplicationError::VariableOutOfScope(out_of_scope.index()))
    );
    let output = builder.output_state()[0];
    assert_eq!(
        builder.affine(output.into()),
        Err(ApplicationError::OutputDependency(output.index()))
    );
    assert_eq!(
        builder.affine(Affine::from(output) * Goldilocks::ZERO),
        Err(ApplicationError::OutputDependency(output.index()))
    );
    assert_eq!(
        builder.affine(Affine::from(output) - Affine::from(output)),
        Err(ApplicationError::OutputDependency(output.index()))
    );
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
