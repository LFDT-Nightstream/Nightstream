use nightstream::application::{Affine, ApplicationBuilder, ApplicationError};
use nightstream::{Circuit, Fq as F, PrimeCharacteristicRing};

fn scalar(value: u64) -> Affine {
    Affine::constant(F::from_u64(value))
}

#[test]
fn product_assertion_checks_supplied_values_without_generated_words() {
    let mut builder = ApplicationBuilder::new(3).unwrap();
    let values = builder.private_inputs().to_vec();
    builder
        .assert_product(values[0].into(), values[1].into(), values[2].into())
        .unwrap();
    let application = builder
        .finish([values[2].into(), scalar(0), scalar(0), scalar(0)])
        .unwrap();
    assert_eq!(application.row_count(), 5);
    assert!(application.generated_range().is_empty());
    let witness = application
        .execute([F::ZERO; 4], &[F::from_u64(3), F::from_u64(4), F::from_u64(12)])
        .unwrap();
    assert_eq!(witness.output_state()[0], F::from_u64(12));
    assert!(application
        .execute([F::ZERO; 4], &[F::from_u64(3), F::from_u64(4), F::from_u64(13)])
        .is_err());
}

#[test]
fn direct_boolean_row_rejects_non_boolean_values() {
    let mut builder = ApplicationBuilder::new(1).unwrap();
    let bit = builder.private_inputs()[0];
    builder
        .assert_product(bit.into(), Affine::from(bit) - scalar(1), scalar(0))
        .unwrap();
    let application = builder
        .finish([bit.into(), scalar(0), scalar(0), scalar(0)])
        .unwrap();
    for value in [F::ZERO, F::ONE] {
        application.execute([F::ZERO; 4], &[value]).unwrap();
    }
    for value in [F::from_u64(2), -F::ONE] {
        assert!(application.execute([F::ZERO; 4], &[value]).is_err());
    }
    Circuit::compile(application).unwrap();
}

#[test]
fn invalid_product_forms_leave_the_builder_unchanged() {
    let other = ApplicationBuilder::new(20).unwrap();
    let invalid = *other.private_inputs().last().unwrap();
    let mut builder = ApplicationBuilder::new(1).unwrap();
    for position in 0..3 {
        let mut forms = [scalar(1), scalar(1), scalar(1)];
        forms[position] = invalid.into();
        let [a, b, c] = forms;
        assert!(matches!(
            builder.assert_product(a, b, c),
            Err(ApplicationError::VariableOutOfScope(_))
        ));
    }
    let application = builder
        .finish([scalar(0), scalar(0), scalar(0), scalar(0)])
        .unwrap();
    assert_eq!(application.row_count(), 4);
    assert!(application.generated_range().is_empty());
    application.execute([F::ZERO; 4], &[F::ZERO]).unwrap();
}
