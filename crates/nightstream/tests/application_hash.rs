use neo_ccs::crypto::poseidon2::{poseidon2_hash, RATE};
use nightstream::application::{Affine, ApplicationBuilder, ApplicationError};
use nightstream::{Circuit, Fq as F, PrimeCharacteristicRing};

#[test]
fn public_hash_matches_native_hash_at_rate_boundaries() {
    for count in [0, 1, RATE - 1, RATE, RATE + 1, 2 * RATE] {
        let mut builder = ApplicationBuilder::new(count).unwrap();
        let preimage = builder
            .private_inputs()
            .iter()
            .copied()
            .map(Affine::from)
            .collect::<Vec<_>>();
        let digest = builder.poseidon2_hash(&preimage).unwrap();
        let application = builder.finish(digest).unwrap();
        let private = (0..count)
            .map(|index| F::from_usize(index + 1))
            .collect::<Vec<_>>();
        let witness = application.execute([F::ZERO; 4], &private).unwrap();
        assert_eq!(witness.output_state(), poseidon2_hash(&private));
        application.check(witness.values()).unwrap();
        for lane in application.output_state() {
            let mut changed = witness.values().to_vec();
            changed[lane.index()] += F::ONE;
            assert!(application.check(&changed).is_err());
        }
        if count == RATE + 1 {
            Circuit::compile(application).unwrap();
        }
    }
}

#[test]
fn invalid_hash_expressions_do_not_leave_native_blocks() {
    let mut builder = ApplicationBuilder::new(1).unwrap();
    let output = builder.output_state()[0];
    assert!(matches!(
        builder.poseidon2_hash(&[output.into()]),
        Err(ApplicationError::OutputDependency(_))
    ));
    let other = ApplicationBuilder::new(20).unwrap();
    let out_of_scope = *other.private_inputs().last().unwrap();
    assert!(matches!(
        builder.poseidon2_hash(&[out_of_scope.into()]),
        Err(ApplicationError::VariableOutOfScope(_))
    ));
    let message = builder.private_inputs()[0];
    let output = builder.poseidon2_hash(&[message.into()]).unwrap();
    let application = builder.finish(output).unwrap();
    let witness = application.execute([F::ZERO; 4], &[F::ONE]).unwrap();
    assert_eq!(witness.output_state(), poseidon2_hash(&[F::ONE]));
    Circuit::compile(application).unwrap();
}
