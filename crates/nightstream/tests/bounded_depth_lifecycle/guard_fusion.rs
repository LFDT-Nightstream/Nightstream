use super::*;

fn application() -> ApplicationCircuit {
    let mut builder = ApplicationBuilder::new(7).unwrap();
    let inputs = builder.private_inputs().to_vec();
    for &bit in &inputs[..4] {
        builder
            .assert_product(bit.into(), bit.into(), bit.into())
            .unwrap();
    }
    let zero = Affine::constant(F::ZERO);
    let one = Affine::constant(F::ONE);
    builder
        .assert_product(
            Affine::from(inputs[0]) + Affine::from(inputs[1]) + Affine::from(inputs[2]),
            one.clone(),
            one.clone(),
        )
        .unwrap();
    for (index, value) in [5, 7, 11].into_iter().enumerate() {
        builder
            .assert_product(
                inputs[index].into(),
                Affine::from(inputs[4]) + Affine::constant(-F::from_u64(value)),
                zero.clone(),
            )
            .unwrap();
    }
    builder
        .assert_product(
            inputs[3].into(),
            Affine::from(inputs[5]) + Affine::constant(-F::from_u64(17)),
            zero.clone(),
        )
        .unwrap();
    builder
        .assert_product(
            one + Affine::from(inputs[3]) * -F::ONE,
            Affine::from(inputs[6]) + Affine::constant(-F::from_u64(19)),
            zero,
        )
        .unwrap();
    let state = builder.input_state();
    builder.finish(state.map(Into::into)).unwrap()
}

#[test]
fn guarded_application_matches_actual_factored_matrices() {
    let application = application();
    let packages = packages()
        .with_application(application.records().clone(), 7, application.shape())
        .unwrap();
    for package in [packages.light(), packages.refresh()] {
        let expanded = package.clone().expand().unwrap();
        let source = package.source_rows().unwrap();
        assert_eq!(source.row_count() - expanded.rows().row_count(), 6);
    }
    super::factored_rows::check_factored(&packages);
}

#[test]
fn guarded_rows_reject_fresh_recommitted_changes_in_both_relations() {
    let application = application();
    let packages = packages()
        .with_application(application.records().clone(), 7, application.shape())
        .unwrap();
    let mut prover = BoundedDepth::new(
        &CompiledSteps::new(packages.clone()).unwrap(),
        MINIMUM_SECURITY_BITS,
        crate::engine::Backend::Optimized,
    )
    .unwrap();
    prover.cap = 1;
    let initial = [F::ZERO; 4];
    let message = [1, 0, 0, 1, 5, 17, 9].map(F::from_u64);
    let mut envelope = prover.initial(initial).unwrap();
    let mut kinds = Vec::new();
    for iteration in 1..=4 {
        envelope = prover.extend(&envelope, &message, initial, None).unwrap();
        prover
            .verify(iteration, initial, initial, &envelope)
            .unwrap();
        kinds.push(envelope.pc);
        let package = if envelope.pc == 1 {
            packages.light()
        } else {
            packages.refresh()
        };
        let source = package.source_rows().unwrap();
        let input = source
            .port(5, 1)
            .find(|(column, _)| *column != 0)
            .unwrap()
            .0;
        let (coordinate, kind) = prover.relations[envelope.pc as usize - 1]
            .step()
            .expanded()
            .layout()
            .slot(input)
            .unwrap();
        assert_eq!(kind.width(), 41);
        let mut changed = envelope.clone();
        let fresh = changed.fresh.as_mut().unwrap();
        let digit = &mut fresh.witness.Z[(coordinate % D, coordinate / D)];
        *digit = if *digit == F::ZERO { F::ONE } else { F::ZERO };
        fresh.claim.c = commit_production_signed_unit_prefix_matrix(&fresh.witness.Z).unwrap();
        let error = prover
            .verify(iteration, initial, initial, &changed)
            .unwrap_err();
        assert!(error.to_string().contains("a relation row fails"), "{error}");
    }
    assert_eq!(kinds, [1, 1, 2, 1]);
}
