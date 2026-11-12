use super::*;

#[test]
fn packed_products_restore_both_sources_and_reject_recommitted_signed_units() {
    let mut builder = ApplicationBuilder::new(2).unwrap();
    let bits = builder.private_inputs().to_vec();
    for &bit in &bits {
        builder
            .assert_product(bit.into(), bit.into(), bit.into())
            .unwrap();
    }
    builder
        .assert_product(
            bits[0].into(),
            Affine::constant(F::ONE) - Affine::from(bits[1]),
            Affine::constant(F::ZERO),
        )
        .unwrap();
    let product = builder.multiply(bits[0].into(), bits[1].into()).unwrap();
    let derived = builder.multiply(product.into(), bits[0].into()).unwrap();
    let state = builder.input_state();
    let application = builder
        .finish([
            Affine::from(state[0]) + Affine::from(derived),
            state[1].into(),
            state[2].into(),
            state[3].into(),
        ])
        .unwrap();
    let packages = packages()
        .with_application(application.records().clone(), 2, application.shape())
        .unwrap();
    for package in [packages.light(), packages.refresh()] {
        assert_eq!(package.packed_product_count(), 2);
        assert_eq!(package.packed_boolean_count(), 4);
        assert_eq!(package.packed_zero_count(), 1);
    }
    let mut prover = BoundedDepth::new(packages.clone(), MINIMUM_SECURITY_BITS).unwrap();
    prover.cap = 1;
    let initial = [F::ZERO; 4];
    let mut envelope = prover.initial(initial).unwrap();
    let mut kinds = Vec::new();
    for iteration in 1..=4 {
        let current = [F::from_u64(iteration), F::ZERO, F::ZERO, F::ZERO];
        envelope = prover
            .extend(&envelope, &[F::ONE, F::ONE], current)
            .unwrap();
        prover
            .verify(iteration, initial, current, &envelope)
            .unwrap();
        kinds.push(envelope.pc);
        let package = if envelope.pc == 1 {
            packages.light()
        } else {
            packages.refresh()
        };
        let layout = package.layout().unwrap();
        let product = package.packed_products().next().unwrap();
        let zero = package.packed_zero_products().next().unwrap();
        for (column, value) in [
            (product[3], F::NEG_ONE),
            (product[4], F::ONE),
            (product[5], F::ZERO),
            (zero[5], F::ZERO),
        ] {
            let (coordinate, kind) = layout.slot(column).unwrap();
            assert_eq!(kind.width(), 1);
            let mut changed = envelope.clone();
            let fresh = changed.fresh.as_mut().unwrap();
            fresh.witness.Z[(coordinate % D, coordinate / D)] = value;
            fresh.claim.c = commit_production_signed_unit_prefix_matrix(&fresh.witness.Z).unwrap();
            let error = prover
                .verify(iteration, initial, current, &changed)
                .unwrap_err();
            assert!(error.to_string().contains("a relation row fails"), "{error}");
        }
    }
    assert_eq!(kinds, [1, 1, 2, 1]);
}
