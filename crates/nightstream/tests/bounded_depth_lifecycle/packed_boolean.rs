use super::*;

#[test]
fn packed_pairs_are_checked_in_both_fold_relations_and_after_recommitment() {
    let mut builder = ApplicationBuilder::new(32).unwrap();
    let bits = builder.private_inputs().to_vec();
    let state = builder.input_state();
    for &bit in &bits {
        builder
            .assert_product(bit.into(), bit.into(), bit.into())
            .unwrap();
    }
    let application = builder
        .finish([
            Affine::from(state[0]) + Affine::from(bits[0]),
            state[1].into(),
            state[2].into(),
            state[3].into(),
        ])
        .unwrap();
    let packages = packages()
        .with_application(application.records().clone(), 32, application.shape())
        .unwrap();
    let mut prover = BoundedDepth::new(
        &CompiledSteps::new(packages.clone()).unwrap(),
        MINIMUM_SECURITY_BITS,
        crate::engine::Backend::Optimized,
    )
    .unwrap();
    prover.cap = 1;
    let initial = [F::ZERO; 4];
    let message = (0..32)
        .map(|index| F::from_usize(index % 2))
        .collect::<Vec<_>>();
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
        let companion = package.sbox_start() + 150 * package.block_count();
        let (coordinate, kind) = prover.relations[envelope.pc as usize - 1]
            .step()
            .expanded()
            .layout()
            .slot(companion)
            .unwrap();
        assert_eq!(kind.width(), 1);
        let mut changed = envelope.clone();
        let fresh = changed.fresh.as_mut().unwrap();
        assert_eq!(fresh.witness.Z[(coordinate % D, coordinate / D)], F::ONE);
        fresh.witness.Z[(coordinate % D, coordinate / D)] = F::ZERO;
        fresh.claim.c = commit_production_signed_unit_prefix_matrix(&fresh.witness.Z).unwrap();
        let error = prover
            .verify(iteration, initial, initial, &changed)
            .unwrap_err();
        assert!(error.to_string().contains("a relation row fails"), "{error}");
    }
    assert_eq!(kinds, [1, 1, 2, 1]);
}
