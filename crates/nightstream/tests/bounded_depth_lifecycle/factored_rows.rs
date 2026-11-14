//! Factored rows equal the plan rows read through the slot layout (Lean
//! `SlotLayout.layout`).

use super::super::{CompiledSteps, Relation};
use super::support::packages;
use super::MINIMUM_SECURITY_BITS;
use neo_ccs::Mat;
use neo_math::{D, F};
use neo_reductions::superneo_eval::{MatrixRows, SuperneoZBlocks};
use p3_field::PrimeCharacteristicRing;

#[test]
fn factored_rows_match_the_slot_layout_rows() {
    let packages = packages();
    check_factored(&packages);
}

#[test]
fn packed_application_rows_match_the_slot_layout_rows() {
    let mut builder = crate::application::ApplicationBuilder::new(32).unwrap();
    for bit in builder.private_inputs().to_vec() {
        builder
            .assert_product(bit.into(), bit.into(), bit.into())
            .unwrap();
    }
    let state = builder.input_state();
    let application = builder.finish(state.map(Into::into)).unwrap();
    let packages = packages()
        .with_application(application.records().clone(), 32, application.shape())
        .unwrap();
    check_factored(&packages);
}

#[test]
fn native_power_and_affine_application_match_actual_factored_matrices() {
    let mut builder = crate::application::ApplicationBuilder::new(1).unwrap();
    let x = crate::application::Affine::from(builder.private_inputs()[0])
        + crate::application::Affine::constant(F::from_u64(19));
    let x2 = builder.multiply(x.clone(), x.clone()).unwrap();
    let x4 = builder.multiply(x2.into(), x2.into()).unwrap();
    let x6 = builder.multiply(x4.into(), x2.into()).unwrap();
    let x7 = builder.multiply(x6.into(), x).unwrap();
    let copy = builder
        .affine(crate::application::Affine::from(x7) * F::from_u64(3))
        .unwrap();
    let mut state = builder.input_state().map(crate::application::Affine::from);
    state[0] = state[0].clone() + crate::application::Affine::from(copy);
    let application = builder.finish(state).unwrap();
    let packages = packages()
        .with_application(application.records().clone(), 1, application.shape())
        .unwrap();
    for step in CompiledSteps::new(packages.clone()).unwrap().steps() {
        let rows = step.expanded().rows();
        assert_eq!(
            (0..rows.ordinary_row_count())
                .filter(|&row| rows.port(row, 3).next().is_some())
                .count(),
            1
        );
        let relation = Relation::new(step, MINIMUM_SECURITY_BITS).unwrap();
        assert_eq!(relation.params().b(), 2);
        assert_eq!(relation.params().k_rho(), 16);
    }
    check_factored(&packages);
}

#[test]
fn affine_reconstruction_restores_native_output_aliases() {
    let mut builder = crate::application::ApplicationBuilder::new(1).unwrap();
    let input = crate::application::Affine::from(builder.private_inputs()[0]);
    let digest = builder.poseidon2_hash(&[input]).unwrap();
    let copy = builder.affine(digest[0].clone() * F::from_u64(3)).unwrap();
    let mut output = digest;
    output[0] = copy.into();
    let application = builder.finish(output).unwrap();
    let packages = packages()
        .with_application(application.records().clone(), 1, application.shape())
        .unwrap();
    for package in [packages.light(), packages.refresh()] {
        let expanded = package.clone().expand().unwrap();
        let (plan, layout) = (expanded.rows(), expanded.layout());
        let mut logical = (0..layout.width())
            .map(|index| F::from_u64((index % 3) as u64) - F::ONE)
            .collect::<Vec<_>>();
        logical[0] = F::ONE;
        let restored = expanded.source_assignment(&logical).unwrap();
        let source = package.source_rows().unwrap();
        assert_eq!(source.row_count() - plan.row_count(), 1);
        assert_eq!(source.residual(0, &restored), F::ZERO);
        for row in 0..4 {
            for port in 0..3 {
                assert_eq!(
                    source.evaluate(row + 1, port, &restored),
                    layout.evaluate(plan, row, port, &logical).unwrap()
                );
            }
        }
    }
}

pub(super) fn check_factored(packages: &nightstream_fprime::BoundedDepthPackages) {
    for step in CompiledSteps::new(packages.clone())
        .expect("compiled steps")
        .steps()
    {
        let relation = Relation::new(step, MINIMUM_SECURITY_BITS).expect("relation");
        let (rows, layout) = (step.expanded().rows(), step.expanded().layout());
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let logical: Vec<F> = (0..layout.width())
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                F::from_u64((state >> 62) % 3) - F::ONE
            })
            .collect();
        let blocks = relation.blocks();
        let witness = Mat::from_row_major(
            D,
            blocks,
            (0..D)
                .flat_map(|lane| (0..blocks).map(move |block| block * D + lane))
                .map(|column| logical.get(column).copied().unwrap_or(F::ZERO))
                .collect(),
        );
        let witness = SuperneoZBlocks::from_witness_mat(&witness, relation.structure().m()).expect("witness");
        let factored = relation
            .factored()
            .expect("factored rows")
            .expect("a factored form");
        let values = factored.slot_values(&witness);
        for row in 0..rows.row_count() {
            for port in 0..4 {
                assert_eq!(
                    factored.row_value(port, row, &values),
                    layout.evaluate(rows, row, port, &logical).expect("row"),
                    "circuit {} row {row} port {port}",
                    step.expanded().circuit()
                );
            }
        }
    }
}
