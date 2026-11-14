use std::{path::PathBuf, sync::Arc};

use super::*;
use crate::{
    load_bounded_depth_packages, ApplicationForm, ApplicationRecordsWriter, ApplicationRowHeader, ApplicationTerm,
    BoundedDepthPackages, NativeShape, BOUNDED_DEPTH_PACKAGES_IDENTITY,
};

fn variable(column: usize) -> Linear {
    Linear {
        constant: F::ZERO,
        terms: vec![(column, F::ONE)],
    }
}

fn constant(value: i64) -> Linear {
    Linear {
        constant: F::from_i64(value),
        terms: Vec::new(),
    }
}

fn guarded(gate: Linear, residual: Linear) -> Row {
    Row {
        a: gate,
        b: residual,
        c: Linear::default(),
    }
}

fn boolean(column: usize) -> Row {
    guarded(variable(column), variable(column).plus(constant(-1)))
}

fn packages(rows: &[Row]) -> BoundedDepthPackages {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../formal/nightstream-fprime/artifacts");
    let base = load_bounded_depth_packages(
        &std::fs::read(root.join("nightstream-fprime-bounded-depth-light-v1.json")).unwrap(),
        &std::fs::read(root.join("nightstream-fprime-bounded-depth-refresh-v1.json")).unwrap(),
        BOUNDED_DEPTH_PACKAGES_IDENTITY,
    )
    .unwrap();
    let mut records = ApplicationRecordsWriter::new().unwrap();
    for row in rows {
        let sides = [&row.a, &row.b, &row.c];
        records
            .append_row(
                ApplicationRowHeader {
                    constants: sides.map(|side| side.constant.as_canonical_u64()),
                    term_counts: sides.map(|side| side.terms.len()),
                },
                [ApplicationForm::A, ApplicationForm::B, ApplicationForm::C]
                    .into_iter()
                    .zip(sides)
                    .flat_map(|(form, side)| {
                        side.terms.iter().map(move |&(variable, coefficient)| {
                            Ok(ApplicationTerm {
                                form,
                                variable,
                                coefficient: coefficient.as_canonical_u64(),
                            })
                        })
                    }),
            )
            .unwrap();
    }
    base.with_application(Arc::new(records.finish().unwrap()), 12, &NativeShape::default())
        .unwrap()
}

fn fixture() -> Vec<Row> {
    let mut rows = [4, 5, 6, 9].map(boolean).to_vec();
    rows.push(guarded(
        variable(4)
            .plus(variable(5))
            .plus(variable(6))
            .plus(constant(-1)),
        one(),
    ));
    for (gate, value) in [(4, 3), (5, 5), (6, -9)] {
        rows.push(guarded(
            variable(gate).scale(F::from_u64(7)),
            variable(7)
                .plus(variable(8).scale(F::TWO))
                .plus(constant(value))
                .scale(F::from_u64(11)),
        ));
    }
    rows.push(guarded(variable(9), variable(10).plus(constant(-7))));
    rows.push(guarded(
        one().plus(variable(9).scale(-F::ONE)),
        variable(11).plus(constant(13)),
    ));
    rows
}

#[test]
fn guarded_rows_preserve_source_and_all_four_matrix_ports() {
    let rows = fixture();
    let packages = packages(&rows);
    for package in [packages.light(), packages.refresh()] {
        let fusion = &package.guard_fusion;
        assert_eq!(fusion.groups.len(), 2);
        assert_eq!(fusion.removed.len(), 3);
        assert_eq!(
            fusion
                .groups
                .iter()
                .filter(|group| group.row.a == one())
                .count(),
            1
        );
        let expanded = package.clone().expand().unwrap();
        let (plan, layout) = (expanded.rows(), expanded.layout());
        let source = package.source_rows().unwrap();
        assert_eq!(source.piece_ranges()[0].len(), rows.len());
        for (active, value) in [(0, 3), (1, 5), (2, -9)] {
            for gate in [0, 1] {
                let mut columns = vec![F::ZERO; package.source_end()];
                columns[0] = F::ONE;
                let first = super::super::application::WITNESS_START;
                columns[first + active] = F::ONE;
                columns[first + 3] = F::from_i64(-value - 34);
                columns[first + 4] = F::from_u64(17);
                columns[first + 5] = F::from_u64(gate);
                columns[first + 6] = F::from_u64(if gate == 1 { 7 } else { 123 });
                columns[first + 7] = F::from_i64(if gate == 0 { -13 } else { 98 });
                for (offset, &column) in package.packed_booleans.sources.iter().enumerate() {
                    columns[package.packed_booleans.start + offset] = F::ONE - columns[column];
                }
                assert!(source.piece_ranges()[0]
                    .clone()
                    .all(|row| source.residual(row, &columns) == F::ZERO));
                assert!(plan.piece_ranges()[0]
                    .clone()
                    .all(|row| plan.residual(row, &columns) == F::ZERO));
                let logical = layout.assignment(&columns).unwrap();
                let restored = expanded.source_assignment(&logical).unwrap();
                assert!(source.piece_ranges()[0]
                    .clone()
                    .all(|row| source.residual(row, &restored) == F::ZERO));
                for row in plan.piece_ranges()[0].clone() {
                    for port in 0..4 {
                        assert_eq!(
                            plan.evaluate(row, port, &columns),
                            layout.evaluate(plan, row, port, &logical).unwrap()
                        );
                    }
                }
                columns[first + 3] += F::ONE;
                assert!(plan.piece_ranges()[0]
                    .clone()
                    .any(|row| plan.residual(row, &columns) != F::ZERO));
                assert!(source.piece_ranges()[0]
                    .clone()
                    .any(|row| source.residual(row, &columns) != F::ZERO));
            }
        }
    }
}

#[test]
fn missing_certificates_keep_only_complementary_pairs() {
    let mut source = fixture();
    source.remove(0);
    let missing = packages(&source);
    assert!(missing
        .light()
        .guard_fusion
        .groups
        .iter()
        .all(|group| group.sources.len() == 2));
}

#[test]
fn uncertified_and_overlapping_guards_cannot_be_fused() {
    let unproved = packages(&[
        guarded(variable(4), variable(5)),
        guarded(one().plus(variable(4).scale(-F::ONE)), variable(6)),
    ]);
    assert!(unproved.light().guard_fusion.groups.is_empty());
    let mut rows = [4, 5, 6].map(boolean).to_vec();
    rows.push(guarded(
        variable(4)
            .plus(variable(5))
            .plus(variable(6))
            .plus(constant(-1)),
        one(),
    ));
    rows.push(guarded(variable(4).plus(variable(5)), variable(7).plus(constant(1))));
    rows.push(guarded(variable(5).plus(variable(6)), variable(7).plus(constant(2))));
    let overlapping = packages(&rows);
    assert!(overlapping.light().guard_fusion.groups.is_empty());
}

#[test]
fn nonboolean_gate_needs_the_retained_boolean_check() {
    let rows = [
        boolean(4),
        guarded(variable(4), variable(5)),
        guarded(one().plus(variable(4).scale(-F::ONE)), variable(6)),
    ];
    let packages = packages(&rows);
    let package = packages.light();
    assert_eq!(package.guard_fusion.removed.len(), 1);
    let first = super::super::application::WITNESS_START;
    let mut columns = vec![F::ZERO; package.source_end()];
    columns[0] = F::ONE;
    columns[first] = F::TWO;
    columns[first + 1] = F::ONE;
    columns[first + 2] = F::TWO;
    columns[package.packed_booleans.start] = -F::ONE;
    let plan = package.plan_rows().unwrap();
    let source = package.source_rows().unwrap();
    assert!(plan.piece_ranges()[0]
        .clone()
        .all(|row| plan.residual(row, &columns) == F::ZERO));
    assert_ne!(source.residual(0, &columns), F::ZERO);
    let layout = package.layout().unwrap();
    let (position, kind) = layout.slot(first).unwrap();
    assert_eq!(kind, super::super::layout::Kind::Bit);
    assert_eq!(layout.assignment(&columns).unwrap()[position], F::TWO);
    assert!(![-F::ONE, F::ZERO, F::ONE].contains(&columns[first]));
}

#[test]
fn subset_complements_remain_valid_when_the_family_total_is_zero() {
    let mut rows = [4, 5, 6, 7].map(boolean).to_vec();
    rows.push(guarded(
        variable(4)
            .plus(variable(5))
            .plus(variable(6))
            .plus(variable(7).scale(-F::ONE)),
        one(),
    ));
    let gate = variable(4).plus(variable(5));
    rows.push(guarded(gate.clone(), variable(8).plus(constant(-3))));
    rows.push(guarded(one().plus(gate.scale(-F::ONE)), variable(9).plus(constant(-5))));
    let packages = packages(&rows);
    let package = packages.light();
    assert_eq!(package.guard_fusion.removed.len(), 1);
    let plan = package.plan_rows().unwrap();
    let source = package.source_rows().unwrap();
    let first = super::super::application::WITNESS_START;
    for active in 0..4 {
        let mut columns = vec![F::ZERO; package.source_end()];
        columns[0] = F::ONE;
        if active < 3 {
            columns[first + active] = F::ONE;
            columns[first + 3] = F::ONE;
        }
        columns[first + 4] = F::from_u64(if active < 2 { 3 } else { 91 });
        columns[first + 5] = F::from_u64(if active >= 2 { 5 } else { 88 });
        for (offset, &column) in package.packed_booleans.sources.iter().enumerate() {
            columns[package.packed_booleans.start + offset] = F::ONE - columns[column];
        }
        assert!(plan.piece_ranges()[0]
            .clone()
            .all(|row| plan.residual(row, &columns) == F::ZERO));
        assert!(source.piece_ranges()[0]
            .clone()
            .all(|row| source.residual(row, &columns) == F::ZERO));
    }
}

#[test]
fn a_family_certificate_cannot_certify_its_own_replacement() {
    let sum = variable(4).plus(variable(5));
    let rows = [
        boolean(4),
        boolean(5),
        guarded(sum.clone().plus(constant(-1)), one()),
        guarded(sum, variable(6)),
    ];
    let packages = packages(&rows);
    for package in [packages.light(), packages.refresh()] {
        let plan = package.plan_rows().unwrap();
        let source = package.source_rows().unwrap();
        let first = super::super::application::WITNESS_START;
        let mut columns = vec![F::ZERO; package.source_end()];
        columns[0] = F::ONE;
        columns[first] = F::ONE;
        columns[first + 1] = F::ONE;
        columns[first + 2] = F::TWO.inverse();
        let columns = package.relation_columns(columns, &plan).unwrap();
        let logical = package.layout().unwrap().assignment(&columns).unwrap();
        assert!(logical
            .iter()
            .all(|value| [F::NEG_ONE, F::ZERO, F::ONE].contains(value)));
        assert!(source.piece_ranges()[0]
            .clone()
            .any(|row| source.residual(row, &columns) != F::ZERO));
        assert!(plan.piece_ranges()[0]
            .clone()
            .any(|row| plan.residual(row, &columns) != F::ZERO));
    }
}
