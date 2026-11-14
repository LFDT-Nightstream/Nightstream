use std::{path::PathBuf, sync::Arc};

use p3_field::PrimeField64;

use super::*;
use crate::{
    load_bounded_depth_packages, ApplicationForm, ApplicationRecordsWriter, ApplicationRowHeader, ApplicationTerm,
    BoundedDepthPackages, NativeShape, BOUNDED_DEPTH_PACKAGES_IDENTITY,
};

const PRIVATE: usize = 7;
const INPUT: usize = super::super::application::WITNESS_START;

fn packages() -> BoundedDepthPackages {
    let artifacts = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../formal/nightstream-fprime/artifacts");
    load_bounded_depth_packages(
        &std::fs::read(artifacts.join("nightstream-fprime-bounded-depth-light-v1.json")).unwrap(),
        &std::fs::read(artifacts.join("nightstream-fprime-bounded-depth-refresh-v1.json")).unwrap(),
        BOUNDED_DEPTH_PACKAGES_IDENTITY,
    )
    .unwrap()
}

fn record_rows(rows: &[Row]) -> Arc<crate::ApplicationRecords> {
    let mut writer = ApplicationRecordsWriter::new().unwrap();
    for row in rows {
        let sides = [&row.a, &row.b, &row.c];
        writer
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
    Arc::new(writer.finish().unwrap())
}

fn chain() -> Vec<Row> {
    let x = unit(4).plus(Linear {
        constant: F::from_u64(19),
        terms: vec![],
    });
    vec![
        Row {
            a: x.clone(),
            b: x.clone(),
            c: unit(5),
        },
        Row {
            a: unit(5),
            b: unit(5),
            c: unit(6),
        },
        Row {
            a: unit(6),
            b: unit(5),
            c: unit(7),
        },
        Row {
            a: unit(7),
            b: x,
            c: unit(8),
        },
    ]
}

fn native(rows: &[Row]) -> BoundedDepthPackages {
    packages()
        .with_application(record_rows(rows), PRIVATE, &NativeShape::default())
        .unwrap()
}

#[test]
fn native_power_and_affine_rows_reconstruct_exact_source_constraints() {
    let mut rows = chain();
    rows.push(Row {
        a: unit(8).scale(F::from_u64(3)),
        b: Linear {
            constant: F::ONE,
            terms: vec![],
        },
        c: unit(9),
    });
    rows.push(Row {
        a: unit(9),
        b: unit(4),
        c: unit(10),
    });
    let native = native(&rows);
    for package in [native.light(), native.refresh()] {
        assert_eq!(package.field_lowering.powers.len(), 1);
        assert_eq!(package.field_lowering.affine.len(), 1);
        let expanded = package.clone().expand().unwrap();
        let (plan, layout) = (expanded.rows(), expanded.layout());
        let source = package.source_rows().unwrap();
        assert_eq!(source.row_count() - plan.row_count(), 4);
        for column in INPUT + 1..=INPUT + 3 {
            assert!(layout.slot(column).is_none());
        }
        assert!(layout.slot(INPUT + 5).is_none());
        for x in [F::ZERO, F::ONE, F::NEG_ONE, F::from_u64(123456)] {
            let input = x + F::from_u64(19);
            let mut columns = vec![F::ZERO; package.source_end()];
            columns[0] = F::ONE;
            columns[INPUT] = x;
            columns[INPUT + 4] = input.exp_u64(7);
            columns[INPUT + 6] = F::from_u64(3) * input.exp_u64(7) * x;
            package.field_lowering.reconstruct(&mut columns).unwrap();
            assert!((0..6).all(|row| source.residual(row, &columns) == F::ZERO));
            assert!((0..2).all(|row| plan.residual(row, &columns) == F::ZERO));
            let logical = layout.assignment(&columns).unwrap();
            assert!(logical
                .iter()
                .all(|value| [-F::ONE, F::ZERO, F::ONE].contains(value)));
            let restored = expanded.source_assignment(&logical).unwrap();
            assert!((0..6).all(|row| source.residual(row, &restored) == F::ZERO));
            for row in 0..2 {
                for port in 0..4 {
                    assert_eq!(
                        plan.evaluate(row, port, &columns),
                        layout.evaluate(plan, row, port, &logical).unwrap()
                    );
                }
            }
            columns[INPUT + 4] += F::ONE;
            columns[INPUT + 6] = F::from_u64(3) * columns[INPUT + 4] * x;
            package.field_lowering.reconstruct(&mut columns).unwrap();
            assert_ne!(plan.residual(0, &columns), F::ZERO);
            assert_ne!(source.residual(3, &columns), F::ZERO);
        }
    }
}

#[test]
fn wrong_chain_rows_and_external_intermediate_uses_cannot_disappear() {
    let base = packages();
    for edited in 0..4 {
        let mut rows = chain();
        rows[edited].c.constant += F::ONE;
        let native = base
            .with_application(record_rows(&rows), PRIVATE, &NativeShape::default())
            .unwrap();
        for package in [native.light(), native.refresh()] {
            assert!(package.field_lowering.powers.is_empty());
        }
    }
    for intermediate in 5..=7 {
        let mut rows = chain();
        rows.push(Row {
            a: unit(intermediate),
            b: unit(4),
            c: unit(10),
        });
        let native = base
            .with_application(record_rows(&rows), PRIVATE, &NativeShape::default())
            .unwrap();
        assert!(native.light().field_lowering.powers.is_empty());
        assert!(native
            .light()
            .layout()
            .unwrap()
            .slot(INPUT + intermediate - 4)
            .is_some());
    }
}

#[test]
fn persistent_and_native_piece_reads_protect_removed_columns() {
    let compiled = native(&chain());
    for native in [compiled.light(), compiled.refresh()] {
        let mut package = native.clone();
        package.field_lowering = FieldLowering::default();
        package.slots = package.rule_slots(&[], &[]);
        package.pieces.push(Piece::Rows(vec![Row {
            a: unit(INPUT + 1),
            b: unit(INPUT),
            c: unit(INPUT + 6),
        }]));
        assert!(FieldLowering::checked(&package).unwrap().powers.is_empty());
        package.pieces.pop();
        package.pieces.push(Piece::Block {
            start: package.application.local_start,
            lanes: vec![unit(INPUT + 1); 16],
        });
        assert!(FieldLowering::checked(&package).unwrap().powers.is_empty());
    }
    let mut rows = chain();
    for row in &mut rows {
        for side in [&mut row.a, &mut row.b, &mut row.c] {
            for (column, _) in &mut side.terms {
                if *column == 5 {
                    *column = 0;
                }
            }
        }
    }
    assert!(native(&rows).light().field_lowering.powers.is_empty());
}
