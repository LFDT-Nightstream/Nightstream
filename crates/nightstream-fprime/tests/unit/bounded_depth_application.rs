//! A native application replaces Lean's only inside the region that Lean
//! writes: a region that claims a later gadget's piece, or fewer locals than
//! the application's pieces read, is refused. The identity binds the
//! application's rows.

use std::path::PathBuf;
use std::sync::Arc;

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use crate::application_records::{
    ApplicationForm, ApplicationRecipeNode, ApplicationRecords, ApplicationRecordsWriter, ApplicationRowHeader,
    ApplicationTerm,
};
use crate::bounded_depth::layout::Kind;
use crate::bounded_depth::{
    load_bounded_depth_packages, parse_atoms, BoundedDepthPackages, Linear, NativeBlock, NativeShape, Piece,
    StepPackage, BOUNDED_DEPTH_PACKAGES_IDENTITY,
};
use crate::package::PackageError;

fn artifact(name: &str) -> Vec<u8> {
    std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../formal/nightstream-fprime/artifacts")
            .join(name),
    )
    .unwrap_or_else(|error| panic!("read {name}: {error}"))
}

/// `(current_k + scale · witness_k) · 1 = next_k`.
fn addition(scale: u64) -> Arc<ApplicationRecords> {
    let mut writer = ApplicationRecordsWriter::new().expect("writer");
    let term = |form, variable, coefficient| {
        Ok(ApplicationTerm {
            form,
            variable,
            coefficient,
        })
    };
    for lane in 0..4 {
        let header = ApplicationRowHeader {
            constants: [0, 1, 0],
            term_counts: [2, 0, 1],
        };
        let terms = [
            term(ApplicationForm::A, lane, 1),
            term(ApplicationForm::A, 4 + lane, scale),
            term(ApplicationForm::C, 8 + lane, 1),
        ];
        writer.append_row(header, terms).expect("row");
    }
    Arc::new(writer.finish().expect("records"))
}

fn packages() -> BoundedDepthPackages {
    load_bounded_depth_packages(
        &artifact("nightstream-fprime-bounded-depth-light-v1.json"),
        &artifact("nightstream-fprime-bounded-depth-refresh-v1.json"),
        BOUNDED_DEPTH_PACKAGES_IDENTITY,
    )
    .expect("bounded-depth packages")
}

#[test]
fn substitution_binds_the_application_and_refuses_a_wrong_region() {
    let packages = packages();
    let identity = |scale| {
        packages
            .with_application(addition(scale), 4, &NativeShape::default())
            .expect("Lean's region holds the application")
            .identity()
    };
    assert_ne!(identity(1), packages.identity());
    assert_ne!(identity(1), identity(2));

    let mut later_piece = packages.clone();
    later_piece.light.application.pieces += 1;
    let mut fewer_locals = packages.clone();
    fewer_locals.refresh.application.locals -= 1;
    for wrong in [later_piece, fewer_locals] {
        assert!(matches!(
            wrong.with_application(addition(1), 4, &NativeShape::default()),
            Err(PackageError::Malformed(
                "bounded-depth application piece reads past its region"
            ))
        ));
    }
}

const FIRST_LOCAL: usize = 12;
const BLOCK: usize = 1096;

/// Four witness words and `locals` locals, each its own recipe
/// `local · 1 = previous` from witness word 0; then `output_k · 1 = read`,
/// and `bit · bit = bit` for each of `bits`.
fn chained(locals: usize, read: usize, bits: &[usize]) -> Arc<ApplicationRecords> {
    let mut writer = ApplicationRecordsWriter::new().expect("writer");
    let copy = |writer: &mut ApplicationRecordsWriter, from: usize, to: usize| {
        let header = ApplicationRowHeader {
            constants: [0, 1, 0],
            term_counts: [1, 0, 1],
        };
        let term = |form, variable| {
            Ok(ApplicationTerm {
                form,
                variable,
                coefficient: 1,
            })
        };
        writer
            .append_row(header, [term(ApplicationForm::A, from), term(ApplicationForm::C, to)])
            .expect("row")
    };
    for local in FIRST_LOCAL..FIRST_LOCAL + locals {
        let row = copy(&mut writer, if local == FIRST_LOCAL { 4 } else { local - 1 }, local);
        writer
            .append_recipe(row, [Ok(ApplicationRecipeNode::Variable(local - 1))])
            .expect("recipe");
    }
    for lane in 0..4 {
        copy(&mut writer, read, 8 + lane);
    }
    for &bit in bits {
        let header = ApplicationRowHeader {
            constants: [0, 0, 0],
            term_counts: [1, 1, 1],
        };
        let term = |form| {
            Ok(ApplicationTerm {
                form,
                variable: bit,
                coefficient: 1,
            })
        };
        writer
            .append_row(
                header,
                [
                    term(ApplicationForm::A),
                    term(ApplicationForm::B),
                    term(ApplicationForm::C),
                ],
            )
            .expect("row");
    }
    Arc::new(writer.finish().expect("records"))
}

fn block(first_local: usize, lane: usize) -> NativeBlock {
    NativeBlock {
        first_local,
        lanes: std::array::from_fn(|_| Linear {
            constant: Goldilocks::ZERO,
            terms: vec![(lane, Goldilocks::ONE)],
        }),
    }
}

#[test]
fn native_blocks_replace_their_rows_and_bits_take_one_coordinate() {
    let packages = packages();
    let records = chained(2 * BLOCK, FIRST_LOCAL + 2 * BLOCK - 1, &[5, 0, FIRST_LOCAL + 1080]);
    let shape = NativeShape {
        blocks: vec![block(FIRST_LOCAL, 4), block(FIRST_LOCAL + BLOCK, 5)],
    };
    let plain = packages
        .with_application(Arc::clone(&records), 4, &NativeShape::default())
        .expect("rows in place");
    let packed = packages
        .with_application(Arc::clone(&records), 4, &shape)
        .expect("blocks in place");
    for (plain, packed, lean) in [
        (plain.light(), packed.light(), packages.light()),
        (plain.refresh(), packed.refresh(), packages.refresh()),
    ] {
        assert_eq!(packed.application.pieces, 3);
        assert!(matches!(&packed.pieces[0], Piece::Rows(rows) if rows.len() == 7));
        assert_eq!(packed.block_count(), lean.block_count() - 5 + 2);
        assert_eq!(plain.block_count(), lean.block_count() - 5);
        let layout = packed.layout().expect("layout");
        let kind = |variable| {
            layout
                .slot(packed.variable_column(variable).expect("column"))
                .map(|(_, kind)| kind)
        };
        assert_eq!(kind(FIRST_LOCAL), None);
        assert_eq!(kind(FIRST_LOCAL + BLOCK + 1079), None);
        assert_eq!(kind(FIRST_LOCAL + 1080), None);
        assert_eq!(kind(FIRST_LOCAL + 1081), None);
        assert_eq!(kind(5), Some(Kind::Bit));
        assert_eq!(kind(4), Some(Kind::Field));
        assert_eq!(kind(0), Some(Kind::Field));
        assert_eq!(kind(6), None);
        assert_eq!(packed.packed_booleans.sources, vec![packed.variable_column(5).unwrap()]);
        for column in [packed.sbox_start(), packed.sbox_start() + 299] {
            assert_eq!(layout.slot(column).map(|(_, kind)| kind), Some(Kind::Field));
        }
        let plain_layout = plain.layout().expect("layout");
        for (local, expected) in [
            (FIRST_LOCAL + 1079, Kind::Bit),
            (FIRST_LOCAL + 1080, Kind::Bit),
            (FIRST_LOCAL + 2 * BLOCK - 1, Kind::Bit),
        ] {
            assert_eq!(
                plain_layout
                    .slot(plain.variable_column(local).unwrap())
                    .unwrap()
                    .1,
                expected
            );
        }
        for local in FIRST_LOCAL..FIRST_LOCAL + 1079 {
            assert!(plain_layout
                .slot(plain.variable_column(local).unwrap())
                .is_none());
        }
        assert_eq!(
            layout.width() - packed.packed_booleans.sources.len() + (2 * BLOCK - 1079),
            plain_layout.width() - plain.packed_booleans.sources.len() + 41 * 2 * 150
        );
        assert_eq!(
            packed.source_rows().expect("rows").row_count() + 2 * BLOCK,
            plain.source_rows().expect("rows").row_count() + 2 * 150
        );
    }
    let one_lane = NativeShape {
        blocks: vec![block(FIRST_LOCAL, 4), block(FIRST_LOCAL + BLOCK, 6)],
    };
    let one_lane = packages
        .with_application(Arc::clone(&records), 4, &one_lane)
        .expect("in place")
        .identity();
    let distinct = [plain.identity(), packed.identity(), one_lane];
    for (index, left) in distinct.iter().enumerate() {
        assert!(distinct[index + 1..].iter().all(|right| right != left));
    }
}

/// Each package that Lean writes has the slots of Lean's rule
/// (`StepPackage.runsOf`) on its own pieces, with its `bit` hints as the bit
/// candidates: Rust's rule is Lean's.
#[test]
fn lean_packages_follow_the_slot_rule() {
    for name in [
        "light",
        "refresh",
        "finish",
        "nebula-v1-light",
        "nebula-v1-refresh",
        "nebula-n1-light",
        "nebula-n1-refresh",
    ] {
        let bytes = artifact(&format!("nightstream-fprime-bounded-depth-{name}-v1.json"));
        let package = StepPackage::decode(&parse_atoms(&bytes).expect("atoms")).expect("package");
        assert_eq!(package.rule_slots(&package.bit_columns()), package.slots, "{name}");
    }
}

/// Blocks that overlap or pass the locals, a lane that reads its own block
/// or a dead local, and a kept row that reads a dead local are refused.
#[test]
fn native_shape_outside_its_rules_is_refused() {
    let packages = packages();
    let records = chained(2 * BLOCK, FIRST_LOCAL + 2 * BLOCK - 1, &[]);
    let two = |first: NativeBlock, second: NativeBlock| NativeShape {
        blocks: vec![first, second],
    };
    let cases = [
        (
            &records,
            two(block(FIRST_LOCAL, 4), block(FIRST_LOCAL + 1000, 4)),
            "bounded-depth native block outside the free locals",
        ),
        (
            &records,
            two(block(FIRST_LOCAL, 4), block(FIRST_LOCAL + BLOCK + 1, 4)),
            "bounded-depth native block outside the free locals",
        ),
        (
            &records,
            two(block(FIRST_LOCAL, FIRST_LOCAL), block(FIRST_LOCAL + BLOCK, 4)),
            "bounded-depth native lane reads past its inputs",
        ),
        (
            &records,
            two(block(FIRST_LOCAL, 4), block(FIRST_LOCAL + BLOCK, FIRST_LOCAL + 5)),
            "bounded-depth native lane reads past its inputs",
        ),
    ];
    for (records, shape, expected) in cases {
        let refused = packages.with_application(Arc::clone(records), 4, &shape);
        assert!(
            matches!(refused, Err(PackageError::Malformed(message)) if message == expected),
            "{expected}: {:?}",
            refused.map(|packages| packages.identity())
        );
    }
    let reads_dead = chained(2 * BLOCK, FIRST_LOCAL + 7, &[]);
    assert!(matches!(
        packages.with_application(
            reads_dead,
            4,
            &NativeShape {
                blocks: vec![block(FIRST_LOCAL, 4)]
            }
        ),
        Err(PackageError::Malformed(
            "bounded-depth application row reads a dead local"
        ))
    ));
}

#[test]
fn inferred_application_bits_keep_the_compact_layout() {
    let records = |explicit: bool| {
        let mut writer = ApplicationRecordsWriter::new().unwrap();
        let mut row = |a, b, c| {
            writer
                .append_row(
                    ApplicationRowHeader {
                        constants: [0; 3],
                        term_counts: [1; 3],
                    },
                    [
                        (ApplicationForm::A, a),
                        (ApplicationForm::B, b),
                        (ApplicationForm::C, c),
                    ]
                    .into_iter()
                    .map(|(form, variable)| {
                        Ok(ApplicationTerm {
                            form,
                            variable,
                            coefficient: 1,
                        })
                    }),
                )
                .unwrap();
        };
        row(4, 4, 4);
        row(5, 5, 5);
        if explicit {
            row(6, 6, 6);
        }
        row(4, 5, 6);
        writer
            .append_row(
                ApplicationRowHeader {
                    constants: [0, 1, 0],
                    term_counts: [1, 1, 0],
                },
                [
                    (ApplicationForm::A, 4, 1),
                    (ApplicationForm::B, 5, Goldilocks::NEG_ONE.as_canonical_u64()),
                ]
                .into_iter()
                .map(|(form, variable, coefficient)| {
                    Ok(ApplicationTerm {
                        form,
                        variable,
                        coefficient,
                    })
                }),
            )
            .unwrap();
        Arc::new(writer.finish().unwrap())
    };
    let base = packages();
    let explicit = base
        .with_application(records(true), 3, &NativeShape::default())
        .unwrap();
    let inferred = base
        .with_application(records(false), 3, &NativeShape::default())
        .unwrap();
    for (explicit, inferred) in [
        (&explicit.light, &inferred.light),
        (&explicit.refresh, &inferred.refresh),
    ] {
        let explicit_layout = explicit.layout().unwrap();
        let inferred_layout = inferred.layout().unwrap();
        for variable in 4..7 {
            let column = explicit.variable_column(variable).unwrap();
            assert_eq!(explicit_layout.slot(column), inferred_layout.slot(column));
        }
        assert_eq!(explicit_layout.width(), inferred_layout.width() + 1);
        assert_eq!(
            explicit.plan_rows().unwrap().row_count(),
            inferred.plan_rows().unwrap().row_count()
        );
        assert_eq!(
            explicit.source_rows().unwrap().row_count(),
            inferred.source_rows().unwrap().row_count() + 1
        );
        let output = inferred.variable_column(6).unwrap();
        assert_eq!(inferred.layout().unwrap().slot(output).unwrap().1, Kind::Bit);
        let mut old_bits = inferred.bit_columns();
        old_bits.extend([
            inferred.variable_column(4).unwrap(),
            inferred.variable_column(5).unwrap(),
            output,
        ]);
        let old_slots = inferred.rule_slots(&old_bits);
        assert_ne!(old_slots, inferred.slots);
        let mut shared = vec![Goldilocks::ZERO; inferred.application.local_start];
        shared[0] = Goldilocks::ONE;
        for variable in 4..7 {
            shared[inferred.variable_column(variable).unwrap()] = Goldilocks::ONE;
        }
        let columns = inferred.circuit_columns(&shared).unwrap();
        let plan = inferred.plan_rows().unwrap();
        let source = inferred.source_rows().unwrap();
        assert_eq!(
            inferred.relation_columns(columns.clone(), &plan).unwrap(),
            inferred.relation_columns(columns, &source).unwrap()
        );
    }
}

#[test]
fn empty_and_unmatched_applications_keep_the_existing_slot_rule() {
    let base = packages();
    for (records, count) in [
        (Arc::new(ApplicationRecordsWriter::new().unwrap().finish().unwrap()), 0),
        (addition(1), 4),
    ] {
        let package = base
            .with_application(records, count, &NativeShape::default())
            .unwrap();
        for step in [&package.light, &package.refresh] {
            let mut bits = step.bit_columns();
            bits.extend((0..count).map(|index| step.variable_column(4 + index).unwrap()));
            assert_eq!(step.slots, step.rule_slots(&bits));
        }
    }
}

#[test]
fn source_and_packed_row_orders_fill_the_same_fresh_columns() {
    let native = packages()
        .with_application(chained(0, 4, &[4, 5, 6]), 4, &NativeShape::default())
        .unwrap();
    for package in [native.light(), native.refresh()] {
        let mut shared = vec![Goldilocks::ZERO; package.application.local_start];
        shared[0] = Goldilocks::ONE;
        shared[package.variable_column(5).unwrap()] = Goldilocks::ONE;
        let columns = package.circuit_columns(&shared).unwrap();
        let plan = package.plan_rows().unwrap();
        let source = package.source_rows().unwrap();
        assert_eq!(source.row_count() - plan.row_count(), 2);
        assert_eq!(
            package.relation_columns(columns.clone(), &plan).unwrap(),
            package.relation_columns(columns, &source).unwrap()
        );
    }
}
