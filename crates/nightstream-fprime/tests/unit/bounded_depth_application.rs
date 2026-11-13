//! A native application replaces Lean's only inside the region that Lean
//! writes: a region that claims a later gadget's piece, or fewer locals than
//! the application's pieces read, is refused. The identity binds the
//! application's rows.

use std::path::PathBuf;
use std::sync::Arc;

use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;

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
            constant: Fq::ZERO,
            terms: vec![(lane, Fq::ONE)],
        }),
    }
}

/// Two blocks of the chain's locals replace their rows: the application's
/// pieces are its four output rows, its three bit rows and the two blocks,
/// a block's locals get no slot (a row reads the form of an output local),
/// and each block gets 150 S-box columns. A witness word that a row forces
/// to 0 or 1 gets one coordinate; a state word that a row forces keeps its
/// field slot. A witness word that no row reads gets no slot. The identity
/// binds the blocks.
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
        for column in [packed.sbox_start(), packed.sbox_start() + 299] {
            assert_eq!(layout.slot(column).map(|(_, kind)| kind), Some(Kind::Field));
        }
        // In the plain package, each local has a field slot, and the forced
        // output local has a bit slot.
        assert_eq!(
            layout.width() + 41 * (2 * BLOCK - 1) + 1,
            plain.layout().expect("layout").width() + 41 * 2 * 150
        );
        assert_eq!(
            packed.plan_rows().expect("rows").row_count() + 2 * BLOCK,
            plain.plan_rows().expect("rows").row_count() + 2 * 150
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
