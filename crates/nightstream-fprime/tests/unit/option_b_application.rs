//! A native application replaces Lean's only inside the region that Lean
//! writes: a region that claims a later gadget's piece, or fewer locals than
//! the application's pieces read, is refused. The identity binds the
//! application's rows.

use std::path::PathBuf;
use std::sync::Arc;

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

use crate::application_records::{
    ApplicationForm, ApplicationRecipeNode, ApplicationRecords, ApplicationRecordsWriter, ApplicationRowHeader,
    ApplicationTerm,
};
use crate::option_b::layout::Kind;
use crate::option_b::{
    load_option_b_packages, Linear, NativeBlock, NativeShape, OptionBPackages, Piece, OPTION_B_PACKAGES_IDENTITY,
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

fn packages() -> OptionBPackages {
    load_option_b_packages(
        &artifact("nightstream-fprime-option-b-light-v1.json"),
        &artifact("nightstream-fprime-option-b-refresh-v1.json"),
        OPTION_B_PACKAGES_IDENTITY,
    )
    .expect("option-B packages")
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
                "option-B application piece reads past its region"
            ))
        ));
    }
}

const FIRST_LOCAL: usize = 12;
const BLOCK: usize = 1096;

/// Four witness words and `locals` locals, each its own recipe
/// `local · 1 = previous` from witness word 0; then `output_k · 1 = read`.
fn chained(locals: usize, read: usize) -> Arc<ApplicationRecords> {
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

/// Two blocks of the chain's locals replace their rows: the application's
/// pieces are its four output rows and the two blocks, the dead locals get
/// no slot, each block gets 150 S-box columns, and a bit word gets one
/// coordinate. The identity binds the blocks and the bits.
#[test]
fn native_blocks_replace_their_rows_and_bits_take_one_coordinate() {
    let packages = packages();
    let records = chained(2 * BLOCK, FIRST_LOCAL + 2 * BLOCK - 1);
    let shape = NativeShape {
        blocks: vec![block(FIRST_LOCAL, 4), block(FIRST_LOCAL + BLOCK, 5)],
        bits: vec![5],
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
        assert!(matches!(&packed.pieces[0], Piece::Rows(rows) if rows.len() == 4));
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
        assert_eq!(kind(FIRST_LOCAL + 1080), Some(Kind::Field));
        assert_eq!(kind(5), Some(Kind::Bit));
        assert_eq!(kind(4), Some(Kind::Field));
        for column in [packed.sbox_start(), packed.sbox_start() + 299] {
            assert_eq!(layout.slot(column).map(|(_, kind)| kind), Some(Kind::Field));
        }
        assert_eq!(
            layout.width() + 41 * 2 * 1080 + 40,
            plain.layout().expect("layout").width() + 41 * 2 * 150
        );
        assert_eq!(
            packed.plan_rows().expect("rows").row_count() + 2 * BLOCK,
            plain.plan_rows().expect("rows").row_count() + 2 * (150 + 16)
        );
    }
    let identity = |shape: &NativeShape| {
        packages
            .with_application(Arc::clone(&records), 4, shape)
            .expect("in place")
            .identity()
    };
    let one_lane = NativeShape {
        blocks: vec![block(FIRST_LOCAL, 4), block(FIRST_LOCAL + BLOCK, 6)],
        ..shape.clone()
    };
    let no_bit = NativeShape {
        bits: Vec::new(),
        ..shape.clone()
    };
    let distinct = [
        plain.identity(),
        packed.identity(),
        identity(&one_lane),
        identity(&no_bit),
    ];
    for (index, left) in distinct.iter().enumerate() {
        assert!(distinct[index + 1..].iter().all(|right| right != left));
    }
}

/// Blocks that overlap or pass the locals, a lane that reads its own block
/// or a dead local, a kept row that reads a dead local, and a bit outside
/// the live witness words and locals are refused.
#[test]
fn native_shape_outside_its_rules_is_refused() {
    let packages = packages();
    let records = chained(2 * BLOCK, FIRST_LOCAL + 2 * BLOCK - 1);
    let two = |first: NativeBlock, second: NativeBlock| NativeShape {
        blocks: vec![first, second],
        bits: Vec::new(),
    };
    let bit = |variable| NativeShape {
        blocks: vec![block(FIRST_LOCAL, 4)],
        bits: vec![variable],
    };
    let cases = [
        (
            &records,
            two(block(FIRST_LOCAL, 4), block(FIRST_LOCAL + 1000, 4)),
            "option-B native block outside the free locals",
        ),
        (
            &records,
            two(block(FIRST_LOCAL, 4), block(FIRST_LOCAL + BLOCK + 1, 4)),
            "option-B native block outside the free locals",
        ),
        (
            &records,
            two(block(FIRST_LOCAL, FIRST_LOCAL), block(FIRST_LOCAL + BLOCK, 4)),
            "option-B native lane reads past its inputs",
        ),
        (
            &records,
            two(block(FIRST_LOCAL, 4), block(FIRST_LOCAL + BLOCK, FIRST_LOCAL + 5)),
            "option-B native lane reads past its inputs",
        ),
        (&records, bit(0), "option-B native bit outside the live words"),
        (&records, bit(9), "option-B native bit outside the live words"),
        (
            &records,
            bit(FIRST_LOCAL + 3),
            "option-B native bit outside the live words",
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
    let reads_dead = chained(2 * BLOCK, FIRST_LOCAL + 7);
    assert!(matches!(
        packages.with_application(reads_dead, 4, &bit(5)),
        Err(PackageError::Malformed("option-B application row reads a dead local"))
    ));
}
