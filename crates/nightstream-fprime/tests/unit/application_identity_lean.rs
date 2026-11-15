//! The word stream of the application identity against Lean
//! (`Lifecycle.BoundedDepth.ApplicationIdentity.Application.words`, SPEC.md
//! section 4.4). Lean emits sample records with their words and their
//! identity (`Export.EncodingParity.applications`). The records go into the
//! Rust types, and the functions of `with_application` must give Lean's
//! words and Lean's hash.

use std::sync::Arc;

use neo_ccs::crypto::poseidon2::poseidon2_hash;
use neo_math::Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde_json::Value;

use super::tests::{block, packages, BLOCK, FIRST_LOCAL};
use super::{layout_words, record_words};
use crate::application_records::{
    ApplicationForm, ApplicationRecipeNode, ApplicationRecords, ApplicationRecordsWriter, ApplicationRowHeader,
    ApplicationTerm,
};
use crate::bounded_depth::packed_boolean::PackedBooleans;
use crate::bounded_depth::packed_product::Product;
use crate::bounded_depth::packed_zero::{Input, ZeroProduct};
use crate::bounded_depth::{guard_fusion, word_packing, Linear, NativeBlock, NativeShape, Piece, Row, StepPackage};
use crate::package::PackageError;

fn list(value: &Value) -> &[Value] {
    value.as_array().expect("a list")
}

fn word(value: &Value) -> u64 {
    value.as_u64().expect("a word")
}

fn index(value: &Value) -> usize {
    word(value) as usize
}

fn words(value: &Value) -> Vec<u64> {
    list(value).iter().map(word).collect()
}

fn indices(value: &Value) -> Vec<usize> {
    list(value).iter().map(index).collect()
}

fn linear(value: &Value) -> Linear {
    Linear {
        constant: Fq::from_u64(word(&value[0])),
        terms: list(&value[1])
            .iter()
            .map(|term| (index(&term[0]), Fq::from_u64(word(&term[1]))))
            .collect(),
    }
}

fn row(value: &Value) -> Row {
    Row {
        a: linear(&value[0]),
        b: linear(&value[1]),
        c: linear(&value[2]),
    }
}

fn empty_row() -> Row {
    row(&serde_json::json!([[0, []], [0, []], [0, []]]))
}

/// The rows and the recipe rows of a Lean application. The identity does not
/// read the nodes of a recipe, so each recipe is one variable.
fn records(application: &Value) -> Arc<ApplicationRecords> {
    let mut writer = ApplicationRecordsWriter::new().expect("writer");
    for row in list(&application[2]) {
        let forms = [ApplicationForm::A, ApplicationForm::B, ApplicationForm::C];
        let header = ApplicationRowHeader {
            constants: std::array::from_fn(|form| word(&row[form][0])),
            term_counts: std::array::from_fn(|form| list(&row[form][1]).len()),
        };
        let terms = forms.into_iter().flat_map(|form| {
            list(&row[form.index()][1]).iter().map(move |term| {
                Ok(ApplicationTerm {
                    form,
                    variable: index(&term[0]),
                    coefficient: word(&term[1]),
                })
            })
        });
        writer.append_row(header, terms).expect("row");
    }
    for recipe in list(&application[3]) {
        writer
            .append_recipe(index(recipe), [Ok(ApplicationRecipeNode::Variable(0))])
            .expect("recipe");
    }
    Arc::new(writer.finish().expect("records"))
}

fn shape(application: &Value) -> NativeShape {
    let blocks = list(&application[4]).iter().map(|block| NativeBlock {
        first_local: index(&block[0]),
        lanes: std::array::from_fn(|lane| linear(&block[1][lane])),
    });
    NativeShape {
        blocks: blocks.collect(),
    }
}

/// `base` with the layout of a Lean application. Each indexed row of the
/// layout goes to its index in the rows piece.
fn with_layout(base: &StepPackage, layout: &Value) -> StepPackage {
    let mut rows = Vec::new();
    let mut indexed = |entries: &Value| -> Vec<usize> {
        let place = |entry: &Value| {
            let position = index(&entry[0]);
            if rows.len() <= position {
                rows.resize(position + 1, empty_row());
            }
            rows[position] = row(&entry[1]);
            position
        };
        list(entries).iter().map(place).collect()
    };
    let fusion = list(&layout[0]).iter().map(|group| guard_fusion::Group {
        source_records: list(&group[0])
            .iter()
            .map(|pair| (index(&pair[0]), index(&pair[1])))
            .collect(),
        sources: indexed(&group[1]),
        certificates: indexed(&group[2]),
        row: row(&group[3]),
    });
    let fusion = fusion.collect();
    let bit = |bit: &Value| match word(&bit[0]) {
        0 => word_packing::Bit::Zero,
        1 => word_packing::Bit::One,
        _ => word_packing::Bit::Column(index(&bit[1])),
    };
    let packing = list(&layout[1]).iter().map(|group| word_packing::Group {
        sources: indexed(&group[0]),
        certificates: indexed(&group[1]),
        guard: linear(&group[2]),
        pairs: list(&group[3])
            .iter()
            .map(|pair| (bit(&pair[0]), bit(&pair[1])))
            .collect(),
    });
    let packing = packing.collect();
    let input = |column: &Value, complement: &Value| Input {
        column: index(column),
        complement: word(complement) == 1,
    };
    StepPackage {
        guard_fusion: guard_fusion::GuardFusion {
            groups: fusion,
            ..Default::default()
        },
        word_packing: word_packing::WordPacking { groups: packing },
        packed_booleans: PackedBooleans {
            start: index(&layout[2]),
            sources: indices(&layout[3]),
            removed: list(&layout[4])
                .iter()
                .map(|row| (index(row), empty_row()))
                .collect(),
            products: list(&layout[5])
                .iter()
                .map(|product| Product {
                    row: index(&product[0]),
                    left: index(&product[1]),
                    right: index(&product[2]),
                    output: index(&product[3]),
                })
                .collect(),
            zeros: list(&layout[6])
                .iter()
                .map(|zero| ZeroProduct {
                    row: index(&zero[0]),
                    left: input(&zero[1], &zero[2]),
                    right: input(&zero[3], &zero[4]),
                })
                .collect(),
        },
        slots: list(&layout[7])
            .iter()
            .map(|slot| (index(&slot[0]), index(&slot[1]), word(&slot[2]) as u8))
            .collect(),
        pieces: vec![Piece::Rows(rows)],
        ..base.clone()
    }
}

#[test]
fn the_identity_words_are_the_lean_words() {
    let file: Value = serde_json::from_str(include_str!(
        "../../../../formal/nightstream-fprime/artifacts/nightstream-fprime-encoding-parity-v1.json"
    ))
    .expect("Lean encoding file");
    assert_eq!(file[0].as_u64(), Some(1), "Lean encoding schema");
    let base = packages();
    let samples = list(&file[4]);
    assert_eq!(samples.len(), 4);
    for sample in samples {
        let application = &sample[0];
        let mut stream = Vec::new();
        let mut update = |word| stream.push(word);
        record_words(
            words(&application[0]).try_into().expect("an identity"),
            &records(application),
            index(&application[1]),
            &shape(application),
            &mut update,
        )
        .expect("record words");
        layout_words(&with_layout(base.light(), &application[5]), &mut update).expect("light words");
        layout_words(&with_layout(base.refresh(), &application[6]), &mut update).expect("refresh words");
        assert_eq!(stream, words(&sample[1]));
        let input = stream.iter().map(|&word| Fq::from_u64(word));
        let identity = poseidon2_hash(&input.collect::<Vec<_>>());
        assert_eq!(identity.map(|word| word.as_canonical_u64()).to_vec(), words(&sample[2]));
    }
}

/// One block of locals from witness word 0, each `local · 1 = previous`, and
/// the output rows. The row of the second local reads `previous + shift`.
fn one_block(shift: u64) -> Arc<ApplicationRecords> {
    let mut writer = ApplicationRecordsWriter::new().expect("writer");
    let mut copy = |from: usize, to: usize| {
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
    let mut rows = vec![
        copy(4, FIRST_LOCAL),
        copy(FIRST_LOCAL + shift as usize, FIRST_LOCAL + 1),
    ];
    rows.extend((FIRST_LOCAL + 2..FIRST_LOCAL + BLOCK).map(|local| copy(local - 1, local)));
    for lane in 0..4 {
        copy(FIRST_LOCAL + BLOCK - 1, 8 + lane);
    }
    for row in rows {
        writer
            .append_recipe(row, [Ok(ApplicationRecipeNode::Variable(0))])
            .expect("recipe");
    }
    Arc::new(writer.finish().expect("records"))
}

/// A block drops the rows of its locals, so no other check reads their
/// variables. A variable `v + q` there has the field element of `v`, and the
/// identity refuses it (Lean `Application.Canonical`).
#[test]
fn a_word_at_or_above_q_is_refused() {
    let packages = packages();
    let shape = NativeShape {
        blocks: vec![block(FIRST_LOCAL, 4)],
    };
    packages
        .with_application(one_block(0), 4, &shape)
        .expect("canonical words");
    assert!(matches!(
        packages.with_application(one_block(Fq::ORDER_U64), 4, &shape),
        Err(PackageError::NonCanonicalField {
            location: "application identity",
            value
        }) if value == Fq::ORDER_U64 + FIRST_LOCAL as u64
    ));
}
