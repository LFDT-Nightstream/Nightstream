//! Compare complete native PiDEC evaluations with independent Lean values.
//! This module only decodes, validates and compares the retained outputs.

use std::{fs, path::Path, time::Instant};

use neo_ajtai::nightstream_fprime_setup::PRODUCTION_MESSAGE_COLUMNS;
use neo_math::{D, F};
use nightstream_fprime::{
    PI_CCS_V1_1_ROUND_COUNT as POINT, PI_DEC_V1_1_CHILD_COUNT as CHILDREN,
    PI_DEC_V1_1_COMMITMENT_WORDS_PER_CHILD as COMMITMENT_WORDS, PI_DEC_V1_1_EVAL_A_MATRICES_PER_CHILD as MATRICES,
    PI_DEC_V1_1_PUBLIC_INPUT_WORDS_PER_CHILD as PUBLIC_WORDS,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

type KWords = [u64; 2];
type PadValues = Vec<Vec<KWords>>;
type MatrixValues = Vec<Vec<Vec<KWords>>>;
type NativeChildren = (Vec<KWords>, Vec<Vec<u64>>, Vec<Vec<u64>>, PadValues, MatrixValues);
type LeanEvaluations = (u64, usize, Vec<KWords>, PadValues, MatrixValues);

struct Evaluations {
    point: Vec<KWords>,
    pad: PadValues,
    matrices: MatrixValues,
}

fn validate(values: &Evaluations, source: &str) {
    assert_eq!(values.point.len(), POINT, "{source}: complete common point");
    assert_eq!(values.pad.len(), CHILDREN, "{source}: all Pad children");
    assert_eq!(values.matrices.len(), CHILDREN, "{source}: all matrix children");
    for (coordinate, pair) in values.point.iter().enumerate() {
        for (component, &word) in pair.iter().enumerate() {
            assert!(
                word < F::ORDER_U64,
                "{source}: noncanonical point coordinate {coordinate}, component {component}"
            );
        }
    }
    for child in 0..CHILDREN {
        assert_eq!(values.pad[child].len(), D, "{source}: Pad lanes at child {child}");
        assert_eq!(
            values.matrices[child].len(),
            MATRICES,
            "{source}: matrix families at child {child}"
        );
        for (lane, pair) in values.pad[child].iter().enumerate() {
            for (component, &word) in pair.iter().enumerate() {
                assert!(
                    word < F::ORDER_U64,
                    "{source}: noncanonical child {child}, family pad, matrix none, lane {lane}, component {component}"
                );
            }
        }
        for matrix in 0..MATRICES {
            assert_eq!(
                values.matrices[child][matrix].len(),
                D,
                "{source}: matrix lanes at child {child}, matrix {matrix}"
            );
            for (lane, pair) in values.matrices[child][matrix].iter().enumerate() {
                for (component, &word) in pair.iter().enumerate() {
                    assert!(word < F::ORDER_U64, "{source}: noncanonical child {child}, family matrix, matrix {matrix}, lane {lane}, component {component}");
                }
            }
        }
    }
}

fn compare_pair(actual: &KWords, expected: &KWords, location: &str) -> Result<(), String> {
    for component in 0..2 {
        if actual[component] != expected[component] {
            return Err(format!("{location}, component {component}"));
        }
    }
    Ok(())
}

// Both inputs have passed the same complete shape and canonical-word checks.
fn compare_values(actual: &Evaluations, expected: &Evaluations) -> Result<(), String> {
    for coordinate in 0..POINT {
        compare_pair(
            &actual.point[coordinate],
            &expected.point[coordinate],
            &format!("point mismatch at coordinate {coordinate}"),
        )?;
    }
    for child in 0..CHILDREN {
        for lane in 0..D {
            compare_pair(
                &actual.pad[child][lane],
                &expected.pad[child][lane],
                &format!("evaluation mismatch at child {child}, family pad, matrix none, lane {lane}"),
            )?;
        }
        for matrix in 0..MATRICES {
            for lane in 0..D {
                compare_pair(
                    &actual.matrices[child][matrix][lane],
                    &expected.matrices[child][matrix][lane],
                    &format!("evaluation mismatch at child {child}, family matrix, matrix {matrix}, lane {lane}"),
                )?;
            }
        }
    }
    Ok(())
}

/// Lean emits [1,blocks,point,pad,matrix], with K encoded as [re,im].
/// Native children use the existing [point,commitments,public,pad,matrix]
/// format. Normalize native field words through F; reject noncanonical Lean
/// words. Commitments and public inputs have their exact transport dimensions
/// checked here; their semantic comparisons remain in their existing gates.
pub fn compare(native_path: &Path, lean_path: &Path) {
    let started = Instant::now();
    let (point, commitments, public, pad, matrices): NativeChildren =
        serde_json::from_slice(&fs::read(native_path).expect("native child evaluations file"))
            .expect("complete native child output schema");
    assert_eq!(commitments.len(), CHILDREN, "native commitment children");
    assert_eq!(public.len(), CHILDREN, "native public-input children");
    for child in 0..CHILDREN {
        assert_eq!(
            commitments[child].len(),
            COMMITMENT_WORDS,
            "native commitment shape at child {child}"
        );
        assert_eq!(
            public[child].len(),
            PUBLIC_WORDS,
            "native public shape at child {child}"
        );
    }
    let mut actual = Evaluations { point, pad, matrices };
    for pair in actual
        .point
        .iter_mut()
        .chain(actual.pad.iter_mut().flatten())
        .chain(actual.matrices.iter_mut().flatten().flatten())
    {
        for word in pair {
            *word = F::from_u64(*word).as_canonical_u64();
        }
    }
    validate(&actual, "native");

    let (schema, blocks, point, pad, matrices): LeanEvaluations =
        serde_json::from_slice(&fs::read(lean_path).expect("Lean evaluations file"))
            .expect("complete Lean evaluation schema");
    assert_eq!(schema, 1, "Lean evaluation schema");
    assert_eq!(blocks, PRODUCTION_MESSAGE_COLUMNS as usize, "complete selected carrier");
    let expected = Evaluations { point, pad, matrices };
    validate(&expected, "Lean");
    compare_values(&actual, &expected).expect("complete independent PiDEC evaluations match");

    let child = CHILDREN - 1;
    let matrix = MATRICES - 1;
    let lane = D - 1;
    let component = 1;
    let word = &mut actual.matrices[child][matrix][lane][component];
    *word = (F::from_u64(*word) + F::ONE).as_canonical_u64();
    assert_eq!(
        compare_values(&actual, &expected).unwrap_err(),
        format!(
            "evaluation mismatch at child {child}, family matrix, matrix {matrix}, lane {lane}, component {component}"
        )
    );
    println!(
        "pidec_evaluation_replay=passed children={CHILDREN} matrices={MATRICES} lanes={D} evaluation_words={} point_words={} blocks={blocks} target_mutation=rejected child={child} family=matrix matrix={matrix} lane={lane} component={component} elapsed={:?}",
        CHILDREN * (1 + MATRICES) * D * 2,
        POINT * 2,
        started.elapsed()
    );
}
