//! Recompose the independently checked child evaluations into the exact
//! preceding PiCCS output. This checks the fixture handoff, not phase closure.

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use serde_json::{json, Value};

#[allow(dead_code)]
#[path = "support/pi_ccs_opening.rs"]
mod opening;
#[allow(dead_code)]
#[path = "per_application_logical_matrix_conformance/reference/mod.rs"]
mod reference;

use opening::{Extension, Ring, DEGREE};
use reference::Field;

const CHILDREN: usize = 16;
const MATRICES: usize = 14;

#[derive(Deserialize)]
struct Inputs {
    structural_identity: [u64; 4],
    verifier_context: [u64; 4],
    folded_metadata: PathBuf,
    preceding_result: PathBuf,
    commitments: PathBuf,
    families: PathBuf,
}

fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).expect("child recomposition input")).expect("numeric JSON")
}

fn ring(value: &Value) -> Ring {
    serde_json::from_value::<Vec<[u64; 2]>>(value.clone())
        .expect("ring words")
        .into_iter()
        .map(|words| Extension::checked(words).expect("canonical extension encoding"))
        .collect::<Vec<_>>()
        .try_into()
        .expect("complete ring")
}

fn rho_product(rho: &[i8; DEGREE], value: &Ring) -> Ring {
    // The selected sampler alphabet is {-2,-1,0,1,2}. Use its two signed
    // binary digits with the independent quotient-ring arithmetic.
    let digit = |bit: u32| {
        rho.map(|coefficient| {
            assert!((-2..=2).contains(&coefficient));
            if (coefficient.unsigned_abs() >> bit) & 1 == 0 {
                0
            } else if coefficient < 0 {
                255
            } else {
                1
            }
        })
    };
    let low = opening::multiply_signed(value, &digit(0));
    let high = opening::multiply_signed(value, &digit(1));
    std::array::from_fn(|index| low[index] + high[index] + high[index])
}

#[test]
#[ignore = "requires the checked child family paths on stdin; run under the 300-second cap"]
fn all_child_evaluations_recompose_to_the_preceding_pi_ccs_output() {
    let inputs: Inputs = serde_json::from_reader(std::io::stdin().lock()).expect("recomposition paths");
    let meta = read(&inputs.folded_metadata);
    assert_eq!(meta.as_array().expect("folded metadata").len(), 13);
    assert_eq!(meta[0], 1);
    assert_eq!(meta[1], json!(inputs.structural_identity));
    assert_eq!(meta[2], json!(inputs.verifier_context));
    assert!(meta[8].as_u64().expect("parent bound") < 1 << CHILDREN);
    let previous = read(&inputs.preceding_result);
    assert_eq!(
        previous
            .as_array()
            .expect("complete preceding Lean result")
            .len(),
        6
    );
    assert_eq!(previous[0], 1);
    assert_eq!(previous[1][0], 2);
    let phase = &previous[5];
    assert_eq!(phase.as_array().expect("complete PiCCS phase result").len(), 15);
    assert_eq!(phase[0], 1, "preceding PiCCS acceptance");
    assert_eq!(phase[6], meta[12], "exact preceding output point");
    assert_eq!(phase[14], meta[5], "exact preceding outgoing transcript state");
    assert_eq!(phase[10][0], previous[1][1], "same fresh commitment");
    assert_eq!(phase[11][0], previous[1][2], "same fresh public input");
    for index in [10, 11, 12, 13] {
        assert_eq!(
            phase[index].as_array().expect("17 preceding claims").len(),
            CHILDREN + 1
        );
    }
    for source in 1..=CHILDREN {
        assert_eq!(phase[10][source], json!(vec![0u64; 22 * DEGREE]));
        assert_eq!(phase[11][source], json!(vec![0u64; 270]));
        assert_eq!(ring(&phase[12][source]), [Extension::ZERO; DEGREE]);
        assert_eq!(
            phase[13][source]
                .as_array()
                .expect("14 matrix families")
                .len(),
            MATRICES
        );
        for matrix in 0..MATRICES {
            assert_eq!(ring(&phase[13][source][matrix]), [Extension::ZERO; DEGREE]);
        }
    }
    let rhos: Vec<Vec<i8>> = serde_json::from_value(meta[6].clone()).expect("recorded sampler output");
    assert_eq!(rhos.len(), CHILDREN + 1);
    assert!(rhos
        .iter()
        .all(|rho| rho.len() == DEGREE && rho.iter().all(|&v| (-2..=2).contains(&v))));
    let rho: [i8; DEGREE] = rhos[0].clone().try_into().unwrap();
    let children = (0..CHILDREN)
        .map(|child| {
            let record = read(&inputs.commitments.join(format!("child-{child}.json")));
            assert_eq!(record.as_array().expect("child commitment").len(), 7);
            assert_eq!(record[0], 1);
            assert_eq!(record[1], meta[1]);
            assert_eq!(record[2], meta[2]);
            assert_eq!(record[3], json!(child));
            assert_eq!(record[4], meta[12]);
            record
        })
        .collect::<Vec<_>>();
    for selected in std::iter::once(None).chain((0..MATRICES).map(Some)) {
        let name = selected.map_or_else(|| "K".to_owned(), |matrix| format!("A{matrix}"));
        let family = read(&inputs.families.join(format!("family-{name}.json")));
        assert_eq!(family.as_array().expect("complete child family").len(), 9);
        assert_eq!(family[0], 1);
        assert_eq!(family[1], meta[1]);
        assert_eq!(family[2], meta[2]);
        assert_eq!(family[3], meta[12]);
        assert_eq!(family[4], json!(u64::from(selected.is_some())));
        assert_eq!(family[5], json!(selected.unwrap_or(0)));
        for index in [6, 7, 8] {
            assert_eq!(family[index].as_array().expect("all child values").len(), CHILDREN);
        }
        let mut combined = [Extension::ZERO; DEGREE];
        for child in 0..CHILDREN {
            assert_eq!(family[6][child], children[child][5], "same checked public input");
            assert_eq!(family[7][child], children[child][6], "same checked commitment");
            let values = ring(&family[8][child]);
            if selected == Some(MATRICES - 1) {
                assert_eq!(values, [Extension::ZERO; DEGREE], "canonical zero matrix");
            }
            let weight = Field::checked(1u64 << child, "binary child weight").unwrap();
            for (target, value) in combined.iter_mut().zip(values) {
                *target += value.scale(weight);
            }
        }
        let preceding = ring(match selected {
            None => &phase[12][0],
            Some(matrix) => &phase[13][0][matrix],
        });
        assert_eq!(combined, rho_product(&rho, &preceding), "complete {name} recomposition");
    }
    println!("child_evaluation_recomposition=passed children={CHILDREN} families={} coefficients={} exact_prior_point_and_outgoing_state=checked", MATRICES + 1, (MATRICES + 1) * DEGREE);
}
