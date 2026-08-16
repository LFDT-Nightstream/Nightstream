//! Exact row-recipe and Lean-artifact gate for one phased PiRLC family.
//!
//! Owns: the canonical local column layout, an independent sparse-matrix
//! materialization, exhaustive recipe comparison, one honest assignment, and
//! deterministic Lean artifact output.
//!
//! Does not own: PiCCS input authority, the Poseidon2 replay rows, recursive
//! orchestration, terminal integration, or a complete F-prime relation.
//!
//! Emits constraints: 43,794 direct selective-CCS product rows over 45,415
//! columns. Only the general-selector, A, B, and C ports are nonzero.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use neo_ccs::CscMat;
use neo_fold_clean::frontends::r1cs_f_prime::lean_manifest::GOLDILOCKS_MODULUS;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryPiRlcFamilySelectiveCcs.lean";

const SCHEMA_VERSION: u64 = 1;
const SOURCE_COUNT: usize = 15;
const LANE_COUNT: usize = 54;
const CHALLENGE_START: usize = 1;
const INPUT_START: usize = CHALLENGE_START + SOURCE_COUNT * LANE_COUNT;
const OUTPUT_START: usize = INPUT_START + SOURCE_COUNT * LANE_COUNT;
const PRODUCT_START: usize = OUTPUT_START + LANE_COUNT;
const PRODUCT_ROWS: usize = SOURCE_COUNT * LANE_COUNT * LANE_COUNT;
const ROWS: usize = PRODUCT_ROWS + LANE_COUNT;
const COLUMNS: usize = PRODUCT_START + PRODUCT_ROWS;
const ROW_VARIABLES: usize = 16;
const PORT_COUNT: usize = 13;
const GENERAL_SELECTOR_PORT: usize = 1;
const A_PORT: usize = 2;
const B_PORT: usize = 3;
const C_PORT: usize = 4;

type Term = (usize, F);
type SourceRow = [Vec<Term>; 3];

fn challenge_column(source: usize, lane: usize) -> usize {
    CHALLENGE_START + source * LANE_COUNT + lane
}

fn input_column(source: usize, lane: usize) -> usize {
    INPUT_START + source * LANE_COUNT + lane
}

fn output_column(lane: usize) -> usize {
    OUTPUT_START + lane
}

fn product_column(source: usize, left: usize, right: usize) -> usize {
    PRODUCT_START + (source * LANE_COUNT + left) * LANE_COUNT + right
}

fn product_row(source: usize, left: usize, right: usize) -> usize {
    (source * LANE_COUNT + left) * LANE_COUNT + right
}

fn reduced_monomial(degree: usize) -> Vec<(usize, F)> {
    if degree < LANE_COUNT {
        vec![(degree, F::ONE)]
    } else if degree < LANE_COUNT + LANE_COUNT / 2 {
        vec![(degree - LANE_COUNT, -F::ONE), (degree - LANE_COUNT / 2, -F::ONE)]
    } else {
        vec![(degree - 3 * LANE_COUNT / 2, F::ONE)]
    }
}

fn build_source_matrices() -> [CscMat<F>; 3] {
    let mut a = Vec::with_capacity(2 * PRODUCT_ROWS + LANE_COUNT);
    let mut b = Vec::with_capacity(PRODUCT_ROWS + 2 * PRODUCT_ROWS);
    let mut c = Vec::with_capacity(ROWS);
    let minus_two = F::from_u64(GOLDILOCKS_MODULUS - 2);

    for source in 0..SOURCE_COUNT {
        for left in 0..LANE_COUNT {
            for right in 0..LANE_COUNT {
                let row = product_row(source, left, right);
                let product = product_column(source, left, right);
                a.push((row, challenge_column(source, left), F::ONE));
                a.push((row, 0, minus_two));
                b.push((row, input_column(source, right), F::ONE));
                c.push((row, product, F::ONE));
                for (output, coefficient) in reduced_monomial(left + right) {
                    b.push((PRODUCT_ROWS + output, product, coefficient));
                }
            }
        }
    }
    for output in 0..LANE_COUNT {
        let row = PRODUCT_ROWS + output;
        a.push((row, 0, F::ONE));
        c.push((row, output_column(output), F::ONE));
    }

    [
        CscMat::from_counted_triplets(a, ROWS, COLUMNS),
        CscMat::from_counted_triplets(b, ROWS, COLUMNS),
        CscMat::from_counted_triplets(c, ROWS, COLUMNS),
    ]
}

fn raw_terms(source: usize, degree: usize, coefficient: F) -> Vec<Term> {
    (0..LANE_COUNT)
        .filter_map(|left| {
            let right = degree.checked_sub(left)?;
            (right < LANE_COUNT).then(|| (product_column(source, left, right), coefficient))
        })
        .collect()
}

fn expected_source_row(row: usize) -> SourceRow {
    if row < PRODUCT_ROWS {
        let source = row / (LANE_COUNT * LANE_COUNT);
        let within_source = row % (LANE_COUNT * LANE_COUNT);
        let left = within_source / LANE_COUNT;
        let right = within_source % LANE_COUNT;
        return [
            vec![
                (0, F::from_u64(GOLDILOCKS_MODULUS - 2)),
                (challenge_column(source, left), F::ONE),
            ],
            vec![(input_column(source, right), F::ONE)],
            vec![(product_column(source, left, right), F::ONE)],
        ];
    }

    let output = row - PRODUCT_ROWS;
    let mut b = Vec::new();
    for source in 0..SOURCE_COUNT {
        b.extend(raw_terms(source, output, F::ONE));
        let folded_degree = if output < LANE_COUNT / 2 {
            output + LANE_COUNT
        } else {
            output + LANE_COUNT / 2
        };
        b.extend(raw_terms(source, folded_degree, -F::ONE));
        if output + 3 * LANE_COUNT / 2 <= 2 * LANE_COUNT - 2 {
            b.extend(raw_terms(source, output + 3 * LANE_COUNT / 2, F::ONE));
        }
    }
    [vec![(0, F::ONE)], b, vec![(output_column(output), F::ONE)]]
}

fn canonical_terms(terms: Vec<Term>) -> Vec<Term> {
    let mut combined = BTreeMap::new();
    for (column, coefficient) in terms {
        *combined.entry(column).or_insert(F::ZERO) += coefficient;
    }
    combined.retain(|_, coefficient| *coefficient != F::ZERO);
    combined.into_iter().collect()
}

fn matrix_coefficient(matrix: &CscMat<F>, row: usize, column: usize) -> Option<F> {
    let range = matrix.column_range(column);
    matrix.row_idx[range.clone()]
        .binary_search(&(row as u32))
        .ok()
        .map(|offset| matrix.vals[range.start + offset])
}

fn assert_matrices_match_recipe(matrices: &[CscMat<F>; 3]) {
    let mut expected_nnz = [0usize; 3];
    for row in 0..ROWS {
        let expected = expected_source_row(row).map(canonical_terms);
        for (matrix_index, terms) in expected.into_iter().enumerate() {
            expected_nnz[matrix_index] += terms.len();
            for (column, coefficient) in terms {
                assert_eq!(
                    matrix_coefficient(&matrices[matrix_index], row, column),
                    Some(coefficient),
                    "matrix {matrix_index}, row {row}, column {column}",
                );
            }
        }
    }
    for (matrix_index, matrix) in matrices.iter().enumerate() {
        assert!(matrix.is_canonical(), "matrix {matrix_index} must be canonical");
        assert_eq!(matrix.vals.len(), expected_nnz[matrix_index]);
    }
}

fn honest_assignment() -> Vec<F> {
    let mut assignment = vec![F::ZERO; COLUMNS];
    assignment[0] = F::ONE;
    let mut output = [F::ZERO; LANE_COUNT];
    for source in 0..SOURCE_COUNT {
        for lane in 0..LANE_COUNT {
            assignment[challenge_column(source, lane)] = F::from_usize((source + lane) % 5);
            assignment[input_column(source, lane)] = F::from_usize(1 + source * 3 + lane * 5);
        }
        for left in 0..LANE_COUNT {
            let challenge = assignment[challenge_column(source, left)] + F::from_u64(GOLDILOCKS_MODULUS - 2);
            for right in 0..LANE_COUNT {
                let product = challenge * assignment[input_column(source, right)];
                assignment[product_column(source, left, right)] = product;
                for (lane, coefficient) in reduced_monomial(left + right) {
                    output[lane] += coefficient * product;
                }
            }
        }
    }
    for (lane, value) in output.into_iter().enumerate() {
        assignment[output_column(lane)] = value;
    }
    assignment
}

fn assert_honest_assignment_satisfies(matrices: &[CscMat<F>; 3]) {
    let assignment = honest_assignment();
    let mut images = std::array::from_fn::<_, 3, _>(|_| vec![F::ZERO; ROWS]);
    for (matrix, image) in matrices.iter().zip(images.iter_mut()) {
        matrix.add_mul_into(&assignment, image, ROWS);
    }
    for row in 0..ROWS {
        assert_eq!(images[0][row] * images[1][row], images[2][row], "row {row}");
    }
}

fn render_artifact() -> String {
    let mut rendered = String::new();
    rendered.push_str(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.PiRlcFamilySelectiveCcsSchema\n\n\
/-! Generated file: compact exact row recipe for one production PiRLC family.\n\n\
Owns: the Rust-declared dimensions, canonical column starts, selective port\n\
indices, and Goldilocks coefficients used by the exhaustive row audit.\n\n\
Does not own: semantic truth, PiCCS input authority, Poseidon2 binding, or the\n\
complete recursive and terminal F-prime relations. Lean recomputes every row.\n\n\
Emits constraints: no. This file contains recipe data, not a trusted digest.\n-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiRlcFamilySelectiveCcs\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcFamilySelectiveCcs.Artifact\n\n\
def rawArtifact : RawArtifact where\n",
    );
    writeln!(rendered, "  schemaVersion := {SCHEMA_VERSION}").unwrap();
    writeln!(rendered, "  sourceCount := {SOURCE_COUNT}").unwrap();
    writeln!(rendered, "  laneCount := {LANE_COUNT}").unwrap();
    writeln!(rendered, "  challengeStart := {CHALLENGE_START}").unwrap();
    writeln!(rendered, "  inputStart := {INPUT_START}").unwrap();
    writeln!(rendered, "  outputStart := {OUTPUT_START}").unwrap();
    writeln!(rendered, "  productStart := {PRODUCT_START}").unwrap();
    writeln!(rendered, "  productRows := {PRODUCT_ROWS}").unwrap();
    writeln!(rendered, "  rows := {ROWS}").unwrap();
    writeln!(rendered, "  columns := {COLUMNS}").unwrap();
    writeln!(rendered, "  rowVariables := {ROW_VARIABLES}").unwrap();
    writeln!(rendered, "  portCount := {PORT_COUNT}").unwrap();
    writeln!(rendered, "  generalSelectorPort := {GENERAL_SELECTOR_PORT}").unwrap();
    writeln!(rendered, "  aPort := {A_PORT}").unwrap();
    writeln!(rendered, "  bPort := {B_PORT}").unwrap();
    writeln!(rendered, "  cPort := {C_PORT}").unwrap();
    writeln!(rendered, "  minusOne := {}", GOLDILOCKS_MODULUS - 1).unwrap();
    writeln!(rendered, "  minusTwo := {}", GOLDILOCKS_MODULUS - 2).unwrap();
    rendered.push_str(
        "\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiRlcFamilySelectiveCcs\n",
    );
    rendered
}

fn artifact_path() -> String {
    format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_PATH)
}

#[test]
fn pi_rlc_family_rows_match_generated_selective_recipe() {
    assert_eq!(ROWS, 43_794);
    assert_eq!(COLUMNS, 45_415);
    assert!(ROWS <= 1 << ROW_VARIABLES);
    assert!((1 << (ROW_VARIABLES - 1)) < ROWS);

    let matrices = build_source_matrices();
    assert_matrices_match_recipe(&matrices);
    assert_honest_assignment_satisfies(&matrices);

    let path = artifact_path();
    let rendered = render_artifact();
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if rendered != committed {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, rendered).expect("write PiRLC family artifact candidate");
        panic!("PiRLC family selective-CCS artifact drifted; inspect {expected} and promote it explicitly");
    }
}

#[test]
#[ignore = "deliberately writes the reviewed generated Lean artifact"]
fn regenerate_pi_rlc_family_selective_ccs_artifact() {
    std::fs::write(artifact_path(), render_artifact()).expect("write generated PiRLC family artifact");
}
