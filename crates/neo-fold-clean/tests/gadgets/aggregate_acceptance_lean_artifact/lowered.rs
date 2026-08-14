//! Lowered-row and sparse-polynomial extraction for aggregate acceptance.
//!
//! Owns: role-discriminated discovery of the exact nine active rows, their
//! global row ranges, arity-56 matrix bindings, and 25 relevant polynomial
//! terms.
//!
//! Does not own: source rows, the projected inverse, or outer decoded-LC input
//! substitution.
//!
//! | Constraint family | Rows/chunk | Nonzero production matrices |
//! |---|---:|---|
//! | Tree output bit pairs | 7 | selector, quadratic-left, quadratic-right |
//! | Radix-3 edge aggregate | 1 | selector, product-left/right 0..13, product-out |
//! | Root/accept binding | 1 | selector, product-left/right 0, product-out |

use std::collections::{BTreeMap, BTreeSet};

use neo_ccs::{CcsMatrix, CcsStructure, CscMat};
use neo_fold_clean::engine::r1cs_circuit::R1csEncodingTrace;
use neo_fold_clean::frontends::f_prime::gadget_native::EncodedGadgetNativeR1cs;
use neo_fold_clean::frontends::f_prime::gadget_native::GadgetNativeCoordinateGateRoles;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::{
    signed, ActiveRow, ChunkGeometry, CoordinateRole, MatrixBinding, MatrixLinearCombination, MatrixRole,
    PolynomialTerm, RoleTerm, VariablePower, ACCEPTANCE_COORDINATES_PER_CHUNK, ACTIVE_ROWS_PER_CHUNK, CHUNKS,
    GATE_ARITY,
};

const SELECTOR: usize = GadgetNativeCoordinateGateRoles::SELECTOR;
const PRODUCT_LEFT: usize = GadgetNativeCoordinateGateRoles::PRODUCT_LEFT;
const PRODUCT_SLOTS: usize = GadgetNativeCoordinateGateRoles::PRODUCT_SLOTS;
const PRODUCT_RIGHT: usize = GadgetNativeCoordinateGateRoles::PRODUCT_RIGHT;
const PRODUCT_OUT: usize = GadgetNativeCoordinateGateRoles::PRODUCT_OUT;
const QUADRATIC_BIT_LEFT: usize = GadgetNativeCoordinateGateRoles::BOOLEAN_PAIR_LEFT;
const QUADRATIC_BIT_RIGHT: usize = GadgetNativeCoordinateGateRoles::BOOLEAN_PAIR_RIGHT;

#[derive(Clone, Debug, PartialEq, Eq)]
struct GlobalTerm {
    matrix: usize,
    column: usize,
    coefficient: F,
}

pub(super) struct LoweredAudit {
    pub matrix_bindings: Vec<MatrixBinding>,
    pub active_rows: Vec<ActiveRow>,
    pub polynomial_terms: Vec<PolynomialTerm>,
}

fn csc_rows(matrix: &CscMat<F>, matrix_index: usize, rows: &mut [Vec<GlobalTerm>]) {
    for column in 0..matrix.ncols {
        for entry in matrix.column_range(column) {
            rows[matrix.row_index(entry)].push(GlobalTerm {
                matrix: matrix_index,
                column,
                coefficient: matrix.vals[entry],
            });
        }
    }
}

fn index_rows(structure: &CcsStructure<F>) -> Vec<Vec<GlobalTerm>> {
    let mut rows = vec![Vec::new(); structure.n];
    for (matrix_index, matrix) in structure.matrices.iter().enumerate() {
        match matrix {
            CcsMatrix::Csc(matrix) => csc_rows(matrix, matrix_index, &mut rows),
            CcsMatrix::Identity { .. } | CcsMatrix::CscWithSeededPhi81 { .. } | CcsMatrix::VerifierArtifact { .. } => {
                panic!("aggregate-acceptance gate matrices must be ordinary CSC")
            }
        }
    }
    for row in &mut rows {
        row.sort_by_key(|term| (term.matrix, term.column));
        assert!(row
            .windows(2)
            .all(|pair| { (pair[0].matrix, pair[0].column) != (pair[1].matrix, pair[1].column) }));
    }
    rows
}

fn csc_rows_for_column(matrix: &CscMat<F>, column: usize) -> Vec<usize> {
    matrix
        .column_range(column)
        .map(|entry| matrix.row_index(entry))
        .collect()
}

fn rows_for_column(structure: &CcsStructure<F>, matrix: usize, column: usize) -> Vec<usize> {
    match &structure.matrices[matrix] {
        CcsMatrix::Csc(matrix) => csc_rows_for_column(matrix, column),
        _ => panic!("aggregate-acceptance role matrix must be ordinary CSC"),
    }
}

fn unique_role_row(structure: &CcsStructure<F>, matrix: usize, column: usize, role: &str) -> usize {
    let rows = rows_for_column(structure, matrix, column);
    assert_eq!(rows.len(), 1, "{role} must identify one exact global row");
    rows[0]
}

fn push(terms: &mut Vec<GlobalTerm>, matrix: usize, column: usize, coefficient: F) {
    if coefficient != F::ZERO {
        terms.push(GlobalTerm {
            matrix,
            column,
            coefficient,
        });
    }
}

fn finish_row(mut terms: Vec<GlobalTerm>) -> Vec<GlobalTerm> {
    terms.sort_by_key(|term| (term.matrix, term.column));
    assert!(terms
        .windows(2)
        .all(|pair| { (pair[0].matrix, pair[0].column) != (pair[1].matrix, pair[1].column) }));
    terms
}

fn edge_columns(inputs: &[usize], outputs: &[usize], index: usize) -> (usize, usize) {
    match index {
        0 => (inputs[0], inputs[1]),
        1 => (inputs[2], inputs[3]),
        2 => (inputs[4], inputs[5]),
        3 => (inputs[6], inputs[7]),
        4 => (outputs[0], outputs[1]),
        5 => (outputs[2], outputs[3]),
        6 => (outputs[4], outputs[5]),
        7 => (inputs[8], inputs[9]),
        8 => (inputs[10], inputs[11]),
        9 => (inputs[12], inputs[13]),
        10 => (inputs[14], inputs[15]),
        11 => (outputs[7], outputs[8]),
        12 => (outputs[9], outputs[10]),
        13 => (outputs[11], outputs[12]),
        _ => unreachable!("fourteen-edge acceptance tree"),
    }
}

fn expected_rows(geometry: &ChunkGeometry, weights: &[F]) -> Vec<Vec<GlobalTerm>> {
    let accept = geometry.encoded_acceptance_columns[0];
    let outputs = &geometry.encoded_acceptance_columns[1..];
    assert_eq!(outputs.len(), ACCEPTANCE_COORDINATES_PER_CHUNK - 1);
    assert_eq!(weights.len(), outputs.len());
    let mut rows = Vec::with_capacity(ACTIVE_ROWS_PER_CHUNK);
    for pair in 0..7 {
        let mut row = Vec::new();
        push(&mut row, SELECTOR, 0, F::ONE);
        push(&mut row, QUADRATIC_BIT_LEFT, outputs[2 * pair], F::ONE);
        push(&mut row, QUADRATIC_BIT_RIGHT, outputs[2 * pair + 1], F::ONE);
        rows.push(finish_row(row));
    }

    let mut aggregate = Vec::new();
    push(&mut aggregate, SELECTOR, 0, F::ONE);
    for index in 0..outputs.len() {
        let (left, right) = edge_columns(&geometry.encoded_input_columns, outputs, index);
        if matches!(index, 0..=3 | 7..=10) {
            push(&mut aggregate, PRODUCT_LEFT + index, 0, weights[index]);
            push(&mut aggregate, PRODUCT_LEFT + index, left, -weights[index]);
            push(&mut aggregate, PRODUCT_RIGHT + index, 0, F::ONE);
            push(&mut aggregate, PRODUCT_RIGHT + index, right, -F::ONE);
        } else {
            push(&mut aggregate, PRODUCT_LEFT + index, left, weights[index]);
            push(&mut aggregate, PRODUCT_RIGHT + index, right, F::ONE);
        }
        push(&mut aggregate, PRODUCT_OUT, outputs[index], weights[index]);
    }
    rows.push(finish_row(aggregate));

    let mut root = Vec::new();
    push(&mut root, SELECTOR, 0, F::ONE);
    push(&mut root, PRODUCT_LEFT, outputs[6], F::ONE);
    push(&mut root, PRODUCT_RIGHT, outputs[13], F::ONE);
    push(&mut root, PRODUCT_OUT, 0, F::ONE);
    push(&mut root, PRODUCT_OUT, accept, -F::ONE);
    rows.push(finish_row(root));
    rows
}

fn coordinate_roles(geometry: &ChunkGeometry) -> BTreeMap<usize, CoordinateRole> {
    let mut roles = BTreeMap::from([(0, CoordinateRole::One)]);
    for (index, &column) in geometry.encoded_input_columns.iter().enumerate() {
        assert!(roles
            .insert(column, CoordinateRole::ChunkBit(index))
            .is_none());
    }
    assert!(roles
        .insert(geometry.encoded_acceptance_columns[0], CoordinateRole::Accept)
        .is_none());
    for (index, &column) in geometry.encoded_acceptance_columns[1..].iter().enumerate() {
        assert!(roles
            .insert(column, CoordinateRole::TreeOutput(index))
            .is_none());
    }
    roles
}

fn matrix_role(index: usize) -> MatrixRole {
    match index {
        SELECTOR => MatrixRole::Selector,
        PRODUCT_LEFT..PRODUCT_RIGHT => MatrixRole::ProductLeft(index - PRODUCT_LEFT),
        PRODUCT_RIGHT..PRODUCT_OUT => MatrixRole::ProductRight(index - PRODUCT_RIGHT),
        PRODUCT_OUT => MatrixRole::ProductOut,
        QUADRATIC_BIT_LEFT => MatrixRole::QuadraticBitLeft,
        QUADRATIC_BIT_RIGHT => MatrixRole::QuadraticBitRight,
        _ => panic!("matrix {index} is not an aggregate-acceptance role"),
    }
}

fn normalize_row(row: &[GlobalTerm], geometry: &ChunkGeometry) -> ActiveRow {
    let roles = coordinate_roles(geometry);
    let mut grouped = BTreeMap::<MatrixRole, Vec<RoleTerm<CoordinateRole>>>::new();
    for term in row {
        grouped
            .entry(matrix_role(term.matrix))
            .or_default()
            .push(RoleTerm {
                role: *roles
                    .get(&term.column)
                    .unwrap_or_else(|| panic!("unowned encoded acceptance coordinate {}", term.column)),
                coefficient: signed(term.coefficient),
            });
    }
    grouped
        .into_iter()
        .map(|(role, terms)| MatrixLinearCombination { role, terms })
        .collect()
}

fn matrix_bindings() -> Vec<MatrixBinding> {
    std::iter::once(MatrixBinding {
        role: MatrixRole::Selector,
        index: SELECTOR,
    })
    .chain((0..PRODUCT_SLOTS).map(|slot| MatrixBinding {
        role: MatrixRole::ProductLeft(slot),
        index: PRODUCT_LEFT + slot,
    }))
    .chain((0..PRODUCT_SLOTS).map(|slot| MatrixBinding {
        role: MatrixRole::ProductRight(slot),
        index: PRODUCT_RIGHT + slot,
    }))
    .chain([
        MatrixBinding {
            role: MatrixRole::ProductOut,
            index: PRODUCT_OUT,
        },
        MatrixBinding {
            role: MatrixRole::QuadraticBitLeft,
            index: QUADRATIC_BIT_LEFT,
        },
        MatrixBinding {
            role: MatrixRole::QuadraticBitRight,
            index: QUADRATIC_BIT_RIGHT,
        },
    ])
    .collect()
}

fn acceptance_matrix_indices() -> BTreeSet<usize> {
    matrix_bindings()
        .into_iter()
        .map(|binding| binding.index)
        .collect()
}

fn polynomial_schema(structure: &CcsStructure<F>) -> Vec<PolynomialTerm> {
    assert_eq!(structure.f.arity(), GATE_ARITY);
    let indices = acceptance_matrix_indices();
    structure
        .f
        .terms()
        .iter()
        .filter_map(|term| {
            let active = term
                .exps
                .iter()
                .enumerate()
                .filter(|&(_, &power)| power != 0)
                .map(|(index, &power)| (index, power))
                .collect::<Vec<_>>();
            if active.is_empty() || !active.iter().all(|(index, _)| indices.contains(index)) {
                return None;
            }
            assert!(active.iter().any(|(index, _)| *index != SELECTOR));
            Some(PolynomialTerm {
                coefficient: signed(term.coeff),
                powers: active
                    .into_iter()
                    .map(|(index, power)| VariablePower {
                        role: matrix_role(index),
                        power,
                    })
                    .collect(),
            })
        })
        .collect()
}

fn expected_polynomial() -> Vec<PolynomialTerm> {
    let mut terms = (0..PRODUCT_SLOTS)
        .map(|slot| PolynomialTerm {
            coefficient: 1,
            powers: vec![
                VariablePower {
                    role: MatrixRole::Selector,
                    power: 1,
                },
                VariablePower {
                    role: MatrixRole::ProductLeft(slot),
                    power: 1,
                },
                VariablePower {
                    role: MatrixRole::ProductRight(slot),
                    power: 1,
                },
            ],
        })
        .collect::<Vec<_>>();
    terms.push(PolynomialTerm {
        coefficient: -1,
        powers: vec![
            VariablePower {
                role: MatrixRole::Selector,
                power: 1,
            },
            VariablePower {
                role: MatrixRole::ProductOut,
                power: 1,
            },
        ],
    });
    for (coefficient, role, power) in [
        (1, MatrixRole::QuadraticBitLeft, 4),
        (-2, MatrixRole::QuadraticBitLeft, 3),
        (1, MatrixRole::QuadraticBitLeft, 2),
        (-7, MatrixRole::QuadraticBitRight, 4),
        (14, MatrixRole::QuadraticBitRight, 3),
        (-7, MatrixRole::QuadraticBitRight, 2),
    ] {
        terms.push(PolynomialTerm {
            coefficient,
            powers: vec![
                VariablePower {
                    role: MatrixRole::Selector,
                    power: 1,
                },
                VariablePower { role, power },
            ],
        });
    }
    terms
}

pub(super) fn extract(
    encoded: &EncodedGadgetNativeR1cs,
    trace: &R1csEncodingTrace,
    chunks: &mut [ChunkGeometry],
) -> LoweredAudit {
    assert_eq!(chunks.len(), CHUNKS);
    assert_eq!(trace.acceptance_chunks().len(), CHUNKS);
    assert_eq!(encoded.structure.matrices.len(), GATE_ARITY);
    let indexed_rows = index_rows(&encoded.structure);
    let mut representative = None;
    let mut previous_end = None;
    for (chunk, geometry) in chunks.iter_mut().enumerate() {
        let audit = encoded
            .plan
            .aggregate_acceptance_audit(chunk)
            .expect("role-specific aggregate-acceptance lowering audit");
        let mut expected_weight = F::ONE;
        for &weight in &audit.radix_weights {
            assert_eq!(weight, expected_weight, "radix-3 weight drift at chunk {chunk}");
            expected_weight *= F::from_u64(3);
        }
        let expected = expected_rows(geometry, &audit.radix_weights);
        let outputs = &geometry.encoded_acceptance_columns[1..];
        let mut global_rows = (0..7)
            .map(|pair| {
                unique_role_row(
                    &encoded.structure,
                    QUADRATIC_BIT_LEFT,
                    outputs[2 * pair],
                    "tree bit-pair",
                )
            })
            .collect::<Vec<_>>();
        global_rows.push(unique_role_row(
            &encoded.structure,
            PRODUCT_OUT,
            outputs[0],
            "radix-3 product aggregate",
        ));
        global_rows.push(unique_role_row(
            &encoded.structure,
            PRODUCT_LEFT,
            outputs[6],
            "root binding",
        ));
        assert_eq!(global_rows.len(), ACTIVE_ROWS_PER_CHUNK);
        assert_eq!(
            global_rows.iter().copied().collect::<BTreeSet<_>>().len(),
            ACTIVE_ROWS_PER_CHUNK
        );
        let start = global_rows[0];
        assert_eq!(global_rows, (start..start + ACTIVE_ROWS_PER_CHUNK).collect::<Vec<_>>());
        if let Some(end) = previous_end {
            assert_eq!(
                start, end,
                "acceptance chunks must occupy one contiguous global row range"
            );
        }
        previous_end = Some(start + ACTIVE_ROWS_PER_CHUNK);
        geometry.active_row_start = start;
        geometry.active_row_end = start + ACTIVE_ROWS_PER_CHUNK;

        for (family_row, (&global, expected_row)) in global_rows.iter().zip(&expected).enumerate() {
            assert_eq!(
                indexed_rows[global], *expected_row,
                "active row family {family_row} drift at chunk {chunk}"
            );
        }
        let normalized = expected
            .iter()
            .map(|row| normalize_row(row, geometry))
            .collect::<Vec<_>>();
        if let Some(rows) = &representative {
            assert_eq!(rows, &normalized, "active row role drift at chunk {chunk}");
        } else {
            representative = Some(normalized);
        }
    }

    let matrix_bindings = matrix_bindings();
    assert_eq!(matrix_bindings.len(), 1 + 2 * PRODUCT_SLOTS + 3);
    let polynomial_terms = polynomial_schema(&encoded.structure);
    assert_eq!(polynomial_terms, expected_polynomial());
    LoweredAudit {
        matrix_bindings,
        active_rows: representative.expect("representative active rows"),
        polynomial_terms,
    }
}
