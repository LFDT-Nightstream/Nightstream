//! Exact executable audit of stage-local coordinate and residual pairing.
//!
//! Owns: a two-stage fixture, schedule order/coverage, exact matrix roles,
//! the three six-term nonresidue-seven polynomials,
//! estimator/materializer parity, witness inversion, canonical-slot pairing,
//! and practical left/right/tail mutation rejection.
//!
//! | Fixture family | Coordinates by stage | Expected rows |
//! |---|---:|---:|
//! | ordinary Boolean | `3 + 2` | `1 pair + 1 tail`, then `1 pair` |
//! | ordinary-private centered | `41 + 41` | `20 pairs + 1 tail` in each stage |
//! | canonical-binary / synthetic | zero | zero, but organizational groups remain visible |
//! | SIS centered residuals | one 41-coordinate opening | 20 pairs + 1 tail |
//! | canonicality residuals | 32 equations per slot | 16 pairs, reset per slot |

use std::collections::BTreeSet;

use neo_ccs::{CcsMatrix, CscMat};
use neo_fold_clean::engine::r1cs_circuit::boolean::enforce_bit;
use neo_fold_clean::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::frontends::f_prime::gadget_native::{
    encode_r1cs_gadget_native, estimate_r1cs_gadget_native, GadgetNativeBooleanFamily, GadgetNativeCenteredFamily,
    GadgetNativeCoordinateGateRoles as Roles, GadgetNativeCoordinateGroupFamily, GadgetNativeCoordinateRowAudit,
    GADGET_NATIVE_RESIDUAL_PAIR_NONRESIDUE,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn staged_fixture() -> (R1csBuilder, Vec<usize>) {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("test.coordinate.stage_a");
    let mut public_bits = Vec::new();
    for value in [F::ZERO, F::ONE, F::ZERO] {
        let bit = builder.alloc(value);
        enforce_bit(&mut builder, bit);
        public_bits.push(bit.col());
    }
    let _field_a = builder.alloc(F::from_u64(0x1234_5678));

    builder.begin_encoding_stage("test.coordinate.stage_b");
    for value in [F::ONE, F::ZERO] {
        let bit = builder.alloc(value);
        enforce_bit(&mut builder, bit);
        public_bits.push(bit.col());
    }
    let _field_b = builder.alloc(F::from_u64(0x9abc_def0));
    builder.begin_encoding_stage("complete");
    assert!(builder.is_satisfied());
    (builder, public_bits)
}

#[test]
fn common_coordinate_schedule_is_exact_stage_local_and_executable() {
    let (builder, public_bits) = staged_fixture();
    let source = builder.snapshot();
    let estimate = estimate_r1cs_gadget_native(&source, builder.encoding_trace(), &public_bits)
        .expect("stage-local coordinate estimate");
    let encoded = encode_r1cs_gadget_native(&source, builder.encoding_trace(), &public_bits)
        .expect("stage-local coordinate lowering");
    let schedule = encoded.plan.coordinate_gate_schedule();

    assert_eq!(estimate.encoded_cols, encoded.structure.m);
    assert_eq!(estimate.encoded_rows, encoded.structure.n);
    assert_eq!(estimate.boolean_pairing, schedule.pairing());
    assert_eq!(encoded.decode_source().expect("exact inverse"), source.witness());
    assert!(encoded.is_satisfied());

    let pairing = schedule.pairing();
    assert_eq!(
        (
            pairing.common.coordinates,
            pairing.common.pair_rows,
            pairing.common.tail_rows
        ),
        (5, 2, 1)
    );
    assert_eq!(
        (
            pairing.source_raw64.coordinates,
            pairing.source_raw64.pair_rows,
            pairing.source_raw64.tail_rows,
        ),
        (0, 0, 0)
    );
    assert_eq!(
        (
            pairing.source_prefix31.coordinates,
            pairing.source_prefix31.pair_rows,
            pairing.source_prefix31.tail_rows,
        ),
        (0, 0, 0)
    );
    for family in [
        pairing.synthetic_ring_raw64,
        pairing.synthetic_ring_prefix31,
        pairing.synthetic_product_sum_raw64,
        pairing.synthetic_product_sum_prefix31,
    ] {
        assert_eq!((family.coordinates, family.pair_rows, family.tail_rows), (0, 0, 0));
    }

    assert_eq!(schedule.groups().len(), 2 * (GadgetNativeBooleanFamily::ALL.len() + 2));
    for (stage, expected_label) in ["test.coordinate.stage_a", "test.coordinate.stage_b"]
        .into_iter()
        .enumerate()
    {
        let groups = &schedule.groups()[stage * 9..(stage + 1) * 9];
        assert!(groups.iter().all(|group| group.stage == expected_label));
        for (group, family) in groups[..7].iter().zip(GadgetNativeBooleanFamily::ALL) {
            assert_eq!(group.family, GadgetNativeCoordinateGroupFamily::Boolean(family));
            assert_eq!(
                group.encoded_rows.len(),
                group.coordinates.len().div_ceil(2),
                "{expected_label} {family:?} row census"
            );
        }
        assert_eq!(
            groups[7].family,
            GadgetNativeCoordinateGroupFamily::CenteredUnit(GadgetNativeCenteredFamily::OrdinaryPrivateField)
        );
        assert_eq!(groups[7].coordinates.len(), 41);
        assert_eq!(groups[7].encoded_rows.len(), 21);
        assert_eq!(
            groups[8].family,
            GadgetNativeCoordinateGroupFamily::CenteredUnit(GadgetNativeCenteredFamily::SisOpening)
        );
        assert!(groups[8].coordinates.is_empty());
        assert!(groups[8].encoded_rows.is_empty());
    }

    let stage_a_common = &schedule.groups()[0];
    let stage_b_common = &schedule.groups()[9];
    assert_eq!(stage_a_common.coordinates, vec![1, 2, 3]);
    assert_eq!(stage_b_common.coordinates, vec![4, 5]);
    assert!(matches!(
        schedule.rows()[stage_a_common.encoded_rows.start],
        GadgetNativeCoordinateRowAudit::BooleanPair { left: 1, right: 2, .. }
    ));
    assert!(matches!(
        schedule.rows()[stage_a_common.encoded_rows.start + 1],
        GadgetNativeCoordinateRowAudit::BooleanTail { coordinate: 3, .. }
    ));
    assert!(matches!(
        schedule.rows()[stage_b_common.encoded_rows.start],
        GadgetNativeCoordinateRowAudit::BooleanPair { left: 4, right: 5, .. }
    ));

    let mut covered = BTreeSet::new();
    for (expected_row, row) in schedule.rows().iter().enumerate() {
        assert_eq!(row.row(), expected_row, "coordinate rows must be gap-free");
        match *row {
            GadgetNativeCoordinateRowAudit::BooleanPair { row, left, right } => {
                assert!(covered.insert(left));
                assert!(covered.insert(right));
                assert_eq!(schedule.row_for_column(left), Some(row));
                assert_eq!(schedule.row_for_column(right), Some(row));
                assert_pair_matrix(&encoded.structure.matrices, row, left, right);
            }
            GadgetNativeCoordinateRowAudit::BooleanTail { row, coordinate } => {
                assert!(covered.insert(coordinate));
                assert_eq!(schedule.row_for_column(coordinate), Some(row));
                assert_single_matrix(&encoded.structure.matrices, row, Roles::BITNESS, coordinate);
            }
            GadgetNativeCoordinateRowAudit::CenteredUnitPair {
                row,
                family,
                left,
                right,
            } => {
                assert!(covered.insert(left));
                assert!(covered.insert(right));
                assert_eq!(schedule.row_for_column(left), Some(row));
                assert_eq!(schedule.row_for_column(right), Some(row));
                assert_eq!(schedule.centered_family_for_column(left), Some(family));
                assert_eq!(schedule.centered_family_for_column(right), Some(family));
                assert_centered_pair_matrix(&encoded.structure.matrices, row, left, right);
            }
            GadgetNativeCoordinateRowAudit::CenteredUnitTail {
                row,
                family,
                coordinate,
            } => {
                assert!(covered.insert(coordinate));
                assert_eq!(schedule.row_for_column(coordinate), Some(row));
                assert_eq!(schedule.centered_family_for_column(coordinate), Some(family));
                assert_single_matrix(&encoded.structure.matrices, row, Roles::CENTERED_UNIT_TAIL, coordinate);
            }
        }
    }
    let logical_public_end = 1 + encoded.plan.public_columns().len();
    let expected = (1..logical_public_end)
        .chain(encoded.plan.public_input_len()..encoded.assignment.len())
        .collect();
    assert_eq!(covered, expected);
    assert_boolean_pair_polynomial(&encoded.structure.f);

    let pair = schedule
        .rows()
        .iter()
        .find_map(|row| match *row {
            GadgetNativeCoordinateRowAudit::BooleanPair { left, right, .. } => Some((left, right)),
            _ => None,
        })
        .expect("at least one pair row");
    let tail = schedule
        .rows()
        .iter()
        .find_map(|row| match *row {
            GadgetNativeCoordinateRowAudit::BooleanTail { coordinate, .. } => Some(coordinate),
            _ => None,
        })
        .expect("odd stage-local tail");
    let mut pair_mutation = encoded.clone();
    pair_mutation.assignment[pair.0] = F::from_u64(2);
    assert!(!pair_mutation.is_satisfied());
    let mut pair_double_mutation = encoded.clone();
    pair_double_mutation.assignment[pair.0] = F::from_u64(2);
    pair_double_mutation.assignment[pair.1] = F::from_u64(3);
    assert!(!pair_double_mutation.is_satisfied());
    let mut tail_mutation = encoded;
    tail_mutation.assignment[tail] = F::from_u64(2);
    assert!(!tail_mutation.is_satisfied());
}

#[test]
fn centered_coordinates_pair_within_their_physical_stage_and_keep_the_odd_tail() {
    let (source, trace, _field) = super::balanced_ternary_relation(F::from_u64(19));
    let estimate = estimate_r1cs_gadget_native(&source, &trace, &[]).expect("centered-pair estimate");
    let encoded = encode_r1cs_gadget_native(&source, &trace, &[]).expect("centered-pair lowering");
    let schedule = encoded.plan.coordinate_gate_schedule();

    assert_eq!(
        schedule.centered_pairing_for(GadgetNativeCenteredFamily::SisOpening),
        neo_fold_clean::frontends::f_prime::gadget_native::GadgetNativePairTailCount {
            coordinates: 41,
            pair_rows: 20,
            tail_rows: 1,
        }
    );
    assert_eq!(estimate.centered_pairing, schedule.centered_pairing());
    assert_eq!(estimate.encoded_rows, encoded.structure.n);
    assert!(encoded.is_satisfied());

    let centered = schedule
        .groups()
        .iter()
        .find(|group| {
            group.family == GadgetNativeCoordinateGroupFamily::CenteredUnit(GadgetNativeCenteredFamily::SisOpening)
                && !group.coordinates.is_empty()
        })
        .expect("one centered family");
    assert_eq!(centered.coordinates.len(), 41);
    assert_eq!(centered.encoded_rows.len(), 21);
    let centered_rows = &schedule.rows()[centered.encoded_rows.clone()];
    for (pair, expected) in centered_rows[..20]
        .iter()
        .zip(centered.coordinates[..40].chunks_exact(2))
    {
        let GadgetNativeCoordinateRowAudit::CenteredUnitPair {
            row,
            family: GadgetNativeCenteredFamily::SisOpening,
            left,
            right,
        } = *pair
        else {
            panic!("first 20 centered rows must be pairs");
        };
        assert_eq!([left, right], [expected[0], expected[1]]);
        assert_centered_pair_matrix(&encoded.structure.matrices, row, left, right);
    }
    let GadgetNativeCoordinateRowAudit::CenteredUnitTail {
        row,
        family: GadgetNativeCenteredFamily::SisOpening,
        coordinate,
    } = centered_rows[20]
    else {
        panic!("41st centered coordinate must be an ordinary tail");
    };
    assert_eq!(coordinate, centered.coordinates[40]);
    assert_single_matrix(&encoded.structure.matrices, row, Roles::CENTERED_UNIT_TAIL, coordinate);
    assert_centered_pair_polynomial(&encoded.structure.f);
    assert!(encoded.balanced_ternary_rows(0).is_ok());

    let mut swapped_roles = encoded.clone();
    swapped_roles
        .structure
        .matrices
        .swap(Roles::CENTERED_PAIR_LEFT, Roles::CENTERED_PAIR_RIGHT);
    assert!(
        swapped_roles.balanced_ternary_rows(0).is_err(),
        "artifact reader must reject a left/right matrix-role swap"
    );

    let GadgetNativeCoordinateRowAudit::CenteredUnitPair { left, right, .. } = centered_rows[0] else {
        unreachable!()
    };
    let mut left_tamper = encoded.clone();
    left_tamper.assignment[left] = F::from_u64(2);
    assert!(!left_tamper.is_satisfied());
    let mut right_tamper = encoded.clone();
    right_tamper.assignment[right] = F::from_u64(2);
    assert!(!right_tamper.is_satisfied());
    let mut tail_tamper = encoded;
    tail_tamper.assignment[coordinate] = F::from_u64(2);
    assert!(!tail_tamper.is_satisfied());
}

#[test]
fn goldilocks_canonicality_pairs_reset_at_each_slot_and_reject_each_side() {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("test.canonical_binary");
    let first = builder.alloc(F::from_u64(7u64 << 32));
    let _ = decompose_var_to_u64_bits(&mut builder, first);
    let second = builder.alloc(F::from_u64(5u64 << 32));
    let _ = decompose_var_to_u64_bits(&mut builder, second);
    builder.begin_encoding_stage("complete");
    let source = builder.snapshot();
    let trace = builder.encoding_trace();
    let estimate = estimate_r1cs_gadget_native(&source, trace, &[]).expect("canonical-pair estimate");
    let encoded = encode_r1cs_gadget_native(&source, trace, &[]).expect("canonical-pair lowering");
    let coordinate_rows = encoded.plan.coordinate_gate_schedule().rows().len();

    assert_eq!(
        encoded.structure.n,
        coordinate_rows + 2 * 16 + estimate.fallback_source_rows
    );
    assert_eq!(estimate.encoded_rows, encoded.structure.n);
    assert!(encoded.is_satisfied());
    for (slot_index, source_column) in [first.col(), second.col()].into_iter().enumerate() {
        let start = encoded
            .plan
            .encoded_range_for_source_column(source_column)
            .expect("canonical slot")
            .start;
        let relations = canonical_relation_terms(start);
        assert_eq!(relations.len(), 32);
        for pair in 0..16 {
            let row = coordinate_rows + slot_index * 16 + pair;
            assert_one_product_pair_matrix(
                &encoded.structure.matrices,
                row,
                &relations[2 * pair],
                &relations[2 * pair + 1],
            );
        }
    }
    assert_one_product_pair_polynomial(&encoded.structure.f);

    let first_start = encoded
        .plan
        .encoded_range_for_source_column(first.col())
        .expect("first canonical slot")
        .start;
    let mut left_tamper = encoded.clone();
    left_tamper.assignment[first_start + 33] = F::ZERO;
    assert!(!left_tamper.is_satisfied(), "left relation must reject independently");
    let mut right_tamper = encoded;
    right_tamper.assignment[first_start + 34] = F::ZERO;
    assert!(!right_tamper.is_satisfied(), "right relation must reject independently");
}

type LinearTerms = Vec<(usize, F)>;
type ProductRelation = (LinearTerms, LinearTerms, LinearTerms);

fn canonical_relation_terms(start: usize) -> Vec<ProductRelation> {
    let aux = start + 64;
    let mut relations = vec![(
        vec![(start + 32, F::ONE)],
        vec![(start + 33, F::ONE)],
        vec![(aux, F::ONE)],
    )];
    for high_offset in 2..32 {
        relations.push((
            vec![(aux + high_offset - 2, F::ONE)],
            vec![(start + 32 + high_offset, F::ONE)],
            vec![(aux + high_offset - 1, F::ONE)],
        ));
    }
    relations.push((
        vec![(aux + 30, F::ONE)],
        (0..32)
            .map(|bit| (start + bit, F::from_u64(1u64 << bit)))
            .collect(),
        Vec::new(),
    ));
    relations
}

fn assert_pair_matrix(matrices: &[CcsMatrix<F>], row: usize, left: usize, right: usize) {
    for (role, matrix) in matrices.iter().enumerate() {
        let terms = matrix_row(matrix, row);
        let expected = if role == Roles::SELECTOR {
            vec![(0, F::ONE)]
        } else if role == Roles::BOOLEAN_PAIR_LEFT {
            vec![(left, F::ONE)]
        } else if role == Roles::BOOLEAN_PAIR_RIGHT {
            vec![(right, F::ONE)]
        } else {
            Vec::new()
        };
        assert_eq!(terms, expected, "matrix role {role} at pair row {row}");
    }
}

fn assert_centered_pair_matrix(matrices: &[CcsMatrix<F>], row: usize, left: usize, right: usize) {
    for (role, matrix) in matrices.iter().enumerate() {
        let terms = matrix_row(matrix, row);
        let expected = if role == Roles::SELECTOR {
            vec![(0, F::ONE)]
        } else if role == Roles::CENTERED_PAIR_LEFT {
            vec![(left, F::ONE)]
        } else if role == Roles::CENTERED_PAIR_RIGHT {
            vec![(right, F::ONE)]
        } else {
            Vec::new()
        };
        assert_eq!(terms, expected, "matrix role {role} at centered pair row {row}");
    }
}

fn assert_one_product_pair_matrix(
    matrices: &[CcsMatrix<F>],
    row: usize,
    left: &ProductRelation,
    right: &ProductRelation,
) {
    for (role, matrix) in matrices.iter().enumerate() {
        let terms = matrix_row(matrix, row);
        let expected = match role {
            Roles::SELECTOR => vec![(0, F::ONE)],
            Roles::ONE_PRODUCT_PAIR_LEFT_A => left.0.clone(),
            Roles::ONE_PRODUCT_PAIR_LEFT_B => left.1.clone(),
            Roles::ONE_PRODUCT_PAIR_LEFT_C => left.2.clone(),
            Roles::ONE_PRODUCT_PAIR_RIGHT_A => right.0.clone(),
            Roles::ONE_PRODUCT_PAIR_RIGHT_B => right.1.clone(),
            Roles::ONE_PRODUCT_PAIR_RIGHT_C => right.2.clone(),
            _ => Vec::new(),
        };
        assert_eq!(terms, expected, "matrix role {role} at one-product pair row {row}");
    }
}

fn assert_single_matrix(matrices: &[CcsMatrix<F>], row: usize, role: usize, coordinate: usize) {
    for (matrix_role, matrix) in matrices.iter().enumerate() {
        let terms = matrix_row(matrix, row);
        let expected = if matrix_role == Roles::SELECTOR {
            vec![(0, F::ONE)]
        } else if matrix_role == role {
            vec![(coordinate, F::ONE)]
        } else {
            Vec::new()
        };
        assert_eq!(terms, expected, "matrix role {matrix_role} at single row {row}");
    }
}

fn matrix_row(matrix: &CcsMatrix<F>, row: usize) -> Vec<(usize, F)> {
    let matrix = matrix
        .as_csc()
        .expect("production gadget-native matrices are CSC");
    csc_row(matrix, row)
}

fn csc_row(matrix: &CscMat<F>, row: usize) -> Vec<(usize, F)> {
    let mut terms = Vec::new();
    for column in 0..matrix.ncols {
        for entry in matrix.column_range(column) {
            if matrix.row_index(entry) == row {
                terms.push((column, matrix.vals[entry]));
            }
        }
    }
    terms
}

fn assert_boolean_pair_polynomial(polynomial: &neo_ccs::SparsePoly<F>) {
    let selector = Roles::SELECTOR;
    let left = Roles::BOOLEAN_PAIR_LEFT;
    let right = Roles::BOOLEAN_PAIR_RIGHT;
    let nonresidue = F::from_u64(GADGET_NATIVE_RESIDUAL_PAIR_NONRESIDUE);
    let expected = [
        (F::ONE, [(selector, 1), (left, 4)]),
        (-F::from_u64(2), [(selector, 1), (left, 3)]),
        (F::ONE, [(selector, 1), (left, 2)]),
        (-nonresidue, [(selector, 1), (right, 4)]),
        (F::from_u64(2) * nonresidue, [(selector, 1), (right, 3)]),
        (-nonresidue, [(selector, 1), (right, 2)]),
    ];
    let actual = polynomial
        .terms()
        .iter()
        .filter(|term| term.exps[left] != 0 || term.exps[right] != 0)
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len());
    for (coefficient, powers) in expected {
        assert!(actual.iter().any(|term| {
            term.coeff == coefficient
                && term.exps.iter().enumerate().all(|(role, &power)| {
                    power
                        == powers
                            .iter()
                            .find_map(|&(expected_role, expected_power)| {
                                (role == expected_role).then_some(expected_power)
                            })
                            .unwrap_or(0)
                })
        }));
    }
}

fn assert_centered_pair_polynomial(polynomial: &neo_ccs::SparsePoly<F>) {
    let selector = Roles::SELECTOR;
    let left = Roles::CENTERED_PAIR_LEFT;
    let right = Roles::CENTERED_PAIR_RIGHT;
    let nonresidue = F::from_u64(GADGET_NATIVE_RESIDUAL_PAIR_NONRESIDUE);
    assert_polynomial_terms(
        polynomial,
        &[left, right],
        &[
            (F::ONE, vec![(selector, 1), (left, 6)]),
            (-F::from_u64(2), vec![(selector, 1), (left, 4)]),
            (F::ONE, vec![(selector, 1), (left, 2)]),
            (-nonresidue, vec![(selector, 1), (right, 6)]),
            (F::from_u64(2) * nonresidue, vec![(selector, 1), (right, 4)]),
            (-nonresidue, vec![(selector, 1), (right, 2)]),
        ],
    );
}

fn assert_one_product_pair_polynomial(polynomial: &neo_ccs::SparsePoly<F>) {
    let selector = Roles::SELECTOR;
    let la = Roles::ONE_PRODUCT_PAIR_LEFT_A;
    let lb = Roles::ONE_PRODUCT_PAIR_LEFT_B;
    let lc = Roles::ONE_PRODUCT_PAIR_LEFT_C;
    let ra = Roles::ONE_PRODUCT_PAIR_RIGHT_A;
    let rb = Roles::ONE_PRODUCT_PAIR_RIGHT_B;
    let rc = Roles::ONE_PRODUCT_PAIR_RIGHT_C;
    let nonresidue = F::from_u64(GADGET_NATIVE_RESIDUAL_PAIR_NONRESIDUE);
    assert_polynomial_terms(
        polynomial,
        &[la, lb, lc, ra, rb, rc],
        &[
            (F::ONE, vec![(selector, 1), (la, 2), (lb, 2)]),
            (-F::from_u64(2), vec![(selector, 1), (la, 1), (lb, 1), (lc, 1)]),
            (F::ONE, vec![(selector, 1), (lc, 2)]),
            (-nonresidue, vec![(selector, 1), (ra, 2), (rb, 2)]),
            (
                F::from_u64(2) * nonresidue,
                vec![(selector, 1), (ra, 1), (rb, 1), (rc, 1)],
            ),
            (-nonresidue, vec![(selector, 1), (rc, 2)]),
        ],
    );
}

fn assert_polynomial_terms(
    polynomial: &neo_ccs::SparsePoly<F>,
    family_roles: &[usize],
    expected: &[(F, Vec<(usize, u32)>)],
) {
    let actual = polynomial
        .terms()
        .iter()
        .filter(|term| family_roles.iter().any(|&role| term.exps[role] != 0))
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len());
    for (coefficient, powers) in expected {
        assert!(actual.iter().any(|term| {
            term.coeff == *coefficient
                && term.exps.iter().enumerate().all(|(role, &power)| {
                    power
                        == powers
                            .iter()
                            .find_map(|&(expected_role, expected_power)| {
                                (role == expected_role).then_some(expected_power)
                            })
                            .unwrap_or(0)
                })
        }));
    }
}
