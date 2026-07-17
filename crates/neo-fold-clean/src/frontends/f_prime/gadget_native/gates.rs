//! Fixed CCS gate schema and row builder for gadget-native lowering.
//!
//! Owns: matrix-role indices, sparse row assembly, and the fixed polynomial.
//!
//! Does not own: source-trace validation or gadget witness materialization.
//!
//! Emits constraints: yes, by materializing the final CCS relation.
//!
//! Authority boundary: callers may emit a custom row only after its source
//! trace has been validated by the mathematical owner of that row family.
//!
//! | Role family | Mathematical obligation | Rust owner | Lean owner |
//! |---|---|---|---|
//! | Common value roots | Boolean tails and centered-unit tails | `coordinate_gates` | `BooleanPairRows` / `ResidualPairFamilies` |
//! | Product sum | `sum_i l_i*r_i = out` for at most 18 products | this file | `ProductSum` |
//! | Poseidon2 / linear | `x^7 = out` and `lhs = rhs` | parent lowering | existing refinements |
//! | Quadratic bit pair | Two Boolean residuals packed with nonresidue seven | this file | `BooleanPairRows.quadraticZeroPair_iff` |
//! | Centered residual pair | `(l^3-l)^2 - 7(r^3-r)^2 = 0` | `coordinate_gates` | `ResidualPairFamilies.centeredUnitPairHolds_iff` |
//! | One-product residual pair | `(A_l B_l-C_l)^2 - 7(A_r B_r-C_r)^2 = 0` | `slots` | `ResidualPairFamilies.oneProductPairHolds_iff` |
//! | Mod-5 residue pair | Five-value centered residue alphabet | `mod5` | `PackedChunkRows` |

use neo_ccs::{CcsMatrix, CcsStructure, CscMat, SparsePoly, Term};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::{mod5, MAX_PRODUCT_TERMS};

pub const GADGET_NATIVE_RESIDUAL_PAIR_NONRESIDUE: u64 = 7;

/// Stable role indices exposed only for exact matrix/polynomial audits.
#[doc(hidden)]
pub struct GadgetNativeCoordinateGateRoles;

impl GadgetNativeCoordinateGateRoles {
    pub const SELECTOR: usize = gate::SELECTOR;
    pub const PRODUCT_LEFT: usize = gate::PRODUCT_LEFT;
    pub const PRODUCT_RIGHT: usize = gate::PRODUCT_RIGHT;
    pub const PRODUCT_OUT: usize = gate::PRODUCT_OUT;
    pub const PRODUCT_SLOTS: usize = MAX_PRODUCT_TERMS;
    pub const BITNESS: usize = gate::BITNESS;
    pub const CENTERED_UNIT_TAIL: usize = gate::CENTERED_UNIT_TAIL;
    pub const BOOLEAN_PAIR_LEFT: usize = gate::QUADRATIC_BIT_LEFT;
    pub const BOOLEAN_PAIR_RIGHT: usize = gate::QUADRATIC_BIT_RIGHT;
    pub const CENTERED_PAIR_LEFT: usize = gate::CENTERED_PAIR_LEFT;
    pub const CENTERED_PAIR_RIGHT: usize = gate::CENTERED_PAIR_RIGHT;
    pub const ONE_PRODUCT_PAIR_LEFT_A: usize = gate::ONE_PRODUCT_PAIR_LEFT_A;
    pub const ONE_PRODUCT_PAIR_LEFT_B: usize = gate::ONE_PRODUCT_PAIR_LEFT_B;
    pub const ONE_PRODUCT_PAIR_LEFT_C: usize = gate::ONE_PRODUCT_PAIR_LEFT_C;
    pub const ONE_PRODUCT_PAIR_RIGHT_A: usize = gate::ONE_PRODUCT_PAIR_RIGHT_A;
    pub const ONE_PRODUCT_PAIR_RIGHT_B: usize = gate::ONE_PRODUCT_PAIR_RIGHT_B;
    pub const ONE_PRODUCT_PAIR_RIGHT_C: usize = gate::ONE_PRODUCT_PAIR_RIGHT_C;
    pub const MOD5_RESIDUE_LEFT: usize = gate::MOD5_RESIDUE_LEFT;
    pub const MOD5_RESIDUE_RIGHT: usize = gate::MOD5_RESIDUE_RIGHT;
    pub const ARITY: usize = gate::ARITY;
}

pub(super) mod gate {
    pub const SELECTOR: usize = 0;
    pub const BITNESS: usize = 1;
    pub const CENTERED_UNIT_TAIL: usize = 2;
    pub const PRODUCT_LEFT: usize = 3;
    pub const PRODUCT_RIGHT: usize = PRODUCT_LEFT + super::MAX_PRODUCT_TERMS;
    pub const PRODUCT_OUT: usize = PRODUCT_RIGHT + super::MAX_PRODUCT_TERMS;
    pub const SBOX_IN: usize = PRODUCT_OUT + 1;
    pub const SBOX_OUT: usize = SBOX_IN + 1;
    pub const LINEAR_LHS: usize = SBOX_OUT + 1;
    pub const LINEAR_RHS: usize = LINEAR_LHS + 1;
    pub const QUADRATIC_BIT_LEFT: usize = LINEAR_RHS + 1;
    pub const QUADRATIC_BIT_RIGHT: usize = QUADRATIC_BIT_LEFT + 1;
    pub const CENTERED_PAIR_LEFT: usize = QUADRATIC_BIT_RIGHT + 1;
    pub const CENTERED_PAIR_RIGHT: usize = CENTERED_PAIR_LEFT + 1;
    pub const ONE_PRODUCT_PAIR_LEFT_A: usize = CENTERED_PAIR_RIGHT + 1;
    pub const ONE_PRODUCT_PAIR_LEFT_B: usize = ONE_PRODUCT_PAIR_LEFT_A + 1;
    pub const ONE_PRODUCT_PAIR_LEFT_C: usize = ONE_PRODUCT_PAIR_LEFT_B + 1;
    pub const ONE_PRODUCT_PAIR_RIGHT_A: usize = ONE_PRODUCT_PAIR_LEFT_C + 1;
    pub const ONE_PRODUCT_PAIR_RIGHT_B: usize = ONE_PRODUCT_PAIR_RIGHT_A + 1;
    pub const ONE_PRODUCT_PAIR_RIGHT_C: usize = ONE_PRODUCT_PAIR_RIGHT_B + 1;
    pub const MOD5_RESIDUE_LEFT: usize = ONE_PRODUCT_PAIR_RIGHT_C + 1;
    pub const MOD5_RESIDUE_RIGHT: usize = MOD5_RESIDUE_LEFT + 1;
    pub const ARITY: usize = MOD5_RESIDUE_RIGHT + 1;
}

/// One already-validated `A * B = C` relation represented by linear forms.
#[derive(Clone)]
pub(super) struct OneProductResidualTerms {
    pub(super) a: Vec<(usize, F)>,
    pub(super) b: Vec<(usize, F)>,
    pub(super) c: Vec<(usize, F)>,
}

pub(super) struct TraceGateBuilder {
    trips: Vec<Vec<(usize, usize, F)>>,
    pub(super) rows: usize,
}

impl TraceGateBuilder {
    pub(super) fn new() -> Self {
        Self {
            trips: (0..gate::ARITY).map(|_| Vec::new()).collect(),
            rows: 0,
        }
    }

    pub(super) fn bitness(&mut self, column: usize) {
        let row = self.begin_row(one_selector());
        self.trips[gate::BITNESS].push((row, column, F::ONE));
    }

    pub(super) fn centered_unit_tail(&mut self, column: usize) {
        let row = self.begin_row(one_selector());
        self.trips[gate::CENTERED_UNIT_TAIL].push((row, column, F::ONE));
    }

    pub(super) fn product_sum(
        &mut self,
        selector: Vec<(usize, F)>,
        products: Vec<(Vec<(usize, F)>, Vec<(usize, F)>)>,
        out: Vec<(usize, F)>,
    ) {
        assert!(products.len() <= MAX_PRODUCT_TERMS);
        let row = self.begin_row(selector);
        for (index, (left, right)) in products.into_iter().enumerate() {
            self.push_terms(gate::PRODUCT_LEFT + index, row, left);
            self.push_terms(gate::PRODUCT_RIGHT + index, row, right);
        }
        self.push_terms(gate::PRODUCT_OUT, row, out);
    }

    pub(super) fn sbox7(&mut self, selector: Vec<(usize, F)>, input: Vec<(usize, F)>, out: Vec<(usize, F)>) {
        let row = self.begin_row(selector);
        self.push_terms(gate::SBOX_IN, row, input);
        self.push_terms(gate::SBOX_OUT, row, out);
    }

    pub(super) fn linear(&mut self, selector: Vec<(usize, F)>, lhs: Vec<(usize, F)>, rhs: Vec<(usize, F)>) {
        let row = self.begin_row(selector);
        self.push_terms(gate::LINEAR_LHS, row, lhs);
        self.push_terms(gate::LINEAR_RHS, row, rhs);
    }

    pub(super) fn quadratic_bit_pair(&mut self, left: Vec<(usize, F)>, right: Vec<(usize, F)>) {
        let row = self.begin_row(one_selector());
        self.push_terms(gate::QUADRATIC_BIT_LEFT, row, left);
        self.push_terms(gate::QUADRATIC_BIT_RIGHT, row, right);
    }

    pub(super) fn centered_unit_pair(&mut self, left: usize, right: usize) {
        let row = self.begin_row(one_selector());
        self.trips[gate::CENTERED_PAIR_LEFT].push((row, left, F::ONE));
        self.trips[gate::CENTERED_PAIR_RIGHT].push((row, right, F::ONE));
    }

    pub(super) fn one_product_residual_pair(&mut self, left: OneProductResidualTerms, right: OneProductResidualTerms) {
        let row = self.begin_row(one_selector());
        self.push_terms(gate::ONE_PRODUCT_PAIR_LEFT_A, row, left.a);
        self.push_terms(gate::ONE_PRODUCT_PAIR_LEFT_B, row, left.b);
        self.push_terms(gate::ONE_PRODUCT_PAIR_LEFT_C, row, left.c);
        self.push_terms(gate::ONE_PRODUCT_PAIR_RIGHT_A, row, right.a);
        self.push_terms(gate::ONE_PRODUCT_PAIR_RIGHT_B, row, right.b);
        self.push_terms(gate::ONE_PRODUCT_PAIR_RIGHT_C, row, right.c);
    }

    pub(super) fn mod5_residue_pair(&mut self, left: Vec<(usize, F)>, right: Vec<(usize, F)>) {
        let row = self.begin_row(one_selector());
        self.push_terms(gate::MOD5_RESIDUE_LEFT, row, left);
        self.push_terms(gate::MOD5_RESIDUE_RIGHT, row, right);
    }

    fn begin_row(&mut self, selector: Vec<(usize, F)>) -> usize {
        let row = self.rows;
        self.rows += 1;
        self.push_terms(gate::SELECTOR, row, selector);
        row
    }

    fn push_terms(&mut self, matrix: usize, row: usize, terms: Vec<(usize, F)>) {
        self.trips[matrix].extend(
            terms
                .into_iter()
                .filter(|(_, coefficient)| *coefficient != F::ZERO)
                .map(|(column, coefficient)| (row, column, coefficient)),
        );
    }

    pub(super) fn finish(self, columns: usize) -> CcsStructure<F> {
        let matrices = self
            .trips
            .into_iter()
            .map(|trips| CcsMatrix::Csc(CscMat::from_triplets(trips, self.rows, columns)))
            .collect();
        let mut terms = Vec::with_capacity(MAX_PRODUCT_TERMS + 32);
        terms.push(poly_term(F::ONE, &[(gate::SELECTOR, 1), (gate::BITNESS, 2)]));
        terms.push(poly_term(-F::ONE, &[(gate::SELECTOR, 1), (gate::BITNESS, 1)]));
        terms.push(poly_term(F::ONE, &[(gate::SELECTOR, 1), (gate::CENTERED_UNIT_TAIL, 3)]));
        terms.push(poly_term(
            -F::ONE,
            &[(gate::SELECTOR, 1), (gate::CENTERED_UNIT_TAIL, 1)],
        ));
        for index in 0..MAX_PRODUCT_TERMS {
            terms.push(poly_term(
                F::ONE,
                &[
                    (gate::SELECTOR, 1),
                    (gate::PRODUCT_LEFT + index, 1),
                    (gate::PRODUCT_RIGHT + index, 1),
                ],
            ));
        }
        terms.push(poly_term(-F::ONE, &[(gate::SELECTOR, 1), (gate::PRODUCT_OUT, 1)]));
        terms.push(poly_term(F::ONE, &[(gate::SELECTOR, 1), (gate::SBOX_IN, 7)]));
        terms.push(poly_term(-F::ONE, &[(gate::SELECTOR, 1), (gate::SBOX_OUT, 1)]));
        terms.push(poly_term(F::ONE, &[(gate::SELECTOR, 1), (gate::LINEAR_LHS, 1)]));
        terms.push(poly_term(-F::ONE, &[(gate::SELECTOR, 1), (gate::LINEAR_RHS, 1)]));
        append_boolean_pair_polynomial_terms(&mut terms);
        append_centered_pair_polynomial_terms(&mut terms);
        append_one_product_pair_polynomial_terms(&mut terms);
        mod5::append_residue_polynomial_terms(&mut terms);
        let polynomial = SparsePoly::new(gate::ARITY, terms);
        CcsStructure::new_sparse(matrices, polynomial).expect("gadget-native CCS is well formed")
    }

    /// Normalize the exact rows already emitted into this builder without
    /// allocating CSC matrices. The outer vector is physical row order; each
    /// row contains only nonempty `(matrix, terms)` images.
    pub(super) fn into_sparse_rows(self) -> Vec<Vec<(usize, Vec<(usize, F)>)>> {
        let mut rows = (0..self.rows)
            .map(|_| (0..gate::ARITY).map(|_| Vec::new()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        for (matrix, trips) in self.trips.into_iter().enumerate() {
            for (row, column, coefficient) in trips {
                rows[row][matrix].push((column, coefficient));
            }
        }
        rows.into_iter()
            .map(|matrices| {
                matrices
                    .into_iter()
                    .enumerate()
                    .filter_map(|(matrix, mut terms)| {
                        terms.sort_unstable_by_key(|&(column, _)| column);
                        let mut normalized = Vec::<(usize, F)>::with_capacity(terms.len());
                        for (column, coefficient) in terms {
                            if let Some((last_column, last_coefficient)) = normalized.last_mut() {
                                if *last_column == column {
                                    *last_coefficient += coefficient;
                                    continue;
                                }
                            }
                            normalized.push((column, coefficient));
                        }
                        normalized.retain(|(_, coefficient)| *coefficient != F::ZERO);
                        (!normalized.is_empty()).then_some((matrix, normalized))
                    })
                    .collect()
            })
            .collect()
    }
}

fn append_boolean_pair_polynomial_terms(terms: &mut Vec<Term<F>>) {
    let selector = gate::SELECTOR;
    let left = gate::QUADRATIC_BIT_LEFT;
    let right = gate::QUADRATIC_BIT_RIGHT;
    let nonresidue = GADGET_NATIVE_RESIDUAL_PAIR_NONRESIDUE;
    terms.extend([
        poly_term(F::ONE, &[(selector, 1), (left, 4)]),
        poly_term(-F::from_u64(2), &[(selector, 1), (left, 3)]),
        poly_term(F::ONE, &[(selector, 1), (left, 2)]),
        poly_term(-F::from_u64(nonresidue), &[(selector, 1), (right, 4)]),
        poly_term(F::from_u64(2 * nonresidue), &[(selector, 1), (right, 3)]),
        poly_term(-F::from_u64(nonresidue), &[(selector, 1), (right, 2)]),
    ]);
}

fn append_centered_pair_polynomial_terms(terms: &mut Vec<Term<F>>) {
    let selector = gate::SELECTOR;
    let left = gate::CENTERED_PAIR_LEFT;
    let right = gate::CENTERED_PAIR_RIGHT;
    let nonresidue = GADGET_NATIVE_RESIDUAL_PAIR_NONRESIDUE;
    terms.extend([
        poly_term(F::ONE, &[(selector, 1), (left, 6)]),
        poly_term(-F::from_u64(2), &[(selector, 1), (left, 4)]),
        poly_term(F::ONE, &[(selector, 1), (left, 2)]),
        poly_term(-F::from_u64(nonresidue), &[(selector, 1), (right, 6)]),
        poly_term(F::from_u64(2 * nonresidue), &[(selector, 1), (right, 4)]),
        poly_term(-F::from_u64(nonresidue), &[(selector, 1), (right, 2)]),
    ]);
}

fn append_one_product_pair_polynomial_terms(terms: &mut Vec<Term<F>>) {
    let selector = gate::SELECTOR;
    let left_a = gate::ONE_PRODUCT_PAIR_LEFT_A;
    let left_b = gate::ONE_PRODUCT_PAIR_LEFT_B;
    let left_c = gate::ONE_PRODUCT_PAIR_LEFT_C;
    let right_a = gate::ONE_PRODUCT_PAIR_RIGHT_A;
    let right_b = gate::ONE_PRODUCT_PAIR_RIGHT_B;
    let right_c = gate::ONE_PRODUCT_PAIR_RIGHT_C;
    let nonresidue = GADGET_NATIVE_RESIDUAL_PAIR_NONRESIDUE;
    terms.extend([
        poly_term(F::ONE, &[(selector, 1), (left_a, 2), (left_b, 2)]),
        poly_term(-F::from_u64(2), &[(selector, 1), (left_a, 1), (left_b, 1), (left_c, 1)]),
        poly_term(F::ONE, &[(selector, 1), (left_c, 2)]),
        poly_term(-F::from_u64(nonresidue), &[(selector, 1), (right_a, 2), (right_b, 2)]),
        poly_term(
            F::from_u64(2 * nonresidue),
            &[(selector, 1), (right_a, 1), (right_b, 1), (right_c, 1)],
        ),
        poly_term(-F::from_u64(nonresidue), &[(selector, 1), (right_c, 2)]),
    ]);
}

pub(super) fn one_selector() -> Vec<(usize, F)> {
    vec![(0, F::ONE)]
}

pub(super) fn poly_term(coefficient: F, powers: &[(usize, u32)]) -> Term<F> {
    let mut exps = vec![0u32; gate::ARITY];
    for &(matrix, power) in powers {
        exps[matrix] = power;
    }
    Term {
        coeff: coefficient,
        exps,
    }
}
