//! Exact export and bit-backed CCS lowering of field-native R1CS synthesis.
//!
//! Owns only column normalization: constant one first, caller-selected
//! public outputs next, then every remaining synthesized wire in original
//! allocation order. Matrix rows and coefficients are preserved exactly.
//! The fixed-shape branch compiler models HyperNova's single augmented
//! relation: base rows are selected by `is_base`, recursive rows by its
//! complement, and both branches share one public output prefix.

use neo_ccs::{CcsMatrix, CcsStructure, CscMat, SparsePoly, Term};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use crate::engine::r1cs_circuit::builder::{
    BalancedTernaryDecomposition, CenteredUnitTrace, PolynomialEvaluationTrace, Poseidon2PermutationTrace,
    Poseidon2SboxTrace, ProductFactorTrace, ProductSumBatchTrace, ProductSumIdentityTrace,
};
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::frontends::direct_ccs::FrontendError;
use crate::frontends::f_prime::structure::MixedGateBuilder;
use crate::frontends::r1cs_f_prime::SparseR1cs;
use crate::paper::relations::Structure;

const BALANCED_TERNARY_FIELD_WIDTH: usize = 41;

/// Sparse relation and matching assignment produced from one synthesis.
#[derive(Debug)]
pub struct LoweredFieldR1cs {
    shape: SparseR1cs,
    assignment: Vec<F>,
}

impl LoweredFieldR1cs {
    pub fn shape(&self) -> &SparseR1cs {
        &self.shape
    }

    pub fn assignment(&self) -> &[F] {
        &self.assignment
    }

    pub fn into_parts(self) -> (SparseR1cs, Vec<F>) {
        (self.shape, self.assignment)
    }
}

#[derive(Debug, Error)]
pub enum FieldR1csLoweringError {
    #[error("field-R1CS lowering: constant-one wire cannot also be listed as a public output")]
    ConstantOneIsImplicit,
    #[error("field-R1CS lowering: public output column {col} is outside synthesized width {cols}")]
    PublicOutputOutOfRange { col: usize, cols: usize },
    #[error("field-R1CS lowering: public output column {col} was listed more than once")]
    DuplicatePublicOutput { col: usize },
    #[error(transparent)]
    Shape(#[from] FrontendError),
    #[error(transparent)]
    SeededPhi81(#[from] neo_ccs::SeededPhi81Error),
}

/// Low-norm CCS encoding of one exact sparse R1CS relation.
///
/// The committed assignment is `[1 || bits(z_1) || ... || bits(z_m)]`;
/// R1CS column zero reuses the CCS constant column and is not duplicated.
/// Public R1CS columns remain first, so `public_input_len` is the exact CCS
/// prefix length after variable-width bit encoding.
#[derive(Debug)]
pub struct LowNormR1cs {
    structure: Structure,
    assignment: Vec<F>,
    public_input_len: usize,
    field_widths: Vec<usize>,
    field_slots: Vec<Option<(usize, usize)>>,
}

/// Selected arm of HyperNova's fixed-shape augmented relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedR1csBranch {
    Base,
    Recursive,
}

/// One low-norm CCS structure containing both base and recursive R1CS arms.
///
/// Public field columns are shared. A caller-selected application-private
/// prefix may also be shared; all remaining private columns occupy disjoint
/// bit regions and the inactive region is filled with zeroes. The CCS
/// polynomial gates every base row by `is_base` and every recursive row by
/// `1 - is_base`, exactly modeling one fixed-shape conditional relation.
#[derive(Debug)]
pub struct FixedShapeLowNormR1cs {
    structure: Structure,
    public_input_len: usize,
    selector_col: usize,
    public_field_count: usize,
    base_slots: Vec<Option<(usize, usize)>>,
    recursive_slots: Vec<Option<(usize, usize)>>,
    base_aliases: Vec<Option<(usize, usize)>>,
    recursive_aliases: Vec<Option<(usize, usize)>>,
}

/// One low-norm CCS structure containing an arbitrary fixed set of
/// one-hot-selected field-R1CS arms.
///
/// Road A uses three arms: base, recursive with an empty accumulator, and
/// steady recursive with `k_rho` running claims. Public columns and a
/// caller-selected application-private prefix are shared by every arm.
/// Branch-local advice reuses one low-norm arena. Exactly one selector is
/// active, so bitness and centered-digit rows are gated to that arm.
#[derive(Debug)]
pub struct MultiBranchLowNormR1cs {
    structure: Structure,
    public_input_len: usize,
    selector_cols: Vec<usize>,
    public_field_count: usize,
    arm_slots: Vec<Vec<Option<(usize, usize)>>>,
    arm_aliases: Vec<Vec<Option<(usize, usize)>>>,
    arm_equal_aliases: Vec<Vec<Option<usize>>>,
    arm_centered_columns: Vec<Vec<bool>>,
    arm_derived_product_sums: Vec<Vec<DerivedProductSumEncoding>>,
}

#[derive(Debug)]
pub(crate) struct DerivedProductSumEncoding {
    pub(crate) slot: (usize, usize),
    pub(crate) factors: Vec<ProductFactorTrace>,
    pub(crate) previous: Option<usize>,
}

impl MultiBranchLowNormR1cs {
    pub(crate) fn from_compiler_parts(
        structure: Structure,
        public_input_len: usize,
        selector_cols: Vec<usize>,
        public_field_count: usize,
        arm_slots: Vec<Vec<Option<(usize, usize)>>>,
        arm_aliases: Vec<Vec<Option<(usize, usize)>>>,
        arm_equal_aliases: Vec<Vec<Option<usize>>>,
        arm_centered_columns: Vec<Vec<bool>>,
        arm_derived_product_sums: Vec<Vec<DerivedProductSumEncoding>>,
    ) -> Self {
        Self {
            structure,
            public_input_len,
            selector_cols,
            public_field_count,
            arm_slots,
            arm_aliases,
            arm_equal_aliases,
            arm_centered_columns,
            arm_derived_product_sums,
        }
    }

    pub fn structure(&self) -> &Structure {
        &self.structure
    }

    pub fn public_input_len(&self) -> usize {
        self.public_input_len
    }

    pub fn selector_cols(&self) -> &[usize] {
        &self.selector_cols
    }

    pub fn field_slot(&self, arm: usize, field_col: usize) -> Option<(usize, usize)> {
        self.arm_slots.get(arm)?.get(field_col).copied().flatten()
    }

    pub fn encode(&self, arm: usize, field_assignment: &[F]) -> Result<Vec<F>, LowNormR1csError> {
        let slots = self
            .arm_slots
            .get(arm)
            .ok_or(LowNormR1csError::ArmIndexOutOfRange {
                arm,
                arms: self.arm_slots.len(),
            })?;
        if field_assignment.len() != slots.len() {
            return Err(LowNormR1csError::AssignmentLength {
                got: field_assignment.len(),
                expected: slots.len(),
            });
        }
        if field_assignment.first().copied() != Some(F::ONE) {
            return Err(LowNormR1csError::ConstantOne);
        }

        let mut assignment = vec![F::ZERO; self.structure.m];
        assignment[0] = F::ONE;
        assignment[self.selector_cols[arm]] = F::ONE;
        for col in 1..self.public_field_count {
            if slots[col].is_none() {
                continue;
            }
            if let Some(source) = self.arm_equal_aliases[arm][col] {
                if field_assignment[col] != field_assignment[source] {
                    return Err(LowNormR1csError::AliasedFieldMismatch {
                        field_col: col,
                        source_col: source,
                    });
                }
                continue;
            }
            write_encoded_value(
                &mut assignment,
                slots[col],
                self.arm_aliases[arm][col],
                self.arm_centered_columns[arm][col],
                field_assignment[col],
                col,
            )?;
        }
        for col in self.public_field_count..slots.len() {
            if slots[col].is_none() {
                continue;
            }
            if let Some(source) = self.arm_equal_aliases[arm][col] {
                if field_assignment[col] != field_assignment[source] {
                    return Err(LowNormR1csError::AliasedFieldMismatch {
                        field_col: col,
                        source_col: source,
                    });
                }
                continue;
            }
            write_encoded_value(
                &mut assignment,
                slots[col],
                self.arm_aliases[arm][col],
                self.arm_centered_columns[arm][col],
                field_assignment[col],
                col,
            )?;
        }
        let mut derived_values = Vec::with_capacity(self.arm_derived_product_sums[arm].len());
        for derived in &self.arm_derived_product_sums[arm] {
            let mut value = derived.factors.iter().fold(F::ZERO, |sum, factor| {
                sum + factor.coefficient
                    * eval_source_lc(&factor.left, field_assignment)
                    * eval_source_lc(&factor.right, field_assignment)
            });
            if let Some(previous) = derived.previous {
                value += derived_values[previous];
            }
            write_encoded_value(&mut assignment, Some(derived.slot), None, false, value, usize::MAX)?;
            derived_values.push(value);
        }
        Ok(assignment)
    }

    pub fn is_satisfied(&self, assignment: &[F]) -> bool {
        is_structure_satisfied(&self.structure, assignment)
    }

    pub(crate) fn first_unsatisfied_row(&self, assignment: &[F]) -> Option<usize> {
        first_unsatisfied_structure_row(&self.structure, assignment)
    }

    pub(crate) fn into_structure(self) -> Structure {
        self.structure
    }
}

impl FixedShapeLowNormR1cs {
    pub fn structure(&self) -> &Structure {
        &self.structure
    }

    pub fn public_input_len(&self) -> usize {
        self.public_input_len
    }

    pub fn selector_col(&self) -> usize {
        self.selector_col
    }

    /// Bit slot occupied by one source field column in the selected arm.
    /// Shared application columns return the same slot for both branches.
    pub fn field_slot(&self, branch: FixedR1csBranch, field_col: usize) -> Option<(usize, usize)> {
        match branch {
            FixedR1csBranch::Base => self.base_slots.get(field_col).copied().flatten(),
            FixedR1csBranch::Recursive => self.recursive_slots.get(field_col).copied().flatten(),
        }
    }

    /// Encode one selected branch. The inactive branch needs no dummy
    /// witness because all of its semantic rows are selector-gated.
    pub fn encode(&self, branch: FixedR1csBranch, field_assignment: &[F]) -> Result<Vec<F>, LowNormR1csError> {
        let slots = match branch {
            FixedR1csBranch::Base => &self.base_slots,
            FixedR1csBranch::Recursive => &self.recursive_slots,
        };
        if field_assignment.len() != slots.len() {
            return Err(LowNormR1csError::AssignmentLength {
                got: field_assignment.len(),
                expected: slots.len(),
            });
        }
        if field_assignment.first().copied() != Some(F::ONE) {
            return Err(LowNormR1csError::ConstantOne);
        }

        let mut assignment = vec![F::ZERO; self.structure.m];
        assignment[0] = F::ONE;
        assignment[self.selector_col] = match branch {
            FixedR1csBranch::Base => F::ONE,
            FixedR1csBranch::Recursive => F::ZERO,
        };
        let aliases = match branch {
            FixedR1csBranch::Base => &self.base_aliases,
            FixedR1csBranch::Recursive => &self.recursive_aliases,
        };
        for col in 1..self.public_field_count {
            write_encoded_value(
                &mut assignment,
                slots[col],
                aliases[col],
                false,
                field_assignment[col],
                col,
            )?;
        }
        for col in self.public_field_count..slots.len() {
            write_encoded_value(
                &mut assignment,
                slots[col],
                aliases[col],
                false,
                field_assignment[col],
                col,
            )?;
        }
        Ok(assignment)
    }

    pub fn is_satisfied(&self, assignment: &[F]) -> bool {
        is_structure_satisfied(&self.structure, assignment)
    }
}

impl LowNormR1cs {
    pub fn structure(&self) -> &Structure {
        &self.structure
    }

    pub fn assignment(&self) -> &[F] {
        &self.assignment
    }

    pub fn public_input_len(&self) -> usize {
        self.public_input_len
    }

    pub fn field_widths(&self) -> &[usize] {
        &self.field_widths
    }

    pub fn field_slot(&self, field_col: usize) -> Option<(usize, usize)> {
        self.field_slots.get(field_col).copied().flatten()
    }

    pub fn is_satisfied(&self, assignment: &[F]) -> bool {
        if assignment.len() != self.structure.m {
            return false;
        }
        let mut matrix_z = vec![vec![F::ZERO; self.structure.n]; self.structure.matrices.len()];
        for (matrix, values) in self.structure.matrices.iter().zip(matrix_z.iter_mut()) {
            matrix.add_mul_into(assignment, values, self.structure.n);
        }
        (0..self.structure.n).all(|row| {
            let point: Vec<F> = matrix_z.iter().map(|values| values[row]).collect();
            self.structure.f.eval(&point) == F::ZERO
        })
    }

    pub fn into_parts(self) -> (Structure, Vec<F>, usize) {
        (self.structure, self.assignment, self.public_input_len)
    }
}

#[derive(Debug, Error)]
pub enum LowNormR1csError {
    #[error(transparent)]
    Shape(#[from] FrontendError),
    #[error("low-norm R1CS lowering: public prefix must begin with the implicit constant-one column")]
    MissingPublicConstant,
    #[error("low-norm fixed-shape lowering: base public field count {base} != recursive count {recursive}")]
    PublicInputArityMismatch { base: usize, recursive: usize },
    #[error(
        "low-norm fixed-shape lowering: public field column {col} has base width {base} but recursive width {recursive}"
    )]
    PublicWidthMismatch {
        col: usize,
        base: usize,
        recursive: usize,
    },
    #[error(
        "low-norm fixed-shape lowering: shared private prefix {requested} exceeds base/recursive private fields ({base}/{recursive})"
    )]
    SharedPrivatePrefixTooLong {
        requested: usize,
        base: usize,
        recursive: usize,
    },
    #[error(
        "low-norm fixed-shape lowering: shared private field offset {offset} has base width {base} but recursive width {recursive}"
    )]
    SharedPrivateWidthMismatch {
        offset: usize,
        base: usize,
        recursive: usize,
    },
    #[error("low-norm multi-branch lowering requires at least two arms, got {0}")]
    TooFewArms(usize),
    #[error("low-norm multi-branch lowering: arm {arm} public field count {actual} != {expected}")]
    ArmPublicInputArity {
        arm: usize,
        actual: usize,
        expected: usize,
    },
    #[error(
        "low-norm multi-branch lowering: arm {arm} field column {col} has width {actual} but arm 0 has {expected}"
    )]
    ArmFieldWidth {
        arm: usize,
        col: usize,
        actual: usize,
        expected: usize,
    },
    #[error("low-norm multi-branch lowering: arm {arm} has {actual} private fields, needs shared prefix {required}")]
    ArmSharedPrefixTooLong {
        arm: usize,
        actual: usize,
        required: usize,
    },
    #[error("low-norm multi-branch lowering: arm index {arm} is outside 0..{arms}")]
    ArmIndexOutOfRange { arm: usize, arms: usize },
    #[error("low-norm multi-branch lowering: shared-prefix alignment modulus must be nonzero")]
    ZeroAlignmentModulus,
    #[error("low-norm R1CS lowering: assignment length {got} != relation width {expected}")]
    AssignmentLength { got: usize, expected: usize },
    #[error("low-norm R1CS lowering: R1CS constant column is not one")]
    ConstantOne,
    #[error("selective low-norm compiler: {0}")]
    SelectiveTrace(String),
    #[error(transparent)]
    SeededPhi81(#[from] neo_ccs::SeededPhi81Error),
    #[error("low-norm R1CS lowering: column {col} value {value} does not fit inferred width {width}")]
    InferredWidthViolation {
        col: usize,
        width: usize,
        value: u64,
    },
    #[error(
        "low-norm R1CS lowering: canonical bit column {bit_col} disagrees with bit {bit} of source field column {field_col}"
    )]
    AliasedBitMismatch {
        field_col: usize,
        bit_col: usize,
        bit: usize,
    },
    #[error("low-norm R1CS lowering: field column {field_col} disagrees with equal source column {source_col}")]
    AliasedFieldMismatch { field_col: usize, source_col: usize },
    #[error("low-norm R1CS lowering: field column {col} does not fit the balanced-ternary field encoding")]
    BalancedTernaryOverflow { col: usize },
}

/// Normalize one live field-native synthesis to the same
/// `[1 || public_outputs || private allocation order]` assignment used by
/// [`lower_field_r1cs`], without rebuilding its sparse matrices. The
/// authoritative F' relation was compiled from those deterministic matrices;
/// per-step proving only needs the matching witness column order.
pub(crate) fn normalized_field_assignment(
    builder: &R1csBuilder,
    public_outputs: &[Var],
) -> Result<Vec<F>, FieldR1csLoweringError> {
    normalized_field_assignment_with_columns(builder, public_outputs).map(|(assignment, _)| assignment)
}

pub(crate) fn normalized_field_assignment_with_columns(
    builder: &R1csBuilder,
    public_outputs: &[Var],
) -> Result<(Vec<F>, Vec<usize>), FieldR1csLoweringError> {
    let witness = builder.witness();
    let cols = witness.len();
    let mut selected = vec![false; cols];
    selected[Var::ONE.col()] = true;
    let mut old_columns = Vec::with_capacity(cols);
    old_columns.push(Var::ONE.col());
    for output in public_outputs {
        let col = output.col();
        if col == Var::ONE.col() {
            return Err(FieldR1csLoweringError::ConstantOneIsImplicit);
        }
        if col >= cols {
            return Err(FieldR1csLoweringError::PublicOutputOutOfRange { col, cols });
        }
        if selected[col] {
            return Err(FieldR1csLoweringError::DuplicatePublicOutput { col });
        }
        selected[col] = true;
        old_columns.push(col);
    }
    old_columns.extend((1..cols).filter(|&col| !selected[col]));
    let assignment = old_columns.iter().map(|&col| witness[col]).collect();
    Ok((assignment, old_columns))
}

/// Preserve one synthesized field-native relation while normalizing its
/// public prefix to `[1 || public_outputs]`.
///
/// `public_outputs` is ordered: that exact order becomes the sparse R1CS
/// public input after the implicit constant-one column. All private columns
/// retain their relative allocation order.
pub fn lower_field_r1cs(
    builder: R1csBuilder,
    public_outputs: &[Var],
) -> Result<LoweredFieldR1cs, FieldR1csLoweringError> {
    let synthesis = builder.into_synthesis();
    let cols = synthesis.witness.len();
    let mut selected = vec![false; cols];
    selected[Var::ONE.col()] = true;

    let mut old_columns = Vec::with_capacity(cols);
    old_columns.push(Var::ONE.col());
    for output in public_outputs {
        let col = output.col();
        if col == Var::ONE.col() {
            return Err(FieldR1csLoweringError::ConstantOneIsImplicit);
        }
        if col >= cols {
            return Err(FieldR1csLoweringError::PublicOutputOutOfRange { col, cols });
        }
        if selected[col] {
            return Err(FieldR1csLoweringError::DuplicatePublicOutput { col });
        }
        selected[col] = true;
        old_columns.push(col);
    }
    old_columns.extend((1..cols).filter(|&col| !selected[col]));

    let mut old_to_new = vec![0usize; cols];
    let mut assignment = Vec::with_capacity(cols);
    for (new_col, old_col) in old_columns.into_iter().enumerate() {
        old_to_new[old_col] = new_col;
        assignment.push(synthesis.witness[old_col]);
    }

    let remap = |trips: Vec<(usize, usize, F)>| {
        trips
            .into_iter()
            .map(|(row, old_col, value)| (row, old_to_new[old_col], value))
            .collect()
    };
    let canonical_u64_decompositions = synthesis
        .canonical_u64_decompositions
        .iter()
        .map(
            |decomposition| crate::engine::r1cs_circuit::builder::CanonicalU64Decomposition {
                field_col: old_to_new[decomposition.field_col],
                bit_cols: decomposition.bit_cols.map(|col| old_to_new[col]),
            },
        )
        .collect();
    let balanced_ternary_decompositions = synthesis
        .balanced_ternary_decompositions
        .iter()
        .map(|decomposition| BalancedTernaryDecomposition {
            field_col: old_to_new[decomposition.field_col],
            digit_cols: decomposition.digit_cols.map(|col| old_to_new[col]),
        })
        .collect();
    let boolean_columns = synthesis
        .boolean_columns
        .iter()
        .map(|&old_col| old_to_new[old_col])
        .collect();
    let centered_unit_columns = synthesis
        .centered_unit_columns
        .iter()
        .map(|&old_col| old_to_new[old_col])
        .collect();
    let centered_unit_traces = synthesis
        .centered_unit_traces
        .iter()
        .map(|trace| CenteredUnitTrace {
            row_start: trace.row_start,
            row_end: trace.row_end,
            allocated_columns: trace
                .allocated_columns
                .iter()
                .map(|&old_col| old_to_new[old_col])
                .collect(),
            value_col: old_to_new[trace.value_col],
        })
        .collect();
    let equality_pairs = synthesis
        .equality_pairs
        .iter()
        .map(|&(row, lhs, rhs)| (row, old_to_new[lhs], old_to_new[rhs]))
        .collect();
    let remap_lc = |lc: &Lc| Lc {
        terms: lc
            .terms
            .iter()
            .map(|&(old_col, coefficient)| (old_to_new[old_col], coefficient))
            .collect(),
        constant: lc.constant,
    };
    let poseidon2_traces = synthesis
        .poseidon2_traces
        .iter()
        .map(|trace| Poseidon2PermutationTrace {
            row_start: trace.row_start,
            row_end: trace.row_end,
            input_cols: trace.input_cols.map(|old_col| old_to_new[old_col]),
            allocated_columns: trace
                .allocated_columns
                .iter()
                .map(|&old_col| old_to_new[old_col])
                .collect(),
            sboxes: trace
                .sboxes
                .iter()
                .map(|sbox| Poseidon2SboxTrace {
                    input: remap_lc(&sbox.input),
                    output_col: old_to_new[sbox.output_col],
                })
                .collect(),
            output_cols: trace.output_cols.map(|old_col| old_to_new[old_col]),
            output_linear_forms: core::array::from_fn(|lane| remap_lc(&trace.output_linear_forms[lane])),
        })
        .collect();
    let polynomial_evaluation_traces = synthesis
        .polynomial_evaluation_traces
        .iter()
        .map(|trace| PolynomialEvaluationTrace {
            row_start: trace.row_start,
            row_end: trace.row_end,
            allocated_columns: trace
                .allocated_columns
                .iter()
                .map(|&old_col| old_to_new[old_col])
                .collect(),
            coefficient_cols: trace
                .coefficient_cols
                .iter()
                .map(|&old_col| old_to_new[old_col])
                .collect(),
            power_cols: trace
                .power_cols
                .iter()
                .map(|cols| cols.map(|old_col| old_to_new[old_col]))
                .collect(),
            output_cols: trace.output_cols.map(|old_col| old_to_new[old_col]),
        })
        .collect();
    let product_sum_batch_traces = synthesis
        .product_sum_batch_traces
        .iter()
        .map(|trace| ProductSumBatchTrace {
            row_start: trace.row_start,
            row_end: trace.row_end,
            allocated_columns: trace
                .allocated_columns
                .iter()
                .map(|&old_col| old_to_new[old_col])
                .collect(),
            retained_columns: trace
                .retained_columns
                .iter()
                .map(|&old_col| old_to_new[old_col])
                .collect(),
            identities: trace
                .identities
                .iter()
                .map(|identity| ProductSumIdentityTrace {
                    factors: identity
                        .factors
                        .iter()
                        .map(|factor| ProductFactorTrace {
                            left: remap_lc(&factor.left),
                            right: remap_lc(&factor.right),
                            coefficient: factor.coefficient,
                        })
                        .collect(),
                    result: remap_lc(&identity.result),
                })
                .collect(),
        })
        .collect();
    let seeded_phi81_a_blocks = synthesis
        .seeded_phi81_a_blocks
        .iter()
        .map(|block| {
            let starts = block
                .word_starts()
                .iter()
                .map(|&old_start| {
                    let new_start = old_to_new[old_start];
                    assert!((0..block.word_width()).all(|offset| old_to_new[old_start + offset] == new_start + offset));
                    new_start
                })
                .collect();
            block.with_geometry(block.row_start(), starts)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let a = CcsMatrix::csc_with_seeded_phi81(
        CscMat::from_triplets(remap(synthesis.a_trips), synthesis.rows, cols),
        seeded_phi81_a_blocks,
    )?;
    let shape = SparseR1cs::new_with_canonical_u64_decompositions(
        a,
        CcsMatrix::Csc(CscMat::from_triplets(remap(synthesis.b_trips), synthesis.rows, cols)),
        CcsMatrix::Csc(CscMat::from_triplets(remap(synthesis.c_trips), synthesis.rows, cols)),
        synthesis.rows,
        cols,
        1 + public_outputs.len(),
        canonical_u64_decompositions,
        balanced_ternary_decompositions,
        boolean_columns,
        centered_unit_columns,
        centered_unit_traces,
        equality_pairs,
        poseidon2_traces,
        polynomial_evaluation_traces,
        product_sum_batch_traces,
        synthesis.row_family_ranges,
    )?;

    Ok(LoweredFieldR1cs { shape, assignment })
}

/// Encode a field-native sparse R1CS directly as one low-norm CCS relation.
///
/// This is deliberately **not** the R1CS-F' shell compiler: the input may
/// already be the authoritative augmented F' relation, so wrapping it in a
/// second verifier shell would create `F''`. Each non-constant field column
/// is represented by verifier-inferred-width bits, every committed coordinate
/// is constrained Boolean, and the original product rows are evaluated over
/// inline bit recompositions. Honest full-width encodings are canonical;
/// private bit strings at least the Goldilocks modulus are harmless aliases
/// because both the source relation and this lowering interpret them in `F`.
pub fn lower_sparse_r1cs_to_low_norm(
    shape: &SparseR1cs,
    field_assignment: &[F],
) -> Result<LowNormR1cs, LowNormR1csError> {
    shape.validate_shape()?;
    if shape.m_in == 0 {
        return Err(LowNormR1csError::MissingPublicConstant);
    }
    if field_assignment.len() != shape.m {
        return Err(LowNormR1csError::AssignmentLength {
            got: field_assignment.len(),
            expected: shape.m,
        });
    }
    if field_assignment.first().copied() != Some(F::ONE) {
        return Err(LowNormR1csError::ConstantOne);
    }

    let field_widths = shape.conservative_var_widths();
    debug_assert_eq!(field_widths.len(), shape.m);
    let field_aliases = canonical_bit_aliases(shape, &field_widths, 0);
    let mut slots = vec![None; shape.m];
    let mut cursor = 1usize;
    for col in 1..shape.m {
        assign_field_slot(&mut slots, &field_widths, &field_aliases, col, &mut cursor);
    }
    let mut assignment = vec![F::ZERO; cursor];
    assignment[0] = F::ONE;
    for col in 1..shape.m {
        write_encoded_value(
            &mut assignment,
            slots[col],
            field_aliases[col],
            false,
            field_assignment[col],
            col,
        )?;
    }

    let public_input_len = 1 + field_widths[1..shape.m_in].iter().sum::<usize>();
    let mut builder = MixedGateBuilder::with_estimated_rows(assignment.len() - 1 + shape.n);
    for col in 1..assignment.len() {
        builder.bitness(col);
    }
    let a_rows = encoded_matrix_rows(&shape.a, &slots, shape.n);
    let b_rows = encoded_matrix_rows(&shape.b, &slots, shape.n);
    let c_rows = encoded_matrix_rows(&shape.c, &slots, shape.n);
    for ((a, b), c) in a_rows.into_iter().zip(b_rows).zip(c_rows) {
        builder.product(a, b, c);
    }
    let structure = builder.finish(assignment.len());
    Ok(LowNormR1cs {
        structure,
        assignment,
        public_input_len,
        field_widths,
        field_slots: slots,
    })
}

/// Compile HyperNova's base/recursive conditional into one low-norm CCS
/// structure without wrapping either arm in another F' verifier shell.
///
/// Both source relations must expose the same public field columns with the
/// same inferred bit widths. Their private widths and row counts may differ.
pub fn build_fixed_shape_low_norm_r1cs(
    base: &SparseR1cs,
    recursive: &SparseR1cs,
) -> Result<FixedShapeLowNormR1cs, LowNormR1csError> {
    build_fixed_shape_low_norm_r1cs_with_shared_private_prefix(base, recursive, 0)
}

/// Compile the base/recursive conditional while sharing the first
/// `shared_private_fields` source-private columns between both arms.
///
/// Authoritative Nebula F' uses this for the current `S_mem` assignment:
/// the application relation is identical in both arms and its lane bits must
/// occupy one fixed assignment region so the product commitment remains one
/// linear map. Branch-specific verifier advice stays disjoint.
pub fn build_fixed_shape_low_norm_r1cs_with_shared_private_prefix(
    base: &SparseR1cs,
    recursive: &SparseR1cs,
    shared_private_fields: usize,
) -> Result<FixedShapeLowNormR1cs, LowNormR1csError> {
    base.validate_shape()?;
    recursive.validate_shape()?;
    if base.m_in == 0 || recursive.m_in == 0 {
        return Err(LowNormR1csError::MissingPublicConstant);
    }
    if base.m_in != recursive.m_in {
        return Err(LowNormR1csError::PublicInputArityMismatch {
            base: base.m_in,
            recursive: recursive.m_in,
        });
    }

    let base_widths = base.conservative_var_widths();
    let recursive_widths = recursive.conservative_var_widths();
    for col in 1..base.m_in {
        if base_widths[col] != recursive_widths[col] {
            return Err(LowNormR1csError::PublicWidthMismatch {
                col,
                base: base_widths[col],
                recursive: recursive_widths[col],
            });
        }
    }
    let base_private = base.m - base.m_in;
    let recursive_private = recursive.m - recursive.m_in;
    if shared_private_fields > base_private || shared_private_fields > recursive_private {
        return Err(LowNormR1csError::SharedPrivatePrefixTooLong {
            requested: shared_private_fields,
            base: base_private,
            recursive: recursive_private,
        });
    }
    for offset in 0..shared_private_fields {
        let base_width = base_widths[base.m_in + offset];
        let recursive_width = recursive_widths[recursive.m_in + offset];
        if base_width != recursive_width {
            return Err(LowNormR1csError::SharedPrivateWidthMismatch {
                offset,
                base: base_width,
                recursive: recursive_width,
            });
        }
    }
    let base_aliases = canonical_bit_aliases(base, &base_widths, shared_private_fields);
    let recursive_aliases = canonical_bit_aliases(recursive, &recursive_widths, shared_private_fields);

    let mut cursor = 1usize;
    let mut base_slots = vec![None; base.m];
    let mut recursive_slots = vec![None; recursive.m];
    for col in 1..base.m_in {
        let slot = Some((cursor, base_widths[col]));
        base_slots[col] = slot;
        recursive_slots[col] = slot;
        cursor += base_widths[col];
    }
    let public_input_len = cursor;
    let selector_col = cursor;
    cursor += 1;
    for offset in 0..shared_private_fields {
        let base_col = base.m_in + offset;
        let recursive_col = recursive.m_in + offset;
        let slot = Some((cursor, base_widths[base_col]));
        base_slots[base_col] = slot;
        recursive_slots[recursive_col] = slot;
        cursor += base_widths[base_col];
    }
    for col in base.m_in + shared_private_fields..base.m {
        assign_field_slot(&mut base_slots, &base_widths, &base_aliases, col, &mut cursor);
    }
    for col in recursive.m_in + shared_private_fields..recursive.m {
        assign_field_slot(
            &mut recursive_slots,
            &recursive_widths,
            &recursive_aliases,
            col,
            &mut cursor,
        );
    }

    let structure = build_fixed_shape_structure(base, recursive, &base_slots, &recursive_slots, selector_col, cursor);
    Ok(FixedShapeLowNormR1cs {
        structure,
        public_input_len,
        selector_col,
        public_field_count: base.m_in,
        base_slots,
        recursive_slots,
        base_aliases,
        recursive_aliases,
    })
}

/// Compile several field-R1CS arms into one one-hot-selected low-norm CCS
/// relation. This is the fixed-shape compiler used by folded Road A.
pub fn build_multi_branch_low_norm_r1cs(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
) -> Result<MultiBranchLowNormR1cs, LowNormR1csError> {
    build_multi_branch_low_norm_r1cs_aligned(arms, shared_private_fields, None)
}

/// Same as [`build_multi_branch_low_norm_r1cs`], but insert verifier-pinned
/// zero bits so the shared private prefix starts at `residue (mod modulus)`.
/// Nebula uses this to preserve the source `S_mem` ring-column geometry.
pub fn build_multi_branch_low_norm_r1cs_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<MultiBranchLowNormR1cs, LowNormR1csError> {
    if modulus == 0 {
        return Err(LowNormR1csError::ZeroAlignmentModulus);
    }
    build_multi_branch_low_norm_r1cs_aligned(arms, shared_private_fields, Some((modulus, residue % modulus)))
}

fn build_multi_branch_low_norm_r1cs_aligned(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    alignment: Option<(usize, usize)>,
) -> Result<MultiBranchLowNormR1cs, LowNormR1csError> {
    if arms.len() < 2 {
        return Err(LowNormR1csError::TooFewArms(arms.len()));
    }
    for arm in arms {
        arm.validate_shape()?;
        if arm.m_in == 0 {
            return Err(LowNormR1csError::MissingPublicConstant);
        }
    }
    let public_field_count = arms[0].m_in;
    let widths: Vec<Vec<usize>> = arms
        .iter()
        .map(SparseR1cs::conservative_var_widths)
        .collect();
    for (arm_idx, arm) in arms.iter().enumerate().skip(1) {
        if arm.m_in != public_field_count {
            return Err(LowNormR1csError::ArmPublicInputArity {
                arm: arm_idx,
                actual: arm.m_in,
                expected: public_field_count,
            });
        }
    }
    for col in 1..public_field_count {
        for arm_idx in 1..arms.len() {
            if widths[arm_idx][col] != widths[0][col] {
                return Err(LowNormR1csError::ArmFieldWidth {
                    arm: arm_idx,
                    col,
                    actual: widths[arm_idx][col],
                    expected: widths[0][col],
                });
            }
        }
    }
    for (arm_idx, arm) in arms.iter().enumerate() {
        let private_fields = arm.m - arm.m_in;
        if private_fields < shared_private_fields {
            return Err(LowNormR1csError::ArmSharedPrefixTooLong {
                arm: arm_idx,
                actual: private_fields,
                required: shared_private_fields,
            });
        }
    }
    for offset in 0..shared_private_fields {
        let expected = widths[0][arms[0].m_in + offset];
        for arm_idx in 1..arms.len() {
            let actual = widths[arm_idx][arms[arm_idx].m_in + offset];
            if actual != expected {
                return Err(LowNormR1csError::ArmFieldWidth {
                    arm: arm_idx,
                    col: arms[arm_idx].m_in + offset,
                    actual,
                    expected,
                });
            }
        }
    }
    let arm_aliases: Vec<Vec<Option<(usize, usize)>>> = arms
        .iter()
        .zip(&widths)
        .map(|(arm, arm_widths)| canonical_bit_aliases(arm, arm_widths, shared_private_fields))
        .collect();

    let mut cursor = 1usize;
    let mut arm_slots: Vec<Vec<Option<(usize, usize)>>> = arms.iter().map(|arm| vec![None; arm.m]).collect();
    for col in 1..public_field_count {
        let slot = Some((cursor, widths[0][col]));
        for slots in &mut arm_slots {
            slots[col] = slot;
        }
        cursor += widths[0][col];
    }
    let public_input_len = cursor;
    let selector_cols: Vec<usize> = (0..arms.len())
        .map(|_| {
            let col = cursor;
            cursor += 1;
            col
        })
        .collect();
    let padding_len = alignment
        .map(|(modulus, residue)| (residue + modulus - cursor % modulus) % modulus)
        .unwrap_or(0);
    let zero_padding_cols: Vec<usize> = (cursor..cursor + padding_len).collect();
    cursor += padding_len;
    for offset in 0..shared_private_fields {
        let slot = Some((cursor, widths[0][arms[0].m_in + offset]));
        for (arm_idx, arm) in arms.iter().enumerate() {
            arm_slots[arm_idx][arm.m_in + offset] = slot;
        }
        cursor += widths[0][arms[0].m_in + offset];
    }
    let branch_private_start = cursor;
    let mut branch_private_end = cursor;
    for (arm_idx, arm) in arms.iter().enumerate() {
        let mut arm_cursor = branch_private_start;
        for col in arm.m_in + shared_private_fields..arm.m {
            assign_field_slot(
                &mut arm_slots[arm_idx],
                &widths[arm_idx],
                &arm_aliases[arm_idx],
                col,
                &mut arm_cursor,
            );
        }
        branch_private_end = branch_private_end.max(arm_cursor);
    }
    cursor = branch_private_end;

    let structure = build_multi_branch_structure(arms, &arm_slots, &selector_cols, &zero_padding_cols, cursor);
    Ok(MultiBranchLowNormR1cs {
        structure,
        public_input_len,
        selector_cols,
        public_field_count,
        arm_slots,
        arm_aliases,
        arm_equal_aliases: arms.iter().map(|arm| vec![None; arm.m]).collect(),
        arm_centered_columns: arms.iter().map(|arm| vec![false; arm.m]).collect(),
        arm_derived_product_sums: (0..arms.len()).map(|_| Vec::new()).collect(),
    })
}

fn build_multi_branch_structure(
    arms: &[SparseR1cs],
    arm_slots: &[Vec<Option<(usize, usize)>>],
    selector_cols: &[usize],
    zero_padding_cols: &[usize],
    cols: usize,
) -> Structure {
    const BIT: usize = 0;
    const SELECTOR: usize = 1;
    const A: usize = 2;
    const B: usize = 3;
    const C: usize = 4;
    const SELECTOR_SUM: usize = 5;
    const ARITY: usize = 6;

    let rows = cols + zero_padding_cols.len() + arms.iter().map(|arm| arm.n).sum::<usize>();
    let mut trips: [Vec<(usize, usize, F)>; ARITY] = std::array::from_fn(|_| Vec::new());
    for col in 1..cols {
        trips[BIT].push((col - 1, col, F::ONE));
    }
    let selector_row = cols - 1;
    trips[SELECTOR_SUM].push((selector_row, 0, -F::ONE));
    for &selector in selector_cols {
        trips[SELECTOR_SUM].push((selector_row, selector, F::ONE));
    }
    for (offset, &col) in zero_padding_cols.iter().enumerate() {
        trips[SELECTOR_SUM].push((cols + offset, col, F::ONE));
    }

    let mut row_start = cols + zero_padding_cols.len();
    for (arm_idx, arm) in arms.iter().enumerate() {
        for row in 0..arm.n {
            let target = row_start + row;
            trips[SELECTOR].push((target, selector_cols[arm_idx], F::ONE));
        }
        append_encoded_matrix_triplets(&mut trips[A], &arm.a, &arm_slots[arm_idx], row_start, arm.n);
        append_encoded_matrix_triplets(&mut trips[B], &arm.b, &arm_slots[arm_idx], row_start, arm.n);
        append_encoded_matrix_triplets(&mut trips[C], &arm.c, &arm_slots[arm_idx], row_start, arm.n);
        row_start += arm.n;
    }

    let term = |coefficient: F, powers: &[(usize, u32)]| {
        let mut exps = vec![0u32; ARITY];
        for &(index, power) in powers {
            exps[index] = power;
        }
        Term {
            coeff: coefficient,
            exps,
        }
    };
    let f = SparsePoly::new(
        ARITY,
        vec![
            term(F::ONE, &[(BIT, 2)]),
            term(-F::ONE, &[(BIT, 1)]),
            term(F::ONE, &[(SELECTOR, 1), (A, 1), (B, 1)]),
            term(-F::ONE, &[(SELECTOR, 1), (C, 1)]),
            term(F::ONE, &[(SELECTOR_SUM, 1)]),
        ],
    );
    let matrices = trips
        .into_iter()
        .map(|matrix_trips| CcsMatrix::Csc(CscMat::from_triplets(matrix_trips, rows, cols)))
        .collect();
    CcsStructure::new_sparse(matrices, f).expect("multi-branch low-norm R1CS structure must be well-formed")
}

fn build_fixed_shape_structure(
    base: &SparseR1cs,
    recursive: &SparseR1cs,
    base_slots: &[Option<(usize, usize)>],
    recursive_slots: &[Option<(usize, usize)>],
    selector_col: usize,
    cols: usize,
) -> Structure {
    const BIT: usize = 0;
    const SELECTOR: usize = 1;
    const BASE_A: usize = 2;
    const BASE_B: usize = 3;
    const BASE_C: usize = 4;
    const RECURSIVE_A: usize = 5;
    const RECURSIVE_B: usize = 6;
    const RECURSIVE_C: usize = 7;
    const ARITY: usize = 8;

    let rows = cols - 1 + base.n + recursive.n;
    let mut trips: [Vec<(usize, usize, F)>; ARITY] = std::array::from_fn(|_| Vec::new());
    for col in 1..cols {
        trips[BIT].push((col - 1, col, F::ONE));
    }

    let base_row_start = cols - 1;
    for row in 0..base.n {
        let target = base_row_start + row;
        trips[SELECTOR].push((target, selector_col, F::ONE));
    }
    append_encoded_matrix_triplets(&mut trips[BASE_A], &base.a, base_slots, base_row_start, base.n);
    append_encoded_matrix_triplets(&mut trips[BASE_B], &base.b, base_slots, base_row_start, base.n);
    append_encoded_matrix_triplets(&mut trips[BASE_C], &base.c, base_slots, base_row_start, base.n);

    let recursive_row_start = base_row_start + base.n;
    for row in 0..recursive.n {
        let target = recursive_row_start + row;
        trips[SELECTOR].push((target, selector_col, F::ONE));
    }
    append_encoded_matrix_triplets(
        &mut trips[RECURSIVE_A],
        &recursive.a,
        recursive_slots,
        recursive_row_start,
        recursive.n,
    );
    append_encoded_matrix_triplets(
        &mut trips[RECURSIVE_B],
        &recursive.b,
        recursive_slots,
        recursive_row_start,
        recursive.n,
    );
    append_encoded_matrix_triplets(
        &mut trips[RECURSIVE_C],
        &recursive.c,
        recursive_slots,
        recursive_row_start,
        recursive.n,
    );

    let term = |coefficient: F, powers: &[(usize, u32)]| {
        let mut exps = vec![0u32; ARITY];
        for &(index, power) in powers {
            exps[index] = power;
        }
        Term {
            coeff: coefficient,
            exps,
        }
    };
    let f = SparsePoly::new(
        ARITY,
        vec![
            term(F::ONE, &[(BIT, 2)]),
            term(-F::ONE, &[(BIT, 1)]),
            term(F::ONE, &[(SELECTOR, 1), (BASE_A, 1), (BASE_B, 1)]),
            term(-F::ONE, &[(SELECTOR, 1), (BASE_C, 1)]),
            term(F::ONE, &[(RECURSIVE_A, 1), (RECURSIVE_B, 1)]),
            term(-F::ONE, &[(RECURSIVE_C, 1)]),
            term(-F::ONE, &[(SELECTOR, 1), (RECURSIVE_A, 1), (RECURSIVE_B, 1)]),
            term(F::ONE, &[(SELECTOR, 1), (RECURSIVE_C, 1)]),
        ],
    );
    let matrices = trips
        .into_iter()
        .map(|matrix_trips| CcsMatrix::Csc(CscMat::from_triplets(matrix_trips, rows, cols)))
        .collect();
    CcsStructure::new_sparse(matrices, f).expect("fixed-shape low-norm R1CS structure must be well-formed")
}

/// Stream one field-native sparse matrix directly into its bit-encoded
/// destination. The previous row-materialization path held `n` small vectors
/// for A, B, and C simultaneously, which dominates memory for authoritative
/// F' arms with millions of rows.
fn append_encoded_matrix_triplets(
    out: &mut Vec<(usize, usize, F)>,
    matrix: &CcsMatrix<F>,
    slots: &[Option<(usize, usize)>],
    row_offset: usize,
    rows: usize,
) {
    let mut append = |row: usize, field_col: usize, coefficient: F| {
        if coefficient == F::ZERO || row >= rows {
            return;
        }
        if field_col == 0 {
            out.push((row_offset + row, 0, coefficient));
            return;
        }
        let (start, width) = slots[field_col].expect("every non-constant R1CS column has a bit slot");
        let mut power = coefficient;
        for bit in 0..width {
            out.push((row_offset + row, start + bit, power));
            power += power;
        }
    };

    match matrix {
        CcsMatrix::Identity { n } => {
            for row in 0..(*n).min(rows).min(slots.len()) {
                append(row, row, F::ONE);
            }
        }
        CcsMatrix::Csc(csc) => {
            for col in 0..csc.ncols.min(slots.len()) {
                for index in csc.col_ptr[col]..csc.col_ptr[col + 1] {
                    append(csc.row_idx[index], col, csc.vals[index]);
                }
            }
        }
        CcsMatrix::CscWithSeededPhi81 { csc, blocks } => {
            for col in 0..csc.ncols.min(slots.len()) {
                for index in csc.col_ptr[col]..csc.col_ptr[col + 1] {
                    append(csc.row_idx[index], col, csc.vals[index]);
                }
            }
            for block in blocks {
                block.for_each_term::<F, _>(|row, col, coefficient| append(row, col, coefficient));
            }
        }
    }
}

/// Alias private decomposition children onto an already-committed canonical
/// field slot. Public bit children and the caller-declared shared private
/// prefix keep independent slots so this optimization cannot change the
/// public statement or cross-arm ownership.
fn canonical_bit_aliases(
    shape: &SparseR1cs,
    widths: &[usize],
    shared_private_fields: usize,
) -> Vec<Option<(usize, usize)>> {
    let mut aliases = vec![None; shape.m];
    let shared_end = shape.m_in + shared_private_fields;
    for decomposition in shape.canonical_u64_decompositions() {
        let field_col = decomposition.field_col;
        if field_col == 0 || field_col >= shape.m || widths[field_col] != 64 {
            continue;
        }
        if (shape.m_in..shared_end).contains(&field_col) {
            continue;
        }
        let usable = decomposition.bit_cols.iter().all(|&bit_col| {
            bit_col > field_col
                && bit_col < shape.m
                && bit_col >= shape.m_in
                && !(shape.m_in..shared_end).contains(&bit_col)
                && widths[bit_col] == 1
                && aliases[bit_col].is_none()
        });
        if !usable {
            continue;
        }
        for (bit, &bit_col) in decomposition.bit_cols.iter().enumerate() {
            aliases[bit_col] = Some((field_col, bit));
        }
    }
    aliases
}

fn assign_field_slot(
    slots: &mut [Option<(usize, usize)>],
    widths: &[usize],
    aliases: &[Option<(usize, usize)>],
    field_col: usize,
    cursor: &mut usize,
) {
    if let Some((source_col, bit)) = aliases[field_col] {
        let (source_start, source_width) = slots[source_col].expect("canonical source slot must precede its bits");
        debug_assert_eq!(source_width, 64);
        slots[field_col] = Some((source_start + bit, 1));
        return;
    }
    slots[field_col] = Some((*cursor, widths[field_col]));
    *cursor += widths[field_col];
}

fn write_encoded_value(
    assignment: &mut [F],
    slot: Option<(usize, usize)>,
    alias: Option<(usize, usize)>,
    centered: bool,
    value: F,
    field_col: usize,
) -> Result<(), LowNormR1csError> {
    let (start, width) = slot.expect("every non-constant field column has a bit slot");
    if centered {
        debug_assert_eq!(width, 1);
        assignment[start] = value;
        return Ok(());
    }
    if width == BALANCED_TERNARY_FIELD_WIDTH {
        return write_balanced_ternary(assignment, start, value, field_col);
    }
    let value = value.as_canonical_u64();
    if width < 64 && value >= (1u64 << width) {
        return Err(LowNormR1csError::InferredWidthViolation {
            col: field_col,
            width,
            value,
        });
    }
    if let Some((source_col, bit)) = alias {
        let encoded = F::from_u64(value);
        if assignment[start] != encoded {
            return Err(LowNormR1csError::AliasedBitMismatch {
                field_col: source_col,
                bit_col: field_col,
                bit,
            });
        }
        return Ok(());
    }
    for bit in 0..width {
        assignment[start + bit] = F::from_u64((value >> bit) & 1);
    }
    Ok(())
}

fn write_balanced_ternary(
    assignment: &mut [F],
    start: usize,
    value: F,
    field_col: usize,
) -> Result<(), LowNormR1csError> {
    let modulus = F::ORDER_U64;
    let canonical = value.as_canonical_u64();
    let negative = canonical > modulus / 2;
    let mut remaining = if negative { modulus - canonical } else { canonical };
    for digit_index in 0..BALANCED_TERNARY_FIELD_WIDTH {
        let residue = remaining % 3;
        let positive_digit = match residue {
            0 => F::ZERO,
            1 => F::ONE,
            2 => -F::ONE,
            _ => unreachable!("remainder modulo three"),
        };
        assignment[start + digit_index] = if negative { -positive_digit } else { positive_digit };
        remaining = remaining / 3 + u64::from(residue == 2);
    }
    if remaining != 0 {
        return Err(LowNormR1csError::BalancedTernaryOverflow { col: field_col });
    }
    Ok(())
}

fn eval_source_lc(lc: &Lc, assignment: &[F]) -> F {
    lc.terms
        .iter()
        .fold(lc.constant, |sum, &(column, coefficient)| {
            sum + coefficient * assignment[column]
        })
}

fn encoded_matrix_rows(matrix: &CcsMatrix<F>, slots: &[Option<(usize, usize)>], rows: usize) -> Vec<Vec<(usize, F)>> {
    let mut out = vec![Vec::new(); rows];
    match matrix {
        CcsMatrix::Identity { n } => {
            for row in 0..(*n).min(rows).min(slots.len()) {
                extend_encoded_terms(&mut out[row], row, F::ONE, slots);
            }
        }
        CcsMatrix::Csc(csc) => {
            for col in 0..csc.ncols.min(slots.len()) {
                for idx in csc.col_ptr[col]..csc.col_ptr[col + 1] {
                    let row = csc.row_idx[idx];
                    if row < rows {
                        extend_encoded_terms(&mut out[row], col, csc.vals[idx], slots);
                    }
                }
            }
        }
        CcsMatrix::CscWithSeededPhi81 { csc, blocks } => {
            for col in 0..csc.ncols.min(slots.len()) {
                for idx in csc.col_ptr[col]..csc.col_ptr[col + 1] {
                    let row = csc.row_idx[idx];
                    if row < rows {
                        extend_encoded_terms(&mut out[row], col, csc.vals[idx], slots);
                    }
                }
            }
            for block in blocks {
                block.for_each_term::<F, _>(|row, col, coefficient| {
                    if row < rows {
                        extend_encoded_terms(&mut out[row], col, coefficient, slots);
                    }
                });
            }
        }
    }
    out
}

fn extend_encoded_terms(out: &mut Vec<(usize, F)>, field_col: usize, coefficient: F, slots: &[Option<(usize, usize)>]) {
    if coefficient == F::ZERO {
        return;
    }
    if field_col == 0 {
        out.push((0, coefficient));
        return;
    }
    let (start, width) = slots[field_col].expect("every non-constant R1CS column has a bit slot");
    let mut power = coefficient;
    for bit in 0..width {
        out.push((start + bit, power));
        power += power;
    }
}

fn is_structure_satisfied(structure: &Structure, assignment: &[F]) -> bool {
    first_unsatisfied_structure_row(structure, assignment).is_none()
}

fn first_unsatisfied_structure_row(structure: &Structure, assignment: &[F]) -> Option<usize> {
    if assignment.len() != structure.m {
        return Some(structure.n);
    }
    let mut matrix_z = vec![vec![F::ZERO; structure.n]; structure.matrices.len()];
    for (matrix, values) in structure.matrices.iter().zip(matrix_z.iter_mut()) {
        matrix.add_mul_into(assignment, values, structure.n);
    }
    (0..structure.n).find(|&row| {
        let point: Vec<F> = matrix_z.iter().map(|values| values[row]).collect();
        structure.f.eval(&point) != F::ZERO
    })
}
