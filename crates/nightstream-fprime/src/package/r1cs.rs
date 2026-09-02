use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use super::{
    scheduled_witnesses, ColumnRef, CompactRowInvocation, LoadedPackage, PackageError, ScheduledInvocation,
    ScheduledWitness, SparseCombination, SparseRow, TemplateCombination, TemplateRow, WitnessInstruction,
};

/// One canonical sparse matrix derived from the Lean-emitted row program.
/// Values are canonical Goldilocks words in CSR row order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageSparseMatrix {
    pub(super) rows: usize,
    pub(super) columns: usize,
    pub(super) values: Vec<u64>,
    pub(super) column_indices: Vec<usize>,
    pub(super) row_offsets: Vec<usize>,
}

impl PackageSparseMatrix {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn nonzero_count(&self) -> usize {
        self.values.len()
    }

    pub fn values(&self) -> &[u64] {
        &self.values
    }

    pub fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }

    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    fn into_final_layout(
        mut self,
        unpadded_constant: usize,
        public_columns: usize,
        domain_size: usize,
    ) -> Result<Self, PackageError> {
        let unpadded_columns = unpadded_constant
            .checked_add(1)
            .and_then(|columns| columns.checked_add(public_columns))
            .ok_or(PackageError::Invalid("final matrix column overflow"))?;
        if self.rows > domain_size || unpadded_constant > domain_size || self.columns != unpadded_columns {
            return Err(PackageError::Invalid("final matrix geometry"));
        }

        let shift = domain_size - unpadded_constant;
        for column in &mut self.column_indices {
            if *column >= unpadded_constant {
                *column = column
                    .checked_add(shift)
                    .ok_or(PackageError::Invalid("final matrix column overflow"))?;
            }
        }
        self.row_offsets.resize(domain_size + 1, self.values.len());
        self.rows = domain_size;
        self.columns = domain_size
            .checked_add(1)
            .and_then(|columns| columns.checked_add(public_columns))
            .ok_or(PackageError::Invalid("final matrix column overflow"))?;
        Ok(self)
    }
}

/// Exact R1CS matrices expanded from one identity-checked package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageR1cs {
    pub(super) a: PackageSparseMatrix,
    pub(super) b: PackageSparseMatrix,
    pub(super) c: PackageSparseMatrix,
}

impl PackageR1cs {
    pub fn a(&self) -> &PackageSparseMatrix {
        &self.a
    }

    pub fn b(&self) -> &PackageSparseMatrix {
        &self.b
    }

    pub fn c(&self) -> &PackageSparseMatrix {
        &self.c
    }

    fn into_final_layout(
        self,
        unpadded_constant: usize,
        public_columns: usize,
        domain_size: usize,
    ) -> Result<Self, PackageError> {
        let Self { a, b, c } = self;
        let (a, (b, c)) = rayon::join(
            || a.into_final_layout(unpadded_constant, public_columns, domain_size),
            || {
                rayon::join(
                    || b.into_final_layout(unpadded_constant, public_columns, domain_size),
                    || c.into_final_layout(unpadded_constant, public_columns, domain_size),
                )
            },
        );
        Ok(Self { a: a?, b: b?, c: c? })
    }
}

#[derive(Clone, Copy)]
enum MatrixSide {
    A,
    B,
    C,
}

struct CsrBuilder {
    rows: usize,
    cols: usize,
    data: Vec<u64>,
    indices: Vec<usize>,
    indptr: Vec<usize>,
    scratch: Vec<(usize, Goldilocks)>,
}

impl CsrBuilder {
    fn new(rows: usize, cols: usize, capacity: usize) -> Result<Self, PackageError> {
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = Vec::new();
        data.try_reserve_exact(capacity)
            .map_err(|_| PackageError::Invalid("matrix data allocation"))?;
        indices
            .try_reserve_exact(capacity)
            .map_err(|_| PackageError::Invalid("matrix index allocation"))?;
        indptr
            .try_reserve_exact(rows + 1)
            .map_err(|_| PackageError::Invalid("matrix row allocation"))?;
        indptr.push(0);
        Ok(Self {
            rows,
            cols,
            data,
            indices,
            indptr,
            scratch: Vec::with_capacity(32),
        })
    }

    fn push_template(
        &mut self,
        combination: &TemplateCombination,
        package: &LoadedPackage,
        invocation: ScheduledInvocation<'_>,
    ) {
        self.scratch.clear();
        self.push_term(package.layout.constant_column, combination.constant);
        for term in &combination.terms {
            match term.column {
                ColumnRef::Local(index) => {
                    self.push_term(invocation.witness_start() + index, term.coefficient);
                }
                ColumnRef::Input(lane) => match invocation {
                    ScheduledInvocation::Hash { chain, ordinal, .. } => {
                        if ordinal > 0 {
                            let previous = chain.witness_start
                                + (ordinal - 1) * package.permutation.local_column_count
                                + package.permutation.output_local_start
                                + lane;
                            self.push_term(previous, term.coefficient);
                        }
                        if ordinal < chain.absorb_count {
                            let input_offset = ordinal * 4 + lane;
                            if lane < 4 && input_offset < chain.input_length {
                                self.push_term(chain.input_start + input_offset, term.coefficient);
                            }
                        } else if lane == 0 {
                            self.push_term(package.layout.constant_column, term.coefficient);
                        }
                    }
                    ScheduledInvocation::Explicit(explicit) => {
                        let input = &explicit.inputs[lane];
                        self.push_term(package.layout.constant_column, term.coefficient * input.constant);
                        for input_term in &input.terms {
                            self.push_term(input_term.column, term.coefficient * input_term.coefficient);
                        }
                    }
                },
            }
        }
        self.finish_row();
    }

    fn push_sparse(&mut self, combination: &SparseCombination, constant_column: usize) {
        self.scratch.clear();
        self.push_term(constant_column, combination.constant);
        for term in &combination.terms {
            self.push_term(term.column, term.coefficient);
        }
        self.finish_row();
    }

    fn push_compact(
        &mut self,
        combination: &TemplateCombination,
        constant_column: usize,
        invocation: &CompactRowInvocation,
    ) {
        self.scratch.clear();
        self.push_term(constant_column, combination.constant);
        for term in &combination.terms {
            let column = match term.column {
                ColumnRef::Input(input) => invocation.input_column(input),
                ColumnRef::Local(local) => invocation.local_start + local,
            };
            self.push_term(column, term.coefficient);
        }
        self.finish_row();
    }

    fn push_term(&mut self, column: usize, coefficient: Goldilocks) {
        if coefficient != Goldilocks::ZERO {
            self.scratch.push((column, coefficient));
        }
    }

    fn finish_row(&mut self) {
        self.scratch.sort_unstable_by_key(|term| term.0);
        let mut cursor = 0usize;
        while cursor < self.scratch.len() {
            let column = self.scratch[cursor].0;
            let mut coefficient = Goldilocks::ZERO;
            while cursor < self.scratch.len() && self.scratch[cursor].0 == column {
                coefficient += self.scratch[cursor].1;
                cursor += 1;
            }
            if coefficient != Goldilocks::ZERO {
                self.indices.push(column);
                self.data.push(coefficient.as_canonical_u64());
            }
        }
        self.indptr.push(self.data.len());
    }

    fn finish(self) -> Result<PackageSparseMatrix, PackageError> {
        if self.indptr.len() != self.rows + 1 {
            return Err(PackageError::Invalid("expanded matrix row count"));
        }
        Ok(PackageSparseMatrix {
            rows: self.rows,
            columns: self.cols,
            values: self.data,
            column_indices: self.indices,
            row_offsets: self.indptr,
        })
    }
}

fn expand_r1cs(package: &LoadedPackage) -> Result<PackageR1cs, PackageError> {
    let schedule = scheduled_witnesses(package)?;
    let (a, (b, c)) = rayon::join(
        || expand_matrix(package, &schedule, MatrixSide::A),
        || {
            rayon::join(
                || expand_matrix(package, &schedule, MatrixSide::B),
                || expand_matrix(package, &schedule, MatrixSide::C),
            )
        },
    );
    Ok(PackageR1cs { a: a?, b: b?, c: c? })
}

fn expand_matrix(
    package: &LoadedPackage,
    schedule: &[ScheduledWitness<'_>],
    side: MatrixSide,
) -> Result<PackageSparseMatrix, PackageError> {
    let mut builder = CsrBuilder::new(
        package.layout.row_count,
        package.layout.total_column_count,
        entry_capacity(package, side)?,
    )?;
    let mut row_cursor = 0usize;
    let mut assertion_cursor = 0usize;
    for &witness in schedule {
        while assertion_cursor < package.assertion_rows.len()
            && package.assertion_rows[assertion_cursor].row_index < witness.row_start()
        {
            push_assertion(
                &package.assertion_rows[assertion_cursor],
                package.layout.constant_column,
                side,
                &mut builder,
            );
            assertion_cursor += 1;
            row_cursor += 1;
        }
        if row_cursor != witness.row_start() {
            return Err(PackageError::Invalid("expanded witness row start"));
        }
        match witness {
            ScheduledWitness::Permutation(invocation) => {
                for row in &package.permutation.rows {
                    builder.push_template(template_side(row, side), package, invocation);
                    row_cursor += 1;
                }
            }
            ScheduledWitness::Compact(invocation) => {
                let template = &package.compact_templates[invocation.template_index];
                for row in &template.rows {
                    builder.push_compact(compact_side(row, side), package.layout.constant_column, invocation);
                    row_cursor += 1;
                }
            }
            ScheduledWitness::Generic(instruction) => {
                push_witness_instruction(instruction, package.layout.constant_column, side, &mut builder);
                row_cursor += 1;
            }
        }
    }
    while assertion_cursor < package.assertion_rows.len() {
        push_assertion(
            &package.assertion_rows[assertion_cursor],
            package.layout.constant_column,
            side,
            &mut builder,
        );
        assertion_cursor += 1;
        row_cursor += 1;
    }
    if row_cursor != package.layout.row_count {
        return Err(PackageError::Invalid("expanded physical row count"));
    }
    builder.finish()
}

fn push_assertion(row: &SparseRow, constant_column: usize, side: MatrixSide, builder: &mut CsrBuilder) {
    builder.push_sparse(sparse_side(row, side), constant_column);
}

fn push_witness_instruction(
    instruction: &WitnessInstruction,
    constant_column: usize,
    side: MatrixSide,
    builder: &mut CsrBuilder,
) {
    match side {
        MatrixSide::A => builder.push_sparse(&instruction.a, constant_column),
        MatrixSide::B => builder.push_sparse(&instruction.b, constant_column),
        MatrixSide::C => {
            builder.scratch.clear();
            builder.push_term(instruction.target, Goldilocks::ONE);
            builder.finish_row();
        }
    }
}

fn entry_capacity(package: &LoadedPackage, side: MatrixSide) -> Result<usize, PackageError> {
    let max_input_entries = package
        .permutation_invocations
        .iter()
        .flat_map(|invocation| &invocation.inputs)
        .map(sparse_entry_bound)
        .max()
        .unwrap_or(0)
        .max(2);
    let per_permutation = package
        .permutation
        .rows
        .iter()
        .map(|row| template_entry_bound(template_side(row, side), max_input_entries))
        .try_fold(0usize, |sum, count| sum.checked_add(count))
        .ok_or(PackageError::Invalid("template entry bound overflow"))?;
    let invocation_count = package
        .hash_chains
        .iter()
        .map(|chain| chain.absorb_count + 1)
        .try_fold(0usize, |sum, count| sum.checked_add(count))
        .and_then(|count| count.checked_add(package.permutation_invocations.len()))
        .ok_or(PackageError::Invalid("invocation count overflow"))?;
    let assertion_entries = package
        .assertion_rows
        .iter()
        .map(|row| sparse_entry_bound(sparse_side(row, side)))
        .try_fold(0usize, |sum, count| sum.checked_add(count))
        .ok_or(PackageError::Invalid("assertion entry bound overflow"))?;
    let witness_entries = package
        .witness_instructions
        .iter()
        .map(|instruction| match side {
            MatrixSide::A => sparse_entry_bound(&instruction.a),
            MatrixSide::B => sparse_entry_bound(&instruction.b),
            MatrixSide::C => 1,
        })
        .try_fold(0usize, |sum, count| sum.checked_add(count))
        .ok_or(PackageError::Invalid("witness entry bound overflow"))?;
    let compact_template_entries = package
        .compact_templates
        .iter()
        .map(|template| {
            template
                .rows
                .iter()
                .map(|row| template_entry_bound(compact_side(row, side), 1))
                .try_fold(0usize, |sum, count| sum.checked_add(count))
                .ok_or(PackageError::Invalid("compact template entry bound overflow"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let compact_entries = package
        .compact_invocations
        .iter()
        .map(|invocation| compact_template_entries[invocation.template_index])
        .try_fold(0usize, |sum, count| sum.checked_add(count))
        .ok_or(PackageError::Invalid("compact entry capacity overflow"))?;
    per_permutation
        .checked_mul(invocation_count)
        .and_then(|count| count.checked_add(assertion_entries))
        .and_then(|count| count.checked_add(witness_entries))
        .and_then(|count| count.checked_add(compact_entries))
        .ok_or(PackageError::Invalid("matrix entry capacity overflow"))
}

fn compact_side(row: &super::compact::CompactTemplateRow, side: MatrixSide) -> &TemplateCombination {
    match side {
        MatrixSide::A => &row.a,
        MatrixSide::B => &row.b,
        MatrixSide::C => &row.c,
    }
}

fn template_side(row: &TemplateRow, side: MatrixSide) -> &TemplateCombination {
    match side {
        MatrixSide::A => &row.a,
        MatrixSide::B => &row.b,
        MatrixSide::C => &row.c,
    }
}

fn sparse_side(row: &SparseRow, side: MatrixSide) -> &SparseCombination {
    match side {
        MatrixSide::A => &row.a,
        MatrixSide::B => &row.b,
        MatrixSide::C => &row.c,
    }
}

fn template_entry_bound(combination: &TemplateCombination, input_entries: usize) -> usize {
    usize::from(combination.constant != Goldilocks::ZERO)
        + combination
            .terms
            .iter()
            .map(|term| match term.column {
                ColumnRef::Input(_) => input_entries,
                ColumnRef::Local(_) => 1,
            })
            .sum::<usize>()
}

fn sparse_entry_bound(combination: &SparseCombination) -> usize {
    usize::from(combination.constant != Goldilocks::ZERO) + combination.terms.len()
}

impl LoadedPackage {
    /// Expand the Lean-emitted schedule and apply the exact final `2^28`
    /// row/private-domain padding proved by `Layout.Stage1.Spartan`.
    pub fn r1cs_matrices(&self) -> Result<PackageR1cs, PackageError> {
        let shift =
            u32::try_from(self.relation.cube_variables()).map_err(|_| PackageError::Invalid("final matrix domain"))?;
        let domain_size = 1usize
            .checked_shl(shift)
            .ok_or(PackageError::Invalid("final matrix domain"))?;
        expand_r1cs(self)?.into_final_layout(
            self.layout.constant_column,
            self.layout.public_column_count,
            domain_size,
        )
    }
}
