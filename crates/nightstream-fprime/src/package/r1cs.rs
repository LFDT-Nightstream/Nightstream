use wip_spartan::SparseMatrix;

use super::{expand_r1cs, LoadedPackage, PackageError, SpartanField};

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

    pub(super) fn into_spartan(self) -> Result<SparseMatrix<SpartanField>, PackageError> {
        let data = self
            .values
            .into_iter()
            .map(SpartanField::from_canonical_u64)
            .collect();
        SparseMatrix::from_csr(self.rows, self.columns, data, self.column_indices, self.row_offsets)
            .map_err(|error| PackageError::Spartan(format!("CSR: {error:?}")))
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
        Ok(Self {
            a: self
                .a
                .into_final_layout(unpadded_constant, public_columns, domain_size)?,
            b: self
                .b
                .into_final_layout(unpadded_constant, public_columns, domain_size)?,
            c: self
                .c
                .into_final_layout(unpadded_constant, public_columns, domain_size)?,
        })
    }
}

impl LoadedPackage {
    /// Expand the Lean-emitted schedule and apply the exact final `2^25`
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
