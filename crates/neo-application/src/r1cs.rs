//! Tagged R1CS construction with authoritative row and gadget catalogs.

use neo_ccs::{sparse_r1cs_to_ccs, CcsMatrix, CcsStructure, CscMat};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::{GadgetDescriptor, GadgetOccurrence};

/// Consumer-defined semantic ownership attached to one constraint row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintTag<Owner> {
    label: &'static str,
    owner: Owner,
}

impl<Owner> ConstraintTag<Owner> {
    pub const fn new(label: &'static str, owner: Owner) -> Self {
        Self { label, owner }
    }

    pub const fn label(&self) -> &'static str {
        self.label
    }

    pub const fn owner(&self) -> &Owner {
        &self.owner
    }
}

/// The three sparse linear expressions in one R1CS row: `(A z) * (B z) = C z`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csRow {
    a_terms: Vec<(usize, F)>,
    b_terms: Vec<(usize, F)>,
    c_terms: Vec<(usize, F)>,
}

impl R1csRow {
    pub fn a_terms(&self) -> &[(usize, F)] {
        &self.a_terms
    }

    pub fn b_terms(&self) -> &[(usize, F)] {
        &self.b_terms
    }

    pub fn c_terms(&self) -> &[(usize, F)] {
        &self.c_terms
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaggedR1csRow<Owner> {
    tag: ConstraintTag<Owner>,
    row: R1csRow,
}

impl<Owner> TaggedR1csRow<Owner> {
    pub const fn tag(&self) -> &ConstraintTag<Owner> {
        &self.tag
    }

    pub const fn row(&self) -> &R1csRow {
        &self.row
    }
}

/// Diagnostic catalog built from the exact rows used to materialize the CCS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintCatalog<Owner> {
    rows: Vec<TaggedR1csRow<Owner>>,
    gadget_occurrences: Vec<GadgetOccurrence<Owner>>,
}

impl<Owner> ConstraintCatalog<Owner> {
    pub fn rows(&self) -> &[TaggedR1csRow<Owner>] {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Structured shared-gadget invocations retained alongside the flat rows.
    pub fn gadget_occurrences(&self) -> &[GadgetOccurrence<Owner>] {
        &self.gadget_occurrences
    }
}

/// A compiled application R1CS together with its verifier-facing public prefix
/// and diagnostic catalog.
#[derive(Clone, Debug)]
pub struct R1csRelation<Owner> {
    structure: CcsStructure<F>,
    public_input_count: usize,
    const_one_column: usize,
    catalog: ConstraintCatalog<Owner>,
}

impl<Owner> R1csRelation<Owner> {
    pub const fn structure(&self) -> &CcsStructure<F> {
        &self.structure
    }

    /// Number of verifier-supplied columns at the start of the assignment.
    ///
    /// The assignment is `z = x || w`, where
    /// `x = z[..public_input_count]` and `w = z[public_input_count..]`.
    pub const fn public_input_count(&self) -> usize {
        self.public_input_count
    }

    pub const fn const_one_column(&self) -> usize {
        self.const_one_column
    }

    pub const fn column_count(&self) -> usize {
        self.structure.m
    }

    pub const fn catalog(&self) -> &ConstraintCatalog<Owner> {
        &self.catalog
    }
}

/// Builds an R1CS relation over the concrete Nightstream field.
#[derive(Clone, Debug)]
pub struct R1csBuilder<Owner> {
    column_count: usize,
    public_input_count: usize,
    const_one_column: usize,
    rows: Vec<TaggedR1csRow<Owner>>,
    gadget_occurrences: Vec<GadgetOccurrence<Owner>>,
}

/// Exclusive view that applies one semantic tag to every emitted row.
///
/// Tags are not hierarchical: [`Self::tagged`] and [`Self::with_tag`] replace
/// this tag for the nested view rather than accumulating a tag path. Once the
/// nested borrow ends, this view keeps its original tag.
pub struct TaggedR1csBuilder<'a, Owner> {
    inner: &'a mut R1csBuilder<Owner>,
    tag: ConstraintTag<Owner>,
}

impl<Owner> R1csBuilder<Owner> {
    /// Create a builder whose assignment is split as `z = x || w`.
    ///
    /// The first `public_input_count` columns are the verifier-supplied `x`;
    /// all remaining columns are the private witness `w`. `const_one_column`
    /// must lie in that public prefix.
    pub fn new(
        column_count: usize,
        public_input_count: usize,
        const_one_column: usize,
    ) -> Result<Self, R1csBuildError> {
        if public_input_count > column_count {
            return Err(R1csBuildError::PublicInputCount {
                count: public_input_count,
                column_count,
            });
        }
        if const_one_column >= column_count {
            return Err(R1csBuildError::ConstantOneOutOfRange {
                column: const_one_column,
                column_count,
            });
        }
        if const_one_column >= public_input_count {
            return Err(R1csBuildError::ConstantOneNotPublic {
                column: const_one_column,
                public_input_count,
            });
        }

        Ok(Self {
            column_count,
            public_input_count,
            const_one_column,
            rows: Vec::new(),
            gadget_occurrences: Vec::new(),
        })
    }

    pub const fn const_one_column(&self) -> usize {
        self.const_one_column
    }

    /// Borrow this builder as a directly usable tagged row emitter.
    pub fn tagged(&mut self, tag: ConstraintTag<Owner>) -> TaggedR1csBuilder<'_, Owner> {
        TaggedR1csBuilder { inner: self, tag }
    }

    /// Emit a scoped family of rows through a borrowing tagged view.
    pub fn with_tag<R>(
        &mut self,
        tag: ConstraintTag<Owner>,
        emit: impl FnOnce(&mut TaggedR1csBuilder<'_, Owner>) -> R,
    ) -> R
    where
        Owner: Clone,
    {
        let mut tagged = self.tagged(tag);
        emit(&mut tagged)
    }

    fn push_row(
        &mut self,
        tag: ConstraintTag<Owner>,
        a_terms: impl IntoIterator<Item = (usize, F)>,
        b_terms: impl IntoIterator<Item = (usize, F)>,
        c_terms: impl IntoIterator<Item = (usize, F)>,
    ) -> &mut Self {
        self.rows.push(TaggedR1csRow {
            tag,
            row: R1csRow {
                a_terms: a_terms.into_iter().collect(),
                b_terms: b_terms.into_iter().collect(),
                c_terms: c_terms.into_iter().collect(),
            },
        });
        self
    }

    fn push_linear_zero(
        &mut self,
        tag: ConstraintTag<Owner>,
        terms: impl IntoIterator<Item = (usize, F)>,
    ) -> &mut Self {
        self.push_row(tag, terms, [(self.const_one_column, F::ONE)], [])
    }

    fn push_boolean(&mut self, tag: ConstraintTag<Owner>, column: usize) -> &mut Self {
        self.push_row(
            tag,
            [(column, F::ONE)],
            [(column, F::ONE), (self.const_one_column, -F::ONE)],
            [],
        )
    }

    pub fn build(self) -> Result<R1csRelation<Owner>, R1csBuildError> {
        if self.rows.is_empty() {
            return Err(R1csBuildError::EmptyRelation);
        }

        let row_count = self.rows.len();
        let column_count = self.column_count;
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();

        for (row_index, tagged) in self.rows.iter().enumerate() {
            append_terms(&mut a, row_index, tagged.row.a_terms(), column_count, R1csSide::A)?;
            append_terms(&mut b, row_index, tagged.row.b_terms(), column_count, R1csSide::B)?;
            append_terms(&mut c, row_index, tagged.row.c_terms(), column_count, R1csSide::C)?;
        }

        let a = CcsMatrix::Csc(CscMat::from_triplets(a, row_count, column_count));
        let b = CcsMatrix::Csc(CscMat::from_triplets(b, row_count, column_count));
        let c = CcsMatrix::Csc(CscMat::from_triplets(c, row_count, column_count));
        let structure = sparse_r1cs_to_ccs(a, b, c).expect("builder constructs matching R1CS matrices");

        Ok(R1csRelation {
            structure,
            public_input_count: self.public_input_count,
            const_one_column: self.const_one_column,
            catalog: ConstraintCatalog {
                rows: self.rows,
                gadget_occurrences: self.gadget_occurrences,
            },
        })
    }
}

impl<Owner: Clone> TaggedR1csBuilder<'_, Owner> {
    pub(crate) fn next_row_index(&self) -> usize {
        self.inner.rows.len()
    }

    pub(crate) fn record_gadget(&mut self, descriptor: GadgetDescriptor, first_row: usize) {
        let row_range = first_row..self.inner.rows.len();
        debug_assert!(!row_range.is_empty());
        self.inner
            .gadget_occurrences
            .push(GadgetOccurrence::new(self.tag.clone(), descriptor, row_range));
    }

    pub fn push_row(
        &mut self,
        a_terms: impl IntoIterator<Item = (usize, F)>,
        b_terms: impl IntoIterator<Item = (usize, F)>,
        c_terms: impl IntoIterator<Item = (usize, F)>,
    ) -> &mut Self {
        self.inner
            .push_row(self.tag.clone(), a_terms, b_terms, c_terms);
        self
    }

    pub fn push_linear_zero(&mut self, terms: impl IntoIterator<Item = (usize, F)>) -> &mut Self {
        self.inner.push_linear_zero(self.tag.clone(), terms);
        self
    }

    pub fn push_boolean(&mut self, column: usize) -> &mut Self {
        self.inner.push_boolean(self.tag.clone(), column);
        self
    }

    pub const fn const_one_column(&self) -> usize {
        self.inner.const_one_column()
    }

    /// Reborrow the underlying builder with a replacement tag.
    ///
    /// The new tag does not contain or otherwise accumulate this view's tag.
    pub fn tagged(&mut self, tag: ConstraintTag<Owner>) -> TaggedR1csBuilder<'_, Owner> {
        TaggedR1csBuilder {
            inner: &mut *self.inner,
            tag,
        }
    }

    /// Emit a nested row family under a replacement tag.
    ///
    /// The nested tag does not accumulate this view's tag, and this view's tag
    /// is unchanged when the closure returns.
    pub fn with_tag<R>(
        &mut self,
        tag: ConstraintTag<Owner>,
        emit: impl FnOnce(&mut TaggedR1csBuilder<'_, Owner>) -> R,
    ) -> R {
        let mut tagged = self.tagged(tag);
        emit(&mut tagged)
    }
}

fn append_terms(
    triplets: &mut Vec<(usize, usize, F)>,
    row: usize,
    terms: &[(usize, F)],
    column_count: usize,
    side: R1csSide,
) -> Result<(), R1csBuildError> {
    for &(column, coefficient) in terms {
        if column >= column_count {
            return Err(R1csBuildError::TermOutOfRange {
                row,
                side,
                column,
                column_count,
            });
        }
        triplets.push((row, column, coefficient));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum R1csSide {
    A,
    B,
    C,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum R1csBuildError {
    #[error("public input count {count} exceeds relation column count {column_count}")]
    PublicInputCount { count: usize, column_count: usize },
    #[error("constant-one column {column} is outside relation column count {column_count}")]
    ConstantOneOutOfRange { column: usize, column_count: usize },
    #[error("constant-one column {column} is outside the public input prefix of length {public_input_count}")]
    ConstantOneNotPublic {
        column: usize,
        public_input_count: usize,
    },
    #[error("R1CS relation requires at least one row")]
    EmptyRelation,
    #[error("row {row} side {side:?} references column {column}, outside relation column count {column_count}")]
    TermOutOfRange {
        row: usize,
        side: R1csSide,
        column: usize,
        column_count: usize,
    },
}
