//! CCS structure for one `enc(F')` step that hosts an R1CS app circuit.
//!
//! Reuses every row the shared F' shell structure
//! ([`crate::frontends::f_prime::structure::build_f_prime_structure`])
//! emits (bit-validity, ring-action shell, state-out / public-x_out
//! digest bindings, selector, Poseidon transitions). On top of the
//! shell we append exactly `r1cs.n()` product rows — one per R1CS
//! constraint — that enforce
//! `(A_i · z_app) * (B_i · z_app) = (C_i · z_app)`, where each variable
//! `z_app[j]` is recomposed from its 64 committed bits in the
//! `app_private` region via `lane_terms(slot)`.

use neo_ccs::{sparse_r1cs_to_ccs, CcsMatrix};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use crate::frontends::direct_ccs::FrontendError;
use crate::frontends::direct_ccs::R1cs;
use crate::frontends::f_prime::image::FPrimeImageLayout;
use crate::frontends::f_prime::structure::{
    emit_shell_rows, f_prime_lane_slots, lane_terms, FPrimeStructure, LaneSlot, MixedGateBuilder,
};
use crate::paper::relations::Structure;

/// Sparse R1CS shape for large app circuits.
#[derive(Clone, Debug)]
pub struct SparseR1cs {
    pub a: CcsMatrix<F>,
    pub b: CcsMatrix<F>,
    pub c: CcsMatrix<F>,
    pub n: usize,
    pub m: usize,
    pub m_in: usize,
}

impl SparseR1cs {
    pub fn new(
        a: CcsMatrix<F>,
        b: CcsMatrix<F>,
        c: CcsMatrix<F>,
        n: usize,
        m: usize,
        m_in: usize,
    ) -> Result<Self, FrontendError> {
        let out = Self { a, b, c, n, m, m_in };
        out.validate_shape()?;
        Ok(out)
    }

    pub fn validate_shape(&self) -> Result<(), FrontendError> {
        let (ar, ac) = (self.a.rows(), self.a.cols());
        let (br, bc) = (self.b.rows(), self.b.cols());
        let (cr, cc) = (self.c.rows(), self.c.cols());
        if ar != self.n || br != self.n || cr != self.n || ac != self.m || bc != self.m || cc != self.m {
            return Err(FrontendError::ShapeMismatch {
                a_rows: ar,
                a_cols: ac,
                b_rows: br,
                b_cols: bc,
                c_rows: cr,
                c_cols: cc,
            });
        }
        if self.m_in > self.m {
            return Err(FrontendError::PublicInputTooLarge {
                m_in: self.m_in,
                m: self.m,
            });
        }
        Ok(())
    }

    pub fn is_satisfied_by(&self, z: &[F]) -> Result<(), FrontendError> {
        if z.len() != self.m {
            return Err(FrontendError::AssignmentLength {
                got: z.len(),
                expected: self.m,
            });
        }
        let mut az = vec![F::ZERO; self.n];
        let mut bz = vec![F::ZERO; self.n];
        let mut cz = vec![F::ZERO; self.n];
        self.a.add_mul_into(z, &mut az, self.n);
        self.b.add_mul_into(z, &mut bz, self.n);
        self.c.add_mul_into(z, &mut cz, self.n);
        for row in 0..self.n {
            if az[row] * bz[row] != cz[row] {
                return Err(FrontendError::Unsatisfied { row });
            }
        }
        Ok(())
    }

    pub fn to_structure(&self) -> Structure {
        sparse_r1cs_to_ccs(self.a.clone(), self.b.clone(), self.c.clone()).expect("valid sparse R1CS structure")
    }
}

/// R1CS representation accepted by the R1CS-F' compiler.
#[derive(Clone, Debug)]
pub enum R1csShape {
    Dense(R1cs),
    Sparse(SparseR1cs),
}

impl From<R1cs> for R1csShape {
    fn from(value: R1cs) -> Self {
        Self::Dense(value)
    }
}

impl From<&R1cs> for R1csShape {
    fn from(value: &R1cs) -> Self {
        Self::Dense(value.clone())
    }
}

impl From<SparseR1cs> for R1csShape {
    fn from(value: SparseR1cs) -> Self {
        Self::Sparse(value)
    }
}

impl From<&SparseR1cs> for R1csShape {
    fn from(value: &SparseR1cs) -> Self {
        Self::Sparse(value.clone())
    }
}

impl From<&R1csShape> for R1csShape {
    fn from(value: &R1csShape) -> Self {
        value.clone()
    }
}

impl R1csShape {
    pub fn validate_shape(&self) -> Result<(), FrontendError> {
        match self {
            Self::Dense(r1cs) => r1cs.validate_shape(),
            Self::Sparse(r1cs) => r1cs.validate_shape(),
        }
    }

    pub fn m(&self) -> usize {
        match self {
            Self::Dense(r1cs) => r1cs.m(),
            Self::Sparse(r1cs) => r1cs.m,
        }
    }

    pub fn n(&self) -> usize {
        match self {
            Self::Dense(r1cs) => r1cs.n(),
            Self::Sparse(r1cs) => r1cs.n,
        }
    }

    pub fn m_in(&self) -> usize {
        match self {
            Self::Dense(r1cs) => r1cs.m_in,
            Self::Sparse(r1cs) => r1cs.m_in,
        }
    }

    pub fn is_satisfied_by(&self, z: &[F]) -> Result<(), FrontendError> {
        match self {
            Self::Dense(r1cs) => r1cs.is_satisfied_by(z),
            Self::Sparse(r1cs) => r1cs.is_satisfied_by(z),
        }
    }

    pub fn to_structure(&self) -> Structure {
        match self {
            Self::Dense(r1cs) => r1cs.to_structure(),
            Self::Sparse(r1cs) => r1cs.to_structure(),
        }
    }
}

/// Layout anchors returned alongside the [`FPrimeStructure`] when the
/// latter was produced by [`build_r1cs_f_prime_structure`]. Tests use
/// the row-start / row-count fields to confirm each R1CS constraint
/// became its own structure row; the encoder reads `app_var_slots` to
/// fill `app_private` in the right order.
#[derive(Clone, Debug)]
pub struct R1csRowAnchors {
    /// Variable assignment slots: `app_var_slots[j]` is the 64-bit lane
    /// for R1CS variable `z[j]`.
    pub app_var_slots: Vec<LaneSlot>,
    /// First row index of the appended R1CS product block.
    pub r1cs_row_start: usize,
    /// Number of R1CS product rows appended (`= r1cs.n()`).
    pub r1cs_row_count: usize,
}

/// Build the CCS structure for an R1CS app step.
///
/// The layout must already reserve `r1cs.m() * 64` bits inside its
/// `app_private` region (set by sizing `plan.limbs = r1cs.m() * 64 + 1`).
/// Each R1CS variable's 64 bits live contiguously at
/// `layout.app_private.offset + j * 64`.
pub fn build_r1cs_f_prime_structure<R>(layout: FPrimeImageLayout, r1cs: R) -> (FPrimeStructure, R1csRowAnchors)
where
    R: Into<R1csShape>,
{
    let r1cs = r1cs.into();
    let image_end = layout.end;
    assert!(
        image_end >= 2,
        "FPrimeImageLayout::end = {image_end} too small; need constant slot + ≥1 bit column"
    );
    assert_eq!(
        layout.app_private.bits,
        r1cs.m() * POSEIDON2_GOLDILOCKS_BITS,
        "layout.app_private must reserve r1cs.m() * 64 bits (set plan.limbs = r1cs.m() * 64 + 1)"
    );

    let lane_slots = f_prime_lane_slots(&layout);
    let app_var_slots: Vec<LaneSlot> = (0..r1cs.m())
        .map(|j| LaneSlot {
            bit_start: layout.app_private.offset + j * POSEIDON2_GOLDILOCKS_BITS,
        })
        .collect();

    let mut builder = MixedGateBuilder::with_estimated_rows(image_end);
    emit_shell_rows(&layout, &lane_slots, &mut builder);

    let r1cs_row_start = builder.rows();
    append_r1cs_rows(&app_var_slots, &r1cs, &mut builder);
    let r1cs_row_count = builder.rows() - r1cs_row_start;
    debug_assert_eq!(r1cs_row_count, r1cs.n());

    let ccs = builder.finish(image_end);
    let structure = FPrimeStructure {
        layout,
        ccs,
        lane_slots,
    };
    let anchors = R1csRowAnchors {
        app_var_slots,
        r1cs_row_start,
        r1cs_row_count,
    };
    (structure, anchors)
}

/// Append one product row per R1CS constraint. For row `i`:
///
/// ```text
/// (Σ_j A[i,j] · lane_terms(z_j)) ·
/// (Σ_j B[i,j] · lane_terms(z_j))
///   = (Σ_j C[i,j] · lane_terms(z_j))
/// ```
///
/// Each variable's 64 bits are recomposed inline via `lane_terms`; no
/// fresh witness columns are minted.
fn append_r1cs_rows(app_var_slots: &[LaneSlot], r1cs: &R1csShape, builder: &mut MixedGateBuilder) {
    match r1cs {
        R1csShape::Dense(r1cs) => {
            for row in 0..r1cs.n() {
                let left = dense_matrix_row_terms(&r1cs.a, row, app_var_slots);
                let right = dense_matrix_row_terms(&r1cs.b, row, app_var_slots);
                let out = dense_matrix_row_terms(&r1cs.c, row, app_var_slots);
                builder.product(left, right, out);
            }
        }
        R1csShape::Sparse(r1cs) => {
            let left = sparse_matrix_row_terms(&r1cs.a, app_var_slots, r1cs.n);
            let right = sparse_matrix_row_terms(&r1cs.b, app_var_slots, r1cs.n);
            let out = sparse_matrix_row_terms(&r1cs.c, app_var_slots, r1cs.n);
            for row in 0..r1cs.n {
                builder.product(
                    left[row].iter().copied(),
                    right[row].iter().copied(),
                    out[row].iter().copied(),
                );
            }
        }
    }
}

/// Expand one matrix row `M[row, ·]` into `(col, coeff)` terms over the
/// F' bit-frame: each nonzero `M[row, j]` contributes a scaled lane
/// sum `M[row, j] · Σ_i 2^i · z[bit_start_j + i]`.
fn dense_matrix_row_terms(m: &neo_ccs::Mat<F>, row: usize, app_var_slots: &[LaneSlot]) -> Vec<(usize, F)> {
    let mut out: Vec<(usize, F)> = Vec::new();
    for (j, slot) in app_var_slots.iter().enumerate() {
        let coeff = m[(row, j)];
        if coeff != F::ZERO {
            for (col, c) in lane_terms(*slot) {
                out.push((col, c * coeff));
            }
        }
    }
    out
}

fn sparse_matrix_row_terms(m: &CcsMatrix<F>, app_var_slots: &[LaneSlot], rows: usize) -> Vec<Vec<(usize, F)>> {
    let mut out = vec![Vec::new(); rows];
    match m {
        CcsMatrix::Identity { n } => {
            for (row, slot) in app_var_slots.iter().take((*n).min(rows)).enumerate() {
                out[row].extend(lane_terms(*slot));
            }
        }
        CcsMatrix::Csc(csc) => {
            for (col, slot) in app_var_slots.iter().enumerate().take(csc.ncols) {
                let start = csc.col_ptr[col];
                let end = csc.col_ptr[col + 1];
                for idx in start..end {
                    let row = csc.row_idx[idx];
                    if row < rows {
                        let coeff = csc.vals[idx];
                        for (lane_col, lane_coeff) in lane_terms(*slot) {
                            out[row].push((lane_col, lane_coeff * coeff));
                        }
                    }
                }
            }
        }
    }
    out
}
