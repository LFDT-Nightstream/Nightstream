//! CCS structure for one `enc(F')` step that hosts an R1CS app circuit.
//!
//! Reuses every row the shared F' shell structure
//! ([`crate::frontends::f_prime_shell::structure::build_f_prime_shell_structure`])
//! emits (bit-validity, ring-action shell, state-out / public-x_out
//! digest bindings, selector, Poseidon transitions). On top of the
//! shell we append exactly `r1cs.n()` product rows — one per R1CS
//! constraint — that enforce
//! `(A_i · z_app) * (B_i · z_app) = (C_i · z_app)`, where each variable
//! `z_app[j]` is recomposed from its 64 committed bits in the
//! `app_private` region via `lane_terms(slot)`.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use crate::frontends::direct_ccs::R1cs;
use crate::frontends::f_prime_shell::image::FPrimeImageLayout;
use crate::frontends::f_prime_shell::structure::{
    emit_shell_rows, f_prime_lane_slots, lane_terms, FPrimeStructure, LaneSlot, MixedGateBuilder,
};

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
pub fn build_r1cs_f_prime_structure(layout: FPrimeImageLayout, r1cs: &R1cs) -> (FPrimeStructure, R1csRowAnchors) {
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
    append_r1cs_rows(&app_var_slots, r1cs, &mut builder);
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
fn append_r1cs_rows(app_var_slots: &[LaneSlot], r1cs: &R1cs, builder: &mut MixedGateBuilder) {
    for row in 0..r1cs.n() {
        let left = matrix_row_terms(&r1cs.a, row, app_var_slots);
        let right = matrix_row_terms(&r1cs.b, row, app_var_slots);
        let out = matrix_row_terms(&r1cs.c, row, app_var_slots);
        builder.product(left, right, out);
    }
}

/// Expand one matrix row `M[row, ·]` into `(col, coeff)` terms over the
/// F' bit-frame: each nonzero `M[row, j]` contributes a scaled lane
/// sum `M[row, j] · Σ_i 2^i · z[bit_start_j + i]`.
fn matrix_row_terms(m: &neo_ccs::Mat<F>, row: usize, app_var_slots: &[LaneSlot]) -> Vec<(usize, F)> {
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
