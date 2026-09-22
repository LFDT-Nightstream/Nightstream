//! Ajtai commitment opening + X projection as R1CS rows.
//!
//! Ports the matrix-row coefficient computation from the prototype's
//! `enforce_ajtai_commitment` into the clean's `R1csBuilder` world. The
//! coefficient matrix comes from the verifier-owned [`AjtaiSModule`], not
//! process-global state.

use std::ops::Range;

use neo_ajtai::{precompute_rot_columns, AjtaiSModule, PP};
use neo_math::ring::Rq;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};
use crate::paper::reductions::pi_dec_circuit::CeClaimWires;

use super::witness::FinalWitnessWires;

#[derive(Debug)]
pub(crate) enum AjtaiOpeningError {
    SetupMissing {
        d: usize,
        cols: usize,
    },
    Shape {
        what: &'static str,
        expected: usize,
        got: usize,
    },
}

pub(crate) fn enforce_ajtai_opening(
    builder: &mut R1csBuilder,
    log: &AjtaiSModule,
    witness: &FinalWitnessWires,
    c_data: &[Var],
    c_d: usize,
    c_kappa: usize,
) -> Result<(), AjtaiOpeningError> {
    let rows = witness.rows;
    let cols = witness.cols;
    if rows != c_d {
        return Err(AjtaiOpeningError::Shape {
            what: "witness rows vs c_d",
            expected: c_d,
            got: rows,
        });
    }
    let coord_count = rows.checked_mul(c_kappa).ok_or(AjtaiOpeningError::Shape {
        what: "c_data length overflow",
        expected: 0,
        got: 0,
    })?;
    if c_data.len() != coord_count {
        return Err(AjtaiOpeningError::Shape {
            what: "c_data length",
            expected: coord_count,
            got: c_data.len(),
        });
    }
    let (log_rows, log_cols) = log.dims();
    if log_rows != rows {
        return Err(AjtaiOpeningError::Shape {
            what: "verifier-owned Ajtai row count",
            expected: rows,
            got: log_rows,
        });
    }
    if log_cols != cols {
        return Err(AjtaiOpeningError::Shape {
            what: "verifier-owned Ajtai column count",
            expected: cols,
            got: log_cols,
        });
    }
    let pp = log
        .materialize_pp()
        .map_err(|_| AjtaiOpeningError::SetupMissing { d: rows, cols })?;
    enforce_ajtai_opening_with_pp(builder, witness, c_data, c_d, c_kappa, 0..cols, &pp)
}

/// Enforce an opening of one whole-column witness slice under an explicit
/// Ajtai matrix. Nebula uses this for its independent `A_ops` and `A_mem`
/// coordinates; the full-witness commitment uses [`enforce_ajtai_opening`].
pub(crate) fn enforce_ajtai_slice_opening(
    builder: &mut R1csBuilder,
    witness: &FinalWitnessWires,
    c_data: &[Var],
    c_d: usize,
    c_kappa: usize,
    columns: Range<usize>,
    pp: &PP<Rq>,
) -> Result<(), AjtaiOpeningError> {
    enforce_ajtai_opening_with_pp(builder, witness, c_data, c_d, c_kappa, columns, pp)
}

fn enforce_ajtai_opening_with_pp(
    builder: &mut R1csBuilder,
    witness: &FinalWitnessWires,
    c_data: &[Var],
    c_d: usize,
    c_kappa: usize,
    columns: Range<usize>,
    pp: &PP<Rq>,
) -> Result<(), AjtaiOpeningError> {
    let rows = witness.rows;
    if rows != c_d {
        return Err(AjtaiOpeningError::Shape {
            what: "witness rows vs c_d",
            expected: c_d,
            got: rows,
        });
    }
    if columns.end > witness.cols {
        return Err(AjtaiOpeningError::Shape {
            what: "witness slice columns",
            expected: columns.end,
            got: witness.cols,
        });
    }
    let cols = columns.len();
    let coord_count = rows.checked_mul(c_kappa).ok_or(AjtaiOpeningError::Shape {
        what: "c_data length overflow",
        expected: 0,
        got: 0,
    })?;
    if c_data.len() != coord_count {
        return Err(AjtaiOpeningError::Shape {
            what: "c_data length",
            expected: coord_count,
            got: c_data.len(),
        });
    }
    if pp.d != rows || pp.m != cols {
        return Err(AjtaiOpeningError::Shape {
            what: "Ajtai matrix dimensions",
            expected: rows * cols,
            got: pp.d * pp.m,
        });
    }
    if pp.m_rows.len() != c_kappa {
        return Err(AjtaiOpeningError::Shape {
            what: "Ajtai kappa",
            expected: c_kappa,
            got: pp.m_rows.len(),
        });
    }
    for pp_row in &pp.m_rows {
        if pp_row.len() != cols {
            return Err(AjtaiOpeningError::Shape {
                what: "Ajtai row width",
                expected: cols,
                got: pp_row.len(),
            });
        }
    }

    // For each commit column `commit_col` (kappa total) and each
    // coordinate-row inside it (rows total), emit one linear
    // equation: `Σ rots[witness_row][coord_row] * Z[witness_row,
    // witness_col] = c_data[commit_col * rows + coord_row]`. The
    // mapping mirrors `ajtai_commitment_rows` in the prototype but
    // streams the constraint directly without materialising the full
    // coefficient matrix.
    for (commit_col, pp_row) in pp.m_rows.iter().enumerate() {
        for coord_row in 0..rows {
            let mut lhs = Lc::zero();
            for (local_col, ring_el) in pp_row.iter().copied().enumerate() {
                let witness_col = columns.start + local_col;
                let mut rots = [[F::ZERO; D]; D];
                precompute_rot_columns(ring_el, &mut rots);
                for witness_row in 0..rows {
                    let coeff = rots[witness_row][coord_row];
                    if coeff != F::ZERO {
                        let z_var = witness
                            .entry(witness_row, witness_col)
                            .ok_or(AjtaiOpeningError::Shape {
                                what: "witness entry",
                                expected: rows * cols,
                                got: 0,
                            })?;
                        lhs.add_term(z_var, coeff);
                    }
                }
            }
            let coord = commit_col * rows + coord_row;
            builder.enforce_eq(&lhs, &Lc::from_var(c_data[coord]));
        }
    }
    Ok(())
}

#[derive(Debug)]
pub(crate) struct XProjectionError {
    what: &'static str,
    expected: usize,
    got: usize,
}

impl XProjectionError {
    pub(crate) fn what(&self) -> &'static str {
        self.what
    }
    pub(crate) fn expected(&self) -> usize {
        self.expected
    }
    pub(crate) fn got(&self) -> usize {
        self.got
    }
}

/// Enforce `claim.X = L_x(Z)` for the compact public-input embedding.
pub(crate) fn enforce_x_projection(
    builder: &mut R1csBuilder,
    witness: &FinalWitnessWires,
    claim: &CeClaimWires,
) -> Result<(), XProjectionError> {
    if claim.x_rows != D {
        return Err(XProjectionError {
            what: "claim.x_rows",
            expected: D,
            got: claim.x_rows,
        });
    }
    if witness.rows != D {
        return Err(XProjectionError {
            what: "witness rows",
            expected: D,
            got: witness.rows,
        });
    }
    if claim.m_in % D != 0 {
        return Err(XProjectionError {
            what: "claim.m_in remainder modulo D",
            expected: 0,
            got: claim.m_in % D,
        });
    }
    let required_cols = claim.m_in / D;
    if claim.x_cols != required_cols {
        return Err(XProjectionError {
            what: "claim.x_cols == m_in / D",
            expected: required_cols,
            got: claim.x_cols,
        });
    }
    if witness.cols < required_cols {
        return Err(XProjectionError {
            what: "witness cols ≥ required_cols",
            expected: required_cols,
            got: witness.cols,
        });
    }
    for col in 0..required_cols {
        for row in 0..D {
            let z_var = witness.entry(row, col).ok_or(XProjectionError {
                what: "witness entry",
                expected: D * required_cols,
                got: 0,
            })?;
            let x_idx = row * claim.x_cols + col;
            let x_var = *claim.x.get(x_idx).ok_or(XProjectionError {
                what: "claim.x slot",
                expected: D * claim.x_cols,
                got: claim.x.len(),
            })?;
            builder.enforce_eq(&Lc::from_var(z_var), &Lc::from_var(x_var));
        }
    }
    Ok(())
}
