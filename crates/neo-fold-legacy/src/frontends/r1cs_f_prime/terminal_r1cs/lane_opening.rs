//! Nebula lane-slice openings for the direct terminal relation.
//!
//! Each published `adv` component is recomputed from a whole-column slice of
//! the same private witness used by the main terminal commitment opening.

use neo_ajtai::{precompute_rot_columns, AjtaiSModule, Commitment};
use neo_ccs::LaneCommitments;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::paper::relations::product_commitment_circuit::{AdvCommitmentDataWires, CommitmentDataWires};
use crate::paper::relations::{LaneRanges, LaneScheme, Structure};

use super::TerminalR1csError;

type Rotations = [[F; D]; D];

pub(super) struct LaneOpeningRows {
    ops: Vec<Vec<Rotations>>,
    mem: Vec<Vec<Rotations>>,
    ranges: LaneRanges,
}

pub(super) fn prepare(
    lanes: Option<&LaneScheme>,
    structure: &Structure,
    verifier_rows: usize,
) -> Result<Option<LaneOpeningRows>, TerminalR1csError> {
    let Some(lanes) = lanes else {
        return Ok(None);
    };
    let ranges = lanes.ranges().clone();
    let witness_columns = structure.m.div_ceil(D);
    if ranges.fs.end > witness_columns {
        return Err(TerminalR1csError::Shape {
            what: "terminal Nebula lane range",
            expected: witness_columns,
            got: ranges.fs.end,
        });
    }
    let ops = rotations(lanes.ops_module(), ranges.ops.len(), verifier_rows)?;
    let mem = rotations(lanes.mem_module(), ranges.is.len(), verifier_rows)?;
    Ok(Some(LaneOpeningRows { ops, mem, ranges }))
}

pub(super) fn validate(
    rows: Option<&LaneOpeningRows>,
    commitments: Option<&LaneCommitments<Commitment>>,
    verifier_rows: usize,
    running: bool,
) -> Result<(), TerminalR1csError> {
    match (rows, commitments) {
        (None, None) => Ok(()),
        (None, Some(_)) => Err(TerminalR1csError::Unsupported(if running {
            "Nebula running commitment sidecars without a lane scheme"
        } else {
            "Nebula fresh commitment sidecars without a lane scheme"
        })),
        (Some(_), None) => Err(TerminalR1csError::Unsupported(if running {
            "Nebula running lane scheme without commitment sidecars"
        } else {
            "Nebula fresh lane scheme without commitment sidecars"
        })),
        (Some(_), Some(commitments)) => {
            validate_commitment(&commitments.ops, verifier_rows)?;
            validate_commitment(&commitments.is, verifier_rows)?;
            validate_commitment(&commitments.fs, verifier_rows)
        }
    }
}

pub(super) fn alloc_public(
    builder: &mut R1csBuilder,
    public_vars: &mut Vec<Var>,
    commitments: Option<&LaneCommitments<Commitment>>,
) -> Option<AdvCommitmentDataWires> {
    commitments.map(|commitments| {
        let alloc =
            |builder: &mut R1csBuilder, public_vars: &mut Vec<Var>, commitment: &Commitment| CommitmentDataWires {
                d: commitment.d,
                kappa: commitment.kappa,
                data: alloc_public_vec(builder, public_vars, &commitment.data),
            };
        LaneCommitments {
            ops: alloc(builder, public_vars, &commitments.ops),
            is: alloc(builder, public_vars, &commitments.is),
            fs: alloc(builder, public_vars, &commitments.fs),
        }
    })
}

pub(super) fn enforce(
    builder: &mut R1csBuilder,
    rows: Option<&LaneOpeningRows>,
    witness: &[Var],
    commitments: Option<&AdvCommitmentDataWires>,
) -> Result<(), TerminalR1csError> {
    match (rows, commitments) {
        (None, None) => Ok(()),
        (Some(rows), Some(commitments)) => {
            enforce_slice(
                builder,
                &rows.ops,
                witness,
                &commitments.ops.data,
                rows.ranges.ops.clone(),
            )?;
            enforce_slice(
                builder,
                &rows.mem,
                witness,
                &commitments.is.data,
                rows.ranges.is.clone(),
            )?;
            enforce_slice(
                builder,
                &rows.mem,
                witness,
                &commitments.fs.data,
                rows.ranges.fs.clone(),
            )
        }
        _ => Err(TerminalR1csError::InvalidState(
            "terminal lane-opening rows and public sidecars differ",
        )),
    }
}

fn rotations(
    module: &AjtaiSModule,
    witness_columns: usize,
    verifier_rows: usize,
) -> Result<Vec<Vec<Rotations>>, TerminalR1csError> {
    require_len("Nebula lane Ajtai ring degree", D, module.dims().0)?;
    require_len("Nebula lane Ajtai witness columns", witness_columns, module.dims().1)?;
    require_len("Nebula lane Ajtai verifier rows", verifier_rows, module.kappa())?;
    let pp = module
        .materialize_pp()
        .map_err(|error| TerminalR1csError::Coefficients(error.to_string()))?;
    require_len("Nebula lane Ajtai materialized rows", verifier_rows, pp.m_rows.len())?;
    let mut all = Vec::with_capacity(pp.m_rows.len());
    for row in &pp.m_rows {
        require_len("Nebula lane Ajtai materialized columns", witness_columns, row.len())?;
        let mut row_rotations = Vec::with_capacity(row.len());
        for &ring_element in row {
            let mut rotations = [[F::ZERO; D]; D];
            precompute_rot_columns(ring_element, &mut rotations);
            row_rotations.push(rotations);
        }
        all.push(row_rotations);
    }
    Ok(all)
}

fn validate_commitment(commitment: &Commitment, verifier_rows: usize) -> Result<(), TerminalR1csError> {
    require_len("Nebula lane commitment ring degree", D, commitment.d)?;
    require_len("Nebula lane commitment verifier rows", verifier_rows, commitment.kappa)?;
    require_len(
        "Nebula lane commitment coordinates",
        verifier_rows * D,
        commitment.data.len(),
    )
}

fn alloc_public_vec(builder: &mut R1csBuilder, public_vars: &mut Vec<Var>, values: &[F]) -> Vec<Var> {
    values
        .iter()
        .map(|&value| {
            let variable = builder.alloc(value);
            public_vars.push(variable);
            variable
        })
        .collect()
}

fn enforce_slice(
    builder: &mut R1csBuilder,
    rotations: &[Vec<Rotations>],
    witness: &[Var],
    commitment: &[Var],
    columns: std::ops::Range<usize>,
) -> Result<(), TerminalR1csError> {
    require_len(
        "Nebula lane commitment coordinates",
        rotations.len() * D,
        commitment.len(),
    )?;
    if columns.end * D > witness.len() {
        return Err(TerminalR1csError::Shape {
            what: "Nebula lane witness range",
            expected: witness.len() / D,
            got: columns.end,
        });
    }
    for (commitment_column, row) in rotations.iter().enumerate() {
        require_len("Nebula lane Ajtai columns", columns.len(), row.len())?;
        for output in 0..D {
            let mut left = Lc::zero();
            for (local_block, block_rotations) in row.iter().enumerate() {
                let block = columns.start + local_block;
                for witness_lane in 0..D {
                    left.add_term(witness[block * D + witness_lane], block_rotations[witness_lane][output]);
                }
            }
            builder.enforce_eq(&left, &Lc::from_var(commitment[commitment_column * D + output]));
        }
    }
    Ok(())
}

fn require_len(what: &'static str, expected: usize, got: usize) -> Result<(), TerminalR1csError> {
    if got == expected {
        Ok(())
    } else {
        Err(TerminalR1csError::Shape { what, expected, got })
    }
}
