use neo_ajtai::Commitment;
use neo_ccs::CeClaim;
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::paper::digest::{
    noncanonical_digest32_lane, params_digest, structure_digest, terminal_ce_public_digest,
    terminal_ce_relation_digest, terminal_children_digest,
};
use crate::paper::params::Params;
use crate::paper::relations::{superneo_public_x_cols, Structure};

/// Public statement for a compact proof of the terminal CE relation.
///
/// This binds the proof to the verifier-owned structure/parameter context and
/// to the exact NIFS-produced terminal children. It is not sufficient by
/// itself: the proof verifier must still prove `exists Z` satisfying the full
/// terminal CE relation for those children.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalCePublic {
    pub relation_digest: [F; 4],
    pub structure_digest: [F; 4],
    pub params_digest: [F; 4],
    pub terminal_children_digest: [F; 4],
    pub claim_count: usize,
}

impl TerminalCePublic {
    pub fn from_terminal_children(
        params: &Params,
        structure: &Structure,
        terminal_children: &[CeClaim<Commitment, F, K>],
    ) -> Result<Self, TerminalCePublicError> {
        validate_terminal_children(params, structure, terminal_children)?;
        Ok(Self {
            relation_digest: terminal_ce_relation_digest(),
            structure_digest: structure_digest(structure),
            params_digest: params_digest(params.inner()),
            terminal_children_digest: terminal_children_digest(terminal_children),
            claim_count: terminal_children.len(),
        })
    }

    pub fn digest(&self) -> [F; 4] {
        terminal_ce_public_digest(
            self.relation_digest,
            self.structure_digest,
            self.params_digest,
            self.terminal_children_digest,
            self.claim_count,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TerminalCePublicError {
    #[error("terminal CE child {index} commitment d ({got}) must equal D ({expected})")]
    CommitmentD {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE child {index} commitment kappa ({got}) must equal params.kappa ({expected})")]
    CommitmentKappa {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE child {index} commitment data length ({got}) must equal d*kappa ({expected})")]
    CommitmentDataLen {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE child {index} X.rows ({got}) must equal D ({expected})")]
    XRows {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE child {index} X.cols ({got}) must equal the compact coefficient width ({expected})")]
    XCols {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE child {index} m_in ({got}) must be a whole number of degree-{degree} ring elements")]
    MInNotWholeRing {
        index: usize,
        got: usize,
        degree: usize,
    },
    #[error("terminal CE child {index} m_in ({got}) must not exceed structure.m ({expected})")]
    MInExceedsStructureM {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE child {index} r length ({got}) must equal row-domain length ({expected})")]
    RLen {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE child {index} y_ring length ({got}) must equal the identity-first matrix count ({expected})")]
    YRingCount {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE child {index} y_ring row {matrix_index} has {got} lanes, expected {expected}")]
    YRingLaneCount {
        index: usize,
        matrix_index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE child {index} ct length ({got}) must equal y_ring length ({expected})")]
    CtLen {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE child {index} ct[{matrix_index}] must equal y_ring[{matrix_index}] lane zero")]
    CtMismatch { index: usize, matrix_index: usize },
    #[error("terminal CE child {index} y_ring[{matrix_index}] padding lane {lane} must be zero")]
    YRingPaddingNonZero {
        index: usize,
        matrix_index: usize,
        lane: usize,
    },
    #[error("terminal CE child {index} fold digest lane {lane} is not a canonical Goldilocks element")]
    NoncanonicalFoldDigest { index: usize, lane: usize },
}

fn validate_terminal_children(
    params: &Params,
    structure: &Structure,
    claims: &[CeClaim<Commitment, F, K>],
) -> Result<(), TerminalCePublicError> {
    let d_pad = D.next_power_of_two();
    let assignment_width = neo_reductions::common::superneo_carrier_width(structure.m);
    let expected_r_len = structure
        .n
        .max(assignment_width)
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    let expected_t = structure.t() + 1;
    for (index, claim) in claims.iter().enumerate() {
        if claim.c.d != D {
            return Err(TerminalCePublicError::CommitmentD {
                index,
                expected: D,
                got: claim.c.d,
            });
        }
        let expected_kappa = params.kappa() as usize;
        if claim.c.kappa != expected_kappa {
            return Err(TerminalCePublicError::CommitmentKappa {
                index,
                expected: expected_kappa,
                got: claim.c.kappa,
            });
        }
        let expected_c_len = claim.c.d * claim.c.kappa;
        if claim.c.data.len() != expected_c_len {
            return Err(TerminalCePublicError::CommitmentDataLen {
                index,
                expected: expected_c_len,
                got: claim.c.data.len(),
            });
        }
        if claim.m_in > structure.m {
            return Err(TerminalCePublicError::MInExceedsStructureM {
                index,
                expected: structure.m,
                got: claim.m_in,
            });
        }
        if claim.m_in % D != 0 {
            return Err(TerminalCePublicError::MInNotWholeRing {
                index,
                got: claim.m_in,
                degree: D,
            });
        }
        if claim.X.rows() != D {
            return Err(TerminalCePublicError::XRows {
                index,
                expected: D,
                got: claim.X.rows(),
            });
        }
        let active_cols = superneo_public_x_cols(claim.m_in);
        if claim.X.cols() != active_cols {
            return Err(TerminalCePublicError::XCols {
                index,
                expected: active_cols,
                got: claim.X.cols(),
            });
        }
        if claim.r.len() != expected_r_len {
            return Err(TerminalCePublicError::RLen {
                index,
                expected: expected_r_len,
                got: claim.r.len(),
            });
        }
        if claim.y_ring.len() != expected_t {
            return Err(TerminalCePublicError::YRingCount {
                index,
                expected: expected_t,
                got: claim.y_ring.len(),
            });
        }
        for (matrix_index, row) in claim.y_ring.iter().enumerate() {
            if row.len() != d_pad {
                return Err(TerminalCePublicError::YRingLaneCount {
                    index,
                    matrix_index,
                    expected: d_pad,
                    got: row.len(),
                });
            }
        }
        if claim.ct.len() != claim.y_ring.len() {
            return Err(TerminalCePublicError::CtLen {
                index,
                expected: expected_t,
                got: claim.ct.len(),
            });
        }
        for (matrix_index, (ct, row)) in claim.ct.iter().zip(claim.y_ring.iter()).enumerate() {
            if *ct != row[0] {
                return Err(TerminalCePublicError::CtMismatch { index, matrix_index });
            }
            for (lane, value) in row.iter().enumerate().skip(D) {
                if *value != K::ZERO {
                    return Err(TerminalCePublicError::YRingPaddingNonZero {
                        index,
                        matrix_index,
                        lane,
                    });
                }
            }
        }
        if let Some(lane) = noncanonical_digest32_lane(claim.fold_digest) {
            return Err(TerminalCePublicError::NoncanonicalFoldDigest { index, lane });
        }
    }
    Ok(())
}
