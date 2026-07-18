//! Exact, non-authoritative audit record for the R1CS-IVC fixed-point compiler.
//!
//! Owns: every fixed-point input/output header, full CCS polynomials, arm
//! shapes, final public-carrier/selector/private-alignment layout, and the
//! final source arms' caller-labeled physical row intervals.
//!
//! Does not own: relation acceptance, witness validity, semantic refinement,
//! row-removal authority, stage-label semantics, expected-tree membership, or
//! performance budgets.
//!
//! Emits constraints: no.
//!
//! Authority boundary: these values are captured from the same compiler run
//! that emits the relation, but remain diagnostics. Lean refinement must
//! compare their raw contents to an independent specification; a digest or a
//! self-consistent audit record is never semantic authority.
//!
//! | Child | Mathematical/physical ownership | Emits constraints? |
//! |---|---|---|
//! | `RelationHeaderAudit` | exact `(n, m, m_in, f)` | no |
//! | `FixedPointRoundAudit` | input header, three arms, output header | no |
//! | `SelectiveCompilerAudit` | exact emitted coordinate layout, width census, and source-row intervals | no |
//! | `R1csIvcCompilationAudit` | ordered rounds plus final compiler audit | no |

use neo_ccs::SparsePoly;
use neo_math::F;

use crate::engine::r1cs_circuit::PhysicalStageRange;
use crate::frontends::r1cs_f_prime::{SelectiveCompilerAudit, SelectiveLayoutAudit, SelectiveLowNormWidthAudit};

#[derive(Clone, Debug)]
pub struct RelationHeaderAudit {
    pub rows: usize,
    pub columns: usize,
    pub public_input_len: usize,
    pub polynomial: SparsePoly<F>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArmShapeAudit {
    pub rows: usize,
    pub columns: usize,
    pub public_columns: usize,
}

#[derive(Clone, Debug)]
pub struct FixedPointRoundAudit {
    pub round: usize,
    pub input: RelationHeaderAudit,
    pub arms: [ArmShapeAudit; 3],
    pub output: RelationHeaderAudit,
}

#[derive(Clone, Debug)]
pub struct R1csIvcCompilationAudit {
    rounds: Vec<FixedPointRoundAudit>,
    selective: SelectiveCompilerAudit,
}

impl R1csIvcCompilationAudit {
    pub(super) fn new(rounds: Vec<FixedPointRoundAudit>, selective: SelectiveCompilerAudit) -> Self {
        Self { rounds, selective }
    }

    pub fn rounds(&self) -> &[FixedPointRoundAudit] {
        &self.rounds
    }

    pub fn layout(&self) -> &SelectiveLayoutAudit {
        self.selective.layout()
    }

    pub fn width(&self) -> &SelectiveLowNormWidthAudit {
        self.selective.width()
    }

    /// Caller-labeled source-row intervals for the final base and recursive
    /// arms. Consumers must validate the expected roots and path vocabulary
    /// before treating these diagnostics as a complete protocol ledger.
    pub fn source_arm_physical_stages(&self) -> &[Vec<PhysicalStageRange>] {
        self.selective.source_arm_physical_stages()
    }
}
