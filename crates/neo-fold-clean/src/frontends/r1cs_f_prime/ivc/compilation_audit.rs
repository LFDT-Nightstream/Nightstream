//! Exact, non-authoritative audit records for R1CS-IVC fixed-point discovery
//! and materialization.
//!
//! Owns: every fixed-point input/output header, full CCS polynomials, arm
//! shapes, final public-carrier/selector/private-alignment layout, the final
//! source arms' caller-labeled physical row intervals, and the recovered
//! steady-recursive PiCCS output-message dimensions. Shape discovery and
//! emitted-relation evidence are different views.
//!
//! Does not own: relation acceptance, witness validity, semantic refinement,
//! row-removal authority, stage-label semantics, expected-tree membership, or
//! performance budgets.
//!
//! Emits constraints: no.
//!
//! Authority boundary: the shape audit stops before matrix materialization;
//! the compilation audit is installed only after the emitted relation is
//! compared with that shape. Both remain diagnostics. Lean refinement must
//! compare their raw contents to an independent specification; a digest or a
//! self-consistent audit record is never semantic authority.
//!
//! | Child | Mathematical/physical ownership | Emits constraints? |
//! |---|---|---|
//! | `RelationHeaderAudit` | exact `(n, m, m_in, f)` | no |
//! | `FixedPointRoundAudit` | input header, three arms, output header | no |
//! | `SelectiveCompilerAudit` | exact planned coordinate layout, width census, and source-row intervals | no |
//! | `PiCcsOutputDigestAudit` | steady-recursive output dimensions and final Poseidon2 schedule recovered from retained physical traces | no |
//! | `R1csIvcFixedPointShapeAudit` | stabilized shape before the materialization budget gate | no |
//! | `R1csIvcCompilationAudit` | borrowed view of the audit owned by the emitted relation | no |

use neo_ccs::SparsePoly;
use neo_math::F;

use crate::engine::r1cs_circuit::PhysicalStageRange;
use crate::frontends::r1cs_f_prime::{
    SelectiveCompilerAudit, SelectiveLayoutAudit, SelectiveLowNormWidthAudit, SelectiveRowMappingAudit,
};

use super::PiCcsOutputDigestAudit;

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

/// Stabilized fixed-point shape before selective matrix materialization.
///
/// This is the audit surface for an oversized candidate: it proves what the
/// compiler planned, not that a relation with those matrices was emitted.
#[derive(Clone, Debug)]
pub struct R1csIvcFixedPointShapeAudit {
    rounds: Vec<FixedPointRoundAudit>,
    selective: SelectiveCompilerAudit,
    pi_ccs_output_digest: PiCcsOutputDigestAudit,
}

impl R1csIvcFixedPointShapeAudit {
    pub(super) fn new(
        rounds: Vec<FixedPointRoundAudit>,
        selective: SelectiveCompilerAudit,
        pi_ccs_output_digest: PiCcsOutputDigestAudit,
    ) -> Self {
        Self {
            rounds,
            selective,
            pi_ccs_output_digest,
        }
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

    /// Exact planned source-disposition and emitted-row intervals. This is
    /// compiler evidence, not a semantic lowering theorem.
    pub fn rows(&self) -> &SelectiveRowMappingAudit {
        self.selective.rows()
    }

    /// Profile and final sponge schedule recovered from the steady-recursive
    /// arm's physical PiCCS output-message path.
    pub fn pi_ccs_output_digest(&self) -> &PiCcsOutputDigestAudit {
        &self.pi_ccs_output_digest
    }

    /// Caller-labeled source-row intervals for the final planned arms.
    /// Labels remain diagnostic until checked against the expected vocabulary.
    pub fn source_arm_physical_stages(&self) -> &[Vec<PhysicalStageRange>] {
        self.selective.source_arm_physical_stages()
    }
}

/// Borrowed audit view after the materialized selective relation was checked
/// equal to the planned shape and compiler ledger.
///
/// The emitted relation owns `SelectiveCompilerAudit`. This view does not copy
/// that large ledger into a second owner.
#[derive(Clone, Copy, Debug)]
pub struct R1csIvcCompilationAudit<'a> {
    rounds: &'a [FixedPointRoundAudit],
    selective: &'a SelectiveCompilerAudit,
    pi_ccs_output_digest: &'a PiCcsOutputDigestAudit,
}

impl<'a> R1csIvcCompilationAudit<'a> {
    pub(super) fn new(
        rounds: &'a [FixedPointRoundAudit],
        selective: &'a SelectiveCompilerAudit,
        pi_ccs_output_digest: &'a PiCcsOutputDigestAudit,
    ) -> Self {
        Self {
            rounds,
            selective,
            pi_ccs_output_digest,
        }
    }

    pub fn rounds(&self) -> &[FixedPointRoundAudit] {
        self.rounds
    }

    pub fn layout(&self) -> &SelectiveLayoutAudit {
        self.selective.layout()
    }

    pub fn width(&self) -> &SelectiveLowNormWidthAudit {
        self.selective.width()
    }

    /// Exact source-disposition and emitted-row intervals checked against the
    /// materialized relation's compiler ledger.
    pub fn rows(&self) -> &SelectiveRowMappingAudit {
        self.selective.rows()
    }

    /// Profile and final sponge schedule recovered from the emitted recursive
    /// arm before the selective relation was checked against its plan.
    pub fn pi_ccs_output_digest(&self) -> &PiCcsOutputDigestAudit {
        self.pi_ccs_output_digest
    }

    /// Caller-labeled source-row intervals for the final base and recursive
    /// arms. Consumers must validate the expected roots and path vocabulary
    /// before treating these diagnostics as a complete protocol ledger.
    pub fn source_arm_physical_stages(&self) -> &[Vec<PhysicalStageRange>] {
        self.selective.source_arm_physical_stages()
    }
}
