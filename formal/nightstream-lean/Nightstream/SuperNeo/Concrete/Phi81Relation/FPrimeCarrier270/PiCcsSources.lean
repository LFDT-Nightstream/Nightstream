import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Fe

/-!
Production-shaped Split-NC source adapter for the five-ring F' carrier.

Protocol: SuperNeo `Pi_CCS`, specialized to the split FE/NC Phi81 semantics.
Phase: caller legacy CCS sources to one authoritative `SplitNc.Sources.Data`.
Constraint family: semantic source ownership only; this file emits no rows.

Owns: the batch semantic shape; one input bundle containing the legacy CCS
structure, legacy-width fresh assignments, full-carrier running assignments,
the prior point, and claimed coefficients; deterministic alignment of the
legacy matrices and fresh assignments; exact source-field ownership; and
equivalence between Split-NC fresh CCS truth and legacy fresh CCS truth.

Does not own: running carried-evaluation truth, NC norm truth, commitments,
Ajtai setup or binding, transcript derivation, SumCheck, Rust sparse storage,
R1CS lowering, row deletion, or constraint counts.

Emits constraints: no.

Authority boundary: legacy matrices, the explicit sparse polynomial, and each
legacy fresh assignment are caller inputs exactly once. Their 13-column
alignment and full-carrier completion are definitions. Running assignments
are already full-carrier values and pass through unchanged; the adapter does
not truncate or reconstruct them. Claimed coefficients remain inputs, so this
file deliberately proves no carried CE truth.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.sources.shape` | Split-NC logical width is the aligned F' width and both carriers coincide | computed | `semanticShape_carrierWidth` |
| `nifs.pi_ccs.sources.matrix` | the sole Split-NC matrix source is the aligned-and-completed legacy matrix | computed | `data_matrixSource_matrix` |
| `nifs.pi_ccs.sources.fresh` | every fresh source is exactly `FPrimeCarrier270.assignment` | computed | `data_freshAssignment_eq` |
| `nifs.pi_ccs.sources.running` | every full-carrier running source is unchanged | direct dataflow | `data_runningAssignment_eq`, `data_assignment_runningIndex_eq` |
| `nifs.pi_ccs.fe.fresh.images` | every fresh matrix image equals its legacy image | derived | `freshMatrixImagesAt_eq` |
| `nifs.pi_ccs.fe.fresh.residual` | every fresh residual equals its legacy residual | derived | `freshResidualAt_eq` |
| `nifs.pi_ccs.fe.fresh.truth` | Split-NC fresh CCS truth iff all legacy fresh assignments satisfy the legacy structure | derived | `freshTruth_iff_legacy` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap

/-- Exact Split-NC batch dimensions for one aligned F' relation. Unlike the
batch-free Phi81 relation shape, fresh and running arities are explicit here. -/
def semanticShape (dimensions : Dimensions)
    (freshCount runningCount : Nat) : SemanticShape where
  rowVariables := dimensions.rowVariables
  logicalWidth := dimensions.alignedLogicalWidth
  freshCount := freshCount
  runningCount := runningCount
  matrixCount := dimensions.matrixCount

@[simp] theorem semanticShape_rowVariables
    (dimensions : Dimensions) (freshCount runningCount : Nat) :
    (semanticShape dimensions freshCount runningCount).rowVariables =
      dimensions.rowVariables := by
  rfl

@[simp] theorem semanticShape_logicalWidth
    (dimensions : Dimensions) (freshCount runningCount : Nat) :
    (semanticShape dimensions freshCount runningCount).logicalWidth =
      dimensions.alignedLogicalWidth := by
  rfl

/-- The production-shaped Split-NC carrier and the typed five-ring relation
carrier are the same exact finite domain. -/
@[simp] theorem semanticShape_carrierWidth
    (dimensions : Dimensions) (freshCount runningCount : Nat) :
    (semanticShape dimensions freshCount runningCount).carrierWidth =
      dimensions.shape.carrierWidth := by
  rfl

/-- Legacy CCS structure at the actual Split-NC batch shape. The fresh and
running counts are retained because this adapter targets `Sources.Data`, not
the batch-free PaperJoint facade. -/
abbrev LegacyBatchStructure (dimensions : Dimensions)
    (freshCount runningCount : Nat) :=
  CCSResidualTable.Structure F
    (semanticShape dimensions freshCount runningCount).paperShape
    dimensions.legacyLogicalWidth

/-- One authoritative input product for the Split-NC source adapter. -/
structure Inputs (dimensions : Dimensions)
    (freshCount runningCount : Nat) where
  legacyStructure :
    LegacyBatchStructure dimensions freshCount runningCount
  freshAssignments : Fin freshCount -> LegacyAssignment dimensions
  runningAssignments :
    Fin runningCount -> Assignment dimensions.shape
  priorPoint : Point dimensions.shape
  claimedCoefficient :
    CarriedCoordinate
      (semanticShape dimensions freshCount runningCount).paperShape -> K

namespace Inputs

/-- Deterministically align caller-owned legacy matrices and fresh
assignments while retaining full-carrier running sources verbatim. -/
def data
    {dimensions : Dimensions} {freshCount runningCount : Nat}
    (inputs : Inputs dimensions freshCount runningCount) :
    Sources.Data (semanticShape dimensions freshCount runningCount) where
  matrices := fun matrix =>
    alignedMatrix dimensions (inputs.legacyStructure.matrices matrix)
  constraintPolynomial := inputs.legacyStructure.constraintPolynomial
  freshAssignments := fun source =>
    alignedLogicalAssignment dimensions (inputs.freshAssignments source)
  runningAssignments := inputs.runningAssignments
  priorPoint := inputs.priorPoint
  claimedCoefficient := inputs.claimedCoefficient

/-- The sole Split-NC completed matrix is exactly the carrier matrix derived
from the caller's one legacy matrix source. -/
theorem data_matrixSource_matrix
    {dimensions : Dimensions} {freshCount runningCount : Nat}
    (inputs : Inputs dimensions freshCount runningCount)
    (matrix : Fin dimensions.matrixCount)
    (vertex : BooleanVertex dimensions.rowVariables)
    (column : Fin dimensions.shape.carrierWidth) :
    inputs.data.matrixSource.matrices matrix vertex column =
      carrierMatrix dimensions (inputs.legacyStructure.matrices matrix)
        vertex column := by
  rfl

/-- The explicit legacy sparse polynomial passes into the Split-NC matrix
source without rewriting or a caller-supplied equivalence. -/
@[simp] theorem data_matrixSource_constraintPolynomial
    {dimensions : Dimensions} {freshCount runningCount : Nat}
    (inputs : Inputs dimensions freshCount runningCount) :
    inputs.data.matrixSource.constraintPolynomial =
      inputs.legacyStructure.constraintPolynomial := by
  rfl

/-- Split-NC's canonical fresh completion is the exact typed five-ring F'
assignment constructor for the corresponding legacy source. -/
theorem data_freshAssignment_eq
    {dimensions : Dimensions} {freshCount runningCount : Nat}
    (inputs : Inputs dimensions freshCount runningCount)
    (source : Fin freshCount) :
    inputs.data.freshAssignment source =
      assignment dimensions (inputs.freshAssignments source) := by
  rfl

/-- The running-assignment field is direct dataflow from the caller's
full-carrier source family. -/
theorem data_runningAssignment_eq
    {dimensions : Dimensions} {freshCount runningCount : Nat}
    (inputs : Inputs dimensions freshCount runningCount)
    (source : Fin runningCount) :
    inputs.data.runningAssignments source = inputs.runningAssignments source := by
  rfl

/-- Reading a running source through the joint fresh/running partition still
returns the caller's full-carrier assignment unchanged. -/
theorem data_assignment_runningIndex_eq
    {dimensions : Dimensions} {freshCount runningCount : Nat}
    (inputs : Inputs dimensions freshCount runningCount)
    (source : Fin runningCount) :
    inputs.data.assignment (Data.runningIndex source) =
      inputs.runningAssignments source := by
  rw [Data.assignment_runningIndex]
  exact data_runningAssignment_eq inputs source

/-! ## Fresh FE semantics -/

/-- The completed Split-NC image of one fresh source is exactly the legacy
matrix image. The finite reindexing is owned by `CcsRefinement`; this adapter
only instantiates that theorem at the batch-shaped source. -/
theorem freshMatrixImagesAt_eq
    {dimensions : Dimensions} {freshCount runningCount : Nat}
    (inputs : Inputs dimensions freshCount runningCount)
    (source : Fin freshCount)
    (vertex : BooleanVertex dimensions.rowVariables) :
    CCSResidualTable.matrixImagesAt ConcreteCarrier.baseOps
        inputs.data.matrixSource.system
        (inputs.data.freshAssignment source) vertex =
      CCSResidualTable.matrixImagesAt ConcreteCarrier.baseOps
        inputs.legacyStructure (inputs.freshAssignments source) vertex := by
  funext matrix
  exact CcsRefinement.carrierMatrixVectorAt_eq dimensions
    (inputs.legacyStructure.matrices matrix)
    (inputs.freshAssignments source) vertex

/-- The completed Split-NC fresh residual is exactly the legacy residual at
every semantic CCS row. -/
theorem freshResidualAt_eq
    {dimensions : Dimensions} {freshCount runningCount : Nat}
    (inputs : Inputs dimensions freshCount runningCount)
    (source : Fin freshCount)
    (vertex : BooleanVertex dimensions.rowVariables) :
    CCSResidualTable.residualAt ConcreteCarrier.baseOps
        inputs.data.matrixSource.system
        (inputs.data.freshAssignment source) vertex =
      CCSResidualTable.residualAt ConcreteCarrier.baseOps
        inputs.legacyStructure (inputs.freshAssignments source) vertex := by
  unfold CCSResidualTable.residualAt
  rw [freshMatrixImagesAt_eq]
  rfl

/-- Split-NC fresh FE truth contains neither more nor less CCS truth than the
caller-owned legacy fresh batch. This theorem does not include carried CE or
NC norm truth. -/
theorem freshTruth_iff_legacy
    {dimensions : Dimensions} {freshCount runningCount : Nat}
    (inputs : Inputs dimensions freshCount runningCount) :
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Fe.FreshTruth
        inputs.data <->
      (forall source,
        CCSResidualTable.ConstraintSatisfied ConcreteCarrier.baseOps
          inputs.legacyStructure (inputs.freshAssignments source)) := by
  change
    (forall source vertex,
      CCSResidualTable.residualAt ConcreteCarrier.baseOps
        inputs.data.matrixSource.system
        (inputs.data.freshAssignment source) vertex =
          ConcreteCarrier.baseOps.zero) <-> _
  constructor
  · intro truth source vertex
    rw [← freshResidualAt_eq inputs source vertex]
    exact truth source vertex
  · intro truth source vertex
    rw [freshResidualAt_eq inputs source vertex]
    exact truth source vertex

end Inputs

end Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources
