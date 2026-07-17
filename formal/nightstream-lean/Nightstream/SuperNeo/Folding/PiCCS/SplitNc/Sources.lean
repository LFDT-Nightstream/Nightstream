import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Parameters
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81MatrixSource

/-!
One authoritative source family for the independent SplitNc semantics.

Protocol: SuperNeo `Pi_CCS`, specialized to the Phi81 carrier.
Phase: source construction before FE and NC residuals.
Constraint family: semantic witness connectivity only; this file emits no
rows.

Owns: original-width matrices and fresh assignments, full-carrier running
assignments, canonical fresh zero extension, the sole derived Phi81 matrix
source, canonical source injections, and exact finite list materializations
used only by later implementation refinement.

Does not own: protocol-level carrier/norm refinement, unified verifier
composition, commitments, public-input projection, FE/NC polynomial mixing,
SumCheck, transcript challenges, Rust witness decoding, R1CS, or constraint
counts.

Emits constraints: no.

Authority boundary: callers provide each mathematical source once. This
module imports only the concrete carrier algebra required to derive matrix
coefficients; it does not depend on the protocol verifier. Fresh carrier
suffixes and all coefficient-expanded matrices are definitions. Running
assignments retain every complete-carrier coordinate; no projection to the
original CCS width is treated as authority.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| coefficient embedding | matrix source | original `M` / Phi81 coefficients | every coefficient matrix is derived from the sole original matrix family |
| `Pi_CCS` | fresh sources | original / complete assignment | suffix is canonical zero extension |
| `Pi_CCS` | running sources | complete assignment | every carrier coordinate remains authoritative |
| FE | source views | CCS / carried evaluation | both views derive from this one `Data` value |
| NC refinement | serialization | typed assignments / lists | canonical `Fin` order is exact and proved |
| imported algebra | matrix derivation | concrete base operations | `ConcreteCarrier.Algebra`, no verifier dependency |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra

/-- Independent mathematical inputs for one SplitNc semantic statement.

There is no caller-supplied coefficient matrix and no caller-supplied fresh
carrier suffix. -/
structure Data (shape : SemanticShape) where
  matrices : Fin shape.matrixCount ->
    BooleanMatrix F shape.rowVariables shape.logicalWidth
  constraintPolynomial :
    CCSResidualTable.ConstraintPolynomial F shape.matrixCount
  freshAssignments : Fin shape.freshCount -> Assignment F shape.logicalWidth
  runningAssignments : Fin shape.runningCount -> Assignment F shape.carrierWidth
  priorPoint : CubePoint K shape.rowVariables
  claimedCoefficient : CarriedCoordinate shape.paperShape -> K

namespace Data

/-- Sole matrix owner, completed and coefficient-expanded by the independent
Phi81 kernel. -/
def matrixSource
    {shape : SemanticShape}
    (data : Data shape) :
    MatrixCoefficientSource.MatrixSource F shape.paperShape shape.carrierWidth
      (Phi81ColumnLayout.blockCount shape.carrierWidth) :=
  Phi81MatrixSource.source shape.rowVariables shape.freshCount
    shape.runningCount shape.matrixCount shape.logicalWidth data.matrices
    data.constraintPolynomial

/-- A fresh source completed to the full carrier with a canonical zero tail. -/
def freshAssignment
    {shape : SemanticShape}
    (data : Data shape)
    (source : Fin shape.freshCount) : Assignment F shape.carrierWidth :=
  Phi81CarrierLayout.extendAssignment 0 (data.freshAssignments source)

/-- Canonical injection of a fresh source into the joint source family. -/
def freshIndex
    {shape : SemanticShape}
    (source : Fin shape.freshCount) : Fin shape.sourceCount :=
  Fin.castAdd shape.runningCount source

/-- Canonical injection of a running source into the joint source family. -/
def runningIndex
    {shape : SemanticShape}
    (source : Fin shape.runningCount) : Fin shape.sourceCount :=
  Fin.natAdd shape.freshCount source

/-- The one authoritative complete-carrier assignment for every source. -/
def assignment
    {shape : SemanticShape}
    (data : Data shape) :
    Fin shape.sourceCount -> Assignment F shape.carrierWidth :=
  Fin.addCases data.freshAssignment data.runningAssignments

/-- The fresh injection reads the canonical fresh completion. -/
theorem assignment_freshIndex
    {shape : SemanticShape}
    (data : Data shape)
    (source : Fin shape.freshCount) :
    data.assignment (freshIndex source) = data.freshAssignment source := by
  simp [assignment, freshIndex]

/-- The running injection reads the caller's full carrier verbatim. -/
theorem assignment_runningIndex
    {shape : SemanticShape}
    (data : Data shape)
    (source : Fin shape.runningCount) :
    data.assignment (runningIndex source) = data.runningAssignments source := by
  simp [assignment, runningIndex]

/-- Fresh and running source coordinates cannot alias. -/
theorem freshIndex_ne_runningIndex
    {shape : SemanticShape}
    (fresh : Fin shape.freshCount)
    (running : Fin shape.runningCount) :
    freshIndex fresh ≠ runningIndex running := by
  intro equal
  have values := congrArg Fin.val equal
  simp [freshIndex, runningIndex, SemanticShape.sourceCount] at values
  omega

/-- Every joint source coordinate comes from exactly one side of the source
partition. -/
theorem source_eq_fresh_or_running
    {shape : SemanticShape}
    (source : Fin shape.sourceCount) :
    (Exists fun fresh => source = freshIndex fresh) \/
      Exists fun running => source = runningIndex running := by
  refine Fin.addCases ?_ ?_ source
  · intro fresh
    exact Or.inl ⟨fresh, by
      simp [freshIndex, SemanticShape.sourceCount]⟩
  · intro running
    exact Or.inr ⟨running, by
      simp [runningIndex, SemanticShape.sourceCount]⟩

/-- FE view of the fresh CCS sources over the completed carrier. -/
def freshBatch
    {shape : SemanticShape}
    (data : Data shape) :
    CCSResidualTable.FreshBatch F shape.paperShape shape.carrierWidth where
  system := data.matrixSource.system
  assignments := data.freshAssignment

/-- FE view of the running carried-evaluation sources. Every coefficient
matrix is derived from `matrixSource`; every assignment is full-carrier. -/
def carriedData
    {shape : SemanticShape}
    (data : Data shape) :
    CarriedEvaluationResidual.EvaluationData F K shape.paperShape
      shape.carrierWidth where
  priorPoint := data.priorPoint
  assignments := data.runningAssignments
  coefficientMatrices :=
    data.matrixSource.coefficientMatrices ConcreteCarrier.baseOps
  claimedCoefficient := data.claimedCoefficient

/-- Canonical increasing materialization of one authoritative assignment.
This is a refinement view, not a second authority source. -/
def orderedAssignment
    {shape : SemanticShape}
    (data : Data shape)
    (source : Fin shape.sourceCount) : List F :=
  (canonicalFinIndices shape.carrierWidth).map (data.assignment source)

/-- Canonical increasing materialization of the whole source family. -/
def orderedAssignments
    {shape : SemanticShape}
    (data : Data shape) : List (List F) :=
  (canonicalFinIndices shape.sourceCount).map data.orderedAssignment

/-- One materialized assignment has exactly the complete carrier width. -/
theorem orderedAssignment_length
    {shape : SemanticShape}
    (data : Data shape)
    (source : Fin shape.sourceCount) :
    (data.orderedAssignment source).length = shape.carrierWidth := by
  simp [orderedAssignment, canonicalFinIndices_length]

/-- Materialization preserves every typed assignment coordinate exactly. -/
theorem orderedAssignment_getD
    {shape : SemanticShape}
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : Fin shape.carrierWidth) :
    (data.orderedAssignment source).getD column.val 0 =
      data.assignment source column := by
  simp [orderedAssignment, canonicalFinIndices, column.isLt]

/-- The materialized batch has exactly the typed source count. -/
theorem orderedAssignments_length
    {shape : SemanticShape}
    (data : Data shape) :
    data.orderedAssignments.length = shape.sourceCount := by
  simp [orderedAssignments, canonicalFinIndices_length]

/-- Every typed source assignment occurs in the materialized batch. -/
theorem orderedAssignment_mem_orderedAssignments
    {shape : SemanticShape}
    (data : Data shape)
    (source : Fin shape.sourceCount) :
    data.orderedAssignment source ∈ data.orderedAssignments := by
  apply List.mem_map.mpr
  exact ⟨source, by simp [canonicalFinIndices], rfl⟩

/-- Every materialized batch member comes from a typed source. -/
theorem mem_orderedAssignments
    {shape : SemanticShape}
    (data : Data shape)
    {assignment : List F}
    (member : assignment ∈ data.orderedAssignments) :
    Exists fun source => assignment = data.orderedAssignment source := by
  rw [orderedAssignments] at member
  rcases List.mem_map.mp member with ⟨source, _, equal⟩
  exact ⟨source, equal.symm⟩

/-- The fresh FE view reads the same completed assignment as the joint source
family. -/
theorem freshBatch_assignment_eq
    {shape : SemanticShape}
    (data : Data shape)
    (source : Fin shape.freshCount) :
    data.freshBatch.assignments source =
      data.assignment (freshIndex source) := by
  symm
  exact data.assignment_freshIndex source

/-- The carried FE view reads the same running assignment as the joint source
family. -/
theorem carriedData_assignment_eq
    {shape : SemanticShape}
    (data : Data shape)
    (source : Fin shape.runningCount) :
    data.carriedData.assignments source =
      data.assignment (runningIndex source) := by
  symm
  exact data.assignment_runningIndex source

end Data

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
