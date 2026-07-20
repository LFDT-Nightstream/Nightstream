import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.OldPointBinding
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.SourceRefinement
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain

/-!
Contract: identify the delayed-projection child table with the authoritative
running assignments of the production 270-coordinate Split-NC statement.

Owns: canonical list materialization of `Sources.Data.runningAssignments`,
exact child/column lookup, fixed-profile domain coverage, and specialization
of the old-point binding theorem to that raw table.

Does not own: construction of `Sources.Data` from Rust witness columns,
production combined-NC acceptance, parent padding rows, recursive-state
continuity, transcript scheduling, commitment binding, `y_ring`, costs, or
row removal.

Emits constraints: no.

Authority boundary: the child values below are read only from the typed raw
running assignments. There is no output message or `CeClaim.y_zcol` input on
this surface. A production refinement must decode the witness tables into the
same `Sources.Data.runningAssignments` values before this theorem applies.

| Stage path | Mathematical obligation | Authority class | Open boundary |
|---|---|---|---|
| `nifs.pi_ccs.nc.delayed_projection.raw_children` | canonical child-major/column-major materialization of raw running assignments | direct dataflow | Rust/R1CS witness-table decoder |
| `nifs.pi_ccs.nc.delayed_projection.raw_children.coverage` | every 270-coordinate child fits the 512-column flat NC domain | derived | fixed-profile decoder conformance |
| `nifs.pi_ccs.nc.delayed_projection.old_point` | accepted compact identity yields the raw-child old-point relation or `BadRoot` | model-level composition | combined-NC acceptance, padding, and transcript derivation |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.ProductionRawChildren

open Nightstream.Implementation.R1CS.PiCcsNc
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection
open Nightstream.Implementation.R1CS.PiCcsNc.SourceRefinement
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

namespace Fixed270

abbrev semanticShape
    (rowVariables freshCount runningCount matrixCount : Nat) :=
  Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.plainShape
    rowVariables freshCount runningCount matrixCount

abbrev domain :=
  Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain

def implementationShape : PiCcsNc.Shape :=
  SourceRefinement.implementationShape domain

/-- One raw running assignment in canonical production column order. -/
def authoritativeRunningChild
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Sources.Data
      (semanticShape rowVariables freshCount runningCount matrixCount))
    (source : Fin runningCount) : List F :=
  List.ofFn fun column => data.runningAssignments source column

/-- The raw running-child table in canonical child order. Fresh outputs and
prover-carried evaluation sidecars are deliberately absent. -/
def authoritativeRunningChildren
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Sources.Data
      (semanticShape rowVariables freshCount runningCount matrixCount)) :
    List (List F) :=
  List.ofFn fun source => authoritativeRunningChild data source

@[simp] theorem authoritativeRunningChild_length
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Sources.Data
      (semanticShape rowVariables freshCount runningCount matrixCount))
    (source : Fin runningCount) :
    (authoritativeRunningChild data source).length =
      (semanticShape rowVariables freshCount runningCount matrixCount).carrierWidth := by
  simp [authoritativeRunningChild]

@[simp] theorem authoritativeRunningChildren_length
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Sources.Data
      (semanticShape rowVariables freshCount runningCount matrixCount)) :
    (authoritativeRunningChildren data).length = runningCount := by
  simp [authoritativeRunningChildren]

/-- Canonical list lookup is exactly the typed raw assignment coordinate. -/
theorem authoritativeRunningChild_getD
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Sources.Data
      (semanticShape rowVariables freshCount runningCount matrixCount))
    (source : Fin runningCount)
    (column : Fin
      (semanticShape rowVariables freshCount runningCount matrixCount).carrierWidth) :
    (authoritativeRunningChild data source).getD column.val 0 =
      data.runningAssignments source column := by
  have columnLt :
      column.val <
        Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.alignedPublicWidth := by
    simpa [semanticShape] using column.isLt
  simp [authoritativeRunningChild, columnLt]

/-- Two-dimensional raw-table lookup is exactly the typed running assignment;
neither a fresh output nor any carried evaluation vector is consulted. -/
theorem authoritativeRunningChildren_getD
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Sources.Data
      (semanticShape rowVariables freshCount runningCount matrixCount))
    (source : Fin runningCount)
    (column : Fin
      (semanticShape rowVariables freshCount runningCount matrixCount).carrierWidth) :
    ((authoritativeRunningChildren data).getD source.val []).getD
        column.val 0 = data.runningAssignments source column := by
  have childEq :
      (authoritativeRunningChildren data).getD source.val [] =
        authoritativeRunningChild data source := by
    unfold authoritativeRunningChildren
    rw [List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem (by simp),
      List.getElem_ofFn]
    rfl
  rw [childEq]
  exact authoritativeRunningChild_getD data source column

/-- The raw child list is the same assignment already owned by the canonical
joint Split-NC source family at the running-source injection. -/
theorem authoritativeRunningChild_eq_orderedAssignment
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Sources.Data
      (semanticShape rowVariables freshCount runningCount matrixCount))
    (source : Fin runningCount) :
    authoritativeRunningChild data source =
      data.orderedAssignment (Sources.Data.runningIndex source) := by
  unfold authoritativeRunningChild Sources.Data.orderedAssignment
    canonicalFinIndices
  rw [List.map_ofFn]
  apply congrArg List.ofFn
  funext column
  simp only [Function.comp_apply, id_eq]
  exact congrFun (data.assignment_runningIndex source).symm column

/-- Every authoritative raw running child fits the fixed 512-column
production flat domain. -/
theorem assignmentsFitColumnDomain
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Sources.Data
      (semanticShape rowVariables freshCount runningCount matrixCount)) :
    MixedPolynomial.AssignmentsFitColumnDomain implementationShape
      (authoritativeRunningChildren data) := by
  intro assignment member
  rcases List.mem_ofFn.mp member with ⟨source, rfl⟩
  rw [authoritativeRunningChild_length]
  exact
    (Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain_covers
      rowVariables freshCount runningCount matrixCount).1

theorem lanesCoverRing : ringDegree ≤ implementationShape.laneDomain := by
  decide

/-- Fixed-profile specialization of `OldPointBinding` to authoritative raw
running assignments. Child coverage and the 54-active/10-padding lane shape
are derived here; production still must derive the old-point length, padding,
and accepted compact identity from its actual acceptance path. -/
theorem acceptedProjectionIdentity_implies_oldPointRelation_or_badRoot
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Sources.Data
      (semanticShape rowVariables freshCount runningCount matrixCount))
    (radix : F) (parent : DelayedParent) (producerBeta : K)
    (pointLength : parent.sCol.length = implementationShape.ellM)
    (parentPadding : ∀ lane,
      ringDegree ≤ lane → lane < implementationShape.laneDomain →
      parent.yZcol lane = K.zero)
    (accepted : Nightstream.SuperNeo.ProjectionCheck.Accepted projectionOps
      (projectionIdentity implementationShape radix
        (authoritativeRunningChildren data) parent.sCol
        (delayedParentActiveCoefficients parent) producerBeta)) :
    OldPointSumcheckRelation implementationShape radix parent
        (authoritativeRunningChildren data) ∨
      Nightstream.SuperNeo.ProjectionCheck.BadRoot projectionOps
        (projectionIdentity implementationShape radix
          (authoritativeRunningChildren data) parent.sCol
          (delayedParentActiveCoefficients parent) producerBeta) := by
  apply
    Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.acceptedProjectionIdentity_implies_oldPointRelation_or_badRoot
      implementationShape radix parent (authoritativeRunningChildren data)
        producerBeta
  · exact ⟨pointLength, assignmentsFitColumnDomain data, lanesCoverRing⟩
  · exact parentPadding
  · exact accepted

end Fixed270

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.ProductionRawChildren
