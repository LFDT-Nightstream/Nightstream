import NightstreamFPrime.Export.Stage1.DirectPrefixPlan
import NightstreamFPrime.Export.Stage1.RunningTransitionDirectPlan

/-!
Owns the one ordered direct 14-matrix plan through the running-instance
transition. The existing pilot/PiCCS/PiRLC direct prefix comes first, followed
by the canonical transition rows.

This remains a phase-local plan. It does not include accumulator,
application, output-hash, terminal, or final package-identity work.
-/

namespace NightstreamFPrime.Export.Stage1.DirectRunningPrefixPlan

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def prefixGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : RunningTransitionRetainedGeometry.Geometry program logicalWidth) :
    PiCCSPoseidonPlan.Geometry program logicalWidth :=
  RunningTransitionRetainedGeometry.poseidonGeometry geometry

def prefixPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : RunningTransitionRetainedGeometry.Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  DirectPrefixPlan.plan payloadForms (prefixGeometry geometry)

def transitionPlan
    {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (geometry : RunningTransitionRetainedGeometry.Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  RunningTransitionDirectPlan.plan relation geometry

theorem rowCount_le
    {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : RunningTransitionRetainedGeometry.Geometry program logicalWidth) :
    (prefixPlan payloadForms geometry).rowCount + (transitionPlan relation geometry).rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [prefixPlan, DirectPrefixPlan.plan_rowCount, transitionPlan,
    RunningTransitionDirectPlan.plan_rowCount]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

def plan
    {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : RunningTransitionRetainedGeometry.Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (prefixPlan payloadForms geometry)
    (transitionPlan relation geometry) (rowCount_le relation payloadForms geometry)

@[simp] theorem plan_rowCount
    {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : RunningTransitionRetainedGeometry.Geometry program logicalWidth) :
    (plan relation payloadForms geometry).rowCount = 5310442 := by
  simp [plan, prefixPlan, transitionPlan]

theorem rowsZero_iff
    {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : RunningTransitionRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :
    (plan relation payloadForms geometry).RowsZero assignment ↔
      (prefixPlan payloadForms geometry).RowsZero assignment ∧
        (transitionPlan relation geometry).RowsZero assignment := by
  exact ProductionRelation.Plan.append_rowsZero_iff _ _
    (rowCount_le relation payloadForms geometry) assignment

structure Encodes
    {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : RunningTransitionRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) : Prop where
  prior : DirectPrefixPlan.Encodes payloadForms (prefixGeometry geometry) assignment
    base groupValue products
  transition : RunningTransitionRetainedGeometry.Encodes geometry assignment
    (PiRLCRetainedPreservation.sourceAssignment
      program base groupValue products)

structure Semantics
    {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : RunningTransitionRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) : Prop where
  prior : DirectPrefixPlan.Semantics payloadForms (prefixGeometry geometry) assignment base
    groupValue products
  transition : NightstreamFPrime.Layout.Stage1.RunningTransitionLayout.PhysicalHolds
    logicalWidth publicFits
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback
      (RunningTransitionDirectPlan.transitionEnv program base))

private theorem prefixOne
    {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : RunningTransitionRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment
      (RunningTransitionRetainedGeometry.oneColumn geometry) = 1) :
    assignment (PiCCSPoseidonPlan.oneColumn (prefixGeometry geometry)) = 1 := by
  have columns :
      PiCCSPoseidonPlan.oneColumn (prefixGeometry geometry) =
        RunningTransitionRetainedGeometry.oneColumn geometry := by
    apply Fin.ext
    rfl
  rw [columns]
  exact one

/-- One direct-plan acceptance proof forces every prior direct semantic and
the canonical running transition on the same per-application package data. -/
theorem rowsZero_implies_semantics
    {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : RunningTransitionRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (RunningTransitionRetainedGeometry.oneColumn geometry) = 1)
    (encodes : Encodes payloadForms geometry assignment base groupValue products)
    (rowsZero : (plan relation payloadForms geometry).RowsZero assignment) :
    Semantics relation payloadForms geometry assignment base groupValue products := by
  have children := (rowsZero_iff relation payloadForms geometry assignment).mp rowsZero
  refine ⟨?_, ?_⟩
  · exact DirectPrefixPlan.rowsZero_implies_semantics
      payloadForms (prefixGeometry geometry) assignment base groupValue products
      (prefixOne geometry assignment one) encodes.prior children.1
  · exact (RunningTransitionDirectPlan.rowsZero_iff_physical
      relation geometry assignment base groupValue products one
      encodes.transition).mp children.2

end NightstreamFPrime.Export.Stage1.DirectRunningPrefixPlan
