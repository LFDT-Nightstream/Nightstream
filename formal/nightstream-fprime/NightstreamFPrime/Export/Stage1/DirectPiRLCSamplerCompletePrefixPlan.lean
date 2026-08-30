import NightstreamFPrime.Export.Stage1.DirectPiDECPrefixPlan
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectPlanSemantics

/-!
Owns the first direct 14-matrix prefix that includes every PiRLC sampler row.

Rows remain in protocol order: the established prefix through sampler
Poseidon2, sampler digest-lane and fail-closed selector rows, PiRLC arithmetic,
PiDEC, and the running transition. This module does not select an application
package or close any phase status.
-/

namespace NightstreamFPrime.Export.Stage1.DirectPiRLCSamplerCompletePrefixPlan

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

variable {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

private theorem append_eq_of_eq
    {logicalWidth : Nat}
    {leftA rightA leftB rightB : ProductionRelation.Plan logicalWidth}
    (leftEq : leftA = leftB) (rightEq : rightA = rightB)
    (fitsA : leftA.rowCount + rightA.rowCount ≤
      2 ^ Lifecycle.cubeVariables)
    (fitsB : leftB.rowCount + rightB.rowCount ≤
      2 ^ Lifecycle.cubeVariables) :
    ProductionRelation.Plan.append leftA rightA fitsA =
      ProductionRelation.Plan.append leftB rightB fitsB := by
  cases leftEq
  cases rightEq
  rfl

def piDecGeometry {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) : PiDECRetainedGeometry.Geometry application logicalWidth :=
  PiRLCSamplerOrdinaryRetainedGeometry.prefixGeometry geometry

def samplerPrefixPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) : ProductionRelation.Plan logicalWidth :=
  DirectPiDECPrefixPlan.samplerPrefixPlan relation (piDecGeometry geometry)

def samplerOrdinaryPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) : ProductionRelation.Plan logicalWidth :=
  PiRLCSamplerOrdinaryDirectPlan.plan relation geometry

def piRlcPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) : ProductionRelation.Plan logicalWidth :=
  DirectPiDECPrefixPlan.piRlcPlan (piDecGeometry geometry)

def piDecPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) : ProductionRelation.Plan logicalWidth :=
  DirectPiDECPrefixPlan.piDecPlan relation (piDecGeometry geometry)

def transitionPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) : ProductionRelation.Plan logicalWidth :=
  DirectPiDECPrefixPlan.transitionPlan relation (piDecGeometry geometry)

@[simp] theorem samplerPrefixPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) :
    (samplerPrefixPlan relation geometry).rowCount = 3711589 := by
  exact DirectPiDECPrefixPlan.samplerPrefixPlan_rowCount relation _

@[simp] theorem samplerOrdinaryPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) :
    (samplerOrdinaryPlan relation geometry).rowCount = 220881 := by
  exact PiRLCSamplerOrdinaryDirectPlan.plan_rowCount relation geometry

@[simp] theorem piRlcPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) :
    (piRlcPlan geometry).rowCount = 1773933 := by
  exact DirectPiDECPrefixPlan.piRlcPlan_rowCount _

private theorem samplerCompleteRowCount_le
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) :
    (samplerPrefixPlan relation geometry).rowCount +
        (samplerOrdinaryPlan relation geometry).rowCount ≤
      2 ^ Lifecycle.cubeVariables := by
  rw [samplerPrefixPlan_rowCount, samplerOrdinaryPlan_rowCount]
  norm_num [Lifecycle.cubeVariables]

def samplerCompletePlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) : ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (samplerPrefixPlan relation geometry)
    (samplerOrdinaryPlan relation geometry)
    (samplerCompleteRowCount_le relation geometry)

@[simp] theorem samplerCompletePlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) :
    (samplerCompletePlan relation geometry).rowCount = 3932470 := by
  simp [samplerCompletePlan]

private theorem piRlcCompleteRowCount_le
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) :
    (samplerCompletePlan relation geometry).rowCount +
        (piRlcPlan geometry).rowCount ≤ 2 ^ Lifecycle.cubeVariables := by
  rw [samplerCompletePlan_rowCount, piRlcPlan_rowCount]
  norm_num [Lifecycle.cubeVariables]

def piRlcCompletePlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) : ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (samplerCompletePlan relation geometry)
    (piRlcPlan geometry) (piRlcCompleteRowCount_le relation geometry)

@[simp] theorem piRlcCompletePlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) :
    (piRlcCompletePlan relation geometry).rowCount = 5706403 := by
  simp [piRlcCompletePlan]

private theorem piDecCompleteRowCount_le
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) :
    (piRlcCompletePlan relation geometry).rowCount +
        (piDecPlan relation geometry).rowCount ≤
      2 ^ Lifecycle.cubeVariables := by
  rw [piRlcCompletePlan_rowCount]
  rw [piDecPlan, DirectPiDECPrefixPlan.piDecPlan,
    PiDECDirectPlan.plan_rowCount]
  norm_num [Lifecycle.cubeVariables]

def piDecCompletePlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) : ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (piRlcCompletePlan relation geometry)
    (piDecPlan relation geometry) (piDecCompleteRowCount_le relation geometry)

@[simp] theorem piDecCompletePlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) :
    (piDecCompletePlan relation geometry).rowCount = 5731675 := by
  simp [piDecCompletePlan, piDecPlan, DirectPiDECPrefixPlan.piDecPlan]

private theorem totalRowCount_le
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) :
    (piDecCompletePlan relation geometry).rowCount +
        (transitionPlan relation geometry).rowCount ≤
      2 ^ Lifecycle.cubeVariables := by
  rw [piDecCompletePlan_rowCount]
  rw [transitionPlan, DirectPiDECPrefixPlan.transitionPlan,
    RunningTransitionDirectPlan.plan_rowCount]
  norm_num [Lifecycle.cubeVariables]

def plan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) : ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (piDecCompletePlan relation geometry)
    (transitionPlan relation geometry) (totalRowCount_le relation geometry)

@[simp] theorem plan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) :
    (plan relation geometry).rowCount = 6052978 := by
  simp [plan, transitionPlan, DirectPiDECPrefixPlan.transitionPlan]

theorem plan_eq_of_same_shape
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (left right : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) : plan left geometry = plan right geometry := by
  have piCcsOrdinaryEq :
      DirectPiDECPrefixPlan.piCcsOrdinaryPlan left (piDecGeometry geometry) =
        DirectPiDECPrefixPlan.piCcsOrdinaryPlan right
          (piDecGeometry geometry) := by
    exact PiCCSOrdinaryDirectPlan.plan_eq_of_same_shape left right _
  have piCcsCoreEq :
      DirectPiDECPrefixPlan.piCcsCorePlan left (piDecGeometry geometry) =
        DirectPiDECPrefixPlan.piCcsCorePlan right
          (piDecGeometry geometry) := by
    exact append_eq_of_eq rfl piCcsOrdinaryEq _ _
  have pilotOrdinaryPrefixEq :
      DirectPiDECPrefixPlan.pilotOrdinaryPrefixPlan left
          (piDecGeometry geometry) =
        DirectPiDECPrefixPlan.pilotOrdinaryPrefixPlan right
          (piDecGeometry geometry) := by
    exact append_eq_of_eq piCcsCoreEq rfl _ _
  have pilotBindingPrefixEq :
      DirectPiDECPrefixPlan.pilotBindingPrefixPlan left
          (piDecGeometry geometry) =
        DirectPiDECPrefixPlan.pilotBindingPrefixPlan right
          (piDecGeometry geometry) := by
    exact append_eq_of_eq pilotOrdinaryPrefixEq rfl _ _
  have piCcsCompleteEq :
      DirectPiDECPrefixPlan.piCcsCompletePlan left (piDecGeometry geometry) =
        DirectPiDECPrefixPlan.piCcsCompletePlan right
          (piDecGeometry geometry) := by
    exact append_eq_of_eq pilotBindingPrefixEq rfl _ _
  have samplerPrefixEq :
      samplerPrefixPlan left geometry = samplerPrefixPlan right geometry := by
    exact append_eq_of_eq piCcsCompleteEq rfl _ _
  have samplerOrdinaryEq :
      samplerOrdinaryPlan left geometry = samplerOrdinaryPlan right geometry :=
    PiRLCSamplerOrdinaryDirectPlan.plan_eq_of_same_shape left right geometry
  have samplerCompleteEq :
      samplerCompletePlan left geometry = samplerCompletePlan right geometry :=
    append_eq_of_eq samplerPrefixEq samplerOrdinaryEq _ _
  have piRlcCompleteEq :
      piRlcCompletePlan left geometry = piRlcCompletePlan right geometry :=
    append_eq_of_eq samplerCompleteEq rfl _ _
  have piDecEq : piDecPlan left geometry = piDecPlan right geometry :=
    PiDECDirectPlan.plan_eq_of_same_shape left right _
  have piDecCompleteEq :
      piDecCompletePlan left geometry = piDecCompletePlan right geometry :=
    append_eq_of_eq piRlcCompleteEq piDecEq _ _
  have transitionEq :
      transitionPlan left geometry = transitionPlan right geometry :=
    RunningTransitionDirectPlan.plan_eq_of_same_shape left right _
  exact append_eq_of_eq piDecCompleteEq transitionEq _ _

theorem rowsZero_iff
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) (assignment : Assignment F logicalWidth) :
    (plan relation geometry).RowsZero assignment ↔
      (samplerPrefixPlan relation geometry).RowsZero assignment ∧
        (samplerOrdinaryPlan relation geometry).RowsZero assignment ∧
          (piRlcPlan geometry).RowsZero assignment ∧
            (piDecPlan relation geometry).RowsZero assignment ∧
              (transitionPlan relation geometry).RowsZero assignment := by
  rw [plan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [piDecCompletePlan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [piRlcCompletePlan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [samplerCompletePlan, ProductionRelation.Plan.append_rowsZero_iff]
  constructor
  · rintro ⟨⟨⟨⟨samplerPrefix, samplerOrdinary⟩, piRlc⟩, piDec⟩,
      transition⟩
    exact ⟨samplerPrefix, samplerOrdinary, piRlc, piDec, transition⟩
  · rintro ⟨samplerPrefix, samplerOrdinary, piRlc, piDec, transition⟩
    exact ⟨⟨⟨⟨samplerPrefix, samplerOrdinary⟩, piRlc⟩, piDec⟩,
      transition⟩

private theorem priorRowsZero
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) (assignment : Assignment F logicalWidth)
    (samplerPrefix : (samplerPrefixPlan relation geometry).RowsZero assignment)
    (piRlc : (piRlcPlan geometry).RowsZero assignment)
    (piDec : (piDecPlan relation geometry).RowsZero assignment)
    (transition : (transitionPlan relation geometry).RowsZero assignment) :
    (DirectPiDECPrefixPlan.plan relation
      (piDecGeometry geometry)).RowsZero assignment := by
  unfold DirectPiDECPrefixPlan.plan
  rw [ProductionRelation.Plan.append_rowsZero_iff]
  unfold DirectPiDECPrefixPlan.piDecPrefixPlan
  rw [ProductionRelation.Plan.append_rowsZero_iff]
  unfold DirectPiDECPrefixPlan.piRlcPrefixPlan
  rw [ProductionRelation.Plan.append_rowsZero_iff]
  exact ⟨⟨⟨samplerPrefix, piRlc⟩, piDec⟩, transition⟩

/-- Complete encoding contract for the sampler-aware prefix. The old prefix
and the two new sampler ordinary blocks encode one canonical nested source
assignment. -/
structure Encodes
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) : Prop where
  prior : DirectPiDECPrefixPlan.Encodes (piDecGeometry geometry) assignment
    base groupValue products
  samplerOrdinary : PiRLCSamplerOrdinaryRetainedGeometry.Encodes geometry
    assignment (PiRLCRetainedPreservation.sourceAssignment application base
      groupValue products)

structure Semantics
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) : Prop where
  prior : DirectPiDECPrefixPlan.Semantics relation (piDecGeometry geometry)
    assignment base groupValue products
  samplerOrdinary : R1CS.RowsHold
    (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment)
    (PiRLCSamplerOrdinaryDirectSource.sourceRows
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits))

theorem rowsZero_implies_semantics
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment base groupValue products)
    (rowsZero : (plan relation geometry).RowsZero assignment) :
    Semantics relation geometry assignment base groupValue products := by
  have children := (rowsZero_iff relation geometry assignment).mp rowsZero
  rcases children with
    ⟨samplerPrefixRows, samplerOrdinaryRows, piRlcRows, piDecRows,
      transitionRows⟩
  have oldRows := priorRowsZero relation geometry assignment samplerPrefixRows
    piRlcRows piDecRows transitionRows
  have priorOne :
      assignment (PiDECRetainedGeometry.oneColumn (piDecGeometry geometry)) = 1 :=
    one
  refine ⟨DirectPiDECPrefixPlan.rowsZero_implies_semantics relation
    (piDecGeometry geometry) assignment base groupValue products priorOne
    encodes.prior oldRows, ?_⟩
  exact (PiRLCSamplerOrdinaryDirectPlan.rowsZero_iff_rowsHold relation geometry
    assignment one).mp samplerOrdinaryRows

end NightstreamFPrime.Export.Stage1.DirectPiRLCSamplerCompletePrefixPlan
