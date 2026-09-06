import NightstreamFPrime.Export.Stage1.DirectRunningPrefixPlan
import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryDirectPlan
import NightstreamFPrime.Export.Stage1.PiCCSPayloadWiring
import NightstreamFPrime.Export.Stage1.PiCCSTranscriptEndpointPlan
import NightstreamFPrime.Export.Stage1.PiDECDirectPlan
import NightstreamFPrime.Export.Stage1.PilotDirectSemantics

/-!
Owns the first complete ordered direct 14-matrix prefix through PiDEC and the
running-instance transition.

Rows occur in protocol order: pilot Poseidon2, PiCCS Poseidon2, PiCCS ordinary,
pilot ordinary and digest-custody rows, PiRLC sampler Poseidon2, PiRLC
arithmetic, PiDEC, then the running transition. This remains phase-local
evidence. It does not select a concrete application, close a package identity,
or include application rows.
-/

namespace NightstreamFPrime.Export.Stage1.DirectPiDECPrefixPlan

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
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

def piCcsOrdinaryGeometry
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    PiCCSOrdinaryRetainedGeometry.Geometry application logicalWidth :=
  PiDECRetainedGeometry.prefixGeometry geometry

def pilotOrdinaryGeometry
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    PilotOrdinaryRetainedGeometry.Geometry application logicalWidth :=
  PiDECRetainedGeometry.pilotOrdinaryGeometry geometry

def runningGeometry
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    RunningTransitionRetainedGeometry.Geometry application logicalWidth :=
  PiCCSOrdinaryRetainedGeometry.prefixGeometry
    (piCcsOrdinaryGeometry geometry)

def poseidonGeometry
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    PiCCSPoseidonPlan.Geometry application logicalWidth :=
  DirectRunningPrefixPlan.prefixGeometry (runningGeometry geometry)

/-- The phase parent selects the declared action values from its source map. -/
def piCcsPayload
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    PiCCSPoseidonPlan.Payload logicalWidth :=
  PiCCSPayloadWiring.form (piCcsOrdinaryGeometry geometry)

def pilotPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  DirectPrefixPlan.pilotPlan (poseidonGeometry geometry)

def piCcsPoseidonPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  DirectPrefixPlan.piCcsPlan (piCcsPayload geometry) (poseidonGeometry geometry)

def piCcsOrdinaryPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PiCCSOrdinaryDirectPlan.plan relation (piCcsOrdinaryGeometry geometry)

def pilotOrdinaryPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PilotOrdinaryDirectPlan.plan (pilotOrdinaryGeometry geometry)

def pilotBindingPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PilotDigestBindingPlan.plan (pilotOrdinaryGeometry geometry)

def piCcsEndpointPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PiCCSTranscriptEndpointPlan.plan (poseidonGeometry geometry)
    (piCcsOrdinaryGeometry geometry)

def samplerPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  DirectPrefixPlan.samplerPlan (poseidonGeometry geometry)

def piRlcPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  DirectPrefixPlan.piRlcPlan (poseidonGeometry geometry)

def piDecPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PiDECDirectPlan.plan relation geometry

def transitionPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  RunningTransitionDirectPlan.plan relation (runningGeometry geometry)

@[simp] theorem pilotPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (pilotPlan geometry).rowCount = 2321800 := by
  simp [pilotPlan, DirectPrefixPlan.pilotPlan]

@[simp] theorem piCcsPoseidonPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (piCcsPoseidonPlan geometry).rowCount = 729984 := by
  simp [piCcsPoseidonPlan, DirectPrefixPlan.piCcsPlan]

@[simp] theorem piCcsEndpointPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (piCcsEndpointPlan geometry).rowCount = 32 := by
  simp [piCcsEndpointPlan]

@[simp] theorem pilotOrdinaryPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (pilotOrdinaryPlan geometry).rowCount = 1330 := by
  simp [pilotOrdinaryPlan]

@[simp] theorem pilotBindingPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (pilotBindingPlan geometry).rowCount = 8 := by
  simp [pilotBindingPlan]

@[simp] theorem samplerPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (samplerPlan geometry).rowCount = 14382 := by
  simp [samplerPlan, DirectPrefixPlan.samplerPlan]

@[simp] theorem piRlcPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (piRlcPlan geometry).rowCount = 1898781 := by
  simp [piRlcPlan, DirectPrefixPlan.piRlcPlan]

private theorem piCcsPoseidonRowCount_le
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (pilotPlan geometry).rowCount + (piCcsPoseidonPlan geometry).rowCount ≤
      2 ^ Lifecycle.cubeVariables := by
  rw [pilotPlan_rowCount, piCcsPoseidonPlan_rowCount]
  norm_num [Lifecycle.cubeVariables]

def piCcsPoseidonPrefix
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (pilotPlan geometry)
    (piCcsPoseidonPlan geometry) (piCcsPoseidonRowCount_le geometry)

@[simp] theorem piCcsPoseidonPrefix_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (piCcsPoseidonPrefix geometry).rowCount = 3051784 := by
  simp [piCcsPoseidonPrefix]

private theorem piCcsCoreRowCount_le
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (piCcsPoseidonPrefix geometry).rowCount +
        (piCcsOrdinaryPlan relation geometry).rowCount ≤
      2 ^ Lifecycle.cubeVariables := by
  rw [piCcsPoseidonPrefix_rowCount]
  rw [piCcsOrdinaryPlan, PiCCSOrdinaryDirectPlan.plan_rowCount]
  norm_num [Lifecycle.cubeVariables]

def piCcsCorePlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (piCcsPoseidonPrefix geometry)
    (piCcsOrdinaryPlan relation geometry)
    (piCcsCoreRowCount_le relation geometry)

@[simp] theorem piCcsCorePlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (piCcsCorePlan relation geometry).rowCount = 3863453 := by
  simp [piCcsCorePlan, piCcsOrdinaryPlan]

private theorem pilotOrdinaryPrefixRowCount_le
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (piCcsCorePlan relation geometry).rowCount +
        (pilotOrdinaryPlan geometry).rowCount ≤
      2 ^ Lifecycle.cubeVariables := by
  rw [piCcsCorePlan_rowCount, pilotOrdinaryPlan_rowCount]
  norm_num [Lifecycle.cubeVariables]

def pilotOrdinaryPrefixPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (piCcsCorePlan relation geometry)
    (pilotOrdinaryPlan geometry)
    (pilotOrdinaryPrefixRowCount_le relation geometry)

@[simp] theorem pilotOrdinaryPrefixPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (pilotOrdinaryPrefixPlan relation geometry).rowCount = 3864783 := by
  simp [pilotOrdinaryPrefixPlan]

private theorem pilotBindingPrefixRowCount_le
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (pilotOrdinaryPrefixPlan relation geometry).rowCount +
        (pilotBindingPlan geometry).rowCount ≤
      2 ^ Lifecycle.cubeVariables := by
  rw [pilotOrdinaryPrefixPlan_rowCount, pilotBindingPlan_rowCount]
  norm_num [Lifecycle.cubeVariables]

def pilotBindingPrefixPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (pilotOrdinaryPrefixPlan relation geometry)
    (pilotBindingPlan geometry)
    (pilotBindingPrefixRowCount_le relation geometry)

@[simp] theorem pilotBindingPrefixPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (pilotBindingPrefixPlan relation geometry).rowCount = 3864791 := by
  simp [pilotBindingPrefixPlan]

private theorem piCcsCompleteRowCount_le
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (pilotBindingPrefixPlan relation geometry).rowCount +
        (piCcsEndpointPlan geometry).rowCount ≤
      2 ^ Lifecycle.cubeVariables := by
  rw [pilotBindingPrefixPlan_rowCount, piCcsEndpointPlan_rowCount]
  norm_num [Lifecycle.cubeVariables]

def piCcsCompletePlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (pilotBindingPrefixPlan relation geometry)
    (piCcsEndpointPlan geometry)
    (piCcsCompleteRowCount_le relation geometry)

@[simp] theorem piCcsCompletePlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (piCcsCompletePlan relation geometry).rowCount = 3864823 := by
  simp [piCcsCompletePlan]

private theorem samplerPrefixRowCount_le
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (piCcsCompletePlan relation geometry).rowCount +
        (samplerPlan geometry).rowCount ≤ 2 ^ Lifecycle.cubeVariables := by
  rw [piCcsCompletePlan_rowCount, samplerPlan_rowCount]
  norm_num [Lifecycle.cubeVariables]

def samplerPrefixPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (piCcsCompletePlan relation geometry)
    (samplerPlan geometry) (samplerPrefixRowCount_le relation geometry)

@[simp] theorem samplerPrefixPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (samplerPrefixPlan relation geometry).rowCount = 3879205 := by
  simp [samplerPrefixPlan]

private theorem piRlcPrefixRowCount_le
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (samplerPrefixPlan relation geometry).rowCount +
        (piRlcPlan geometry).rowCount ≤ 2 ^ Lifecycle.cubeVariables := by
  rw [samplerPrefixPlan_rowCount, piRlcPlan_rowCount]
  norm_num [Lifecycle.cubeVariables]

def piRlcPrefixPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (samplerPrefixPlan relation geometry)
    (piRlcPlan geometry) (piRlcPrefixRowCount_le relation geometry)

@[simp] theorem piRlcPrefixPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (piRlcPrefixPlan relation geometry).rowCount = 5777986 := by
  simp [piRlcPrefixPlan]

private theorem piDecPrefixRowCount_le
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (piRlcPrefixPlan relation geometry).rowCount +
        (piDecPlan relation geometry).rowCount ≤
      2 ^ Lifecycle.cubeVariables := by
  rw [piRlcPrefixPlan_rowCount]
  rw [piDecPlan, PiDECDirectPlan.plan_rowCount]
  norm_num [Lifecycle.cubeVariables]

def piDecPrefixPlan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (piRlcPrefixPlan relation geometry)
    (piDecPlan relation geometry) (piDecPrefixRowCount_le relation geometry)

@[simp] theorem piDecPrefixPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (piDecPrefixPlan relation geometry).rowCount = 5803474 := by
  simp [piDecPrefixPlan, piDecPlan]

private theorem totalRowCount_le
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (piDecPrefixPlan relation geometry).rowCount +
        (transitionPlan relation geometry).rowCount ≤
      2 ^ Lifecycle.cubeVariables := by
  rw [piDecPrefixPlan_rowCount]
  rw [transitionPlan, RunningTransitionDirectPlan.plan_rowCount]
  norm_num [Lifecycle.cubeVariables]

def plan
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (piDecPrefixPlan relation geometry)
    (transitionPlan relation geometry) (totalRowCount_le relation geometry)

@[simp] theorem plan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    (plan relation geometry).rowCount = 6148969 := by
  simp [plan, transitionPlan]

/-- The complete ordered prefix depends on the verified relation shape, but
not on matrix entries. Each relation-sensitive leaf has this property before
the plans are appended. -/
theorem plan_eq_of_same_shape
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (left right : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth) :
    plan left geometry = plan right geometry := by
  have piCcsCoreEq :
      piCcsCorePlan left geometry = piCcsCorePlan right geometry := by
    apply append_eq_of_eq
    · rfl
    · exact PiCCSOrdinaryDirectPlan.plan_eq_of_same_shape left right _
  have pilotOrdinaryPrefixEq :
      pilotOrdinaryPrefixPlan left geometry =
        pilotOrdinaryPrefixPlan right geometry := by
    exact append_eq_of_eq piCcsCoreEq rfl _ _
  have pilotBindingPrefixEq :
      pilotBindingPrefixPlan left geometry =
        pilotBindingPrefixPlan right geometry := by
    exact append_eq_of_eq pilotOrdinaryPrefixEq rfl _ _
  have piCcsCompleteEq :
      piCcsCompletePlan left geometry = piCcsCompletePlan right geometry := by
    exact append_eq_of_eq pilotBindingPrefixEq rfl _ _
  have samplerPrefixEq :
      samplerPrefixPlan left geometry = samplerPrefixPlan right geometry := by
    exact append_eq_of_eq piCcsCompleteEq rfl _ _
  have piRlcPrefixEq :
      piRlcPrefixPlan left geometry = piRlcPrefixPlan right geometry := by
    exact append_eq_of_eq samplerPrefixEq rfl _ _
  have piDecPrefixEq :
      piDecPrefixPlan left geometry = piDecPrefixPlan right geometry := by
    exact append_eq_of_eq piRlcPrefixEq
      (PiDECDirectPlan.plan_eq_of_same_shape left right _) _ _
  exact append_eq_of_eq piDecPrefixEq
    (RunningTransitionDirectPlan.plan_eq_of_same_shape left right _) _ _

theorem rowsZero_iff
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) :
    (plan relation geometry).RowsZero assignment ↔
      (pilotPlan geometry).RowsZero assignment ∧
        (piCcsPoseidonPlan geometry).RowsZero assignment ∧
          (piCcsOrdinaryPlan relation geometry).RowsZero assignment ∧
            (pilotOrdinaryPlan geometry).RowsZero assignment ∧
              (pilotBindingPlan geometry).RowsZero assignment ∧
                (piCcsEndpointPlan geometry).RowsZero assignment ∧
                  (samplerPlan geometry).RowsZero assignment ∧
                    (piRlcPlan geometry).RowsZero assignment ∧
                      (piDecPlan relation geometry).RowsZero assignment ∧
                        (transitionPlan relation geometry).RowsZero assignment := by
  rw [plan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [piDecPrefixPlan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [piRlcPrefixPlan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [samplerPrefixPlan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [piCcsCompletePlan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [pilotBindingPrefixPlan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [pilotOrdinaryPrefixPlan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [piCcsCorePlan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [piCcsPoseidonPrefix, ProductionRelation.Plan.append_rowsZero_iff]
  constructor
  · rintro ⟨⟨⟨⟨⟨⟨⟨⟨⟨pilot, piCcsPoseidon⟩, piCcsOrdinary⟩,
        pilotOrdinary⟩, pilotBinding⟩, piCcsEndpoint⟩, sampler⟩, piRlc⟩,
        piDec⟩, transition⟩
    exact ⟨pilot, piCcsPoseidon, piCcsOrdinary, pilotOrdinary, pilotBinding,
      piCcsEndpoint, sampler, piRlc, piDec, transition⟩
  · rintro ⟨pilot, piCcsPoseidon, piCcsOrdinary, pilotOrdinary, pilotBinding,
      piCcsEndpoint, sampler, piRlc, piDec, transition⟩
    exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨pilot, piCcsPoseidon⟩, piCcsOrdinary⟩,
      pilotOrdinary⟩, pilotBinding⟩, piCcsEndpoint⟩, sampler⟩, piRlc⟩,
      piDec⟩, transition⟩

structure Encodes
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) : Prop where
  running : DirectRunningPrefixPlan.Encodes (piCcsPayload geometry) (runningGeometry geometry) assignment
    base groupValue products
  pilotOrdinary : PilotOrdinaryDirectPlan.Encodes
    (pilotOrdinaryGeometry geometry) assignment base groupValue products
  piDec : PiDECRetainedGeometry.Encodes geometry assignment
    (PiRLCRetainedPreservation.sourceAssignment application base groupValue
      products)

structure Semantics
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) : Prop where
  prior : DirectPrefixPlan.Semantics (piCcsPayload geometry) (poseidonGeometry geometry) assignment base
    groupValue products
  pilot : Lifecycle.Pilot.SpecHolds PilotProduction.interface
    PilotProduction.witnessOffset
    (PilotSpartan.pullback
      (PilotOrdinaryDirectPlan.pilotEnv application base))
  piCcsTranscript : PiCCSInvocations.TranscriptSpecs relationLogicalWidth
    relationPublicFits
    (PiCCSTranscriptEndpointPlan.transcriptEnv application base groupValue
      products)
  piCcsEndpoint : (piCcsEndpointPlan geometry).RowsZero assignment
  piCcsOrdinary : R1CS.RowsHold
    (RunningTransitionDirectPlan.transitionEnv application base)
    (PiCCSOrdinaryDirectSource.sourceRows relationLogicalWidth
      relationPublicFits)
  piDec : R1CS.RowsHold
    (RunningTransitionDirectPlan.transitionEnv application base)
    (PiDECOrdinaryDirectSource.sourceRows relationLogicalWidth
      relationPublicFits)
  transition : RunningTransitionLayout.PhysicalHolds relationLogicalWidth
    relationPublicFits
    (Spartan.pullback
      (RunningTransitionDirectPlan.transitionEnv application base))

theorem rowsZero_implies_semantics
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (PiDECRetainedGeometry.oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment base groupValue products)
    (rowsZero : (plan relation geometry).RowsZero assignment) :
    Semantics relation geometry assignment base groupValue products := by
  have children := (rowsZero_iff relation geometry assignment).mp rowsZero
  rcases children with ⟨pilotRows, piCcsPoseidonRows, piCcsOrdinaryRows,
    pilotOrdinaryRows, pilotBindingRows, piCcsEndpointRows, samplerRows,
    piRlcRows, piDecRows, transitionRows⟩
  have directPrefixRows :
      (DirectPrefixPlan.plan (piCcsPayload geometry) (poseidonGeometry geometry)).RowsZero assignment := by
    apply (DirectPrefixPlan.rowsZero_iff (piCcsPayload geometry) (poseidonGeometry geometry)
      assignment).mpr
    exact ⟨pilotRows, piCcsPoseidonRows, samplerRows, piRlcRows⟩
  have prior := DirectPrefixPlan.rowsZero_implies_semantics
      (piCcsPayload geometry) (poseidonGeometry geometry) assignment base groupValue products one
      encodes.running.prior directPrefixRows
  have pilotOne : assignment
      (PilotOrdinaryDirectPlan.oneColumn
        (pilotOrdinaryGeometry geometry)) = 1 := by
    exact one
  have pilotOrdinaryRowsHold :=
    (PilotOrdinaryDirectPlan.rowsZero_iff_rowsHold
      (pilotOrdinaryGeometry geometry) assignment base groupValue products
      pilotOne encodes.pilotOrdinary).mp pilotOrdinaryRows
  have pilotBinding :=
    (PilotDigestBindingPlan.rowsZero_iff_matches
      (pilotOrdinaryGeometry geometry) assignment pilotOne).mp pilotBindingRows
  have pilotEncoding : PilotPoseidonPreservation.Encoding
      (PilotDirectSemantics.poseidonGeometry
        (pilotOrdinaryGeometry geometry)) assignment base groupValue products :=
    { priorInput := encodes.running.prior.pilotPriorInput
      outputInput := encodes.running.prior.pilotOutputInput }
  have pilotHashes := PilotPoseidonPreservation.semantics_imply_hashFacts
    (PilotDirectSemantics.poseidonGeometry (pilotOrdinaryGeometry geometry))
    assignment (PilotOrdinaryDirectPlan.pilotEnv application base) pilotOne
    (PilotPoseidonPreservation.priorInputForm_eval _ assignment base groupValue
      products pilotEncoding)
    (PilotPoseidonPreservation.outputInputForm_eval _ assignment base groupValue
      products pilotEncoding)
    prior.pilot
  have pilotSpec := PilotDirectSemantics.implies_spec
    (pilotOrdinaryGeometry geometry) assignment
    (PilotOrdinaryDirectPlan.pilotEnv application base) pilotHashes
    pilotOrdinaryRowsHold pilotBinding
    (fun lane => PilotOrdinaryDirectPlan.Location.form_eval
      (pilotOrdinaryGeometry geometry) assignment base groupValue products
      encodes.pilotOrdinary (.priorDigest lane))
    (fun lane => PilotOrdinaryDirectPlan.Location.form_eval
      (pilotOrdinaryGeometry geometry) assignment base groupValue products
      encodes.pilotOrdinary (.outputState lane))
  have traces := PiCCSTranscriptDirectSemantics.indexedSemantics_implies_traces
    (poseidonGeometry geometry) assignment
    (PiCCSActionPayloadBlock.packageEnv application
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue
        products))
    (PiCCSPoseidonPreservation.indexedSemantics (poseidonGeometry geometry)
      assignment
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue
        products) prior.piCcs)
  refine ⟨prior, pilotSpec, ?_, piCcsEndpointRows, ?_, ?_, ?_⟩
  · exact PiCCSTranscriptEndpointPlan.traces_and_endpoints_imply_transcriptSpecs
      (relationLogicalWidth := relationLogicalWidth)
      (relationPublicFits := relationPublicFits)
      (poseidonGeometry geometry) assignment
      (PiCCSTranscriptEndpointPlan.transcriptEnv application base groupValue products)
      traces
      (PiCCSTranscriptEndpointPlan.rowsZero_implies_endpointState
        (poseidonGeometry geometry) (piCcsOrdinaryGeometry geometry) assignment
        base groupValue products one encodes.pilotOrdinary.prior piCcsEndpointRows)
  · exact (PiCCSOrdinaryDirectPlan.rowsZero_iff_rowsHold relation
      (piCcsOrdinaryGeometry geometry) assignment base groupValue products one
      encodes.pilotOrdinary.prior).mp piCcsOrdinaryRows
  · exact (PiDECDirectPlan.rowsZero_iff_rowsHold relation geometry assignment
      base groupValue products one encodes.piDec).mp piDecRows
  · exact (RunningTransitionDirectPlan.rowsZero_iff_physical relation
      (runningGeometry geometry) assignment base groupValue products one
      encodes.running.transition).mp transitionRows

theorem rowsZero_implies_piCcsSpecHolds
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (PiDECRetainedGeometry.oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment base groupValue products)
    (rowsZero : (plan relation geometry).RowsZero assignment) :
    Lifecycle.PiCCS.v1_1.Formal.SpecHolds relation
      (PiCCSInvocations.parentInterface relationLogicalWidth
        relationPublicFits)
      PiCCSInputs.phaseOffset
      (Spartan.pullback
        (PiCCSTranscriptEndpointPlan.transcriptEnv application base groupValue
          products)) := by
  have semantics := rowsZero_implies_semantics relation geometry assignment base
    groupValue products one encodes rowsZero
  have ordinaryAtTranscript : R1CS.RowsHold
      (PiCCSTranscriptEndpointPlan.transcriptEnv application base groupValue
        products)
      (PiCCSOrdinaryDirectSource.sourceRows relationLogicalWidth
        relationPublicFits) := by
    apply R1CS.rowsHold_of_agree_below
      (PiCCSOrdinaryDirectSource.sourceRows relationLogicalWidth
        relationPublicFits)
      Spartan.spartanColumnCount
      (RunningTransitionDirectPlan.transitionEnv application base)
      (PiCCSTranscriptEndpointPlan.transcriptEnv application base groupValue
        products)
      (PiCCSOrdinaryDirectSource.sourceRows_varsBelow relation)
    · intro column bound
      exact PiCCSTranscriptEndpointPlan.transcriptEnv_eq_transitionEnv_of_lt
        application base groupValue products column bound
    · exact semantics.piCcsOrdinary
  have packets := PiCCSArithmetic.arithmeticRows_imply_packetHolds
    relationLogicalWidth relationPublicFits
    (PiCCSTranscriptEndpointPlan.transcriptEnv application base groupValue
      products) ordinaryAtTranscript
  have assumptions :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.production relation
      (PiCCSInvocations.parentInterface relationLogicalWidth
        relationPublicFits)
      PiCCSInputs.phaseOffset
      (PiCCSInputs.externalInputsLinear relationLogicalWidth
        relationPublicFits)
      (Spartan.pullback
        (PiCCSTranscriptEndpointPlan.transcriptEnv application base groupValue
          products))
  have arithmetic := PiCCSArithmetic.packetHolds_imply_arithmeticSpecs
    relationLogicalWidth relationPublicFits relation
    (PiCCSTranscriptEndpointPlan.transcriptEnv application base groupValue
      products) assumptions packets
  refine {
    statementBinding := arithmetic.statementBinding_parent
    statementAbsorption :=
      semantics.piCcsTranscript.statementAbsorption_parent
    challenge := semantics.piCcsTranscript.challengeDerivation_parent
    roundTranscript := semantics.piCcsTranscript.roundTranscript_parent
    initialClaim := arithmetic.initialClaim_parent
    sumcheck := arithmetic.sumcheck_parent
    eval_K := arithmetic.evalK_parent
    eval_A := arithmetic.evalA_parent
    ccs := arithmetic.ccs_parent
    norm := arithmetic.norm_parent
    finalIdentity := arithmetic.finalIdentity_parent
    outputBinding := semantics.piCcsTranscript.outputBinding_parent relation }

theorem rowsZero_implies_piCcsPhaseHolds
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (ajtai : AjtaiKey
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits))
    (template : Proof (ProductionKey.degreeBound relation))
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (PiDECRetainedGeometry.oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment base groupValue products)
    (rowsZero : (plan relation geometry).RowsZero assignment) :
    Lifecycle.PiCCS.v1_1.Formal.PhaseHolds relation ajtai
      (PiCCSInvocations.parentInterface relationLogicalWidth
        relationPublicFits)
      PiCCSInputs.phaseOffset
      (Spartan.pullback
        (PiCCSTranscriptEndpointPlan.transcriptEnv application base groupValue
          products)) template := by
  apply Lifecycle.PiCCS.v1_1.Formal.spec_implies_phaseHolds relation ajtai
  exact rowsZero_implies_piCcsSpecHolds relation geometry assignment base
    groupValue products one encodes rowsZero

theorem rowsZero_implies_piDecPhaseHolds
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (ajtai : AjtaiKey
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits))
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (PiDECRetainedGeometry.oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment base groupValue products)
    (assumptions : Lifecycle.PiDEC.v1_1.Formal.Assumptions relation
      (PiDECArithmetic.phaseInterface relationLogicalWidth
        relationPublicFits)
      PiDECInputs.phaseOffset
      (Spartan.pullback
        (RunningTransitionDirectPlan.transitionEnv application base)))
    (rowsZero : (plan relation geometry).RowsZero assignment) :
    Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds relation ajtai
      (PiDECArithmetic.phaseInterface relationLogicalWidth relationPublicFits)
      PiDECInputs.phaseOffset
      (Spartan.pullback
        (RunningTransitionDirectPlan.transitionEnv application base)) := by
  have children := (rowsZero_iff relation geometry assignment).mp rowsZero
  rcases children with ⟨_, _, _, _, _, _, _, _, piDecRows, _⟩
  exact PiDECDirectPlan.rowsZero_implies_phaseHolds relation ajtai geometry
    assignment base groupValue products one encodes.piDec assumptions
    piDecRows

end NightstreamFPrime.Export.Stage1.DirectPiDECPrefixPlan
