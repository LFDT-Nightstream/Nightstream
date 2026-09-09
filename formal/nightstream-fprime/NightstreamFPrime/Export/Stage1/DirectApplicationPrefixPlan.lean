import NightstreamFPrime.Export.Stage1.ApplicationDirectPlan
import NightstreamFPrime.Export.Stage1.DirectPiRLCSamplerCompletePrefixPlan
import NightstreamFPrime.Export.Stage1.NextPreimageDirectPlan
import NightstreamFPrime.Export.Stage1.RecursivePublicOutputPlan

/-!
Owns the ordered direct 14-matrix plan through one verifier-selected
application. The running prefix comes first; application rows follow and read
the actual pilot preimage forms. Only application witness/local values add
retained coordinates.
-/

namespace NightstreamFPrime.Export.Stage1.DirectApplicationPrefixPlan

open NightstreamFPrime.Circuit
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

def prefixGeometry {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    PiRLCSamplerOrdinaryRetainedGeometry.Geometry application logicalWidth :=
  ApplicationRetainedGeometry.prefixGeometry geometry

def prefixPlan
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  DirectPiRLCSamplerCompletePrefixPlan.plan relation (prefixGeometry geometry)

def applicationPlan
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ApplicationDirectPlan.plan fits geometry

def piDecGeometry
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    PiDECRetainedGeometry.Geometry application logicalWidth :=
  DirectPiRLCSamplerCompletePrefixPlan.piDecGeometry (prefixGeometry geometry)

def pilotOrdinaryGeometry
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    PilotOrdinaryRetainedGeometry.Geometry application logicalWidth :=
  DirectPiDECPrefixPlan.pilotOrdinaryGeometry (piDecGeometry geometry)

def piCcsOrdinaryGeometry
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    PiCCSOrdinaryRetainedGeometry.Geometry application logicalWidth :=
  PilotOrdinaryDirectPlan.piCcsGeometry (pilotOrdinaryGeometry geometry)

def nextPreimagePlan
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  NextPreimageDirectPlan.plan (piCcsOrdinaryGeometry geometry)

def publicOutputPlan
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  RecursivePublicOutputPlan.plan geometry

theorem rowCount_le
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    (((prefixPlan relation geometry).rowCount +
        (applicationPlan fits geometry).rowCount) +
        (nextPreimagePlan geometry).rowCount) +
        (publicOutputPlan geometry).rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  have packageRows := fits.rows
  rw [PerApplicationPackage.package_rowCount] at packageRows
  have baseRows : PerApplicationPackage.basePackage.layout.rowCount =
      29218024 := by
    simpa [PerApplicationPackage.basePackage] using
      NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.1
  rw [baseRows] at packageRows
  rw [prefixPlan, DirectPiRLCSamplerCompletePrefixPlan.plan_rowCount,
    applicationPlan, ApplicationDirectPlan.plan_rowCount,
    nextPreimagePlan, NextPreimageDirectPlan.plan_rowCount,
    publicOutputPlan, RecursivePublicOutputPlan.plan_rowCount]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables] at packageRows ⊢
  omega

private theorem prefixApplicationRowCount_le
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    (prefixPlan relation geometry).rowCount +
        (applicationPlan fits geometry).rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  have complete := rowCount_le relation fits geometry
  omega

private theorem throughNextPreimageRowCount_le
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    ((prefixPlan relation geometry).rowCount +
        (applicationPlan fits geometry).rowCount) +
        (nextPreimagePlan geometry).rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  have complete := rowCount_le relation fits geometry
  omega

def prefixApplicationPlan
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (prefixPlan relation geometry)
    (applicationPlan fits geometry)
    (prefixApplicationRowCount_le relation fits geometry)

def throughNextPreimagePlan
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (prefixApplicationPlan relation fits geometry)
    (nextPreimagePlan geometry)
    (throughNextPreimageRowCount_le relation fits geometry)

def plan
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append
    (throughNextPreimagePlan relation fits geometry)
    (publicOutputPlan geometry) (rowCount_le relation fits geometry)

@[simp] theorem plan_rowCount
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    (plan relation fits geometry).rowCount =
      6369850 + (PerApplicationPackage.applicationPlan application).rowCount +
        9 := by
  simp [plan, throughNextPreimagePlan, prefixApplicationPlan, prefixPlan,
    applicationPlan, nextPreimagePlan, publicOutputPlan]

/-- The ordered plan through the application depends on relation shape only.
The application program and retained placement are verifier-owned inputs. -/
theorem plan_eq_of_same_shape
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (left right : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    plan left fits geometry = plan right fits geometry := by
  unfold plan throughNextPreimagePlan prefixApplicationPlan prefixPlan
  have prefixEq :=
    DirectPiRLCSamplerCompletePrefixPlan.plan_eq_of_same_shape left right
      (prefixGeometry geometry)
  cases prefixEq
  rfl

theorem rowsZero_iff
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) :
    (plan relation fits geometry).RowsZero assignment ↔
      (((prefixPlan relation geometry).RowsZero assignment ∧
        (applicationPlan fits geometry).RowsZero assignment) ∧
          (nextPreimagePlan geometry).RowsZero assignment) ∧
        (publicOutputPlan geometry).RowsZero assignment := by
  rw [plan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [throughNextPreimagePlan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [prefixApplicationPlan, ProductionRelation.Plan.append_rowsZero_iff]

/-- The application source is contained in the complete package source
prefix used by the canonical retained assignment. -/
theorem applicationSourceWidth_le_baseSourceWidth
    (application : Lifecycle.Stage1.Application.Program) :
    ApplicationRetainedBlocks.sourceWidth application ≤
      PiRLCProductPlan.baseSourceWidth application := by
  unfold ApplicationRetainedBlocks.sourceWidth
    ApplicationDirectSource.sourceWidth ApplicationPackage.r1csFreshStart
    Layout.Stage1.ApplicationInputs.localStart
    Layout.Stage1.ApplicationInputs.witnessStart
    PiRLCProductPlan.baseSourceWidth
  rw [PerApplicationPackage.package_totalColumnCount]
  unfold PerApplicationPackage.addedPrivateColumnCount
  have baseTotal : PerApplicationPackage.basePackage.layout.totalColumnCount =
      29336725 := by
    exact Package.circuitPackage_layout_values.2.2.2.2
  have privateCount : Layout.Stage1.Spartan.privateColumnCount = 29336446 := by
    exact Layout.Stage1.Spartan.privateColumnCount_eq
  rw [baseTotal, privateCount]
  change 29336446 + application.witnessWordCount +
        localLength (ApplicationPackage.operations application
          (ApplicationPackage.productionColumns application)
          (29336446 + application.witnessWordCount)) +
        R1CS.totalFreshCount
          (ApplicationPackage.constraints application
            (ApplicationPackage.productionColumns application)
            (29336446 + application.witnessWordCount)) ≤
      29336725 + (application.witnessWordCount +
        (PerApplicationPackage.applicationPlan application).privateCount)
  change _ ≤ 29336725 + (application.witnessWordCount +
    (localLength (ApplicationPackage.operations application
      (ApplicationPackage.productionColumns application)
      (29336446 + application.witnessWordCount)) +
    R1CS.totalFreshCount
      (ApplicationPackage.constraints application
        (ApplicationPackage.productionColumns application)
        (29336446 + application.witnessWordCount))))
  omega

/-- The application reads the same complete package source assignment as the
NIFS prefix. There is no independently selectable application source. -/
def applicationSource
    (application : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F) :
    Fin (ApplicationRetainedBlocks.sourceWidth application) → F :=
  fun column => base ⟨column.val,
    lt_of_lt_of_le column.isLt
      (applicationSourceWidth_le_baseSourceWidth application)⟩

/-- The application-prefix view is the total physical-base environment at
every application source column. -/
@[simp] theorem applicationSource_eq_sourceEnv
    (application : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (column : Fin (ApplicationRetainedBlocks.sourceWidth application)) :
    applicationSource application base column =
      SourceCompiler.sourceEnv base column.val := by
  unfold applicationSource SourceCompiler.sourceEnv
  rw [dif_pos (lt_of_lt_of_le column.isLt
    (applicationSourceWidth_le_baseSourceWidth application))]

structure Encodes
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) : Prop where
  runningPrefix : DirectPiRLCSamplerCompletePrefixPlan.Encodes
    (prefixGeometry geometry) assignment base groupValue products
  /-- Input/output facts refer to the existing pilot coordinates. Only the
  witness/local fields describe new application suffix coordinates. -/
  applicationEncoding : ApplicationRetainedGeometry.Encodes geometry assignment
    (applicationSource application base)

structure Semantics
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) : Prop where
  runningPrefix : DirectPiRLCSamplerCompletePrefixPlan.Semantics relation
    (prefixGeometry geometry) assignment base groupValue products
  applicationSemantics : Lifecycle.Stage1.Application.Holds application.step
    (Layout.Stage1.ApplicationInputs.interface application)
    (Layout.Stage1.ApplicationInputs.localStart application)
    (ApplicationDirectPlan.sourceEnv (applicationSource application base))
  nextPreimage : Lifecycle.Stage1.NextPreimage.SpecHolds
    Layout.Stage1.NextPreimageInputs.sourceInterface
    Layout.Stage1.RunningTransitionInputs.phaseOffset
    (Spartan.pullback (RunningTransitionDirectPlan.transitionEnv application base))
  publicOutput : RecursivePublicOutputPlan.Matches geometry assignment

private theorem prefixOne
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (ApplicationRetainedGeometry.oneColumn geometry) = 1) :
    assignment
      (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn
        (prefixGeometry geometry)) =
        1 := by
  exact one

theorem rowsZero_implies_semantics
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (ApplicationRetainedGeometry.oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment base groupValue products)
    (rowsZero : (plan relation fits geometry).RowsZero assignment) :
    Semantics relation geometry assignment base groupValue products := by
  have children := (rowsZero_iff relation fits geometry assignment).mp rowsZero
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact DirectPiRLCSamplerCompletePrefixPlan.rowsZero_implies_semantics relation
      (prefixGeometry geometry) assignment base groupValue products
      (prefixOne geometry assignment one) encodes.runningPrefix children.1.1.1
  · have rows := (ApplicationDirectPlan.rowsZero_iff_rowsHold fits geometry
      assignment (applicationSource application base)
      encodes.applicationEncoding one).mp children.1.1.2
    exact ApplicationDirectSource.rowsHold_implies_applicationHolds application
      (ApplicationDirectPlan.sourceEnv (applicationSource application base)) rows
  · exact NextPreimageDirectPlan.rowsZero_implies_spec
      (piCcsOrdinaryGeometry geometry) assignment base groupValue products
      encodes.runningPrefix.prior.pilotOrdinary.prior one children.1.2
  · exact (RecursivePublicOutputPlan.rowsZero_iff_matches geometry assignment
      one).mp children.2

end NightstreamFPrime.Export.Stage1.DirectApplicationPrefixPlan
