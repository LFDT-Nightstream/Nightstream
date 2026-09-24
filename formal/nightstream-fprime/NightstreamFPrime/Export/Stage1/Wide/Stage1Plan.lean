import NightstreamFPrime.Export.Stage1.Wide.RetainedLayout

/-! Assemble the candidate Stage 1 rows with the wide PiRLC allocation.
Unchanged phase plans use the proved coordinate renaming. Source coverage and
whole-package witness preservation remain separate obligations. -/

namespace NightstreamFPrime.Export.Stage1.Wide.Stage1Plan

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation
open NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev Program := Lifecycle.Stage1.Application.Program

def referenceGeometry (program : Program) := PerApplicationFixedPoint.geometry program
def piDecGeometry (program : Program) :=
  DirectApplicationPrefixPlan.piDecGeometry (referenceGeometry program)
def poseidonGeometry (program : Program) :=
  DirectPiDECPrefixPlan.poseidonGeometry (piDecGeometry program)
def piCcsGeometry (program : Program) :=
  DirectPiDECPrefixPlan.piCcsOrdinaryGeometry (piDecGeometry program)

def rename (program : Program) (plan : ProductionRelation.Plan (PerApplicationFixedPoint.logicalWidth program)) :
    ProductionRelation.Plan (RetainedLayout.logicalWidth program) :=
  plan.mapColumns (RetainedLayout.column program)

def initialState (program : Program) : PoseidonSboxPlan.State (RetainedLayout.logicalWidth program) :=
  fun lane => (PiRLCSamplerPoseidonPlan.piCcsFinalOutput (poseidonGeometry program) lane).mapColumns
    (RetainedLayout.column program)

def value (program : Program) (ring : PiRLCGeometry.RingIndex) :
    Phi81ProductPlan.State (RetainedLayout.logicalWidth program) :=
  fun lane => (PiRLCValueWiring.form (piCcsGeometry program)
    (PiRLCProductRingSchedule.laneInvocation ring lane)).mapColumns (RetainedLayout.column program)

def piRlcInterface (program : Program) : PiRLCGeometry.Interface (RetainedLayout.logicalWidth program) where
  oneColumn := RetainedLayout.column program (ApplicationRetainedGeometry.oneColumn (referenceGeometry program))
  initialState := initialState program
  value := value program
  start := RetainedLayout.commonCount program
  fits := Nat.le_refl _

variable {relationWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationWidth}

def prefixPlan (program : Program) (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits) :=
  rename program (DirectPiDECPrefixPlan.piCcsCompletePlan relation (piDecGeometry program))

def piRlc (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled) :=
  PiRLCGeometry.plan compiled (piRlcInterface program)

def piDec (program : Program) (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits) :=
  rename program (DirectPiDECPrefixPlan.piDecPlan relation (piDecGeometry program))

def running (program : Program) :=
  rename program (RunningTransitionReducedPlan.plan
    (DirectPiDECPrefixPlan.runningGeometry (piDecGeometry program)))

@[simp] theorem prefix_rows (program : Program)
    (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits) :
    (prefixPlan program relation).rowCount = 3054685 := by
  rw [prefixPlan, rename, Plan.mapColumns_rowCount, DirectPiDECPrefixPlan.piCcsCompletePlan_rowCount]

@[simp] theorem piRlc_rows (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled) :
    (piRlc program compiled).rowCount = 119153 := PiRLCGeometry.rowCount_eq _ _

@[simp] theorem piDec_rows (program : Program)
    (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits) :
    (piDec program relation).rowCount = 25488 := by
  rw [piDec, rename, Plan.mapColumns_rowCount, DirectPiDECPrefixPlan.piDecPlan, PiDECDirectPlan.plan_rowCount]
  rfl

@[simp] theorem running_rows (program : Program) : (running program).rowCount = 49359 := by
  rw [running, rename, Plan.mapColumns_rowCount, RunningTransitionReducedPlan.plan_rowCount]

def throughPiRlc (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits) :=
  Plan.append (prefixPlan program relation) (piRlc program compiled) (by
    rw [prefix_rows, piRlc_rows]; decide)

@[simp] theorem throughPiRlc_rows (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits) :
    (throughPiRlc program compiled relation).rowCount = 3173838 := by
  simp [throughPiRlc]

def throughPiDec (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits) :=
  Plan.append (throughPiRlc program compiled relation) (piDec program relation) (by
    rw [throughPiRlc_rows, piDec_rows]; decide)

@[simp] theorem throughPiDec_rows (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits) :
    (throughPiDec program compiled relation).rowCount = 3199326 := by
  simp [throughPiDec]

def beforeApplication (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits) :=
  Plan.append (throughPiDec program compiled relation) (running program) (by
    rw [throughPiDec_rows, running_rows]; decide)

@[simp] theorem beforeApplication_rows (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits) :
    (beforeApplication program compiled relation).rowCount = 3248685 := by
  simp [beforeApplication]

def application (program : Program) (fits : PerApplicationPackage.FitsTwoPow28 program) :=
  rename program (ApplicationDirectPlan.plan fits (referenceGeometry program))

def nextPreimage (program : Program) :=
  rename program (DirectApplicationPrefixPlan.nextPreimagePlan (referenceGeometry program))

def publicOutput (program : Program) :=
  rename program (DirectApplicationPrefixPlan.publicOutputPlan (referenceGeometry program))

theorem totalFits (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 program) :
    ((beforeApplication program compiled relation).rowCount + (application program fits).rowCount) +
      (nextPreimage program).rowCount + (publicOutput program).rowCount ≤ 2 ^ Lifecycle.cubeVariables := by
  have baseline := DirectApplicationPrefixPlan.rowCount_le relation fits (referenceGeometry program)
  simp only [DirectApplicationPrefixPlan.prefixPlan,
    DirectPiRLCSamplerCompletePrefixPlan.plan_rowCount,
    DirectApplicationPrefixPlan.applicationPlan, ApplicationDirectPlan.plan_rowCount,
    DirectApplicationPrefixPlan.nextPreimagePlan, NextPreimageDirectPlan.plan_rowCount,
    DirectApplicationPrefixPlan.publicOutputPlan, RecursivePublicOutputPlan.plan_rowCount] at baseline
  simp only [beforeApplication_rows, application, nextPreimage, publicOutput, rename,
    Plan.mapColumns_rowCount, ApplicationDirectPlan.plan_rowCount,
    DirectApplicationPrefixPlan.nextPreimagePlan, NextPreimageDirectPlan.plan_rowCount,
    DirectApplicationPrefixPlan.publicOutputPlan, RecursivePublicOutputPlan.plan_rowCount]
  omega

def throughApplication (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 program) :=
  Plan.append (beforeApplication program compiled relation) (application program fits)
    (by have bound := totalFits program compiled relation fits; omega)

def throughNextPreimage (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 program) :=
  Plan.append (throughApplication program compiled relation fits) (nextPreimage program)
    (by have bound := totalFits program compiled relation fits; change _ + _ + _ ≤ _; omega)

def plan (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 program) :=
  Plan.append (throughNextPreimage program compiled relation fits) (publicOutput program)
    (totalFits program compiled relation fits)

theorem plan_rows (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 program) :
    (plan program compiled relation fits).rowCount = 3248694 + ApplicationDirectPlan.rowCount program := by
  simp only [plan, throughNextPreimage, throughApplication, Plan.append_rowCount, beforeApplication_rows,
    application, nextPreimage, publicOutput, rename, DirectApplicationPrefixPlan.nextPreimagePlan,
    DirectApplicationPrefixPlan.publicOutputPlan, Plan.mapColumns_rowCount,
    ApplicationDirectPlan.plan_rowCount, NextPreimageDirectPlan.plan_rowCount,
    RecursivePublicOutputPlan.plan_rowCount]
  omega

theorem rows_iff (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 program)
    (assignment : Assignment F (RetainedLayout.logicalWidth program)) :
    (plan program compiled relation fits).RowsZero assignment ↔
      (prefixPlan program relation).RowsZero assignment ∧
      (piRlc program compiled).RowsZero assignment ∧
      (piDec program relation).RowsZero assignment ∧
      (running program).RowsZero assignment ∧
      (application program fits).RowsZero assignment ∧
      (nextPreimage program).RowsZero assignment ∧
      (publicOutput program).RowsZero assignment := by
  simp only [plan, throughNextPreimage, throughApplication, beforeApplication,
    throughPiDec, throughPiRlc, Plan.append_rowsZero_iff, and_assoc]

end NightstreamFPrime.Export.Stage1.Wide.Stage1Plan
