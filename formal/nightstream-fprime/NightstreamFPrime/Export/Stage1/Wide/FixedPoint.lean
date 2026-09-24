import NightstreamFPrime.Export.Stage1.Wide.Stage1Plan
import NightstreamFPrime.Lifecycle.PiRLC.Wide.Key

/-! Derive the candidate recursive relation from its own matrices and width.
The seed has no semantic authority. Reassembly with the derived relation
produces the same plan. This does not select the production package. -/

namespace NightstreamFPrime.Export.Stage1.Wide.FixedPoint

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation
open NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint

abbrev Program := RetainedLayout.Program

private theorem append_congr {columns : Nat} {a b c d : ProductionRelation.Plan columns}
    (left : a = c) (right : b = d) (fits : a.rowCount + b.rowCount ≤ 2 ^ Lifecycle.cubeVariables)
    (fits' : c.rowCount + d.rowCount ≤ 2 ^ Lifecycle.cubeVariables) :
    Plan.append a b fits = Plan.append c d fits' := by
  cases left
  cases right
  rfl

theorem plan_same_shape (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    {width : Nat} {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
    (left right : Lifecycle.ProductionKey.LogicalRelation width publicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 program) :
    Stage1Plan.plan program compiled left fits = Stage1Plan.plan program compiled right fits := by
  let geometry := Stage1Plan.piDecGeometry program
  have core : DirectPiDECPrefixPlan.piCcsCorePlan left geometry =
      DirectPiDECPrefixPlan.piCcsCorePlan right geometry := by
    apply append_congr
    · rfl
    · exact PiCCSOrdinaryDirectPlan.plan_eq_of_same_shape left right _
  have pilot : DirectPiDECPrefixPlan.pilotOrdinaryPrefixPlan left geometry =
      DirectPiDECPrefixPlan.pilotOrdinaryPrefixPlan right geometry := append_congr core rfl _ _
  have binding : DirectPiDECPrefixPlan.pilotBindingPrefixPlan left geometry =
      DirectPiDECPrefixPlan.pilotBindingPrefixPlan right geometry := append_congr pilot rfl _ _
  have complete : DirectPiDECPrefixPlan.piCcsCompletePlan left geometry =
      DirectPiDECPrefixPlan.piCcsCompletePlan right geometry := append_congr binding rfl _ _
  have prefixEq : Stage1Plan.prefixPlan program left = Stage1Plan.prefixPlan program right :=
    Stage1Plan.rename_congr program complete _ _
  have piDecEq : Stage1Plan.piDec program left = Stage1Plan.piDec program right :=
    Stage1Plan.rename_congr program (PiDECDirectPlan.plan_eq_of_same_shape left right geometry) _ _
  have throughPiRlc : Stage1Plan.throughPiRlc program compiled left =
      Stage1Plan.throughPiRlc program compiled right := append_congr prefixEq rfl _ _
  have throughPiDec : Stage1Plan.throughPiDec program compiled left =
      Stage1Plan.throughPiDec program compiled right := append_congr throughPiRlc piDecEq _ _
  have throughRunning : Stage1Plan.beforeApplication program compiled left =
      Stage1Plan.beforeApplication program compiled right := append_congr throughPiDec rfl _ _
  have throughApplication : Stage1Plan.throughApplication program compiled left fits =
      Stage1Plan.throughApplication program compiled right fits := append_congr throughRunning rfl _ _
  have throughNext : Stage1Plan.throughNextPreimage program compiled left fits =
      Stage1Plan.throughNextPreimage program compiled right fits := append_congr throughApplication rfl _ _
  exact append_congr throughNext rfl _ _

def publicFits (program : Program) :
    ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth (RetainedLayout.logicalWidth program) := by
  apply Nat.le_trans (m := RetainedLayout.logicalWidth program)
  · rw [RetainedLayout.logicalWidth_eq]
    change 270 ≤ _
    omega
  · exact Phi81CarrierLayout.logicalWidth_le_carrierWidth _

theorem carrierFits (program : Program) (fits : PerApplicationFixedPoint.FitsTwoPow28 program) :
    Phi81CarrierLayout.carrierWidth (RetainedLayout.logicalWidth program) ≤ 2 ^ Lifecycle.cubeVariables := by
  have widths : RetainedLayout.logicalWidth program ≤ PerApplicationFixedPoint.logicalWidth program := by
    rw [RetainedLayout.logicalWidth_eq, RetainedLayout.referenceWidth_eq]
    omega
  apply Nat.le_trans _ fits.carrier
  unfold Phi81CarrierLayout.carrierWidth Phi81ColumnLayout.blockCount
  norm_num only [ringDegree]
  omega

def seed (program : Program) (fits : PerApplicationFixedPoint.FitsTwoPow28 program) :
    Lifecycle.ProductionKey.LogicalRelation (RetainedLayout.logicalWidth program) (publicFits program) where
  matrices := fun _ _ _ => 0
  cubeFits := carrierFits program fits

def structuralPlan (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 program) :=
  Stage1Plan.plan program compiled (seed program fits) fits.package

def relation (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 program) :
    Lifecycle.ProductionKey.LogicalRelation (RetainedLayout.logicalWidth program) (publicFits program) :=
  (structuralPlan program compiled fits).logicalRelation (carrierFits program fits)

theorem plan_fixedPoint (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 program) :
    Stage1Plan.plan program compiled (relation program compiled fits) fits.package =
      structuralPlan program compiled fits :=
  plan_same_shape program compiled (relation program compiled fits) (seed program fits) fits.package

theorem relation_matrices (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 program) :
    (relation program compiled fits).matrices = (structuralPlan program compiled fits).matrix := rfl

noncomputable def key (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 program)
    (ajtai : AjtaiKey (logicalWidth := RetainedLayout.logicalWidth program) (publicFits := publicFits program)) :=
  Lifecycle.PiRLC.Wide.Key.key (relation program compiled fits) ajtai

end NightstreamFPrime.Export.Stage1.Wide.FixedPoint
