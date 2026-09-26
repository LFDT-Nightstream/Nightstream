import NightstreamFPrime.Export.Stage1.Wide.PilotPoseidonSupport

namespace NightstreamFPrime.Export.Stage1.Wide.ReadSupport

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation FormSupport

theorem application_block (program : Program) {sourceWidth : Nat} (retained : LowNormBlock.Block sourceWidth)
    (start : Nat) (fits : start + retained.coordinateCount ≤ PerApplicationFixedPoint.logicalWidth program)
    (slot : Fin retained.slotCount) (lower : RetainedLayout.applicationStart program ≤ start) :
    CopiedForm program (retained.form start fits slot) := by
  apply block
  intro column low high
  have width := RetainedLayout.referenceWidth_eq program
  have appStart := (RetainedLayout.boundaries program).2.2.2
  exact Or.inr (Or.inr ⟨by omega, by omega⟩)

theorem application_input (program : Program)
    (geometry : ApplicationOrdinaryGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (lane : Lifecycle.Stage1.Application.StateIndex) : CopiedForm program (ApplicationDirectPlan.inputForm geometry lane) := by
  rw [ApplicationDirectPlan.inputForm_eq_pilot]
  exact common_copied program _ (prior_word program (ApplicationOrdinaryGeometry.pilotGeometry geometry) _)

theorem application_output (program : Program)
    (geometry : ApplicationOrdinaryGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (lane : Lifecycle.Stage1.Application.StateIndex) : CopiedForm program (ApplicationDirectPlan.outputForm geometry lane) := by
  rw [ApplicationDirectPlan.outputForm_eq_pilot]
  exact common_copied program _ (output_word program (ApplicationOrdinaryGeometry.pilotGeometry geometry) _)

theorem application_location (program : Program)
    (geometry : ApplicationOrdinaryGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (value : ApplicationOrdinaryPlan.Location program) :
    CopiedForm program (value.form geometry) := by
  cases value <;> dsimp only [ApplicationOrdinaryPlan.Location.form]
  case input index => exact application_input program geometry index
  case output index => exact application_output program geometry index
  case witness index => exact application_block program _ _ _ _ (Nat.le_refl _)
  case localValues index =>
    apply application_block
    change RetainedLayout.applicationStart program ≤ RetainedLayout.applicationStart program + _
    omega

theorem application (program : Program) (fits : PerApplicationPackage.FitsTwoPow28 program)
    (geometry : ApplicationOrdinaryGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program)) :
    CopiedPlans program (ApplicationDirectPlan.plan fits geometry) := by
  unfold ApplicationDirectPlan.plan
  apply source_plan
  · left
    change 0 < RetainedLayout.hashEnd program
    rw [(RetainedLayout.boundaries program).1]
    decide
  · intro row column
    dsimp only [ApplicationOrdinaryPlan.inputs, ApplicationOrdinaryPlan.sourceMap]
    split
    · exact empty _
    · exact application_location program geometry _

end NightstreamFPrime.Export.Stage1.Wide.ReadSupport
