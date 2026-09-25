import NightstreamFPrime.Export.Stage1.Wide.SourceAssignment
import NightstreamFPrime.Export.Stage1.Wide.PiDECOutput

/-! Connect PiDEC retained forms to the wide physical source and direct fold.
Proof/split/scratch fields are copied; parent views read the direct outputs. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiDECWitnessInputs

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev Program := RetainedLayout.Program

theorem reference_value (program : Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (location : PiDECDirectPlan.Location) :
    (location.form (Stage1Plan.piDecGeometry program)).eval (SourceAssignment.raw program env application).assignment =
      PiDECSource.value env location := by
  let raw := SourceAssignment.raw program env application
  have encoded := (PerApplicationCanonicalEncodes.samplerPrefixEncodes raw).prior.piDec
  rw [PiDECDirectPlan.Location.form_eval _ _ _ encoded location]
  rw [RunningTransitionDirectPlan.sourceAssignment_packageSource program raw.base raw.groupValue raw.products
    location.sourceColumn location.sourceColumn_lt]
  have copied := PerApplicationSourceAssignment.packageEnv_ofCompleted program
    (SourceAssignment.targetEnv env) application (Layout.Stage1.Spartan.sourceToSpartan location.sourceColumn)
    (by
      have upper := Layout.Stage1.Spartan.sourceToSpartan_lt _ location.sourceColumn_lt
      simpa only [PerApplicationPackage.basePackage_totalColumnCount_eq,
        Layout.Stage1.Spartan.spartanColumnCount_eq] using! upper)
  exact copied.trans ((SourceAssignment.targetEnv_source env location.sourceColumn location.sourceColumn_lt).trans
    (SourceAssignment.sourceEnv_piDec env location))

def Local : PiDECDirectPlan.Location → Prop
  | .proof _ | .logical _ | .fresh _ => True
  | _ => False

private theorem shared_common (program : Program) {sourceWidth : Nat} (retained : LowNormBlock.Block sourceWidth)
    (start : Nat) (fits : start + retained.coordinateCount ≤ PerApplicationFixedPoint.logicalWidth program)
    (slot : Fin retained.slotCount) (lower : RetainedLayout.sharedStart program ≤ start)
    (upper : start + retained.coordinateCount ≤ RetainedLayout.sharedEnd program) :
    FormSupport.Supported (FormSupport.Common program) (retained.form start fits slot) := by
  apply FormSupport.block
  intro column low high
  exact Or.inr ⟨by omega, by omega⟩

theorem local_common (program : Program) (location : PiDECDirectPlan.Location) (owned : Local location) :
    FormSupport.Supported (FormSupport.Common program) (location.form (Stage1Plan.piDecGeometry program)) := by
  let within : PiDECRetainedGeometry.Geometry program (RetainedLayout.sharedEnd program) := ⟨Nat.le_refl _⟩
  cases location <;> dsimp only [Local] at owned
  case proof index =>
    apply shared_common
    · change RetainedLayout.sharedStart program ≤ RetainedLayout.sharedStart program + _ + _
      omega
    · exact PiDECRetainedGeometry.proofFits within
  case logical index =>
    apply shared_common
    · rw [(RetainedLayout.boundaries program).2.1]
      unfold PiDECRetainedGeometry.logicalStart PiDECRetainedGeometry.prefixLogicalWidth
      rw [PilotOrdinaryRetainedGeometry.completeLogicalWidth_eq]
      decide
    · exact PiDECRetainedGeometry.logicalFits within
  case fresh index =>
    apply shared_common
    · rw [(RetainedLayout.boundaries program).2.1]
      unfold PiDECRetainedGeometry.freshStart PiDECRetainedGeometry.logicalStart PiDECRetainedGeometry.prefixLogicalWidth
      rw [PilotOrdinaryRetainedGeometry.completeLogicalWidth_eq]
      omega
    · exact PiDECRetainedGeometry.freshFits within

/-- Direct PiRLC completion cannot change any PiDEC proof or split coordinate. -/
theorem local_value (program : Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (location : PiDECDirectPlan.Location) (owned : Local location) :
    (RetainedLayout.renameForm program (location.form (Stage1Plan.piDecGeometry program))
      (ReadSupport.piDec_location program _ location)).eval (SourceAssignment.assignment program env application) =
      PiDECSource.value env location := by
  let raw := SourceAssignment.raw program env application
  change (RetainedLayout.renameForm program _ _).eval (AssignmentProjection.assignment program raw.assignment) = _
  rw [AssignmentProjection.assignment_eq_project]
  exact (Stage1Witness.common_form_unchanged program (AssignmentProjection.project program raw.assignment)
    _ (local_common program location owned)).trans
      ((AssignmentProjection.project_form program raw.assignment _ (ReadSupport.piDec_location program _ location)).trans
        (reference_value program env application location))

/-- All four parent families read the direct ordered fold, including the two
extension cells. They do not read copied reference product coordinates. -/
theorem parent_value (program : Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (family : PiRLCOutput.Family) (block : Fin family.blockCount) (cell : Fin family.cellCount)
    (lane : Fin ringDegree) :
    (RetainedLayout.renameForm program
      ((PiDECOutput.parentView family block cell lane).form (Stage1Plan.piDecGeometry program))
      (ReadSupport.piDec_location program _ _)).eval (SourceAssignment.assignment program env application) =
      PiRLCOutput.ordered
        (PiRLCWitness.initial (Stage1Plan.piRlcInterface program)
          (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment))
        (PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program)
          (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment))
        family block cell lane := by
  rw [PiDECOutput.parent_form]
  exact congrFun (PiRLCOutput.witness_output _ _ family block cell) lane

end NightstreamFPrime.Export.Stage1.Wide.PiDECWitnessInputs
