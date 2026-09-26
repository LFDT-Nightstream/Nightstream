import NightstreamFPrime.Export.Stage1.Wide.DecodedPrefix
import NightstreamFPrime.Export.Stage1.Wide.PiDECOutput
import NightstreamFPrime.Export.Stage1.PiDECFormSemantics
import NightstreamFPrime.Export.Stage1.PiDECMatrixProgramSubstitution
import NightstreamFPrime.Layout.Stage1.PiDECInputBounds

/-! Decode PiDEC from the candidate's retained forms. Locations outside the
PiDEC packet keep the PiCCS view, including its evaluation point. The phase
proof applies to arbitrary assignments, without a retained encoding premise. -/

namespace NightstreamFPrime.Export.Stage1.Wide.DecodedPiDEC

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open ProductionRelation Layout.Stage1

def targetEnv (program : RetainedLayout.Program)
    (assignment : Assignment F (RetainedLayout.logicalWidth program)) : Env := fun column =>
  match PiDECDirectPlan.classifyTarget column with
  | some decoded => (decoded.location.form (Stage1Plan.piDecGeometry program)).eval
      (AssignmentPullback.assignment program assignment)
  | none => PiCCSAssignmentSoundness.decodedEnv (Stage1Plan.piCcsGeometry program)
      (AssignmentPullback.assignment program assignment) column

def env (program : RetainedLayout.Program)
    (assignment : Assignment F (RetainedLayout.logicalWidth program)) : Env :=
  Spartan.pullback (targetEnv program assignment)

theorem before_parent (program : RetainedLayout.Program)
    (assignment : Assignment F (RetainedLayout.logicalWidth program))
    (column : Nat) (before : column < PiDECSourceSupport.parentCommitmentStart) :
    env program assignment column = DecodedPrefix.piCcsEnv program assignment column := by
  have bounded : column < Spartan.SourceColumnCount := lt_of_lt_of_le before (by decide)
  have absent : PiDECDirectPlan.classifySource column = none := by
    cases found : PiDECDirectPlan.classifySource column with
    | none => rfl
    | some located =>
      have lower := PiDECSourceSupport.parentStart_le_source located.location.sourceSupport
      rw [located.owns] at lower
      omega
  simp only [env, Spartan.pullback, targetEnv, PiDECDirectPlan.classifyTarget,
    Spartan.spartanToSource_sourceToSpartan column bounded, absent]
  rfl

theorem form_eval (program : RetainedLayout.Program)
    (assignment : Assignment F (RetainedLayout.logicalWidth program))
    (column : Fin Spartan.spartanColumnCount) (supported : PiDECSourceSupport.Target column.val) :
    ((PiDECDirectPlan.sourceMap (Stage1Plan.piDecGeometry program)).form column).eval
      (AssignmentPullback.assignment program assignment) = targetEnv program assignment column.val := by
  obtain ⟨decoded, found, _⟩ := PiDECDirectPlan.classifyTarget_complete supported
  simp only [PiDECDirectPlan.sourceMap, targetEnv, found]

theorem location (program : RetainedLayout.Program)
    (assignment : Assignment F (RetainedLayout.logicalWidth program))
    (value : PiDECDirectPlan.Location) :
    env program assignment value.sourceColumn =
      (value.form (Stage1Plan.piDecGeometry program)).eval (AssignmentPullback.assignment program assignment) := by
  let column : Fin Spartan.spartanColumnCount :=
    ⟨Spartan.sourceToSpartan value.sourceColumn, Spartan.sourceToSpartan_lt _ value.sourceColumn_lt⟩
  have support : PiDECSourceSupport.Target column.val := ⟨value.sourceColumn, value.sourceSupport, rfl⟩
  have mapped := PiDECMatrixProgram.substitution_agrees_on_target (Stage1Plan.piDecGeometry program) column support
  have selected := PiDECMatrixProgram.substitution_location_form? (Stage1Plan.piDecGeometry program) value
  have same := Option.some.inj (mapped.symm.trans selected)
  exact (form_eval program assignment column support).symm.trans (congrArg (fun form =>
    form.eval (AssignmentPullback.assignment program assignment)) same)

theorem physical {width : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
    (program : RetainedLayout.Program) (relation : ProductionKey.LogicalRelation width publicFits)
    (assignment : Assignment F (RetainedLayout.logicalWidth program))
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (accepted : (Stage1Plan.piDec program relation).RowsZero assignment) :
    R1CS.RowsHold (env program assignment) (PiDECArithmetic.canonicalLayoutPlan relation).rows := by
  have referenceRows := (AssignmentPullback.rowsZero_iff program assignment _ _).mp accepted
  have rows := (PiDECFormSemantics.rowsZero_iff relation (Stage1Plan.piDecGeometry program)
    (AssignmentPullback.assignment program assignment) (targetEnv program assignment)
    (DecodedPrefix.reference_one program assignment one) (form_eval program assignment)).mp referenceRows
  rw [PiDECOrdinaryDirectSource.sourceRows_eq_canonical] at rows
  have exactRows := PiDECArithmetic.Plan.rows_to_layout
    (PiDECArithmetic.canonicalPlan width publicFits) (PiDECArithmetic.canonicalLayoutPlan relation)
    (PiDECArithmetic.canonicalPlan_matches relation)
  have remapped : R1CS.RowsHold (targetEnv program assignment)
      (Spartan.remapRows (PiDECArithmetic.canonicalLayoutPlan relation).rows) := by
    rw [← exactRows]
    exact rows
  have physical := (Spartan.remapRows_hold (targetEnv program assignment)
    (PiDECArithmetic.canonicalLayoutPlan relation).rows).mp remapped
  exact physical

theorem phase {width : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
    (program : RetainedLayout.Program) (relation : ProductionKey.LogicalRelation width publicFits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := publicFits))
    (assignment : Assignment F (RetainedLayout.logicalWidth program))
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (accepted : (Stage1Plan.piDec program relation).RowsZero assignment) :
    PiDEC.v1_1.Semantics.PhaseHolds relation ajtai (PiDECArithmetic.phaseInterface width publicFits)
      PiDECInputs.phaseOffset (env program assignment) := by
  exact Layout.PiDEC.v1_1.physical_implies_phaseHolds relation ajtai
    (PiDECArithmetic.phaseInterface width publicFits) PiDECInputs.phaseOffset (env program assignment)
    (PiDECInputs.assumptions relation (env program assignment))
    (physical program relation assignment one accepted)

/-- Each decoded PiDEC parent coefficient is forced to be the corresponding
ordered wide fold, including both extension-field cells. -/
theorem parent (program : RetainedLayout.Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (assignment : Assignment F (RetainedLayout.logicalWidth program))
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (rows : (Stage1Plan.piRlc program compiled).RowsZero assignment)
    (family : PiRLCOutput.Family) (block : Fin family.blockCount) (cell : Fin family.cellCount)
    (lane : Fin ringDegree) :
    env program assignment (PiDECOutput.parentView family block cell lane).sourceColumn =
      PiRLCOutput.ordered (PiRLCWitness.initial (Stage1Plan.piRlcInterface program) assignment)
        (PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program) assignment) family block cell lane := by
  rw [location]
  rw [← AssignmentPullback.form_eval program assignment _ (ReadSupport.piDec_location program _ _)]
  rw [PiDECOutput.parent_form]
  exact congrFun (PiRLCOutput.soundness compiled (Stage1Plan.piRlcInterface program)
    assignment one rows family block cell) lane

end NightstreamFPrime.Export.Stage1.Wide.DecodedPiDEC
