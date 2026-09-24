import NightstreamFPrime.Export.Stage1.Wide.SourceAssignment
import NightstreamFPrime.Export.Stage1.NextPreimageCompleteness
import NightstreamFPrime.Export.Stage1.CanonicalPublicOutput

/-! The next-state framing and public digest use the same copied state values
as the pilot. The direct PiRLC constructor preserves all these coordinates. -/

namespace NightstreamFPrime.Export.Stage1.Wide.FinalBindings

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.Stage1

private theorem prior_source (env : Env) (source : Nat) (before : source < SourceAssignment.prefixEnd) :
    Layout.Stage1.Spartan.pullback (SourceAssignment.targetEnv env) source = env source := by
  change SourceAssignment.targetEnv env (Layout.Stage1.Spartan.sourceToSpartan source) = env source
  rw [SourceAssignment.targetEnv_source env source (by
    change source < 19513117 at before
    rw [Layout.Stage1.Spartan.sourceColumnCount_eq]; omega)]
  exact SourceAssignment.sourceEnv_prefix env source before

theorem nextPreimage (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (specification : NextPreimage.SpecHolds Layout.Stage1.NextPreimageInputs.sourceInterface
      Layout.Stage1.Wide.RunningTransitionInputs.phaseOffset env) :
    (Stage1Plan.nextPreimage program).RowsZero (SourceAssignment.assignment program env application) := by
  have view : NextPreimage.SpecHolds Layout.Stage1.NextPreimageInputs.sourceInterface
      Layout.Stage1.RunningTransitionInputs.phaseOffset
      (Layout.Stage1.Spartan.pullback (SourceAssignment.targetEnv env)) := by
    apply NextPreimage.SpecHolds.of_cross_values_eq _ _ _ _ env _ _ _ _ _ specification
    · exact (prior_source env 28 (by decide)).symm
    · exact (prior_source env 49691 (by decide)).symm
    · intro index
      exact (prior_source env (30 + index.val) (by
        have bound : index.val < 4 := index.isLt
        change 30 + index.val < 19513117; omega)).symm
    · intro index
      exact (prior_source env (49693 + index.val) (by
        have bound : index.val < 4 := index.isLt
        change 49693 + index.val < 19513117; omega)).symm
  apply (AssignmentProjection.common_rowsZero_iff program (SourceAssignment.raw program env application).assignment
    _ (ReadSupport.nextPreimage program _) _).mpr
  exact NextPreimageCompleteness.rowsZero_of_base program (SourceAssignment.targetEnv env) application
    (SourceAssignment.raw program env application) rfl view

theorem publicOutput (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F) :
    (Stage1Plan.publicOutput program).RowsZero (SourceAssignment.assignment program env application) := by
  exact (AssignmentProjection.common_rowsZero_iff program (SourceAssignment.raw program env application).assignment
    _ (ReadSupport.public_output program _) _).mpr
      (CanonicalPublicOutput.rowsZero (SourceAssignment.raw program env application))

end NightstreamFPrime.Export.Stage1.Wide.FinalBindings
