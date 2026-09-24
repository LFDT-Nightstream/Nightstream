import NightstreamFPrime.Export.Stage1.Wide.SourceAssignment
import NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupport

/-! Values used by the running transition in the wide and reference layouts.
PiCCS and state fields keep their addresses; PiDEC children move as one block. -/

namespace NightstreamFPrime.Export.Stage1.Wide.RunningSourceValues

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint
open Lifecycle.PiCCS.v1_1 Lifecycle.Stage1 Circuit.Quadratic

private abbrev Old := Layout.Stage1.RunningTransitionInputs.interface
private abbrev New := Layout.Stage1.Wide.RunningTransitionInputs.interface
private abbrev oldOffset := Layout.Stage1.RunningTransitionInputs.phaseOffset
private abbrev newOffset := Layout.Stage1.Wide.RunningTransitionInputs.phaseOffset

variable {width : Nat}
  {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}

theorem suffix (env : Env) (index : Nat) (bounded : index < 363476) :
    SourceAssignment.sourceEnv env (28421542 + index) = env (27496062 + index) := by
  rw [SourceAssignment.sourceEnv, SourceAssignment.source?_suffix]
  · simp only [Option.map_some, Option.getD_some]
    change env (27496062 + (28421542 + index - 28421542)) = _
    rw [Nat.add_sub_cancel_left]
  · change 28421542 ≤ 28421542 + index ∧ 28421542 + index < 28785018
    omega

private theorem shifted (env : Env) (before after : Nat)
    (same : before = after + 925480) (bounded : 27496062 ≤ after ∧ after < 27859538) :
    SourceAssignment.sourceEnv env before = env after := by
  have copied := suffix env (after - 27496062) (by omega)
  have oldEq : 28421542 + (after - 27496062) = before := by omega
  have newEq : 27496062 + (after - 27496062) = after := by omega
  rw [oldEq, newEq] at copied
  exact copied

theorem iteration (env : Env) :
    ((Old width fits).iteration oldOffset).eval (SourceAssignment.sourceEnv env) =
      ((New width fits).iteration newOffset).eval env :=
  SourceAssignment.sourceEnv_prefix env 28 (by decide)

theorem initial (env : Env) (index : RunningTransition.StateIndex) :
    ((Old width fits).initialState oldOffset index).eval (SourceAssignment.sourceEnv env) =
      ((New width fits).initialState newOffset index).eval env := by
  apply SourceAssignment.sourceEnv_prefix
  have bound : index.val < 4 := index.isLt
  change 30 + index.val < 19513117
  omega

theorem current (env : Env) (index : RunningTransition.StateIndex) :
    ((Old width fits).currentState oldOffset index).eval (SourceAssignment.sourceEnv env) =
      ((New width fits).currentState newOffset index).eval env := by
  apply SourceAssignment.sourceEnv_prefix
  have bound : index.val < 4 := index.isLt
  change 35 + index.val < 19513117
  omega

private theorem running_values_eq
    (left right : StatementAbsorption.RunningExpr width fits) (before after : Env)
    (point : ∀ index, (left.point index).eval before = (right.point index).eval after)
    (commitment : ∀ child row lane, (left.commitment child row lane).eval before =
      (right.commitment child row lane).eval after)
    (publicInput : ∀ child coordinate, (left.publicInput child coordinate).eval before =
      (right.publicInput child coordinate).eval after)
    (evalK : ∀ child coefficient, ((left.evaluation child).eval_K coefficient).eval before =
      ((right.evaluation child).eval_K coefficient).eval after)
    (evalA : ∀ child matrix coefficient, ((left.evaluation child).eval_A matrix coefficient).eval before =
      ((right.evaluation child).eval_A matrix coefficient).eval after) :
    StatementAbsorption.evalRunning left before = StatementAbsorption.evalRunning right after := by
  unfold StatementAbsorption.evalRunning
  congr 1
  · have same : (fun index => (left.point index).eval before) =
        (fun index => (right.point index).eval after) := funext point
    simp only [StatementAbsorption.evalPoint, same]
  · exact funext fun child => funext fun row => funext fun lane => commitment child row lane
  · exact funext fun child => funext fun coordinate => publicInput child coordinate
  · funext child
    unfold StatementAbsorption.evalEvaluation
    congr 1
    · exact funext (evalK child)
    · exact funext fun matrix => funext (evalA child matrix)

theorem recursive (env : Env) :
    StatementAbsorption.evalRunning ((Old width fits).recursive oldOffset) (SourceAssignment.sourceEnv env) =
      StatementAbsorption.evalRunning ((New width fits).recursive newOffset) env := by
  apply running_values_eq
  · intro coordinate
    change ((Layout.Stage1.RunningTransitionInputs.recursiveRunningExpr width fits).point coordinate).eval _ =
      ((Layout.Stage1.RunningTransitionInputs.recursiveRunningExpr width fits).point coordinate).eval _
    rw [Layout.Stage1.RunningTransitionInputs.recursivePoint_eq_direct]
    apply KExpr.eval_eq_of_agree_below _ SourceAssignment.prefixEnd
    · have bound : coordinate.val < 28 := coordinate.isLt
      change (15027676 + coordinate.val * 5328 + 4136 < 19513117) ∧
        (15027676 + coordinate.val * 5328 + 4728 < 19513117)
      omega
    · exact SourceAssignment.sourceEnv_prefix env
  · intro child row lane
    change SourceAssignment.sourceEnv env (28421542 + child.val * 1188 + row.val * 54 + lane.val) =
      env (27496062 + child.val * 1188 + row.val * 54 + lane.val)
    have hc : child.val < 16 := child.isLt
    have hr : row.val < 22 := row.isLt
    have hl : lane.val < 54 := lane.isLt
    exact shifted env _ _ (by omega) (by omega)
  · intro child coordinate
    change SourceAssignment.sourceEnv env (28466470 + child.val * 270 + coordinate.val) =
      env (27540990 + child.val * 270 + coordinate.val)
    have hc : child.val < 16 := child.isLt
    have hi : coordinate.val < 270 := coordinate.isLt
    exact shifted env _ _ (by omega) (by omega)
  · intro child coefficient
    change K.mk
        (SourceAssignment.sourceEnv env (28440550 + child.val * 108 + coefficient.val * 2))
        (SourceAssignment.sourceEnv env (28440550 + child.val * 108 + coefficient.val * 2 + 1)) =
      K.mk (env (27515070 + child.val * 108 + coefficient.val * 2))
        (env (27515070 + child.val * 108 + coefficient.val * 2 + 1))
    congr 1
    all_goals
      have hc : child.val < 16 := child.isLt
      have hi : coefficient.val < 54 := coefficient.isLt
      exact shifted env _ _ (by omega) (by omega)
  · intro child matrix coefficient
    change K.mk
        (SourceAssignment.sourceEnv env (28442278 + child.val * 1512 + matrix.val * 108 + coefficient.val * 2))
        (SourceAssignment.sourceEnv env (28442278 + child.val * 1512 + matrix.val * 108 + coefficient.val * 2 + 1)) =
      K.mk (env (27516798 + child.val * 1512 + matrix.val * 108 + coefficient.val * 2))
        (env (27516798 + child.val * 1512 + matrix.val * 108 + coefficient.val * 2 + 1))
    congr 1
    all_goals
      have hc : child.val < 16 := child.isLt
      have hm : matrix.val < 14 := matrix.isLt
      have hi : coefficient.val < 54 := coefficient.isLt
      exact shifted env _ _ (by omega) (by omega)

theorem output_word (env : Env) (index : RunningTransition.WordIndex) :
    (RunningTransition.runningWord ((Old width fits).output oldOffset) index).eval (SourceAssignment.sourceEnv env) =
      (RunningTransition.runningWord ((New width fits).output newOffset) index).eval env := by
  change (RunningTransition.runningWord (Layout.Stage1.RunningTransitionInputs.outputRunningExpr width fits) index).eval _ =
    (RunningTransition.runningWord (Layout.Stage1.RunningTransitionInputs.outputRunningExpr width fits) index).eval _
  apply Expr.eval_eq_of_agree_below _ Layout.PilotProduction.outputDigestStart
  · exact RunningTransition.runningWord_varsBelow _ _
      (Layout.Stage1.RunningTransitionInputs.outputRunningBelowOutputDigestStart width fits) index
  · intro source below
    apply SourceAssignment.sourceEnv_prefix
    change source < 19513117
    change source < 99056 at below
    omega

theorem recursive_word (env : Env) (index : RunningTransition.WordIndex) :
    (RunningTransition.runningWord ((Old width fits).recursive oldOffset) index).eval (SourceAssignment.sourceEnv env) =
      (RunningTransition.runningWord ((New width fits).recursive newOffset) index).eval env := by
  rw [RunningTransition.runningWord_eval, RunningTransition.runningWord_eval, recursive]

end NightstreamFPrime.Export.Stage1.Wide.RunningSourceValues
