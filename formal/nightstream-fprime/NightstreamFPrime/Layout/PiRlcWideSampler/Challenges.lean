import NightstreamFPrime.Layout.PiRlcWideSampler.Completeness

/-! Direct centered challenge forms for the ring-product verifier. Three
checked bits and a constant replace each independent 41-coordinate field. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.Challenges

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling
open ProductionRelation BatchPlan BatchSemantics
open Spec.Folding.PiCCS.PaperJoint
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def combination (position : Fin 54) : R1CS.LinearCombination :=
  ⟨-2, [(1803 + 3 * position.val, 1), (1804 + 3 * position.val, 2), (1805 + 3 * position.val, 4)]⟩

theorem bounded (position : Fin 54) : SourceCompiler.CombinationBounded 2025 (combination position) := by
  intro term member
  simp only [combination, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl <;> dsimp only <;> omega

theorem combination_eval (position : Fin 54) (env : Env) :
    (combination position).eval env = (WideReduction.Program.outputChallenge 4 position).eval env := by
  have core : WideReduction.Program.coreOffset 4 = 1408 := rfl
  have digits : WideReduction.digitStart 1408 = 1803 := rfl
  simp only [combination, R1CS.LinearCombination.eval, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, WideReduction.Program.outputChallenge, Expr.eval_sub,
    WideReduction.Program.outputWord, WideReduction.linearExpr, WideReduction.digitBit, core, digits,
    WideReduction.digitBitCount]
  change (-2 : F) + (1 * env (1803 + 3 * position.val) +
    (2 * env (1804 + 3 * position.val) + (4 * env (1805 + 3 * position.val) + 0))) =
      (1 * env (1803 + 3 * position.val + 0) +
        (2 * env (1803 + 3 * position.val + 1) + (4 * env (1803 + 3 * position.val + 2) + 0))) - 2
  rw [show 1803 + 3 * position.val + 1 = 1804 + 3 * position.val by omega,
    show 1803 + 3 * position.val + 2 = 1805 + 3 * position.val by omega]
  simp only [Nat.add_zero]
  rw [sub_eq_add_neg]
  exact add_comm _ _

def form {columns : Nat} (interface : Interface columns) (source : Fin 17) (position : Fin 54) : SparseForm columns :=
  SourceCompiler.compileCombination (rangeSource interface source) interface.oneColumn
    (combination position) (bounded position)

theorem form_eval {columns : Nat} (interface : Interface columns) (assignment : Assignment F columns)
    (one : assignment interface.oneColumn = 1) (source : Fin 17) (position : Fin 54) :
    (form interface source position).eval assignment =
      (WideReduction.Program.outputChallenge 4 position).eval (rangeEnv interface assignment source) := by
  rw [form, SourceCompiler.compileCombination_eval _ _ _ _ _ _ one (rangeSource_preserves interface assignment source),
    combination_eval]

theorem exact_challenge {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns)
    (assignment : Assignment F columns) (one : assignment interface.oneColumn = 1)
    (rows : (plan compiled interface).RowsZero assignment) (source : Fin 17) :
    (fun position => (form interface source position).eval assignment) =
      Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.challengeAt
        (StateSemantics.state assignment interface.initialState) source.val := by
  funext position
  rw [form_eval interface assignment one source position]
  change (WideReduction.Program.outputWord _ _ - 2).eval _ = _
  rw [Expr.eval_sub, WideReduction.Program.outputWord_eval]
  change WideReduction.fieldOfNat (WideReduction.digitValue _ 1408 position.val) - 2 = _
  rw [StateSemantics.sampled_digits compiled interface assignment one rows source position]
  exact Lifecycle.PiRLC.Wide.Batch.centered_digit _

end NightstreamFPrime.Layout.PiRlcWideSampler.Challenges
