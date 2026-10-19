import NightstreamFPrime.Lifecycle.PiRLC.v1_2.PhaseTransport

/-! Sampler-owned outputs are stable when their generated coordinates agree.
The lemmas inspect only direct output expressions and preserve the same total
sampler specification. -/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_2

open NightstreamFPrime.Circuit NightstreamFPrime.Gadgets.Poseidon2 NightstreamFPrime.Spec

namespace Sampler

theorem outputWord_eq_of_agree_from (offset : Nat) (left right : Env)
    (agrees : ∀ index, offset ≤ index → left index = right index) (position : Fin ringDegree) :
    (outputWord offset position).eval left = (outputWord offset position).eval right := by
  change left (wordsOffset offset + position.val) = right (wordsOffset offset + position.val)
  apply agrees
  unfold wordsOffset advanceOffset rangeOffset
  omega

theorem outputState_eq_of_agree_from (interface : Interface)
    (coordinate offset : Nat) (left right : Env)
    (agrees : ∀ index, offset ≤ index → left index = right index) :
    evalState left (outputState interface coordinate offset) =
      evalState right (outputState interface coordinate offset) := by
  apply congrArg List.ofFn
  funext lane
  change left (advanceOffset offset + 1080 + lane.val) = right (advanceOffset offset + 1080 + lane.val)
  apply agrees
  unfold advanceOffset rangeOffset
  omega

end Sampler

namespace SamplerChain

def evalStateAt (interface : Interface) (offset : Nat) (env : Env) (count : Nat) : Poseidon2.State :=
  Sampler.evalState env (stateAtExpr interface offset count)

def evalChallenges (offset : Nat) (env : Env) : Fin sourceCount → RingF :=
  fun source position => (outputChallenge offset source position).eval env

private theorem sourceOffset_le (offset source : Nat) : offset ≤ sourceOffset offset source := by
  unfold sourceOffset
  omega

theorem evalStateAt_eq_of_initial_and_agree_from
    (interface : Interface) (offset : Nat) (left right : Env)
    (initialEq : evalInitialState interface offset left = evalInitialState interface offset right)
    (agrees : ∀ index, offset ≤ index → left index = right index) :
    ∀ count, evalStateAt interface offset left count = evalStateAt interface offset right count := by
  intro count
  cases count with
  | zero => exact initialEq
  | succ source =>
      apply Sampler.outputState_eq_of_agree_from
      intro index bounded
      exact agrees index (Nat.le_trans (sourceOffset_le offset source) bounded)

theorem evalChallenges_eq_of_outputWord_eq
    (leftOffset rightOffset : Nat) (left right : Env)
    (wordsEq : ∀ source : Fin sourceCount, ∀ position : Fin ringDegree,
      (Sampler.outputWord (sourceOffset leftOffset source.val) position).eval left =
        (Sampler.outputWord (sourceOffset rightOffset source.val) position).eval right) :
    evalChallenges leftOffset left = evalChallenges rightOffset right := by
  funext source position
  simp only [evalChallenges, outputChallenge, Sampler.outputChallenge, Expr.eval_sub, wordsEq]
  rfl

theorem evalChallenges_eq_of_agree_from (offset : Nat) (left right : Env)
    (agrees : ∀ index, offset ≤ index → left index = right index) :
    evalChallenges offset left = evalChallenges offset right := by
  apply evalChallenges_eq_of_outputWord_eq
  intro source position
  exact Sampler.outputWord_eq_of_agree_from _ left right
    (fun index bounded => agrees index (Nat.le_trans (sourceOffset_le offset source.val) bounded)) position

theorem specHolds_of_initial_and_agree_from
    (interface : Interface) (offset : Nat) (left right : Env)
    (initialEq : evalInitialState interface offset left = evalInitialState interface offset right)
    (agrees : ∀ index, offset ≤ index → left index = right index)
    (specification : SpecHolds interface offset left) : SpecHolds interface offset right := by
  apply SpecHolds.of_eval_eq interface offset left right initialEq _ _ specification
  · intro source position
    exact Sampler.outputWord_eq_of_agree_from _ left right
      (fun index bounded => agrees index (Nat.le_trans (sourceOffset_le offset source.val) bounded)) position
  · exact evalStateAt_eq_of_initial_and_agree_from interface offset left right initialEq agrees sourceCount

end SamplerChain

end NightstreamFPrime.Lifecycle.PiRLC.v1_2
