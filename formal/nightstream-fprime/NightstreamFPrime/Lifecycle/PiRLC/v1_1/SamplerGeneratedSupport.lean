import NightstreamFPrime.Lifecycle.PiRLC.v1_1.PhaseTransport

/-!
Owns lower-support transport for PiRLC sampler-generated values. Every result
is derived from direct fresh-variable offsets; sampler semantics and rows stay
owned by the existing sampler modules.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec

namespace Sampler

/-- The 54 selector outputs are unchanged when environments agree from the
sampler start onward. -/
theorem outputCoefficients_eq_of_agree_from (offset : Nat) (left right : Env)
    (agrees : ∀ index, offset ≤ index → left index = right index) :
    outputCoefficients left offset = outputCoefficients right offset := by
  unfold outputCoefficients First54.evalOutput
  apply congrArg List.ofFn
  funext slot
  apply Expr.eval_eq_of_agree_satisfy _ (fun index => offset ≤ index)
    left right
  · simp only [First54.output, First54ValueStep.output, Expr.VarsSatisfy]
    unfold First54.valueOffset First54.positionOffset selectorOffset windowBase
    omega
  · exact agrees

/-- The eight final permutation lanes are unchanged when environments agree
from the sampler start onward. -/
theorem outputState_eq_of_agree_from (interface : Interface)
    (coordinate offset : Nat) (left right : Env)
    (agrees : ∀ index, offset ≤ index → left index = right index) :
    evalState left (outputState interface coordinate offset) =
      evalState right (outputState interface coordinate offset) := by
  unfold evalState
  apply congrArg List.ofFn
  funext lane
  apply Expr.eval_eq_of_agree_satisfy _ (fun index => offset ≤ index)
    left right
  · simp only [outputState, DigestWindow.output, Permutation.Owned.output,
      Permutation.scheduleOutput, Permutation.freshState, Expr.VarsSatisfy]
    unfold windowOffset windowBase DigestWindow.permutationOffset
    omega
  · exact agrees

/-- The centered 54-coordinate challenge is unchanged when environments
agree from the sampler start onward. -/
theorem evalOutputChallenge_eq_of_agree_from (offset : Nat)
    (left right : Env)
    (agrees : ∀ index, offset ≤ index → left index = right index) :
    evalOutputChallenge left offset = evalOutputChallenge right offset := by
  funext position
  unfold evalOutputChallenge outputChallenge outputWord
  rw [Expr.eval_sub, Expr.eval_sub]
  congr 1
  apply Expr.eval_eq_of_agree_satisfy _ (fun index => offset ≤ index)
    left right
  · simp only [First54.output, First54ValueStep.output, Expr.VarsSatisfy]
    unfold First54.valueOffset First54.positionOffset selectorOffset windowBase
    omega
  · exact agrees

/-- Centering the selector words preserves an equality that can cross both
an environment boundary and a sampler-offset boundary. -/
theorem evalOutputChallenge_eq_of_outputWord_eq
    (leftOffset rightOffset : Nat) (left right : Env)
    (wordsEq : ∀ position : Fin ringDegree,
      (outputWord leftOffset position).eval left =
        (outputWord rightOffset position).eval right) :
    evalOutputChallenge left leftOffset =
      evalOutputChallenge right rightOffset := by
  funext position
  unfold evalOutputChallenge outputChallenge
  rw [Expr.eval_sub, Expr.eval_sub, wordsEq position]
  change (outputWord rightOffset position).eval right - (2 : F) =
    (outputWord rightOffset position).eval right - (2 : F)
  rfl

end Sampler

namespace SamplerChain

private theorem sourceOffset_le (offset source : Nat) :
    offset ≤ sourceOffset offset source := by
  unfold sourceOffset
  omega

/-- Every symbolic chain state evaluates equally after the initial state when
all sampler-owned variables are preserved. -/
theorem evalStateAt_eq_of_initial_and_agree_from
    (interface : Interface) (offset : Nat) (left right : Env)
    (initialEq : evalInitialState interface offset left =
      evalInitialState interface offset right)
    (agrees : ∀ index, offset ≤ index → left index = right index) :
    ∀ count, evalStateAt interface offset left count =
      evalStateAt interface offset right count := by
  intro count
  cases count with
  | zero => exact initialEq
  | succ source =>
      apply Sampler.outputState_eq_of_agree_from
      intro index bounded
      exact agrees index (Nat.le_trans (sourceOffset_le offset source) bounded)

/-- All 17 centered challenges are unchanged when environments agree on the
sampler-chain suffix. -/
theorem evalChallenges_eq_of_agree_from
    (interface : Interface) (offset : Nat) (left right : Env)
    (agrees : ∀ index, offset ≤ index → left index = right index) :
    evalChallenges interface offset left = evalChallenges interface offset right := by
  funext source
  exact Sampler.evalOutputChallenge_eq_of_agree_from
    (sourceOffset offset source.val) left right (fun index bounded =>
      agrees index (Nat.le_trans (sourceOffset_le offset source.val) bounded))

/-- The complete challenge vector is equal when every centered source word
is evaluated from equal selector outputs. -/
theorem evalChallenges_eq_of_outputWord_eq
    (leftInterface rightInterface : Interface)
    (leftOffset rightOffset : Nat) (left right : Env)
    (wordsEq : ∀ source : Fin sourceCount, ∀ position : Fin ringDegree,
      (Sampler.outputWord (sourceOffset leftOffset source.val) position).eval left =
        (Sampler.outputWord (sourceOffset rightOffset source.val) position).eval
          right) :
    evalChallenges leftInterface leftOffset left =
      evalChallenges rightInterface rightOffset right := by
  funext source
  exact Sampler.evalOutputChallenge_eq_of_outputWord_eq
    (sourceOffset leftOffset source.val)
    (sourceOffset rightOffset source.val) left right (wordsEq source)

/-- The complete sampler-chain relation is stable under equal initial state
and preservation of the chain-owned suffix. -/
theorem relationHolds_of_initial_and_agree_from
    (interface : Interface) (offset : Nat) (left right : Env)
    (initialEq : evalInitialState interface offset left =
      evalInitialState interface offset right)
    (agrees : ∀ index, offset ≤ index → left index = right index)
    (relation : RelationHolds interface offset left) :
    RelationHolds interface offset right := by
  let stateEq := evalStateAt_eq_of_initial_and_agree_from interface offset
    left right initialEq agrees
  apply RelationHolds.of_eval_eq interface offset left right
  · intro count _bounded
    exact stateEq count
  · intro source
    apply Sampler.outputCoefficients_eq_of_agree_from
    intro index bounded
    exact agrees index
      (Nat.le_trans (sourceOffset_le offset source.val) bounded)
  · intro source
    apply Sampler.outputState_eq_of_agree_from
    intro index bounded
    exact agrees index
      (Nat.le_trans (sourceOffset_le offset source.val) bounded)
  · exact evalChallenges_eq_of_agree_from interface offset left right agrees
  · exact relation

end SamplerChain

end NightstreamFPrime.Lifecycle.PiRLC.v1_1
