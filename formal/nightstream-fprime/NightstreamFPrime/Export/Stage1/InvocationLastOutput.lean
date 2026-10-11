import NightstreamFPrime.Export.Stage1.Invocations

/-!
Owns the structural last-output theorem for a nonempty Duplex invocation
schedule. The proof follows action and absorb-block structure. It does not
evaluate a concrete production action list.
-/

namespace NightstreamFPrime.Export.Stage1.InvocationLastOutput

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Stage1.Invocations
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex

private theorem invocationCount_cons (action : Formal.Action)
    (actions : List Formal.Action) :
    invocationCount (action :: actions) =
      Action.invocationCount action + invocationCount actions := by
  simp [invocationCount]

private theorem lastOffset_cons (witnessStart headCount tailCount : Nat)
    (tailPositive : 0 < tailCount) :
    witnessStart + headCount * 1096 + (tailCount - 1) * 1096 =
      witnessStart + (headCount + tailCount - 1) * 1096 := by
  omega

theorem compileBlocks_state_last (phase rowStart witnessStart : Nat)
    (state : EState) (blocks : List (List Expr)) (nonempty : blocks ≠ []) :
    (compileBlocks phase rowStart witnessStart state blocks).state =
      permutationOutput
        (witnessStart + (blocks.length - 1) * 1096) := by
  induction blocks generalizing rowStart witnessStart state with
  | nil => exact False.elim (nonempty rfl)
  | cons block blocks inductionHypothesis =>
      cases blocks with
      | nil =>
          change permutationOutput witnessStart = _
          apply congrArg permutationOutput
          simp
      | cons next rest =>
          have tail := inductionHypothesis (rowStart + 1096)
            (witnessStart + 1096) (permutationOutput witnessStart) (by simp)
          calc
            (compileBlocks phase rowStart witnessStart state
                (block :: next :: rest)).state =
                (compileBlocks phase (rowStart + 1096) (witnessStart + 1096)
                  (permutationOutput witnessStart) (next :: rest)).state := rfl
            _ = permutationOutput
                (witnessStart + 1096 + ((next :: rest).length - 1) * 1096) :=
              tail
            _ = permutationOutput
                (witnessStart + ((block :: next :: rest).length - 1) * 1096) := by
              apply congrArg permutationOutput
              simp only [List.length_cons]
              omega

/-- A schedule without permutations leaves the incoming state unchanged. -/
theorem compileActions_state_of_invocationCount_zero
    (phase rowStart witnessStart : Nat) (state : EState)
    (actions : List Formal.Action) (zero : invocationCount actions = 0) :
    (compileActions phase rowStart witnessStart state actions).state = state := by
  induction actions generalizing rowStart witnessStart state with
  | nil => rfl
  | cons action actions inductionHypothesis =>
      rw [invocationCount_cons] at zero
      cases action with
      | absorb input =>
          have empty : Hash.inputChunks input = [] := by
            apply List.eq_nil_of_length_eq_zero
            simp only [Action.invocationCount] at zero
            omega
          simp only [compileActions, empty, compileBlocks]
          exact inductionHypothesis _ _ _ (by omega)
      | readK pair expected =>
          simp only [Action.invocationCount, Nat.zero_add] at zero
          exact inductionHypothesis _ _ _ zero

theorem compileActions_state_last (phase rowStart witnessStart : Nat)
    (state : EState) (actions : List Formal.Action)
    (positive : 0 < invocationCount actions) :
    (compileActions phase rowStart witnessStart state actions).state =
      permutationOutput
        (witnessStart + (invocationCount actions - 1) * 1096) := by
  induction actions generalizing rowStart witnessStart state with
  | nil => simp [invocationCount] at positive
  | cons action actions inductionHypothesis =>
      rw [invocationCount_cons] at positive ⊢
      cases action with
      | absorb input =>
          let absorbed := compileBlocks phase rowStart witnessStart state
            (Hash.inputChunks input)
          change
            (compileActions phase absorbed.rowNext absorbed.witnessNext
              absorbed.state actions).state = _
          simp only [Action.invocationCount] at positive ⊢
          by_cases tailZero : invocationCount actions = 0
          · rw [compileActions_state_of_invocationCount_zero _ _ _ _ _
              tailZero, tailZero]
            have nonempty : Hash.inputChunks input ≠ [] := by
              intro empty
              rw [empty, tailZero] at positive
              simp at positive
            rw [compileBlocks_state_last phase rowStart witnessStart state _
              nonempty, Nat.add_zero]
          · rw [inductionHypothesis _ _ _ (by omega), compileBlocks_witnessNext]
            apply congrArg permutationOutput
            exact lastOffset_cons witnessStart _ _ (by omega)
      | readK pair expected =>
          simp only [Action.invocationCount, Nat.zero_add] at positive ⊢
          exact inductionHypothesis rowStart witnessStart state positive

theorem compileActions_state_scheduleOutput
    (phase rowStart witnessStart : Nat) (state : EState)
    (actions : List Formal.Action)
    (positive : 0 < invocationCount actions) :
    (compileActions phase rowStart witnessStart state actions).state =
      Permutation.scheduleOutput
        (witnessStart + (invocationCount actions - 1) * 1096) := by
  exact compileActions_state_last phase rowStart witnessStart state actions
    positive

end NightstreamFPrime.Export.Stage1.InvocationLastOutput
