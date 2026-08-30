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

def ActionsPositive (actions : List Formal.Action) : Prop :=
  ∀ action ∈ actions, 0 < Action.invocationCount action

private theorem invocationCount_cons (action : Formal.Action)
    (actions : List Formal.Action) :
    invocationCount (action :: actions) =
      Action.invocationCount action + invocationCount actions := by
  simp [invocationCount]

private theorem lastOffset_cons (witnessStart headCount tailCount : Nat)
    (tailPositive : 0 < tailCount) :
    witnessStart + headCount * 592 + (tailCount - 1) * 592 =
      witnessStart + (headCount + tailCount - 1) * 592 := by
  omega

theorem compileBlocks_state_last (phase rowStart witnessStart : Nat)
    (state : EState) (blocks : List (List Expr)) (nonempty : blocks ≠ []) :
    (compileBlocks phase rowStart witnessStart state blocks).state =
      permutationOutput
        (witnessStart + (blocks.length - 1) * 592) := by
  induction blocks generalizing rowStart witnessStart state with
  | nil => exact False.elim (nonempty rfl)
  | cons block blocks inductionHypothesis =>
      cases blocks with
      | nil =>
          change permutationOutput witnessStart = _
          apply congrArg permutationOutput
          simp
      | cons next rest =>
          have tail := inductionHypothesis (rowStart + 592)
            (witnessStart + 592) (permutationOutput witnessStart) (by simp)
          calc
            (compileBlocks phase rowStart witnessStart state
                (block :: next :: rest)).state =
                (compileBlocks phase (rowStart + 592) (witnessStart + 592)
                  (permutationOutput witnessStart) (next :: rest)).state := rfl
            _ = permutationOutput
                (witnessStart + 592 + ((next :: rest).length - 1) * 592) :=
              tail
            _ = permutationOutput
                (witnessStart + ((block :: next :: rest).length - 1) * 592) := by
              apply congrArg permutationOutput
              simp only [List.length_cons]
              omega

theorem compileActions_singleton_state (phase rowStart witnessStart : Nat)
    (state : EState) (action : Formal.Action)
    (positive : 0 < Action.invocationCount action) :
    (compileActions phase rowStart witnessStart state [action]).state =
      permutationOutput
        (witnessStart + (Action.invocationCount action - 1) * 592) := by
  cases action with
  | absorb input =>
      have chunksNonempty : Hash.inputChunks input ≠ [] := by
        intro empty
        simp [Action.invocationCount, empty] at positive
      simpa [compileActions, Action.invocationCount] using
        compileBlocks_state_last phase rowStart witnessStart state
          (Hash.inputChunks input) chunksNonempty
  | squeezeK expected =>
      change permutationOutput (witnessStart + 592) =
        permutationOutput (witnessStart + (2 - 1) * 592)
      apply congrArg permutationOutput
      omega

theorem compileActions_state_last (phase rowStart witnessStart : Nat)
    (state : EState) (actions : List Formal.Action) (nonempty : actions ≠ [])
    (positive : ActionsPositive actions) :
    (compileActions phase rowStart witnessStart state actions).state =
      permutationOutput
        (witnessStart + (invocationCount actions - 1) * 592) := by
  induction actions generalizing rowStart witnessStart state with
  | nil => exact False.elim (nonempty rfl)
  | cons action actions inductionHypothesis =>
      cases actions with
      | nil =>
          exact compileActions_singleton_state phase rowStart witnessStart state
            action (positive action (by simp))
      | cons next rest =>
          have tailPositive : ActionsPositive (next :: rest) := by
            intro current member
            exact positive current (by simp [member])
          have tailCountPositive : 0 < invocationCount (next :: rest) := by
            have headPositive := tailPositive next (by simp)
            simp only [invocationCount, List.map_cons, List.sum_cons]
            omega
          cases action with
          | absorb input =>
              let absorbed := compileBlocks phase rowStart witnessStart state
                (Hash.inputChunks input)
              change
                (compileActions phase absorbed.rowNext absorbed.witnessNext
                  absorbed.state (next :: rest)).state = _
              have tail := inductionHypothesis absorbed.rowNext
                absorbed.witnessNext absorbed.state (by simp) tailPositive
              rw [tail, compileBlocks_witnessNext]
              rw [invocationCount_cons (.absorb input) (next :: rest)]
              simp only [Action.invocationCount]
              apply congrArg permutationOutput
              exact lastOffset_cons witnessStart
                (Hash.inputChunks input).length
                (invocationCount (next :: rest)) tailCountPositive
          | squeezeK expected =>
              change
                (compileActions phase (rowStart + 1184) (witnessStart + 1184)
                  (permutationOutput (witnessStart + 592))
                  (next :: rest)).state = _
              have tail := inductionHypothesis (rowStart + 1184)
                (witnessStart + 1184)
                (permutationOutput (witnessStart + 592)) (by simp) tailPositive
              rw [tail]
              rw [invocationCount_cons (.squeezeK expected) (next :: rest)]
              simp only [Action.invocationCount]
              apply congrArg permutationOutput
              simpa using lastOffset_cons witnessStart 2
                (invocationCount (next :: rest)) tailCountPositive

theorem compileActions_state_scheduleOutput
    (phase rowStart witnessStart : Nat) (state : EState)
    (actions : List Formal.Action) (nonempty : actions ≠ [])
    (positive : ActionsPositive actions) :
    (compileActions phase rowStart witnessStart state actions).state =
      Permutation.scheduleOutput
        (witnessStart + (invocationCount actions - 1) * 592) := by
  exact compileActions_state_last phase rowStart witnessStart state actions
    nonempty positive

end NightstreamFPrime.Export.Stage1.InvocationLastOutput
