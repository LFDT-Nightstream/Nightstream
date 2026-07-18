import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.TranscriptMachine
import Nightstream.Implementation.R1CS.Core.TranscriptCertificate

/-!
Profile-independent refinement for one compact Poseidon2 transcript call.

Owns: decoding one call's input/output columns as canonical Goldilocks states
and proving that independent `CallAccepted` execution equals one transition of
the handwritten transcript machine.

Does not own: a protocol schedule, generated artifacts, inter-call wiring,
constant pins, row ownership, sampler semantics, costs, or row removal.

Emits constraints: no.

Authority boundary: `CallAccepted` independently replays the fixed 600-row
Poseidon2 SSA program. Call metadata selects columns but supplies no accepted
digest or semantic transcript event.

| Layer | Object | Proven obligation |
|---|---|---|
| physical | `callInputState` | exact eight assignment columns named by the call |
| physical | `callOutputState` | exact eight independently replayed output columns |
| refinement | `callAccepted_permute` | accepted call equals one semantic permutation |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.CallRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

set_option maxRecDepth 65536

/-- Canonical field element read from one artifact assignment column. -/
def fieldAt (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (column : Nat) : Field :=
  ⟨assignment column, canonical column⟩

@[simp] theorem fieldAt_val
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (column : Nat) :
    (fieldAt assignment canonical column).val = assignment column := by
  rfl

/-- The eight input lanes selected by one exact renamed Poseidon2 call. -/
def callInputState (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (call : Poseidon2Call.Call) (absorbed : Fin (rate + 1)) : State where
  lanes := fun lane => fieldAt assignment canonical
    (call.columnMap (lane.val + 1))
  absorbed := absorbed

@[simp] theorem callInputState_lane
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (call : Poseidon2Call.Call) (absorbed : Fin (rate + 1))
    (lane : Fin width) :
    (callInputState assignment canonical call absorbed).lanes lane =
      fieldAt assignment canonical (call.columnMap (lane.val + 1)) := by
  rfl

@[simp] theorem callInputState_absorbed
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (call : Poseidon2Call.Call) (absorbed : Fin (rate + 1)) :
    (callInputState assignment canonical call absorbed).absorbed = absorbed := by
  rfl

/-- The eight output lanes selected by one exact renamed Poseidon2 call. -/
def callOutputState (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (call : Poseidon2Call.Call) : State where
  lanes := fun lane => fieldAt assignment canonical
    (call.columnMap (601 + lane.val))
  absorbed := ⟨0, by decide⟩

@[simp] theorem callOutputState_lane
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (call : Poseidon2Call.Call) (lane : Fin width) :
    (callOutputState assignment canonical call).lanes lane =
      fieldAt assignment canonical (call.columnMap (601 + lane.val)) := by
  rfl

@[simp] theorem callOutputState_absorbed
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (call : Poseidon2Call.Call) :
    (callOutputState assignment canonical call).absorbed = ⟨0, by decide⟩ := by
  rfl

private theorem outputColumnsKnown :
    ∀ column ∈ [601, 602, 603, 604, 605, 606, 607, 608],
      column ∈ Program.knownAfter Poseidon2Permutation.inputColumns
        Poseidon2Permutation.definitions := by
  decide

private theorem callAccepted_lane
    {assignment : Nat → Nat}
    (call : Poseidon2Call.Call)
    (one : assignment 0 = 1)
    (accepted : TranscriptCertificate.CallAccepted call assignment)
    (lane : Nat) (laneLt : lane < width) :
    assignment (call.columnMap (601 + lane)) =
      Poseidon2PermutationSound.permute
        (fun inputLane => assignment (call.columnMap (inputLane + 1))) lane := by
  have outputMember :
      601 + lane ∈ Poseidon2Permutation.outputColumns := by
    change 601 + lane ∈ [601, 602, 603, 604, 605, 606, 607, 608]
    simp only [List.mem_cons, List.not_mem_nil, or_false]
    simp [width] at laneLt
    omega
  have known := outputColumnsKnown (601 + lane) (by
    simpa only using outputMember)
  have agrees := accepted (601 + lane) known
  have inputsAgree : Program.AgreeOn
      (pullAssignment assignment call.columnMap)
      (Poseidon2PermutationSound.inputOnly
        (pullAssignment assignment call.columnMap))
      Poseidon2Permutation.inputColumns := by
    intro column member
    simp only [Poseidon2Permutation.inputColumns, List.mem_cons,
      List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Poseidon2PermutationSound.inputOnly]
  have functional := Program.run_congr
    Poseidon2Permutation.definitions_wellFormed inputsAgree
    (601 + lane) known
  calc
    assignment (call.columnMap (601 + lane)) =
        pullAssignment assignment call.columnMap (601 + lane) := rfl
    _ = Poseidon2PermutationSound.interpret
          (pullAssignment assignment call.columnMap) (601 + lane) :=
        agrees.symm
    _ = Poseidon2PermutationSound.permuteState
          (pullAssignment assignment call.columnMap) (601 + lane) := by
        simpa [Poseidon2PermutationSound.interpret,
          Poseidon2PermutationSound.permuteState] using functional
    _ = Poseidon2PermutationSound.permute
          (fun inputLane => assignment (call.columnMap (inputLane + 1))) lane := by
      rw [Poseidon2PermutationSound.permute_eq]
      apply congrArg (fun state =>
        Poseidon2PermutationSound.interpret state (601 + lane))
      funext column
      unfold Poseidon2PermutationSound.inputOnly
      by_cases columnLt : column < 9
      · simp only [columnLt, ↓reduceIte]
        by_cases columnZero : column = 0
        · subst column
          simp [pullAssignment, Poseidon2Call.Call.columnMap,
            Poseidon2PermutationSound.permutationAssignment, one]
        · have columnPositive : 0 < column := Nat.pos_of_ne_zero columnZero
          simp [pullAssignment, Poseidon2Call.Call.columnMap, columnZero,
            columnLt, Poseidon2PermutationSound.permutationAssignment,
            Nat.sub_add_cancel columnPositive]
      · simp [columnLt]

/-- Independent acceptance of a renamed call refines one pure machine
permutation over exactly the assignment columns named by the call. -/
theorem callAccepted_permute
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (call : Poseidon2Call.Call)
    (absorbed : Fin (rate + 1))
    (accepted : TranscriptCertificate.CallAccepted call assignment) :
    permute (callInputState assignment canonical call absorbed) =
      callOutputState assignment canonical call := by
  unfold permute callOutputState
  congr
  funext lane
  apply Fin.ext
  change
    (Poseidon2PermutationSound.permute
      (laneNat (callInputState assignment canonical call absorbed))
      lane.val) % goldilocksP =
    assignment (call.columnMap (601 + lane.val))
  have inputLanes :
      ∀ inputLane, inputLane < width →
        laneNat (callInputState assignment canonical call absorbed)
            inputLane =
          assignment (call.columnMap (inputLane + 1)) := by
    intro inputLane inputLaneLt
    simp [laneNat, callInputState, fieldAt, inputLaneLt]
  rw [Poseidon2PermutationSound.permute_congr inputLanes lane.val]
  have output := callAccepted_lane call one accepted lane.val lane.isLt
  rw [← output, Nat.mod_eq_of_lt (canonical _)]

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.CallRefinement
