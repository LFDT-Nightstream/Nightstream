import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Schedule
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.TranscriptMachine

/-!
Semantic bridge for exact Poseidon call pieces used by the Π_RLC sampler.

Owns: a reusable theorem turning independent call acceptance into equality
between the pure transcript-machine permutation and the artifact assignment's
eight output columns, plus extraction of acceptance for each scheduled call.

Does not own: constant pins, inter-call wire connectivity, lane bit/chunk
decomposition, rejection/selection semantics, the reached post-PiCCS state,
native Rust conformance, or costs.

Emits constraints: no.

Authority boundary: the theorem consumes `TranscriptCertificate.CallAccepted`,
which independently replays the fixed Poseidon2 SSA interpreter. The generated
piece descriptor merely selects which assignment columns instantiate that
semantics.

| Protocol | Phase | Constraint family | Proven guarantee |
|---|---|---|---|
| `Pi_RLC` | transcript | one Poseidon call | accepted call outputs equal the pure machine permutation of the same eight inputs |
| `Pi_RLC` | scalar domain | scheduled call | owner acceptance exposes independent acceptance of the domain-boundary call |
| `Pi_RLC` | digest blocks | scheduled calls | owner acceptance exposes independent acceptance of all five block permutations |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

set_option maxRecDepth 65536

/-- Canonical field element read from one artifact assignment column. -/
def fieldAt (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (column : Nat) : Field :=
  ⟨assignment column, canonical column⟩

/-- The eight input lanes selected by one exact renamed Poseidon call. -/
def callInputState (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (call : Poseidon2Call.Call) (absorbed : Fin (rate + 1)) : State where
  lanes := fun lane => fieldAt assignment canonical
    (call.columnMap (lane.val + 1))
  absorbed := absorbed

/-- The eight output lanes selected by one exact renamed Poseidon call. -/
def callOutputState (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (call : Poseidon2Call.Call) : State where
  lanes := fun lane => fieldAt assignment canonical
    (call.columnMap (601 + lane.val))
  absorbed := ⟨0, by decide⟩

/-- Kernel-reduced membership of the eight final SSA outputs. This replaces
the older `native_decide`-backed convenience theorem so the active refinement
path does not acquire `Lean.trustCompiler`. -/
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

private theorem acceptedScheduledCall
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (piece : Piece)
    (member : piece ∈
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces)
    (call : Poseidon2Call.Call)
    (payload : piece.payload = .poseidon call) :
    TranscriptCertificate.CallAccepted call assignment := by
  have pieceAccepted := accepted piece member
  rw [Piece.Accepted, payload] at pieceAccepted
  exact pieceAccepted

theorem enterScalarCallAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    TranscriptCertificate.CallAccepted Schedule.Artifact.enterScalarCall
      assignment := by
  exact acceptedScheduledCall accepted Schedule.Artifact.enterScalarPiece
    Schedule.Artifact.enterScalarPiece_mem
    Schedule.Artifact.enterScalarCall rfl

theorem block0FullCursorCallAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    TranscriptCertificate.CallAccepted
      Schedule.Artifact.block0FullCursorCall assignment := by
  exact acceptedScheduledCall accepted Schedule.Artifact.block0FullCursorPiece
    Schedule.Artifact.block0FullCursorPiece_mem
    Schedule.Artifact.block0FullCursorCall rfl

theorem block0DigestCallAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    TranscriptCertificate.CallAccepted Schedule.Artifact.block0DigestCall
      assignment := by
  exact acceptedScheduledCall accepted Schedule.Artifact.block0DigestPiece
    Schedule.Artifact.block0DigestPiece_mem
    Schedule.Artifact.block0DigestCall rfl

theorem block1DigestCallAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    TranscriptCertificate.CallAccepted Schedule.Artifact.block1DigestCall
      assignment := by
  exact acceptedScheduledCall accepted Schedule.Artifact.block1DigestPiece
    Schedule.Artifact.block1DigestPiece_mem
    Schedule.Artifact.block1DigestCall rfl

theorem block2DigestCallAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    TranscriptCertificate.CallAccepted Schedule.Artifact.block2DigestCall
      assignment := by
  exact acceptedScheduledCall accepted Schedule.Artifact.block2DigestPiece
    Schedule.Artifact.block2DigestPiece_mem
    Schedule.Artifact.block2DigestCall rfl

theorem block3DigestCallAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    TranscriptCertificate.CallAccepted Schedule.Artifact.block3DigestCall
      assignment := by
  exact acceptedScheduledCall accepted Schedule.Artifact.block3DigestPiece
    Schedule.Artifact.block3DigestPiece_mem
    Schedule.Artifact.block3DigestCall rfl

/-- The exact recursive owner exposes every call needed by the one-scalar
domain/digest schedule. This packages call semantics, not inter-call wiring. -/
structure ScheduledCallsAccepted (assignment : Nat → Nat) : Prop where
  enterScalar : TranscriptCertificate.CallAccepted
    Schedule.Artifact.enterScalarCall assignment
  block0FullCursor : TranscriptCertificate.CallAccepted
    Schedule.Artifact.block0FullCursorCall assignment
  block0Digest : TranscriptCertificate.CallAccepted
    Schedule.Artifact.block0DigestCall assignment
  block1Digest : TranscriptCertificate.CallAccepted
    Schedule.Artifact.block1DigestCall assignment
  block2Digest : TranscriptCertificate.CallAccepted
    Schedule.Artifact.block2DigestCall assignment
  block3Digest : TranscriptCertificate.CallAccepted
    Schedule.Artifact.block3DigestCall assignment

theorem scheduledCallsAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    ScheduledCallsAccepted assignment :=
  { enterScalar := enterScalarCallAccepted accepted
    block0FullCursor := block0FullCursorCallAccepted accepted
    block0Digest := block0DigestCallAccepted accepted
    block1Digest := block1DigestCallAccepted accepted
    block2Digest := block2DigestCallAccepted accepted
    block3Digest := block3DigestCallAccepted accepted }

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds
