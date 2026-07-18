import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.DigestRounds
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.OutputDigestPins
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.ScheduleRefinement

/-!
State-level refinement of the terminal `Pi_CCS` output-digest handoff into
the `Pi_RLC` sampler.

Assurance tier: implementation/R1CS correspondence. This module proves that
accepted equations and independently replayed Poseidon2 calls implement the
separately specified output-digest transcript transition.

Owns: interpretation of the post-catch-up state; the four digest-field input
columns; both full-rate boundaries; final buffering at cursor two; and exact
equality with the initial state consumed by terminal rho sampling.

Does not own: why the catch-up input state is the complete accepted `Pi_CCS`
transcript; why columns `2553433..2553436` are the recomputed digest of every
accepted `Pi_CCS` output; native Rust conformance; challenge-sampler algebra;
row necessity, row removal, or cost totals.

Emits constraints: no.

Authority boundary: this theorem never treats a supplied digest or generated
state as authority. It proves deterministic absorption of four named field
columns. Full authority still requires the separately named obligations that
derive those columns and the catch-up input from accepted `Pi_CCS` semantics.

| Protocol | Phase | Constraint family | Lean guarantee |
|---|---|---|---|
| `Pi_CCS` | catch-up | Poseidon2 | accepted call computes the state used by the handoff |
| `Pi_RLC` | label boundary | constants + Poseidon2 | exact packed label fills and crosses the first rate window |
| `Pi_RLC` | digest boundary | count/digest + Poseidon2 | count and first digest lanes fill and cross the second rate window |
| `Pi_RLC` | sampler entry | buffered digest lanes | final state equals the rho sampler's cursor-two initial state |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.OutputDigestBinding

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds

abbrev CanonicalAssignment (assignment : Nat -> Nat) :=
  forall column, assignment column < goldilocksP

/-- Computed state produced by the exact terminal `Pi_CCS` catch-up call. -/
def postCatchupState (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment) : State :=
  callOutputState assignment canonical OutputDigestSchedule.catchupCall

/-- Four fields absorbed by the output-digest handoff. This definition does
not claim that they are authoritative; that is a preceding proof obligation. -/
def outputDigest (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment) : Fin 4 -> Field :=
  fun lane => fieldAt assignment canonical
    ([2553433, 2553434, 2553435, 2553436].getD lane.val 0)

/-- Explicit state shape after four fresh-cursor label absorbs. Keeping this
shape named prevents proof simplification from expanding the permutation
interpreter while checking ordinary overwrite connectivity. -/
def labelPrefixState (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment) : State where
  lanes := overwriteLane
    (overwriteLane
      (overwriteLane
        (overwriteLane (postCatchupState assignment canonical).lanes
          0 (wordField 26))
        1 (wordField 13338641331874160))
      2 (wordField 27970976485502569))
    3 (wordField 28252447032566124)
  absorbed := ⟨4, by decide⟩

theorem labelPrefix_shape
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment) :
    OutputDigestSemantics.labelPrefix
        (postCatchupState assignment canonical) =
      labelPrefixState assignment canonical := by
  rfl

/-- Explicit full-cursor state before the second digest-binding call. -/
def secondBoundaryInputState (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment) : State where
  lanes := overwriteLane
    (overwriteLane
      (overwriteLane
        (overwriteLane
          (callOutputState assignment canonical
            OutputDigestSchedule.firstBoundaryCall).lanes
          0 (wordField 500152231785))
        1 (wordField 4))
      2 (outputDigest assignment canonical ⟨0, by decide⟩))
    3 (outputDigest assignment canonical ⟨1, by decide⟩)
  absorbed := ⟨4, by decide⟩

private theorem laneValueCases (lane : Fin width) :
    lane.val = 0 ∨ lane.val = 1 ∨ lane.val = 2 ∨ lane.val = 3 ∨
    lane.val = 4 ∨ lane.val = 5 ∨ lane.val = 6 ∨ lane.val = 7 := by
  have laneLt : lane.val < 8 := by
    simpa [width] using lane.isLt
  omega

private theorem stateExt {left right : State}
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) : left = right := by
  cases left
  cases right
  simp_all

private theorem fieldAt_eq_wordField
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    {column value : Nat}
    (equal : assignment column = value)
    (valueLtU64 : value < u64Modulus)
    (valueLtField : value < goldilocksP) :
    fieldAt assignment canonical column = wordField value := by
  apply Fin.ext
  simp [fieldAt, wordField, fieldValue, equal,
    Nat.mod_eq_of_lt valueLtU64, Nat.mod_eq_of_lt valueLtField]

private theorem acceptedScheduledCall
    {assignment : Nat -> Nat}
    (piece : Piece)
    (pieceAccepted : piece.Accepted assignment)
    (call : Poseidon2Call.Call)
    (payload : piece.payload = .poseidon call) :
    TranscriptCertificate.CallAccepted call assignment := by
  rw [Piece.Accepted, payload, Payload.Accepted] at pieceAccepted
  exact pieceAccepted

theorem catchupCallAccepted
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsCatchup.Accepted assignment) :
    TranscriptCertificate.CallAccepted OutputDigestSchedule.catchupCall
      assignment := by
  exact acceptedScheduledCall OutputDigestSchedule.catchupPiece
    (accepted OutputDigestSchedule.catchupPiece
      OutputDigestSchedule.catchupPiece_mem)
    OutputDigestSchedule.catchupCall
    (by rw [OutputDigestSchedule.catchupPiece_eq]; rfl)

theorem firstBoundaryCallAccepted
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    TranscriptCertificate.CallAccepted
      OutputDigestSchedule.firstBoundaryCall assignment := by
  exact acceptedScheduledCall OutputDigestSchedule.firstBoundaryPiece
    (accepted OutputDigestSchedule.firstBoundaryPiece
      OutputDigestSchedule.firstBoundaryPiece_mem)
    OutputDigestSchedule.firstBoundaryCall
    (by rw [OutputDigestSchedule.firstBoundaryPiece_eq]; rfl)

theorem secondBoundaryCallAccepted
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    TranscriptCertificate.CallAccepted
      OutputDigestSchedule.secondBoundaryCall assignment := by
  exact acceptedScheduledCall OutputDigestSchedule.secondBoundaryPiece
    (accepted OutputDigestSchedule.secondBoundaryPiece
      OutputDigestSchedule.secondBoundaryPiece_mem)
    OutputDigestSchedule.secondBoundaryCall
    (by rw [OutputDigestSchedule.secondBoundaryPiece_eq]; rfl)

/-- The catch-up output is computed by the independent Poseidon2 interpreter,
not accepted as a carried state value. -/
theorem catchupState_computed
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsCatchup.Accepted assignment) :
    permute (callInputState assignment canonical
      OutputDigestSchedule.catchupCall ⟨1, by decide⟩) =
      postCatchupState assignment canonical := by
  exact callAccepted_permute canonical one OutputDigestSchedule.catchupCall
    ⟨1, by decide⟩ (catchupCallAccepted accepted)

/-- The first four independently encoded label fields are exactly the input
state of the first artifact call. -/
theorem labelPrefix_eq_firstBoundaryInput
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (pins : OutputDigestPins.Facts assignment) :
    OutputDigestSemantics.labelPrefix
        (postCatchupState assignment canonical) =
      callInputState assignment canonical
        OutputDigestSchedule.firstBoundaryCall ⟨4, by decide⟩ := by
  have label0 : assignment 2553445 = 26 :=
    pins.label (2553445, 26) (by simp [OutputDigestSchedule.labelPins])
  have label1 : assignment 2553446 = 13338641331874160 :=
    pins.label (2553446, 13338641331874160)
      (by simp [OutputDigestSchedule.labelPins])
  have label2 : assignment 2553447 = 27970976485502569 :=
    pins.label (2553447, 27970976485502569)
      (by simp [OutputDigestSchedule.labelPins])
  have label3 : assignment 2553448 = 28252447032566124 :=
    pins.label (2553448, 28252447032566124)
      (by simp [OutputDigestSchedule.labelPins])
  have label0Field := fieldAt_eq_wordField canonical label0
    (by decide) (by decide)
  have label1Field := fieldAt_eq_wordField canonical label1
    (by decide) (by decide)
  have label2Field := fieldAt_eq_wordField canonical label2
    (by decide) (by decide)
  have label3Field := fieldAt_eq_wordField canonical label3
    (by decide) (by decide)
  rw [labelPrefix_shape]
  apply stateExt
  · funext lane
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [labelPrefixState, postCatchupState,
        OutputDigestSchedule.catchupCall,
        OutputDigestSchedule.firstBoundaryCall,
        Poseidon2Call.Call.columnMap, overwriteLane,
        h, label0Field, label1Field, label2Field, label3Field]
  · rfl

/-- Crossing the first full cursor computes the exact first-boundary output
and then buffers the final label limb in lane zero. -/
theorem firstBoundary_refines
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (pins : OutputDigestPins.Facts assignment)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    OutputDigestSemantics.afterFirstBoundary
        (postCatchupState assignment canonical) =
      absorbElem
        (callOutputState assignment canonical
          OutputDigestSchedule.firstBoundaryCall)
        (wordField 500152231785) := by
  unfold OutputDigestSemantics.afterFirstBoundary
  rw [labelPrefix_eq_firstBoundaryInput canonical pins]
  have permutation := callAccepted_permute canonical one
    OutputDigestSchedule.firstBoundaryCall ⟨4, by decide⟩
    (firstBoundaryCallAccepted accepted)
  change
    { lanes := overwriteLane
        (permute (callInputState assignment canonical
          OutputDigestSchedule.firstBoundaryCall ⟨4, by decide⟩)).lanes
        0 (wordField 500152231785)
      absorbed := ⟨1, by decide⟩ } =
      absorbElem
        (callOutputState assignment canonical
          OutputDigestSchedule.firstBoundaryCall)
        (wordField 500152231785)
  rw [permutation]
  rfl

/-- The count and first two digest fields are exactly the input state of the
second artifact call. -/
theorem beforeSecondBoundary_eq_input
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (pins : OutputDigestPins.Facts assignment)
    (firstBoundary :
      OutputDigestSemantics.afterFirstBoundary
          (postCatchupState assignment canonical) =
        absorbElem
          (callOutputState assignment canonical
            OutputDigestSchedule.firstBoundaryCall)
          (wordField 500152231785)) :
    OutputDigestSemantics.beforeSecondBoundary
        (postCatchupState assignment canonical)
        (outputDigest assignment canonical) =
      callInputState assignment canonical
        OutputDigestSchedule.secondBoundaryCall ⟨4, by decide⟩ := by
  have label4 : assignment 2553449 = 500152231785 :=
    pins.label (2553449, 500152231785)
      (by simp [OutputDigestSchedule.labelPins])
  have fieldCount : assignment 2554050 = 4 :=
    pins.fieldCount (2554050, 4)
      (by simp [OutputDigestSchedule.fieldCountPins])
  have label4Field := fieldAt_eq_wordField canonical label4
    (by decide) (by decide)
  have fieldCountField := fieldAt_eq_wordField canonical fieldCount
    (by decide) (by decide)
  rw [OutputDigestSemantics.beforeSecondBoundary, firstBoundary]
  change secondBoundaryInputState assignment canonical =
    callInputState assignment canonical
      OutputDigestSchedule.secondBoundaryCall ⟨4, by decide⟩
  apply stateExt
  · funext lane
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [secondBoundaryInputState, outputDigest,
        OutputDigestSchedule.firstBoundaryCall,
        OutputDigestSchedule.secondBoundaryCall,
        Poseidon2Call.Call.columnMap, overwriteLane,
        h, label4Field, fieldCountField]
  · rfl

/-- The second boundary is independently replayed; buffering digest lanes two
and three yields exactly the state consumed by scalar zero. -/
theorem completeBinding_eq_initialState
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (pins : OutputDigestPins.Facts assignment)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    OutputDigestSemantics.completeBinding
        (postCatchupState assignment canonical)
        (outputDigest assignment canonical) =
      ScheduleRefinement.initialState assignment canonical := by
  have first := firstBoundary_refines canonical one pins accepted
  have secondInput := beforeSecondBoundary_eq_input canonical pins first
  unfold OutputDigestSemantics.completeBinding
  unfold OutputDigestSemantics.afterSecondBoundary
  rw [secondInput]
  have permutation := callAccepted_permute canonical one
    OutputDigestSchedule.secondBoundaryCall ⟨4, by decide⟩
    (secondBoundaryCallAccepted accepted)
  change
    absorbElem
      { lanes := overwriteLane
          (permute (callInputState assignment canonical
            OutputDigestSchedule.secondBoundaryCall ⟨4, by decide⟩)).lanes
          0 (outputDigest assignment canonical ⟨2, by decide⟩)
        absorbed := ⟨1, by decide⟩ }
      (outputDigest assignment canonical ⟨3, by decide⟩) =
      ScheduleRefinement.initialState assignment canonical
  rw [permutation]
  change
    { lanes := overwriteLane
        (overwriteLane
          (callOutputState assignment canonical
            OutputDigestSchedule.secondBoundaryCall).lanes
          0 (outputDigest assignment canonical ⟨2, by decide⟩))
        1 (outputDigest assignment canonical ⟨3, by decide⟩)
      absorbed := ⟨2, by decide⟩ } =
      ScheduleRefinement.initialState assignment canonical
  apply stateExt
  · funext lane
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [outputDigest, ScheduleRefinement.initialState,
        OutputDigestSchedule.secondBoundaryCall,
        Poseidon2Call.Call.columnMap, overwriteLane, h]
  · rfl

/-- Accepted terminal handoff rows refine the independent output-digest
transition all the way to the already-audited rho sampler state. This theorem
does not discharge the preceding digest-authority or catch-up-input obligations. -/
theorem accepted_refines_outputDigestBinding
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (catchupAccepted :
      FPrimeFullHistoryTerminalPiCcsCatchup.Accepted assignment)
    (rlcAccepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    permute (callInputState assignment canonical
        OutputDigestSchedule.catchupCall ⟨1, by decide⟩) =
        postCatchupState assignment canonical /\
      OutputDigestSemantics.appendInputClaimsDigest
          (postCatchupState assignment canonical)
          (outputDigest assignment canonical) =
        ScheduleRefinement.initialState assignment canonical := by
  have pins := OutputDigestPins.facts canonical one catchupAccepted rlcAccepted
  constructor
  · exact catchupState_computed canonical one catchupAccepted
  · rw [OutputDigestSemantics.appendInputClaimsDigest_eq_completeBinding]
    exact completeBinding_eq_initialState canonical one pins rlcAccepted

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.OutputDigestBinding
