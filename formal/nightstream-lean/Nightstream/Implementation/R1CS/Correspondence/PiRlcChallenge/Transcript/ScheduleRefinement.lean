import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.DigestRounds
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.PinSchedule

/-!
State-level refinement of the recursive-profile Π_RLC transcript schedule.

Owns: the artifact-column interpretation of the state entering scalar zero,
the states produced by its scalar-domain transition and four digest blocks,
and proofs that the exact accepted calls and pins compose as the independent
overwrite transcript machine specifies.

Does not own: why the initial columns are the post-Π_CCS state, canonical-u64
bit/chunk decomposition, rejection selection, coefficient assembly, native
Rust conformance, any other scalar coordinate/profile, or rows/costs.

Emits constraints: no.

Authority boundary: this module consumes independently decoded pin equations
and independently replayed Poseidon2 calls. Generated row order alone does
not imply a transcript transition; every overwrite and preserved lane is
connected here.

| Protocol | Phase | State transition | Exact artifact columns |
|---|---|---|---|
| `Pi_RLC` | scalar 0 input | post-`Pi_CCS` state, cursor `2` | `348830`, `348831`, `350040..350045` |
| `Pi_RLC` | scalar domain | `enterScalar state 0` | overwritten lane `350048`, preserved outputs `350642..350648` |
| `Pi_RLC` | digest block 0 | `digestBlock state 0` | output lanes `351846..351853` |
| `Pi_RLC` | digest block 1 | `digestBlock state 1` | output lanes `353082..353089` |
| `Pi_RLC` | digest block 2 | `digestBlock state 2` | output lanes `354318..354325` |
| `Pi_RLC` | digest block 3 | `digestBlock state 3` | output lanes `355554..355561` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ScheduleRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

abbrev CanonicalAssignment (assignment : Nat → Nat) :=
  ∀ column, assignment column < goldilocksP

/-- The verifier state immediately before scalar coordinate zero. Its
connection to the complete preceding `Pi_CCS` transcript is intentionally a
separate proof obligation. -/
def initialState (assignment : Nat → Nat)
    (canonical : CanonicalAssignment assignment) : State where
  lanes := fun lane => DigestRounds.fieldAt assignment canonical
    ([348830, 348831, 350040, 350041, 350042, 350043, 350044, 350045].getD
      lane.val 0)
  absorbed := ⟨2, by decide⟩

/-- Artifact state after the scalar-domain permutation and coordinate-zero
overwrite. -/
def afterEnterState (assignment : Nat → Nat)
    (canonical : CanonicalAssignment assignment) : State where
  lanes := overwriteLane
    (DigestRounds.callOutputState assignment canonical
      Schedule.Artifact.enterScalarCall).lanes
    0 (DigestRounds.fieldAt assignment canonical 350048)
  absorbed := ⟨1, by decide⟩

/-- Artifact state after digest block zero. -/
def block0State (assignment : Nat → Nat)
    (canonical : CanonicalAssignment assignment) : State :=
  DigestRounds.callOutputState assignment canonical
    Schedule.Artifact.block0DigestCall

/-- Artifact state after digest block one. -/
def block1State (assignment : Nat → Nat)
    (canonical : CanonicalAssignment assignment) : State :=
  DigestRounds.callOutputState assignment canonical
    Schedule.Artifact.block1DigestCall

/-- Artifact state after digest block two. -/
def block2State (assignment : Nat → Nat)
    (canonical : CanonicalAssignment assignment) : State :=
  DigestRounds.callOutputState assignment canonical
    Schedule.Artifact.block2DigestCall

/-- Artifact state after digest block three. -/
def block3State (assignment : Nat → Nat)
    (canonical : CanonicalAssignment assignment) : State :=
  DigestRounds.callOutputState assignment canonical
    Schedule.Artifact.block3DigestCall

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

private theorem fieldAt_eq_wordField_zero
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    {column : Nat} (equal : assignment column = 0) :
    DigestRounds.fieldAt assignment canonical column = wordField 0 := by
  apply Fin.ext
  simp [DigestRounds.fieldAt, wordField, fieldValue, equal]

/-- The two in-rate scalar-domain words produce exactly the input state named
by the generated rate-boundary call. -/
theorem enterScalarCallInput
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (pins : PinSchedule.Facts assignment) :
    absorbElem
        (absorbElem (initialState assignment canonical) (wordField 2))
        (wordField 0) =
      DigestRounds.callInputState assignment canonical
        Schedule.Artifact.enterScalarCall ⟨4, by decide⟩ := by
  change
    { lanes := overwriteLane
        (overwriteLane (initialState assignment canonical).lanes
          2 (wordField 2))
        3 (wordField 0)
      absorbed := ⟨4, by decide⟩ } =
      DigestRounds.callInputState assignment canonical
        Schedule.Artifact.enterScalarCall ⟨4, by decide⟩
  apply stateExt
  · funext lane
    apply Fin.ext
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [initialState, overwriteLane,
        DigestRounds.callInputState, DigestRounds.fieldAt,
        Schedule.Artifact.enterScalarCall, Poseidon2Call.Call.columnMap,
        wordField, fieldValue, u64Modulus, goldilocksP, h, pins.enterLength,
        pins.enterDomain]
  · rfl

/-- Exact owner acceptance refines the scalar-zero domain transition of the
independent transcript machine. -/
theorem enterScalar_refines
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    enterScalar (initialState assignment canonical) 0 =
      afterEnterState assignment canonical := by
  have pins := PinSchedule.facts canonical one accepted
  have callAccepted := DigestRounds.enterScalarCallAccepted accepted
  unfold enterScalar appendRawPair
  rw [enterScalarCallInput canonical pins]
  change
    { lanes := overwriteLane
        (permute (DigestRounds.callInputState assignment canonical
          Schedule.Artifact.enterScalarCall ⟨4, by decide⟩)).lanes
        0 (wordField 0)
      absorbed := ⟨1, by decide⟩ } =
      afterEnterState assignment canonical
  rw [DigestRounds.callAccepted_permute canonical one
    Schedule.Artifact.enterScalarCall ⟨4, by decide⟩ callAccepted]
  have coordinate := fieldAt_eq_wordField_zero canonical pins.enterCoordinate
  rw [← coordinate]
  rfl

/-- Block zero's three raw-pair words fill exactly the state named by the
generated full-cursor call. -/
theorem block0FullCursorCallInput
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (pins : PinSchedule.Facts assignment) :
    appendRawPair (afterEnterState assignment canonical) 1 0 =
      DigestRounds.callInputState assignment canonical
        Schedule.Artifact.block0FullCursorCall ⟨4, by decide⟩ := by
  change
    { lanes := overwriteLane
        (overwriteLane
          (overwriteLane (afterEnterState assignment canonical).lanes
            1 (wordField 2))
          2 (wordField 1))
        3 (wordField 0)
      absorbed := ⟨4, by decide⟩ } =
      DigestRounds.callInputState assignment canonical
        Schedule.Artifact.block0FullCursorCall ⟨4, by decide⟩
  apply stateExt
  · funext lane
    apply Fin.ext
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [afterEnterState, overwriteLane, DigestRounds.callInputState,
        DigestRounds.callOutputState, DigestRounds.fieldAt,
        Schedule.Artifact.enterScalarCall,
        Schedule.Artifact.block0FullCursorCall,
        Poseidon2Call.Call.columnMap, wordField, fieldValue, u64Modulus,
        goldilocksP, h, pins.block0Length, pins.block0Domain,
        pins.block0Counter]
  · rfl

/-- The full-cursor call output plus squeeze word is exactly the input named
by block zero's digest call. -/
theorem block0DigestCallInput
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (pins : PinSchedule.Facts assignment)
    (accepted : TranscriptCertificate.CallAccepted
      Schedule.Artifact.block0FullCursorCall assignment) :
    absorbElem
        (DigestRounds.callInputState assignment canonical
          Schedule.Artifact.block0FullCursorCall ⟨4, by decide⟩)
        (wordField 1) =
      DigestRounds.callInputState assignment canonical
        Schedule.Artifact.block0DigestCall ⟨1, by decide⟩ := by
  change
    { lanes := overwriteLane
        (permute (DigestRounds.callInputState assignment canonical
          Schedule.Artifact.block0FullCursorCall ⟨4, by decide⟩)).lanes
        0 (wordField 1)
      absorbed := ⟨1, by decide⟩ } =
      DigestRounds.callInputState assignment canonical
        Schedule.Artifact.block0DigestCall ⟨1, by decide⟩
  rw [DigestRounds.callAccepted_permute canonical one
    Schedule.Artifact.block0FullCursorCall ⟨4, by decide⟩ accepted]
  apply stateExt
  · funext lane
    apply Fin.ext
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [overwriteLane, DigestRounds.callInputState,
        DigestRounds.callOutputState,
        Schedule.Artifact.block0FullCursorCall,
        Schedule.Artifact.block0DigestCall,
        Poseidon2Call.Call.columnMap, wordField, fieldValue, u64Modulus,
        goldilocksP, h, pins.block0Squeeze]
  · rfl

/-- Exact owner acceptance refines digest block zero's successor state. -/
theorem digestBlock0_refines
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    (digestBlock (afterEnterState assignment canonical) 0).1 =
      block0State assignment canonical := by
  have pins := PinSchedule.facts canonical one accepted
  have fullAccepted := DigestRounds.block0FullCursorCallAccepted accepted
  have digestAccepted := DigestRounds.block0DigestCallAccepted accepted
  change
    permute
        (absorbElem
          (appendRawPair (afterEnterState assignment canonical) 1 0)
          (wordField 1)) =
      block0State assignment canonical
  rw [block0FullCursorCallInput canonical pins]
  rw [block0DigestCallInput canonical one pins fullAccepted]
  simpa [block0State] using
    DigestRounds.callAccepted_permute canonical one
      Schedule.Artifact.block0DigestCall ⟨1, by decide⟩ digestAccepted

/-- Block one's four overwrite words produce exactly its one-call digest
input, preserving only block zero's high four lanes. -/
theorem block1DigestCallInput
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (pins : PinSchedule.Facts assignment) :
    absorbElem
        (appendRawPair (block0State assignment canonical) 1 1)
        (wordField 1) =
      DigestRounds.callInputState assignment canonical
        Schedule.Artifact.block1DigestCall ⟨4, by decide⟩ := by
  change
    { lanes := overwriteLane
        (overwriteLane
          (overwriteLane
            (overwriteLane (block0State assignment canonical).lanes
              0 (wordField 2))
            1 (wordField 1))
          2 (wordField 1))
        3 (wordField 1)
      absorbed := ⟨4, by decide⟩ } =
      DigestRounds.callInputState assignment canonical
        Schedule.Artifact.block1DigestCall ⟨4, by decide⟩
  apply stateExt
  · funext lane
    apply Fin.ext
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [block0State, overwriteLane, DigestRounds.callInputState,
        DigestRounds.callOutputState,
        Schedule.Artifact.block0DigestCall,
        Schedule.Artifact.block1DigestCall,
        Poseidon2Call.Call.columnMap, wordField, fieldValue, u64Modulus,
        goldilocksP, h, pins.block1Length, pins.block1Domain,
        pins.block1Counter, pins.block1Squeeze]
  · rfl

/-- Exact owner acceptance refines digest block one's successor state. -/
theorem digestBlock1_refines
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    (digestBlock (block0State assignment canonical) 1).1 =
      block1State assignment canonical := by
  have pins := PinSchedule.facts canonical one accepted
  have digestAccepted := DigestRounds.block1DigestCallAccepted accepted
  change
    permute
        (absorbElem
          (appendRawPair (block0State assignment canonical) 1 1)
          (wordField 1)) =
      block1State assignment canonical
  rw [block1DigestCallInput canonical pins]
  simpa [block1State] using
    DigestRounds.callAccepted_permute canonical one
      Schedule.Artifact.block1DigestCall ⟨4, by decide⟩ digestAccepted

/-- Block two's four overwrite words produce exactly its one-call digest
input, preserving only block one's high four lanes. -/
theorem block2DigestCallInput
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (pins : PinSchedule.Facts assignment) :
    absorbElem
        (appendRawPair (block1State assignment canonical) 1 2)
        (wordField 1) =
      DigestRounds.callInputState assignment canonical
        Schedule.Artifact.block2DigestCall ⟨4, by decide⟩ := by
  change
    { lanes := overwriteLane
        (overwriteLane
          (overwriteLane
            (overwriteLane (block1State assignment canonical).lanes
              0 (wordField 2))
            1 (wordField 1))
          2 (wordField 2))
        3 (wordField 1)
      absorbed := ⟨4, by decide⟩ } =
      DigestRounds.callInputState assignment canonical
        Schedule.Artifact.block2DigestCall ⟨4, by decide⟩
  apply stateExt
  · funext lane
    apply Fin.ext
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [block1State, overwriteLane, DigestRounds.callInputState,
        DigestRounds.callOutputState,
        Schedule.Artifact.block1DigestCall,
        Schedule.Artifact.block2DigestCall,
        Poseidon2Call.Call.columnMap, wordField, fieldValue, u64Modulus,
        goldilocksP, h, pins.block2Length, pins.block2Domain,
        pins.block2Counter, pins.block2Squeeze]
  · rfl

/-- Exact owner acceptance refines digest block two's successor state. -/
theorem digestBlock2_refines
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    (digestBlock (block1State assignment canonical) 2).1 =
      block2State assignment canonical := by
  have pins := PinSchedule.facts canonical one accepted
  have digestAccepted := DigestRounds.block2DigestCallAccepted accepted
  change
    permute
        (absorbElem
          (appendRawPair (block1State assignment canonical) 1 2)
          (wordField 1)) =
      block2State assignment canonical
  rw [block2DigestCallInput canonical pins]
  simpa [block2State] using
    DigestRounds.callAccepted_permute canonical one
      Schedule.Artifact.block2DigestCall ⟨4, by decide⟩ digestAccepted

/-- Block three's four overwrite words produce exactly its one-call digest
input, preserving only block two's high four lanes. -/
theorem block3DigestCallInput
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (pins : PinSchedule.Facts assignment) :
    absorbElem
        (appendRawPair (block2State assignment canonical) 1 3)
        (wordField 1) =
      DigestRounds.callInputState assignment canonical
        Schedule.Artifact.block3DigestCall ⟨4, by decide⟩ := by
  change
    { lanes := overwriteLane
        (overwriteLane
          (overwriteLane
            (overwriteLane (block2State assignment canonical).lanes
              0 (wordField 2))
            1 (wordField 1))
          2 (wordField 3))
        3 (wordField 1)
      absorbed := ⟨4, by decide⟩ } =
      DigestRounds.callInputState assignment canonical
        Schedule.Artifact.block3DigestCall ⟨4, by decide⟩
  apply stateExt
  · funext lane
    apply Fin.ext
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [block2State, overwriteLane, DigestRounds.callInputState,
        DigestRounds.callOutputState,
        Schedule.Artifact.block2DigestCall,
        Schedule.Artifact.block3DigestCall,
        Poseidon2Call.Call.columnMap, wordField, fieldValue, u64Modulus,
        goldilocksP, h, pins.block3Length, pins.block3Domain,
        pins.block3Counter, pins.block3Squeeze] <;> rfl
  · rfl

/-- Exact owner acceptance refines digest block three's successor state. -/
theorem digestBlock3_refines
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    (digestBlock (block2State assignment canonical) 3).1 =
      block3State assignment canonical := by
  have pins := PinSchedule.facts canonical one accepted
  have digestAccepted := DigestRounds.block3DigestCallAccepted accepted
  change
    permute
        (absorbElem
          (appendRawPair (block2State assignment canonical) 1 3)
          (wordField 1)) =
      block3State assignment canonical
  rw [block3DigestCallInput canonical pins]
  simpa [block3State] using
    DigestRounds.callAccepted_permute canonical one
      Schedule.Artifact.block3DigestCall ⟨4, by decide⟩ digestAccepted

/-- State-only semantic closure for the exact one-scalar/four-block recursive
artifact schedule. Candidate chunks remain a separate refinement layer. -/
structure RefinesStateSchedule
    (assignment : Nat → Nat)
    (canonical : CanonicalAssignment assignment) : Prop where
  enterScalar : enterScalar (initialState assignment canonical) 0 =
    afterEnterState assignment canonical
  block0 : (digestBlock (afterEnterState assignment canonical) 0).1 =
    block0State assignment canonical
  block1 : (digestBlock (block0State assignment canonical) 1).1 =
    block1State assignment canonical
  block2 : (digestBlock (block1State assignment canonical) 2).1 =
    block2State assignment canonical
  block3 : (digestBlock (block2State assignment canonical) 3).1 =
    block3State assignment canonical

/-- Accepted artifact equations force the exact independent state schedule.
This theorem still assumes, rather than proves, that `initialState` is the
state reached by the complete preceding `Pi_CCS` transcript. -/
theorem accepted_refines_stateSchedule
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    RefinesStateSchedule assignment canonical :=
  { enterScalar := enterScalar_refines canonical one accepted
    block0 := digestBlock0_refines canonical one accepted
    block1 := digestBlock1_refines canonical one accepted
    block2 := digestBlock2_refines canonical one accepted
    block3 := digestBlock3_refines canonical one accepted }

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ScheduleRefinement
