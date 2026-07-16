import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.DigestRounds
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.PinSchedule
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Binding
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.DigestRounds
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryTerminalTranscriptSound

/-!
State-level refinement of the terminal `Pi_CCS` binding and challenge prefix.

Assurance tier: implementation/R1CS correspondence. This module composes
accepted constant equations and independently replayed Poseidon2 calls into
the independently specified overwrite transcript machine.

Owns: the assignment interpretation of the fixed terminal input state; typed
header, instance-digest, running-count, and checked-parent fields; explicit
state continuity across the binding calls; and eventually the two challenge
call families.

Does not own: derivation of the fixed header from public parameters,
recomputation of the four instance-digest fields, validation of the four
checked-parent fields, FE/NC SumCheck, catch-up, Rust conformance, costs, or
row removal.

Emits constraints: no.

Authority boundary: dynamic digest and parent fields are only interpreted
here. Separate producer theorems must prove they are verifier-derived before
the completed transcript can authorize challenges. Every permutation in this
file comes from `TranscriptCertificate.CallAccepted`, never from row order or
a carried digest.

| Protocol | Phase | Constraint family | Mathematical obligation |
|---|---|---|---|
| `Pi_CCS` | input | terminal state | interpret eight fixed-profile columns at cursor `2` |
| `Pi_CCS` | header | two Poseidon2 boundaries | execute raw `[11, header[0..4]]` exactly |
| `Pi_CCS` | instance | Poseidon2 boundary | bind raw `[12, instanceDigest[0..4]]` |
| `Pi_CCS` | running authority | three Poseidon2 boundaries | bind `[4]`, `[5,14]`, and `[13,parent[0..4]]` |
| `Pi_CCS` | main challenges | seven Poseidon2 calls | replay the `[2]` response stream |
| `Pi_CCS` | `beta_m` | five Poseidon2 calls | replay the `[3]` response stream |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.ScheduleRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiCcsTranscript
open Nightstream.Implementation.R1CS.PiCcsTranscript.Binding
open Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds

set_option maxHeartbeats 4000000

abbrev CanonicalAssignment (assignment : Nat -> Nat) :=
  forall column, assignment column < goldilocksP

def headerBoundaryIndex : Fin Schedule.bindingCount := ⟨0, by decide⟩
def headerPayloadIndex : Fin Schedule.bindingCount := ⟨1, by decide⟩
def instanceBoundaryIndex : Fin Schedule.bindingCount := ⟨2, by decide⟩
def runningDomainIndex : Fin Schedule.bindingCount := ⟨3, by decide⟩
def countParentBoundaryIndex : Fin Schedule.bindingCount := ⟨4, by decide⟩
def parentPayloadIndex : Fin Schedule.bindingCount := ⟨5, by decide⟩

def countParentBoundaryCall : Poseidon2Call.Call :=
  { rowStart := 6651, rowEnd := 7251
    inputColumns := [1625695, 1625693, 1625694, 1625697,
      1625689, 1625690, 1625691, 1625692]
    firstAllocatedColumn := 1625698 }

def parentPayloadCall : Poseidon2Call.Call :=
  { rowStart := 7251, rowEnd := 7851
    inputColumns := [1625696, 1619039, 1619040, 1619041,
      1626294, 1626295, 1626296, 1626297]
    firstAllocatedColumn := 1626298 }

theorem countParentBoundaryCall_eq :
    Schedule.bindingCall countParentBoundaryIndex = countParentBoundaryCall := by
  rfl

theorem parentPayloadCall_eq :
    Schedule.bindingCall parentPayloadIndex = parentPayloadCall := by
  rfl

private theorem countParentOutputLane
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (lane : Fin width) :
    (callOutputState assignment canonical countParentBoundaryCall).lanes lane =
      fieldAt assignment canonical (1626290 + lane.val) := by
  unfold callOutputState
  apply congrArg (fieldAt assignment canonical)
  have large : ¬ 601 + lane.val < 9 := by omega
  simp [Poseidon2Call.Call.columnMap, countParentBoundaryCall, large]
  omega

private theorem parentPayloadInputLane
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (lane : Fin width) :
    (callInputState assignment canonical parentPayloadCall
      ⟨4, by decide⟩).lanes lane =
      fieldAt assignment canonical
        ([1625696, 1619039, 1619040, 1619041,
          1626294, 1626295, 1626296, 1626297].getD lane.val 0) := by
  unfold callInputState
  apply congrArg (fieldAt assignment canonical)
  have small : lane.val + 1 < 9 := by
    simpa [Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine.width]
      using Nat.succ_lt_succ lane.isLt
  simp [Poseidon2Call.Call.columnMap, parentPayloadCall, small]

def initialColumns : List Nat :=
  [1132300, 1132301, 1132302, 1132303,
   1132304, 1132305, 1132306, 1132307]

/-- Assignment-backed state immediately before the terminal `Pi_CCS`
binding prefix. A separate theorem below binds these columns to the fixed
terminal initialization owner. -/
def initialState (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment) : State where
  lanes := fun lane => fieldAt assignment canonical
    (initialColumns.getD lane.val 0)
  absorbed := ⟨2, by decide⟩

def headerColumns : List Nat := [1623283, 1623284, 1623285, 1623286]
def instanceDigestColumns : List Nat := [1623275, 1623276, 1623277, 1623278]
def checkedParentColumns : List Nat := [1619039, 1619040, 1619041, 1619042]

def fieldVector (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment)
    (columns : List Nat) : Fin 4 -> Field :=
  fun lane => fieldAt assignment canonical (columns.getD lane.val 0)

/-- Typed semantic input read from exact artifact columns. This construction
does not itself grant producer authority to either digest vector. -/
def bindingInput (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment) : Input where
  headerBundle := fieldVector assignment canonical headerColumns
  instanceDigest := fieldVector assignment canonical instanceDigestColumns
  runningCount := 14
  checkedParentHandle := fieldVector assignment canonical checkedParentColumns

def headerBundleFields (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment) : List Field :=
  [fieldVector assignment canonical headerColumns ⟨0, by decide⟩,
   fieldVector assignment canonical headerColumns ⟨1, by decide⟩,
   fieldVector assignment canonical headerColumns ⟨2, by decide⟩,
   fieldVector assignment canonical headerColumns ⟨3, by decide⟩]

def instanceDigestFields (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment) : List Field :=
  [fieldVector assignment canonical instanceDigestColumns ⟨0, by decide⟩,
   fieldVector assignment canonical instanceDigestColumns ⟨1, by decide⟩,
   fieldVector assignment canonical instanceDigestColumns ⟨2, by decide⟩,
   fieldVector assignment canonical instanceDigestColumns ⟨3, by decide⟩]

def checkedParentFields (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment) : List Field :=
  [fieldVector assignment canonical checkedParentColumns ⟨0, by decide⟩,
   fieldVector assignment canonical checkedParentColumns ⟨1, by decide⟩,
   fieldVector assignment canonical checkedParentColumns ⟨2, by decide⟩,
   fieldVector assignment canonical checkedParentColumns ⟨3, by decide⟩]

/-- Artifact-backed state after the instance boundary call and the remaining
two instance-digest lanes. -/
def afterInstanceState (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment) : State :=
  absorbAll
    (callOutputState assignment canonical
      (Schedule.bindingCall instanceBoundaryIndex))
    [(instanceDigestFields assignment canonical).getD 2 (wordField 0),
     (instanceDigestFields assignment canonical).getD 3 (wordField 0)]

/-- Artifact-backed state after the running-count message and before the
checked-parent message. -/
def afterRunningCountState (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment) : State :=
  absorbAll
    (callOutputState assignment canonical
      (Schedule.bindingCall runningDomainIndex))
    [wordField 2, wordField 5, wordField 14]

/-- Exact state after the final parent-handle lane is absorbed. -/
def afterParentState (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment) : State :=
  absorbElem
    (callOutputState assignment canonical
      (Schedule.bindingCall parentPayloadIndex))
    ((checkedParentFields assignment canonical).getD 3 (wordField 0))

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

/-- An independently accepted call at a full cursor is exactly the boundary
permutation performed before the next element is overwritten. -/
private theorem absorbAcceptedFull
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (call : Poseidon2Call.Call)
    (accepted : TranscriptCertificate.CallAccepted call assignment)
    (value : Field) :
    absorbElem (callInputState assignment canonical call ⟨4, by decide⟩)
        value =
      absorbElem (callOutputState assignment canonical call) value := by
  have full : ¬
      (callInputState assignment canonical call ⟨4, by decide⟩).absorbed.val <
        rate := by
    change ¬ 4 < rate
    simp [rate]
  have room :
      (callOutputState assignment canonical call).absorbed.val < rate := by
    change 0 < rate
    simp [rate]
  unfold absorbElem
  rw [dif_neg full, dif_pos room]
  dsimp only
  rw [callAccepted_permute canonical one call ⟨4, by decide⟩ accepted]
  simp [callOutputState]

/-- Eager normalization of a full accepted call input reaches the exact
assignment-backed call output. -/
private theorem normalizeAcceptedFull
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (call : Poseidon2Call.Call)
    (accepted : TranscriptCertificate.CallAccepted call assignment) :
    normalizeFull
        (callInputState assignment canonical call ⟨4, by decide⟩) =
      callOutputState assignment canonical call := by
  unfold normalizeFull
  rw [if_pos (by rfl)]
  exact callAccepted_permute canonical one call ⟨4, by decide⟩ accepted

/-- The first two header words fill the terminal cursor and name exactly the
first binding-call input. -/
theorem headerBoundaryCallInput
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (pins : PinSchedule.Facts assignment) :
    absorbElem
        (absorbElem (initialState assignment canonical) (wordField 5))
        (wordField 11) =
      callInputState assignment canonical
        (Schedule.bindingCall headerBoundaryIndex) ⟨4, by decide⟩ := by
  have lengthField : fieldAt assignment canonical 1623288 = wordField 5 :=
    fieldAt_eq_wordField canonical pins.headerLength (by decide) (by decide)
  have tagField : fieldAt assignment canonical 1623287 = wordField 11 :=
    fieldAt_eq_wordField canonical pins.headerTag (by decide) (by decide)
  change
    { lanes := overwriteLane
        (overwriteLane (initialState assignment canonical).lanes
          2 (wordField 5))
        3 (wordField 11)
      absorbed := ⟨4, by decide⟩ } =
      callInputState assignment canonical
        (Schedule.bindingCall headerBoundaryIndex) ⟨4, by decide⟩
  apply stateExt
  · funext lane
    apply Fin.ext
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [initialState, initialColumns, overwriteLane, callInputState,
        Schedule.bindingCall, Schedule.bindingCalls,
        headerBoundaryIndex, Poseidon2Call.Call.columnMap, h,
        lengthField, tagField]
  · rfl

/-- After the first header boundary call, the four header fields exactly fill
the second call's input state. -/
theorem headerPayloadCallInput
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment) :
    absorbAll
        (callOutputState assignment canonical
          (Schedule.bindingCall headerBoundaryIndex))
        (headerBundleFields assignment canonical) =
      callInputState assignment canonical
        (Schedule.bindingCall headerPayloadIndex) ⟨4, by decide⟩ := by
  change
    absorbElem
      (absorbElem
        (absorbElem
          (absorbElem
            (callOutputState assignment canonical
              (Schedule.bindingCall headerBoundaryIndex))
            (fieldAt assignment canonical 1623283))
          (fieldAt assignment canonical 1623284))
        (fieldAt assignment canonical 1623285))
      (fieldAt assignment canonical 1623286) =
    callInputState assignment canonical
      (Schedule.bindingCall headerPayloadIndex) ⟨4, by decide⟩
  change
    { lanes := overwriteLane
        (overwriteLane
          (overwriteLane
            (overwriteLane
              (callOutputState assignment canonical
                (Schedule.bindingCall headerBoundaryIndex)).lanes
              0 (fieldAt assignment canonical 1623283))
            1 (fieldAt assignment canonical 1623284))
          2 (fieldAt assignment canonical 1623285))
        3 (fieldAt assignment canonical 1623286)
      absorbed := ⟨4, by decide⟩ } =
    callInputState assignment canonical
      (Schedule.bindingCall headerPayloadIndex) ⟨4, by decide⟩
  apply stateExt
  · funext lane
    apply Fin.ext
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [callOutputState, callInputState, fieldAt, overwriteLane,
        Schedule.bindingCall, Schedule.bindingCalls, headerBoundaryIndex,
        headerPayloadIndex, Poseidon2Call.Call.columnMap, h]
  · rfl

/-- The independently specified header append is exactly the two-call
artifact execution. -/
theorem afterHeader_refines
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : FPrimeFullHistoryTerminalPiCcsTranscript.Accepted assignment) :
    afterHeader (initialState assignment canonical)
        (bindingInput assignment canonical) =
      callOutputState assignment canonical
        (Schedule.bindingCall headerPayloadIndex) := by
  have pins := PinSchedule.facts canonical one accepted
  have boundaryAccepted :=
    DigestRounds.bindingCallAccepted accepted headerBoundaryIndex
  have payloadAccepted :=
    DigestRounds.bindingCallAccepted accepted headerPayloadIndex
  unfold afterHeader headerFields bindingInput
  change appendRaw (initialState assignment canonical)
      (wordField 11 :: headerBundleFields assignment canonical) = _
  unfold appendRaw appendRawLazy
  change normalizeFull
      (absorbAll
        (absorbElem
          (absorbElem (initialState assignment canonical) (wordField 5))
          (wordField 11))
        (headerBundleFields assignment canonical)) = _
  rw [headerBoundaryCallInput canonical pins]
  unfold headerBundleFields
  simp only [absorbAll]
  rw [absorbAcceptedFull canonical one
    (Schedule.bindingCall headerBoundaryIndex) boundaryAccepted]
  change normalizeFull
      (absorbAll
        (callOutputState assignment canonical
          (Schedule.bindingCall headerBoundaryIndex))
        (headerBundleFields assignment canonical)) = _
  rw [headerPayloadCallInput canonical]
  exact normalizeAcceptedFull canonical one
    (Schedule.bindingCall headerPayloadIndex) payloadAccepted

/-- Length, tag, and the first two digest lanes exactly fill the instance
boundary call. -/
theorem instanceBoundaryCallInput
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (pins : PinSchedule.Facts assignment) :
    absorbAll
        (callOutputState assignment canonical
          (Schedule.bindingCall headerPayloadIndex))
        [wordField 5, wordField 12,
         (instanceDigestFields assignment canonical).getD 0 (wordField 0),
         (instanceDigestFields assignment canonical).getD 1 (wordField 0)] =
      callInputState assignment canonical
        (Schedule.bindingCall instanceBoundaryIndex) ⟨4, by decide⟩ := by
  have lengthField : fieldAt assignment canonical 1624490 = wordField 5 :=
    fieldAt_eq_wordField canonical pins.instanceLength (by decide) (by decide)
  have tagField : fieldAt assignment canonical 1624489 = wordField 12 :=
    fieldAt_eq_wordField canonical pins.instanceTag (by decide) (by decide)
  change
    { lanes := overwriteLane
        (overwriteLane
          (overwriteLane
            (overwriteLane
              (callOutputState assignment canonical
                (Schedule.bindingCall headerPayloadIndex)).lanes
              0 (wordField 5))
            1 (wordField 12))
          2 (fieldAt assignment canonical 1623275))
        3 (fieldAt assignment canonical 1623276)
      absorbed := ⟨4, by decide⟩ } =
    callInputState assignment canonical
      (Schedule.bindingCall instanceBoundaryIndex) ⟨4, by decide⟩
  apply stateExt
  · funext lane
    apply Fin.ext
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [callOutputState, callInputState, overwriteLane,
        Schedule.bindingCall, Schedule.bindingCalls, headerPayloadIndex,
        instanceBoundaryIndex, Poseidon2Call.Call.columnMap, h,
        lengthField, tagField]
  · rfl

/-- The independently specified instance append reaches the exact artifact
state, conditional only on the interpreted digest wires—not on their producer
authority. -/
theorem afterInstance_refines
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : FPrimeFullHistoryTerminalPiCcsTranscript.Accepted assignment) :
    afterInstance (initialState assignment canonical)
        (bindingInput assignment canonical) =
      afterInstanceState assignment canonical := by
  have pins := PinSchedule.facts canonical one accepted
  have boundaryAccepted :=
    DigestRounds.bindingCallAccepted accepted instanceBoundaryIndex
  unfold afterInstance instanceFields
  rw [afterHeader_refines canonical one accepted]
  unfold bindingInput
  change appendRaw
      (callOutputState assignment canonical
        (Schedule.bindingCall headerPayloadIndex))
      (wordField 12 :: instanceDigestFields assignment canonical) = _
  unfold appendRaw appendRawLazy
  unfold instanceDigestFields
  change normalizeFull
      (absorbAll
        (absorbAll
          (callOutputState assignment canonical
            (Schedule.bindingCall headerPayloadIndex))
          [wordField 5, wordField 12,
           fieldAt assignment canonical 1623275,
           fieldAt assignment canonical 1623276])
        [fieldAt assignment canonical 1623277,
         fieldAt assignment canonical 1623278]) = _
  have boundaryInput :
      absorbAll
          (callOutputState assignment canonical
            (Schedule.bindingCall headerPayloadIndex))
          [wordField 5, wordField 12,
           fieldAt assignment canonical 1623275,
           fieldAt assignment canonical 1623276] =
        callInputState assignment canonical
          (Schedule.bindingCall instanceBoundaryIndex) ⟨4, by decide⟩ := by
    simpa [instanceDigestFields, fieldVector, instanceDigestColumns] using
      instanceBoundaryCallInput canonical pins
  rw [boundaryInput]
  simp only [absorbAll]
  rw [absorbAcceptedFull canonical one
    (Schedule.bindingCall instanceBoundaryIndex) boundaryAccepted]
  rfl

/-- The running-domain raw message fills the cursor left by the instance
digest and names exactly its accepted boundary call. -/
theorem runningDomainCallInput
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (pins : PinSchedule.Facts assignment) :
    absorbAll (afterInstanceState assignment canonical)
        [wordField 1, wordField 4] =
      callInputState assignment canonical
        (Schedule.bindingCall runningDomainIndex) ⟨4, by decide⟩ := by
  have lengthField : fieldAt assignment canonical 1625092 = wordField 1 :=
    fieldAt_eq_wordField canonical pins.runningDomainLength
      (by decide) (by decide)
  have tagField : fieldAt assignment canonical 1625091 = wordField 4 :=
    fieldAt_eq_wordField canonical pins.runningDomainTag
      (by decide) (by decide)
  unfold afterInstanceState instanceDigestFields
  change
    { lanes := overwriteLane
        (overwriteLane
          (overwriteLane
            (overwriteLane
              (callOutputState assignment canonical
                (Schedule.bindingCall instanceBoundaryIndex)).lanes
              0 (fieldAt assignment canonical 1623277))
            1 (fieldAt assignment canonical 1623278))
          2 (wordField 1))
        3 (wordField 4)
      absorbed := ⟨4, by decide⟩ } =
    callInputState assignment canonical
      (Schedule.bindingCall runningDomainIndex) ⟨4, by decide⟩
  apply stateExt
  · funext lane
    apply Fin.ext
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [callOutputState, callInputState, overwriteLane,
        Schedule.bindingCall, Schedule.bindingCalls, instanceBoundaryIndex,
        runningDomainIndex, Poseidon2Call.Call.columnMap, h,
        lengthField, tagField]
  · rfl

theorem afterRunningDomain_refines
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : FPrimeFullHistoryTerminalPiCcsTranscript.Accepted assignment) :
    afterRunningDomain (initialState assignment canonical)
        (bindingInput assignment canonical) =
      callOutputState assignment canonical
        (Schedule.bindingCall runningDomainIndex) := by
  have pins := PinSchedule.facts canonical one accepted
  have domainAccepted :=
    DigestRounds.bindingCallAccepted accepted runningDomainIndex
  unfold afterRunningDomain runningDomainFields
  rw [afterInstance_refines canonical one accepted]
  change appendRaw (afterInstanceState assignment canonical)
      [wordField 4] = _
  unfold appendRaw appendRawLazy
  change normalizeFull
      (absorbAll (afterInstanceState assignment canonical)
        [wordField 1, wordField 4]) = _
  rw [runningDomainCallInput canonical pins]
  exact normalizeAcceptedFull canonical one
    (Schedule.bindingCall runningDomainIndex) domainAccepted

/-- The count message is non-boundary absorption and reaches the exact
cursor-3 artifact state. -/
theorem afterRunningCount_refines
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : FPrimeFullHistoryTerminalPiCcsTranscript.Accepted assignment) :
    afterRunningCount (initialState assignment canonical)
        (bindingInput assignment canonical) =
      afterRunningCountState assignment canonical := by
  unfold afterRunningCount runningCountFields
  rw [afterRunningDomain_refines canonical one accepted]
  unfold bindingInput
  change appendRaw
      (callOutputState assignment canonical
        (Schedule.bindingCall runningDomainIndex))
      [wordField 5, wordField 14] = _
  rfl

/-- Parent raw length fills the cursor after `[5,14]`, producing the first
checked-parent boundary call. -/
theorem countParentBoundaryCallInput
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (pins : PinSchedule.Facts assignment) :
    absorbElem (afterRunningCountState assignment canonical) (wordField 5) =
      callInputState assignment canonical
        (Schedule.bindingCall countParentBoundaryIndex) ⟨4, by decide⟩ := by
  have countLength : fieldAt assignment canonical 1625695 = wordField 2 :=
    fieldAt_eq_wordField canonical pins.runningCountLength
      (by decide) (by decide)
  have countTag : fieldAt assignment canonical 1625693 = wordField 5 :=
    fieldAt_eq_wordField canonical pins.runningCountTag
      (by decide) (by decide)
  have countValue : fieldAt assignment canonical 1625694 = wordField 14 :=
    fieldAt_eq_wordField canonical pins.runningCount
      (by decide) (by decide)
  have parentLength : fieldAt assignment canonical 1625697 = wordField 5 :=
    fieldAt_eq_wordField canonical pins.parentLength
      (by decide) (by decide)
  unfold afterRunningCountState
  change
    { lanes := overwriteLane
        (overwriteLane
          (overwriteLane
            (overwriteLane
              (callOutputState assignment canonical
                (Schedule.bindingCall runningDomainIndex)).lanes
              0 (wordField 2))
            1 (wordField 5))
          2 (wordField 14))
        3 (wordField 5)
      absorbed := ⟨4, by decide⟩ } =
    callInputState assignment canonical
      (Schedule.bindingCall countParentBoundaryIndex) ⟨4, by decide⟩
  apply stateExt
  · funext lane
    apply Fin.ext
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [callOutputState, callInputState, overwriteLane,
        Schedule.bindingCall, Schedule.bindingCalls, runningDomainIndex,
        countParentBoundaryIndex, Poseidon2Call.Call.columnMap, h,
        countLength, countTag, countValue, parentLength]
  · rfl

/-- After the count/parent boundary call, tag `13` and the first three parent
lanes exactly fill the final binding call. -/
theorem parentPayloadCallInput
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (pins : PinSchedule.Facts assignment) :
    absorbAll
        (callOutputState assignment canonical
          (Schedule.bindingCall countParentBoundaryIndex))
        [wordField 13,
         (checkedParentFields assignment canonical).getD 0 (wordField 0),
         (checkedParentFields assignment canonical).getD 1 (wordField 0),
         (checkedParentFields assignment canonical).getD 2 (wordField 0)] =
      callInputState assignment canonical
        (Schedule.bindingCall parentPayloadIndex) ⟨4, by decide⟩ := by
  have parentTag : fieldAt assignment canonical 1625696 = wordField 13 :=
    fieldAt_eq_wordField canonical pins.parentTag (by decide) (by decide)
  unfold checkedParentFields fieldVector checkedParentColumns
  rw [countParentBoundaryCall_eq, parentPayloadCall_eq]
  change
    { lanes := overwriteLane
        (overwriteLane
          (overwriteLane
            (overwriteLane
              (callOutputState assignment canonical
                countParentBoundaryCall).lanes
              0 (wordField 13))
            1 (fieldAt assignment canonical 1619039))
          2 (fieldAt assignment canonical 1619040))
        3 (fieldAt assignment canonical 1619041)
      absorbed := ⟨4, by decide⟩ } =
    callInputState assignment canonical parentPayloadCall ⟨4, by decide⟩
  rw [← parentTag]
  apply stateExt
  · funext lane
    apply Fin.ext
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [overwriteLane, h, countParentOutputLane,
        parentPayloadInputLane]
  · rfl

/-- Complete five-message binding refinement. The result is transcript-
correct for the interpreted vectors; producer authority remains a distinct
premise for the later end-to-end theorem. -/
theorem bindingRun_refines
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : FPrimeFullHistoryTerminalPiCcsTranscript.Accepted assignment) :
    run (initialState assignment canonical)
        (bindingInput assignment canonical) =
      afterParentState assignment canonical := by
  have pins := PinSchedule.facts canonical one accepted
  have countBoundaryAccepted :=
    DigestRounds.bindingCallAccepted accepted countParentBoundaryIndex
  have parentPayloadAccepted :=
    DigestRounds.bindingCallAccepted accepted parentPayloadIndex
  unfold run parentHandleFields
  rw [afterRunningCount_refines canonical one accepted]
  unfold bindingInput
  change appendRaw (afterRunningCountState assignment canonical)
      (wordField 13 :: checkedParentFields assignment canonical) = _
  unfold appendRaw appendRawLazy
  change normalizeFull
      (absorbAll
        (absorbElem (afterRunningCountState assignment canonical)
          (wordField 5))
        (wordField 13 :: checkedParentFields assignment canonical)) = _
  rw [countParentBoundaryCallInput canonical pins]
  unfold checkedParentFields
  simp only [absorbAll]
  rw [absorbAcceptedFull canonical one
    (Schedule.bindingCall countParentBoundaryIndex) countBoundaryAccepted]
  have payloadInput :
      absorbAll
          (callOutputState assignment canonical
            (Schedule.bindingCall countParentBoundaryIndex))
          [wordField 13,
           fieldAt assignment canonical 1619039,
           fieldAt assignment canonical 1619040,
           fieldAt assignment canonical 1619041] =
        callInputState assignment canonical
          (Schedule.bindingCall parentPayloadIndex) ⟨4, by decide⟩ := by
    simpa [checkedParentFields, fieldVector, checkedParentColumns] using
      parentPayloadCallInput canonical pins
  change normalizeFull
      (absorbAll
        (absorbAll
          (callOutputState assignment canonical
            (Schedule.bindingCall countParentBoundaryIndex))
          [wordField 13,
           fieldAt assignment canonical 1619039,
           fieldAt assignment canonical 1619040,
           fieldAt assignment canonical 1619041])
        [fieldAt assignment canonical 1619042]) = _
  rw [payloadInput]
  simp only [absorbAll]
  rw [absorbAcceptedFull canonical one
    (Schedule.bindingCall parentPayloadIndex) parentPayloadAccepted]
  rfl

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.ScheduleRefinement
