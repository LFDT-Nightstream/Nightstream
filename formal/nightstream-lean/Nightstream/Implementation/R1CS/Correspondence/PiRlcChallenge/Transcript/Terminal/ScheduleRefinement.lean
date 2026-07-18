import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.DigestRounds
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.PinSchedule

/-!
State-level refinement of the complete terminal `Pi_RLC` transcript schedule.

Assurance tier: implementation/R1CS correspondence. This module proves that
the independently accepted Poseidon2 calls and constant equations compose as
one execution of the independently specified overwrite transcript machine.

Owns: the artifact interpretation of the state entering scalar zero; the
scalar-zero entry transition; predecessor-state selection for later scalars;
the shared successor entry/block-zero boundary; all four digest transitions;
and state continuity between adjacent scalar coordinates.

Does not own: why the initial columns are the state reached by the preceding
`Pi_CCS` transcript, canonical lane decomposition, rejection selection,
coefficient assembly, Rust conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: neither generated row order nor a carried state digest is
accepted as a transcript. Every overwrite word is decoded from accepted pin
equations, every permutation is replayed through the independent Poseidon2
interpreter, and every preserved lane is connected explicitly.

| Protocol | Phase | Constraint family | Mathematical obligation |
|---|---|---|---|
| `Pi_RLC` | scalar 0 input | transcript state | interpret the eight post-`Pi_CCS` columns at cursor 2 |
| `Pi_RLC` | scalar entry | domain separation | execute raw pair `[0, rho]` with exact overwrite/cursor semantics |
| `Pi_RLC` | block 0 | shared boundary and digest | handle the scalar-zero two-call shape and successor one-boundary shape |
| `Pi_RLC` | blocks 1-3 | digest transitions | execute `[1, rho+b]`, squeeze, and Poseidon2 for every scalar |
| `Pi_RLC` | scalar continuity | successor state | block-3 output of `rho` is the input state of `rho+1` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.ScheduleRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal

abbrev CanonicalAssignment (assignment : Nat -> Nat) :=
  forall column, assignment column < goldilocksP

def zeroScalar : Fin ScalarRows.scalarCount := ⟨0, by decide⟩

/-- The verifier state immediately before terminal scalar zero. Connecting it
to the complete preceding `Pi_CCS` transcript remains a separate obligation. -/
def initialState (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment) : State where
  lanes := fun lane => fieldAt assignment canonical
    ([2553435, 2553436, 2554645, 2554646,
      2554647, 2554648, 2554649, 2554650].getD lane.val 0)
  absorbed := ⟨2, by decide⟩

/-- Artifact state after scalar zero crosses its entry boundary and overwrites
lane zero with the scalar coordinate. -/
def zeroAfterEnterState (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment) : State where
  lanes := overwriteLane
    (callOutputState assignment canonical
      (Schedule.entryBoundaryCall zeroScalar)).lanes
    0 (fieldAt assignment canonical
      (PinSchedule.Artifact.coordinateColumn zeroScalar))
  absorbed := ⟨1, by decide⟩

/-- Artifact state after digest block zero for scalar `rho`. -/
def block0State (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) : State :=
  callOutputState assignment canonical (Schedule.block0DigestCall rho)

/-- Artifact state after digest block one for scalar `rho`. -/
def block1State (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) : State :=
  callOutputState assignment canonical (Schedule.block1DigestCall rho)

/-- Artifact state after digest block two for scalar `rho`. -/
def block2State (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) : State :=
  callOutputState assignment canonical (Schedule.block2DigestCall rho)

/-- Artifact state after digest block three for scalar `rho`. -/
def block3State (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) : State :=
  callOutputState assignment canonical (Schedule.block3DigestCall rho)

/-- State entering digest block `block + 1`. The three cases are exactly the
outputs of blocks zero, one, and two; no carried digest is trusted. -/
def priorLaterState (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) : State :=
  { lanes :=
      if zero : block.val = 0 then
        (block0State assignment canonical rho).lanes
      else if one : block.val = 1 then
        (block1State assignment canonical rho).lanes
      else
        (block2State assignment canonical rho).lanes
    absorbed := ⟨0, by decide⟩ }

/-- Exact artifact state reached by digest block `block + 1`. -/
def laterBlockState (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) : State :=
  callOutputState assignment canonical (Schedule.laterDigestCall rho block)

def previousScalar (rho : Fin ScalarRows.scalarCount)
    (nonzero : rho.val ≠ 0) : Fin ScalarRows.scalarCount :=
  ⟨rho.val - 1, by
    have rhoLt := rho.isLt
    simp only [ScalarRows.scalarCount] at rhoLt ⊢
    omega⟩

def nextScalar (rho : Fin ScalarRows.scalarCount)
    (hasNext : rho.val + 1 < ScalarRows.scalarCount) :
    Fin ScalarRows.scalarCount :=
  ⟨rho.val + 1, hasNext⟩

/-- Scalar zero starts at the post-`Pi_CCS` state. Every later scalar starts
at the preceding scalar's block-three output. -/
def stateBeforeScalar (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) : State :=
  if zero : rho.val = 0 then initialState assignment canonical
  else block3State assignment canonical (previousScalar rho zero)

/-- Artifact-backed state after the scalar-domain transition. Scalar zero has
already crossed a Poseidon2 boundary; successors have not. -/
def afterEnterState (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) : State :=
  if zero : rho.val = 0 then zeroAfterEnterState assignment canonical
  else enterScalar (stateBeforeScalar assignment canonical rho) rho.val

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

theorem stateBeforeScalar_zero
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment) :
    stateBeforeScalar assignment canonical zeroScalar =
      initialState assignment canonical := by
  simp [stateBeforeScalar, zeroScalar]

theorem stateBeforeScalar_nonzero
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) (nonzero : rho.val ≠ 0) :
    stateBeforeScalar assignment canonical rho =
      block3State assignment canonical (previousScalar rho nonzero) := by
  simp [stateBeforeScalar, nonzero]

/-- The first two scalar-domain words fill the initial cursor and produce
exactly the state named by scalar zero's entry-boundary call. -/
theorem entryBoundaryCallInput_zero
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (pins : PinSchedule.Facts assignment) :
    absorbElem
        (absorbElem (initialState assignment canonical) (wordField 2))
        (wordField 0) =
      callInputState assignment canonical
        (Schedule.entryBoundaryCall zeroScalar) ⟨4, by decide⟩ := by
  have entryLength : assignment 2554651 = 2 := by
    simpa [PinSchedule.Artifact.entryLengthColumn, zeroScalar] using
      pins.entryLength zeroScalar
  have entryDomain : assignment 2554652 = 0 := by
    simpa [PinSchedule.Artifact.entryDomainColumn,
      PinSchedule.Artifact.entryLengthColumn, zeroScalar] using
      pins.entryDomain zeroScalar
  change
    { lanes := overwriteLane
        (overwriteLane (initialState assignment canonical).lanes
          2 (wordField 2))
        3 (wordField 0)
      absorbed := ⟨4, by decide⟩ } =
      callInputState assignment canonical
        (Schedule.entryBoundaryCall zeroScalar) ⟨4, by decide⟩
  apply stateExt
  · funext lane
    apply Fin.ext
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [initialState, overwriteLane, callInputState, fieldAt,
        Schedule.entryBoundaryCall, zeroScalar,
        Poseidon2Call.Call.columnMap, wordField, fieldValue, u64Modulus,
        goldilocksP, h, entryLength, entryDomain]
  · rfl

/-- Exact pins and independent Poseidon2 acceptance refine scalar zero's
domain transition. -/
theorem enterScalar_zero_refines
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    enterScalar (initialState assignment canonical) 0 =
      zeroAfterEnterState assignment canonical := by
  have pins := PinSchedule.facts canonical one accepted
  have callAccepted := DigestRounds.entryBoundaryCallAccepted accepted zeroScalar
  unfold enterScalar appendRawPair
  rw [entryBoundaryCallInput_zero canonical pins]
  change
    { lanes := overwriteLane
        (permute (callInputState assignment canonical
          (Schedule.entryBoundaryCall zeroScalar) ⟨4, by decide⟩)).lanes
        0 (wordField 0)
      absorbed := ⟨1, by decide⟩ } =
      zeroAfterEnterState assignment canonical
  rw [callAccepted_permute canonical one
    (Schedule.entryBoundaryCall zeroScalar) ⟨4, by decide⟩ callAccepted]
  have coordinatePin : assignment 2554653 = 0 := by
    simpa [PinSchedule.Artifact.coordinateColumn,
      PinSchedule.Artifact.entryLengthColumn, zeroScalar] using
      pins.entryCoordinate zeroScalar
  have coordinate : fieldAt assignment canonical 2554653 = wordField 0 :=
    fieldAt_eq_wordField canonical coordinatePin (by decide) (by decide)
  unfold zeroAfterEnterState
  simp only [PinSchedule.Artifact.coordinateColumn,
    PinSchedule.Artifact.entryLengthColumn, zeroScalar, ↓reduceIte]
  rw [← coordinate]

/-- All scalar-domain transitions match the independent machine. For
successors this is definitionally pre-bound to the preceding block-three
state; scalar zero requires the accepted entry permutation above. -/
theorem enterScalar_refines
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    enterScalar (stateBeforeScalar assignment canonical rho) rho.val =
      afterEnterState assignment canonical rho := by
  by_cases zero : rho.val = 0
  · have rhoEq : rho = zeroScalar := Fin.ext zero
    subst rho
    simpa [afterEnterState, zeroScalar, stateBeforeScalar_zero] using
      enterScalar_zero_refines canonical one accepted
  · simp [afterEnterState, zero]

/-- Scalar zero's block-zero raw pair fills its cursor and produces exactly
the input state named by the scalar-zero-only full-cursor call. -/
theorem scalar0Block0FullCursorCallInput
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (pins : PinSchedule.Facts assignment) :
    appendRawPair (zeroAfterEnterState assignment canonical) 1 0 =
      callInputState assignment canonical
        Schedule.scalar0Block0FullCursorCall ⟨4, by decide⟩ := by
  have blockLength : assignment 2555255 = 2 := by
    simpa [PinSchedule.Artifact.block0LengthColumn, zeroScalar] using
      pins.block0Length zeroScalar
  have blockDomain : assignment 2555256 = 1 := by
    simpa [PinSchedule.Artifact.block0DomainColumn,
      PinSchedule.Artifact.block0LengthColumn, zeroScalar] using
      pins.block0Domain zeroScalar
  have blockCounter : assignment 2555257 = 0 := by
    simpa [PinSchedule.Artifact.block0CounterColumn, zeroScalar] using
      pins.block0Counter zeroScalar
  change
    { lanes := overwriteLane
        (overwriteLane
          (overwriteLane (zeroAfterEnterState assignment canonical).lanes
            1 (wordField 2))
          2 (wordField 1))
        3 (wordField 0)
      absorbed := ⟨4, by decide⟩ } =
      callInputState assignment canonical
        Schedule.scalar0Block0FullCursorCall ⟨4, by decide⟩
  apply stateExt
  · funext lane
    apply Fin.ext
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [zeroAfterEnterState, overwriteLane, callInputState,
        callOutputState, fieldAt, Schedule.entryBoundaryCall,
        Schedule.scalar0Block0FullCursorCall, zeroScalar,
        PinSchedule.Artifact.coordinateColumn,
        PinSchedule.Artifact.entryLengthColumn,
        Poseidon2Call.Call.columnMap, wordField, fieldValue, u64Modulus,
        goldilocksP, h, blockLength, blockDomain, blockCounter]
  · rfl

/-- The scalar-zero full-cursor output plus squeeze word is exactly the input
state named by block zero's digest call. -/
theorem scalar0Block0DigestCallInput
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (pins : PinSchedule.Facts assignment)
    (accepted : TranscriptCertificate.CallAccepted
      Schedule.scalar0Block0FullCursorCall assignment) :
    absorbElem
        (callInputState assignment canonical
          Schedule.scalar0Block0FullCursorCall ⟨4, by decide⟩)
        (wordField 1) =
      callInputState assignment canonical
        (Schedule.block0DigestCall zeroScalar) ⟨1, by decide⟩ := by
  have squeeze : assignment 2555258 = 1 := by
    simpa [PinSchedule.Artifact.block0SqueezeColumn,
      PinSchedule.Artifact.block0CounterColumn, zeroScalar] using
      pins.block0Squeeze zeroScalar
  change
    { lanes := overwriteLane
        (permute (callInputState assignment canonical
          Schedule.scalar0Block0FullCursorCall ⟨4, by decide⟩)).lanes
        0 (wordField 1)
      absorbed := ⟨1, by decide⟩ } =
      callInputState assignment canonical
        (Schedule.block0DigestCall zeroScalar) ⟨1, by decide⟩
  rw [callAccepted_permute canonical one
    Schedule.scalar0Block0FullCursorCall ⟨4, by decide⟩ accepted]
  apply stateExt
  · funext lane
    apply Fin.ext
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [overwriteLane, callInputState, callOutputState, fieldAt,
        Schedule.scalar0Block0FullCursorCall, Schedule.block0DigestCall,
        Schedule.block0InputColumns, zeroScalar,
        Poseidon2Call.Call.columnMap, wordField, fieldValue, u64Modulus,
        goldilocksP, h, squeeze]
  · rfl

/-- Scalar zero's independent transcript execution reaches the exact block
zero artifact state. -/
theorem digestBlock0_zero_refines
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    (digestBlock (zeroAfterEnterState assignment canonical) 0).1 =
      block0State assignment canonical zeroScalar := by
  have pins := PinSchedule.facts canonical one accepted
  have fullAccepted :=
    DigestRounds.scalar0Block0FullCursorCallAccepted accepted
  have digestAccepted :=
    DigestRounds.block0DigestCallAccepted accepted zeroScalar
  change
    permute
        (absorbElem
          (appendRawPair (zeroAfterEnterState assignment canonical) 1 0)
          (wordField 1)) =
      block0State assignment canonical zeroScalar
  rw [scalar0Block0FullCursorCallInput canonical pins]
  rw [scalar0Block0DigestCallInput canonical one pins fullAccepted]
  simpa [block0State] using
    callAccepted_permute canonical one
      (Schedule.block0DigestCall zeroScalar) ⟨1, by decide⟩ digestAccepted

/-- For every nonzero scalar, the scalar-domain words followed by block
zero's length word fill exactly the shared entry/block-zero boundary call. -/
theorem successorEntryBoundaryCallInput
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (pins : PinSchedule.Facts assignment)
    (rho : Fin ScalarRows.scalarCount) (nonzero : rho.val ≠ 0) :
    absorbElem (afterEnterState assignment canonical rho) (wordField 2) =
      callInputState assignment canonical
        (Schedule.entryBoundaryCall rho) ⟨4, by decide⟩ := by
  have rhoLt : rho.val < 15 := by
    simpa [ScalarRows.scalarCount] using rho.isLt
  have lengthPin : assignment (Schedule.domainBase rho) = 2 := by
    simpa [PinSchedule.Artifact.entryLengthColumn, nonzero] using
      pins.entryLength rho
  have domainPin : assignment (Schedule.domainBase rho + 1) = 0 := by
    simpa [PinSchedule.Artifact.entryDomainColumn,
      PinSchedule.Artifact.entryLengthColumn, nonzero] using
      pins.entryDomain rho
  have coordinatePin : assignment (Schedule.domainBase rho + 2) = rho.val := by
    simpa [PinSchedule.Artifact.coordinateColumn,
      PinSchedule.Artifact.entryLengthColumn, nonzero] using
      pins.entryCoordinate rho
  have blockLengthPin : assignment (Schedule.domainBase rho + 4) = 2 := by
    simpa [PinSchedule.Artifact.block0LengthColumn, nonzero] using
      pins.block0Length rho
  simp only [Schedule.domainBase, Schedule.scalarColumnStride] at lengthPin
  simp only [Schedule.domainBase, Schedule.scalarColumnStride] at domainPin
  simp only [Schedule.domainBase, Schedule.scalarColumnStride] at coordinatePin
  simp only [Schedule.domainBase, Schedule.scalarColumnStride] at blockLengthPin
  have rhoLtU64 : rho.val < u64Modulus := by
    have bound : 15 < u64Modulus := by decide
    omega
  have rhoLtField : rho.val < goldilocksP := by
    have bound : 15 < goldilocksP := by decide
    omega
  rw [show afterEnterState assignment canonical rho =
      enterScalar (stateBeforeScalar assignment canonical rho) rho.val by
    simp [afterEnterState, nonzero]]
  rw [stateBeforeScalar_nonzero canonical rho nonzero]
  change
    { lanes := overwriteLane
        (overwriteLane
          (overwriteLane
            (overwriteLane (block3State assignment canonical
              (previousScalar rho nonzero)).lanes 0 (wordField 2))
            1 (wordField 0))
          2 (wordField rho.val))
        3 (wordField 2)
      absorbed := ⟨4, by decide⟩ } =
      callInputState assignment canonical
        (Schedule.entryBoundaryCall rho) ⟨4, by decide⟩
  apply stateExt
  · funext lane
    apply Fin.ext
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [block3State, overwriteLane, callInputState, callOutputState,
        fieldAt, Schedule.entryBoundaryCall, Schedule.block3DigestCall,
        Schedule.laterDigestCall, Schedule.laterBlockPinBase,
        Schedule.previousHighColumns, Schedule.entryFirstAllocated,
        Schedule.domainBase, Schedule.scalarColumnStride,
        previousScalar, Poseidon2Call.Call.columnMap, nonzero, h,
        wordField, fieldValue, u64Modulus, goldilocksP,
        Nat.mod_eq_of_lt rhoLtU64, Nat.mod_eq_of_lt rhoLtField,
        lengthPin, domainPin, coordinatePin, blockLengthPin] <;>
      congr 1 <;> omega
  · rfl

/-- The shared successor boundary output, block-zero domain/counter words,
and squeeze word form exactly the block-zero digest input. -/
theorem successorBlock0DigestCallInput
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (pins : PinSchedule.Facts assignment)
    (rho : Fin ScalarRows.scalarCount) (nonzero : rho.val ≠ 0)
    (accepted : TranscriptCertificate.CallAccepted
      (Schedule.entryBoundaryCall rho) assignment) :
    absorbElem
        (absorbElem
          (absorbElem
            (callInputState assignment canonical
              (Schedule.entryBoundaryCall rho) ⟨4, by decide⟩)
            (wordField 1))
          (wordField rho.val))
        (wordField 1) =
      callInputState assignment canonical
        (Schedule.block0DigestCall rho) ⟨3, by decide⟩ := by
  have rhoLt : rho.val < 15 := by
    simpa [ScalarRows.scalarCount] using rho.isLt
  have domainPin : assignment (Schedule.domainBase rho + 5) = 1 := by
    simpa [PinSchedule.Artifact.block0DomainColumn,
      PinSchedule.Artifact.block0LengthColumn, nonzero] using
      pins.block0Domain rho
  have counterPin :
      assignment (Schedule.entryFirstAllocated rho + 600) = rho.val := by
    simpa [PinSchedule.Artifact.block0CounterColumn, nonzero] using
      pins.block0Counter rho
  have squeezePin :
      assignment (Schedule.entryFirstAllocated rho + 601) = 1 := by
    simpa [PinSchedule.Artifact.block0SqueezeColumn,
      PinSchedule.Artifact.block0CounterColumn, nonzero] using
      pins.block0Squeeze rho
  simp only [Schedule.entryFirstAllocated, Schedule.domainBase,
    Schedule.scalarColumnStride, nonzero, ↓reduceIte] at domainPin
  simp only [Schedule.entryFirstAllocated, Schedule.domainBase,
    Schedule.scalarColumnStride, nonzero, ↓reduceIte] at counterPin
  simp only [Schedule.entryFirstAllocated, Schedule.domainBase,
    Schedule.scalarColumnStride, nonzero, ↓reduceIte] at squeezePin
  have rhoLtU64 : rho.val < u64Modulus := by
    have bound : 15 < u64Modulus := by decide
    omega
  have rhoLtField : rho.val < goldilocksP := by
    have bound : 15 < goldilocksP := by decide
    omega
  change
    { lanes := overwriteLane
        (overwriteLane
          (overwriteLane
            (permute (callInputState assignment canonical
              (Schedule.entryBoundaryCall rho) ⟨4, by decide⟩)).lanes
            0 (wordField 1))
          1 (wordField rho.val))
        2 (wordField 1)
      absorbed := ⟨3, by decide⟩ } =
      callInputState assignment canonical
        (Schedule.block0DigestCall rho) ⟨3, by decide⟩
  rw [callAccepted_permute canonical one (Schedule.entryBoundaryCall rho)
    ⟨4, by decide⟩ accepted]
  apply stateExt
  · funext lane
    apply Fin.ext
    rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
      simp [overwriteLane, callInputState, callOutputState, fieldAt,
        Schedule.entryBoundaryCall, Schedule.block0DigestCall,
        Schedule.block0InputColumns, Schedule.entryFirstAllocated,
        Schedule.domainBase, Schedule.scalarColumnStride,
        Poseidon2Call.Call.columnMap, nonzero, h,
        wordField, fieldValue, u64Modulus, goldilocksP,
        Nat.mod_eq_of_lt rhoLtU64, Nat.mod_eq_of_lt rhoLtField,
        domainPin, counterPin, squeezePin] <;>
      omega
  · rfl

/-- Every nonzero scalar's block-zero transition reaches its exact artifact
state through the shared boundary call. -/
theorem digestBlock0_successor_refines
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) (nonzero : rho.val ≠ 0) :
    (digestBlock (afterEnterState assignment canonical rho) rho.val).1 =
      block0State assignment canonical rho := by
  have pins := PinSchedule.facts canonical one accepted
  have boundaryAccepted := DigestRounds.entryBoundaryCallAccepted accepted rho
  have digestAccepted := DigestRounds.block0DigestCallAccepted accepted rho
  change
    permute
        (absorbElem
          (appendRawPair (afterEnterState assignment canonical rho)
            1 rho.val)
          (wordField 1)) =
      block0State assignment canonical rho
  unfold appendRawPair
  rw [successorEntryBoundaryCallInput canonical pins rho nonzero]
  rw [successorBlock0DigestCallInput canonical one pins rho nonzero
    boundaryAccepted]
  simpa [block0State] using
    callAccepted_permute canonical one (Schedule.block0DigestCall rho)
      ⟨3, by decide⟩ digestAccepted

/-- Complete block-zero state refinement for all fifteen scalar coordinates. -/
theorem digestBlock0_refines
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    (digestBlock (afterEnterState assignment canonical rho) rho.val).1 =
      block0State assignment canonical rho := by
  by_cases zero : rho.val = 0
  · have rhoEq : rho = zeroScalar := Fin.ext zero
    subst rho
    simpa [afterEnterState, zeroScalar] using
      digestBlock0_zero_refines canonical one accepted
  · exact digestBlock0_successor_refines canonical one accepted rho zero

/-- The prior digest state and the four verifier-pinned words form exactly
the scheduled input for each of digest blocks one through three. -/
theorem laterDigestCallInput
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (pins : PinSchedule.Facts assignment)
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) :
    absorbElem
        (appendRawPair (priorLaterState assignment canonical rho block)
          1 (rho.val + block.val + 1))
        (wordField 1) =
      callInputState assignment canonical
        (Schedule.laterDigestCall rho block) ⟨4, by decide⟩ := by
  have counterLt : rho.val + block.val + 1 < 18 := by
    have rhoLt := rho.isLt
    have blockLt := block.isLt
    simp only [ScalarRows.scalarCount] at rhoLt
    omega
  have lengthField :
      fieldAt assignment canonical
          (Schedule.laterBlockPinBase rho block) = wordField 2 :=
    fieldAt_eq_wordField canonical (pins.laterLength rho block)
      (by decide) (by decide)
  have domainField :
      fieldAt assignment canonical
          (Schedule.laterBlockPinBase rho block + 1) = wordField 1 :=
    fieldAt_eq_wordField canonical (pins.laterDomain rho block)
      (by decide) (by decide)
  have counterField :
      fieldAt assignment canonical
          (Schedule.laterBlockPinBase rho block + 2) =
        wordField (rho.val + block.val + 1) :=
    fieldAt_eq_wordField canonical (pins.laterCounter rho block)
      (by
        have bound : 18 < u64Modulus := by decide
        omega)
      (by
        have bound : 18 < goldilocksP := by decide
        omega)
  have squeezeField :
      fieldAt assignment canonical
          (Schedule.laterBlockPinBase rho block + 3) = wordField 1 :=
    fieldAt_eq_wordField canonical (pins.laterSqueeze rho block)
      (by decide) (by decide)
  unfold appendRawPair
  change
    { lanes := overwriteLane
        (overwriteLane
          (overwriteLane
            (overwriteLane (priorLaterState assignment canonical rho block).lanes
              0 (wordField 2))
            1 (wordField 1))
          2 (wordField (rho.val + block.val + 1)))
        3 (wordField 1)
      absorbed := ⟨4, by decide⟩ } =
      callInputState assignment canonical
        (Schedule.laterDigestCall rho block) ⟨4, by decide⟩
  have blockCases : block.val = 0 ∨ block.val = 1 ∨ block.val = 2 := by
    have blockLt := block.isLt
    omega
  rcases blockCases with zero | one | two <;>
    apply stateExt
  all_goals
    first
    | rfl
    | (funext lane
       apply Fin.ext
       rcases laneValueCases lane with h | h | h | h | h | h | h | h <;>
         simp_all [priorLaterState, block0State, block1State, block2State,
           overwriteLane, callInputState, callOutputState,
           Schedule.laterDigestCall, Schedule.block0DigestCall,
           Schedule.block1DigestCall, Schedule.block2DigestCall,
           Schedule.block0InputColumns, Schedule.entryFirstAllocated,
           Schedule.priorDigestHighBase, Schedule.domainBase,
           Schedule.scalarColumnStride, Poseidon2Call.Call.columnMap] <;>
         simp [fieldAt, Schedule.laterBlockPinBase,
           Schedule.scalarColumnStride] <;>
         congr 1 <;> omega)

/-- Each later digest transition replays the exact accepted Poseidon2 call,
starting from the output state of the preceding digest block. -/
theorem laterDigestBlock_refines
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) :
    (digestBlock (priorLaterState assignment canonical rho block)
      (rho.val + block.val + 1)).1 =
      laterBlockState assignment canonical rho block := by
  have pins := PinSchedule.facts canonical one accepted
  have digestAccepted :=
    DigestRounds.laterDigestCallAccepted accepted rho block
  change
    permute
        (absorbElem
          (appendRawPair (priorLaterState assignment canonical rho block)
            1 (rho.val + block.val + 1))
          (wordField 1)) =
      laterBlockState assignment canonical rho block
  rw [laterDigestCallInput canonical pins rho block]
  simpa [laterBlockState] using
    callAccepted_permute canonical one
      (Schedule.laterDigestCall rho block) ⟨4, by decide⟩ digestAccepted

/-- Named block-one refinement for the phase tree. -/
theorem digestBlock1_refines
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    (digestBlock (block0State assignment canonical rho) (rho.val + 1)).1 =
      block1State assignment canonical rho := by
  simpa [priorLaterState, laterBlockState, block1State,
    Schedule.block1DigestCall] using
    laterDigestBlock_refines canonical one accepted rho ⟨0, by decide⟩

/-- Named block-two refinement for the phase tree. -/
theorem digestBlock2_refines
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    (digestBlock (block1State assignment canonical rho) (rho.val + 2)).1 =
      block2State assignment canonical rho := by
  simpa [priorLaterState, laterBlockState, block2State,
    Schedule.block2DigestCall] using
    laterDigestBlock_refines canonical one accepted rho ⟨1, by decide⟩

/-- Named block-three refinement for the phase tree. -/
theorem digestBlock3_refines
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    (digestBlock (block2State assignment canonical rho) (rho.val + 3)).1 =
      block3State assignment canonical rho := by
  simpa [priorLaterState, laterBlockState, block3State,
    Schedule.block3DigestCall] using
    laterDigestBlock_refines canonical one accepted rho ⟨2, by decide⟩

theorem previousScalar_nextScalar
    (rho : Fin ScalarRows.scalarCount)
    (hasNext : rho.val + 1 < ScalarRows.scalarCount) :
    previousScalar (nextScalar rho hasNext) (by
      simp [nextScalar]) = rho := by
  apply Fin.ext
  simp [previousScalar, nextScalar]

/-- Block three of a scalar is exactly the state consumed by the next scalar.
This closes every internal edge of the fifteen-scalar transcript chain. -/
theorem stateBeforeScalar_next
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount)
    (hasNext : rho.val + 1 < ScalarRows.scalarCount) :
    stateBeforeScalar assignment canonical (nextScalar rho hasNext) =
      block3State assignment canonical rho := by
  have nonzero : (nextScalar rho hasNext).val ≠ 0 := by
    simp [nextScalar]
  rw [stateBeforeScalar_nonzero canonical (nextScalar rho hasNext) nonzero]
  rw [previousScalar_nextScalar rho hasNext]

/-- One auditable package for the complete terminal state schedule. It starts
from `initialState`; binding that state to the preceding `Pi_CCS` transcript is
deliberately outside this theorem. -/
structure StateScheduleRefined
    (assignment : Nat -> Nat)
    (canonical : CanonicalAssignment assignment) : Prop where
  enter : forall rho : Fin ScalarRows.scalarCount,
    enterScalar (stateBeforeScalar assignment canonical rho) rho.val =
      afterEnterState assignment canonical rho
  block0 : forall rho : Fin ScalarRows.scalarCount,
    (digestBlock (afterEnterState assignment canonical rho) rho.val).1 =
      block0State assignment canonical rho
  block1 : forall rho : Fin ScalarRows.scalarCount,
    (digestBlock (block0State assignment canonical rho) (rho.val + 1)).1 =
      block1State assignment canonical rho
  block2 : forall rho : Fin ScalarRows.scalarCount,
    (digestBlock (block1State assignment canonical rho) (rho.val + 2)).1 =
      block2State assignment canonical rho
  block3 : forall rho : Fin ScalarRows.scalarCount,
    (digestBlock (block2State assignment canonical rho) (rho.val + 3)).1 =
      block3State assignment canonical rho
  scalarEdge : forall (rho : Fin ScalarRows.scalarCount)
      (hasNext : rho.val + 1 < ScalarRows.scalarCount),
    stateBeforeScalar assignment canonical (nextScalar rho hasNext) =
      block3State assignment canonical rho

theorem stateScheduleRefined
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    StateScheduleRefined assignment canonical :=
  { enter := enterScalar_refines canonical one accepted
    block0 := digestBlock0_refines canonical one accepted
    block1 := digestBlock1_refines canonical one accepted
    block2 := digestBlock2_refines canonical one accepted
    block3 := digestBlock3_refines canonical one accepted
    scalarEdge := stateBeforeScalar_next canonical }

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.ScheduleRefinement
