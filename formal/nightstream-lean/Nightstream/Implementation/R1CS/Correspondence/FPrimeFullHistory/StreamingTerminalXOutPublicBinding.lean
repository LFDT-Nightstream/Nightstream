import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutLifecycleBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPublicHashRowSound

/-!
Contract: the exact terminal public-word rows bind the Poseidon2 XOut hash to
the verifier-owned four-word public input.

Owns the typed interpretation of the 256 generated public-bit columns and the
composition from public-word row soundness to the lifecycle digest type.

Does not own terminal source-to-final assignment transport, the 32-field XOut
decoder, final selective-row transport, or terminal acceptance.

Assurance tier: artifact-checked for the Nightstream b2/k16 terminal profile.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPublicBinding

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.Nebula.StateOutputPoseidonBinding
open Nightstream.Protocol.Nebula
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPublicHash.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPublicHash
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallArtifact
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPublicHashRowSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation

private abbrev LifecycleDigest :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Digest

private theorem publicWords_length : publicWords.length = 4 := by
  have lengths := publicWords_paired.length_eq
  simpa using lengths.symm

/-- Exact generated public word for one terminal XOut digest lane. -/
def publicWordAt (lane : Fin 4) : PublicWord :=
  publicWords.get ⟨lane.val, by
    rw [publicWords_length]
    exact lane.isLt⟩

theorem publicWordAt_member (lane : Fin 4) :
    publicWordAt lane ∈ publicWords := by
  exact List.get_mem publicWords _

theorem publicWordAt_paired (lane : Fin 4) :
    (publicWordAt lane).fieldColumn =
        trace.outputColumns.getD lane.val 0 ∧
      (publicWordAt lane).Valid := by
  have paired := publicWords_paired.get
    (i := lane.val)
    (by simpa using lane.isLt)
    (by rw [publicWords_length]; exact lane.isLt)
  simpa [publicWordAt] using paired

/-- Numeric value of one verifier-owned little-endian public word. -/
def publicWordValue (assignment : Nat → Nat) (lane : Fin 4) : Nat :=
  Nat.ofDigits 2
    ((List.range CanonicalFieldBits.bitCount).map fun index =>
      assignment
        ((publicWordAt lane).publicBitColumns.getD index 0))

/-- Typed interpretation of the exact generated public columns. This is a
public-input placement, not an asserted digest equality. -/
structure PublicAssignmentBinding
    (assignment : Nat → Nat) (publicXOut : LifecycleDigest) : Prop where
  value : ∀ lane,
    publicWordValue assignment lane = digestValues publicXOut lane

private theorem publicWord_decode_eq_value
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.Satisfied assignment)
    (lane : Fin 4) :
    CanonicalFieldBits.decode
        (publicWordOfRows assignment canonical one satisfied
          (publicWordAt lane) (publicWordAt_member lane)
          (publicWordAt_paired lane).2) =
      publicWordValue assignment lane := by
  unfold publicWordValue CanonicalFieldBits.decode
  rw [publicWordOfRows_val]

private theorem rounds_length : rounds.length = 9 := by
  rfl

private theorem round_schedule_of_getD
    (index : Nat) (bounded : index < rounds.length)
    (schedule : ValueSchedule)
    (exact : (rounds.getD index default).valueSchedule = schedule) :
    (rounds.get ⟨index, bounded⟩).valueSchedule = schedule := by
  rw [← List.getD_eq_get]
  exact exact

private theorem round0_schedule :
    (rounds.getD 0 default).valueSchedule = .absorb 4 := by
  unfold Round.valueSchedule
  rw [round0_kind]
  rfl

private theorem round1_schedule :
    (rounds.getD 1 default).valueSchedule = .absorb 4 := by
  unfold Round.valueSchedule
  rw [round1_kind]
  rfl

private theorem round2_schedule :
    (rounds.getD 2 default).valueSchedule = .absorb 4 := by
  unfold Round.valueSchedule
  rw [round2_kind]
  rfl

private theorem round3_schedule :
    (rounds.getD 3 default).valueSchedule = .absorb 4 := by
  unfold Round.valueSchedule
  rw [round3_kind]
  rfl

private theorem round4_schedule :
    (rounds.getD 4 default).valueSchedule = .absorb 4 := by
  unfold Round.valueSchedule
  rw [round4_kind]
  rfl

private theorem round5_schedule :
    (rounds.getD 5 default).valueSchedule = .absorb 4 := by
  unfold Round.valueSchedule
  rw [round5_kind]
  rfl

private theorem round6_schedule :
    (rounds.getD 6 default).valueSchedule = .absorb 4 := by
  unfold Round.valueSchedule
  rw [round6_kind]
  rfl

private theorem round7_schedule :
    (rounds.getD 7 default).valueSchedule = .absorb 4 := by
  unfold Round.valueSchedule
  rw [round7_kind]
  rfl

private theorem round8_schedule :
    (rounds.getD 8 default).valueSchedule = .pad := by
  unfold Round.valueSchedule
  rw [round8_kind]

/-- The nine generated terminal rounds have the canonical eight-absorb and
one-pad state-output schedule. The proof reuses the nine isolated round-kind
certificates and does not reduce the generated round payloads. -/
theorem valueSchedules_exact :
    valueSchedules rounds =
      Nightstream.Implementation.Nebula.StateOutputPoseidonRows.expectedSchedule := by
  apply List.ext_get
  · simp [valueSchedules, rounds_length,
      Nightstream.Implementation.Nebula.StateOutputPoseidonRows.expectedSchedule]
  · intro index leftBound rightBound
    have indexBound : index < 9 := by
      simpa [valueSchedules, rounds_length] using leftBound
    interval_cases index
    · change (rounds.get ⟨0, by simpa [rounds_length]⟩).valueSchedule =
        .absorb 4
      exact round_schedule_of_getD 0 (by simpa [rounds_length]) _
        round0_schedule
    · change (rounds.get ⟨1, by simpa [rounds_length]⟩).valueSchedule =
        .absorb 4
      exact round_schedule_of_getD 1 (by simpa [rounds_length]) _
        round1_schedule
    · change (rounds.get ⟨2, by simpa [rounds_length]⟩).valueSchedule =
        .absorb 4
      exact round_schedule_of_getD 2 (by simpa [rounds_length]) _
        round2_schedule
    · change (rounds.get ⟨3, by simpa [rounds_length]⟩).valueSchedule =
        .absorb 4
      exact round_schedule_of_getD 3 (by simpa [rounds_length]) _
        round3_schedule
    · change (rounds.get ⟨4, by simpa [rounds_length]⟩).valueSchedule =
        .absorb 4
      exact round_schedule_of_getD 4 (by simpa [rounds_length]) _
        round4_schedule
    · change (rounds.get ⟨5, by simpa [rounds_length]⟩).valueSchedule =
        .absorb 4
      exact round_schedule_of_getD 5 (by simpa [rounds_length]) _
        round5_schedule
    · change (rounds.get ⟨6, by simpa [rounds_length]⟩).valueSchedule =
        .absorb 4
      exact round_schedule_of_getD 6 (by simpa [rounds_length]) _
        round6_schedule
    · change (rounds.get ⟨7, by simpa [rounds_length]⟩).valueSchedule =
        .absorb 4
      exact round_schedule_of_getD 7 (by simpa [rounds_length]) _
        round7_schedule
    · change (rounds.get ⟨8, by simpa [rounds_length]⟩).valueSchedule =
        .pad
      exact round_schedule_of_getD 8 (by simpa [rounds_length]) _
        round8_schedule

/-- Exact public-word source rows derive the terminal public XOut from the
verifier-owned public bits. No digest supplied by the prover is authority. -/
theorem public_rows_imply_outer_hash
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.Satisfied assignment)
    (publicXOut : LifecycleDigest)
    (binding : PublicAssignmentBinding assignment publicXOut) :
    outerHash (trace.inputColumns.map assignment) =
      digestValues publicXOut := by
  have schedules :
      valueSchedules trace.rounds =
        valueSchedules
          Nightstream.Implementation.Nebula.StateOutputPoseidonRows.representativeRounds := by
    exact valueSchedules_exact.trans
      Nightstream.Implementation.Nebula.StateOutputPoseidonRows.representativeRounds_schedule.symm
  have sameHash := runValueRounds_eq_of_schedules schedules
    (trace.inputColumns.map assignment) (fun _ => 0)
  funext lane
  have paired := publicWordAt_paired lane
  calc
    outerHash (trace.inputColumns.map assignment) lane =
        runValueRounds
          Nightstream.Implementation.Nebula.StateOutputPoseidonRows.representativeRounds
          (trace.inputColumns.map assignment) (fun _ => 0) lane.val := rfl
    _ = runValueRounds trace.rounds (trace.inputColumns.map assignment)
          (fun _ => 0) lane.val := (congrFun sameHash lane.val).symm
    _ = CanonicalFieldBits.decode
          (publicWordOfRows assignment canonical one satisfied
            (publicWordAt lane) (publicWordAt_member lane) paired.2) :=
      (publicWord_hash_sound assignment canonical one satisfied lane.isLt
        (publicWordAt_member lane) paired.1 paired.2).symm
    _ = publicWordValue assignment lane :=
      publicWord_decode_eq_value assignment canonical one satisfied lane
    _ = digestValues publicXOut lane := binding.value lane

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPublicBinding
