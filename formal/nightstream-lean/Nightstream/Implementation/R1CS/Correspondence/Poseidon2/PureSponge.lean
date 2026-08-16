import Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge
import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.ExtractedReference

/-!
Contract: semantic bridge from the column-free extracted Poseidon2 sponge
runner to the canonical additive Poseidon2 sponge.

Owns full-rate absorb schedules, their exact value chunks, the final padding
round, and equality of every output lane with the selected canonical
Poseidon2 reference.

Does not own generated rows, input-frame authority, collision resistance, or
protocol domain separation.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 10000000

namespace Nightstream.Implementation.R1CS.Poseidon2PureSponge

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge

/-- A column-free witness of one value schedule. Only `kind` affects
`runValueRounds`. -/
def representativeRound : ValueSchedule → Round
  | .absorb count =>
      { (default : Round) with
        kind := .absorb (List.replicate count 0) }
  | .pad =>
      { (default : Round) with kind := .pad }

theorem representativeRound_schedule (schedule : ValueSchedule) :
    (representativeRound schedule).valueSchedule = schedule := by
  cases schedule <;> simp [representativeRound, Round.valueSchedule]

/-- `count` full four-field absorbs followed by the required padding call. -/
def fullRateRounds : Nat → List Round
  | 0 => [representativeRound .pad]
  | count + 1 => representativeRound (.absorb 4) :: fullRateRounds count

def fullRateSchedule : Nat → List ValueSchedule
  | 0 => [.pad]
  | count + 1 => .absorb 4 :: fullRateSchedule count

theorem fullRateRounds_schedule (count : Nat) :
    valueSchedules (fullRateRounds count) = fullRateSchedule count := by
  induction count with
  | zero => simp [fullRateRounds, fullRateSchedule, valueSchedules,
      representativeRound_schedule]
  | succ count inductionHypothesis =>
      simp only [fullRateRounds, fullRateSchedule, valueSchedules,
        List.map_cons, List.cons.injEq]
      exact ⟨representativeRound_schedule _, by
        simpa only [valueSchedules] using inductionHypothesis⟩

/-- The exact four-field chunks consumed by `fullRateRounds`. -/
def fullRateChunks : List Nat → Nat →
    List Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.RateChunk
  | _, 0 => []
  | values, count + 1 =>
      { values := values.take 4
        bounded := by
          simp only [List.length_take,
            Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.rate]
          omega } ::
        fullRateChunks (values.drop 4) count

def toReferenceState (state : Nat → Nat) :
    Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference.Values :=
  fun lane => state lane.val

private theorem absorb_round_refines
    (count : Nat) (values : List Nat) (state : Nat → Nat)
    (valuesLength : values.length = count)
    (stateCanonical : ∀ lane, state lane < goldilocksP)
    (lane : Fin 8) :
    valueRound (representativeRound (.absorb count)) values state lane.val =
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference.referencePermutation
        Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
        (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorbChunk
          values (toReferenceState state)) lane := by
  have inputCanonical :
      ∀ inputLane, inputLane < 8 →
        valueInput (representativeRound (.absorb count)).kind values state
            inputLane < goldilocksP := by
    intro inputLane _inputLaneLt
    simp only [representativeRound, valueInput, List.length_replicate]
    split
    · exact Nat.mod_lt _ (by decide)
    · exact stateCanonical inputLane
  calc
    valueRound (representativeRound (.absorb count)) values state lane.val =
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference.referencePermutation
          Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
          (fun inputLane =>
            valueInput (representativeRound (.absorb count)).kind values state
              inputLane.val) lane := by
      exact Nightstream.Implementation.R1CS.Poseidon2ExtractedReference.permute_eq_reference
        inputCanonical lane
    _ = Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference.referencePermutation
          Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
          (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorbChunk
            values (toReferenceState state)) lane := by
      congr 2
      funext inputLane
      by_cases inChunk : inputLane.val < count
      · have inValues : inputLane.val < values.length := by
          simpa [valuesLength] using inChunk
        simp [representativeRound, valueInput,
          Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorbChunk,
          toReferenceState, inChunk, List.getD_eq_getElem?_getD,
          List.getElem?_eq_getElem inValues]
      · have beyond : values.length ≤ inputLane.val := by
          omega
        calc
          valueInput (representativeRound (.absorb count)).kind values state
              inputLane.val = state inputLane.val := by
            simp [representativeRound, valueInput, inChunk]
          _ = Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorbChunk
                values (toReferenceState state) inputLane :=
            (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorbChunk_beyond_chunk
              values (toReferenceState state) inputLane beyond).symm

private theorem pad_round_refines
    (state : Nat → Nat)
    (stateCanonical : ∀ lane, state lane < goldilocksP)
    (lane : Fin 8) :
    valueRound (representativeRound .pad) [] state lane.val =
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference.referencePermutation
        Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
        (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.pad
          (toReferenceState state)) lane := by
  have inputCanonical :
      ∀ inputLane, inputLane < 8 →
        valueInput (representativeRound .pad).kind [] state inputLane <
          goldilocksP := by
    intro inputLane _inputLaneLt
    simp only [representativeRound, valueInput]
    split
    · exact Nat.mod_lt _ (by decide)
    · exact stateCanonical inputLane
  calc
    valueRound (representativeRound .pad) [] state lane.val =
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference.referencePermutation
          Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
          (fun inputLane =>
            valueInput (representativeRound .pad).kind [] state inputLane.val)
          lane := by
      exact Nightstream.Implementation.R1CS.Poseidon2ExtractedReference.permute_eq_reference
        inputCanonical lane
    _ = Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference.referencePermutation
          Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
          (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.pad
            (toReferenceState state)) lane := by
      congr 2
      funext inputLane
      by_cases zero : inputLane.val = 0
      · have inputLaneEq : inputLane = ⟨0, by decide⟩ := Fin.ext zero
        subst inputLane
        rfl
      · simp [representativeRound, valueInput,
          Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.pad,
          toReferenceState, zero]

/-- A complete full-rate extracted schedule is the selected canonical
Poseidon2 absorption over the same chunks and the required padding chunk. -/
theorem fullRateRounds_refine
    (count : Nat) (values : List Nat) (state : Nat → Nat)
    (valuesLength : values.length = 4 * count)
    (valuesCanonical : ∀ value ∈ values, value < goldilocksP)
    (stateCanonical : ∀ lane, state lane < goldilocksP)
    (lane : Fin 8) :
    runValueRounds (fullRateRounds count) values state lane.val =
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorb
        Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
        (fullRateChunks values count ++
          [Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.paddingChunk])
        (toReferenceState state) lane := by
  induction count generalizing values state with
  | zero =>
      have pad := pad_round_refines state stateCanonical lane
      simpa [fullRateRounds, fullRateChunks, runValueRounds,
        representativeRound,
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorb,
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.paddingChunk_absorbs]
        using pad
  | succ count inductionHypothesis =>
      let headValues := values.take 4
      let nextState :=
        valueRound (representativeRound (.absorb 4)) headValues state
      have headLength : headValues.length = 4 := by
        unfold headValues
        rw [List.length_take, valuesLength]
        omega
      have headCanonical : ∀ value ∈ headValues, value < goldilocksP := by
        intro value member
        exact valuesCanonical value (List.mem_of_mem_take member)
      have tailLength : (values.drop 4).length = 4 * count := by
        rw [List.length_drop, valuesLength]
        omega
      have tailCanonical :
          ∀ value ∈ values.drop 4, value < goldilocksP := by
        intro value member
        exact valuesCanonical value (List.mem_of_mem_drop member)
      have nextCanonical : ∀ stateLane, nextState stateLane < goldilocksP := by
        exact valueRound_canonical (representativeRound (.absorb 4)) headValues
          state stateCanonical
      have nextReference :
          toReferenceState nextState =
            Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference.referencePermutation
              Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
              (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorbChunk
                headValues (toReferenceState state)) := by
        funext stateLane
        exact absorb_round_refines 4 headValues state headLength stateCanonical
          stateLane
      calc
        runValueRounds (fullRateRounds (count + 1)) values state lane.val =
            runValueRounds (fullRateRounds count) (values.drop 4) nextState
              lane.val := by rfl
        _ = Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorb
              Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
              (fullRateChunks (values.drop 4) count ++
                [Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.paddingChunk])
              (toReferenceState nextState) lane :=
          inductionHypothesis (values.drop 4) nextState tailLength tailCanonical
            nextCanonical
        _ = Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorb
              Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
              (fullRateChunks (values.drop 4) count ++
                [Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.paddingChunk])
              (Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference.referencePermutation
                Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
                (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorbChunk
                  headValues (toReferenceState state))) lane := by
          rw [nextReference]
        _ = Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorb
              Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
              (fullRateChunks values (count + 1) ++
                [Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.paddingChunk])
              (toReferenceState state) lane := by
          rfl

/-- Digest-lane form of `fullRateRounds_refine`. -/
theorem fullRateRounds_compute_digest
    (count : Nat) (values : List Nat)
    (valuesLength : values.length = 4 * count)
    (valuesCanonical : ∀ value ∈ values, value < goldilocksP)
    (lane : Fin 4) :
    runValueRounds (fullRateRounds count) values (fun _ => 0) lane.val =
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digest
        Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
        (fullRateChunks values count) lane := by
  let outputLane : Fin 8 := ⟨lane.val, by omega⟩
  calc
    runValueRounds (fullRateRounds count) values (fun _ => 0) lane.val =
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorb
          Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
          (fullRateChunks values count ++
            [Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.paddingChunk])
          Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.initialSpongeState
          outputLane := by
      exact fullRateRounds_refine count values (fun _ => 0) valuesLength
        valuesCanonical (by
          intro _stateLane
          exact (by decide : 0 < goldilocksP)) outputLane
    _ = Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digest
          Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
          (fullRateChunks values count) lane := by
      exact (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digest_eq_absorb_padding
        Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
        (fullRateChunks values count) lane).symm

end Nightstream.Implementation.R1CS.Poseidon2PureSponge
