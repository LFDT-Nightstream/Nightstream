import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonReplayTransition

/-!
Contract: structural call-sequence semantics for one production PiRLC
Poseidon2 replay run.

Assurance tier: artifact-checked same-assignment call-chain semantics for the
Nightstream b2/k16 profile.

Owns: exact replay of the ordered words consumed by all calls in one run into
the final independent Poseidon2 call result.

Does not own: the final unpermuted tail, equality with the complete 918-word
or 54-word authoritative frame, carried family-state placement, final
matrix-slice identity, complete PiRLC semantics, or lifecycle soundness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplaySequence

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallFamily
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedCapacity
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCompactTrace
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.Wire
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplayTransition
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonRunValues

private def fin2 : Fin 4 := ⟨2, by decide⟩
private def fin3 : Fin 4 := ⟨3, by decide⟩

def firstAbsorbed (run : Run) : Nat :=
  match run.raw.firstClass with
  | .direct => 0
  | .partialStart => 2

/-- The state immediately before the first run call. Rate lanes that the first
call overwrites are set to their already selected call-input values. -/
def initialState (run : Run)
    (assignment : Fin productionFinalColumns → F) : Poseidon2Duplex.State where
  lanes := callInputs run 0 assignment
  absorbed :=
    match run.raw.firstClass with
    | .direct => 0
    | .partialStart => 2

/-- The normalized state produced by one independently checked call. -/
def callState (run : Run) (index : Nat)
    (assignment : Fin productionFinalColumns → F) : Poseidon2Duplex.State where
  lanes := callReference run index assignment
  absorbed := 0

/-- Exact words consumed by one call. A partial first call consumes two words;
all other calls consume four. -/
def callWordsAt (run : Run) (index : Nat)
    (assignment : Fin productionFinalColumns → F) : List Nat :=
  match index, run.raw.firstClass with
  | 0, .partialStart =>
      [callInputs run 0 assignment (rateLane fin2),
        callInputs run 0 assignment (rateLane fin3)]
  | _, _ =>
      List.ofFn fun lane : Fin 4 =>
        callInputs run index assignment (rateLane lane)

/-- Ordered words consumed by the first `count` calls. -/
def callWordsPrefix (run : Run) (count : Nat)
    (assignment : Fin productionFinalColumns → F) : List Nat :=
  (List.range count).flatMap fun index => callWordsAt run index assignment

/-- Exact replay-relevant placement of a carried state at the first call.
Rate lanes at or above the cursor are omitted because overwrite absorption
replaces them before the first permutation. -/
structure ReplayStartPlaced (run : Run)
    (assignment : Fin productionFinalColumns → F)
    (prior : Poseidon2Duplex.State) : Prop where
  absorbed : prior.absorbed = firstAbsorbed run
  carried : ∀ lane : Fin 4, lane.val < firstAbsorbed run →
    callInputs run 0 assignment (rateLane lane) =
      prior.lanes (rateLane lane)
  capacity : ∀ lane : Fin 4,
    callInputs run 0 assignment (capacityLane lane) =
      prior.lanes (capacityLane lane)

private theorem callInputs_canonical (run : Run) (index : Nat)
    (assignment : Fin productionFinalColumns → F) (lane : Fin width) :
    callInputs run index assignment lane < goldilocksP := by
  exact (sourceInput (sourceAt run index assignment) lane).isLt

private theorem state_ext
    (left right : Poseidon2Duplex.State)
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) : left = right := by
  cases left
  cases right
  simp_all

private theorem full_absorb_exact
    (current prior : Values)
    (currentCanonical : ∀ lane, current lane < goldilocksP)
    (capacityExact : ∀ lane : Fin 4,
      current (capacityLane lane) = prior (capacityLane lane)) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        (List.ofFn fun lane : Fin 4 => current (rateLane lane))
        { lanes := prior, absorbed := 0 } =
      { lanes := referencePermutation Poseidon2CanonicalConstants.selected current,
        absorbed := 0 } := by
  have capacity0 : prior ⟨4, by decide⟩ = current ⟨4, by decide⟩ := by
    simpa [capacityLane] using (capacityExact ⟨0, by decide⟩).symm
  have capacity1 : prior ⟨5, by decide⟩ = current ⟨5, by decide⟩ := by
    simpa [capacityLane] using (capacityExact ⟨1, by decide⟩).symm
  have capacity2 : prior ⟨6, by decide⟩ = current ⟨6, by decide⟩ := by
    simpa [capacityLane] using (capacityExact ⟨2, by decide⟩).symm
  have capacity3 : prior ⟨7, by decide⟩ = current ⟨7, by decide⟩ := by
    simpa [capacityLane] using (capacityExact ⟨3, by decide⟩).symm
  apply state_ext
  · funext lane
    fin_cases lane <;>
      simp [Poseidon2Duplex.absorbSlice, Poseidon2Duplex.absorbList,
        Poseidon2Duplex.absorbElem, Poseidon2Duplex.guarded,
        Poseidon2Duplex.permute, Poseidon2Sponge.rate, rateLane, capacityLane,
        Nat.mod_eq_of_lt (currentCanonical _)]
    all_goals
      apply congrArg (fun values =>
        referencePermutation Poseidon2CanonicalConstants.selected values _)
      funext inputLane
      fin_cases inputLane <;>
        simp [capacity0, capacity1, capacity2, capacity3]
  · simp [Poseidon2Duplex.absorbSlice, Poseidon2Duplex.absorbList,
      Poseidon2Duplex.absorbElem, Poseidon2Duplex.guarded,
      Poseidon2Duplex.permute, Poseidon2Sponge.rate]

private theorem partial_absorb_exact
    (current : Values)
    (currentCanonical : ∀ lane, current lane < goldilocksP) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        [current (rateLane fin2), current (rateLane fin3)]
        { lanes := current, absorbed := 2 } =
      { lanes := referencePermutation Poseidon2CanonicalConstants.selected current,
        absorbed := 0 } := by
  apply state_ext
  · funext lane
    fin_cases lane <;>
      simp [fin2, fin3, Poseidon2Duplex.absorbSlice,
        Poseidon2Duplex.absorbList, Poseidon2Duplex.absorbElem,
        Poseidon2Duplex.guarded, Poseidon2Duplex.permute,
        Poseidon2Sponge.rate, rateLane,
        Nat.mod_eq_of_lt (currentCanonical _)]
    all_goals
      apply congrArg (fun values =>
        referencePermutation Poseidon2CanonicalConstants.selected values _)
      funext inputLane
      fin_cases inputLane <;> simp [fin2, fin3]
  · simp [fin2, fin3, Poseidon2Duplex.absorbSlice,
      Poseidon2Duplex.absorbList, Poseidon2Duplex.absorbElem,
      Poseidon2Duplex.guarded, Poseidon2Duplex.permute,
      Poseidon2Sponge.rate]

private theorem partial_absorb_relevant_exact
    (current prior : Values)
    (currentCanonical : ∀ lane, current lane < goldilocksP)
    (carried : ∀ lane : Fin 4, lane.val < 2 →
      current (rateLane lane) = prior (rateLane lane))
    (capacityExact : ∀ lane : Fin 4,
      current (capacityLane lane) = prior (capacityLane lane)) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        [current (rateLane fin2), current (rateLane fin3)]
        { lanes := prior, absorbed := 2 } =
      { lanes := referencePermutation Poseidon2CanonicalConstants.selected current,
        absorbed := 0 } := by
  have rate0 : prior ⟨0, by decide⟩ = current ⟨0, by decide⟩ := by
    simpa [rateLane] using
      (carried ⟨0, by decide⟩ (by decide)).symm
  have rate1 : prior ⟨1, by decide⟩ = current ⟨1, by decide⟩ := by
    simpa [rateLane] using
      (carried ⟨1, by decide⟩ (by decide)).symm
  have capacity0 : prior ⟨4, by decide⟩ = current ⟨4, by decide⟩ := by
    simpa [capacityLane] using (capacityExact ⟨0, by decide⟩).symm
  have capacity1 : prior ⟨5, by decide⟩ = current ⟨5, by decide⟩ := by
    simpa [capacityLane] using (capacityExact ⟨1, by decide⟩).symm
  have capacity2 : prior ⟨6, by decide⟩ = current ⟨6, by decide⟩ := by
    simpa [capacityLane] using (capacityExact ⟨2, by decide⟩).symm
  have capacity3 : prior ⟨7, by decide⟩ = current ⟨7, by decide⟩ := by
    simpa [capacityLane] using (capacityExact ⟨3, by decide⟩).symm
  apply state_ext
  · funext lane
    fin_cases lane <;>
      simp [fin2, fin3, Poseidon2Duplex.absorbSlice,
        Poseidon2Duplex.absorbList, Poseidon2Duplex.absorbElem,
        Poseidon2Duplex.guarded, Poseidon2Duplex.permute,
        Poseidon2Sponge.rate, rateLane,
        Nat.mod_eq_of_lt (currentCanonical _)]
    all_goals
      apply congrArg (fun values =>
        referencePermutation Poseidon2CanonicalConstants.selected values _)
      funext inputLane
      fin_cases inputLane <;>
        simp [fin2, fin3, rate0, rate1, capacity0, capacity1, capacity2,
          capacity3]
  · simp [fin2, fin3, Poseidon2Duplex.absorbSlice,
      Poseidon2Duplex.absorbList, Poseidon2Duplex.absorbElem,
      Poseidon2Duplex.guarded, Poseidon2Duplex.permute,
      Poseidon2Sponge.rate]

private theorem first_call_exact (run : Run)
    (assignment : Fin productionFinalColumns → F) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        (callWordsAt run 0 assignment) (initialState run assignment) =
      callState run 0 assignment := by
  cases first : run.raw.firstClass with
  | direct =>
      simpa [callWordsAt, initialState, callState, callReference, first] using
        full_absorb_exact (callInputs run 0 assignment)
          (callInputs run 0 assignment)
          (callInputs_canonical run 0 assignment) (fun _ => rfl)
  | partialStart =>
      simpa [callWordsAt, initialState, callState, callReference, first] using
        partial_absorb_exact (callInputs run 0 assignment)
          (callInputs_canonical run 0 assignment)

/-- The first call depends only on the carried rate prefix, capacity lanes,
and cursor. It does not require equality of rate lanes that the call
overwrites. -/
theorem first_call_from_placed_start_exact
    (run : Run) (assignment : Fin productionFinalColumns → F)
    (prior : Poseidon2Duplex.State)
    (placed : ReplayStartPlaced run assignment prior) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        (callWordsAt run 0 assignment) prior =
      callState run 0 assignment := by
  cases first : run.raw.firstClass with
  | direct =>
      have absorbed : prior.absorbed = 0 := by
        simpa [firstAbsorbed, first] using placed.absorbed
      have priorExact : prior = { lanes := prior.lanes, absorbed := 0 } := by
        cases prior
        simp_all
      rw [priorExact]
      simpa [callWordsAt, callState, callReference, first] using
        full_absorb_exact (callInputs run 0 assignment) prior.lanes
          (callInputs_canonical run 0 assignment) placed.capacity
  | partialStart =>
      have absorbed : prior.absorbed = 2 := by
        simpa [firstAbsorbed, first] using placed.absorbed
      have priorExact : prior = { lanes := prior.lanes, absorbed := 2 } := by
        cases prior
        simp_all
      rw [priorExact]
      simpa [callWordsAt, callState, callReference, first] using
        partial_absorb_relevant_exact (callInputs run 0 assignment) prior.lanes
          (callInputs_canonical run 0 assignment)
          (by
            intro lane bounded
            exact placed.carried lane (by
              simpa [firstAbsorbed, first] using bounded))
          placed.capacity

private theorem next_call_exact
    (run : Run) (assignment : Fin productionFinalColumns → F)
    (freshWord : Nat → F)
    (transition : RunReplayTransition run assignment freshWord)
    (index : Nat) (currentInRange : index.succ < run.raw.callCount) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        (callWordsAt run index.succ assignment)
        (callState run index assignment) =
      callState run index.succ assignment := by
  have capacityExact : ∀ lane : Fin 4,
      callInputs run index.succ assignment (capacityLane lane) =
        callReference run index assignment (capacityLane lane) := by
    intro lane
    exact transition.chained index currentInRange lane
  simpa [callWordsAt, callState, callReference] using
    full_absorb_exact (callInputs run index.succ assignment)
      (callReference run index assignment)
      (callInputs_canonical run index.succ assignment) capacityExact

theorem callWordsPrefix_succ (run : Run) (count : Nat)
    (assignment : Fin productionFinalColumns → F) :
    callWordsPrefix run count.succ assignment =
      callWordsPrefix run count assignment ++ callWordsAt run count assignment := by
  simp [callWordsPrefix, List.range_succ, List.flatMap_append]

/-- Every nonempty prefix of a retained run evaluates to the corresponding
independent call result. The induction is over calls, not generated rows. -/
theorem replay_prefix_exact
    (run : Run) (assignment : Fin productionFinalColumns → F)
    (freshWord : Nat → F)
    (transition : RunReplayTransition run assignment freshWord)
    (count : Nat) (positive : 0 < count)
    (bounded : count ≤ run.raw.callCount) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        (callWordsPrefix run count assignment) (initialState run assignment) =
      callState run (count - 1) assignment := by
  induction count with
  | zero => omega
  | succ count inductionHypothesis =>
      cases count with
      | zero =>
          simpa [callWordsPrefix_succ, callWordsPrefix] using
            first_call_exact run assignment
      | succ prior =>
          rw [callWordsPrefix_succ, Poseidon2Duplex.absorbSlice_append]
          rw [inductionHypothesis (by omega) (by omega)]
          simpa using next_call_exact run assignment freshWord transition prior
            (by omega)

/-- Every nonempty prefix has the same exact result from any replay-relevant
placed start state. -/
theorem replay_prefix_from_placed_start_exact
    (run : Run) (assignment : Fin productionFinalColumns → F)
    (freshWord : Nat → F)
    (transition : RunReplayTransition run assignment freshWord)
    (prior : Poseidon2Duplex.State)
    (placed : ReplayStartPlaced run assignment prior)
    (count : Nat) (positive : 0 < count)
    (bounded : count ≤ run.raw.callCount) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        (callWordsPrefix run count assignment) prior =
      callState run (count - 1) assignment := by
  induction count with
  | zero => omega
  | succ count inductionHypothesis =>
      cases count with
      | zero =>
          simpa [callWordsPrefix_succ, callWordsPrefix] using
            first_call_from_placed_start_exact run assignment prior placed
      | succ priorIndex =>
          rw [callWordsPrefix_succ, Poseidon2Duplex.absorbSlice_append]
          rw [inductionHypothesis (by omega) (by omega)]
          simpa using next_call_exact run assignment freshWord transition
            priorIndex (by omega)

/-- The complete retained run evaluates to its last independent call result. -/
theorem run_replay_exact
    (run : Run) (productionRun : run ∈ runs)
    (assignment : Fin productionFinalColumns → F)
    (freshWord : Nat → F)
    (transition : RunReplayTransition run assignment freshWord) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        (callWordsPrefix run run.raw.callCount assignment)
        (initialState run assignment) =
      callState run (run.raw.callCount - 1) assignment := by
  exact replay_prefix_exact run assignment freshWord transition
    run.raw.callCount (runs_valid run productionRun).callCountPositive le_rfl

/-- The complete retained run is exact from any replay-relevant placed start
state. -/
theorem run_replay_from_placed_start_exact
    (run : Run) (productionRun : run ∈ runs)
    (assignment : Fin productionFinalColumns → F)
    (freshWord : Nat → F)
    (transition : RunReplayTransition run assignment freshWord)
    (prior : Poseidon2Duplex.State)
    (placed : ReplayStartPlaced run assignment prior) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        (callWordsPrefix run run.raw.callCount assignment) prior =
      callState run (run.raw.callCount - 1) assignment := by
  exact replay_prefix_from_placed_start_exact run assignment freshWord
    transition prior placed run.raw.callCount
      (runs_valid run productionRun).callCountPositive le_rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplaySequence
