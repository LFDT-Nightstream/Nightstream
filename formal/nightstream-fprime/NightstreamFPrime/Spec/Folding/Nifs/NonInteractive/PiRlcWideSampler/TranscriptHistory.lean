import Mathlib.Tactic.IntervalCases
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.ScheduleLaw

/-! The block-oracle query histories replay the exact additive Poseidon2
schedule. A rate block is one four-field answer; a zero block represents
the permutation after the read. This is a value correspondence, not a
random-oracle assumption about the concrete permutation. -/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.TranscriptHistory

open NightstreamFPrime.Spec OracleModel ScheduleLaw

def replay (initial : Poseidon2.State) (history : List Draw) : Poseidon2.State :=
  history.foldl (fun state block => Poseidon2.absorbBlock state (List.ofFn block)) initial

theorem replay_append (initial : Poseidon2.State) (before after : List Draw) :
    replay initial (before ++ after) = replay (replay initial before) after :=
  by simp only [replay, List.foldl_append]

private theorem rounds_length
    (step : Nat → Poseidon2.State → Poseidon2.State)
    (lengths : ∀ index state, (step index state).length = Poseidon2.width)
    (indices : List Nat) (state : Poseidon2.State)
    (fixed : state.length = Poseidon2.width) :
    (indices.foldl (fun state index => step index state) state).length = Poseidon2.width := by
  induction indices generalizing state with
  | nil => exact fixed
  | cons index rest ih => exact ih _ (lengths index state)

private theorem permute_length (state : Poseidon2.State) :
    (Poseidon2.permute state).length = Poseidon2.width := by
  unfold Poseidon2.permute Poseidon2.rounds
  apply rounds_length
  · intro index state
    simp [Poseidon2.fullRound, Poseidon2.externalLayer]
  · apply rounds_length
    · intro index state
      simp [Poseidon2.partialRound, Poseidon2.internalLayer]
    · apply rounds_length
      · intro index state
        simp [Poseidon2.fullRound, Poseidon2.externalLayer]
      · simp [Poseidon2.externalLayer]

private theorem absorb_zero (state : Poseidon2.State)
    (fixed : state.length = Poseidon2.width) :
    Poseidon2.absorbBlock state (List.ofFn zeroBlock) = Poseidon2.permute state := by
  unfold Poseidon2.absorbBlock
  apply congrArg Poseidon2.permute
  have zeros : List.ofFn zeroBlock = [0, 0, 0, 0] := rfl
  rw [zeros]
  have values : (List.range Poseidon2.width).map (fun index =>
      state.getD index 0 + [0, 0, 0, 0].getD index 0) =
      (List.range Poseidon2.width).map (fun index => state.getD index 0) := by
    apply List.map_congr_left
    intro index member
    have bound : index < 8 := List.mem_range.mp member
    interval_cases index <;> simp
  rw [values, ← fixed]
  apply List.ext_getElem
  · simp
  · intro index _bound bound
    simp only [List.getElem_map, List.getElem_range]
    exact (List.getElem_eq_getD 0).symm

private def domain (coordinate : Nat) : Draw := fun lane =>
  if lane.val = 0 then Poseidon2.ofNat 4
  else if lane.val = 1 then Poseidon2.ofNat coordinate else 0

private theorem absorb_domain (state : Poseidon2.State) (coordinate : Nat) :
    Poseidon2.absorbBlock state (List.ofFn (domain coordinate)) = Transcript.enter state coordinate := by
  unfold Poseidon2.absorbBlock Transcript.enter
  apply congrArg Poseidon2.permute
  apply List.map_congr_left
  intro index member
  have bound : index < 8 := List.mem_range.mp member
  interval_cases index <;> simp [domain, drawWidth]

private theorem enter_length (state : Poseidon2.State) (coordinate : Nat) :
    (Transcript.enter state coordinate).length = Poseidon2.width := by
  unfold Transcript.enter Poseidon2.absorbBlock
  exact permute_length _

private theorem historyAt_succ (initial : List Draw) (coordinate : Nat) :
    historyAt initial (coordinate + 1) =
      historyAt initial coordinate ++ [domain coordinate, zeroBlock] := by
  simp only [historyAt, List.range_succ, List.flatMap_append, List.flatMap_cons,
    List.flatMap_nil, List.append_nil, List.append_assoc]
  rfl

/-- The concrete state before every scalar entry is the replay of its full
normalized history. The initial history may include all earlier phases. -/
theorem stateAt_replay (seed : Poseidon2.State) (initial : List Draw) (coordinate : Nat) :
    replay seed (historyAt initial coordinate) =
      Transcript.stateAt (replay seed initial) coordinate := by
  induction coordinate with
  | zero => simp [historyAt, Transcript.stateAt]
  | succ coordinate ih =>
      rw [historyAt_succ, replay_append, ih]
      simp only [replay, List.foldl_cons, List.foldl_nil]
      rw [Transcript.stateAt_succ, absorb_domain]
      exact absorb_zero
        (Transcript.enter (Transcript.stateAt (replay seed initial) coordinate) coordinate)
        (enter_length _ _)

def answer (seed : Poseidon2.State) (query : Query) : Draw :=
  Transcript.block (replay seed query.val)

/-- Each of the 17 query keys reads exactly the four lanes used by the
candidate, with its existing [4,i] domain entry and one digest advance. -/
theorem queryAt_answer (seed : Poseidon2.State) (initial : List Draw) (coordinate : Fin 17) :
    answer seed (queryAt initial coordinate) =
      Transcript.drawAt (replay seed initial) coordinate.val := by
  unfold answer queryAt scalarQuery
  simp only [List.replicate_zero, List.append_nil]
  rw [replay_append, stateAt_replay]
  simp only [replay, List.foldl_cons, List.foldl_nil, Transcript.drawAt]
  exact congrArg Transcript.block (absorb_domain
    (Transcript.stateAt (replay seed initial) coordinate.val) coordinate.val)

theorem queryAt_sample (seed : Poseidon2.State) (initial : List Draw) (coordinate : Fin 17) :
    sample (answer seed (queryAt initial coordinate)) =
      Transcript.scalarAt (replay seed initial) coordinate.val :=
  congrArg sample (queryAt_answer seed initial coordinate)

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.TranscriptHistory
