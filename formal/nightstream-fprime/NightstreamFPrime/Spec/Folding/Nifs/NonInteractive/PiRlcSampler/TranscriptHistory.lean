import Mathlib.Tactic.IntervalCases
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ScheduleLaw

/-! The block-oracle query histories replay the exact additive Poseidon2
schedule. A rate block is one four-field answer; a zero block represents
the permutation after the read. This is a value correspondence, not a
random-oracle assumption about the concrete permutation. -/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.TranscriptHistory

open NightstreamFPrime.Spec OracleModel ScheduleLaw

def replay (initial : Poseidon2.State) (history : List Draw) : Poseidon2.State :=
  history.foldl (fun state block => Poseidon2.absorbBlock state (List.ofFn block)) initial

theorem replay_append (initial : Poseidon2.State) (before after : List Draw) :
    replay initial (before ++ after) = replay (replay initial before) after :=
  by simp only [replay, List.foldl_append]

private theorem zeroBlock_getD (index : Nat) : (List.ofFn zeroBlock).getD index 0 = 0 := by
  rw [show List.ofFn zeroBlock = [0, 0, 0, 0] from rfl]
  rcases index with _ | _ | _ | _ | index <;> simp

private def domain (coordinate : Nat) : Draw := fun lane =>
  if lane.val = 0 then Poseidon2.ofNat 4
  else if lane.val = 1 then Poseidon2.ofNat coordinate else 0

private theorem absorb_domain (state : Poseidon2.State) (coordinate : Nat) :
    Poseidon2.absorbBlock state (List.ofFn (domain coordinate)) = Transcript.enter state coordinate := by
  unfold Poseidon2.absorbBlock Transcript.enter
  apply congrArg Poseidon2.permute
  apply List.map_congr_left
  intro index member
  have bound : index < 16 := List.mem_range.mp member
  interval_cases index <;> simp [domain, drawWidth]

private theorem enter_length (state : Poseidon2.State) (coordinate : Nat) :
    (Transcript.enter state coordinate).length = Poseidon2.width := by
  unfold Transcript.enter Poseidon2.absorbBlock
  exact Poseidon2.permute_length _

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
      exact Poseidon2.absorbBlock_zero
        (Transcript.enter (Transcript.stateAt (replay seed initial) coordinate) coordinate)
        (enter_length _ _) zeroBlock_getD

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

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.TranscriptHistory
