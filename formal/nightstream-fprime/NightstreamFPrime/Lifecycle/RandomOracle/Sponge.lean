import NightstreamFPrime.Lifecycle.Transcript
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.TranscriptHistory

/-! The Poseidon2 sponge of the fold transcripts as a list of permutation calls (`Call`), for the
random-oracle model. A transcript absorbs words in padded rate-sized chunks (`absorbCalls`),
reads `K` coins from the state (`readK`), refreshes the state after the last coin of a chunk
(`refresh`), and reads a `Π_RLC` challenge from one sampler draw (`readRho`). These facts do not
depend on the fold; `TranscriptCoverage` uses them for the one-circuit fold. -/

namespace NightstreamFPrime.Lifecycle.RandomOracle.Sponge

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle

abbrev Call := Fin Poseidon2.rate → F

def squeeze : Call := fun _ => 0

def step (state : Transcript.State) (chunk : Call) : Transcript.State :=
  Poseidon2.absorbBlock state (List.ofFn chunk)

def run (state : Transcript.State) (calls : List Call) : Transcript.State :=
  calls.foldl step state

@[simp] theorem run_nil (state : Transcript.State) : run state [] = state := rfl

@[simp] theorem run_cons (state : Transcript.State) (call : Call) (calls : List Call) :
    run state (call :: calls) = run (step state call) calls := rfl

theorem run_append (state : Transcript.State) (left right : List Call) :
    run state (left ++ right) = run (run state left) right := by
  simp [run, List.foldl_append]

def padChunk (words : List F) : Call :=
  fun lane => words.getD lane.val 0

theorem absorbBlock_padChunk (state : Transcript.State) {words : List F}
    (short : words.length ≤ Poseidon2.rate) :
    Poseidon2.absorbBlock state words =
      Poseidon2.absorbBlock state (List.ofFn (padChunk words)) := by
  unfold Poseidon2.absorbBlock
  congr 1
  apply List.map_congr_left
  intro lane _
  congr 1
  by_cases inside : lane < Poseidon2.rate
  · simp [padChunk, List.getD_eq_getElem?_getD, inside]
  · have outside : Poseidon2.rate ≤ lane := Nat.le_of_not_lt inside
    rw [List.getD_eq_default _ _ (Nat.le_trans short outside),
      List.getD_eq_default _ _ (by simpa using outside)]

def absorbCalls (words : List F) : List Call :=
  (List.range ((words.length + Poseidon2.rate - 1) / Poseidon2.rate)).map fun chunk =>
    padChunk ((words.drop (chunk * Poseidon2.rate)).take Poseidon2.rate)

theorem absorb_eq_run (state : Transcript.State) (words : List F) :
    Transcript.absorb state words = run state (absorbCalls words) := by
  unfold Transcript.absorb absorbCalls run
  rw [List.foldl_map, List.foldl_map]
  apply List.foldl_ext
  intro current chunk _
  exact absorbBlock_padChunk current (List.length_take_le _ _)

@[simp] theorem absorbCalls_length (words : List F) :
    (absorbCalls words).length = (words.length + Poseidon2.rate - 1) / Poseidon2.rate := by
  simp [absorbCalls]

theorem absorbCalls_ne_nil {words : List F} (nonempty : words ≠ []) :
    absorbCalls words ≠ [] := by
  intro empty
  have length := congrArg List.length empty
  have positive : 0 < words.length := List.length_pos_iff.mpr nonempty
  simp only [absorbCalls_length, List.length_nil] at length
  unfold Poseidon2.rate at length
  omega

theorem absorbCalls_injective {left right : List F}
    (lengthEqual : left.length = right.length)
    (same : absorbCalls left = absorbCalls right) : left = right := by
  have chunks : ∀ chunk ∈ List.range ((left.length + Poseidon2.rate - 1) / Poseidon2.rate),
      padChunk ((left.drop (chunk * Poseidon2.rate)).take Poseidon2.rate) =
        padChunk ((right.drop (chunk * Poseidon2.rate)).take Poseidon2.rate) := by
    unfold absorbCalls at same
    rw [← lengthEqual] at same
    exact List.map_inj_left.mp same
  apply List.ext_getElem lengthEqual
  intro position leftBound rightBound
  have chunkBound : position / Poseidon2.rate <
      (left.length + Poseidon2.rate - 1) / Poseidon2.rate := by
    unfold Poseidon2.rate at *
    omega
  have lane := congrFun (chunks (position / Poseidon2.rate) (List.mem_range.mpr chunkBound))
    ⟨position % Poseidon2.rate, Nat.mod_lt _ (by decide)⟩
  have split : position / Poseidon2.rate * Poseidon2.rate +
      position % Poseidon2.rate = position := by
    rw [Nat.mul_comm]
    exact Nat.div_add_mod position Poseidon2.rate
  simp only [padChunk, List.getD_eq_getElem?_getD, List.getElem?_take,
    Nat.mod_lt _ (show 0 < Poseidon2.rate by decide), ite_true,
    List.getElem?_drop, split] at lane
  simpa [List.getElem?_eq_getElem leftBound, List.getElem?_eq_getElem rightBound]
    using lane

abbrev Slot := Fin Transcript.coinsPerChunk

def slotOf (position : Nat) : Slot :=
  ⟨position % Transcript.coinsPerChunk, Nat.mod_lt _ (by decide)⟩

def readK (slot : Slot) (state : Transcript.State) : K :=
  ⟨state.getD (2 * slot.val) 0, state.getD (2 * slot.val + 1) 0⟩

theorem absorbCalls_zeroChunk : absorbCalls Transcript.zeroChunk = [squeeze] := by
  have chunk : padChunk ((Transcript.zeroChunk.drop (0 * Poseidon2.rate)).take Poseidon2.rate) =
      squeeze := by
    funext lane
    simp [padChunk, squeeze, Transcript.zeroChunk, List.getD_eq_getElem?_getD]
  unfold absorbCalls
  rw [show (Transcript.zeroChunk.length + Poseidon2.rate - 1) / Poseidon2.rate = 1 from rfl,
    List.range_one, List.map_cons, List.map_nil, chunk]

def refresh (position : Nat) : List Call :=
  if position % Transcript.coinsPerChunk = Transcript.coinsPerChunk - 1 then [squeeze] else []

theorem replicate_refresh (position : Nat) :
    List.replicate (position / Transcript.coinsPerChunk) squeeze ++ refresh position =
      List.replicate ((position + 1) / Transcript.coinsPerChunk) squeeze := by
  have six : Transcript.coinsPerChunk = 6 := rfl
  unfold refresh
  split
  · rename_i last
    rw [six] at last ⊢
    rw [show (position + 1) / 6 = position / 6 + 1 by omega, List.replicate_succ']
  · rename_i notLast
    rw [six] at notLast ⊢
    rw [show (position + 1) / 6 = position / 6 by omega, List.append_nil]

def widen (draw : Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Draw) : Call := fun lane =>
  if inside : lane.val < Spec.Folding.Nifs.NonInteractive.PiRlcSampler.drawWidth then
    draw ⟨lane.val, inside⟩
  else 0

def readRho (state : Transcript.State) : RingF :=
  Phi81StrongSet.embedScalar
    (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample
      (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.block state))

theorem step_widen (state : Transcript.State)
    (draw : Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Draw) :
    step state (widen draw) = Poseidon2.absorbBlock state (List.ofFn draw) := by
  unfold step Poseidon2.absorbBlock
  congr 1

theorem run_map_widen (state : Transcript.State)
    (history : List Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Draw) :
    run state (history.map widen) =
      Spec.Folding.Nifs.NonInteractive.PiRlcSampler.TranscriptHistory.replay state history := by
  induction history generalizing state with
  | nil => rfl
  | cons draw history inductionHypothesis =>
      show run (step state (widen draw)) (history.map widen) =
        Spec.Folding.Nifs.NonInteractive.PiRlcSampler.TranscriptHistory.replay
          (Poseidon2.absorbBlock state (List.ofFn draw)) history
      rw [step_widen]
      exact inductionHypothesis _

theorem absorbCalls_head? {words : List F} (nonempty : words ≠ []) :
    (absorbCalls words).head? = some (padChunk (words.take Poseidon2.rate)) := by
  have positive : 0 < (words.length + Poseidon2.rate - 1) / Poseidon2.rate := by
    have length : 0 < words.length := List.length_pos_iff.mpr nonempty
    unfold Poseidon2.rate
    omega
  obtain ⟨count, count_eq⟩ : ∃ count, (words.length + Poseidon2.rate - 1) / Poseidon2.rate =
      count + 1 := ⟨_, (Nat.succ_pred_eq_of_pos positive).symm⟩
  unfold absorbCalls
  rw [count_eq, List.range_succ_eq_map, List.map_cons, List.head?_cons, Nat.zero_mul, List.drop_zero]

/-- One squeeze at coin position `position`: the coin it reads and the state after it. -/
theorem squeezeAt_eq (state : Transcript.State) (position : Nat) :
    Transcript.squeezeAt state position =
      (readK (slotOf position) state, run state (refresh position)) := by
  by_cases last : position % Transcript.coinsPerChunk = Transcript.coinsPerChunk - 1
  · rw [Transcript.squeezeAt, ite_eq_left last, refresh, ite_eq_left last, absorb_eq_run,
      absorbCalls_zeroChunk]
    rfl
  · rw [Transcript.squeezeAt, ite_eq_right last, refresh, ite_eq_right last]
    rfl

end NightstreamFPrime.Lifecycle.RandomOracle.Sponge
