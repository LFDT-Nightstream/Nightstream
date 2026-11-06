import CompressionLean.Sim.Algebra

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Replay (Word)

def queueWords (words : Array Word) : Array Word :=
  (Array.range rate).map fun i => words.reverse.getD i 0

def cursorWords (count : ℕ) : Array Word :=
  (Array.range 13).map fun i => if i = count then 1 else 0

@[simp] theorem queueWords_size (words : Array Word) : (queueWords words).size = rate := by
  simp [queueWords]

@[simp] theorem cursorWords_size (count : ℕ) : (cursorWords count).size = 13 := by
  simp [cursorWords]

theorem queueWords_get (words : Array Word) (i : ℕ) (hi : i < rate) :
    (queueWords words)[i]! = words.reverse.getD i 0 := by
  simp [queueWords, getElem!_pos, hi]

theorem cursorWords_get (count i : ℕ) (hi : i < 13) :
    (cursorWords count)[i]! = if i = count then 1 else 0 := by
  simp [cursorWords, getElem!_pos, hi]

theorem queueWords_getD (words : Array Word) (bound : words.size ≤ rate) (i : ℕ) :
    (queueWords words).getD i 0 = words.reverse.getD i 0 := by
  by_cases hi : i < rate
  · simpa [Array.getD, hi, queueWords_size, ← getElem!_pos] using queueWords_get words i hi
  · have hw : ¬i < words.size := by omega
    simp [Array.getD, hi, hw]

theorem reverse_pop_getD (words : Array Word) (i : ℕ) :
    words.pop.reverse.getD i 0 = words.reverse.getD (i + 1) 0 := by
  have e : words.pop.reverse.toList = words.reverse.toList.tail := by simp
  simp only [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, ← List.getD_eq_getElem?_getD]
  rw [e]
  cases words.reverse.toList <;> rfl

theorem queueWords_shift (words : Array Word) (bound : words.size ≤ rate) :
    (Array.range rate).map (fun i => (queueWords words).getD (i + 1) 0) = queueWords words.pop := by
  apply Array.ext
  · simp
  · intro i hi hj
    simp only [Array.size_map, Array.size_range] at hi
    rw [Array.getElem_map, Array.getElem_range, queueWords_getD words bound]
    rw [← reverse_pop_getD, ← queueWords_get words.pop i hi, getElem!_pos (queueWords words.pop) i hj]

theorem cursorWords_shift (count : ℕ) (pos : 0 < count) (bound : count ≤ rate) :
    (Array.range 13).map (fun i => (cursorWords count).getD (i + 1) 0) = cursorWords (count - 1) := by
  apply Array.ext
  · simp
  · intro i hi hj
    simp only [Array.size_map, Array.size_range] at hi
    rw [Array.getElem_map, Array.getElem_range, ← getElem!_pos (cursorWords (count - 1)) i hj,
      cursorWords_get _ _ hi]
    by_cases hn : i + 1 < 13
    · rw [show (cursorWords count).getD (i + 1) 0 = (cursorWords count)[i + 1]! by
        simp [Array.getD, hn, getElem!_pos]]
      rw [cursorWords_get _ _ hn]
      have e : (i + 1 = count) ↔ (i = count - 1) := by omega
      simp only [e]
    · have hc : i ≠ count - 1 := by dsimp [rate] at bound; omega
      simp [Array.getD, hn, hc]

theorem map_range_words (words : Array Word) (count : ℕ) (size : words.size = count) :
    (Array.range count).map (fun i => words[i]!) = words := by
  apply Array.ext
  · simp [size]
  · intro i _ hi
    rw [Array.getElem_map, Array.getElem_range, getElem!_pos words i hi]

theorem queueWords_head (words : Array Word) (pos : 0 < words.size) :
    (queueWords words)[0]! = words.back! := by
  rw [queueWords_get _ _ (by decide)]
  simp [Array.getD, Array.back!, pos, getElem!_pos]

theorem queueWords_extract (words : Array Word) (size : words.size = 16) :
    queueWords (words.extract 0 rate) = (Array.range rate).map fun i => words[rate - 1 - i]! := by
  apply Array.ext
  · simp
  · intro i hi _
    simp only [queueWords_size] at hi
    rw [← getElem!_pos (queueWords (words.extract 0 rate)) i (by simpa using hi), queueWords_get _ _ hi]
    change i < 12 at hi
    have hj : 11 - i < words.size := by rw [size]; omega
    simp [Array.getD, size, rate, hi, getElem!_pos words (11 - i) hj]

theorem duplexing_empty (r : Replay.Duplex) (empty : r.input = #[]) :
    r.duplexing = ⟨Replay.permute r.state, #[], (Replay.permute r.state).extract 0 Replay.rate⟩ := by
  simp [Replay.Duplex.duplexing, empty]

theorem sample_input_empty (r : Replay.Duplex) : r.sample.2.input = #[] := by
  unfold Replay.Duplex.sample
  split
  · rfl
  · rename_i h
    have hi : r.input.isEmpty = true := by
      cases hi : r.input.isEmpty <;> simp_all
    exact Array.empty_of_isEmpty hi

end CompressionLean.Sim
