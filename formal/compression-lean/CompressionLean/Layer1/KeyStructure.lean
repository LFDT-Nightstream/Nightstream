import CompressionLean.Layer1.ClosedForms

/-!
Owns the verifier's copy of the run and slot structure
(`matrix.rs::Structure`) and the step functions it evaluates over it:
`matrix_step` over the runs and `slot_step` over the slots.

The model's tables are the setup's: `matrixColumn` and the lane splits of the
slots. Two Rust facts make them step functions of the key:

- `Checked`, the part of `Structure::check` the step functions use: run
  bounds that start at zero and grow, segments that start at slot zero, grow
  strictly and end before `slots`, and classes of positive length.
- `Describes`, what `Tables::from_rows` builds: run `k` of matrix `j` lies in
  `[runBound j, runBound (j + 1))`, and every slot of a segment has the
  segment's class and lane.

Under both, `matrixStep_eq`, `slotStep_head` and `slotStep_tail` prove the
Rust step formulas equal to the extensions of the model's tables.
-/

namespace CompressionLean.Layer1

open CompressionLean.Lib

/-- The verifier's run and slot structure (`matrix.rs::Structure`). -/
structure Structure (F : Type) where
  /-- The first run of matrix `j`; at `j = matrices`, the run count. -/
  runBound : ℕ → ℕ
  /-- Class `c < classes`: the run length and the ratio. -/
  classLength : ℕ → ℕ
  classRatio : ℕ → F
  classes : ℕ
  /-- Segment `i < segments`: its first slot, its class and its lane. -/
  segmentFirst : ℕ → ℕ
  segmentClass : ℕ → ℕ
  segmentLane : ℕ → ℕ
  segments : ℕ
  slots : ℕ

namespace Structure

variable {F : Type} (key : Structure F) (s : Shape)

/-- The first slot of segment `i`, and `slots` past the last segment (`slot_ranges`). -/
def segmentStart (i : ℕ) : ℕ := if i < key.segments then key.segmentFirst i else key.slots

/-- The part of `Structure::check` the step functions use. -/
structure Checked : Prop where
  run_zero : key.runBound 0 = 0
  run_mono : ∀ j, j < s.matrices → key.runBound j ≤ key.runBound (j + 1)
  segment_zero : 0 < key.segments → key.segmentFirst 0 = 0
  segment_mono : ∀ i, i + 1 < key.segments → key.segmentFirst i < key.segmentFirst (i + 1)
  segment_last : ∀ i, i + 1 = key.segments → key.segmentFirst i < key.slots
  class_named : ∀ i, i < key.segments → key.segmentClass i < key.classes
  length_pos : ∀ c, c < key.classes → 1 ≤ key.classLength c

/-- What `Tables::from_rows` builds: the setup's run matrices and slot shapes
are the key's step functions. -/
structure Describes (S : Setup F) : Prop where
  runs : S.runs = key.runBound s.matrices
  run_matrix : ∀ j k, j < s.matrices → key.runBound j ≤ k → k < key.runBound (j + 1) → S.runMatrix k = j
  slots : S.slots = key.slots
  cover : key.segments = 0 → key.slots = 0
  slot_shape : ∀ i σ, i < key.segments → key.segmentStart i ≤ σ → σ < key.segmentStart (i + 1) →
    S.slotLength σ = key.classLength (key.segmentClass i) ∧
      S.slotRatio σ = key.classRatio (key.segmentClass i) ∧ S.slotLane σ = key.segmentLane i

end Structure

section Steps

variable {K : Type} [CommRing K]

theorem stepTable_range' (a : ℕ → ℕ) (v : ℕ → K) (x : ℕ) :
    ∀ m k, stepTable (a (k + m)) ((List.range' k m).map fun i => (a i, v i)) x =
      ∑ i ∈ Finset.Ico k (k + m), if a i ≤ x ∧ x < a (i + 1) then v i else 0 := by
  intro m
  induction m with
  | zero => intro k; simp [stepTable]
  | succ m ih =>
      intro k
      have next : nextStart (a (k + (m + 1))) ((List.range' (k + 1) m).map fun i => (a i, v i)) = a (k + 1) := by
        cases m with
        | zero => rfl
        | succ m => rfl
      rw [List.range'_succ, List.map_cons, stepTable, next, Finset.sum_eq_sum_Ico_succ_bot (by omega),
        show k + (m + 1) = k + 1 + m by omega, ih (k + 1)]

/-- The step table of `n` segments with starts `a 0, …, a (n - 1)` and stop `a n`. -/
theorem stepTable_range (n : ℕ) (a : ℕ → ℕ) (v : ℕ → K) (x : ℕ) :
    stepTable (a n) ((List.range n).map fun i => (a i, v i)) x =
      ∑ i ∈ Finset.range n, if a i ≤ x ∧ x < a (i + 1) then v i else 0 := by
  have := stepTable_range' a v x n 0
  rw [List.range_eq_range', Finset.range_eq_Ico]
  simpa only [Nat.zero_add] using this

omit [CommRing K] in
theorem ordered_range' (a : ℕ → ℕ) (v : ℕ → K) (n : ℕ) (mono : ∀ i, i < n → a i ≤ a (i + 1)) :
    ∀ m k, k + m = n → Ordered (a n) ((List.range' k m).map fun i => (a i, v i)) := by
  intro m
  induction m with
  | zero => intro k _; trivial
  | succ m ih =>
      intro k sum
      have next : nextStart (a n) ((List.range' (k + 1) m).map fun i => (a i, v i)) = a (k + 1) := by
        cases m with
        | zero => simp [nextStart, ← sum]
        | succ m => rfl
      rw [List.range'_succ, List.map_cons]
      exact ⟨next ▸ mono k (by omega), ih (k + 1) (by omega)⟩

/-- Segments whose starts grow from zero cover `[0, a n)` once: the step table
is `g` when `g` is `v i` on segment `i` and zero from `a n` on. -/
theorem stepTable_eq (n : ℕ) (a : ℕ → ℕ) (v g : ℕ → K) (zero : a 0 = 0)
    (mono : ∀ i, i < n → a i ≤ a (i + 1))
    (inside : ∀ i x, i < n → a i ≤ x → x < a (i + 1) → x < a n → g x = v i)
    (outside : ∀ x, a n ≤ x → g x = 0)
    (x : ℕ) : stepTable (a n) ((List.range n).map fun i => (a i, v i)) x = g x := by
  have le : ∀ i j, i ≤ j → j ≤ n → a i ≤ a j := by
    intro i j ij jn
    induction j with
    | zero => rw [Nat.le_zero.mp ij]
    | succ j ih =>
        rcases Nat.lt_or_ge i (j + 1) with lt | ge
        · exact (ih (by omega) (by omega)).trans (mono j (by omega))
        · rw [show i = j + 1 by omega]
  rw [stepTable_range]
  by_cases past : a n ≤ x
  · rw [outside x past]
    exact Finset.sum_eq_zero fun i member => if_neg fun ⟨_, below⟩ =>
      absurd (below.trans_le (le (i + 1) n (by simp at member; omega) le_rfl)) (not_lt.mpr past)
  push Not at past
  set j := Nat.findGreatest (fun i => a i ≤ x) n
  have low : a j ≤ x := Nat.findGreatest_spec (P := fun i => a i ≤ x) (Nat.zero_le n) (by rw [zero]; omega)
  have small : j < n := lt_of_le_of_ne (Nat.findGreatest_le n) fun same =>
    absurd ((same ▸ low : a n ≤ x)) (not_le.mpr past)
  have high : x < a (j + 1) := by
    have := Nat.findGreatest_is_greatest (P := fun i => a i ≤ x) (k := j + 1) (Nat.lt_succ_self j) small
    simpa using this
  rw [Finset.sum_eq_single j, if_pos ⟨low, high⟩, inside j x small low high past]
  · intro i _ ne
    apply if_neg
    rintro ⟨lo, hi⟩
    rcases Nat.lt_or_gt_of_ne ne with lt | gt
    · exact absurd (hi.trans_le (le (i + 1) j (by omega) small.le)) (not_lt.mpr low)
    · exact absurd (high.trans_le (le (j + 1) i (by omega) (by simp at *; omega))) (not_lt.mpr lo)
  · intro out
    exact absurd (Finset.mem_range.mpr small) out

end Steps

section Key

variable {F : Type} (s : Shape) (S : Setup F) (key : Structure F)

theorem segmentStart_zero (checked : key.Checked s) (describes : key.Describes s S) : key.segmentStart 0 = 0 := by
  unfold Structure.segmentStart
  split_ifs with some
  · exact checked.segment_zero some
  · exact describes.cover (by omega)

theorem segmentStart_mono (checked : key.Checked s) :
    ∀ i, i < key.segments → key.segmentStart i ≤ key.segmentStart (i + 1) := by
  intro i lt
  unfold Structure.segmentStart
  rw [if_pos lt]
  split_ifs with next
  · exact (checked.segment_mono i next).le
  · exact (checked.segment_last i (by omega)).le

theorem segmentStart_end : key.segmentStart key.segments = key.slots := by
  simp [Structure.segmentStart]

/-- The slot segments as `slot_step` reads them: first slot and value. -/
theorem segments_eq {K : Type} (value : ℕ → K) :
    (List.range key.segments).map (fun i => (key.segmentFirst i, value i)) =
      (List.range key.segments).map (fun i => (key.segmentStart i, value i)) := by
  apply List.map_congr_left
  intro i member
  rw [Structure.segmentStart, if_pos (List.mem_range.mp member)]

/-- `matrix.rs::matrix_step` is the extension of the matrix column. -/
theorem matrixStep_eq {K : Type} [Field F] [Field K] (fits : S.Fits s) (checked : key.Checked s)
    (describes : key.Describes s S)
    (p : List K) :
    stepValue p (key.runBound s.matrices) ((List.range s.matrices).map fun j => (key.runBound j, (j : K))) =
      mle p (matrixColumn S) := by
  rw [step_eq p _ _ (ordered_range' _ _ s.matrices checked.run_mono s.matrices 0 (by omega) |>
    fun h => by rwa [← List.range_eq_range'] at h)]
  congr 1
  funext x
  apply stepTable_eq s.matrices key.runBound (fun j => (j : K)) (matrixColumn S) checked.run_zero
    checked.run_mono
  · intro j k lt lo hi _
    simp [matrixColumn, describes.run_matrix j k lt lo hi]
  · intro k past
    simp [matrixColumn, (fits.padding k (describes.runs ▸ past)).2.2.2]

/-- The order `slot_step` needs: segment starts that grow to `slots`. -/
theorem segments_ordered {K : Type} (checked : key.Checked s) (value : ℕ → K) :
    Ordered key.slots ((List.range key.segments).map fun i => (key.segmentStart i, value i)) := by
  have := ordered_range' key.segmentStart value key.segments (segmentStart_mono s key checked) key.segments 0
    (by omega)
  rwa [segmentStart_end, ← List.range_eq_range'] at this

theorem slotStep_eq {K : Type} [CommRing K] (checked : key.Checked s) (describes : key.Describes s S)
    (value : ℕ → K) (g : ℕ → K)
    (inside : ∀ i σ, i < key.segments → key.segmentStart i ≤ σ → σ < key.segmentStart (i + 1) →
      σ < key.slots → g σ = value i)
    (outside : ∀ σ, key.slots ≤ σ → g σ = 0) (p : List K) :
    stepValue p key.slots ((List.range key.segments).map fun i => (key.segmentFirst i, value i)) = mle p g := by
  rw [segments_eq, step_eq p _ _ (segments_ordered s key checked value)]
  congr 1
  funext σ
  have := stepTable_eq key.segments key.segmentStart value g (segmentStart_zero s S key checked describes)
    (segmentStart_mono s key checked) (by rwa [segmentStart_end]) (by rwa [segmentStart_end]) σ
  rwa [segmentStart_end] at this

/-- `matrix.rs::slot_step` at `t = 0`: each segment's value is its class's
head lane table at its lane; this is the extension of the model's head split. -/
theorem slotStep_head {K : Type} [Field F] [Field K] [Algebra F K] (fits : S.Fits s) (checked : key.Checked s)
    (describes : key.Describes s S) (τ : ℕ → K) (p : List K) :
    stepValue p key.slots ((List.range key.segments).map fun i => (key.segmentFirst i,
        headValue s.D (key.classLength (key.segmentClass i)) (algebraMap F K (key.classRatio (key.segmentClass i))) τ
          (key.segmentLane i))) =
      mle p (laneSplit s S τ 0) := by
  apply slotStep_eq s S key checked describes
  · intro i σ lt lo hi below
    obtain ⟨length, ratio, lane⟩ := describes.slot_shape i σ lt lo hi
    rw [laneSplit_head s S τ σ (describes.slots ▸ below) (fits.lane_lt σ)
      (length ▸ checked.length_pos _ (checked.class_named i lt)), length, ratio, lane]
  · intro σ past
    simp [laneSplit, describes.slots, past]

/-- `matrix.rs::slot_step` at `t = 1`: the same with the tail lane tables. -/
theorem slotStep_tail {K : Type} [Field F] [Field K] [Algebra F K] (fits : S.Fits s) (checked : key.Checked s)
    (describes : key.Describes s S) (τ : ℕ → K) (p : List K) :
    stepValue p key.slots ((List.range key.segments).map fun i => (key.segmentFirst i,
        tailValue s.D (key.classLength (key.segmentClass i)) (algebraMap F K (key.classRatio (key.segmentClass i))) τ
          (key.segmentLane i))) =
      mle p (laneSplit s S τ 1) := by
  apply slotStep_eq s S key checked describes
  · intro i σ lt lo hi below
    obtain ⟨length, ratio, lane⟩ := describes.slot_shape i σ lt lo hi
    rw [laneSplit_tail s S τ σ (describes.slots ▸ below) (fits.lane_lt σ) (fits.length_le σ), length, ratio, lane]
  · intro σ past
    simp [laneSplit, describes.slots, past]

end Key

end CompressionLean.Layer1
