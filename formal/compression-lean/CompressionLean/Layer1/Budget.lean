import CompressionLean.Layer1.Cost

/-!
Owns the layer-1 error at production sizes (`production_error`) and the
layer-1 budget with the two WHIR opening errors as named terms
(`production_budget`).

`Shape.Production` and the hypotheses `L0, L1 ≤ 32` and `123 ≤ queries`
are premises, not facts about every key. The Rust test
`production_shape_reaches_the_required_security`
(`crates/neo-spartan/tests/internal/whir.rs`) asserts them for the
production key of `finish.rs::relation`. The cube bound `2^29` covers every
key that `key()` accepts (at most `MAX_MESSAGE_COLUMNS` blocks of 64 lanes).
The challenge field is the cubic Goldilocks extension, with
`p^3 ≥ 2^189` elements.
-/

namespace CompressionLean.Layer1

/-- Size bounds of the production layer-1 shape. -/
structure Shape.Production (s : Shape) : Prop where
  cube : s.cubeVars ≤ 29
  runs : s.runVars ≤ 28
  rows : s.rowVars ≤ 28
  pairs : s.slotVars + s.matrixVars ≤ 28
  slots : s.slotVars + 1 ≤ 28
  norm : s.normBound ≤ 2 ^ 12
  rowCount : s.rows ≤ 2 ^ 8
  ring : s.D ≤ 64
  setup : 3 ≤ s.setupVars

theorem gkrLayerCost_le (s : GkrShape) (small : ∀ t, s.factors t ≤ 2) (n : ℕ) (E : ℕ → ℕ) (e k : ℕ)
    (ride : ∀ k, E k ≤ e) :
    gkrLayerCost s n E k ≤ n * (s.trees + 2) + e + k * (4 * n + e) := by
  have active : (s.active k).length ≤ s.trees :=
    (List.length_filter_le _ _).trans (by simp)
  have degree : s.degree k ≤ 4 := s.degree_le k 4 (by norm_num) fun t _ => by
    unfold GkrShape.partFactors
    split
    · have := small t; omega
    · omega
  have h1 : n * ((s.active k).length - 1) ≤ n * s.trees := Nat.mul_le_mul_left _ (by omega)
  have h2 : (k - 1) * (n * s.degree k + E k) ≤ k * (4 * n + e) :=
    Nat.mul_le_mul (by omega) (by have := Nat.mul_le_mul_left n degree; have := ride k; omega)
  have h3 := ride k
  unfold gkrLayerCost
  nlinarith

theorem gkrCost_le (s : GkrShape) (small : ∀ t, s.factors t ≤ 2) (n : ℕ) (E : ℕ → ℕ) (e bound : ℕ)
    (ride : ∀ k, E k ≤ e) :
    ∀ remaining k, k + remaining ≤ bound + 1 →
      gkrCost s n E remaining k ≤ remaining * (n * (s.trees + 2) + e + bound * (4 * n + e)) := by
  intro remaining
  induction remaining with
  | zero => intro k _; simp [gkrCost]
  | succ r ih =>
      intro k fits
      simp only [gkrCost]
      have layer := gkrLayerCost_le s small n E e k ride
      have scaled : k * (4 * n + e) ≤ bound * (4 * n + e) := Nat.mul_le_mul_right _ (by omega)
      have rest := ih (k + 1) (by omega)
      rw [Nat.succ_mul]
      omega

/-- At production sizes the layer-1 bad-challenge count is below `2^40`. -/
theorem layer1Count_le (s : Shape) (prod : s.Production) (L0 L1 : ℕ) (h0 : L0 ≤ 32) (h1 : L1 ≤ 32) :
    layer1Count s L0 L1 ≤ 2 ^ 40 := by
  have lanes : s.laneVars ≤ 29 := by have := prod.cube; unfold Shape.cubeVars at this; omega
  have blocks : s.blockVars ≤ 29 := by have := prod.cube; unfold Shape.cubeVars at this; omega
  have earlyLayers : (earlyShape s).layers ≤ 29 := GkrShape.layers_le _ _ fun t _ => by
    simp only [earlyShape]
    have := prod.cube; have := prod.runs; have := prod.rows; have := prod.pairs
    split_ifs <;> omega
  have lateLayers : (lateShape s).layers ≤ 29 := GkrShape.layers_le _ _ fun t _ => by
    simp only [lateShape]
    have := prod.slots
    split_ifs <;> omega
  have earlyGkr := gkrCost_le (earlyShape s) (fun t => by simp only [earlyShape]; split_ifs <;> omega)
    L0 (earlyRide s L0) L0 29 (fun k => by unfold earlyRide; split_ifs <;> omega) (earlyShape s).layers 1
    (by omega)
  have lateGkr := gkrCost_le (lateShape s) (fun t => by simp only [lateShape]; split_ifs <;> omega)
    (L0 * L1) (fun _ => 0) 0 29 (fun _ => le_rfl) (lateShape s).layers 1 (by omega)
  have early : earlyCount s L0 ≤ 32 * (2 ^ 29 + (2 * 2 ^ 12 + 1)) +
      (32 * (2 * 2 ^ 28 + (2 ^ 28 + 2 ^ 28)) + (32 * (1 + (2 ^ 28 + 2 ^ 28)) +
        29 * (32 * (5 + 2) + 32 + 29 * (4 * 32 + 32)))) := by
    unfold earlyCount
    have trees : (earlyShape s).trees = 5 := rfl
    rw [trees] at earlyGkr
    have := prod.cube; have := prod.runs; have := prod.rows; have := prod.pairs; have := prod.norm
    have := earlyGkr.trans (Nat.mul_le_mul earlyLayers
      (show L0 * (5 + 2) + L0 + 29 * (4 * L0 + L0) ≤ 32 * (5 + 2) + 32 + 29 * (4 * 32 + 32) by omega))
    gcongr <;> omega
  have linear : linearCount s L0 ≤ 32 * (2 ^ 8 - 1) + (32 * (2 * 64 - 2) + (29 * (32 * 2) + 29 * (32 * 2))) := by
    unfold linearCount
    have := prod.rowCount; have := prod.ring
    gcongr
  have late : lateCount s L0 L1 ≤ 32 * 32 * (2 ^ 28 + 2 ^ 29) +
      29 * (32 * 32 * (2 + 2) + 0 + 29 * (4 * (32 * 32) + 0)) := by
    unfold lateCount
    have trees : (lateShape s).trees = 2 := rfl
    rw [trees] at lateGkr
    have product : L0 * L1 ≤ 32 * 32 := Nat.mul_le_mul h0 h1
    have := lateGkr.trans (Nat.mul_le_mul lateLayers
      (show L0 * L1 * (2 + 2) + 0 + 29 * (4 * (L0 * L1) + 0) ≤ 32 * 32 * (2 + 2) + 0 + 29 * (4 * (32 * 32) + 0)
        by omega))
    have := prod.slots
    gcongr <;> omega
  unfold layer1Count
  have := Nat.add_le_add (Nat.add_le_add (Nat.add_le_add early linear) (le_refl 9)) late
  refine this.trans ?_
  norm_num

/-- Layer-1 error at production sizes: the field part is at most `2^-149`;
the query part is at most `32·2^-t` with `2^(setupVars - 2)` query leaves. -/
theorem production_error (s : Shape) (prod : s.Production) (L0 L1 size : ℕ) (h0 : L0 ≤ 32) (h1 : L1 ≤ 32)
    (big : 2 ^ 189 ≤ size) :
    layer1Error s L0 L1 (2 ^ (s.setupVars - 2)) size ≤ (2 : ℝ) ^ 40 / 2 ^ 189 + 32 * (1 / 2) ^ s.queries := by
  unfold layer1Error queryError
  have positive : (0 : ℝ) < 2 ^ 189 := by positivity
  have sizeBig : (2 : ℝ) ^ 189 ≤ size := by exact_mod_cast big
  have count : (layer1Count s L0 L1 : ℝ) ≤ 2 ^ 40 := by exact_mod_cast layer1Count_le s prod L0 L1 h0 h1
  have field : (layer1Count s L0 L1 : ℝ) / size ≤ 2 ^ 40 / 2 ^ 189 :=
    div_le_div₀ (by positivity) count positive sizeBig
  have half : (((2 ^ (s.setupVars - 3) - 1 : ℕ) : ℝ) / ((2 ^ (s.setupVars - 2) : ℕ) : ℝ)) ≤ 1 / 2 := by
    have split : s.setupVars - 2 = (s.setupVars - 3) + 1 := by have := prod.setup; omega
    rw [split, pow_succ, div_le_div_iff₀ (by positivity) (by norm_num)]
    have : ((2 ^ (s.setupVars - 3) - 1 : ℕ) : ℝ) ≤ ((2 ^ (s.setupVars - 3) : ℕ) : ℝ) := by
      exact_mod_cast Nat.sub_le _ _
    push_cast at this ⊢
    linarith
  have query : (L1 : ℝ) * ((((2 ^ (s.setupVars - 3) - 1 : ℕ) : ℝ) / ((2 ^ (s.setupVars - 2) : ℕ) : ℝ))) ^ s.queries ≤
      32 * (1 / 2) ^ s.queries :=
    mul_le_mul (by exact_mod_cast h1) (pow_le_pow_left₀ (by positivity) half _) (by positivity) (by norm_num)
  linarith

/-- The layer-1 budget: with at least 123 setup queries (`query_count` at a
114-bit minimum) and WHIR opening errors of at most `2^-116` each, the total
is at most `2^-114`. -/
theorem production_budget (s : Shape) (prod : s.Production) (L0 L1 size : ℕ) (h0 : L0 ≤ 32) (h1 : L1 ≤ 32)
    (big : 2 ^ 189 ≤ size) (queries : 123 ≤ s.queries) (whirP0 whirP1 : ℝ)
    (p0 : whirP0 ≤ 1 / 2 ^ 116) (p1 : whirP1 ≤ 1 / 2 ^ 116) :
    whirP0 + whirP1 + layer1Error s L0 L1 (2 ^ (s.setupVars - 2)) size ≤ 1 / 2 ^ 114 := by
  have error := production_error s prod L0 L1 size h0 h1 big
  have query : (1 / 2 : ℝ) ^ s.queries ≤ (1 / 2) ^ 123 := pow_le_pow_of_le_one (by norm_num) (by norm_num) queries
  have : (2 : ℝ) ^ 40 / 2 ^ 189 + 32 * (1 / 2) ^ 123 + 2 * (1 / 2 ^ 116) ≤ 1 / 2 ^ 114 := by norm_num
  linarith

end CompressionLean.Layer1
