import CompressionLean.Lib.Combination
import CompressionLean.Lib.Fraction

/-!
Owns the algebra of one layer of the batched fraction-sum GKR of
`crates/neo-spartan/src/gkr.rs`.

At layer `k`, every tree of depth at least `k` (`GkrShape.active`) has parts
at level `k`: numerator factors and a denominator (one factor inside the
tree, the leaf factors at its leaf layer). The layer's sum-check runs over
`k - 1` variables on `eq(ρ, y)·Σ_i ν^i·(p₀q₁ + p₁q₀ + μ·q₀q₁)`, where `₀`/`₁`
are the halves `y` and `y + 2^{k-1}` (the top variable is the child bit).

- `layer_truth_nil`: the sum is the batched claim of the true level `k - 1`.
- `layer_truth_final`: at the end, it is `eq` times the batched combination
  of the true halves at the sum-check point (`trueChildren`).
- `reduce_true`: true halves reduce to the extension at `next ++ [τ]`.
-/

namespace CompressionLean.Layer1

open CompressionLean.Lib
open Polynomial

/-- `Σ_i ν^i·f(lᵢ)` by Horner's rule, the batching of `gkr::verify`. -/
def wsum {R : Type} [CommSemiring R] (ν : R) (f : ℕ → R) : List ℕ → R
  | [] => 0
  | t :: rest => f t + ν * wsum ν f rest

theorem wsum_congr {R : Type} [CommSemiring R] (ν : R) {f g : ℕ → R} {l : List ℕ}
    (equal : ∀ t ∈ l, f t = g t) : wsum ν f l = wsum ν g l := by
  induction l with
  | nil => rfl
  | cons t rest ih =>
      simp only [wsum]
      rw [equal t (by simp), ih fun u inside => equal u (List.mem_cons_of_mem t inside)]

/-- The trees of one batch as the verifier knows them. -/
structure GkrShape where
  trees : ℕ
  depth : ℕ → ℕ
  factors : ℕ → ℕ

namespace GkrShape

variable (s : GkrShape)

/-- The trees that reach layer `k`, in tree order. -/
def active (k : ℕ) : List ℕ := (List.range s.trees).filter fun t => k ≤ s.depth t

/-- Numerator factors of tree `t` at layer `k`: one inside, the leaf factors at its leaves. -/
def partFactors (k t : ℕ) : ℕ := if s.depth t = k then s.factors t else 1

/-- The round degree of layer `k`: 3, or `factors + 2` at a product leaf layer. -/
def degree (k : ℕ) : ℕ := (s.active k).foldr (fun t d => max d (s.partFactors k t + 2)) 3

/-- The number of layers: the largest depth. -/
def layers : ℕ := (List.range s.trees).foldr (fun t d => max d (s.depth t)) 0

theorem mem_active {k t : ℕ} : t ∈ s.active k ↔ t < s.trees ∧ k ≤ s.depth t := by
  simp [active]

private theorem foldr_max_base (l : List ℕ) (g : ℕ → ℕ) (b : ℕ) :
    b ≤ l.foldr (fun t d => max d (g t)) b := by
  induction l with
  | nil => exact le_rfl
  | cons t rest ih => exact ih.trans (le_max_left _ _)

private theorem foldr_max_mem (l : List ℕ) (g : ℕ → ℕ) (b : ℕ) {t : ℕ} (member : t ∈ l) :
    g t ≤ l.foldr (fun t d => max d (g t)) b := by
  induction l with
  | nil => simp at member
  | cons u rest ih =>
      rcases List.mem_cons.mp member with rfl | inside
      · exact le_max_right _ _
      · exact (ih inside).trans (le_max_left _ _)

theorem le_degree (k : ℕ) : 3 ≤ s.degree k := foldr_max_base _ _ _

theorem partFactors_le_degree {k t : ℕ} (member : t ∈ s.active k) :
    s.partFactors k t + 2 ≤ s.degree k :=
  foldr_max_mem _ (fun t => s.partFactors k t + 2) 3 member

theorem depth_le_layers {t : ℕ} (tree : t < s.trees) : s.depth t ≤ s.layers :=
  foldr_max_mem _ s.depth 0 (List.mem_range.mpr tree)

private theorem foldr_max_le (l : List ℕ) (g : ℕ → ℕ) (b bound : ℕ) (hb : b ≤ bound)
    (each : ∀ t ∈ l, g t ≤ bound) : l.foldr (fun t d => max d (g t)) b ≤ bound := by
  induction l with
  | nil => exact hb
  | cons t rest ih =>
      exact max_le (ih fun u member => each u (List.mem_cons_of_mem t member)) (each t (by simp))

theorem degree_le (s : GkrShape) (k bound : ℕ) (three : 3 ≤ bound)
    (factors : ∀ t ∈ s.active k, s.partFactors k t + 2 ≤ bound) : s.degree k ≤ bound :=
  foldr_max_le _ _ _ _ three factors

theorem layers_le (s : GkrShape) (bound : ℕ) (depths : ∀ t, t < s.trees → s.depth t ≤ bound) :
    s.layers ≤ bound :=
  foldr_max_le _ _ _ _ (Nat.zero_le _) fun t member => depths t (List.mem_range.mp member)

end GkrShape

variable {K : Type} [Field K]

/-- The leaves of every tree of one world: tree `t` ↦ its leaf level. -/
abbrev Batch (K : Type) := ℕ → Level K

/-- Each tree's leaves have the shape's factor count. -/
def Batch.Fits (batch : Batch K) (s : GkrShape) : Prop :=
  ∀ t, t < s.trees → (batch t).factors.length = s.factors t

/-- Level `k` of tree `t`. -/
def treeLevel (s : GkrShape) (batch : Batch K) (t k : ℕ) : Level K :=
  (batch t).above (s.depth t) (s.depth t - k)

/-- Part `j` of tree `t` at layer `k`: factor `j`, or the denominator for `j = partFactors`. -/
def treePart (s : GkrShape) (batch : Batch K) (k t j : ℕ) : ℕ → K :=
  if j < s.partFactors k t then (treeLevel s batch t k).factors.getD j (fun _ => 0)
  else (treeLevel s batch t k).denominator

/-- The true claim of tree `t` at layer `k`: level `k - 1` at `ρ`. -/
def trueClaim (s : GkrShape) (batch : Batch K) (k t : ℕ) (ρ : List K) : K × K :=
  (mle ρ (treeLevel s batch t (k - 1)).numerator, mle ρ (treeLevel s batch t (k - 1)).denominator)

/-- The true halves of tree `t`'s parts at the sum-check point `next`. -/
def trueChildren (s : GkrShape) (batch : Batch K) (k t : ℕ) (next : List K) (j : ℕ) : K × K :=
  (mle next (treePart s batch k t j), mle next (fun y => treePart s batch k t j (y + 2 ^ (k - 1))))

/-- `p₀q₁ + p₁q₀ + μ·q₀q₁` from part pairs: factors `0 ..< f`, denominator `f`. -/
def combineVals (f : ℕ) (pairs : ℕ → K × K) (μ : K) : K :=
  ((List.range f).map fun j => (pairs j).1).prod * (pairs f).2 +
    ((List.range f).map fun j => (pairs j).2).prod * (pairs f).1 + μ * (pairs f).1 * (pairs f).2

/-- One pair reduced at the top variable. -/
def reduce (pair : K × K) (τ : K) : K := pair.1 + τ * (pair.2 - pair.1)

section Levels

variable (s : GkrShape) (batch : Batch K)

theorem treeLevel_pred {t k : ℕ} (low : 1 ≤ k) (high : k ≤ s.depth t) :
    treeLevel s batch t (k - 1) = (treeLevel s batch t k).up (2 ^ (k - 1)) := by
  unfold treeLevel
  rw [show s.depth t - (k - 1) = (s.depth t - k) + 1 by omega]
  simp only [Level.above]
  congr 2
  omega

theorem list_prod_getD (l : List (ℕ → K)) (y : ℕ) :
    (l.map fun f => f y).prod = ((List.range l.length).map fun j => l.getD j (fun _ => 0) y).prod := by
  induction l with
  | nil => simp
  | cons a rest ih =>
      rw [List.length_cons, List.range_succ_eq_map, List.map_cons, List.map_cons, List.prod_cons,
        List.prod_cons, List.map_map, ih]
      simp [Function.comp_def]

theorem numerator_parts (fits : batch.Fits s) {t k : ℕ} (tree : t < s.trees) (high : k ≤ s.depth t)
    (y : ℕ) :
    (treeLevel s batch t k).numerator y =
      ((List.range (s.partFactors k t)).map fun j => treePart s batch k t j y).prod := by
  have count : (treeLevel s batch t k).factors.length = s.partFactors k t := by
    unfold GkrShape.partFactors
    split
    · rename_i leaf
      simp only [treeLevel, leaf, Nat.sub_self, Level.above]
      exact fits t tree
    · rename_i inner
      obtain ⟨m, hm⟩ : ∃ m, s.depth t - k = m + 1 := ⟨s.depth t - k - 1, by omega⟩
      simp [treeLevel, hm, Level.above, Level.up]
  rw [Level.numerator, list_prod_getD, count]
  congr 1
  apply List.map_congr_left
  intro j member
  simp only [treePart, if_pos (List.mem_range.mp member)]

end Levels

section Comb

open MvPolynomial (X)

/-- Variables of a layer polynomial: the eq table, or half `b` of part `j` of tree `t`. -/
abbrev LayerVar := Option (ℕ × ℕ × Bool)

/-- One tree's term `p₀q₁ + p₁q₀ + μ·q₀q₁` with `f` numerator factors. -/
noncomputable def treeTerm (f t : ℕ) (μ : K) : MvPolynomial LayerVar K :=
  ((List.range f).map fun j => X (some (t, j, false))).prod * X (some (t, f, true)) +
    ((List.range f).map fun j => X (some (t, j, true))).prod * X (some (t, f, false)) +
    MvPolynomial.C μ * X (some (t, f, false)) * X (some (t, f, true))

/-- `Σ_i ν^i·term(lᵢ)` as a polynomial. -/
noncomputable def wsumM (ν : K) (term : ℕ → MvPolynomial LayerVar K) : List ℕ → MvPolynomial LayerVar K
  | [] => 0
  | t :: rest => term t + MvPolynomial.C ν * wsumM ν term rest

/-- The polynomial of layer `k`'s sum-check. -/
noncomputable def layerComb (s : GkrShape) (k : ℕ) (μ ν : K) : MvPolynomial LayerVar K :=
  X none * wsumM ν (fun t => treeTerm (s.partFactors k t) t μ) (s.active k)

/-- The tables of layer `k` at the GKR point `ρ`. -/
def layerTables (s : GkrShape) (batch : Batch K) (k : ℕ) (ρ : List K) : LayerVar → ℕ → K
  | none => eqAt ρ
  | some (t, j, b) => fun y => treePart s batch k t j (y + if b then 2 ^ (k - 1) else 0)

theorem eval_wsumM (v : LayerVar → K) (ν : K) (term : ℕ → MvPolynomial LayerVar K) (l : List ℕ) :
    MvPolynomial.eval v (wsumM ν term l) = wsum ν (fun t => MvPolynomial.eval v (term t)) l := by
  induction l with
  | nil => simp [wsumM, wsum]
  | cons t rest ih => simp [wsumM, wsum, ih]

theorem eval_treeTerm (v : LayerVar → K) (f t : ℕ) (μ : K) :
    MvPolynomial.eval v (treeTerm f t μ) =
      combineVals f (fun j => (v (some (t, j, false)), v (some (t, j, true)))) μ := by
  simp [treeTerm, combineVals, map_list_prod, List.map_map, Function.comp_def]

theorem totalDegree_treeTerm (f t : ℕ) (μ : K) : (treeTerm f t μ).totalDegree ≤ max (f + 1) 2 := by
  have product (b : Bool) :
      (((List.range f).map fun j => (X (some (t, j, b)) : MvPolynomial LayerVar K)).prod).totalDegree ≤ f := by
    refine (MvPolynomial.totalDegree_list_prod _).trans (le_of_eq ?_)
    rw [List.map_map]
    have each : (MvPolynomial.totalDegree ∘ fun j => (X (some (t, j, b)) : MvPolynomial LayerVar K)) =
        fun _ => 1 := by
      funext j
      exact MvPolynomial.totalDegree_X _
    rw [each, List.map_const', List.sum_replicate, List.length_range, smul_eq_mul, mul_one]
  have variable_ (v : LayerVar) : (X v : MvPolynomial LayerVar K).totalDegree ≤ 1 :=
    (MvPolynomial.totalDegree_X v).le
  unfold treeTerm
  refine (MvPolynomial.totalDegree_add _ _).trans (max_le ?_ ?_)
  · refine (MvPolynomial.totalDegree_add _ _).trans (max_le ?_ ?_)
    · exact ((MvPolynomial.totalDegree_mul _ _).trans (add_le_add (product false) (variable_ _))).trans
        (le_max_left _ _)
    · exact ((MvPolynomial.totalDegree_mul _ _).trans (add_le_add (product true) (variable_ _))).trans
        (le_max_left _ _)
  · refine (MvPolynomial.totalDegree_mul _ _).trans ?_
    refine (add_le_add ((MvPolynomial.totalDegree_mul _ _).trans
      (add_le_add (MvPolynomial.totalDegree_C μ).le (variable_ _))) (variable_ _)).trans ?_
    exact le_max_right _ _

theorem totalDegree_wsumM (ν : K) (term : ℕ → MvPolynomial LayerVar K) (l : List ℕ) (d : ℕ)
    (each : ∀ t ∈ l, (term t).totalDegree ≤ d) : (wsumM ν term l).totalDegree ≤ d := by
  induction l with
  | nil => simp [wsumM]
  | cons t rest ih =>
      refine (MvPolynomial.totalDegree_add _ _).trans (max_le (each t (by simp)) ?_)
      refine (MvPolynomial.totalDegree_mul _ _).trans ?_
      rw [MvPolynomial.totalDegree_C, zero_add]
      exact ih fun u inside => each u (List.mem_cons_of_mem t inside)

theorem totalDegree_layerComb (s : GkrShape) (k : ℕ) (μ ν : K) :
    (layerComb s k μ ν).totalDegree ≤ s.degree k := by
  unfold layerComb
  refine (MvPolynomial.totalDegree_mul _ _).trans ?_
  rw [MvPolynomial.totalDegree_X]
  have inner : (wsumM ν (fun t => treeTerm (s.partFactors k t) t μ) (s.active k)).totalDegree ≤
      s.degree k - 1 := by
    apply totalDegree_wsumM
    intro t member
    refine (totalDegree_treeTerm _ _ _).trans (max_le ?_ ?_)
    · have := s.partFactors_le_degree member
      omega
    · have := s.le_degree k
      omega
  have := s.le_degree k
  omega

end Comb

section Truth

variable (s : GkrShape) (batch : Batch K)

theorem cubeSum_mul_wsum (n : ℕ) (a : ℕ → K) (ν : K) (g : ℕ → ℕ → K) (l : List ℕ) :
    cubeSum n (fun y => a y * wsum ν (fun t => g t y) l) =
      wsum ν (fun t => cubeSum n (fun y => a y * g t y)) l := by
  induction l with
  | nil => simp only [wsum, mul_zero]; exact cubeSum_zero n
  | cons t rest ih =>
      have split : (fun y => a y * wsum ν (fun t => g t y) (t :: rest)) =
          fun y => a y * g t y + ν * (a y * wsum ν (fun t => g t y) rest) := by
        funext y
        simp only [wsum]
        ring
      rw [split, cubeSum_add, cubeSum_mul_left, ih]
      rfl

/-- At the start of the layer, the sum is the batched true claim. -/
theorem layer_truth_nil (fits : batch.Fits s) (k : ℕ) (low : 1 ≤ k) (ρ : List K)
    (length : ρ.length = k - 1) (μ ν : K) :
    combTruth (k - 1) (layerTables s batch k ρ) (layerComb s k μ ν) [] =
      wsum ν (fun t => (trueClaim s batch k t ρ).1 + μ * (trueClaim s batch k t ρ).2) (s.active k) := by
  rw [combTruth_nil]
  have summand (y : ℕ) :
      MvPolynomial.eval (fun v => layerTables s batch k ρ v y) (layerComb s k μ ν) =
        eqAt ρ y * wsum ν (fun t => (treeLevel s batch t (k - 1)).numerator y +
          μ * (treeLevel s batch t (k - 1)).denominator y) (s.active k) := by
    simp only [layerComb, map_mul, MvPolynomial.eval_X, layerTables, eval_wsumM, eval_treeTerm]
    congr 1
    apply wsum_congr
    intro t member
    obtain ⟨tree, reach⟩ := (s.mem_active).mp member
    rw [treeLevel_pred s batch low reach, Level.up_numerator, Level.up_denominator,
      numerator_parts s batch fits tree reach, numerator_parts s batch fits tree reach]
    simp only [combineVals, if_false, if_true, Nat.add_zero, Bool.false_eq_true, treePart, lt_irrefl]
    ring
  simp_rw [summand]
  rw [cubeSum_mul_wsum]
  apply wsum_congr
  intro t _
  simp only [trueClaim]
  rw [mle_eq_cubeSum, mle_eq_cubeSum, length, ← cubeSum_mul_left, ← cubeSum_add]
  congr 1
  funext y
  ring

/-- At the end of the layer's sum-check, the polynomial is `eq` times the
batched combination of the true halves. -/
theorem layer_truth_final (k : ℕ) (ρ next : List K) (length : next.length = ρ.length) (μ ν : K) :
    MvPolynomial.eval (fun v => mle next (layerTables s batch k ρ v)) (layerComb s k μ ν) =
      eqEval ρ next *
        wsum ν (fun t => combineVals (s.partFactors k t) (trueChildren s batch k t next) μ) (s.active k) := by
  simp only [layerComb, map_mul, MvPolynomial.eval_X, layerTables, eval_wsumM, eval_treeTerm]
  rw [mle_eqAt ρ next length]
  rfl

/-- True halves reduce to the extension at `next ++ [τ]`. -/
theorem reduce_trueChildren (k t j : ℕ) (next : List K) (length : next.length = k - 1) (τ : K) :
    reduce (trueChildren s batch k t next j) τ = mle (next ++ [τ]) (treePart s batch k t j) := by
  rw [mle_append_top, length]
  simp only [reduce, trueChildren]
  ring

end Truth

end CompressionLean.Layer1
