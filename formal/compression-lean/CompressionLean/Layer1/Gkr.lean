import CompressionLean.Layer1.GkrLayer
import CompressionLean.Game.SumCheck

namespace CompressionLean.Layer1

open CompressionLean.Lib
open Polynomial

variable {K : Type} [Field K]

noncomputable def wpoly (δ : ℕ → K) : List ℕ → K[X]
  | [] => 0
  | t :: rest => C (δ t) + X * wpoly δ rest

theorem wpoly_eval (δ : ℕ → K) (ν : K) (l : List ℕ) : (wpoly δ l).eval ν = wsum ν δ l := by
  induction l with
  | nil => simp [wpoly, wsum]
  | cons t rest ih => simp [wpoly, wsum, ih]

theorem wpoly_natDegree (δ : ℕ → K) (l : List ℕ) : (wpoly δ l).natDegree ≤ l.length - 1 := by
  induction l with
  | nil => simp [wpoly]
  | cons t rest ih =>
      cases rest with
      | nil => simp [wpoly]
      | cons u more =>
          unfold wpoly
          refine (natDegree_add_le _ _).trans (max_le (by simp) ?_)
          refine (natDegree_mul_le).trans ?_
          have := natDegree_X_le (R := K)
          simp only [List.length_cons] at ih ⊢
          omega

theorem wpoly_ne_zero {δ : ℕ → K} {l : List ℕ} {t : ℕ} (member : t ∈ l) (live : δ t ≠ 0) :
    wpoly δ l ≠ 0 := by
  induction l with
  | nil => simp at member
  | cons u rest ih =>
      intro zero
      have zeroth := congrArg (fun p => p.coeff 0) zero
      simp only [wpoly, coeff_add, coeff_C_zero, coeff_X_mul_zero, add_zero, coeff_zero] at zeroth
      have tail : wpoly δ rest = 0 := by
        have shifted : X * wpoly δ rest = 0 := by
          have := zero
          simp only [wpoly] at this
          rw [zeroth, C_0, zero_add] at this
          exact this
        exact (mul_eq_zero.mp shifted).resolve_left X_ne_zero
      rcases List.mem_cons.mp member with rfl | inside
      · exact live zeroth
      · exact ih inside tail

theorem wsum_sub (ν : K) (f g : ℕ → K) (l : List ℕ) :
    wsum ν f l - wsum ν g l = wsum ν (fun t => f t - g t) l := by
  induction l with
  | nil => simp [wsum]
  | cons t rest ih => simp only [wsum]; rw [← ih]; ring

theorem exists_ne_of_wsum_ne {ν : K} {f g : ℕ → K} {l : List ℕ} (different : wsum ν f l ≠ wsum ν g l) :
    ∃ t ∈ l, f t ≠ g t := by
  by_contra none
  push Not at none
  exact different (wsum_congr ν none)

theorem exists_ne_of_combineVals_ne {pairs pairs' : ℕ → K × K} {μ : K} {n : ℕ}
    (different : combineVals n pairs μ ≠ combineVals n pairs' μ) : ∃ j ≤ n, pairs j ≠ pairs' j := by
  by_contra none
  push Not at none
  apply different
  unfold combineVals
  have same : ∀ j ∈ List.range n, pairs j = pairs' j :=
    fun j member => none j (Nat.le_of_lt (List.mem_range.mp member))
  rw [List.map_congr_left (fun j m => congrArg Prod.fst (same j m)),
    List.map_congr_left (fun j m => congrArg Prod.snd (same j m)), none n le_rfl]

structure GkrClaims (K : Type) where
  roots : ℕ → K × K
  point : List K
  current : ℕ → K × K
  leafPoint : ℕ → List K
  leafValues : ℕ → ℕ → K

def advance (s : GkrShape) (k : ℕ) (st : GkrClaims K) (next : List K) (children : ℕ → ℕ → K × K)
    (τ : K) : GkrClaims K where
  roots := st.roots
  point := next ++ [τ]
  current := fun t =>
    if k < s.depth t then (reduce (children t 0) τ, reduce (children t 1) τ) else st.current t
  leafPoint := fun t => if s.depth t = k then next ++ [τ] else st.leafPoint t
  leafValues := fun t j => if s.depth t = k then reduce (children t j) τ else st.leafValues t j

variable [Fintype K] [DecidableEq K] {W : Type}

section Bad

variable (s : GkrShape) (batch : W → Batch K) (k : ℕ) (st : GkrClaims K)

open Classical in
noncomputable def badMu (w : W) : Finset K :=
  if h : ∃ t ∈ s.active k, st.current t ≠ trueClaim s (batch w) k t st.point then
    zeros (linePoly ((st.current h.choose).1 - (trueClaim s (batch w) k h.choose st.point).1)
      ((st.current h.choose).2 - (trueClaim s (batch w) k h.choose st.point).2))
  else ∅

noncomputable def badNu (μ : K) (w : W) : Finset K :=
  zeros (wpoly (fun t => ((st.current t).1 + μ * (st.current t).2) -
    ((trueClaim s (batch w) k t st.point).1 + μ * (trueClaim s (batch w) k t st.point).2)) (s.active k))

open Classical in
noncomputable def badTau (next : List K) (children : ℕ → ℕ → K × K) (w : W) : Finset K :=
  if h : ∃ t ∈ s.active k, ∃ j ≤ s.partFactors k t,
      children t j ≠ trueChildren s (batch w) k t next j then
    let t := h.choose
    let j := h.choose_spec.2.choose
    let pair := trueChildren s (batch w) k t next j
    zeros (linePoly ((children t j).1 - pair.1)
      (((children t j).2 - (children t j).1) - (pair.2 - pair.1)))
  else ∅

theorem badMu_spec {w : W} (h : ∃ t ∈ s.active k, st.current t ≠ trueClaim s (batch w) k t st.point) :
    ∃ t ∈ s.active k, st.current t ≠ trueClaim s (batch w) k t st.point ∧
      badMu s batch k st w =
        zeros (linePoly ((st.current t).1 - (trueClaim s (batch w) k t st.point).1)
          ((st.current t).2 - (trueClaim s (batch w) k t st.point).2)) := by
  classical
  exact ⟨h.choose, h.choose_spec.1, h.choose_spec.2, by simp only [badMu, dite_eq_left h]⟩

theorem badTau_spec {next : List K} {children : ℕ → ℕ → K × K} {w : W}
    (h : ∃ t ∈ s.active k, ∃ j ≤ s.partFactors k t, children t j ≠ trueChildren s (batch w) k t next j) :
    ∃ t ∈ s.active k, ∃ j ≤ s.partFactors k t, children t j ≠ trueChildren s (batch w) k t next j ∧
      badTau s batch k next children w =
        zeros (linePoly ((children t j).1 - (trueChildren s (batch w) k t next j).1)
          (((children t j).2 - (children t j).1) -
            ((trueChildren s (batch w) k t next j).2 - (trueChildren s (batch w) k t next j).1))) := by
  classical
  exact ⟨h.choose, h.choose_spec.1, h.choose_spec.2.choose, h.choose_spec.2.choose_spec.1,
    h.choose_spec.2.choose_spec.2, by simp only [badTau, dite_eq_left h]⟩

end Bad

noncomputable def gkrLayers (s : GkrShape) (worlds : Finset W) (batch : W → Batch K)
    (extra : ℕ → List K → Finset K) : ℕ → ℕ → GkrClaims K → Game (Option (GkrClaims K))
  | 0, _, st => .pure (some st)
  | remaining + 1, k, st =>
      .draw K (worlds.biUnion (badMu s batch k st)) fun μ =>
      .draw K (worlds.biUnion (badNu s batch k st μ)) fun ν =>
      Game.bind
        (Game.sumcheck worlds (s.degree k - 1)
          (fun w => combRound (k - 1) (layerTables s (batch w) k st.point) (layerComb s k μ ν))
          (extra k) (k - 1) [] (wsum ν (fun t => (st.current t).1 + μ * (st.current t).2) (s.active k)))
        fun result =>
          .recv (ℕ → ℕ → K × K) fun children =>
            if result.2 = eqEval st.point result.1 *
                wsum ν (fun t => combineVals (s.partFactors k t) (children t) μ) (s.active k) then
              .draw K (worlds.biUnion (badTau s batch k result.1 children) ∪ extra k result.1) fun τ =>
                gkrLayers s worlds batch extra remaining (k + 1) (advance s k st result.1 children τ)
            else .pure none

noncomputable def gkr (s : GkrShape) (worlds : Finset W) (batch : W → Batch K)
    (extra : ℕ → List K → Finset K) : Game (Option (GkrClaims K)) :=
  .recv (ℕ → K × K) fun roots =>
    gkrLayers s worlds batch extra s.layers 1
      { roots, point := [], current := roots, leafPoint := fun _ => [], leafValues := fun _ _ => 0 }

section Soundness

variable (s : GkrShape) (batch : Batch K)

def trueLeaf (t : ℕ) (point : List K) (j : ℕ) : K :=
  mle point (treePart s batch (s.depth t) t j)

def trueRoot (t : ℕ) : K × K :=
  ((treeLevel s batch t 0).numerator 0, (treeLevel s batch t 0).denominator 0)

def Doomed (k : ℕ) (st : GkrClaims K) : Prop :=
  (∃ t ∈ s.active k, st.current t ≠ trueClaim s batch k t st.point) ∨
    ∃ t, t < s.trees ∧ s.depth t < k ∧ ∃ j ≤ s.factors t,
      st.leafValues t j ≠ trueLeaf s batch t (st.leafPoint t) j

def LeafFalse (st : GkrClaims K) : Prop :=
  ∃ t, t < s.trees ∧ ∃ j ≤ s.factors t, st.leafValues t j ≠ trueLeaf s batch t (st.leafPoint t) j

omit [Field K] [Fintype K] [DecidableEq K] in
theorem Path.append_single {extra : List K → Finset K} {point : List K} :
    ∀ {challenges : List K} {c : K}, Path extra point challenges →
      c ∉ extra (point ++ challenges) → Path extra point (challenges ++ [c]) := by
  intro challenges
  induction challenges generalizing point with
  | nil => intro c _ outside; simpa [Path] using outside
  | cons d rest ih =>
      intro c path outside
      exact ⟨path.1, ih path.2 (by simpa using outside)⟩

theorem doomed_step (fits : batch.Fits s) {k : ℕ} (low : 1 ≤ k) {st : GkrClaims K}
    (length : st.point.length = k - 1) {μ ν τ : K} {next : List K} {last : K}
    {children : ℕ → ℕ → K × K} (nextLength : next.length = k - 1)
    (hμ : μ ∉ badMu s (fun _ : Unit => batch) k st ())
    (hν : ν ∉ badNu s (fun _ : Unit => batch) k st μ ())
    (transfer : wsum ν (fun t => (st.current t).1 + μ * (st.current t).2) (s.active k) ≠
        combTruth (k - 1) (layerTables s batch k st.point) (layerComb s k μ ν) [] →
      last ≠ combTruth (k - 1) (layerTables s batch k st.point) (layerComb s k μ ν) next)
    (check : last = eqEval st.point next *
      wsum ν (fun t => combineVals (s.partFactors k t) (children t) μ) (s.active k))
    (hτ : τ ∉ badTau s (fun _ : Unit => batch) k next children ())
    (doomed : Doomed s batch k st) : Doomed s batch (k + 1) (advance s k st next children τ) := by
  classical
  rcases doomed with ⟨t, member, false_claim⟩ | ⟨t, tree, ended, j, le, false_leaf⟩
  ·
    obtain ⟨t₁, t₁_member, t₁_false, mu_bad⟩ :=
      badMu_spec s (fun _ : Unit => batch) k st (w := ()) ⟨t, member, false_claim⟩
    have mu_live : (st.current t₁).1 - (trueClaim s batch k t₁ st.point).1 +
        μ * ((st.current t₁).2 - (trueClaim s batch k t₁ st.point).2) ≠ 0 := by
      have nonzero : linePoly ((st.current t₁).1 - (trueClaim s batch k t₁ st.point).1)
          ((st.current t₁).2 - (trueClaim s batch k t₁ st.point).2) ≠ 0 := by
        apply linePoly_ne_zero
        by_contra both
        push Not at both
        exact t₁_false (Prod.ext (sub_eq_zero.mp both.1) (sub_eq_zero.mp both.2))
      intro zero
      apply hμ
      rw [mu_bad]
      unfold zeros
      rw [ite_eq_right nonzero]
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, linePoly_eval]
      exact zero
    have batch_false : wsum ν (fun t => (st.current t).1 + μ * (st.current t).2) (s.active k) ≠
        combTruth (k - 1) (layerTables s batch k st.point) (layerComb s k μ ν) [] := by
      rw [layer_truth_nil s batch fits k low st.point length]
      intro equal
      rw [← sub_eq_zero, wsum_sub, ← wpoly_eval] at equal
      apply hν
      unfold badNu zeros
      rw [ite_eq_right (wpoly_ne_zero t₁_member (by intro h; apply mu_live; rw [← h]; ring))]
      simpa using equal
    have last_false := transfer batch_false
    rw [show k - 1 = next.length by omega, combTruth_final,
      layer_truth_final s batch k st.point next (by omega), check] at last_false
    have sums_differ :
        wsum ν (fun t => combineVals (s.partFactors k t) (children t) μ) (s.active k) ≠
          wsum ν (fun t => combineVals (s.partFactors k t) (trueChildren s batch k t next) μ)
            (s.active k) := by
      intro equal
      exact last_false (by rw [equal])
    obtain ⟨t₂, member₂, combined⟩ := exists_ne_of_wsum_ne sums_differ
    obtain ⟨j₂, le₂, pair_false⟩ := exists_ne_of_combineVals_ne combined
    obtain ⟨t₃, t₃_member, j₃, j₃_le, j₃_false, tau_bad⟩ :=
      badTau_spec s (fun _ : Unit => batch) k (w := ()) ⟨t₂, member₂, j₂, le₂, pair_false⟩
    have reduced_false : reduce (children t₃ j₃) τ ≠ mle (next ++ [τ]) (treePart s batch k t₃ j₃) := by
      rw [← reduce_trueChildren s batch k t₃ j₃ next nextLength τ]
      have nonzero : linePoly ((children t₃ j₃).1 - (trueChildren s batch k t₃ next j₃).1)
          (((children t₃ j₃).2 - (children t₃ j₃).1) -
            ((trueChildren s batch k t₃ next j₃).2 - (trueChildren s batch k t₃ next j₃).1)) ≠ 0 := by
        apply linePoly_ne_zero
        by_contra both
        push Not at both
        apply j₃_false
        have first := sub_eq_zero.mp both.1
        have second := sub_eq_zero.mp both.2
        rw [first] at second
        exact Prod.ext first (sub_left_inj.mp second)
      intro equal
      apply hτ
      rw [tau_bad]
      unfold zeros
      rw [ite_eq_right nonzero]
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, linePoly_eval]
      simp only [reduce] at equal
      rw [← sub_eq_zero] at equal
      rw [← equal]
      ring
    obtain ⟨tree₃, reach₃⟩ := (s.mem_active).mp t₃_member
    rcases Nat.lt_or_ge k (s.depth t₃) with inner | leaf
    ·
      left
      refine ⟨t₃, (s.mem_active).mpr ⟨tree₃, inner⟩, ?_⟩
      have one : s.partFactors k t₃ = 1 := by simp [GkrShape.partFactors, Nat.ne_of_lt' inner]
      intro equal
      simp only [advance, ite_eq_left inner, trueClaim, Nat.add_sub_cancel] at equal
      have numerator : (treeLevel s batch t₃ k).numerator = treePart s batch k t₃ 0 := by
        funext y
        rw [numerator_parts s batch fits tree₃ reach₃, one]
        simp
      have denominator : (treeLevel s batch t₃ k).denominator = treePart s batch k t₃ 1 := by
        funext y
        simp [treePart, one]
      rw [numerator, denominator] at equal
      rw [one] at j₃_le
      rcases Nat.le_one_iff_eq_zero_or_eq_one.mp j₃_le with zero | one'
      · exact reduced_false (by rw [zero] at *; exact (Prod.ext_iff.mp equal).1)
      · exact reduced_false (by rw [one'] at *; exact (Prod.ext_iff.mp equal).2)
    ·
      have depth : s.depth t₃ = k := le_antisymm leaf reach₃
      right
      refine ⟨t₃, tree₃, by omega, j₃, ?_, ?_⟩
      · simpa [GkrShape.partFactors, depth] using j₃_le
      · simp only [advance, trueLeaf, depth, ite_true]
        exact reduced_false
  ·
    right
    have other : s.depth t ≠ k := by omega
    refine ⟨t, tree, by omega, j, le, ?_⟩
    simp only [advance]
    rw [ite_eq_right other, ite_eq_right other]
    exact false_leaf

variable {s} in
theorem gkrLayers_avoids (worlds : Finset W) (batch : W → Batch K) (extra : ℕ → List K → Finset K)
    (fits : ∀ w ∈ worlds, (batch w).Fits s)
    (distinct : ∀ k, 1 ≤ k → k ≤ s.layers → NodesDistinct K (s.degree k - 1)) :
    ∀ (remaining k : ℕ) (st : GkrClaims K), 1 ≤ k → k + remaining = s.layers + 1 →
      st.point.length = k - 1 →
      (∀ t, t < s.trees → s.depth t < k →
        (st.leafPoint t).length = s.depth t ∧ Path (extra (s.depth t)) [] (st.leafPoint t)) →
      (∀ t t', s.depth t < k → s.depth t = s.depth t' → st.leafPoint t = st.leafPoint t') →
      (gkrLayers s worlds batch extra remaining k st).Avoids fun r =>
        r = none ∨ ∃ fin, r = some fin ∧ fin.roots = st.roots ∧
          (∀ w ∈ worlds, Doomed s (batch w) k st → LeafFalse s (batch w) fin) ∧
          (∀ t, t < s.trees → s.depth t ≤ s.layers →
            (fin.leafPoint t).length = s.depth t ∧ Path (extra (s.depth t)) [] (fin.leafPoint t)) ∧
          ∀ t t', s.depth t ≤ s.layers → s.depth t = s.depth t' → fin.leafPoint t = fin.leafPoint t' := by
  intro remaining
  induction remaining with
  | zero =>
      intro k st _ last _ ended shared
      refine Or.inr ⟨st, rfl, rfl, ?_, fun t tree deep => ended t tree (by omega),
        fun t t' deep same => shared t t' (by omega) same⟩
      intro w _ doomed
      rcases doomed with ⟨t, member, _⟩ | ⟨t, tree, _, j, le, false_leaf⟩
      · obtain ⟨tree, reach⟩ := (s.mem_active).mp member
        have := s.depth_le_layers tree
        omega
      · exact ⟨t, tree, j, le, false_leaf⟩
  | succ remaining ih =>
      intro k st low fits_layers length ended shared μ hμ ν hν
      have inside : k ≤ s.layers := by omega
      rw [Game.avoids_bind]
      have degree : ∀ μ ν : K, (layerComb s k μ ν).totalDegree ≤ s.degree k - 1 + 1 := by
        intro μ ν
        have := s.le_degree k
        have := totalDegree_layerComb s k μ ν
        omega
      refine (Game.sumcheck_avoids worlds (s.degree k - 1) (distinct k low inside)
        (fun w => combTruth (k - 1) (layerTables s (batch w) k st.point) (layerComb s k μ ν))
        (fun w => combRound (k - 1) (layerTables s (batch w) k st.point) (layerComb s k μ ν))
        (extra k) (k - 1)
        (fun w _ => combRound_truth (k - 1) (s.degree k - 1) _ _ (degree μ ν))
        (k - 1) [] _ (by simp)).mono ?_
      rintro ⟨next, last⟩ ⟨⟨challenges, shape, count, path⟩, transfer⟩ children
      simp only [List.nil_append] at shape transfer
      subst shape
      dsimp only
      split_ifs with check
      · intro τ hτ
        have stays : ∀ t, t < s.trees → s.depth t < k + 1 →
            ((advance s k st next children τ).leafPoint t).length = s.depth t ∧
              Path (extra (s.depth t)) [] ((advance s k st next children τ).leafPoint t) := by
          intro t tree shallow
          by_cases here : s.depth t = k
          · simp only [advance, ite_eq_left here]
            refine ⟨by simp [count]; omega, ?_⟩
            rw [here]
            exact Path.append_single path
              (fun hit => hτ (Finset.mem_union_right _ (by simpa using hit)))
          · simp only [advance, ite_eq_right here]
            exact ended t tree (by omega)
        have stays_shared : ∀ t t', s.depth t < k + 1 → s.depth t = s.depth t' →
            (advance s k st next children τ).leafPoint t = (advance s k st next children τ).leafPoint t' := by
          intro t t' shallow same
          by_cases here : s.depth t = k
          · simp only [advance, ite_eq_left here, ite_eq_left (same ▸ here)]
          · simp only [advance, ite_eq_right here, ite_eq_right (same ▸ here)]
            exact shared t t' (by omega) same
        refine (ih (k + 1) _ (by omega) (by omega) (by simp [advance, count]; omega) stays
          stays_shared).mono ?_
        rintro r (none | ⟨fin, rfl, roots, doomed, structural, shared'⟩)
        · exact Or.inl none
        refine Or.inr ⟨fin, rfl, roots, ?_, structural, shared'⟩
        intro w member before
        apply doomed w member
        exact doomed_step s (batch w) (fits w member) low length (by omega)
          (fun hit => hμ (Finset.mem_biUnion.mpr ⟨w, member, hit⟩))
          (fun hit => hν (Finset.mem_biUnion.mpr ⟨w, member, hit⟩))
          (transfer w member) check
          (fun hit => hτ (Finset.mem_union_left _ (Finset.mem_biUnion.mpr ⟨w, member, hit⟩)))
          before
      · exact Or.inl rfl

variable {s} in
theorem gkr_avoids (worlds : Finset W) (batch : W → Batch K) (extra : ℕ → List K → Finset K)
    (fits : ∀ w ∈ worlds, (batch w).Fits s) (deep : ∀ t, t < s.trees → 1 ≤ s.depth t)
    (distinct : ∀ k, 1 ≤ k → k ≤ s.layers → NodesDistinct K (s.degree k - 1)) :
    (gkr s worlds batch extra).Avoids fun r =>
      r = none ∨ ∃ fin, r = some fin ∧
        (∀ w ∈ worlds, (∃ t, t < s.trees ∧ fin.roots t ≠ trueRoot s (batch w) t) →
          LeafFalse s (batch w) fin) ∧
        (∀ t, t < s.trees →
          (fin.leafPoint t).length = s.depth t ∧ Path (extra (s.depth t)) [] (fin.leafPoint t)) ∧
        ∀ t t', t < s.trees → s.depth t = s.depth t' → fin.leafPoint t = fin.leafPoint t' := by
  intro roots
  refine (gkrLayers_avoids worlds batch extra fits distinct s.layers 1 _ le_rfl (by omega) rfl
    (fun t tree shallow => absurd (deep t tree) (by omega))
    (fun t t' _ _ => rfl)).mono ?_
  rintro r (none | ⟨fin, rfl, root_eq, doomed, structural, shared⟩)
  · exact Or.inl none
  refine Or.inr ⟨fin, rfl, ?_, fun t tree => structural t tree (s.depth_le_layers tree),
    fun t t' tree same => shared t t' (s.depth_le_layers tree) same⟩
  intro w member ⟨t, tree, false_root⟩
  apply doomed w member
  left
  refine ⟨t, (s.mem_active).mpr ⟨tree, deep t tree⟩, ?_⟩
  rw [root_eq] at false_root
  simpa [trueClaim, trueRoot, mle, fold] using false_root

end Soundness

section Cost

variable (s : GkrShape)

def gkrLayerCost (n : ℕ) (E : ℕ → ℕ) (k : ℕ) : ℕ :=
  n + n * ((s.active k).length - 1) + (k - 1) * (n * s.degree k + E k) + (n + E k)

def gkrCost (n : ℕ) (E : ℕ → ℕ) : ℕ → ℕ → ℕ
  | 0, _ => 0
  | remaining + 1, k => gkrLayerCost s n E k + gkrCost n E remaining (k + 1)

variable {s}

theorem card_badMu (batch : W → Batch K) (k : ℕ) (st : GkrClaims K) (worlds : Finset W) :
    (worlds.biUnion (badMu s batch k st)).card ≤ worlds.card * 1 := by
  apply card_biUnion_le_mul
  intro w _
  unfold badMu
  split
  · exact zeros_card_le_of_natDegree_le (linePoly_natDegree _ _)
  · simp

theorem card_badNu (batch : W → Batch K) (k : ℕ) (st : GkrClaims K) (μ : K) (worlds : Finset W) :
    (worlds.biUnion (badNu s batch k st μ)).card ≤ worlds.card * ((s.active k).length - 1) :=
  card_biUnion_le_mul _ _ _ fun _ _ => zeros_card_le_of_natDegree_le (wpoly_natDegree _ _)

theorem card_badTau (batch : W → Batch K) (k : ℕ) (next : List K) (children : ℕ → ℕ → K × K)
    (worlds : Finset W) : (worlds.biUnion (badTau s batch k next children)).card ≤ worlds.card * 1 := by
  apply card_biUnion_le_mul
  intro w _
  unfold badTau
  split
  · exact zeros_card_le_of_natDegree_le (linePoly_natDegree _ _)
  · simp

theorem gkrLayers_cost (worlds : Finset W) (batch : W → Batch K) (extra : ℕ → List K → Finset K)
    (E : ℕ → ℕ) (extra_card : ∀ k point, (extra k point).card ≤ E k)
    (distinct : ∀ k, 1 ≤ k → k ≤ s.layers → NodesDistinct K (s.degree k - 1)) :
    ∀ (remaining k : ℕ) (st : GkrClaims K), 1 ≤ k → k + remaining = s.layers + 1 →
      (gkrLayers s worlds batch extra remaining k st).CostLe
        (gkrCost s worlds.card E remaining k / (Fintype.card K : ℝ)) := by
  intro remaining
  induction remaining with
  | zero => intro k st _ _; simp [gkrLayers, gkrCost, Game.CostLe]
  | succ remaining ih =>
      intro k st low fits_layers
      have inside : k ≤ s.layers := by omega
      have three := s.le_degree k
      have positive : (0 : ℝ) < Fintype.card K := by exact_mod_cast Fintype.card_pos
      set n := worlds.card
      set later := gkrCost s n E remaining (k + 1)
      set rounds := (k - 1) * (n * s.degree k + E k)
      have total : gkrCost s n E (remaining + 1) k =
          n * 1 + (n * ((s.active k).length - 1) + (rounds + ((n * 1 + E k) + later))) := by
        simp only [gkrCost, gkrLayerCost, rounds, later]
        ring
      rw [total]
      refine Game.draw_step _ _ _ _ (Game.density_le _ _ (card_badMu batch k st worlds)) fun μ => ?_
      refine Game.draw_step _ _ _ _ (Game.density_le _ _ (card_badNu batch k st μ worlds)) fun ν => ?_
      rw [Nat.cast_add, add_div]
      refine Game.costLe_bind ?_ ?_
      · have cost := Game.sumcheck_cost worlds (s.degree k - 1) (distinct k low inside)
          (fun w => combRound (k - 1) (layerTables s (batch w) k st.point) (layerComb s k μ ν))
          (extra k) (k - 1) (E k)
          (fun w _ point short => (combRound_truth (k - 1) (s.degree k - 1) _ _
            (by have := totalDegree_layerComb s k μ ν; omega)).degree point short)
          (extra_card k) (k - 1) [] (wsum ν (fun t => (st.current t).1 + μ * (st.current t).2) (s.active k))
          (by simp)
        refine cost.mono (le_of_eq ?_)
        rw [show s.degree k - 1 + 1 = s.degree k by omega]
        simp only [rounds]
        push_cast
        ring
      · intro result children
        dsimp only
        split_ifs
        · refine Game.draw_step _ _ (n * 1 + E k) _ (Game.density_le _ _ ?_) fun τ => ih (k + 1) _ (by omega) (by omega)
          exact (Finset.card_union_le _ _).trans (add_le_add (card_badTau batch k _ children worlds)
            (extra_card k _))
        · exact div_nonneg (Nat.cast_nonneg _) positive.le

theorem gkr_cost (worlds : Finset W) (batch : W → Batch K) (extra : ℕ → List K → Finset K)
    (E : ℕ → ℕ) (extra_card : ∀ k point, (extra k point).card ≤ E k)
    (distinct : ∀ k, 1 ≤ k → k ≤ s.layers → NodesDistinct K (s.degree k - 1)) :
    (gkr s worlds batch extra).CostLe (gkrCost s worlds.card E s.layers 1 / (Fintype.card K : ℝ)) :=
  fun _ => gkrLayers_cost worlds batch extra E extra_card distinct s.layers 1 _ le_rfl (by omega)

end Cost

end CompressionLean.Layer1
