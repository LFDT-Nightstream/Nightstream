import CompressionLean.Layer1.Early
import Mathlib.Tactic.Linarith

namespace CompressionLean.Layer1

open CompressionLean.Lib
open Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

def SetupTrue (s : Shape) (S : Setup F) (o : EarlyOut F K) : Prop :=
  o.rowClaim = mle o.runPoint (rowColumn S) ∧
    o.slotClaim s S = mle o.runPoint (slotColumn S) ∧
    o.coefClaim = mle o.runPoint (coefColumn S) ∧
    o.embeddedBlock s = mle o.runPoint (blockColumn S)

def EarlyConsistent (s : Shape) (o : EarlyOut F K) (w : P0Tables F) : Prop :=
  mle o.normPoint (fun x => ι (w.z x)) = o.zLeaf ∧
    mle o.runPoint (fun k => ι (w.e k).1) = o.values.eRe ∧
    mle o.runPoint (fun k => ι (w.e k).2) = o.values.eIm ∧
    (∀ j, j < s.matrices →
      mle (o.pairPoint.take s.slotVars) (fun σ => ι (w.U j σ).1) = (o.values.pairs j).1 ∧
        mle (o.pairPoint.take s.slotVars) (fun σ => ι (w.U j σ).2) = (o.values.pairs j).2) ∧
    mle (o.runPoint.take s.slotVars) (fun σ => ι (w.blockCopy σ)) = o.values.block ∧
    mle o.rowPoint (fun i => ι (w.mult i)) = o.multValue

theorem treePart_leaf (s : GkrShape) (batch : Batch K) (t j : ℕ) :
    treePart s batch (s.depth t) t j =
      if j < s.factors t then (batch t).factors.getD j (fun _ => 0) else (batch t).denominator := by
  simp [treePart, GkrShape.partFactors, treeLevel, Level.above]

theorem trueRoot_eq (s : GkrShape) (batch : Batch K) (t : ℕ) :
    trueRoot s batch t =
      (((batch t).above (s.depth t) (s.depth t)).numerator 0,
        ((batch t).above (s.depth t) (s.depth t)).denominator 0) := by
  simp [trueRoot, treeLevel]

theorem mle_tuple (p : List K) (a γ : K) (A B C : ℕ → K) :
    mle p (fun k => a - tuple γ (A k) (B k) (C k)) = a - tuple γ (mle p A) (mle p B) (mle p C) := by
  have expand : (fun k => a - tuple γ (A k) (B k) (C k)) =
      fun k => a - (A k + (γ * B k + γ * γ * C k)) := by
    funext k
    simp only [tuple]
    ring
  rw [expand, mle_sub, mle_const, mle_add, mle_add, mle_mul_left, mle_mul_left]
  simp only [tuple]
  ring

section Leaves

variable (s : Shape) (S : Setup F) (st : Statement F) (o : EarlyOut F K) (w : P0Tables F)

noncomputable def earlyLeaf (t j : ℕ) : K :=
  trueLeaf (earlyShape s) (earlyBatch s S st o.challenges w) t (o.gkr.leafPoint t) j

theorem earlyLeaf_eq (t j : ℕ) :
    earlyLeaf s S st o w t j =
      mle (o.gkr.leafPoint t)
        (if j < (earlyShape s).factors t then
          (earlyBatch s S st o.challenges w t).factors.getD j (fun _ => 0)
        else (earlyBatch s S st o.challenges w t).denominator) := by
  unfold earlyLeaf trueLeaf
  rw [treePart_leaf]

end Leaves

theorem mle_pairFactor (s : Shape) (point : List K) (μ : K) (U : ℕ → ℕ → K × K)
    (length : point.length = s.slotVars + s.matrixVars) (matrices_le : s.matrices ≤ 2 ^ s.matrixVars) :
    mle point (fun x => if x / 2 ^ s.slotVars < s.matrices then
        (U (x / 2 ^ s.slotVars) (x % 2 ^ s.slotVars)).1 +
          μ * (U (x / 2 ^ s.slotVars) (x % 2 ^ s.slotVars)).2 else 0) =
      ∑ j ∈ Finset.range s.matrices, eqAt (point.drop s.slotVars) j *
        (mle (point.take s.slotVars) (fun σ => (U j σ).1) +
          μ * mle (point.take s.slotVars) (fun σ => (U j σ).2)) := by
  have split : point = point.take s.slotVars ++ point.drop s.slotVars := (List.take_append_drop _ _).symm
  have take_length : (point.take s.slotVars).length = s.slotVars := by
    rw [List.length_take]; omega
  have drop_length : (point.drop s.slotVars).length = s.matrixVars := by
    rw [List.length_drop]; omega
  conv_lhs => rw [split]
  rw [mle_append_split, take_length, mle_eq_cubeSum, cubeSum_eq_sum_range, drop_length]
  rw [← Finset.sum_subset (Finset.range_mono matrices_le)]
  · apply Finset.sum_congr rfl
    intro j inside
    congr 1
    rw [← mle_mul_left, ← mle_add]
    apply mle_congr
    intro y lt
    rw [take_length] at lt
    have divide : (y + 2 ^ s.slotVars * j) / 2 ^ s.slotVars = j := by
      rw [Nat.add_mul_div_left _ _ (by positivity), Nat.div_eq_of_lt lt, zero_add]
    have remainder : (y + 2 ^ s.slotVars * j) % 2 ^ s.slotVars = y := by
      rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt lt]
    rw [divide, remainder, ite_eq_left (Finset.mem_range.mp inside)]
  · intro j _ outside
    rw [mul_eq_zero]
    right
    calc _ = mle (point.take s.slotVars) (fun _ => (0 : K)) := by
          apply mle_congr
          intro y lt
          rw [take_length] at lt
          have divide : (y + 2 ^ s.slotVars * j) / 2 ^ s.slotVars = j := by
            rw [Nat.add_mul_div_left _ _ (by positivity), Nat.div_eq_of_lt lt, zero_add]
          rw [divide, ite_eq_right (fun h => outside (Finset.mem_range.mpr h))]
      _ = 0 := mle_const _ _

section Trees

variable (s : Shape) (S : Setup F) (st : Statement F) (ch : EarlyChallenges K) (w : P0Tables F)

theorem early_factors : (earlyShape s).factors 0 = 0 ∧ (earlyShape s).factors 1 = 0 ∧
    (earlyShape s).factors 2 = 1 ∧ (earlyShape s).factors 3 = 2 ∧ (earlyShape s).factors 4 = 1 := by
  simp [earlyShape]

theorem early_depths : (earlyShape s).depth 0 = s.cubeVars ∧ (earlyShape s).depth 1 = s.runVars ∧
    (earlyShape s).depth 2 = s.rowVars ∧ (earlyShape s).depth 3 = s.runVars ∧
    (earlyShape s).depth 4 = s.slotVars + s.matrixVars := by
  simp [earlyShape]

theorem earlyBatch_0 : earlyBatch s S st ch w 0 = ⟨[], fun x => ch.beta - ι (w.z x)⟩ := by
  simp [earlyBatch]

theorem earlyBatch_1 : earlyBatch s S st ch w 1 =
    ⟨[], fun k => ch.lookup.1 - tuple ch.lookup.2 (rowColumn S k) (ι (w.e k).1) (ι (w.e k).2)⟩ := by
  simp [earlyBatch]

theorem earlyBatch_2 : earlyBatch s S st ch w 2 =
    ⟨[fun i => ι (w.mult i)],
      fun i => ch.lookup.1 - tuple ch.lookup.2 (i : K) (ι (st.chi i).1) (ι (st.chi i).2)⟩ := by
  simp [earlyBatch]

theorem earlyBatch_3 : earlyBatch s S st ch w 3 =
    ⟨[coefColumn S, fun k => ι (w.e k).1 + ch.scatter.2 * ι (w.e k).2],
      fun k => ch.scatter.1 - (slotColumn S k + (2 ^ s.slotVars : K) * matrixColumn S k)⟩ := by
  simp [earlyBatch]

theorem earlyBatch_4 : earlyBatch s S st ch w 4 =
    ⟨[fun x => if x / 2 ^ s.slotVars < s.matrices then
        ι (w.U (x / 2 ^ s.slotVars) (x % 2 ^ s.slotVars)).1 +
          ch.scatter.2 * ι (w.U (x / 2 ^ s.slotVars) (x % 2 ^ s.slotVars)).2 else 0],
      fun x => ch.scatter.1 - (x : K)⟩ := by
  simp [earlyBatch]

end Trees

theorem early_leaves_true (s : Shape) (S : Setup F) (st : Statement F) (o : EarlyOut F K)
    (w : P0Tables F) (checks : EarlyChecks s st o) (setup : SetupTrue s S o)
    (consistent : EarlyConsistent s o w) (shared : o.gkr.leafPoint 1 = o.gkr.leafPoint 3)
    (pairLength : (o.gkr.leafPoint 4).length = s.slotVars + s.matrixVars)
    (matrices_le : s.matrices ≤ 2 ^ s.matrixVars) :
    ¬ LeafFalse (earlyShape s) (earlyBatch s S st o.challenges w) o.gkr := by
  rintro ⟨t, tree, j, le, false_leaf⟩
  apply false_leaf
  change o.gkr.leafValues t j = earlyLeaf s S st o w t j
  rw [earlyLeaf_eq]
  obtain ⟨_, _, _, _, _, lookupRun, lookupRow, scatterRun, scatterPair, scatterIndex⟩ := checks
  obtain ⟨rowTrue, slotTrue, coefTrue, _⟩ := setup
  obtain ⟨zTrue, eReTrue, eImTrue, pairsTrue, _, multTrue⟩ := consistent
  have cases : t = 0 ∨ t = 1 ∨ t = 2 ∨ t = 3 ∨ t = 4 := by
    simp only [earlyShape] at tree
    omega
  obtain ⟨f0, f1, f2, f3, f4⟩ := early_factors s
  rcases cases with rfl | rfl | rfl | rfl | rfl
  · obtain rfl : j = 0 := by rw [f0] at le; omega
    rw [f0, earlyBatch_0, ite_eq_right (lt_irrefl 0), mle_sub, mle_const]
    unfold EarlyOut.normPoint EarlyOut.zLeaf at zTrue
    rw [zTrue]
    ring
  · obtain rfl : j = 0 := by rw [f1] at le; omega
    rw [f1, earlyBatch_1, ite_eq_right (lt_irrefl 0)]
    dsimp only
    rw [mle_tuple, shared]
    unfold EarlyOut.runPoint at eReTrue eImTrue rowTrue
    unfold EarlyOut.rowClaim at rowTrue
    rw [← rowTrue, eReTrue, eImTrue, lookupRun]
  · rw [f2] at le
    rw [f2, earlyBatch_2]
    rcases (show j = 0 ∨ j = 1 by omega) with rfl | rfl
    · rw [ite_eq_left (by norm_num)]
      simp only [List.getD_cons_zero]
      unfold EarlyOut.rowPoint EarlyOut.multValue at multTrue
      exact multTrue.symm
    · rw [ite_eq_right (lt_irrefl 1)]
      dsimp only
      rw [mle_tuple, lookupRow]
      rfl
  · rw [f3] at le
    rw [f3, earlyBatch_3]
    rcases (show j = 0 ∨ j = 1 ∨ j = 2 by omega) with rfl | rfl | rfl
    · rw [ite_eq_left (by norm_num)]
      simp only [List.getD_cons_zero]
      unfold EarlyOut.runPoint EarlyOut.coefClaim at coefTrue
      exact coefTrue
    · rw [ite_eq_left (by norm_num)]
      simp only [List.getD_cons_succ, List.getD_cons_zero]
      rw [mle_add, mle_mul_left, scatterRun]
      unfold EarlyOut.runPoint at eReTrue eImTrue
      rw [eReTrue, eImTrue]
    · rw [ite_eq_right (lt_irrefl 2)]
      dsimp only
      rw [mle_sub, mle_const, mle_add, mle_mul_left]
      unfold EarlyOut.slotClaim EarlyOut.runPoint at slotTrue
      rw [← slotTrue]
      ring
  · rw [f4] at le
    rw [f4, earlyBatch_4]
    rcases (show j = 0 ∨ j = 1 by omega) with rfl | rfl
    · rw [ite_eq_left (by norm_num)]
      simp only [List.getD_cons_zero]
      rw [scatterPair]
      have factor := mle_pairFactor (K := K) s (o.gkr.leafPoint 4) o.challenges.scatter.2
        (fun j σ => (ι (w.U j σ).1, ι (w.U j σ).2)) pairLength matrices_le
      simp only at factor
      rw [factor]
      apply Finset.sum_congr rfl
      intro j inside
      have pair := pairsTrue j (Finset.mem_range.mp inside)
      unfold EarlyOut.pairPoint at pair
      rw [pair.1, pair.2]
      rfl
    · rw [ite_eq_right (lt_irrefl 1)]
      dsimp only
      rw [mle_sub, mle_const, scatterIndex]
      rfl

def EarlyFacts (s : Shape) (S : Setup F) (st : Statement F) (w : P0Tables F) : Prop :=
  (∀ x, x < 2 ^ s.cubeVars → ∃ t, t < 2 * s.normBound + 1 ∧ ι (w.z x) = slotValue s.normBound t) ∧
    (∀ k, k < 2 ^ s.runVars → w.e k = st.chi (S.runRow k)) ∧
    (∀ j σ, j < s.matrices → σ < 2 ^ s.slotVars →
      ι (w.U j σ).1 = ∑ k ∈ (Finset.range (2 ^ s.runVars)).filter
          (fun k => S.runMatrix k = j ∧ S.runSlot k = σ), coefColumn S k * ι (w.e k).1 ∧
      ι (w.U j σ).2 = ∑ k ∈ (Finset.range (2 ^ s.runVars)).filter
          (fun k => S.runMatrix k = j ∧ S.runSlot k = σ), coefColumn S k * ι (w.e k).2) ∧
    ∀ σ, σ < 2 ^ s.slotVars → ι (w.blockCopy σ) = blockColumn S σ

theorem root_sum (s : GkrShape) (batch : Batch K) (t : ℕ) (root : K × K)
    (true_root : root = trueRoot s batch t) (nonzero : root.2 ≠ 0) :
    (∀ x, x < 2 ^ s.depth t → (batch t).denominator x ≠ 0) ∧
      root.1 / root.2 =
        ∑ x ∈ Finset.range (2 ^ s.depth t), (batch t).numerator x / (batch t).denominator x := by
  rw [true_root, trueRoot_eq] at nonzero ⊢
  exact Level.root_fraction _ _ nonzero

theorem pair_fraction {x y : K × K} (check : PairCheck x y) : x.1 / x.2 = y.1 / y.2 := by
  obtain ⟨hx, hy, cross⟩ := check
  rw [div_eq_div_iff hx hy, cross]

set_option backward.isDefEq.respectTransparency false in
theorem early_world (s : Shape) (S : Setup F) (st : Statement F) (o : EarlyOut F K) (w : P0Tables F)
    (N : ℕ) (cast : ∀ a b : ℕ, a < N → b < N → (a : K) = b → a = b)
    (big_cube : 2 ^ s.cubeVars < N) (big_runs : 2 ^ s.runVars < N) (big_rows : 2 ^ s.rowVars ≤ N)
    (big_pairs : 2 ^ (s.slotVars + s.matrixVars) ≤ N)
    (fits : S.Fits s) (slot_le_run : s.slotVars ≤ s.runVars) (matrices_le : s.matrices ≤ 2 ^ s.matrixVars)
    (checks : EarlyChecks s st o) (setup : SetupTrue s S o) (consistent : EarlyConsistent s o w)
    [Fintype K] [DecidableEq K]
    (hβ : o.challenges.beta ∉ zeros (normData s o.histogram w))
    (hL : o.challenges.lookup ∉ (lookupData s S st w).bad)
    (hS : o.challenges.scatter ∉ (scatterData s S w).bad)
    (gkrRoot : (∃ t, t < 5 ∧ o.gkr.roots t ≠ trueRoot (earlyShape s) (earlyBatch s S st o.challenges w) t) →
      LeafFalse (earlyShape s) (earlyBatch s S st o.challenges w) o.gkr)
    (shared : o.gkr.leafPoint 1 = o.gkr.leafPoint 3)
    (pairLength : (o.gkr.leafPoint 4).length = s.slotVars + s.matrixVars)
    (runLength : (o.gkr.leafPoint 3).length = s.runVars)
    (path : Path (blockTest s S w) [] o.runPoint) :
    EarlyFacts (K := K) s S st w := by
  have leaves := early_leaves_true s S st o w checks setup consistent shared pairLength matrices_le
  have roots : ∀ t, t < 5 → o.gkr.roots t = trueRoot (earlyShape s) (earlyBatch s S st o.challenges w) t := by
    by_contra wrong
    push Not at wrong
    exact leaves (gkrRoot wrong)
  obtain ⟨betaTable, root0, norm0, pair12, pair34, _⟩ := checks
  obtain ⟨d0, d1, d2, d3, d4⟩ := early_depths s
  have count : ∀ bound, bound < N → ∀ c : ℕ, 0 < c → c ≤ bound → (c : K) ≠ 0 := by
    intro bound big c positive le zero
    have := cast c 0 (by omega) (by omega) (by simpa using zero)
    omega
  refine ⟨?_, ?_, ?_, ?_⟩
  ·
    obtain ⟨poles, sum⟩ := root_sum (earlyShape s) _ 0 _ (roots 0 (by norm_num)) root0
    rw [d0, earlyBatch_0] at poles sum
    simp only [Level.numerator, List.map_nil, List.prod_nil] at sum
    have equal : ∑ x ∈ Finset.range (2 ^ s.cubeVars), 1 / (o.challenges.beta - ι (w.z x)) =
        tableSum s.normBound o.histogram o.challenges.beta := by
      rw [← sum, norm0, mul_div_cancel_left₀ _ root0]
    exact norm_sound s.cubeVars s.normBound (fun x => ι (w.z x)) (fun t => ι (o.histogram t))
      (count _ big_cube) o.challenges.beta hβ
      (fun x lt => sub_ne_zero.mp (poles x lt)) betaTable equal
  ·
    obtain ⟨poles1, sum1⟩ := root_sum (earlyShape s) _ 1 _ (roots 1 (by norm_num)) pair12.1
    obtain ⟨poles2, sum2⟩ := root_sum (earlyShape s) _ 2 _ (roots 2 (by norm_num)) pair12.2.1
    rw [d1, earlyBatch_1] at poles1 sum1
    rw [d2, earlyBatch_2] at poles2 sum2
    simp only [Level.numerator, List.map_nil, List.prod_nil, List.map_cons, List.prod_cons,
      mul_one] at sum1 sum2
    have inTable := (lookupData s S st w).sound (count _ big_runs) o.challenges.lookup.1
      o.challenges.lookup.2 hL (fun k lt => sub_ne_zero.mp (poles1 k lt))
      (fun i lt => sub_ne_zero.mp (poles2 i lt)) (by
        simp only [lookupData, Lookup.runTuple, Lookup.tableTuple]
        rw [← sum1, ← sum2, pair_fraction pair12])
    intro k lt
    obtain ⟨i, i_lt, same⟩ := inTable k lt
    simp only [lookupData, Prod.mk.injEq] at same
    obtain ⟨row, re, im⟩ := same
    have i_small : i < 2 ^ s.rowVars := by simpa [lookupData] using i_lt
    have row_eq : S.runRow k = i := cast _ _ (by have := fits.row_lt k; omega) (by omega) row
    subst row_eq
    exact Prod.ext ((algebraMap F K).injective re) ((algebraMap F K).injective im)
  ·
    obtain ⟨poles3, sum3⟩ := root_sum (earlyShape s) _ 3 _ (roots 3 (by norm_num)) pair34.1
    obtain ⟨poles4, sum4⟩ := root_sum (earlyShape s) _ 4 _ (roots 4 (by norm_num)) pair34.2.1
    rw [d3, earlyBatch_3] at poles3 sum3
    rw [d4, earlyBatch_4] at poles4 sum4
    simp only [Level.numerator, List.map_nil, List.prod_nil, List.map_cons, List.prod_cons,
      mul_one] at sum3 sum4
    have distinct : ∀ x y : ℕ, x < 2 ^ (s.slotVars + s.matrixVars) →
        y < 2 ^ (s.slotVars + s.matrixVars) → (x : K) = (y : K) → x = y :=
      fun x y hx hy => cast x y (by omega) (by omega)
    have pairs := (scatterData s S w).sound distinct o.challenges.scatter.1 o.challenges.scatter.2 hS
      (fun k lt => sub_ne_zero.mp (poles3 k lt)) (fun x lt => sub_ne_zero.mp (poles4 x lt)) (by
        simp only [scatterData]
        rw [← sum3]
        have : ∑ x ∈ Finset.range (2 ^ (s.slotVars + s.matrixVars)),
            ((if x / 2 ^ s.slotVars < s.matrices then
              (ι (w.U (x / 2 ^ s.slotVars) (x % 2 ^ s.slotVars)).1,
                ι (w.U (x / 2 ^ s.slotVars) (x % 2 ^ s.slotVars)).2) else (0, 0)).1 +
              o.challenges.scatter.2 * (if x / 2 ^ s.slotVars < s.matrices then
              (ι (w.U (x / 2 ^ s.slotVars) (x % 2 ^ s.slotVars)).1,
                ι (w.U (x / 2 ^ s.slotVars) (x % 2 ^ s.slotVars)).2) else (0, 0)).2) /
              (o.challenges.scatter.1 - (x : K)) =
            (o.gkr.roots 4).1 / (o.gkr.roots 4).2 := by
          rw [sum4]
          apply Finset.sum_congr rfl
          intro x _
          split <;> simp
        rw [this, pair_fraction pair34])
    intro j σ hj hσ
    have x_lt : σ + 2 ^ s.slotVars * j < 2 ^ (s.slotVars + s.matrixVars) := by
      rw [pow_add]
      have : j < 2 ^ s.matrixVars := by omega
      nlinarith
    have pair := pairs (σ + 2 ^ s.slotVars * j) x_lt
    have divide : (σ + 2 ^ s.slotVars * j) / 2 ^ s.slotVars = j := by
      rw [Nat.add_mul_div_left _ _ (by positivity), Nat.div_eq_of_lt hσ, zero_add]
    have remainder : (σ + 2 ^ s.slotVars * j) % 2 ^ s.slotVars = σ := by
      rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hσ]
    simp only [scatterData, divide, remainder, ite_eq_left hj, Scatter.sums] at pair
    have filters : (Finset.range (2 ^ s.runVars)).filter
        (fun k => slotColumn S k + (2 ^ s.slotVars : K) * matrixColumn S k =
          ((σ + 2 ^ s.slotVars * j : ℕ) : K)) =
        (Finset.range (2 ^ s.runVars)).filter (fun k => S.runMatrix k = j ∧ S.runSlot k = σ) := by
      apply Finset.filter_congr
      intro k _
      have slot_lt := fits.slot_lt k
      have matrix_lt := fits.matrix_lt k
      have tag_lt : S.runSlot k + 2 ^ s.slotVars * S.runMatrix k < 2 ^ (s.slotVars + s.matrixVars) := by
        rw [pow_add]; nlinarith
      constructor
      · intro equal
        have cast_eq : ((S.runSlot k + 2 ^ s.slotVars * S.runMatrix k : ℕ) : K) =
            ((σ + 2 ^ s.slotVars * j : ℕ) : K) := by
          rw [← equal]; simp [slotColumn, matrixColumn]
        have nat_eq := distinct _ _ tag_lt x_lt cast_eq
        have hm : S.runMatrix k = j := by
          have := congrArg (· / 2 ^ s.slotVars) nat_eq
          simp only [Nat.add_mul_div_left _ _ (show 0 < 2 ^ s.slotVars by positivity),
            Nat.div_eq_of_lt slot_lt, Nat.div_eq_of_lt hσ, zero_add] at this
          exact this
        refine ⟨hm, ?_⟩
        rw [hm] at nat_eq
        omega
      · rintro ⟨hm, hs⟩
        simp [slotColumn, matrixColumn, hm, hs]
    rw [filters] at pair
    obtain ⟨re, im⟩ := Prod.ext_iff.mp pair
    exact ⟨re, im⟩
  ·
    obtain ⟨_, _, _, embedded⟩ := setup
    obtain ⟨_, _, _, _, blockTrue, _⟩ := consistent
    unfold EarlyOut.embeddedBlock EarlyOut.runPoint at embedded
    unfold EarlyOut.runPoint at blockTrue path
    set rp := o.gkr.leafPoint 3
    have split : rp = rp.take s.slotVars ++ rp.drop s.slotVars := (List.take_append_drop _ _).symm
    have take_length : (rp.take s.slotVars).length = s.slotVars := by
      rw [List.length_take]; omega
    have low_zero : ∀ σ, 2 ^ (rp.take s.slotVars).length ≤ σ → (blockColumn S σ : K) = 0 := by
      intro σ big
      rw [take_length] at big
      simp only [blockColumn]
      rw [ite_eq_right (by have := fits.slots_le; omega)]
    rw [split, mle_low_support _ _ _ low_zero] at embedded
    rw [← split] at embedded
    have path_split := (Path.append_iff (blockTest s S w) [] (rp.take s.slotVars) (rp.drop s.slotVars)).mp
      (by rw [← split]; exact path)
    have high_ne : ∀ x ∈ rp.drop s.slotVars, x ≠ 1 :=
      Path.ne_one s.slotVars (fun q long => by simp [blockTest, show ¬ q.length < s.slotVars by omega])
        _ _ (by simp [take_length]) path_split.2
    have product_ne : ((rp.drop s.slotVars).map fun x => 1 - x).prod ≠ 0 := by
      apply List.prod_ne_zero
      simp only [List.mem_map, not_exists, not_and]
      intro x member zero
      exact high_ne x member (by rw [sub_eq_zero] at zero; exact zero.symm)
    have block_value := mul_right_cancel₀ product_ne embedded
    by_contra wrong
    push Not at wrong
    obtain ⟨σ, hσ, unequal⟩ := wrong
    have low_path : Path (foldBad (fun σ => ι (w.blockCopy σ) - blockColumn S σ) s.slotVars) []
        (rp.take s.slotVars) :=
      Path.mono_short s.slotVars (fun q short => by simp [blockTest, short]) _ _
        (by simp [take_length]) path_split.1
    have nonzero := mle_ne_zero_of_path' (fun σ => ι (w.blockCopy σ) - blockColumn S σ)
      (rp.take s.slotVars) ⟨σ, by rw [take_length]; exact hσ, sub_ne_zero.mpr unequal⟩
      (by rw [take_length]; exact low_path)
    apply nonzero
    rw [mle_sub, blockTrue, block_value, sub_self]

end CompressionLean.Layer1
