import CompressionLean.Layer1.EarlySound

namespace CompressionLean.Layer1

open CompressionLean.Lib
open Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

noncomputable def laneWeight (s : Shape) (bar : ℕ → ℕ → F) (zeta : K) (l : ℕ) : K :=
  ∑ p ∈ Finset.range s.D, ι (bar p l) * zeta ^ p

noncomputable def ubar (s : Shape) (lam : K) (U : ℕ → ℕ → F × F) (σ : ℕ) : K :=
  ∑ j ∈ Finset.range s.matrices,
    (lam ^ s.evalARow j 0 * ι (U j σ).1 + lam ^ s.evalARow j 1 * ι (U j σ).2)

def slotStart (s : Shape) (S : Setup F) (σ : ℕ) : ℕ := s.D * S.slotBlock σ + S.slotLane σ

open Classical in
noncomputable def runWeight (s : Shape) (S : Setup F) (σ c : ℕ) : K :=
  if slotStart s S σ ≤ c ∧ c < slotStart s S σ + S.slotLength σ then
    ι (S.slotRatio σ) ^ (c - slotStart s S σ) else 0

noncomputable def laneSplit (s : Shape) (S : Setup F) (τ : ℕ → K) (part σ : ℕ) : K :=
  if σ < S.slots then
    ∑ l ∈ Finset.range s.D, τ l * runWeight s S σ (s.D * (S.slotBlock σ + part) + l)
  else 0

def lateShape (s : Shape) : GkrShape where
  trees := 2
  depth t := if t = 0 then s.slotVars + 1 else s.blockVars
  factors t := if t = 0 then 2 else 1

noncomputable def lateBatch (s : Shape) (S : Setup F) (lam : K) (τ : ℕ → K) (beta : K)
    (w : P0Tables F) (v : P1Tables K) : Batch K := fun t =>
  if t = 0 then
    ⟨[fun x => ubar s lam w.U (x % 2 ^ s.slotVars),
        fun x => laneSplit s S τ (x / 2 ^ s.slotVars) (x % 2 ^ s.slotVars)],
      fun x => beta - (ι (w.blockCopy (x % 2 ^ s.slotVars)) + ((x / 2 ^ s.slotVars : ℕ) : K))⟩
  else ⟨[v.omegaA], fun b => beta - (b : K)⟩

structure LateOut (K : Type) where
  beta : K
  gkr : GkrClaims K
  pairs : ℕ → K × K

namespace LateOut

variable (s : Shape) (o : LateOut K)

def slotPoint : List K := (o.gkr.leafPoint 0).take s.slotVars
def top : K := (o.gkr.leafPoint 0).getD s.slotVars 0
def blockPoint : List K := o.gkr.leafPoint 1
def block : K := o.beta - o.gkr.leafValues 0 2 - o.top s
def omega : K := o.gkr.leafValues 1 0

end LateOut

def LateChecks (s : Shape) (S : Setup F) (lam : K) (τ : ℕ → K) (o : LateOut K) : Prop :=
  PairCheck (o.gkr.roots 0) (o.gkr.roots 1) ∧
  o.gkr.leafValues 0 0 = ∑ j ∈ Finset.range s.matrices,
      (lam ^ s.evalARow j 0 * (o.pairs j).1 + lam ^ s.evalARow j 1 * (o.pairs j).2) ∧
  o.gkr.leafValues 0 1 = (1 - o.top s) * mle (o.slotPoint s) (laneSplit s S τ 0) +
      o.top s * mle (o.slotPoint s) (laneSplit s S τ 1) ∧
  o.gkr.leafValues 1 1 = o.beta - mle o.blockPoint fun b => (b : K)

def LateConsistent (s : Shape) (o : LateOut K) (w : P0Tables F) (v : P1Tables K) : Prop :=
  (∀ j, j < s.matrices →
    mle (o.slotPoint s) (fun σ => ι (w.U j σ).1) = (o.pairs j).1 ∧
      mle (o.slotPoint s) (fun σ => ι (w.U j σ).2) = (o.pairs j).2) ∧
    mle (o.slotPoint s) (fun σ => ι (w.blockCopy σ)) = o.block s ∧
    mle o.blockPoint v.omegaA = o.omega

open Classical in
noncomputable def slotWeights (s : Shape) (S : Setup F) (lam : K) (τ : ℕ → K) (U : ℕ → ℕ → F × F)
    (b : ℕ) : K :=
  ∑ x ∈ (Finset.range (2 ^ (s.slotVars + 1))).filter
      (fun x => blockColumn S (x % 2 ^ s.slotVars) + ((x / 2 ^ s.slotVars : ℕ) : K) = (b : K)),
    ubar s lam U (x % 2 ^ s.slotVars) * laneSplit s S τ (x / 2 ^ s.slotVars) (x % 2 ^ s.slotVars)

noncomputable def blockPoly [DecidableEq K] (s : Shape) (S : Setup F) (lam : K) (τ : ℕ → K)
    (w : P0Tables F) (v : P1Tables K) : K[X] :=
  twoSidedPoly (Finset.range (2 ^ (s.slotVars + 1)))
    (fun x => ubar s lam w.U (x % 2 ^ s.slotVars) * laneSplit s S τ (x / 2 ^ s.slotVars) (x % 2 ^ s.slotVars))
    (fun x => ι (w.blockCopy (x % 2 ^ s.slotVars)) + ((x / 2 ^ s.slotVars : ℕ) : K))
    (Finset.range (2 ^ s.blockVars)) v.omegaA (fun b => (b : K))

theorem late_factors (s : Shape) : (lateShape s).factors 0 = 2 ∧ (lateShape s).factors 1 = 1 := by
  simp [lateShape]

theorem late_depths (s : Shape) :
    (lateShape s).depth 0 = s.slotVars + 1 ∧ (lateShape s).depth 1 = s.blockVars := by
  simp [lateShape]

theorem mle_top_split (sp : List K) (τ : K) (f : ℕ → ℕ → K) :
    mle (sp ++ [τ]) (fun x => f (x / 2 ^ sp.length) (x % 2 ^ sp.length)) =
      (1 - τ) * mle sp (f 0) + τ * mle sp (f 1) := by
  rw [mle_append_top]
  congr 2
  · apply mle_congr
    intro y lt
    rw [Nat.div_eq_of_lt lt, Nat.mod_eq_of_lt lt]
  · apply mle_congr
    intro y lt
    rw [show (y + 2 ^ sp.length) / 2 ^ sp.length = 1 by
        rw [Nat.add_div_right _ (by positivity), Nat.div_eq_of_lt lt],
      Nat.add_mod_right, Nat.mod_eq_of_lt lt]

theorem late_world [Fintype K] [DecidableEq K] (s : Shape) (S : Setup F) (st : Statement F)
    (lam : K) (τ : ℕ → K) (o : LateOut K) (w : P0Tables F) (v : P1Tables K)
    (N : ℕ) (cast : ∀ a b : ℕ, a < N → b < N → (a : K) = b → a = b) (big_blocks : 2 ^ s.blockVars ≤ N)
    (checks : LateChecks s S lam τ o) (consistent : LateConsistent s o w v)
    (early : EarlyFacts (K := K) s S st w)
    (hβ : o.beta ∉ zeros (blockPoly s S lam τ w v))
    (gkrRoot : (∃ t, t < 2 ∧ o.gkr.roots t ≠ trueRoot (lateShape s) (lateBatch s S lam τ o.beta w v) t) →
      LeafFalse (lateShape s) (lateBatch s S lam τ o.beta w v) o.gkr)
    (pairLength : (o.gkr.leafPoint 0).length = s.slotVars + 1) :
    ∀ b, b < 2 ^ s.blockVars → v.omegaA b = slotWeights s S lam τ w.U b := by
  obtain ⟨pair, ubarCheck, splitCheck, indexCheck⟩ := checks
  obtain ⟨pairsTrue, blockTrue, omegaTrue⟩ := consistent
  obtain ⟨f0, f1⟩ := late_factors s
  obtain ⟨d0, d1⟩ := late_depths s
  have point : o.gkr.leafPoint 0 = o.slotPoint s ++ [o.top s] := by
    unfold LateOut.slotPoint LateOut.top
    conv_lhs => rw [← List.take_append_drop s.slotVars (o.gkr.leafPoint 0)]
    congr 1
    rw [List.drop_eq_getElem_cons (by omega)]
    simp [List.getD_eq_getElem?_getD, List.drop_eq_nil_of_le, pairLength]
  have spLength : (o.slotPoint s).length = s.slotVars := by
    unfold LateOut.slotPoint; rw [List.length_take]; omega
  have leaves : ¬ LeafFalse (lateShape s) (lateBatch s S lam τ o.beta w v) o.gkr := by
    rintro ⟨t, tree, j, le, false_leaf⟩
    apply false_leaf
    unfold trueLeaf
    rw [treePart_leaf]
    have cases : t = 0 ∨ t = 1 := by simp only [lateShape] at tree; omega
    rcases cases with rfl | rfl
    · rw [f0] at le ⊢
      simp only [lateBatch, ite_true]
      rw [point]
      rcases (show j = 0 ∨ j = 1 ∨ j = 2 by omega) with rfl | rfl | rfl
      · rw [ite_eq_left (by norm_num)]
        simp only [List.getD_cons_zero]
        have split := mle_top_split (o.slotPoint s) (o.top s) (fun _ σ => ubar s lam w.U σ)
        rw [spLength] at split
        rw [split, ← add_mul, sub_add_cancel, one_mul, ubarCheck]
        unfold ubar
        rw [mle_finset_sum]
        apply Finset.sum_congr rfl
        intro j inside
        rw [mle_add, mle_mul_left, mle_mul_left, (pairsTrue j (Finset.mem_range.mp inside)).1,
          (pairsTrue j (Finset.mem_range.mp inside)).2]
      · rw [ite_eq_left (by norm_num)]
        simp only [List.getD_cons_succ, List.getD_cons_zero]
        have split := mle_top_split (o.slotPoint s) (o.top s) (laneSplit s S τ)
        rw [spLength] at split
        rw [split, splitCheck]
      · rw [ite_eq_right (lt_irrefl 2)]
        have split := mle_top_split (o.slotPoint s) (o.top s)
          (fun a σ => ι (w.blockCopy σ) + ((a : ℕ) : K))
        rw [spLength] at split
        rw [mle_sub, mle_const, split, mle_add, mle_add, mle_const, mle_const, blockTrue]
        unfold LateOut.block
        push_cast
        ring
    · rw [f1] at le ⊢
      simp only [lateBatch, show ¬ ((1 : ℕ) = 0) by omega, ite_false]
      rcases (show j = 0 ∨ j = 1 by omega) with rfl | rfl
      · rw [ite_eq_left (by norm_num)]
        simp only [List.getD_cons_zero]
        unfold LateOut.blockPoint LateOut.omega at omegaTrue
        exact omegaTrue.symm
      · rw [ite_eq_right (lt_irrefl 1)]
        rw [mle_sub, mle_const, indexCheck]
        rfl
  have roots : ∀ t, t < 2 → o.gkr.roots t = trueRoot (lateShape s) (lateBatch s S lam τ o.beta w v) t := by
    by_contra wrong
    push Not at wrong
    exact leaves (gkrRoot wrong)
  obtain ⟨poles0, sum0⟩ := root_sum (lateShape s) _ 0 _ (roots 0 (by norm_num)) pair.1
  obtain ⟨poles1, sum1⟩ := root_sum (lateShape s) _ 1 _ (roots 1 (by norm_num)) pair.2.1
  rw [d0] at poles0 sum0
  rw [d1] at poles1 sum1
  simp only [lateBatch, ite_true, show ¬ ((1 : ℕ) = 0) by omega, ite_false, Level.numerator,
    List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one] at poles0 sum0 poles1 sum1
  have equal := pair_fraction pair
  rw [sum0, sum1] at equal
  intro b hb
  by_contra wrong
  apply hβ
  apply twoSided _ _ _ _ _ _ (v := (b : K)) _ o.beta
    (fun x m => sub_ne_zero.mp (poles0 x (Finset.mem_range.mp m)))
    (fun c m => sub_ne_zero.mp (poles1 c (Finset.mem_range.mp m))) equal
  rw [residue_range_single _ _ (fun x y hx hy => cast x y (by omega) (by omega)) hb]
  intro same
  apply wrong
  rw [← same]
  unfold residue slotWeights
  rw [Finset.sum_filter, Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro x _
  have σ_lt : x % 2 ^ s.slotVars < 2 ^ s.slotVars := Nat.mod_lt _ (by positivity)
  beta_reduce
  rw [early.2.2.2 _ σ_lt]
  by_cases h : blockColumn S (x % 2 ^ s.slotVars) + ((x / 2 ^ s.slotVars : ℕ) : K) = (b : K) <;>
    simp [h]

end CompressionLean.Layer1
