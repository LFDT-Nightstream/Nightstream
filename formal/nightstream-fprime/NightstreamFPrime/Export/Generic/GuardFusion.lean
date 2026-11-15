import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Fintype.BigOperators
import NightstreamFPrime.Export.Generic.NativePower
import NightstreamFPrime.Export.Generic.BooleanInference

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.GuardFusion

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open Fin.CommRing
open scoped BigOperators

theorem complementary_iff (gate left right : F) (bit : gate = 0 ∨ gate = 1) :
    gate * (left - right) = -right ↔ gate * left = 0 ∧ (1 - gate) * right = 0 := by
  rcases bit with rfl | rfl <;> simp

theorem sum_bit_exclusive {n : Nat} (gates : Fin n → F)
    (small : n < fieldModulus) (bits : ∀ i, gates i = 0 ∨ gates i = 1)
    (total : (∑ i, gates i) = 0 ∨ (∑ i, gates i) = 1) :
    ∀ i j, i ≠ j → gates i * gates j = 0 := by
  classical
  let active := Finset.univ.filter (fun i => gates i = 1)
  have count : active.card ≤ n := by
    simpa using Finset.card_le_card (Finset.filter_subset (fun i => gates i = 1) Finset.univ)
  have sum : (∑ i, gates i) = (active.card : F) := by
    calc
      (∑ i, gates i) = ∑ i, if gates i = 1 then (1 : F) else 0 := by
        apply Finset.sum_congr rfl
        intro i _
        rcases bits i with bit | bit <;> simp [bit]
      _ = (active.card : F) := by simp [active]
  have countSmall : active.card < fieldModulus := lt_of_le_of_lt count small
  have atMostOne : active.card ≤ 1 := by
    rw [sum] at total
    rcases total with zero | one
    · have values := congrArg Fin.val zero
      have equal : active.card = 0 := by
        simpa only [Fin.val_natCast, Nat.mod_eq_of_lt countSmall, Fin.val_zero] using values
      omega
    · have values := congrArg Fin.val one
      have oneVal : (1 : F).val = 1 := by decide
      have equal : active.card = 1 := by
        simpa only [Fin.val_natCast, Nat.mod_eq_of_lt countSmall, oneVal] using values
      omega
  intro i j different
  rcases bits i with first | first
  · simp [first]
  rcases bits j with second | second
  · simp [second]
  have same := Finset.card_le_one.mp atMostOne i (by simp [active, first]) j (by simp [active, second])
  exact False.elim (different same)

theorem subset_bit {n : Nat} (gates : Fin n → F) (subset : Finset (Fin n))
    (bits : ∀ i, gates i = 0 ∨ gates i = 1)
    (exclusive : ∀ i j, i ≠ j → gates i * gates j = 0) :
    (∑ i ∈ subset, gates i) = 0 ∨ (∑ i ∈ subset, gates i) = 1 := by
  classical
  by_cases active : ∃ i ∈ subset, gates i ≠ 0
  · obtain ⟨i, member, nonzero⟩ := active
    have others : ∀ j, j ≠ i → gates j = 0 := by
      intro j different
      exact (FieldPrime.baseFieldNoZeroDivisors _ _
        (exclusive i j (Ne.symm different))).resolve_left nonzero
    have sum : (∑ j ∈ subset, gates j) = gates i := by
      apply Finset.sum_eq_single i
      · intro j _ different
        exact others j different
      · exact fun missing => False.elim (missing member)
    rw [sum]
    exact bits i
  · left
    apply Finset.sum_eq_zero
    intro i member
    by_contra nonzero
    exact active ⟨i, member, nonzero⟩

theorem disjoint_sums {n : Nat} (gates : Fin n → F) (left right : Finset (Fin n))
    (exclusive : ∀ i j, i ≠ j → gates i * gates j = 0)
    (disjoint : Disjoint left right) :
    (∑ i ∈ left, gates i) * (∑ j ∈ right, gates j) = 0 := by
  classical
  rw [Finset.sum_mul]
  apply Finset.sum_eq_zero
  intro i member
  rw [Finset.mul_sum]
  apply Finset.sum_eq_zero
  intro j other
  apply exclusive i j
  intro equal
  subst j
  exact Finset.disjoint_left.mp disjoint member other

theorem family_iff {n : Nat} (gates constants : Fin n → F) (common : F)
    (exclusive : ∀ i j, i ≠ j → gates i * gates j = 0) :
    (∑ i, gates i) * common = -(∑ i, constants i * gates i) ↔
      ∀ i, gates i * (common + constants i) = 0 := by
  classical
  constructor
  · intro fused i
    by_cases zero : gates i = 0
    · simp [zero]
    have others : ∀ j, j ≠ i → gates j = 0 := by
      intro j different
      exact (FieldPrime.baseFieldNoZeroDivisors _ _ (exclusive i j (Ne.symm different))).resolve_left zero
    have sumGates : (∑ j, gates j) = gates i := by
      apply Finset.sum_eq_single i
      · intro j _ different
        exact others j different
      · simp
    have sumConstants : (∑ j, constants j * gates j) = constants i * gates i := by
      apply Finset.sum_eq_single i
      · intro j _ different
        simp [others j different]
      · simp
    rw [sumGates, sumConstants] at fused
    calc
      gates i * (common + constants i) = gates i * common + constants i * gates i := by ring
      _ = 0 := by rw [fused, neg_add_cancel]
  · intro source
    have zero : (∑ i, gates i * (common + constants i)) = 0 :=
      Finset.sum_eq_zero (fun i _ => source i)
    have distribute : (∑ i, gates i * (common + constants i)) =
        (∑ i, gates i) * common + ∑ i, constants i * gates i := by
      simp only [mul_add, Finset.sum_add_distrib, Finset.sum_mul]
      congr 1
      apply Finset.sum_congr rfl
      intro i _
      ring
    rw [distribute] at zero
    exact eq_neg_of_add_eq_zero_left zero

def guarded (gate residual : R1CS.LinearCombination) : R1CS.Row :=
  ⟨gate, residual, .zero⟩

def complementary (gate left right : R1CS.LinearCombination) : R1CS.Row :=
  ⟨gate, left.add (right.scale (-1)), right.scale (-1)⟩

theorem complementary_rows (gate left right : R1CS.LinearCombination) (env : Env)
    (bit : gate.eval env = 0 ∨ gate.eval env = 1) :
    (complementary gate left right).Holds env ↔
      R1CS.RowsHold env [guarded gate left, guarded (R1CS.LinearCombination.one.add (gate.scale (-1))) right] := by
  simpa [complementary, guarded, R1CS.Row.Holds, R1CS.RowsHold, sub_eq_add_neg]
    using complementary_iff (gate.eval env) (left.eval env) (right.eval env) bit

def sumForms (forms : List R1CS.LinearCombination) : R1CS.LinearCombination :=
  forms.foldr R1CS.LinearCombination.add .zero

@[simp] theorem sumForms_eval (forms : List R1CS.LinearCombination) (env : Env) :
    (sumForms forms).eval env = (forms.map (fun form => form.eval env)).sum := by
  induction forms with
  | nil => simp [sumForms]
  | cons form rest ih => simpa [sumForms] using congrArg (fun value => form.eval env + value) ih

def family {n : Nat} (gates : Fin n → R1CS.LinearCombination)
    (constants : Fin n → F) (common : R1CS.LinearCombination) : R1CS.Row :=
  ⟨sumForms (List.ofFn gates), common,
    (sumForms (List.ofFn (fun i => (gates i).scale (constants i)))).scale (-1)⟩

def familySources {n : Nat} (gates : Fin n → R1CS.LinearCombination)
    (constants : Fin n → F) (common : R1CS.LinearCombination) : List R1CS.Row :=
  List.ofFn (fun i => guarded (gates i) (common.add (.const (constants i))))

theorem family_rows {n : Nat} (gates : Fin n → R1CS.LinearCombination)
    (constants : Fin n → F) (common : R1CS.LinearCombination) (env : Env)
    (exclusive : ∀ i j, i ≠ j → (gates i).eval env * (gates j).eval env = 0) :
    (family gates constants common).Holds env ↔
      R1CS.RowsHold env (familySources gates constants common) := by
  simpa [family, familySources, guarded, R1CS.Row.Holds, R1CS.RowsHold,
    List.map_ofFn, List.sum_ofFn, List.mem_ofFn]
    using family_iff (fun i => (gates i).eval env) constants (common.eval env) exclusive

structure Rewrite where
  source : List R1CS.Row
  target : List R1CS.Row
  certificates : List R1CS.Row
  equivalent : ∀ env, R1CS.RowsHold env certificates →
    (R1CS.RowsHold env target ↔ R1CS.RowsHold env source)

theorem restore_sources (rewrites : List Rewrite) (retained : List R1CS.Row) (env : Env)
    (covered : ∀ rewrite ∈ rewrites, ∀ row ∈ rewrite.certificates, row ∈ retained) :
    R1CS.RowsHold env (retained ++ rewrites.flatMap Rewrite.target) ↔
      R1CS.RowsHold env (retained ++ rewrites.flatMap Rewrite.source) := by
  rw [R1CS.rowsHold_append, R1CS.rowsHold_append]
  constructor
  · rintro ⟨kept, lowered⟩
    refine ⟨kept, ?_⟩
    intro row member
    obtain ⟨rewrite, member, source⟩ := List.mem_flatMap.mp member
    exact (rewrite.equivalent env (fun r h => kept r (covered rewrite member r h))).mp
      (fun r h => lowered r (List.mem_flatMap.mpr ⟨rewrite, member, h⟩)) row source
  · rintro ⟨kept, original⟩
    refine ⟨kept, ?_⟩
    intro row member
    obtain ⟨rewrite, member, target⟩ := List.mem_flatMap.mp member
    exact (rewrite.equivalent env (fun r h => kept r (covered rewrite member r h))).mpr
      (fun r h => original r (List.mem_flatMap.mpr ⟨rewrite, member, h⟩)) row target

theorem affine_restore_sources (rewrites : List Rewrite) (retained : List R1CS.Row)
    (column : Nat) (form : R1CS.LinearCombination) (env : Env)
    (covered : ∀ rewrite ∈ rewrites, ∀ row ∈ rewrite.certificates, row ∈ retained)
    (acyclic : form.VarsSatisfy (fun index => index ≠ column))
    (holds : R1CS.RowsHold env
      ((retained ++ rewrites.flatMap Rewrite.target).map
        (Substitute.row (NativePower.affineForms column form)))) :
    R1CS.RowsHold (Substitute.refill (NativePower.affineForms column form) env)
      (NativePower.affineRow column form :: (retained ++ rewrites.flatMap Rewrite.source)) := by
  have restored := NativePower.affine_restore_source column form _ env acyclic holds
  have tail : R1CS.RowsHold (Substitute.refill (NativePower.affineForms column form) env)
      (retained ++ rewrites.flatMap Rewrite.target) :=
    fun row member => restored row (List.mem_cons_of_mem _ member)
  have source := (restore_sources rewrites retained _ covered).mp tail
  intro row member
  rcases List.mem_cons.mp member with rfl | member
  · exact restored _ List.mem_cons_self
  · exact source row member

theorem power_restore_sources (chain : NativePower.Chain) (rewrites : List Rewrite)
    (retained : List R1CS.Row) (env : Env) (valid : chain.Valid)
    (covered : ∀ rewrite ∈ rewrites, ∀ row ∈ rewrite.certificates, row ∈ retained)
    (framed : ∀ row ∈ retained ++ rewrites.flatMap Rewrite.target,
      row.VarsSatisfy chain.live)
    (native : chain.Holds env)
    (holds : R1CS.RowsHold env (retained ++ rewrites.flatMap Rewrite.target)) :
    R1CS.RowsHold (chain.restore env)
      (chain.rows ++ (retained ++ rewrites.flatMap Rewrite.source)) := by
  have restored := chain.restore_source _ env valid framed native holds
  have parts := (R1CS.rowsHold_append _ _ _).mp restored
  exact (R1CS.rowsHold_append _ _ _).mpr
    ⟨parts.1, (restore_sources rewrites retained _ covered).mp parts.2⟩

end NightstreamFPrime.Export.Generic.GuardFusion
