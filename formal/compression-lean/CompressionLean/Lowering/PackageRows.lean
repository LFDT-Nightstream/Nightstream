import CompressionLean.Ports.Terminal.Package
import CompressionLean.Lowering.Exec
import NightstreamFPrime.Export.PackageExpansion

namespace CompressionLean.Replay.Word

open NightstreamFPrime.Spec NightstreamFPrime.Export.NativePoseidon2

open Fin.CommRing in
theorem cast_ofF (x : F) : (ofF x).cast = x := by
  have denote := ofNat64_denote x.val
  have canonical := ofNat64_canonical x.val
  have value : (ofNat64 x.val).toNat = x.val := by
    have := congrArg Fin.val denote
    simpa [UInt64.denote, Poseidon2.ofNat, Nat.mod_eq_of_lt canonical, Nat.mod_eq_of_lt x.isLt] using this
  show ((ofNat64 x.val).toNat : F) = x
  rw [value]
  exact Fin.cast_val_eq_self x

end CompressionLean.Replay.Word

namespace CompressionLean.Lowering.Package

open CompressionLean.Terminal.Package
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open CompressionLean.Shrink (R1csRow)
open CompressionLean.Replay (Word)
open Fin.CommRing

section Sides

variable (env : ℕ → NightstreamFPrime.Spec.F)

def termsValue (terms : List (ℕ × NightstreamFPrime.Spec.F)) : NightstreamFPrime.Spec.F :=
  (terms.map fun t => t.2 * env t.1).sum

theorem termsValue_filter (terms : List (ℕ × NightstreamFPrime.Spec.F)) :
    termsValue env (terms.filter (·.2 ≠ 0)) = termsValue env terms := by
  induction terms with
  | nil => rfl
  | cons t rest ih =>
      by_cases zero : t.2 = 0
      · simp [zero, termsValue] at ih ⊢
        exact ih
      · simp [zero, termsValue] at ih ⊢
        rw [ih]

theorem termsValue_collect (terms : List (ℕ × NightstreamFPrime.Spec.F)) :
    termsValue env (collect terms) = termsValue env terms := by
  induction terms using collect.induct with
  | case1 x d y rest ih =>
      rw [collect.eq_1, ite_eq_left rfl, ih]
      simp only [termsValue, List.map_cons, List.sum_cons]
      ring
  | case2 c x d y rest unequal ih =>
      rw [collect.eq_1, ite_eq_right unequal]
      simp only [termsValue, List.map_cons, List.sum_cons] at ih ⊢
      rw [ih]
  | case3 terms short =>
      rw [collect.eq_2 terms short]

theorem words_value (terms : List (ℕ × NightstreamFPrime.Spec.F)) :
    ((terms.filterMap fun (c, x) => if x = 0 then none else some (c, Word.ofF x)).map
      fun t => env t.1 * Word.cast t.2).sum = termsValue env terms := by
  induction terms with
  | nil => rfl
  | cons t rest ih =>
      obtain ⟨c, x⟩ := t
      by_cases zero : x = 0
      · simp only [List.filterMap_cons, zero, ite_true] at ih ⊢
        rw [ih]
        simp [termsValue]
      · simp only [List.filterMap_cons, zero, ite_false, List.map_cons, List.sum_cons] at ih ⊢
        rw [ih, Word.cast_ofF]
        simp [termsValue, mul_comm]

theorem side_value (constant : ℕ) (one : env constant = 1) (combination : R1CS.LinearCombination) :
    ((side constant combination).map fun t => env t.1 * Word.cast t.2).sum = combination.eval env := by
  unfold side
  rw [words_value, termsValue_collect, termsValue, ((List.mergeSort_perm _ _).map _).sum_eq,
    ← termsValue, termsValue_filter]
  simp [termsValue, R1CS.LinearCombination.eval, one]

theorem row_holds (constant : ℕ) (one : env constant = 1) (r : R1CS.Row) :
    R1csRow.Holds env (row constant r) ↔ r.Holds env := by
  simp only [R1csRow.Holds, row, R1CS.Row.Holds, side_value env constant one]

end Sides

end CompressionLean.Lowering.Package
