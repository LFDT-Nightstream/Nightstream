import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Nat.ModEq
import Mathlib.Tactic.Ring
import NightstreamFPrime.Circuit.Basic
import NightstreamFPrime.Spec.GoldilocksPrime

/-!
Owns linear forms with natural coefficients over expressions that evaluate to
bits. Evaluation is exact modulo `p`, reduction of coefficients preserves the
value modulo any `m`, and a bound on coefficients bounds the value. Proof cost
does not depend on the number of terms.
-/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

/-- Canonical field element of a natural number. -/
def fieldOfNat (value : Nat) : F :=
  ⟨value % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

theorem fieldOfNat_add (left right : Nat) :
    fieldOfNat left + fieldOfNat right = fieldOfNat (left + right) := by
  apply Fin.ext
  simp [fieldOfNat, Fin.val_add, Nat.add_mod]

theorem fieldOfNat_mul (left right : Nat) :
    fieldOfNat left * fieldOfNat right = fieldOfNat (left * right) := by
  apply Fin.ext
  simp [fieldOfNat, Fin.val_mul, Nat.mul_mod]

theorem fieldOfNat_val (value : F) : fieldOfNat value.val = value := by
  apply Fin.ext
  simp [fieldOfNat, Nat.mod_eq_of_lt value.isLt]

theorem fieldOfNat_inj {left right : Nat} (leftBound : left < goldilocksModulus)
    (rightBound : right < goldilocksModulus) (same : fieldOfNat left = fieldOfNat right) :
    left = right := by
  have values := congrArg Fin.val same
  simpa [fieldOfNat, Nat.mod_eq_of_lt leftBound, Nat.mod_eq_of_lt rightBound] using values

/-- `Σ a · x` with natural coefficients. -/
def linearExpr : List (Nat × Expr) → Expr
  | [] => Expr.const (fieldOfNat 0)
  | term :: rest => Expr.const (fieldOfNat term.1) * term.2 + linearExpr rest

/-- Natural value `Σ a · val(x)` of a linear form. -/
def linearValue (env : Env) : List (Nat × Expr) → Nat
  | [] => 0
  | term :: rest => term.1 * (term.2.eval env).val + linearValue env rest

theorem linearExpr_eval (env : Env) (terms : List (Nat × Expr)) :
    (linearExpr terms).eval env = fieldOfNat (linearValue env terms) := by
  induction terms with
  | nil => rfl
  | cons term rest inductionHypothesis =>
      change Expr.eval env (Expr.const (fieldOfNat term.1)) * term.2.eval env +
          (linearExpr rest).eval env = _
      rw [inductionHypothesis, Expr.eval, ← fieldOfNat_val (term.2.eval env), fieldOfNat_mul,
        fieldOfNat_add]
      rfl

theorem linearValue_append (env : Env) (left right : List (Nat × Expr)) :
    linearValue env (left ++ right) = linearValue env left + linearValue env right := by
  induction left with
  | nil => simp [linearValue]
  | cons term rest inductionHypothesis =>
      simp only [List.cons_append, linearValue, inductionHypothesis]
      ring

theorem linearValue_rangeMap (env : Env) (count : Nat) (coefficient : Nat → Nat)
    (atom : Nat → Expr) :
    linearValue env ((List.range count).map fun index => (coefficient index, atom index)) =
      ∑ index ∈ Finset.range count, coefficient index * ((atom index).eval env).val := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.map_append, linearValue_append, inductionHypothesis,
        Finset.sum_range_succ]
      simp [linearValue]

/-- Coefficients reduced modulo `m`. -/
def reduceTerms (modulus : Nat) (terms : List (Nat × Expr)) : List (Nat × Expr) :=
  terms.map fun term => (term.1 % modulus, term.2)

theorem linearValue_reduce_modEq (env : Env) (modulus : Nat) (terms : List (Nat × Expr)) :
    linearValue env (reduceTerms modulus terms) ≡ linearValue env terms [MOD modulus] := by
  induction terms with
  | nil => rfl
  | cons term rest inductionHypothesis =>
      exact Nat.ModEq.add ((Nat.mod_modEq _ _).mul_right _) inductionHypothesis

/-- Bits and bounded coefficients bound the value by `length * bound`. -/
theorem linearValue_le (env : Env) (terms : List (Nat × Expr)) (bound : Nat)
    (coefficients : ∀ term ∈ terms, term.1 ≤ bound)
    (bits : ∀ term ∈ terms, (term.2.eval env).val ≤ 1) :
    linearValue env terms ≤ terms.length * bound := by
  induction terms with
  | nil => simp [linearValue]
  | cons term rest inductionHypothesis =>
      have head : term.1 * (term.2.eval env).val ≤ bound :=
        calc term.1 * (term.2.eval env).val ≤ bound * 1 :=
              Nat.mul_le_mul (coefficients term (by simp)) (bits term (by simp))
          _ = bound := Nat.mul_one _
      have tail := inductionHypothesis (fun t member => coefficients t (by simp [member]))
        (fun t member => bits t (by simp [member]))
      simp only [linearValue, List.length_cons]
      calc term.1 * (term.2.eval env).val + linearValue env rest
          ≤ bound + rest.length * bound := Nat.add_le_add head tail
        _ = (rest.length + 1) * bound := by ring

theorem reduceTerms_length (modulus : Nat) (terms : List (Nat × Expr)) :
    (reduceTerms modulus terms).length = terms.length := by
  simp [reduceTerms]

theorem reduceTerms_bound (modulus : Nat) (positive : 0 < modulus) (terms : List (Nat × Expr)) :
    ∀ term ∈ reduceTerms modulus terms, term.1 ≤ modulus - 1 := by
  intro term member
  simp only [reduceTerms, List.mem_map] at member
  obtain ⟨original, _, rfl⟩ := member
  exact Nat.le_sub_one_of_lt (Nat.mod_lt _ positive)

theorem reduceTerms_bits (env : Env) (modulus : Nat) (terms : List (Nat × Expr))
    (bits : ∀ term ∈ terms, (term.2.eval env).val ≤ 1) :
    ∀ term ∈ reduceTerms modulus terms, (term.2.eval env).val ≤ 1 := by
  intro term member
  simp only [reduceTerms, List.mem_map] at member
  obtain ⟨original, originalMember, rfl⟩ := member
  exact bits original originalMember

/-- A Booleanity row `x (x - 1) = 0` makes `x` a bit. -/
theorem bit_of_boolean (env : Env) (atom : Expr)
    (row : (atom * (atom - 1)).eval env = 0) : (atom.eval env).val ≤ 1 := by
  change atom.eval env * (atom - 1).eval env = 0 at row
  rw [Expr.eval_sub] at row
  rcases NightstreamFPrime.Spec.GoldilocksPrime.baseFieldNoZeroDivisors _ _ row with zero | one
  · rw [zero]; exact Nat.zero_le _
  · have : atom.eval env = 1 := by
      have := sub_eq_zero.mp one
      simpa [Expr.eval] using this
    rw [this]
    decide

end NightstreamFPrime.Gadgets.Sampling.WideReduction
