import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Tactic

/-!
Rational enclosures for natural logarithms. The executable endpoints use the
atanh series; the theorems connect them to `Real.log`.
-/

open scoped BigOperators

namespace Ajtai.LogInterval

def atanhSum (x : ℚ) (terms : Nat) : ℚ :=
  ∑ i ∈ Finset.range terms, x ^ (2 * i + 1) / (2 * i + 1)

def logRatioLower (x : ℚ) (terms : Nat) : ℚ :=
  2 * atanhSum x terms

def logRatioUpper (x : ℚ) (terms : Nat) : ℚ :=
  2 * (atanhSum x terms + x ^ (2 * terms + 1) / (1 - x ^ 2))

theorem log_ratio_bounds
    (x : ℚ) (terms : Nat) (nonneg : 0 ≤ x) (ltOne : x < 1) :
    ((logRatioLower x terms : ℚ) : ℝ) ≤
        Real.log ((1 + (x : ℝ)) / (1 - (x : ℝ))) ∧
      Real.log ((1 + (x : ℝ)) / (1 - (x : ℝ))) ≤
        ((logRatioUpper x terms : ℚ) : ℝ) := by
  have nonnegReal : 0 ≤ (x : ℝ) := by exact_mod_cast nonneg
  have ltOneReal : (x : ℝ) < 1 := by exact_mod_cast ltOne
  have lower := Real.sum_range_le_log_div nonnegReal ltOneReal terms
  have upper := Real.log_div_le_sum_range_add nonnegReal ltOneReal terms
  constructor
  · unfold logRatioLower atanhSum
    push_cast
    linarith
  · unfold logRatioUpper atanhSum
    push_cast
    linarith

def rangeX (n scale : Nat) : ℚ :=
  ((n : ℚ) - (scale : ℚ)) / ((n : ℚ) + (scale : ℚ))

def logNatLower (n exponent terms : Nat) : ℚ :=
  exponent * logRatioLower (1 / 3) terms +
    logRatioLower (rangeX n (2 ^ exponent)) terms

def logNatUpper (n exponent terms : Nat) : ℚ :=
  exponent * logRatioUpper (1 / 3) terms +
    logRatioUpper (rangeX n (2 ^ exponent)) terms

theorem log_nat_bounds
    (n exponent terms : Nat)
    (lower : 2 ^ exponent ≤ n) :
    ((logNatLower n exponent terms : ℚ) : ℝ) ≤ Real.log (n : ℝ) ∧
      Real.log (n : ℝ) ≤ ((logNatUpper n exponent terms : ℚ) : ℝ) := by
  have scalePos : 0 < 2 ^ exponent := by positivity
  have nPos : 0 < n := Nat.lt_of_lt_of_le scalePos lower
  have xNonneg : 0 ≤ rangeX n (2 ^ exponent) := by
    unfold rangeX
    apply div_nonneg
    · exact sub_nonneg.mpr (by exact_mod_cast lower)
    · positivity
  have xLtOne : rangeX n (2 ^ exponent) < 1 := by
    unfold rangeX
    rw [div_lt_one (by positivity)]
    have scaleRatPos : (0 : ℚ) < (2 ^ exponent : Nat) := by
      exact_mod_cast scalePos
    linarith
  have twoBounds :=
    log_ratio_bounds (1 / 3) terms (by norm_num) (by norm_num)
  norm_num at twoBounds
  have ratioBounds :=
    log_ratio_bounds (rangeX n (2 ^ exponent)) terms xNonneg xLtOne
  have ratioIdentity :
      (1 + ((rangeX n (2 ^ exponent) : ℚ) : ℝ)) /
          (1 - ((rangeX n (2 ^ exponent) : ℚ) : ℝ)) =
        (n : ℝ) / (2 ^ exponent : Nat) := by
    unfold rangeX
    push_cast
    field_simp
    ring
  have logScale :
      Real.log ((2 ^ exponent : Nat) : ℝ) =
        (exponent : ℝ) * Real.log 2 := by
    norm_num [Nat.cast_pow, Real.log_pow]
  have logDecompose :
      Real.log (n : ℝ) =
        (exponent : ℝ) * Real.log 2 +
          Real.log
            ((1 + ((rangeX n (2 ^ exponent) : ℚ) : ℝ)) /
              (1 - ((rangeX n (2 ^ exponent) : ℚ) : ℝ))) := by
    rw [ratioIdentity, ← logScale, ← Real.log_mul]
    · congr 1
      field_simp
    · positivity
    · positivity
  constructor
  · unfold logNatLower
    push_cast
    rw [logDecompose]
    gcongr
    · exact twoBounds.1
    · exact ratioBounds.1
  · unfold logNatUpper
    push_cast
    rw [logDecompose]
    gcongr
    · exact twoBounds.2
    · exact ratioBounds.2

def precisionTerms : Nat := 24

def lnNatLower (n : Nat) : ℚ :=
  logNatLower n n.log2 precisionTerms

def lnNatUpper (n : Nat) : ℚ :=
  logNatUpper n n.log2 precisionTerms

theorem ln_nat_bounds {n : Nat} (nonzero : n ≠ 0) :
    ((lnNatLower n : ℚ) : ℝ) ≤ Real.log (n : ℝ) ∧
      Real.log (n : ℝ) ≤ ((lnNatUpper n : ℚ) : ℝ) := by
  exact log_nat_bounds n n.log2 precisionTerms (Nat.log2_self_le nonzero)

def lnRatLower (numerator denominator : Nat) : ℚ :=
  lnNatLower numerator - lnNatUpper denominator

def lnRatUpper (numerator denominator : Nat) : ℚ :=
  lnNatUpper numerator - lnNatLower denominator

theorem ln_rat_bounds
    {numerator denominator : Nat}
    (numeratorNonzero : numerator ≠ 0)
    (denominatorNonzero : denominator ≠ 0) :
    ((lnRatLower numerator denominator : ℚ) : ℝ) ≤
        Real.log ((numerator : ℝ) / (denominator : ℝ)) ∧
      Real.log ((numerator : ℝ) / (denominator : ℝ)) ≤
        ((lnRatUpper numerator denominator : ℚ) : ℝ) := by
  have numeratorBounds := ln_nat_bounds numeratorNonzero
  have denominatorBounds := ln_nat_bounds denominatorNonzero
  rw [Real.log_div (by positivity) (by positivity)]
  constructor
  · unfold lnRatLower
    push_cast
    linarith
  · unfold lnRatUpper
    push_cast
    linarith

end Ajtai.LogInterval
