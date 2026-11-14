import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightKnowledge
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshKnowledge
import NightstreamFPrime.Spec.Folding.Light.ErrorBudget

/-! The fold gate of Rust `Params::for_ccs_shape` in integers. Rust reads `Counts` from a CCS
shape: the cube variables `ℓ`, the degree bound of a round polynomial and the CCS matrices. Its
census numerator (`Counts.numerator`) counts every fresh source that the Π_RLC guard admits. The
gate charges `error = (2ℓ + 1 + 17) · numerator/q² + 17² · (1/|C| + 2^−173)` as
`count/denominator`, and gives `bits`, the largest `k` with `2^k · count ≤ denominator`.

- `bits_spec`: `2^bits · count ≤ denominator < 2^(bits + 1) · count`.
- `error_le_bits`: when `bits` is positive, `error ≤ 2^−bits`.
- `refresh_statisticalError_le`, `light_statisticalError_le`: at the production counts, the
  statistical term of each fold's random-oracle result with `Q` queries is at most
  `(Q + 1) · error`. `refresh_statisticalError_le_bits`, `light_statisticalError_le_bits`: it is
  at most `(Q + 1)/2^bits`.
- `production_bits`: `bits` is 109 at the production counts. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.FoldGate

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.RandomOracle

/-- What the gate reads from a CCS shape: the cube variables, the degree bound of a round
polynomial and the CCS matrices. -/
structure Counts where
  cubeVariables : ℕ
  width : ℕ
  matrices : ℕ

/-- The census shape: every fresh source that the Π_RLC guard admits and `k` running sources. -/
def Counts.shape (c : Counts) : Shape :=
  RingMatrixSource.ringShape c.cubeVariables productionGlobalParams.maxFresh
    productionGlobalParams.k c.matrices

/-- The census numerator over `q²`, SuperNeo v1.2 Appendix B.2, equation (14). -/
def Counts.numerator (c : Counts) : ℕ :=
  Spec.Folding.Light.ErrorBudget.testNumerator c.shape c.width

/-- The Π_RLC sources of a refresh fold, `1 + k`. -/
def sources : ℕ := productionProfile.piRlcInputs

/-- The challenges at which the gate charges the test error, `2ℓ + 1 + sources`. -/
def Counts.challenges (c : Counts) : ℕ := 2 * c.cubeVariables + 1 + sources

/-- The error that the gate charges. -/
def error (c : Counts) : ℚ :=
  ((c.challenges * c.numerator : ℕ) : ℚ) / ((fieldModulus ^ 2 : ℕ) : ℚ) +
    ((sources ^ 2 : ℕ) : ℚ) * (1 / (productionChallengeSetCardinality : ℚ) + 1 / 2 ^ 173)

/-- `q² · |C| · 2^173`. -/
def denominator : ℕ := fieldModulus ^ 2 * productionChallengeSetCardinality * 2 ^ 173

/-- The numerator of `error` over `denominator`. -/
def count (c : Counts) : ℕ :=
  c.challenges * c.numerator * productionChallengeSetCardinality * 2 ^ 173 +
    sources ^ 2 * (fieldModulus ^ 2 * 2 ^ 173 + fieldModulus ^ 2 * productionChallengeSetCardinality)

/-- The largest `k` with `2^k · count ≤ denominator`. -/
def bits (c : Counts) : ℕ := Nat.log 2 (denominator / count c)

/-- The counts of each step circuit: `ℓ = 28`, degree 8 and 4 matrices. -/
def production : Counts where
  cubeVariables := Lifecycle.cubeVariables
  width := Nifs.SelectedTestNumerator.width
  matrices := productionProfile.ccsMatrices

theorem production_numerator : production.numerator = 6377 := by
  rfl

private theorem cardinality_eq : productionChallengeSetCardinality = 5 ^ 64 := rfl

private theorem sources_eq : sources = 17 := rfl

theorem error_eq (c : Counts) : error c = (count c : ℚ) / (denominator : ℚ) := by
  have card : (0 : ℚ) < (productionChallengeSetCardinality : ℚ) := by
    rw [cardinality_eq]
    positivity
  have modulus : (0 : ℚ) < (fieldModulus : ℚ) := by norm_num [fieldModulus]
  unfold error count denominator
  push_cast
  field_simp
  ring

theorem count_pos (c : Counts) : 0 < count c := by
  have tail : 0 < sources ^ 2 *
      (fieldModulus ^ 2 * 2 ^ 173 + fieldModulus ^ 2 * productionChallengeSetCardinality) := by
    decide
  unfold count
  exact Nat.lt_of_lt_of_le tail (Nat.le_add_left _ _)

/-- `bits` is the largest `k` with `2^k · count ≤ denominator`. -/
theorem bits_spec (c : Counts) (le : count c ≤ denominator) :
    2 ^ bits c * count c ≤ denominator ∧ denominator < 2 ^ (bits c + 1) * count c := by
  have pos := count_pos c
  unfold bits
  exact ⟨(Nat.le_div_iff_mul_le pos).mp (Nat.pow_log_le_self 2 (Nat.div_pos le pos).ne'),
    (Nat.div_lt_iff_lt_mul pos).mp (Nat.lt_pow_succ_log_self (by norm_num) _)⟩

/-- When `bits` is positive, `error ≤ 2^−bits`. -/
theorem error_le_bits (c : Counts) (pos : 0 < bits c) : (error c : ℝ) ≤ 1 / 2 ^ bits c := by
  have le : count c ≤ denominator := by
    by_contra above
    unfold bits at pos
    rw [Nat.div_eq_of_lt (not_le.mp above), Nat.log_zero_right] at pos
    exact lt_irrefl _ pos
  have spec : count c * 2 ^ bits c ≤ denominator := by
    rw [Nat.mul_comm]
    exact (bits_spec c le).1
  have positive : (0 : ℝ) < (denominator : ℝ) := by
    exact_mod_cast Nat.lt_of_lt_of_le (count_pos c) le
  rw [error_eq]
  push_cast
  rw [div_le_div_iff₀ positive (by positivity), one_mul]
  exact_mod_cast spec

theorem production_bits : bits production = 109 := by
  unfold bits
  rw [Nat.log_eq_iff (Or.inl (by norm_num))]
  constructor <;> decide

private theorem sampleError_lt : Fork.sampleError < 1 / 5 ^ 64 + 1 / 2 ^ 173 := by
  have distance := Spec.Folding.Nifs.NonInteractive.PiRlcSampler.distance_lt
  have count : (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.scalarCount : ℝ) = 5 ^ 64 := by
    norm_num [Spec.Folding.Nifs.NonInteractive.PiRlcSampler.scalarCount,
      RingStrongSet.alphabetSize, RingStrongSet.coefficientCount, ringDegree]
  unfold Fork.sampleError
  rw [count]
  linarith

private theorem error_production :
    ((error production : ℚ) : ℝ) =
      74 * 6377 * (1 / (fieldModulus : ℝ) ^ 2) + 289 * (1 / 5 ^ 64 + 1 / 2 ^ 173) := by
  have challenges : production.challenges = 74 := rfl
  unfold error
  rw [challenges, production_numerator, sources_eq, cardinality_eq]
  push_cast
  ring

/-- The statistical term of the refresh fold with `Q` queries is at most `(Q + 1) · error`. -/
theorem refresh_statisticalError_le (queries : ℕ) :
    RefreshKnowledge.statisticalError queries ≤ ((queries : ℝ) + 1) * (error production : ℝ) := by
  rw [error_production]
  have sample := sampleError_lt
  have nonnegative := Fork.sampleError_nonnegative
  have inverse : (0 : ℝ) < 1 / (fieldModulus : ℝ) ^ 2 := by norm_num [fieldModulus]
  have queriesNonnegative : (0 : ℝ) ≤ queries := Nat.cast_nonneg queries
  unfold RefreshKnowledge.statisticalError
  push_cast
  rw [div_eq_mul_one_div (5389 : ℝ)]
  nlinarith [mul_nonneg queriesNonnegative inverse.le,
    mul_nonneg queriesNonnegative (sub_nonneg.mpr sample.le),
    mul_nonneg queriesNonnegative nonnegative]

/-- The statistical term of the light fold with `Q` queries is at most `(Q + 1) · error`. -/
theorem light_statisticalError_le (queries : ℕ) :
    LightKnowledge.statisticalError queries ≤ ((queries : ℝ) + 1) * (error production : ℝ) := by
  rw [error_production]
  have sample := sampleError_lt
  have nonnegative := Fork.sampleError_nonnegative
  have inverse : (0 : ℝ) < 1 / (fieldModulus : ℝ) ^ 2 := by norm_num [fieldModulus]
  have queriesNonnegative : (0 : ℝ) ≤ queries := Nat.cast_nonneg queries
  unfold LightKnowledge.statisticalError
  push_cast
  rw [div_eq_mul_one_div (574 : ℝ)]
  nlinarith [mul_nonneg queriesNonnegative inverse.le,
    mul_nonneg queriesNonnegative (sub_nonneg.mpr sample.le),
    mul_nonneg queriesNonnegative nonnegative]

private theorem bound_le_bits (queries : ℕ) :
    ((queries : ℝ) + 1) * (error production : ℝ) ≤ ((queries : ℝ) + 1) / 2 ^ bits production := by
  have bound := error_le_bits production (by rw [production_bits]; norm_num)
  rw [div_eq_mul_one_div]
  exact mul_le_mul_of_nonneg_left bound (by positivity)

/-- The statistical term of the refresh fold with `Q` queries is at most `(Q + 1)/2^bits`. -/
theorem refresh_statisticalError_le_bits (queries : ℕ) :
    RefreshKnowledge.statisticalError queries ≤ ((queries : ℝ) + 1) / 2 ^ bits production :=
  (refresh_statisticalError_le queries).trans (bound_le_bits queries)

/-- The statistical term of the light fold with `Q` queries is at most `(Q + 1)/2^bits`. -/
theorem light_statisticalError_le_bits (queries : ℕ) :
    LightKnowledge.statisticalError queries ≤ ((queries : ℝ) + 1) / 2 ^ bits production :=
  (light_statisticalError_le queries).trans (bound_le_bits queries)

end NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.FoldGate
