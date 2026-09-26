import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Hybrid

/-! Conditional averaging retains the four raw field values in the security
experiment. Replacing a sampled challenge by a uniform challenge must also
sample a consistent raw preimage; it must not expose an unrelated uniform
challenge next to an unchanged raw block. -/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler

open Finset
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet

section Conditional

variable {α β : Type*} [Fintype α] [Fintype β] [DecidableEq β]

noncomputable def fiberAverage (map : α → β) (test : α → ℝ) (value : β) : ℝ :=
  average (fun preimage : {input // map input = value} => test preimage.val)

omit [Fintype β] in
theorem fiberAverage_nonnegative (map : α → β) (test : α → ℝ)
    (nonnegative : ∀ input, 0 ≤ test input) (value : β) :
    0 ≤ fiberAverage map test value :=
  average_nonnegative _ (fun input => nonnegative input.val)

omit [Fintype β] in
theorem fiberAverage_le_one (map : α → β) (test : α → ℝ)
    (atMostOne : ∀ input, test input ≤ 1) (value : β) :
    fiberAverage map test value ≤ 1 := by
  by_cases inhabited : Nonempty {input // map input = value}
  · letI := inhabited
    exact average_le_one _ (fun input => atMostOne input.val)
  · letI : IsEmpty {input // map input = value} := not_nonempty_iff.mp inhabited
    simp [fiberAverage, average]

/-- Averaging a conditional expectation over the original uniform input
recovers the original expectation. Empty fibers contribute zero. -/
theorem average_fiberAverage (map : α → β) (test : α → ℝ) :
    average (fun input => fiberAverage map test (map input)) = average test := by
  let equivalence := Equiv.sigmaFiberEquiv map
  rw [← average_comp_equiv equivalence (fun input => fiberAverage map test (map input)),
    ← average_comp_equiv equivalence test]
  unfold average
  rw [Fintype.sum_sigma, Fintype.sum_sigma]
  congr 1
  apply sum_congr rfl
  intro value _
  have same : ∀ input : {input // map input = value},
      fiberAverage map test (map (equivalence ⟨value, input⟩)) = fiberAverage map test value := by
    intro input
    exact congrArg (fiberAverage map test) input.property
  simp only [same, sum_const, card_univ, nsmul_eq_mul]
  change (Fintype.card {input // map input = value} : ℝ) *
    ((∑ input : {input // map input = value}, test input.val) /
      Fintype.card {input // map input = value}) = _
  by_cases inhabited : Nonempty {input // map input = value}
  · letI := inhabited
    have nonzero : (Fintype.card {input // map input = value} : ℝ) ≠ 0 := by
      exact_mod_cast Fintype.card_ne_zero
    field_simp
    rfl
  · letI : IsEmpty {input // map input = value} := not_nonempty_iff.mp inhabited
    simp

omit [Fintype β] in
theorem fiberAverage_comp (map : α → β) (test : β → ℝ) (value : β)
    (inhabited : Nonempty {input // map input = value}) :
    fiberAverage map (test ∘ map) value = test value := by
  letI := inhabited
  have same : ∀ input : {input // map input = value}, test (map input.val) = test value :=
    fun input => congrArg test input.property
  have nonzero : (Fintype.card {input // map input = value} : ℝ) ≠ 0 := by
    exact_mod_cast Fintype.card_ne_zero
  unfold fiberAverage average
  simp only [Function.comp_apply, same, sum_const, card_univ, nsmul_eq_mul]
  field_simp

end Conditional

/-- A computable preimage witnesses that every scalar has a nonempty fiber. -/
def representative (scalar : Scalar) : Draw :=
  drawIndex.symm ⟨(scalarIndex scalar).val, lt_of_lt_of_le (scalarIndex scalar).isLt (by decide)⟩

theorem sample_representative (scalar : Scalar) : sample (representative scalar) = scalar := by
  apply (sample_eq_iff _ scalar).mpr
  rw [representative, Equiv.apply_symm_apply]
  exact Nat.mod_eq_of_lt (scalarIndex scalar).isLt

def sampleVector {Index : Type*} (blocks : Index → Draw) : Index → Scalar :=
  fun index => sample (blocks index)

theorem sampleVector_surjective {Index : Type*} : Function.Surjective (sampleVector (Index := Index)) := by
  intro vector
  refine ⟨fun index => representative (vector index), ?_⟩
  funext index
  exact sample_representative (vector index)

/-- Uniform scalar vector, followed by a uniform raw preimage conditional
on that vector. The raw block and its sampled scalar remain consistent. -/
noncomputable def balancedAverage {Index : Type*} [Fintype Index] [DecidableEq Index]
    (test : (Index → Draw) → ℝ) : ℝ :=
  average (fiberAverage sampleVector test)

theorem balanced_sampleVector {Index : Type*} [Fintype Index] [DecidableEq Index]
    (test : (Index → Scalar) → ℝ) :
    balancedAverage (test ∘ sampleVector) = average test := by
  unfold balancedAverage
  congr 1
  funext vector
  apply fiberAverage_comp
  obtain ⟨blocks, same⟩ := sampleVector_surjective vector
  exact ⟨⟨blocks, same⟩⟩

/-- The same bias bound holds even when a test sees every raw four-field
block, rather than only the resulting scalars. -/
theorem raw_blocks_difference_abs_le {Index : Type*} [Fintype Index] [DecidableEq Index]
    (test : (Index → Draw) → ℝ) (nonnegative : ∀ blocks, 0 ≤ test blocks)
    (atMostOne : ∀ blocks, test blocks ≤ 1) :
    |average test - balancedAverage test| ≤ Fintype.card Index * distance := by
  have comparison := vector_average_difference_abs_le (fiberAverage sampleVector test)
    (fiberAverage_nonnegative sampleVector test nonnegative)
    (fiberAverage_le_one sampleVector test atMostOne)
  change |average (fun blocks => fiberAverage sampleVector test (sampleVector blocks)) -
    balancedAverage test| ≤ _ at comparison
  rwa [average_fiberAverage] at comparison

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler
