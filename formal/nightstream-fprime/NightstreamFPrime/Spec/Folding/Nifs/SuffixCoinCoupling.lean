import Mathlib.Probability.ProbabilityMassFunction.Constructions
import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
Probability coupling for the relayed prover in SuperNeo B.1. A finite table
of suffix outcomes is independent of the verifier coins; looking up the
selected coin tuple has exactly the intended conditional suffix law.
The table is a proof device, not an executed sampler or a running-time model.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.SuffixCoinCoupling

open scoped BigOperators

variable {Request Endpoint : Type*} [Fintype Request] [DecidableEq Request]
  [Fintype Endpoint] [DecidableEq Endpoint]

omit [DecidableEq Endpoint] in
private theorem total (law : PMF Endpoint) : ∑ endpoint, law endpoint = 1 := by
  simpa only [tsum_fintype] using law.tsum_coe

/-- Independent suffix choices for every possible verifier coin tuple. -/
noncomputable def tableLaw (laws : Request → PMF Endpoint) : PMF (Request → Endpoint) :=
  PMF.ofFintype (fun table => ∏ request, laws request (table request)) (by
    rw [← Fintype.prod_sum]
    simp only [total, Finset.prod_const_one])

/-- Only the selected entry is observed. Its distribution is the original
suffix law for that request, including every abort or rejection outcome. -/
theorem selected_law (laws : Request → PMF Endpoint) (request : Request) :
    (tableLaw laws).map (fun table => table request) = laws request := by
  classical
  apply PMF.ext
  intro endpoint
  let selected := fun (index : Request) (value : Endpoint) =>
    if index = request then (if endpoint = value then laws index value else 0)
    else laws index value
  have product : ∀ table : Request → Endpoint,
      (∏ index, selected index (table index)) =
        if endpoint = table request then ∏ index, laws index (table index) else 0 := by
    intro table
    by_cases equal : endpoint = table request
    · rw [if_pos equal]
      apply Finset.prod_congr rfl
      intro index _
      by_cases same : index = request
      · subst index
        simp only [selected, ↓reduceIte, equal]
      · simp only [selected, if_neg same]
    · rw [if_neg equal]
      apply Finset.prod_eq_zero (Finset.mem_univ request)
      simp only [selected, ↓reduceIte, if_neg equal]
  rw [PMF.map_apply, tsum_fintype]
  simp only [tableLaw, PMF.ofFintype_apply]
  simp_rw [← product]
  rw [← Fintype.prod_sum]
  rw [Finset.prod_eq_single request]
  · simp [selected]
  · intro index _ different
    simp only [selected, if_neg different, total]
  · simp

/-- A finite pushforward preserves the mean of every real-valued observable. -/
theorem mean_map {Source Target : Type*} [Fintype Source] [Fintype Target]
    [DecidableEq Target] (law : PMF Source) (select : Source → Target) (value : Target → ℝ) :
    (∑ source, (law source).toReal * value (select source)) =
      ∑ target, (law.map select target).toReal * value target := by
  have entry : ∀ target,
      (law.map select target).toReal =
        ∑ source, if target = select source then (law source).toReal else 0 := by
    intro target
    rw [PMF.map_apply, tsum_fintype, ENNReal.toReal_sum]
    · apply Finset.sum_congr rfl
      intro source _
      split_ifs <;> simp
    · intro source _
      split_ifs
      · exact law.apply_ne_top source
      · exact ENNReal.zero_ne_top
  simp_rw [entry, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro source _
  simp only [ite_mul, zero_mul]
  simp

/-- This equality connects the coupled private tape to the actual selected
suffix experiment before applying the strong reduction's probability bound. -/
theorem selected_mean (laws : Request → PMF Endpoint) (request : Request)
    (value : Endpoint → ℝ) :
    (∑ table, (tableLaw laws table).toReal * value (table request)) =
      ∑ endpoint, (laws request endpoint).toReal * value endpoint := by
  rw [mean_map (tableLaw laws) (fun table => table request) value, selected_law]

end NightstreamFPrime.Spec.Folding.Nifs.SuffixCoinCoupling
