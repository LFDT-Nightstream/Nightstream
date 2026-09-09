import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedMixingRoots
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanEvaluation
import NightstreamFPrime.Spec.SumCheck.GoldilocksCausal

/-!
Uniform alpha root bound for the existing Boolean-table multilinear evaluator
over K. The table is fixed before sampling. The proof follows its actual
`low + alpha * (high - low)` recursion and uses the one-variable K root count.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanMixingProbability

open scoped BigOperators
open NightstreamFPrime.Spec SumCheck.Finite
open GoldilocksCausal ConcreteCarrier

attribute [local instance] Classical.propDecidable

private theorem linear_probability_le (samples : Finset K) (low high : K)
    (nonzero : low ≠ K.zero ∨ high ≠ K.zero) :
    (𝔼 point ∈ samples, if
      extensionOps.add low (extensionOps.mul point (extensionOps.sub high low)) = K.zero
      then (1 : ℝ) else 0) ≤ 1 / (samples.card : ℝ) := by
  have coefficient : ∃ value ∈ [low, extensionOps.sub high low], value ≠ K.zero := by
    by_cases lowZero : low = K.zero
    · have highNonzero := nonzero.resolve_left (fun different => different lowZero)
      have subZero : extensionOps.sub high K.zero = high := by
        change extensionOps.add high (extensionOps.neg extensionOps.zero) = high
        rw [extensionZeroLaws.neg_zero, extensionLaws.add_zero]
      exact ⟨extensionOps.sub high low, by simp, by simpa [lowZero, subZero] using highNonzero⟩
    · exact ⟨low, by simp, lowZero⟩
  have evaluation (point : K) :
      Message.evaluateCoefficients GoldilocksRoots.ops point [low, extensionOps.sub high low] =
        extensionOps.add low (extensionOps.mul point (extensionOps.sub high low)) := by
    change extensionOps.add low
      (extensionOps.mul point (extensionOps.add (extensionOps.sub high low)
        (extensionOps.mul point extensionOps.zero))) = _
    rw [extensionLaws.mul_zero, extensionLaws.add_zero]
  simpa only [evaluation, List.length_cons, List.length_nil, Nat.reduceAdd, Nat.reduceSub, Nat.cast_one]
    using SignedMixingRoots.coefficient_root_probability_le [low, extensionOps.sub high low] samples coefficient

/-- Independent alpha draws evaluated by the canonical table recursion. -/
noncomputable def zeroProbability (samples : Finset K) {dimension : Nat}
    (table : BooleanTable K dimension) : ℝ :=
  uniformAverage samples dimension (fun coordinates =>
    if table.evaluateCoordinates extensionOps coordinates = K.zero then 1 else 0)

/-- A fixed nonzero Boolean-table MLE vanishes on uniform alpha with
probability at most its number of variables divided by the sample-set size. -/
theorem zeroProbability_le (samples : Finset K) (nonempty : samples.Nonempty)
    {dimension : Nat} (table : BooleanTable K dimension)
    (nonzero : ¬ table.AllEntriesZero extensionOps) :
    zeroProbability samples table ≤ (dimension : ℝ) / samples.card := by
  induction table with
  | leaf value =>
      have different : value ≠ K.zero := by
        simpa [BooleanTable.AllEntriesZero, BooleanTable.entries] using nonzero
      simp [zeroProbability, uniformAverage, BooleanTable.evaluateCoordinates, different]
  | @branch dimension low high lowIH highIH =>
      have selected : ∃ table : BooleanTable K dimension,
          (table = low ∨ table = high) ∧
          zeroProbability samples table ≤ (dimension : ℝ) / samples.card := by
        by_cases lowNonzero : ¬ low.AllEntriesZero extensionOps
        · exact ⟨low, Or.inl rfl, lowIH lowNonzero⟩
        · have lowZero : low.AllEntriesZero extensionOps := Classical.not_not.mp lowNonzero
          have highNonzero : ¬ high.AllEntriesZero extensionOps := by
            intro highZero
            apply nonzero
            intro value member
            rcases List.mem_append.mp member with inside | inside
            · exact lowZero value inside
            · exact highZero value inside
          exact ⟨high, Or.inr rfl, highIH highNonzero⟩
      obtain ⟨selected, fromBranch, selectedBound⟩ := selected
      have each (rest : List K) :
          (𝔼 point ∈ samples, if extensionOps.add
            (low.evaluateCoordinates extensionOps rest)
            (extensionOps.mul point (extensionOps.sub
              (high.evaluateCoordinates extensionOps rest)
              (low.evaluateCoordinates extensionOps rest))) = K.zero then (1 : ℝ) else 0) ≤
          (if selected.evaluateCoordinates extensionOps rest = K.zero then 1 else 0) +
            1 / (samples.card : ℝ) := by
        by_cases zero : selected.evaluateCoordinates extensionOps rest = K.zero
        · have atMostOne :
              (𝔼 point ∈ samples, if extensionOps.add
                (low.evaluateCoordinates extensionOps rest)
                (extensionOps.mul point (extensionOps.sub
                  (high.evaluateCoordinates extensionOps rest)
                  (low.evaluateCoordinates extensionOps rest))) = K.zero then (1 : ℝ) else 0) ≤ 1 := by
            apply Finset.expect_le nonempty
            intro point _
            split <;> norm_num
          rw [if_pos zero]
          have positive : (0 : ℝ) ≤ 1 / (samples.card : ℝ) := by positivity
          linarith
        · rw [if_neg zero, zero_add]
          apply linear_probability_le
          rcases fromBranch with equal | equal
          · exact Or.inl (by simpa only [equal] using zero)
          · exact Or.inr (by simpa only [equal] using zero)
      change (𝔼 point ∈ samples, uniformAverage samples dimension (fun rest =>
        if extensionOps.add (low.evaluateCoordinates extensionOps rest)
          (extensionOps.mul point (extensionOps.sub
            (high.evaluateCoordinates extensionOps rest)
            (low.evaluateCoordinates extensionOps rest))) = K.zero then (1 : ℝ) else 0)) ≤ _
      rw [← uniformAverage_expect samples dimension samples]
      have averaged := uniformAverage_mono samples dimension _ _ (fun rest _ => each rest)
      apply averaged.trans
      rw [uniformAverage_add, uniformAverage_const samples nonempty]
      have bound := _root_.add_le_add selectedBound (le_refl (1 / (samples.card : ℝ)))
      simpa only [zeroProbability, Nat.cast_add, Nat.cast_one, add_div] using bound

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanMixingProbability
