import NightstreamFPrime.Export.Stage1.PiCCSNormRangeMerge

/-! Proof-only connection of ordered polynomial worker results to one complete
range. Proportional cut arithmetic is reused from the checked norm merger. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSPolynomialRangeMerge

open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uField
variable {Field : Type uField}

private theorem fold_adjacent_eq_range (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {degree : Nat}
    (term : Nat → FixedPolynomial Field degree)
    (start : Nat) (cuts : Nat → Nat) (first : cuts 0 = 0) (count : Nat) :
    (∀ index, index < count → cuts index ≤ cuts (index + 1)) →
    Nat.fold count (fun index _ accumulated =>
      FixedPolynomial.add ops.toOps accumulated
        (PiCCSPolynomialRange.range ops (start + cuts index)
          (cuts (index + 1) - cuts index) term))
      (FixedPolynomial.zero ops.toOps degree) =
      PiCCSPolynomialRange.range ops start (cuts count) term := by
  induction count with
  | zero =>
      intro _
      rw [first]
      rfl
  | succ count ih =>
      intro ordered
      have previous := ih
        (fun index live => ordered index (Nat.lt_trans live (Nat.lt_succ_self count)))
      have completeCount : cuts count + (cuts (count + 1) - cuts count) = cuts (count + 1) :=
        Nat.add_sub_of_le (ordered count (Nat.lt_succ_self count))
      have joined := PiCCSPolynomialRange.range_append ops laws start
        (cuts count) (cuts (count + 1) - cuts count) term
      rw [completeCount] at joined
      rw [Nat.fold_succ, previous]
      exact joined.symm

/-- Each worker computes its exact proportional subrange. Reading those
workers in array order and adding all stored coefficients gives the complete
requested range. Empty individual parts and first = last are included. -/
theorem array_proportional_eq_range (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {degree : Nat}
    (term : Nat → FixedPolynomial Field degree) (first last parts : Nat)
    (positive : 0 < parts) (ordered : first ≤ last)
    (worker : Fin parts → FixedPolynomial Field degree)
    (worker_value : ∀ index,
      worker index = PiCCSPolynomialRange.range ops
        (first + (last - first) * index.val / parts)
        ((first + (last - first) * (index.val + 1) / parts) -
          (first + (last - first) * index.val / parts)) term) :
    (Array.ofFn worker).foldl (FixedPolynomial.add ops.toOps)
        (FixedPolynomial.zero ops.toOps degree) =
      PiCCSPolynomialRange.range ops first (last - first) term := by
  let cuts : Nat → Nat := fun index => (last - first) * index / parts
  have properties := PiCCSNormRangeMerge.proportionalCuts first last parts positive ordered
  have firstCut : cuts 0 = 0 := by simp [cuts]
  have increasing : ∀ index, index < parts → cuts index ≤ cuts (index + 1) := by
    intro index inside
    have step := (properties.2.2 index inside).1
    change first + cuts index ≤ first + cuts (index + 1) at step
    exact Nat.le_of_add_le_add_left step
  have finalLength : cuts parts = last - first := by
    have endpoint := properties.2.1
    change first + cuts parts = last at endpoint
    omega
  have merged := fold_adjacent_eq_range ops laws term first cuts firstCut parts increasing
  rw [finalLength] at merged
  rw [← Array.foldl_toList, Array.toList_ofFn, List.ofFn_eq_map, List.foldl_map]
  simp_rw [worker_value]
  rw [Nat.fold_eq_finRange_foldl] at merged
  simpa only [cuts, Nat.add_sub_add_left] using merged

end NightstreamFPrime.Export.Stage1.PiCCSPolynomialRangeMerge
