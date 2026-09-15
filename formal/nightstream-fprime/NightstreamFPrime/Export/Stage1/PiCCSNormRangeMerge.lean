import NightstreamFPrime.Export.Stage1.PiCCSNormScanCorrectness

/-! Ordered merging of the implemented norm scan's worker polynomials.
Every partial is computed from the same powers, weights and original masks.
The equalities preserve the complete coefficient object. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSNormRangeMerge

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open PiCCSNormScanCorrectness

/-- Ascending contiguous cuts give exactly one scan range, including empty
parts. Each worker value is the actual finished scan, not a supplied reference. -/
theorem fold_adjacent_eq_range (powers weight : Nat → K)
    (masks : Array (Array (Nat × Nat))) (start : Nat) (cuts : Nat → Nat)
    (first : cuts 0 = 0) (count : Nat) :
    (∀ index, index < count → cuts index ≤ cuts (index + 1)) →
    Nat.fold count (fun index _ accumulated =>
      FixedPolynomial.add extensionOps.toOps accumulated
        (PiCCSNormBuckets.finish powers
          (PiCCSNormScan.range weight masks (start + cuts index)
            (cuts (index + 1) - cuts index))))
      (FixedPolynomial.zero extensionOps.toOps 3) =
      PiCCSNormBuckets.finish powers (PiCCSNormScan.range weight masks start (cuts count)) := by
  induction count with
  | zero =>
      intro _
      have emptyRange : PiCCSNormScan.range weight masks start (cuts 0) =
          PiCCSNormBuckets.empty := by
        rw [first]
        rfl
      exact ((congrArg (PiCCSNormBuckets.finish powers) emptyRange).trans
        (PiCCSNormBuckets.finish_empty powers)).symm
  | succ count inductionHypothesis =>
      intro ordered
      have previous := inductionHypothesis
        (fun index live => ordered index (Nat.lt_trans live (Nat.lt_succ_self count)))
      have completeCount : cuts count + (cuts (count + 1) - cuts count) = cuts (count + 1) :=
        Nat.add_sub_of_le (ordered count (Nat.lt_succ_self count))
      have joined := PiCCSPolynomialRange.range_append extensionOps extensionLaws start
        (cuts count) (cuts (count + 1) - cuts count)
        (fun index => blockNorm powers weight index (masks[index]?.getD #[]))
      rw [completeCount] at joined
      rw [Nat.fold_succ, previous]
      simp only [range_finish]
      exact joined.symm

/-- The runner's proportional cuts start at first, end at last, and remain
ordered. This arithmetic uses no proof or matrix data to select a cut. -/
theorem proportionalCuts (first last parts : Nat) (positive : 0 < parts)
    (ordered : first ≤ last) :
    first + (last - first) * 0 / parts = first ∧
      first + (last - first) * parts / parts = last ∧
      ∀ index, index < parts →
        first + (last - first) * index / parts ≤
          first + (last - first) * (index + 1) / parts ∧
        first + (last - first) * (index + 1) / parts ≤ last := by
  have lastCut : first + (last - first) * parts / parts = last := by
    rw [Nat.mul_div_cancel (last - first) positive, Nat.add_sub_of_le ordered]
  refine ⟨by simp, lastCut, ?_⟩
  intro index inside
  constructor
  · exact Nat.add_le_add_left
      (Nat.div_le_div_right (Nat.mul_le_mul_left (last - first) (Nat.le_succ index))) first
  · calc
      first + (last - first) * (index + 1) / parts ≤
          first + (last - first) * parts / parts :=
        Nat.add_le_add_left
          (Nat.div_le_div_right (Nat.mul_le_mul_left (last - first) (by omega))) first
      _ = last := lastCut

/-- The exact cuts used by replayNorm merge to its requested complete range.
No positivity condition is needed for each part's length. -/
theorem fold_proportional_eq_range (powers weight : Nat → K)
    (masks : Array (Array (Nat × Nat))) (first last parts : Nat)
    (positive : 0 < parts) (ordered : first ≤ last) :
    Nat.fold parts (fun index _ accumulated =>
      FixedPolynomial.add extensionOps.toOps accumulated
        (PiCCSNormBuckets.finish powers
          (PiCCSNormScan.range weight masks (first + (last - first) * index / parts)
            ((first + (last - first) * (index + 1) / parts) -
              (first + (last - first) * index / parts)))))
      (FixedPolynomial.zero extensionOps.toOps 3) =
      PiCCSNormBuckets.finish powers (PiCCSNormScan.range weight masks first (last - first)) := by
  let cuts : Nat → Nat := fun index => (last - first) * index / parts
  have properties := proportionalCuts first last parts positive ordered
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
  have merged := fold_adjacent_eq_range powers weight masks first cuts firstCut parts increasing
  rw [finalLength] at merged
  simpa only [cuts, Nat.add_sub_add_left] using merged

/-- Collecting worker results in canonical index order and folding the array
performs the same exact merge as the numeric loop. -/
theorem array_proportional_eq_range (powers weight : Nat → K)
    (masks : Array (Array (Nat × Nat))) (first last parts : Nat)
    (positive : 0 < parts) (ordered : first ≤ last) :
    (Array.ofFn (fun index : Fin parts =>
      PiCCSNormBuckets.finish powers
        (PiCCSNormScan.range weight masks (first + (last - first) * index.val / parts)
          ((first + (last - first) * (index.val + 1) / parts) -
            (first + (last - first) * index.val / parts))))).foldl
      (FixedPolynomial.add extensionOps.toOps) (FixedPolynomial.zero extensionOps.toOps 3) =
      PiCCSNormBuckets.finish powers (PiCCSNormScan.range weight masks first (last - first)) := by
  rw [← Array.foldl_toList, Array.toList_ofFn, List.ofFn_eq_map, List.foldl_map]
  have merged := fold_proportional_eq_range powers weight masks first last parts positive ordered
  rw [Nat.fold_eq_finRange_foldl] at merged
  exact merged

end NightstreamFPrime.Export.Stage1.PiCCSNormRangeMerge
