import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NumericBooleanDomain
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanReproduction

/-!
Owns executable folding of a numeric table prefix and its exact MLE meaning.
Adjacent entries fix the least-significant remaining bit. Missing entries
are zero, and every challenge is consumed, including after the prefix has
length one. No matrix, assignment-validity, or protocol premise is required.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PrefixFold

universe uField

variable {Field : Type uField}

/-- The same affine interpolation used by `BooleanTable.evaluate`. -/
def interpolate (ops : InterpolationOps Field)
    (coordinate low high : Field) : Field :=
  ops.add low (ops.mul coordinate (ops.sub high low))

/-- Embed an ascending numeric prefix into the existing little-endian cube.
The prefix bound is stated by the refinement theorem, not hidden in a read. -/
def zeroExtend (ops : InterpolationOps Field)
    (variables : Nat) (values : Array Field) : BooleanTable Field variables :=
  BooleanTable.tabulate fun vertex =>
    values.getD (NumericBooleanDomain.index vertex) ops.zero

/-- Fix the next low bit. An odd final entry is paired with implicit zero. -/
def foldOne (ops : InterpolationOps Field)
    (values : Array Field) (challenge : Field) : Array Field :=
  Array.ofFn fun pair : Fin ((values.size + 1) / 2) =>
    interpolate ops challenge
      (values.getD (2 * pair.val) ops.zero)
      (values.getD (2 * pair.val + 1) ops.zero)

/-- Consume challenges in coordinate order. A singleton is still folded
against zero; its value is not treated as a constant on the remaining cube. -/
def foldPrefix (ops : InterpolationOps Field) :
    Array Field → List Field → Array Field
  | values, [] => values
  | values, challenge :: challenges =>
      foldPrefix ops (foldOne ops values challenge) challenges

/-- The executable fold retains exactly the required adjacent pairs. -/
theorem foldOne_size (ops : InterpolationOps Field)
    (values : Array Field) (challenge : Field) :
    (foldOne ops values challenge).size = (values.size + 1) / 2 := by
  simp only [foldOne, Array.size_ofFn]

private theorem getD_outside (ops : InterpolationOps Field)
    (values : Array Field) (index : Nat) (outside : values.size ≤ index) :
    values.getD index ops.zero = ops.zero := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none outside,
    Option.getD_none]

/-- Every numeric result coordinate is the interpolation of entries `2u`
and `2u+1`, including coordinates beyond the stored prefix. -/
theorem foldOne_getD (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (values : Array Field) (challenge : Field) (index : Nat) :
    (foldOne ops values challenge).getD index ops.zero =
      interpolate ops challenge
        (values.getD (2 * index) ops.zero)
        (values.getD (2 * index + 1) ops.zero) := by
  unfold foldOne
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_ofFn]
  by_cases live : index < (values.size + 1) / 2
  · simp only [dif_pos live, Option.getD_some]
  · have lowOutside : values.size ≤ 2 * index := by omega
    have highOutside : values.size ≤ 2 * index + 1 := by omega
    rw [dif_neg live, Option.getD_none,
      getD_outside ops values _ lowOutside,
      getD_outside ops values _ highOutside]
    unfold interpolate InterpolationOps.sub
    rw [laws.add_neg, laws.mul_zero, laws.add_zero]

/-- Folding a covered prefix preserves coverage by the remaining cube. -/
theorem foldOne_fits (ops : InterpolationOps Field)
    (values : Array Field) (challenge : Field) (variables : Nat)
    (fits : values.size ≤ 2 ^ (variables + 1)) :
    (foldOne ops values challenge).size ≤ 2 ^ variables := by
  rw [foldOne_size]
  rw [Nat.pow_succ] at fits
  omega

/-- All executable prefix folds remain within the exact remaining domain. -/
theorem foldPrefix_fits (ops : InterpolationOps Field)
    (values : Array Field) (challenges : List Field) (variables : Nat)
    (fits : values.size ≤ 2 ^ (variables + challenges.length)) :
    (foldPrefix ops values challenges).size ≤ 2 ^ variables := by
  induction challenges generalizing values with
  | nil => simpa only [foldPrefix, List.length_nil, Nat.add_zero] using fits
  | cons challenge challenges inductionHypothesis =>
      apply inductionHypothesis (foldOne ops values challenge)
      exact foldOne_fits ops values challenge (variables + challenges.length) fits

private theorem mul_interpolate (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (weight coordinate low high : Field) :
    ops.mul weight (interpolate ops coordinate low high) =
      ops.add (ops.mul weight low)
        (ops.mul coordinate
          (ops.sub (ops.mul weight high) (ops.mul weight low))) := by
  unfold interpolate
  rw [laws.left_distrib]
  congr 1
  calc
    ops.mul weight (ops.mul coordinate (ops.sub high low)) =
        ops.mul (ops.mul weight coordinate) (ops.sub high low) :=
      (laws.mul_assoc _ _ _).symm
    _ = ops.mul (ops.mul coordinate weight) (ops.sub high low) := by
      rw [laws.mul_comm weight coordinate]
    _ = ops.mul coordinate (ops.mul weight (ops.sub high low)) :=
      laws.mul_assoc _ _ _
    _ = _ := by rw [FiniteSumAlgebra.mul_sub ops laws]

private theorem evaluate_tabulate_interpolate (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables : Nat} (coordinate : Field)
    (low high : BooleanVertex variables → Field)
    (point : CubePoint Field variables) :
    (BooleanTable.tabulate fun vertex =>
      interpolate ops coordinate (low vertex) (high vertex)).evaluate ops point =
      interpolate ops coordinate
        ((BooleanTable.tabulate low).evaluate ops point)
        ((BooleanTable.tabulate high).evaluate ops point) := by
  rw [← BooleanReproduction.equalityWeighted_tabulate_eq_evaluate ops laws point,
    ← BooleanReproduction.equalityWeighted_tabulate_eq_evaluate ops laws point low,
    ← BooleanReproduction.equalityWeighted_tabulate_eq_evaluate ops laws point high]
  unfold BooleanReproduction.equalityWeighted
  calc
    _ = FiniteSumAlgebra.sumMap ops (BooleanVertex.all variables) (fun vertex =>
        ops.add (ops.mul (vertex.equalityWeight ops point) (low vertex))
          (ops.mul coordinate
            (ops.sub (ops.mul (vertex.equalityWeight ops point) (high vertex))
              (ops.mul (vertex.equalityWeight ops point) (low vertex))))) := by
      apply FiniteSumAlgebra.sumMap_congr
      intro vertex _
      exact mul_interpolate ops laws _ _ _ _
    _ = _ := by
      unfold interpolate
      rw [FiniteSumAlgebra.sumMap_add ops laws,
        FiniteSumAlgebra.sumMap_mul_left ops laws,
        FiniteSumAlgebra.sumMap_sub ops laws]

/-- One adjacent numeric fold fixes precisely the head MLE coordinate.
No assumption is made about the table values or the challenge. -/
theorem foldOne_evaluate (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (values : Array Field) (challenge : Field)
    {variables : Nat} (suffix : CubePoint Field variables) :
    (zeroExtend ops variables (foldOne ops values challenge)).evaluate ops suffix =
      (zeroExtend ops (variables + 1) values).evaluate ops
        ⟨challenge :: suffix.coordinates, by simp [suffix.dimension]⟩ := by
  let low : BooleanVertex variables → Field := fun vertex =>
    values.getD (2 * NumericBooleanDomain.index vertex) ops.zero
  let high : BooleanVertex variables → Field := fun vertex =>
    values.getD (1 + 2 * NumericBooleanDomain.index vertex) ops.zero
  have folded : zeroExtend ops variables (foldOne ops values challenge) =
      BooleanTable.tabulate (fun vertex =>
        interpolate ops challenge (low vertex) (high vertex)) := by
    unfold zeroExtend
    apply congrArg BooleanTable.tabulate
    funext vertex
    simpa only [low, high, Nat.add_comm] using
      foldOne_getD ops laws values challenge (NumericBooleanDomain.index vertex)
  have original : zeroExtend ops (variables + 1) values =
      BooleanTable.branch (BooleanTable.tabulate low) (BooleanTable.tabulate high) := by
    unfold zeroExtend
    change BooleanTable.branch _ _ = BooleanTable.branch _ _
    apply congrArg (fun lowTable : BooleanTable Field variables =>
      BooleanTable.branch lowTable (BooleanTable.tabulate high))
    apply congrArg BooleanTable.tabulate
    funext vertex
    change values.getD (0 + 2 * NumericBooleanDomain.index vertex) ops.zero =
      values.getD (2 * NumericBooleanDomain.index vertex) ops.zero
    rw [Nat.zero_add]
  rw [folded, original]
  exact evaluate_tabulate_interpolate ops laws challenge low high suffix

/-- Exact executable prefix refinement. The original array fits the full
cube; after all prefix challenges, its zero extension has the same MLE at
the suffix as the original table at the concatenated point. -/
theorem foldPrefix_evaluate (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (values : Array Field) (challenges : List Field)
    {variables : Nat} (suffix : CubePoint Field variables)
    (fits : values.size ≤ 2 ^ (variables + challenges.length)) :
    (zeroExtend ops variables (foldPrefix ops values challenges)).evaluate ops suffix =
      (zeroExtend ops (variables + challenges.length) values).evaluate ops
        ⟨challenges ++ suffix.coordinates, by simp [suffix.dimension, Nat.add_comm]⟩ := by
  induction challenges generalizing values with
  | nil => rfl
  | cons challenge challenges inductionHypothesis =>
      have nextFits : (foldOne ops values challenge).size ≤
          2 ^ (variables + challenges.length) :=
        foldOne_fits ops values challenge (variables + challenges.length) fits
      rw [foldPrefix, inductionHypothesis (foldOne ops values challenge) nextFits]
      exact foldOne_evaluate ops laws values challenge
        ⟨challenges ++ suffix.coordinates, by simp [suffix.dimension, Nat.add_comm]⟩

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PrefixFold
