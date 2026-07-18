import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction

/-!
Focused regression for canonical Boolean-selector reproduction.

Owns: one two-variable integer fixture with an off-cube target, the neutral
Bool-to-field encoding, and an order-sensitive little-endian tensor-selector
reproduction check.

Does not own: generic soundness, protocol tables, SumCheck, Rust, R1CS, or
constraint counts.

| Stage path | Fixture | Expected result |
|---|---|---|
| `pi_ccs.boolean.selector.partition.regression` | target `[2,3]` | four equality weights sum to one |
| `pi_ccs.boolean.selector.reproduce.regression` | table values `1,2,4,8` | weighted sum equals recursive MLE |
| `pi_ccs.boolean.vertex.field_point.regression` | vertex bits `[false,true]` | field coordinates `[0,1]` |
| `pi_ccs.boolean.selector.tensor.regression` | numeric index `2 = 0b10`, target `[2,3]` | reproduced tensor weight is `-3` |
| `pi_ccs.boolean.selector.order.regression` | `BooleanVertex.all 2` | mapped indices are `[0,2,1,3]`, never assumed numeric order |
-/

namespace tests.PiCcsPaperJointBooleanReproduction

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction

private def integerOps : InterpolationOps Int where
  zero := 0
  one := 1
  add := Int.add
  mul := Int.mul
  neg := Int.neg

private def integerLaws : InterpolationEvaluationLaws integerOps where
  add_assoc := by intros; simp [integerOps, Int.add_assoc]
  add_comm := by intros; simp [integerOps, Int.add_comm]
  zero_add := by intro; simp [integerOps]
  add_zero := by intro; simp [integerOps]
  mul_assoc := by intro; simp [integerOps, Int.mul_assoc]
  mul_comm := by intro; simp [integerOps, Int.mul_comm]
  one_mul := by intro; simp [integerOps]
  mul_one := by intro; simp [integerOps]
  mul_zero := by intro; simp [integerOps]
  left_distrib := by intro; simp [integerOps, Int.mul_add]
  right_distrib := by intro; simp [integerOps, Int.add_mul]
  add_neg := by
    intro value
    change value + -value = 0
    exact Int.add_right_neg value
  neg_add := by
    intro left right
    change -(left + right) = -left + -right
    exact Int.neg_add
  neg_mul := by
    intro left right
    change -left * right = -(left * right)
    exact Int.neg_mul left right

private def target : CubePoint Int 2 where
  coordinates := [2, 3]
  dimension := by decide

private def values : BooleanVertex 2 -> Int
  | .cons false (.cons false .nil) => 1
  | .cons false (.cons true .nil) => 2
  | .cons true (.cons false .nil) => 4
  | .cons true (.cons true .nil) => 8

private def binaryTen : Fin (2 ^ 2) := ⟨2, by decide⟩

private def falseTrue : BooleanVertex 2 :=
  .cons false (.cons true .nil)

/-- The theorem specializes to an off-cube target where individual equality
weights are neither zero nor one. -/
example :
    FiniteSumAlgebra.sumMap integerOps (BooleanVertex.all 2)
        (fun vertex => vertex.equalityWeight integerOps target) = 1 :=
  equalityWeight_sum_eq_one integerOps integerLaws target

/-- The same off-cube fixture reproduces the independently recursive table
MLE, guarding both enumeration order and low/high interpolation. -/
example :
    equalityWeighted integerOps target values =
      (BooleanTable.tabulate values).evaluate integerOps target :=
  equalityWeighted_tabulate_eq_evaluate integerOps integerLaws target values

/-- The concrete value makes the order-sensitive fixture executable. -/
example : equalityWeighted integerOps target values = 28 := by
  decide

/-- The neutral Boolean point encoder preserves prepended, little-endian
coordinate order. -/
example : falseTrue.fieldCoordinates integerOps = [0, 1] := by
  rfl

/-- The generic tensor-selector reproduction theorem specializes to the
order-sensitive index-two fixture. -/
example :
    FiniteSumAlgebra.sumMap integerOps (BooleanVertex.all 2) (fun sampled =>
        integerOps.mul (sampled.equalityWeight integerOps target)
          (NumericBooleanDomain.tensorWeight integerOps binaryTen
            (sampled.toCubePoint integerOps))) =
      NumericBooleanDomain.tensorWeight integerOps binaryTen target :=
  equalityWeighted_tensorWeight_eq_tensorWeight
    integerOps integerLaws binaryTen target

/-- Index two has bits `[false,true]`, so its target weight is
`(1 - 2) * 3 = -3`. -/
example :
    NumericBooleanDomain.tensorWeight integerOps binaryTen target = -3 := by
  decide

/-- The semantic low/high traversal is not numeric-index order. This catches
any accidental replacement of `BooleanVertex.all` by `canonicalFinIndices`. -/
example :
    (BooleanVertex.all 2).map NumericBooleanDomain.index = [0, 2, 1, 3] := by
  decide

example :
    (BooleanVertex.all 2).map NumericBooleanDomain.index ≠ [0, 1, 2, 3] := by
  decide

end tests.PiCcsPaperJointBooleanReproduction
