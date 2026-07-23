import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Equality

/-!
Contract: exact one-coordinate zero test used by `iterationZero`.

The input coordinate is tested directly; no synthetic zero column and no
literal-pin row are allocated.  The two witness equations are

```text
input * inverse = 1 - equal
input * equal   = 0
```

and one activation-gated row binds `equal` to the visible Boolean output.

Emits constraints: exactly three rows and two temporary columns.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

open Nightstream.Implementation.Lowering.Goldilocks

private def zeroInverseRow
    (one input inverse equal : ColumnId) : Row where
  a := singleton input 1
  b := singleton inverse 1
  c := oneMinus one equal

private def zeroAnnihilatorRow
    (input equal : ColumnId) : Row where
  a := singleton input 1
  b := singleton equal 1
  c := []

private def zeroFinalRow
    (active equal output : ColumnId) : Row where
  a := singleton active 1
  b := difference equal output
  c := []

private theorem singleton_eval
    (assignment : ColumnId -> Field)
    (column : ColumnId) :
    (Goldilocks.singleton column 1).eval assignment =
      assignment column := by
  simp only [Goldilocks.singleton, LinearCombination.eval, Fin.one_mul,
    Fin.add_zero]

private theorem oneMinus_eval
    (assignment : ColumnId -> Field)
    (one value : ColumnId)
    (constantOne : assignment one = 1) :
    (oneMinus one value).eval assignment =
      1 - assignment value := by
  simp only [oneMinus, LinearCombination.eval, constantOne, Fin.one_mul,
    Fin.add_zero, Lean.Grind.Fin.neg_mul, Fin.sub_eq_add_neg]

private theorem difference_eval
    (assignment : ColumnId -> Field)
    (left right : ColumnId) :
    (difference left right).eval assignment =
      assignment left - assignment right := by
  simp only [difference, LinearCombination.eval, Fin.one_mul,
    Fin.add_zero, Lean.Grind.Fin.neg_mul, Fin.sub_eq_add_neg]

private theorem zeroInverseRow_iff
    (assignment : ColumnId -> Field)
    (one input inverse equal : ColumnId)
    (constantOne : assignment one = 1) :
    (zeroInverseRow one input inverse equal).Holds assignment ↔
      assignment input * assignment inverse =
        1 - assignment equal := by
  unfold zeroInverseRow Row.Holds
  rw [singleton_eval, singleton_eval,
    oneMinus_eval assignment one equal constantOne]

private theorem zeroAnnihilatorRow_iff
    (assignment : ColumnId -> Field)
    (input equal : ColumnId) :
    (zeroAnnihilatorRow input equal).Holds assignment ↔
      assignment input * assignment equal = 0 := by
  unfold zeroAnnihilatorRow Row.Holds
  rw [singleton_eval, singleton_eval, LinearCombination.eval_nil]

private theorem zeroFinalRow_active_iff
    (assignment : ColumnId -> Field)
    (active equal output : ColumnId)
    (activeOne : assignment active = 1) :
    (zeroFinalRow active equal output).Holds assignment ↔
      assignment output = assignment equal := by
  unfold zeroFinalRow Row.Holds
  rw [singleton_eval, difference_eval, LinearCombination.eval_nil,
    activeOne, Fin.one_mul]
  constructor
  · exact fun differenceZero =>
      (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp differenceZero).symm
  · exact fun same =>
      Lean.Grind.AddCommGroup.sub_eq_zero_iff.mpr same.symm

private theorem zeroFinalRow_inactive
    (assignment : ColumnId -> Field)
    (active equal output : ColumnId)
    (activeZero : assignment active = 0) :
    (zeroFinalRow active equal output).Holds assignment := by
  unfold zeroFinalRow Row.Holds
  rw [singleton_eval, difference_eval, LinearCombination.eval_nil,
    activeZero, Fin.zero_mul]

/-- One physical zero-test occurrence. -/
structure ZeroRecipe where
  owner : PhysicalOwner
  one : ColumnId
  active : ColumnId
  input : OwnedColumn
  output : OwnedColumn
  inverse : OwnedColumn
  equal : OwnedColumn

namespace ZeroRecipe

def rawRows (recipe : ZeroRecipe) : List Row :=
  [zeroInverseRow recipe.one recipe.input.id recipe.inverse.id
      recipe.equal.id,
    zeroAnnihilatorRow recipe.input.id recipe.equal.id,
    zeroFinalRow recipe.active recipe.equal.id recipe.output.id]

def rows (recipe : ZeroRecipe) : List OwnedRow :=
  ownRows recipe.owner recipe.rawRows

@[simp] theorem row_count (recipe : ZeroRecipe) :
    recipe.rows.length = 3 := by
  simp [rows, rawRows]

@[simp] theorem temporary_count (recipe : ZeroRecipe) :
    [recipe.inverse, recipe.equal].length = 2 :=
  rfl

theorem rows_owned
    (recipe : ZeroRecipe)
    (row : OwnedRow)
    (member : row ∈ recipe.rows) :
    row.id.owner = recipe.owner :=
  ownRows_owner recipe.owner recipe.rawRows row member

theorem row_ids_nodup (recipe : ZeroRecipe) :
    (recipe.rows.map fun row => row.id).Nodup :=
  ownRows_ids_nodup recipe.owner recipe.rawRows

theorem active_sound
    (laws : FieldLaws)
    (recipe : ZeroRecipe)
    (assignment : ColumnId -> Field)
    (constantOne : assignment recipe.one = 1)
    (activeOne : assignment recipe.active = 1)
    (holds : Satisfies recipe.rows assignment) :
    assignment recipe.output.id =
      (if assignment recipe.input.id = 0 then 1 else 0) := by
  have raw :=
    (satisfies_ownRows_iff recipe.owner recipe.rawRows assignment).mp holds
  have inverseEquation :=
    (zeroInverseRow_iff assignment recipe.one recipe.input.id
      recipe.inverse.id recipe.equal.id constantOne).mp raw.1
  have annihilatorEquation :=
    (zeroAnnihilatorRow_iff assignment recipe.input.id
      recipe.equal.id).mp raw.2.1
  have output :=
    (zeroFinalRow_active_iff assignment recipe.active recipe.equal.id
      recipe.output.id activeOne).mp raw.2.2.1
  by_cases inputZero : assignment recipe.input.id = 0
  · have equalOne : assignment recipe.equal.id = 1 := by
      rw [inputZero, Fin.zero_mul] at inverseEquation
      exact
        (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp
          inverseEquation.symm).symm
    simp [inputZero, output, equalOne]
  · rcases laws.noZeroDivisors _ _ annihilatorEquation with
      impossible | equalZero
    · exact False.elim (inputZero impossible)
    · simp [inputZero, output, equalZero]

theorem complete
    (inverseLaw : InverseLaw)
    (recipe : ZeroRecipe)
    (assignment : ColumnId -> Field)
    (constantOne : assignment recipe.one = 1)
    (activeValue :
      assignment recipe.active = 1 ∨ assignment recipe.active = 0)
    (inverseValue :
      assignment recipe.inverse.id =
        coordinateInverseValue inverseLaw
          (assignment recipe.input.id) 0)
    (equalValue :
      assignment recipe.equal.id =
        coordinateEqualValue (assignment recipe.input.id) 0)
    (outputValue :
      assignment recipe.active = 1 ->
        assignment recipe.output.id =
          (if assignment recipe.input.id = 0 then 1 else 0)) :
    Satisfies recipe.rows assignment := by
  apply
    (satisfies_ownRows_iff recipe.owner recipe.rawRows assignment).mpr
  have witness :
      (zeroInverseRow recipe.one recipe.input.id recipe.inverse.id
          recipe.equal.id).Holds assignment ∧
        (zeroAnnihilatorRow recipe.input.id recipe.equal.id).Holds
          assignment := by
    by_cases inputZero : assignment recipe.input.id = 0
    · have inverseZero : assignment recipe.inverse.id = 0 := by
        rw [inverseValue]
        simp [coordinateInverseValue, inputZero]
      have equalOne : assignment recipe.equal.id = 1 := by
        rw [equalValue]
        simp [coordinateEqualValue, inputZero]
      constructor
      · apply
          (zeroInverseRow_iff assignment recipe.one recipe.input.id
            recipe.inverse.id recipe.equal.id constantOne).mpr
        rw [inputZero, inverseZero, equalOne, Fin.zero_mul]
        exact Lean.Grind.AddCommGroup.sub_self (1 : Field)
      · apply
          (zeroAnnihilatorRow_iff assignment recipe.input.id
            recipe.equal.id).mpr
        rw [inputZero, Fin.zero_mul]
    · have inputNe : assignment recipe.input.id ≠ 0 := inputZero
      have inverseExact :
          assignment recipe.inverse.id =
            inverseLaw.inverse (assignment recipe.input.id) := by
        rw [inverseValue]
        simp [coordinateInverseValue, inputZero, Fin.sub_eq_add_neg,
          Lean.Grind.AddCommGroup.neg_zero, Fin.add_zero]
      have equalZero : assignment recipe.equal.id = 0 := by
        rw [equalValue]
        simp [coordinateEqualValue, inputZero]
      constructor
      · apply
          (zeroInverseRow_iff assignment recipe.one recipe.input.id
            recipe.inverse.id recipe.equal.id constantOne).mpr
        rw [inverseExact, equalZero]
        simpa only [Fin.sub_eq_add_neg, Lean.Grind.AddCommGroup.neg_zero,
          Fin.add_zero] using
            inverseLaw.mul_inverse_of_ne_zero
              (assignment recipe.input.id) inputNe
      · apply
          (zeroAnnihilatorRow_iff assignment recipe.input.id
            recipe.equal.id).mpr
        rw [equalZero, Fin.mul_zero]
  refine ⟨witness.1, witness.2, ?_, trivial⟩
  rcases activeValue with activeOne | activeZero
  · apply
      (zeroFinalRow_active_iff assignment recipe.active recipe.equal.id
        recipe.output.id activeOne).mpr
    rw [equalValue]
    simpa [coordinateEqualValue] using outputValue activeOne
  · exact zeroFinalRow_inactive assignment recipe.active recipe.equal.id
      recipe.output.id activeZero

end ZeroRecipe

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
