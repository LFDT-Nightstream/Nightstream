import Nightstream.Implementation.Lowering.Goldilocks.TypedNumericRowBridge

/-!
Focused checks for exact typed-to-numeric Goldilocks row lowering.

The countermodel shows why artifact completeness requires an injective column
allocation. Two distinct typed columns that share one numeric index cannot
carry different field values in any numeric assignment.
-/

set_option autoImplicit false

namespace NightstreamTests.TypedNumericRowBridge

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.TypedNumericRowBridge
open Nightstream.SuperNeo.Concrete

#check residue_lcEval
#check numericRow_holds_iff
#check rows_satisfied_iff
#check typedAssignment_lift
#check exists_numeric_assignment_of_satisfies
#check SplitEmbedding.satisfies
#check SplitEmbedding.complete

private def firstColumn : ColumnId where
  owner := .prelude
  bundleIndex := 0
  coordinateIndex := 0

private def secondColumn : ColumnId where
  owner := .prelude
  bundleIndex := 1
  coordinateIndex := 0

private def collidingIndex (_ : ColumnId) : Nat := 0

private def distinguishingAssignment (column : ColumnId) : F :=
  if column = firstColumn then 0 else 1

theorem firstColumn_ne_secondColumn : firstColumn ≠ secondColumn := by
  decide

/-- A non-injective allocation cannot lift every typed assignment. This is a
deterministic completeness failure, not a cryptographic event. -/
theorem noninjective_allocation_has_no_exact_lift :
    forall numericAssignment : Nat -> Nat,
      typedAssignment collidingIndex numericAssignment ≠
        distinguishingAssignment := by
  intro numericAssignment equal
  have firstEqual := congrFun equal firstColumn
  have secondEqual := congrFun equal secondColumn
  have pulledEqual :
      typedAssignment collidingIndex numericAssignment firstColumn =
        typedAssignment collidingIndex numericAssignment secondColumn := rfl
  have secondColumn_ne_firstColumn : secondColumn ≠ firstColumn :=
    fun columnsEqual => firstColumn_ne_secondColumn columnsEqual.symm
  have falseEquality : (0 : F) = 1 := by
    calc
      0 = distinguishingAssignment firstColumn := by
        simp [distinguishingAssignment]
      _ = typedAssignment collidingIndex numericAssignment firstColumn :=
        firstEqual.symm
      _ = typedAssignment collidingIndex numericAssignment secondColumn :=
        pulledEqual
      _ = distinguishingAssignment secondColumn := secondEqual
      _ = 1 := by
        simp [distinguishingAssignment, secondColumn_ne_firstColumn]
  exact Fin.zero_ne_one falseEquality

theorem collidingIndex_not_injective :
    ¬ Function.Injective collidingIndex := by
  intro injective
  exact firstColumn_ne_secondColumn (injective rfl)

end NightstreamTests.TypedNumericRowBridge
