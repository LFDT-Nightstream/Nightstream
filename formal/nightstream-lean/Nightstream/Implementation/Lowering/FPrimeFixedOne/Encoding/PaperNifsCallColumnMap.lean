import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence
import Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
import Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge

/-!
Contract: locate typed base- and quadratic-extension projections inside one
shared numeric column map.

Canonical NIFS gadgets use natural-number column indices.  `CallRecipe` uses
stable `ColumnId`s.  `FLocation` and `KLocation` record the numeric indices
assigned to already-existing typed coordinates and prove that all are
translated by the same global map.  Neither allocates a second copy.

This is intentionally weaker than an independent placement abstraction:
every component of the eventual `nifsVerify` program must use one common
`columnMap`, so producer and consumer reads are definitionally shared.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
open Nightstream.Implementation.R1CS.ProjectionProgram

private theorem concreteK_eq
    (left right : Nightstream.SuperNeo.Concrete.K)
    (c0Equal : left.c0 = right.c0)
    (c1Equal : left.c1 = right.c1) :
    left = right := by
  cases left
  cases right
  simp only at c0Equal c1Equal
  cases c0Equal
  cases c1Equal
  rfl

/-- One physical base-field value located in a shared numeric column
namespace. -/
structure FLocation
    (columnMap : Nat → ColumnId)
    (typed : FColumnId) where
  numeric : Nat
  mapped : columnMap numeric = typed.column

/-- One physical quadratic-extension value located in a shared numeric
column namespace. -/
structure KLocation
    (columnMap : Nat → ColumnId)
    (typed : KColumnIds) where
  numeric : KColumns
  c0Mapped : columnMap numeric.c0 = typed.c0
  c1Mapped : columnMap numeric.c1 = typed.c1

/-- The singleton combination consumed by canonical base-field row gadgets. -/
def FLocation.carried
    {columnMap : Nat → ColumnId}
    {typed : FColumnId}
    (location : FLocation columnMap typed) :
    List (Nat × Nat) :=
  [(location.numeric, 1)]

/-- The singleton combinations consumed by canonical `K` row gadgets. -/
def KLocation.carried
    {columnMap : Nat → ColumnId}
    {typed : KColumnIds}
    (location : KLocation columnMap typed) : KMul.Carried where
  low := [(location.numeric.c0, 1)]
  high := [(location.numeric.c1, 1)]

/-- Pulling a typed assignment through the shared map reads exactly the
located physical pair. -/
theorem KLocation.numeric_value_eq
    {columnMap : Nat → ColumnId}
    {typed : KColumnIds}
    (location : KLocation columnMap typed)
    (assignment : ColumnId → Field) :
    ofProjection
        (location.numeric.value
          (numericAssignment columnMap assignment)) =
      typed.value assignment := by
  apply concreteK_eq
  · apply Fin.ext
    simp only [KColumns.value, baseAt, ProjectionProgram.residue,
      numericAssignment, location.c0Mapped, KColumnIds.value, ofProjection]
    exact Nat.mod_eq_of_lt (by
      simpa [Nightstream.Implementation.R1CS.goldilocksP,
        Nightstream.SuperNeo.Concrete.goldilocksModulus] using
          (assignment typed.c0).isLt)
  · apply Fin.ext
    simp only [KColumns.value, baseAt, ProjectionProgram.residue,
      numericAssignment, location.c1Mapped, KColumnIds.value, ofProjection]
    exact Nat.mod_eq_of_lt (by
      simpa [Nightstream.Implementation.R1CS.goldilocksP,
        Nightstream.SuperNeo.Concrete.goldilocksModulus] using
          (assignment typed.c1).isLt)

/-- Pulling a typed assignment through the shared map reads exactly the
located base-field value. -/
theorem FLocation.numeric_value_eq
    {columnMap : Nat → ColumnId}
    {typed : FColumnId}
    (location : FLocation columnMap typed)
    (assignment : ColumnId → Field) :
    NumericRowBridge.residue
        (numericAssignment columnMap assignment location.numeric) =
      typed.value assignment := by
  apply Fin.ext
  simp only [NumericRowBridge.residue, numericAssignment, location.mapped,
    FColumnId.value]
  exact Nat.mod_eq_of_lt (assignment typed.column).isLt

/-- Evaluating the singleton carried expression reaches the exact typed
physical base-field value. -/
theorem FLocation.carried_value_eq
    {columnMap : Nat → ColumnId}
    {typed : FColumnId}
    (location : FLocation columnMap typed)
    (assignment : ColumnId → Field) :
    NumericRowBridge.residue
        (Nightstream.Implementation.R1CS.lcEval
          (numericAssignment columnMap assignment) location.carried) =
      typed.value assignment := by
  rw [← location.numeric_value_eq assignment]
  apply Fin.ext
  simp [FLocation.carried, Nightstream.Implementation.R1CS.lcEval,
    Nightstream.Implementation.R1CS.Program.rawLcEval,
    NumericRowBridge.residue, numericAssignment,
    Nightstream.Implementation.R1CS.goldilocksP,
    Nightstream.SuperNeo.Concrete.goldilocksModulus]

/-- Decoding the singleton carried expression reaches the exact typed
physical value. -/
theorem KLocation.decodeCarried_eq
    {columnMap : Nat → ColumnId}
    {typed : KColumnIds}
    (location : KLocation columnMap typed)
    (assignment : ColumnId → Field) :
    ofProjection
        (KFixedPhaseSumCheck.decodeCarried
          (numericAssignment columnMap assignment) location.carried) =
      typed.value assignment := by
  change
    ofProjection
        (KFixedPhaseSumCheck.decodeCarried
          (numericAssignment columnMap assignment)
          (KFixedPhaseSemanticOccurrence.carried location.numeric)) =
      typed.value assignment
  rw [KFixedPhaseSemanticOccurrence.decodeCarried_carried]
  exact location.numeric_value_eq assignment

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
