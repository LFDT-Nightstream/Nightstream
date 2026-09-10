import NightstreamFPrime.Layout.R1CS

/-! Exact R1CS column renaming and its linear-combination laws. -/

namespace NightstreamFPrime.Layout.R1CS
open NightstreamFPrime.Spec
open NightstreamFPrime.Layout

/-- Exact column renaming for one R1CS linear combination. This small common
definition lets the three package instantiation forms share one custody
proof without exposing the private summation implementation. -/
def mapCombinationColumns (column : Nat → Nat)
    (combination : R1CS.LinearCombination) : R1CS.LinearCombination :=
  ⟨combination.constant,
    combination.terms.map fun term => (column term.1, term.2)⟩

def mapRowColumns (column : Nat → Nat) (row : R1CS.Row) : R1CS.Row :=
  ⟨mapCombinationColumns column row.a,
    mapCombinationColumns column row.b,
    mapCombinationColumns column row.c⟩

@[simp] theorem mapCombinationColumns_zero (column : Nat → Nat) :
    mapCombinationColumns column R1CS.LinearCombination.zero =
      R1CS.LinearCombination.zero := by
  rfl

@[simp] theorem mapCombinationColumns_const (column : Nat → Nat) (value : F) :
    mapCombinationColumns column (R1CS.LinearCombination.const value) =
      R1CS.LinearCombination.const value := by
  rfl

@[simp] theorem mapCombinationColumns_one (column : Nat → Nat) :
    mapCombinationColumns column R1CS.LinearCombination.one =
      R1CS.LinearCombination.one := by
  rfl

@[simp] theorem mapCombinationColumns_ofVar (column : Nat → Nat)
    (index : Nat) :
    mapCombinationColumns column (R1CS.LinearCombination.ofVar index) =
      R1CS.LinearCombination.ofVar (column index) := by
  rfl

@[simp] theorem mapCombinationColumns_add (column : Nat → Nat)
    (left right : R1CS.LinearCombination) :
    mapCombinationColumns column (R1CS.LinearCombination.add left right) =
      R1CS.LinearCombination.add (mapCombinationColumns column left)
        (mapCombinationColumns column right) := by
  cases left
  cases right
  simp [mapCombinationColumns, R1CS.LinearCombination.add, List.map_append]

@[simp] theorem mapCombinationColumns_scale (column : Nat → Nat)
    (coefficient : F) (combination : R1CS.LinearCombination) :
    mapCombinationColumns column
        (R1CS.LinearCombination.scale coefficient combination) =
      R1CS.LinearCombination.scale coefficient
        (mapCombinationColumns column combination) := by
  cases combination
  simp [mapCombinationColumns, R1CS.LinearCombination.scale, List.map_map,
    Function.comp_def]

end NightstreamFPrime.Layout.R1CS
