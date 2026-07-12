import Nightstream.Implementation.R1CS.Correspondence.CanonicalU64.CanonicalU64Complete

/-!
Focused regression for source/interpreter completeness of the generated
canonical-u64 block.  The field runtime remains universally quantified; the
test does not assume a particular accepting assignment.
-/

namespace NightstreamTests.CanonicalU64Complete

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CanonicalU64
open Nightstream.Implementation.R1CS.CanonicalU64Complete

def zeroSource : Source where
  bit := fun _ => false

def localColumnMap : List Nat := List.range 68

example : wordValue zeroSource = 0 := by native_decide

example (field : FieldInverse) :
    Satisfies rows (interpret field zeroSource) := by
  apply complete field zeroSource
  native_decide

example (field : FieldInverse) :
    ∀ column, interpret field zeroSource column < goldilocksP := by
  apply interpret_canonical field zeroSource
  native_decide

/-- Regression for the finite-map fallback used by production relabeling. -/
example (field : FieldInverse) (source : Source) :
    Relabel.assignment localColumnMap (interpret field source) =
      interpret field source := by
  funext column
  by_cases bounded : column < 68
  · have getDValue : localColumnMap.getD column 0 = column := by
      simp [localColumnMap, List.getD, bounded]
    change interpret field source (localColumnMap.getD column 0) =
      interpret field source column
    rw [getDValue]
  · have getDValue : localColumnMap.getD column 0 = 0 := by
      simp [localColumnMap, List.getD, bounded]
    change interpret field source (localColumnMap.getD column 0) =
      interpret field source column
    rw [getDValue]
    have outside : 67 < column := by omega
    have notZero : column ≠ 0 := by omega
    have notVar : column ≠ varCol := by simp [varCol]; omega
    have notBits : ¬ (2 ≤ column ∧ column < 66) := by omega
    have notFlag : column ≠ 66 := by omega
    have notInverse : column ≠ 67 := by omega
    simp [interpret, notZero, notVar, notBits, notFlag, notInverse]

example (columnMap : List Nat) (field : FieldInverse)
    (assignment : Nat → Nat)
    (witness : ExecutionWitness field
      (Relabel.assignment columnMap assignment)) :
    Satisfies (rows.map (Relabel.row columnMap)) assignment := by
  exact mapped_complete columnMap witness

end NightstreamTests.CanonicalU64Complete
