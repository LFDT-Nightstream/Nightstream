import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityCompilerDescription

/-!
Focused theorem-surface checks for the exact
`nightstream-sparse-structure-v1` encoding.
-/

set_option autoImplicit false

namespace tests.PaddedRowIdentityCompilerDescription

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityCompilerDescription

#check polynomialTerms_perm
#check polynomialTerms_strictLex
#check polynomialTerms_constraints_exact
#check structureFields_injective
#check fields_injective
#check fields_prefixFree
#check codec_canonical
#check matrices_eq_of_structureFields_eq
#check matrices_eq_of_fields_eq

def emptyMatrixDescription : MatrixDescription where
  entries := []
  canonical := by
    simp [CanonicalMatrixEntries]
    decide

def emptyDescription : Description where
  sections := fun _ => emptyMatrixDescription
  streamFits := by native_decide

example : structureHeader.length = 10 := structureHeader_length

example : polynomialTerms.length = 66 := polynomialTerms_count_exact

example : polynomialFields.length = 925 := polynomialFields_length

example : emptyDescription.entryCount = 0 := by native_decide

example : (structureFields emptyDescription).length = 961 := by
  rw [structureFields_length]
  native_decide

example : (fields emptyDescription).length = 962 := by
  rw [fields_length]
  native_decide

example (index : Nat) :
    (matrixFields index emptyMatrixDescription).length = 2 := by
  rw [matrixFields_length]
  rfl

end tests.PaddedRowIdentityCompilerDescription
