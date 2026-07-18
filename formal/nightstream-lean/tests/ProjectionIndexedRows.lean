import Nightstream.Implementation.R1CS.Correspondence.Projection.ArtifactProgram

/-! Focused fail-closed regressions for profile-neutral projection row matching. -/

namespace Nightstream.Tests.ProjectionIndexedRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionIndexedRows

def productDefinition : Definition where
  output := 3
  rhs := .product [(1, 2), (2, 3)] [(0, 1)]

def permutedProductRow : Row where
  a := [(2, 3), (1, 2)]
  b := [(0, 1)]
  c := [(3, 1)]

example : indexedRowsMatch
    [(7, permutedProductRow)] [(7, productDefinition)] = true := by
  decide

example : indexedRowsMatch
    [(8, permutedProductRow)] [(7, productDefinition)] = false := by
  decide

example : indexedRowsMatch
    [(7, permutedProductRow)] [] = false := by
  decide

example : indexedRowsMatch
    [] [(7, productDefinition)] = false := by
  decide

example :
    ¬ SourceRowsEmbedded [(3, permutedProductRow)] [permutedProductRow] := by
  intro embedded
  have found := embedded (3, permutedProductRow) (by simp)
  simp at found

end Nightstream.Tests.ProjectionIndexedRows
