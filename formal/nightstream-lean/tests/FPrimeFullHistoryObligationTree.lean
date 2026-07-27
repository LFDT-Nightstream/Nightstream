import Nightstream.Assurance.FPrimeFullHistoryObligationTree

/-! Focused regressions for exact full-history obligation ownership. -/

namespace NightstreamTests.FPrimeFullHistoryObligationTree

open Nightstream.Assurance.FPrimeFullHistoryObligationTree
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursiveManifest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryManifest

def range (name : String) (rowStart rowEnd : Nat) : RowRange where
  name := name
  rowStart := rowStart
  rowEnd := rowEnd
  nonzeroEntries := 0
  sha256 := ""

def validSmall : List RowRange :=
  [range "left" 0 1, range "right" 1 3]

def gap : List RowRange :=
  [range "left" 0 1, range "right" 2 3]

def overlap : List RowRange :=
  [range "left" 0 2, range "right" 1 3]

def duplicate : List RowRange :=
  [range "same" 0 1, range "same" 0 1]

def reversed : List RowRange :=
  [range "bad" 0 2, range "backwards" 2 1]

example : covers 0 3 validSmall = true := by decide
example : covers 0 3 gap = false := by decide
example : covers 0 3 overlap = false := by decide
example : covers 0 1 duplicate = false := by decide
example : covers 0 1 reversed = false := by decide

example :
    ∃ owner, (owner ∈ validSmall ∧ OwnsRow owner 2) ∧
      ∀ other, other ∈ validSmall ∧ OwnsRow other 2 → other = owner :=
  covers_has_exact_owner (start := 0) (finish := 3)
    (ranges := validSmall) (row := 2) (by decide) (by omega) (by omega)

example : ¬OwnsRow (range "zero" 7 7) 7 := by
  simp [OwnsRow, range]

example :
    ¬({ parent := range "root" 0 3, children := gap } : Branch).Exact := by
  decide

example :
    ¬({ parent := range "root" 0 3, children := overlap } : Branch).Exact := by
  decide

example : allLeaves.length = 61 := by exact exact_leaf_census.1
example : materializedLeaves.length = 59 := by exact exact_leaf_census.2.1
example : zeroCostLeaves.length = 2 := by exact exact_leaf_census.2.2

example : formulaOnlyEstimates.length = 1 := by decide
example : materializedLeafRanges.Nodup := materialized_leaf_ranges_nodup
example : covers 0 totalRows materializedLeafRanges = true :=
  materialized_leaf_ranges_cover

example :
    FPrimeFullHistoryTerminalLink.rows ≠ FPrimeTerminalLink.rows :=
  obligation_tree_retains_terminal_drift

end NightstreamTests.FPrimeFullHistoryObligationTree
