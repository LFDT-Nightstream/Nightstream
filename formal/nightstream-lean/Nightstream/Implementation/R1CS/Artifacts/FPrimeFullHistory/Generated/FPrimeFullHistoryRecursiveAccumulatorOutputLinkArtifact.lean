import Nightstream.Implementation.R1CS.Core.EqualityPins

/-! Generated recursive claimed/recomputed output accumulator digest link. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorOutputLink

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "380b6c9e49cdb01db1066a1dbd4f5042f9abe5729cb929db0ecaafc61e2b7b35"
def rowStart : Nat := 1118772
def rowEnd : Nat := 1118776
def rowCount : Nat := 4

def pairRuns : List EqualityPins.PairRun :=
  [ ⟨872562, 1127468, 1, 1, 4⟩
  ]

def pairs : List (Nat × Nat) := EqualityPins.expandRuns pairRuns
def rows : List Row := EqualityPins.rows pairs

theorem pairs_length : pairs.length = rowCount := by
rw [pairs, EqualityPins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, EqualityPins.rows] using pairs_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorOutputLink
