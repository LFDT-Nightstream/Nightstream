import Nightstream.Implementation.R1CS.Core.EqualityPins

/-! Generated recursive claimed/recomputed output accumulator digest link. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorOutputLink

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "7ec8ee1beebaf1787443c269052f43b70fb23881f11e466f01a5771b0c3e1dfa"
def rowStart : Nat := 924687
def rowEnd : Nat := 924691
def rowCount : Nat := 4

def pairRuns : List EqualityPins.PairRun :=
  [ ⟨887605, 924511, 1, 1, 4⟩
  ]

def pairs : List (Nat × Nat) := EqualityPins.expandRuns pairRuns
def rows : List Row := EqualityPins.rows pairs

theorem pairs_length : pairs.length = rowCount := by
rw [pairs, EqualityPins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, EqualityPins.rows] using pairs_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorOutputLink
