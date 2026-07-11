import Nightstream.Implementation.R1CS.EqualityPins

/-! Generated recursive running-accumulator input digest link. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorRunningLink

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "c96b1cd632f868b145f7674271366be660f05b7f814186739433af8c18c2fcd2"
def rowStart : Nat := 887388
def rowEnd : Nat := 887392
def rowCount : Nat := 4

def pairRuns : List EqualityPins.PairRun :=
  [ ⟨216073, 10856, 1, 1, 4⟩
  ]

def pairs : List (Nat × Nat) := EqualityPins.expandRuns pairRuns
def rows : List Row := EqualityPins.rows pairs

theorem pairs_length : pairs.length = rowCount := by
rw [pairs, EqualityPins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, EqualityPins.rows] using pairs_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorRunningLink
