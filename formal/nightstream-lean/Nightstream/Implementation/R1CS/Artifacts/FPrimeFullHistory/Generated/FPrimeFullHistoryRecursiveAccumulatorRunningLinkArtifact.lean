import Nightstream.Implementation.R1CS.Core.EqualityPins

/-! Generated recursive running-accumulator input digest link. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorRunningLink

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "764e43cc93af3fdf6e8682cb66d6e18daf927b7559a9049aab98e6a2075b7501"
def rowStart : Nat := 863857
def rowEnd : Nat := 863861
def rowCount : Nat := 4

def pairRuns : List EqualityPins.PairRun :=
  [ ⟨213659, 10856, 1, 1, 4⟩
  ]

def pairs : List (Nat × Nat) := EqualityPins.expandRuns pairRuns
def rows : List Row := EqualityPins.rows pairs

theorem pairs_length : pairs.length = rowCount := by
rw [pairs, EqualityPins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, EqualityPins.rows] using pairs_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorRunningLink
