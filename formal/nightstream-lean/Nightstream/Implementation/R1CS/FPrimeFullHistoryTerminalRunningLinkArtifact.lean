import Nightstream.Implementation.R1CS.EqualityPins

/-! Generated terminal running-accumulator digest continuity. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLink

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "5a9524a717542f8528db0a4e4caa8e04e4da897b92cc113dd190b4c98f30d25a"
def rowStart : Nat := 3502232
def rowEnd : Nat := 3502236
def rowCount : Nat := 4

def pairRuns : List EqualityPins.PairRun :=
  [ ⟨1396970, 924511, 1, 1, 4⟩
  ]

def pairs : List (Nat × Nat) := EqualityPins.expandRuns pairRuns
def rows : List Row := EqualityPins.rows pairs

theorem pairs_length : pairs.length = rowCount := by
rw [pairs, EqualityPins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, EqualityPins.rows] using pairs_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLink
