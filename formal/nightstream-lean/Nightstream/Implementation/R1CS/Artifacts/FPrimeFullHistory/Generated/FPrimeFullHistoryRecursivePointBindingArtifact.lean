import Nightstream.Implementation.R1CS.Core.EqualityPins

/-! Generated recursive NIFS PiCCS/PiDEC point binding. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePointBinding

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "8ab83fb7ebcf3e552f9bde0a8f8c6ec44b29b8c15b6893c5107269b3fb487274"
def rowStart : Nat := 858623
def rowEnd : Nat := 858625
def rowCount : Nat := 2

def pairRuns : List EqualityPins.PairRun :=
  [ ⟨360263, 226962, 1, 1, 2⟩
  ]

def pairs : List (Nat × Nat) := EqualityPins.expandRuns pairRuns
def rows : List Row := EqualityPins.rows pairs

theorem pairs_length : pairs.length = rowCount := by
rw [pairs, EqualityPins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, EqualityPins.rows] using pairs_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePointBinding
