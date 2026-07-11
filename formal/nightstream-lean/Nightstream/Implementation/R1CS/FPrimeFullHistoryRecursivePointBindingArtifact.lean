import Nightstream.Implementation.R1CS.EqualityPins

/-! Generated recursive NIFS PiCCS/PiDEC point binding. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePointBinding

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "5ca9aa691a070d733dd7393b4fd344400631ed8cfeaf4e600041a686f790c1a2"
def rowStart : Nat := 882154
def rowEnd : Nat := 882156
def rowCount : Nat := 2

def pairRuns : List EqualityPins.PairRun :=
  [ ⟨373514, 229377, 1, 1, 2⟩
  ]

def pairs : List (Nat × Nat) := EqualityPins.expandRuns pairRuns
def rows : List Row := EqualityPins.rows pairs

theorem pairs_length : pairs.length = rowCount := by
rw [pairs, EqualityPins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, EqualityPins.rows] using pairs_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePointBinding
