import Nightstream.Implementation.R1CS.EqualityPins

/-! Generated terminal delayed public-link rows. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLink

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "04ffa425a0e31c1fbf9fd722a82712b7e836f62397ea98b11cd5dfc63793ab5e"
def rowStart : Nat := 3517633
def rowEnd : Nat := 3517890
def rowCount : Nat := 257

def pairRuns : List EqualityPins.PairRun :=
  [ ⟨930325, 0, 0, 0, 1⟩
  , ⟨930326, 16766, 1, 1, 256⟩
  ]

def pairs : List (Nat × Nat) := EqualityPins.expandRuns pairRuns
def rows : List Row := EqualityPins.rows pairs

theorem pairs_length : pairs.length = rowCount := by
rw [pairs, EqualityPins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, EqualityPins.rows] using pairs_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLink
