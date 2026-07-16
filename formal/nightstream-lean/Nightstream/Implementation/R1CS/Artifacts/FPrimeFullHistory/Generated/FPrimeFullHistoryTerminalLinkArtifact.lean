import Nightstream.Implementation.R1CS.Core.EqualityPins

/-! Generated terminal delayed public-link rows. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLink

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "7f2b66ba038e073025694908b8c8172401eddc38f080df4863d38e61da1ad81b"
def rowStart : Nat := 3418329
def rowEnd : Nat := 3418586
def rowCount : Nat := 257

def pairRuns : List EqualityPins.PairRun :=
  [ ⟨1133282, 0, 0, 0, 1⟩
  , ⟨1133283, 16766, 1, 1, 256⟩
  ]

def pairs : List (Nat × Nat) := EqualityPins.expandRuns pairRuns
def rows : List Row := EqualityPins.rows pairs

theorem pairs_length : pairs.length = rowCount := by
rw [pairs, EqualityPins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, EqualityPins.rows] using pairs_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLink
