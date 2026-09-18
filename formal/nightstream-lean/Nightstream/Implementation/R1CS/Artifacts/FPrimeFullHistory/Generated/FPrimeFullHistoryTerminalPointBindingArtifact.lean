import Nightstream.Implementation.R1CS.Core.EqualityPins

/-! Generated terminal NIFS PiCCS/PiDEC point binding. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPointBinding

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "3895dd2602d9c6a836baa4667c07a884434bcda659348029431bd4f4dbe0ff28"
def rowStart : Nat := 3502230
def rowEnd : Nat := 3502232
def rowCount : Nat := 2

def pairRuns : List EqualityPins.PairRun :=
  [ ⟨2611124, 1424315, 1, 1, 2⟩
  ]

def pairs : List (Nat × Nat) := EqualityPins.expandRuns pairRuns
def rows : List Row := EqualityPins.rows pairs

theorem pairs_length : pairs.length = rowCount := by
rw [pairs, EqualityPins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, EqualityPins.rows] using pairs_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPointBinding
