import Nightstream.Implementation.R1CS.Core.EqualityPins

/-! Generated terminal NIFS PiCCS/PiDEC point binding. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPointBinding

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "fae4150101fde4ef8e5d31299148eaf3910fa7b831ca466c702c77a19da5ba50"
def rowStart : Nat := 3402926
def rowEnd : Nat := 3402928
def rowCount : Nat := 2

def pairRuns : List EqualityPins.PairRun :=
  [ ⟨2676644, 1652431, 1, 1, 2⟩
  ]

def pairs : List (Nat × Nat) := EqualityPins.expandRuns pairRuns
def rows : List Row := EqualityPins.rows pairs

theorem pairs_length : pairs.length = rowCount := by
rw [pairs, EqualityPins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, EqualityPins.rows] using pairs_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPointBinding
