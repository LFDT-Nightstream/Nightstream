import Nightstream.Implementation.R1CS.Core.AffinePins
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiCcsTerminalOutputBindingRuns0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiCcsTerminalOutputBindingRuns1

/-! Generated exact affine-pin phase. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsTerminalOutputBinding

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "2d62a36a1ee0d1d78173bcd43045e4acc1df4c6f27748f8bd9a95bd9ba5e8aa9"
def rowStart : Nat := 1424530
def rowEnd : Nat := 1634049
def rowCount : Nat := 209519

def pinRuns : List AffinePins.Run :=
    FPrimeFullHistoryPiCcsTerminalOutputBindingRuns0.pinRuns ++
    FPrimeFullHistoryPiCcsTerminalOutputBindingRuns1.pinRuns

def pins : List AffinePins.Pin := AffinePins.expandRuns pinRuns
def rows : List Row := AffinePins.rows pins

theorem pins_canonical : AffinePins.PinsCanonical pins := by native_decide
theorem pins_length : pins.length = rowCount := by
rw [pins, AffinePins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, AffinePins.rows] using pins_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsTerminalOutputBinding
