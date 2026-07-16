import Nightstream.Implementation.R1CS.Core.AffinePins
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiCcsTerminalOutputBindingRuns0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiCcsTerminalOutputBindingRuns1

/-! Generated exact affine-pin phase. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsTerminalOutputBinding

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "58c8dfc0a4b0ca75e54eaa3c01d605455c4fdece73f1aaf55be4f5b99f4a29ad"
def rowStart : Nat := 1642352
def rowEnd : Nat := 1851871
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
