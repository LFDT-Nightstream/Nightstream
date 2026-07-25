import Nightstream.Implementation.R1CS.Core.AffinePins

/-! Generated exact affine-pin phase. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkPlacement

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "eb24e545d0ad36ef8215dd25a32b30dd13da0d8c27643ab25eeef94f150408d2"
def rowStart : Nat := 9673389
def rowEnd : Nat := 9673659
def rowCount : Nat := 270

def pinRuns : List AffinePins.Run :=
  [ .constant 4090877 0 1 0 1
  , .equal 4090878 16766 1 1 256
  , .zero 4091134 1 13
  ]

def pins : List AffinePins.Pin := AffinePins.expandRuns pinRuns
def rows : List Row := AffinePins.rows pins

theorem pins_canonical : AffinePins.PinsCanonical pins := by native_decide
theorem pins_length : pins.length = rowCount := by
rw [pins, AffinePins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, AffinePins.rows] using pins_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkPlacement
