import Nightstream.Implementation.R1CS.AffinePins

/-! Generated exact affine-pin phase. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsRecursiveAllocation

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "31d0a148cb926037e412347bec06b6053142e68849a178d513b1425d1287036d"
def rowStart : Nat := 30759
def rowEnd : Nat := 30768
def rowCount : Nat := 9

def pinRuns : List AffinePins.Run :=
  [ .constant 30292 1 54 0 1
  , .constant 30293 1230 18 239 2
  , .zero 31524 0 1
  , .constant 31795 1 54 0 1
  , .constant 31796 973 18 36 2
  , .constant 32770 539 257 0 2
  ]

def pins : List AffinePins.Pin := AffinePins.expandRuns pinRuns
def rows : List Row := AffinePins.rows pins

theorem pins_canonical : AffinePins.PinsCanonical pins := by native_decide
theorem pins_length : pins.length = rowCount := by
rw [pins, AffinePins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, AffinePins.rows] using pins_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsRecursiveAllocation
