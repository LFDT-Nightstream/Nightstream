import Nightstream.Implementation.R1CS.AffinePins

/-! Generated exact affine-pin phase. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcRecursiveLinearFolds

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "d724de012fe58e8ddedcd2df33fbf1bd7e9441db1e243f064abb64d512548a2b"
def rowStart : Nat := 385106
def rowEnd : Nat := 385128
def rowCount : Nat := 22

def pinRuns : List AffinePins.Run :=
  [ .equal 32773 373516 1 1 18
  , .equal 373662 33310 1 1 4
  ]

def pins : List AffinePins.Pin := AffinePins.expandRuns pinRuns
def rows : List Row := AffinePins.rows pins

theorem pins_canonical : AffinePins.PinsCanonical pins := by native_decide
theorem pins_length : pins.length = rowCount := by
rw [pins, AffinePins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, AffinePins.rows] using pins_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcRecursiveLinearFolds
