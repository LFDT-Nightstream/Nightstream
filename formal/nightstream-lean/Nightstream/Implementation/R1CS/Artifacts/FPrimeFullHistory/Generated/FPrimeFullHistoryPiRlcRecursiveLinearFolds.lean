import Nightstream.Implementation.R1CS.Core.AffinePins

/-! Generated exact affine-pin phase. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcRecursiveLinearFolds

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "bc4baa99637163b4a8af9040b205bd2f1e4de847ffb58593fb2263b55f37f32d"
def rowStart : Nat := 361575
def rowEnd : Nat := 361597
def rowCount : Nat := 22

def pinRuns : List AffinePins.Run :=
  [ .equal 32773 360265 1 1 18
  , .equal 33310 360411 1 1 4
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
