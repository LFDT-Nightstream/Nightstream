import Nightstream.Implementation.R1CS.Core.AffinePins

/-! Generated exact affine-pin phase. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsRecursiveOutputBinding

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "9378696a0ee32f8655d29535c12ec498dc97de00cdf2dc5b385a35d385be1590"
def rowStart : Nat := 267449
def rowEnd : Nat := 268704
def rowCount : Nat := 1255

def pinRuns : List AffinePins.Run :=
  [ .equal 32771 226962 1 1 2
  , .equal 32773 241731 1 1 2
  , .equal 32775 243561 1 1 2
  , .equal 32777 245391 1 1 2
  , .equal 32779 247221 1 1 2
  , .equal 32781 249051 1 1 2
  , .equal 32783 250881 1 1 2
  , .equal 32785 252711 1 1 2
  , .equal 32787 254541 1 1 2
  , .equal 32789 256371 1 1 2
  , .zero 31524 0 1
  , .equal 31795 30292 1 1 2
  , .constant 32769 0 54 0 1
  , .equal 32770 31523 539 0 2
  , .equal 31797 30294 1 1 972
  , .equal 31525 31266 5 1 54
  , .equal 31526 31320 5 1 54
  , .equal 31527 31374 5 1 54
  , .equal 31528 31428 5 1 54
  , .equal 31529 31482 5 1 41
  ]

def pins : List AffinePins.Pin := AffinePins.expandRuns pinRuns
def rows : List Row := AffinePins.rows pins

theorem pins_canonical : AffinePins.PinsCanonical pins := by native_decide
theorem pins_length : pins.length = rowCount := by
rw [pins, AffinePins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, AffinePins.rows] using pins_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsRecursiveOutputBinding
