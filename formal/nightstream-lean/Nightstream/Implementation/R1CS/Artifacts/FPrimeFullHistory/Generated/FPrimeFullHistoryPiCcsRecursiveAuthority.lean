import Nightstream.Implementation.R1CS.Core.AffinePins

/-! Generated exact affine-pin phase. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsRecursiveAuthority

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "ff7a9f47535cb5de6f6af5ffb383f27bc67ad0e281a769887a975ac5c7306a8f"
def rowStart : Nat := 30768
def rowEnd : Nat := 30834
def rowCount : Nat := 66

def pinRuns : List AffinePins.Run :=
  [ .equal 33175 32791 1 1 2
  , .equal 33177 32919 1 1 2
  , .equal 33179 33047 1 1 2
  , .zero 32899 1 20
  , .zero 33027 1 20
  , .zero 33155 1 20
  ]

def pins : List AffinePins.Pin := AffinePins.expandRuns pinRuns
def rows : List Row := AffinePins.rows pins

theorem pins_canonical : AffinePins.PinsCanonical pins := by native_decide
theorem pins_length : pins.length = rowCount := by
rw [pins, AffinePins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, AffinePins.rows] using pins_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsRecursiveAuthority
