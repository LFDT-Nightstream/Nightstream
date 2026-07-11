import Nightstream.Implementation.R1CS.AffinePins

/-! Generated exact affine-pin phase. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcTerminalLinearFolds

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "352d7836d9f04674295b3df4b3333eaed09921802809bf589af829be3f0bf0e7"
def rowStart : Nat := 2953186
def rowEnd : Nat := 2953516
def rowCount : Nat := 330

def pinRuns : List AffinePins.Run :=
  [ .equal 958682 2611126 1 1 18
  , .equal 960472 2611126 1 1 18
  , .equal 962262 2611126 1 1 18
  , .equal 964052 2611126 1 1 18
  , .equal 965842 2611126 1 1 18
  , .equal 967632 2611126 1 1 18
  , .equal 969422 2611126 1 1 18
  , .equal 971212 2611126 1 1 18
  , .equal 973002 2611126 1 1 18
  , .equal 974792 2611126 1 1 18
  , .equal 976582 2611126 1 1 18
  , .equal 978372 2611126 1 1 18
  , .equal 980162 2611126 1 1 18
  , .equal 981952 2611126 1 1 18
  , .equal 983742 2611126 1 1 18
  , .equal 2611272 959219 1 1 4
  , .equal 2611272 961009 1 1 4
  , .equal 2611272 962799 1 1 4
  , .equal 2611272 964589 1 1 4
  , .equal 2611272 966379 1 1 4
  , .equal 2611272 968169 1 1 4
  , .equal 2611272 969959 1 1 4
  , .equal 2611272 971749 1 1 4
  , .equal 2611272 973539 1 1 4
  , .equal 2611272 975329 1 1 4
  , .equal 2611272 977119 1 1 4
  , .equal 2611272 978909 1 1 4
  , .equal 2611272 980699 1 1 4
  , .equal 2611272 982489 1 1 4
  , .equal 2611272 984279 1 1 4
  ]

def pins : List AffinePins.Pin := AffinePins.expandRuns pinRuns
def rows : List Row := AffinePins.rows pins

theorem pins_canonical : AffinePins.PinsCanonical pins := by native_decide
theorem pins_length : pins.length = rowCount := by
rw [pins, AffinePins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, AffinePins.rows] using pins_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcTerminalLinearFolds
