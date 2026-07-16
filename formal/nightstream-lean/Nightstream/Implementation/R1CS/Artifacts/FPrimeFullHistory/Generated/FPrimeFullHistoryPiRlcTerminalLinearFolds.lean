import Nightstream.Implementation.R1CS.Core.AffinePins

/-! Generated exact affine-pin phase. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcTerminalLinearFolds

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "c3eeaa987a199e94e0a5a0749b700ad6c9aaae48b2ee3acf09500538e61878c0"
def rowStart : Nat := 2853882
def rowEnd : Nat := 2854212
def rowCount : Nat := 330

def pinRuns : List AffinePins.Run :=
  [ .equal 1159847 2676646 1 1 18
  , .equal 1161637 2676646 1 1 18
  , .equal 1163427 2676646 1 1 18
  , .equal 1165217 2676646 1 1 18
  , .equal 1167007 2676646 1 1 18
  , .equal 1168797 2676646 1 1 18
  , .equal 1170587 2676646 1 1 18
  , .equal 1172377 2676646 1 1 18
  , .equal 1174167 2676646 1 1 18
  , .equal 1175957 2676646 1 1 18
  , .equal 1177747 2676646 1 1 18
  , .equal 1179537 2676646 1 1 18
  , .equal 1181327 2676646 1 1 18
  , .equal 1183117 2676646 1 1 18
  , .equal 1184907 2676646 1 1 18
  , .equal 1160384 2676792 1 1 4
  , .equal 1162174 2676792 1 1 4
  , .equal 1163964 2676792 1 1 4
  , .equal 1165754 2676792 1 1 4
  , .equal 1167544 2676792 1 1 4
  , .equal 1169334 2676792 1 1 4
  , .equal 1171124 2676792 1 1 4
  , .equal 1172914 2676792 1 1 4
  , .equal 1174704 2676792 1 1 4
  , .equal 1176494 2676792 1 1 4
  , .equal 1178284 2676792 1 1 4
  , .equal 1180074 2676792 1 1 4
  , .equal 1181864 2676792 1 1 4
  , .equal 1183654 2676792 1 1 4
  , .equal 1185444 2676792 1 1 4
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
