import Nightstream.Implementation.R1CS.Core.AffinePins

/-! Generated exact affine-pin phase. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcTerminalShape

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "fa0681a152b2404443fc3b2f5bb56eec7d480622aab988de5918db1365905f99"
def rowStart : Nat := 2853717
def rowEnd : Nat := 2853882
def rowCount : Nat := 165

def pinRuns : List AffinePins.Run :=
  [ .zero 2675983 0 1
  , .constant 2676796 1 54 0 1
  , .constant 2676797 1 18 36 2
  , .constant 2676799 1 257 0 2
  , .zero 2677773 0 1
  , .constant 2678458 1 54 0 1
  , .constant 2678459 1 18 36 2
  , .constant 2678461 1 257 0 2
  , .zero 2679435 0 1
  , .constant 2680120 1 54 0 1
  , .constant 2680121 1 18 36 2
  , .constant 2680123 1 257 0 2
  , .zero 2681097 0 1
  , .constant 2681782 1 54 0 1
  , .constant 2681783 1 18 36 2
  , .constant 2681785 1 257 0 2
  , .zero 2682759 0 1
  , .constant 2683444 1 54 0 1
  , .constant 2683445 1 18 36 2
  , .constant 2683447 1 257 0 2
  , .zero 2684421 0 1
  , .constant 2685106 1 54 0 1
  , .constant 2685107 1 18 36 2
  , .constant 2685109 1 257 0 2
  , .zero 2686083 0 1
  , .constant 2686768 1 54 0 1
  , .constant 2686769 1 18 36 2
  , .constant 2686771 1 257 0 2
  , .zero 2687745 0 1
  , .constant 2688430 1 54 0 1
  , .constant 2688431 1 18 36 2
  , .constant 2688433 1 257 0 2
  , .zero 2689407 0 1
  , .constant 2690092 1 54 0 1
  , .constant 2690093 1 18 36 2
  , .constant 2690095 1 257 0 2
  , .zero 2691069 0 1
  , .constant 2691754 1 54 0 1
  , .constant 2691755 1 18 36 2
  , .constant 2691757 1 257 0 2
  , .zero 2692731 0 1
  , .constant 2693416 1 54 0 1
  , .constant 2693417 1 18 36 2
  , .constant 2693419 1 257 0 2
  , .zero 2694393 0 1
  , .constant 2695078 1 54 0 1
  , .constant 2695079 1 18 36 2
  , .constant 2695081 1 257 0 2
  , .zero 2696055 0 1
  , .constant 2696740 1 54 0 1
  , .constant 2696741 1 18 36 2
  , .constant 2696743 1 257 0 2
  , .zero 2697717 0 1
  , .constant 2698402 1 54 0 1
  , .constant 2698403 1 18 36 2
  , .constant 2698405 1 257 0 2
  , .zero 2699379 0 1
  , .constant 2700064 1 54 0 1
  , .constant 2700065 1 18 36 2
  , .constant 2700067 1 257 0 2
  , .equal 1160659 1158869 1 1 2
  , .equal 1161633 1159843 1 1 2
  , .equal 1162173 1160383 276 0 1
  , .equal 1162449 1158869 1 1 2
  , .equal 1163423 1159843 1 1 2
  , .equal 1163963 1160383 276 0 1
  , .equal 1164239 1158869 1 1 2
  , .equal 1165213 1159843 1 1 2
  , .equal 1165753 1160383 276 0 1
  , .equal 1166029 1158869 1 1 2
  , .equal 1167003 1159843 1 1 2
  , .equal 1167543 1160383 276 0 1
  , .equal 1167819 1158869 1 1 2
  , .equal 1168793 1159843 1 1 2
  , .equal 1169333 1160383 276 0 1
  , .equal 1169609 1158869 1 1 2
  , .equal 1170583 1159843 1 1 2
  , .equal 1171123 1160383 276 0 1
  , .equal 1171399 1158869 1 1 2
  , .equal 1172373 1159843 1 1 2
  , .equal 1172913 1160383 276 0 1
  , .equal 1173189 1158869 1 1 2
  , .equal 1174163 1159843 1 1 2
  , .equal 1174703 1160383 276 0 1
  , .equal 1174979 1158869 1 1 2
  , .equal 1175953 1159843 1 1 2
  , .equal 1176493 1160383 276 0 1
  , .equal 1176769 1158869 1 1 2
  , .equal 1177743 1159843 1 1 2
  , .equal 1178283 1160383 276 0 1
  , .equal 1178559 1158869 1 1 2
  , .equal 1179533 1159843 1 1 2
  , .equal 1180073 1160383 276 0 1
  , .equal 1180349 1158869 1 1 2
  , .equal 1181323 1159843 1 1 2
  , .equal 1181863 1160383 276 0 1
  , .equal 1182139 1158869 1 1 2
  , .equal 1183113 1159843 1 1 2
  , .equal 1183653 1160383 276 0 1
  , .equal 1183929 1158869 1 1 2
  , .equal 1184903 1159843 1 1 2
  , .equal 1185443 1160383 1491353 0 1
  , .equal 2676796 1158869 1 1 2
  , .equal 2676798 1159843 1 1 2
  , .equal 2676800 1160383 0 0 1
  ]

def pins : List AffinePins.Pin := AffinePins.expandRuns pinRuns
def rows : List Row := AffinePins.rows pins

theorem pins_canonical : AffinePins.PinsCanonical pins := by native_decide
theorem pins_length : pins.length = rowCount := by
rw [pins, AffinePins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, AffinePins.rows] using pins_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcTerminalShape
