import Nightstream.Implementation.R1CS.Core.AffinePins

/-! Generated exact affine-pin phase. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsTerminalAuthorityTail

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "db37f9af25b44be7f45391c5c65a56a833e6a361cc240f81b86a10e8746ec30d"
def rowStart : Nat := 940800
def rowEnd : Nat := 941790
def rowCount : Nat := 990

def pinRuns : List AffinePins.Run :=
  [ .equal 959084 958700 1 1 2
  , .equal 959086 958828 1 1 2
  , .equal 959088 958956 1 1 2
  , .zero 958808 1 20
  , .zero 958936 1 20
  , .zero 959064 1 20
  , .equal 960874 960490 1 1 2
  , .equal 960876 960618 1 1 2
  , .equal 960878 960746 1 1 2
  , .zero 960598 1 20
  , .zero 960726 1 20
  , .zero 960854 1 20
  , .equal 962664 962280 1 1 2
  , .equal 962666 962408 1 1 2
  , .equal 962668 962536 1 1 2
  , .zero 962388 1 20
  , .zero 962516 1 20
  , .zero 962644 1 20
  , .equal 964454 964070 1 1 2
  , .equal 964456 964198 1 1 2
  , .equal 964458 964326 1 1 2
  , .zero 964178 1 20
  , .zero 964306 1 20
  , .zero 964434 1 20
  , .equal 966244 965860 1 1 2
  , .equal 966246 965988 1 1 2
  , .equal 966248 966116 1 1 2
  , .zero 965968 1 20
  , .zero 966096 1 20
  , .zero 966224 1 20
  , .equal 968034 967650 1 1 2
  , .equal 968036 967778 1 1 2
  , .equal 968038 967906 1 1 2
  , .zero 967758 1 20
  , .zero 967886 1 20
  , .zero 968014 1 20
  , .equal 969824 969440 1 1 2
  , .equal 969826 969568 1 1 2
  , .equal 969828 969696 1 1 2
  , .zero 969548 1 20
  , .zero 969676 1 20
  , .zero 969804 1 20
  , .equal 971614 971230 1 1 2
  , .equal 971616 971358 1 1 2
  , .equal 971618 971486 1 1 2
  , .zero 971338 1 20
  , .zero 971466 1 20
  , .zero 971594 1 20
  , .equal 973404 973020 1 1 2
  , .equal 973406 973148 1 1 2
  , .equal 973408 973276 1 1 2
  , .zero 973128 1 20
  , .zero 973256 1 20
  , .zero 973384 1 20
  , .equal 975194 974810 1 1 2
  , .equal 975196 974938 1 1 2
  , .equal 975198 975066 1 1 2
  , .zero 974918 1 20
  , .zero 975046 1 20
  , .zero 975174 1 20
  , .equal 976984 976600 1 1 2
  , .equal 976986 976728 1 1 2
  , .equal 976988 976856 1 1 2
  , .zero 976708 1 20
  , .zero 976836 1 20
  , .zero 976964 1 20
  , .equal 978774 978390 1 1 2
  , .equal 978776 978518 1 1 2
  , .equal 978778 978646 1 1 2
  , .zero 978498 1 20
  , .zero 978626 1 20
  , .zero 978754 1 20
  , .equal 980564 980180 1 1 2
  , .equal 980566 980308 1 1 2
  , .equal 980568 980436 1 1 2
  , .zero 980288 1 20
  , .zero 980416 1 20
  , .zero 980544 1 20
  , .equal 982354 981970 1 1 2
  , .equal 982356 982098 1 1 2
  , .equal 982358 982226 1 1 2
  , .zero 982078 1 20
  , .zero 982206 1 20
  , .zero 982334 1 20
  , .equal 984144 983760 1 1 2
  , .equal 984146 983888 1 1 2
  , .equal 984148 984016 1 1 2
  , .zero 983868 1 20
  , .zero 983996 1 20
  , .zero 984124 1 20
  ]

def pins : List AffinePins.Pin := AffinePins.expandRuns pinRuns
def rows : List Row := AffinePins.rows pins

theorem pins_canonical : AffinePins.PinsCanonical pins := by native_decide
theorem pins_length : pins.length = rowCount := by
rw [pins, AffinePins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, AffinePins.rows] using pins_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsTerminalAuthorityTail
