import Nightstream.Implementation.R1CS.Core.AffinePins

/-! Generated exact affine-pin phase. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcTerminalShape

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "c1941de032a45ad4e483400af8ee2c0101d677941ef1facbbd5ea40bcdea8ef3"
def rowStart : Nat := 2953021
def rowEnd : Nat := 2953186
def rowCount : Nat := 165

def pinRuns : List AffinePins.Run :=
  [ .zero 2610463 0 1
  , .constant 2611276 1 54 0 1
  , .constant 2611277 1 18 36 2
  , .constant 2611279 1 257 0 2
  , .zero 2612253 0 1
  , .constant 2613066 1 54 0 1
  , .constant 2613067 1 18 36 2
  , .constant 2613069 1 257 0 2
  , .zero 2614043 0 1
  , .constant 2614856 1 54 0 1
  , .constant 2614857 1 18 36 2
  , .constant 2614859 1 257 0 2
  , .zero 2615833 0 1
  , .constant 2616646 1 54 0 1
  , .constant 2616647 1 18 36 2
  , .constant 2616649 1 257 0 2
  , .zero 2617623 0 1
  , .constant 2618436 1 54 0 1
  , .constant 2618437 1 18 36 2
  , .constant 2618439 1 257 0 2
  , .zero 2619413 0 1
  , .constant 2620226 1 54 0 1
  , .constant 2620227 1 18 36 2
  , .constant 2620229 1 257 0 2
  , .zero 2621203 0 1
  , .constant 2622016 1 54 0 1
  , .constant 2622017 1 18 36 2
  , .constant 2622019 1 257 0 2
  , .zero 2622993 0 1
  , .constant 2623806 1 54 0 1
  , .constant 2623807 1 18 36 2
  , .constant 2623809 1 257 0 2
  , .zero 2624783 0 1
  , .constant 2625596 1 54 0 1
  , .constant 2625597 1 18 36 2
  , .constant 2625599 1 257 0 2
  , .zero 2626573 0 1
  , .constant 2627386 1 54 0 1
  , .constant 2627387 1 18 36 2
  , .constant 2627389 1 257 0 2
  , .zero 2628363 0 1
  , .constant 2629176 1 54 0 1
  , .constant 2629177 1 18 36 2
  , .constant 2629179 1 257 0 2
  , .zero 2630153 0 1
  , .constant 2630966 1 54 0 1
  , .constant 2630967 1 18 36 2
  , .constant 2630969 1 257 0 2
  , .zero 2631943 0 1
  , .constant 2632756 1 54 0 1
  , .constant 2632757 1 18 36 2
  , .constant 2632759 1 257 0 2
  , .zero 2633733 0 1
  , .constant 2634546 1 54 0 1
  , .constant 2634547 1 18 36 2
  , .constant 2634549 1 257 0 2
  , .zero 2635523 0 1
  , .constant 2636336 1 54 0 1
  , .constant 2636337 1 18 36 2
  , .constant 2636339 1 257 0 2
  , .equal 959494 957704 1 1 2
  , .equal 960468 958678 1 1 2
  , .equal 961008 959218 276 0 1
  , .equal 961284 957704 1 1 2
  , .equal 962258 958678 1 1 2
  , .equal 962798 959218 276 0 1
  , .equal 963074 957704 1 1 2
  , .equal 964048 958678 1 1 2
  , .equal 964588 959218 276 0 1
  , .equal 964864 957704 1 1 2
  , .equal 965838 958678 1 1 2
  , .equal 966378 959218 276 0 1
  , .equal 966654 957704 1 1 2
  , .equal 967628 958678 1 1 2
  , .equal 968168 959218 276 0 1
  , .equal 968444 957704 1 1 2
  , .equal 969418 958678 1 1 2
  , .equal 969958 959218 276 0 1
  , .equal 970234 957704 1 1 2
  , .equal 971208 958678 1 1 2
  , .equal 971748 959218 276 0 1
  , .equal 972024 957704 1 1 2
  , .equal 972998 958678 1 1 2
  , .equal 973538 959218 276 0 1
  , .equal 973814 957704 1 1 2
  , .equal 974788 958678 1 1 2
  , .equal 975328 959218 276 0 1
  , .equal 975604 957704 1 1 2
  , .equal 976578 958678 1 1 2
  , .equal 977118 959218 276 0 1
  , .equal 977394 957704 1 1 2
  , .equal 978368 958678 1 1 2
  , .equal 978908 959218 276 0 1
  , .equal 979184 957704 1 1 2
  , .equal 980158 958678 1 1 2
  , .equal 980698 959218 276 0 1
  , .equal 980974 957704 1 1 2
  , .equal 981948 958678 1 1 2
  , .equal 982488 959218 276 0 1
  , .equal 982764 957704 1 1 2
  , .equal 983738 958678 1 1 2
  , .equal 984278 959218 1626998 0 1
  , .equal 2611276 957704 1 1 2
  , .equal 2611278 958678 1 1 2
  , .equal 2611280 959218 0 0 1
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
