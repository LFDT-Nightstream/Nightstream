import Nightstream.Implementation.R1CS.Core.AffinePins

/-! Generated exact affine-pin phase. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcRecursiveShape

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "dcc076dc32d592c9bdcbcbb2c9b0ba4a62dbb2992d3684d877b97ffe84bf779d"
def rowStart : Nat := 361480
def rowEnd : Nat := 361575
def rowCount : Nat := 95

def pinRuns : List AffinePins.Run :=
  [ .zero 359602 0 1
  , .constant 360415 1 54 0 1
  , .constant 360416 1 18 36 2
  , .constant 360418 1 257 0 2
  , .zero 361392 0 1
  , .constant 362077 1 54 0 1
  , .constant 362078 1 18 36 2
  , .constant 362080 1 257 0 2
  , .zero 363054 0 1
  , .constant 363739 1 54 0 1
  , .constant 363740 1 18 36 2
  , .constant 363742 1 257 0 2
  , .zero 364716 0 1
  , .constant 365401 1 54 0 1
  , .constant 365402 1 18 36 2
  , .constant 365404 1 257 0 2
  , .zero 366378 0 1
  , .constant 367063 1 54 0 1
  , .constant 367064 1 18 36 2
  , .constant 367066 1 257 0 2
  , .zero 368040 0 1
  , .constant 368725 1 54 0 1
  , .constant 368726 1 18 36 2
  , .constant 368728 1 257 0 2
  , .zero 369702 0 1
  , .constant 370387 1 54 0 1
  , .constant 370388 1 18 36 2
  , .constant 370390 1 257 0 2
  , .zero 371364 0 1
  , .constant 372049 1 54 0 1
  , .constant 372050 1 18 36 2
  , .constant 372052 1 257 0 2
  , .zero 373026 0 1
  , .constant 373711 1 54 0 1
  , .constant 373712 1 18 36 2
  , .constant 373714 1 257 0 2
  , .zero 374688 0 1
  , .constant 375373 1 54 0 1
  , .constant 375374 1 18 36 2
  , .constant 375376 1 257 0 2
  , .zero 376350 0 1
  , .constant 377035 1 54 0 1
  , .constant 377036 1 18 36 2
  , .constant 377038 1 257 0 2
  , .zero 378012 0 1
  , .constant 378697 1 54 0 1
  , .constant 378698 1 18 36 2
  , .constant 378700 1 257 0 2
  , .zero 379674 0 1
  , .constant 380359 1 54 0 1
  , .constant 380360 1 18 36 2
  , .constant 380362 1 257 0 2
  , .zero 381336 0 1
  , .constant 382021 1 54 0 1
  , .constant 382022 1 18 36 2
  , .constant 382024 1 257 0 2
  , .zero 382998 0 1
  , .constant 383683 1 54 0 1
  , .constant 383684 1 18 36 2
  , .constant 383686 1 257 0 2
  , .equal 360415 31795 1 1 2
  , .equal 360417 32769 1 1 2
  , .equal 360419 33309 0 0 1
  ]

def pins : List AffinePins.Pin := AffinePins.expandRuns pinRuns
def rows : List Row := AffinePins.rows pins

theorem pins_canonical : AffinePins.PinsCanonical pins := by native_decide
theorem pins_length : pins.length = rowCount := by
rw [pins, AffinePins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, AffinePins.rows] using pins_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcRecursiveShape
