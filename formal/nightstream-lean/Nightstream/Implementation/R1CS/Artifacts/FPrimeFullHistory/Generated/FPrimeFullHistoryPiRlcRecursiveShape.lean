import Nightstream.Implementation.R1CS.Core.AffinePins

/-! Generated exact affine-pin phase. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcRecursiveShape

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "12ff4726ff7de817942ba5e9332cca9909c48e3c8f3a1d8c02b67cd0dcfa48ae"
def rowStart : Nat := 385011
def rowEnd : Nat := 385106
def rowCount : Nat := 95

def pinRuns : List AffinePins.Run :=
  [ .zero 372853 0 1
  , .constant 373666 1 54 0 1
  , .constant 373667 1 18 36 2
  , .constant 373669 1 257 0 2
  , .zero 374643 0 1
  , .constant 375456 1 54 0 1
  , .constant 375457 1 18 36 2
  , .constant 375459 1 257 0 2
  , .zero 376433 0 1
  , .constant 377246 1 54 0 1
  , .constant 377247 1 18 36 2
  , .constant 377249 1 257 0 2
  , .zero 378223 0 1
  , .constant 379036 1 54 0 1
  , .constant 379037 1 18 36 2
  , .constant 379039 1 257 0 2
  , .zero 380013 0 1
  , .constant 380826 1 54 0 1
  , .constant 380827 1 18 36 2
  , .constant 380829 1 257 0 2
  , .zero 381803 0 1
  , .constant 382616 1 54 0 1
  , .constant 382617 1 18 36 2
  , .constant 382619 1 257 0 2
  , .zero 383593 0 1
  , .constant 384406 1 54 0 1
  , .constant 384407 1 18 36 2
  , .constant 384409 1 257 0 2
  , .zero 385383 0 1
  , .constant 386196 1 54 0 1
  , .constant 386197 1 18 36 2
  , .constant 386199 1 257 0 2
  , .zero 387173 0 1
  , .constant 387986 1 54 0 1
  , .constant 387987 1 18 36 2
  , .constant 387989 1 257 0 2
  , .zero 388963 0 1
  , .constant 389776 1 54 0 1
  , .constant 389777 1 18 36 2
  , .constant 389779 1 257 0 2
  , .zero 390753 0 1
  , .constant 391566 1 54 0 1
  , .constant 391567 1 18 36 2
  , .constant 391569 1 257 0 2
  , .zero 392543 0 1
  , .constant 393356 1 54 0 1
  , .constant 393357 1 18 36 2
  , .constant 393359 1 257 0 2
  , .zero 394333 0 1
  , .constant 395146 1 54 0 1
  , .constant 395147 1 18 36 2
  , .constant 395149 1 257 0 2
  , .zero 396123 0 1
  , .constant 396936 1 54 0 1
  , .constant 396937 1 18 36 2
  , .constant 396939 1 257 0 2
  , .zero 397913 0 1
  , .constant 398726 1 54 0 1
  , .constant 398727 1 18 36 2
  , .constant 398729 1 257 0 2
  , .equal 373666 31795 1 1 2
  , .equal 373668 32769 1 1 2
  , .equal 373670 33309 0 0 1
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
