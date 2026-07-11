import Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound

/-!
Lean twins of the Rust terminal delayed-link vectors. The honest assignment
satisfies all 257 exact rows; affine-one and bit mutations fail at the same
first row as the production isolation harness.
-/

set_option maxRecDepth 32768

namespace NightstreamTests.FPrimeTerminalLink

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeTerminalLink

def honestAssignment (column : Nat) : Nat :=
  if column = 0 then 1
  else if column ≤ 256 then (column - 1) % 2
  else if column = 257 then 1
  else if column < 514 then (column - 258) % 2
  else 0

example : Satisfies rows honestAssignment := by native_decide

def wrongOneAssignment (column : Nat) : Nat :=
  if column = freshOneCol then 0 else honestAssignment column

example : ¬ Satisfies rows wrongOneAssignment := by native_decide

example : ¬ RowHolds wrongOneAssignment (rows.head (by decide)) := by
  native_decide

def wrongBitAssignment (column : Nat) : Nat :=
  if column = freshBitCol 37 then 1 - honestAssignment column
  else honestAssignment column

example : ¬ Satisfies rows wrongBitAssignment := by native_decide

example : ∀ row ∈ rows.take 38, RowHolds wrongBitAssignment row := by
  native_decide

example : ¬ RowHolds wrongBitAssignment ((rows.drop 38).head (by decide)) := by
  native_decide

end NightstreamTests.FPrimeTerminalLink
