import Nightstream.Implementation.R1CS.FPrimeStateLinkSound

/-!
Exact state-link witnesses: the all-zero plain states satisfy every direct
wire equality, while changing only the next step-count wire fails row 9.
-/

set_option maxRecDepth 32768

namespace NightstreamTests.FPrimeStateLink

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeStateLink

def honestAssignment (column : Nat) : Nat :=
  if column = 0 then 1 else 0

example : Satisfies rows honestAssignment := by native_decide

def forgedAssignment (column : Nat) : Nat :=
  if column = 41 then 1 else honestAssignment column

example : ¬ Satisfies rows forgedAssignment := by native_decide

example : ∀ row ∈ rows.take 9, RowHolds forgedAssignment row := by
  native_decide

example : ¬ RowHolds forgedAssignment ((rows.drop 9).head (by decide)) := by
  native_decide

end NightstreamTests.FPrimeStateLink
