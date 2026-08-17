import Nightstream.Implementation.R1CS.Core.Semantics

/-!
Exact verifier-owned base-state pin program emitted by Rust
`engine::decider::enforce_base_state_constants` for the seeded plain-chain
fixture. The 31 rows pin every state-in authority coordinate. Columns 32-35
are dummy `x_out` lanes allocated by the isolation wrapper and are deliberately
outside this row family's ownership.
-/

namespace Nightstream.Implementation.R1CS.FPrimeBaseState

open Nightstream.Implementation.R1CS

def schemaVersion : Nat := 1
def artifactKind : String := "r1cs/f-prime-base-state"
def sourceAnchor : String := "enforce_base_state_constants"
def artifactSha256 : String := "fbd0ee586476eb86c2d22b4cd19207e1091120769dc61cdf85cb74b9e9891ab7"

def witnessSha256 : String := "b3dc2c405814c8b1d3f9ded2ef0d3f730ffb41f9a41942136b5928796f323831"

def rowCount : Nat := 31
def colCount : Nat := 36

def pinnedColumns : List Nat :=
  [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 28, 29, 30, 31, 24, 25, 26, 27, 9, 10, 19]

def pinnedValues : List Nat :=
  [2085410891484037567, 1039630161256871293, 15877356545814311282, 16259737719260645155, 18056588915562328838, 15420768804774625128, 3057609135217599288, 17281897161951014799, 11740281419716179039, 6454719492364217882, 10952734345602844620, 13864950562540229939, 11740281419716179039, 6454719492364217882, 10952734345602844620, 13864950562540229939, 0, 0, 0, 0, 9657258947962146095, 10386410260596289940, 6585180676873716216, 3948890690718989481, 6050346961767540117, 6831654115071457408, 13584604561938226767, 10634950855421314340, 0, 0, 1]

def pins : List (Nat × Nat) := pinnedColumns.zip pinnedValues

/-- Exact `(wire - verifierConstant) * 1 = 0` row, omitting a zero
coefficient exactly as the Rust sparse builder does. -/
def pinRow (pin : Nat × Nat) : Row :=
  if pin.2 = 0 then
    ⟨[(pin.1, 1)], [(0, 1)], []⟩
  else
    ⟨[(pin.1, 1), (0, goldilocksP - pin.2)], [(0, 1)], []⟩

def rows : List Row := pins.map pinRow

def honestWitness : List Nat :=
  [1, 2085410891484037567, 1039630161256871293, 15877356545814311282, 16259737719260645155, 18056588915562328838, 15420768804774625128, 3057609135217599288, 17281897161951014799, 0, 0, 11740281419716179039, 6454719492364217882, 10952734345602844620, 13864950562540229939, 11740281419716179039, 6454719492364217882, 10952734345602844620, 13864950562540229939, 1, 0, 0, 0, 0, 6050346961767540117, 6831654115071457408, 13584604561938226767, 10634950855421314340, 9657258947962146095, 10386410260596289940, 6585180676873716216, 3948890690718989481, 0, 0, 0, 0]

theorem rows_length : rows.length = rowCount := by decide
theorem pins_canonical : ∀ pin ∈ pins, pin.2 < goldilocksP := by decide

end Nightstream.Implementation.R1CS.FPrimeBaseState
