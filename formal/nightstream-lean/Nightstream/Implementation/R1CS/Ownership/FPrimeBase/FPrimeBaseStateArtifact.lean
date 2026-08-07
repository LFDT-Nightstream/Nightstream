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
def artifactSha256 : String := "aaf33fe0c29c897ea93305679c50539c1b64d966603badd022cc0e16a592bd26"
def witnessSha256 : String := "65d1b9220f7b27f92f19ad68c4b29b5e636f38b01bb4fad90b4c6e0f5d8cd8a9"

def rowCount : Nat := 31
def colCount : Nat := 36

def pinnedColumns : List Nat :=
  [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 28, 29, 30, 31, 24, 25, 26, 27, 9, 10, 19]

def pinnedValues : List Nat :=
  [6555185143986702626, 6579791271889664458, 7317575425397797303, 2592549744166101920, 18056588915562328838, 15420768804774625128, 3057609135217599288, 17281897161951014799, 11740281419716179039, 6454719492364217882, 10952734345602844620, 13864950562540229939, 11740281419716179039, 6454719492364217882, 10952734345602844620, 13864950562540229939, 0, 0, 0, 0, 9657258947962146095, 10386410260596289940, 6585180676873716216, 3948890690718989481, 6050346961767540117, 6831654115071457408, 13584604561938226767, 10634950855421314340, 0, 0, 1]

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
  [1, 6555185143986702626, 6579791271889664458, 7317575425397797303, 2592549744166101920, 18056588915562328838, 15420768804774625128, 3057609135217599288, 17281897161951014799, 0, 0, 11740281419716179039, 6454719492364217882, 10952734345602844620, 13864950562540229939, 11740281419716179039, 6454719492364217882, 10952734345602844620, 13864950562540229939, 1, 0, 0, 0, 0, 6050346961767540117, 6831654115071457408, 13584604561938226767, 10634950855421314340, 9657258947962146095, 10386410260596289940, 6585180676873716216, 3948890690718989481, 0, 0, 0, 0]

theorem rows_length : rows.length = rowCount := by decide
theorem pins_canonical : ∀ pin ∈ pins, pin.2 < goldilocksP := by decide

end Nightstream.Implementation.R1CS.FPrimeBaseState
