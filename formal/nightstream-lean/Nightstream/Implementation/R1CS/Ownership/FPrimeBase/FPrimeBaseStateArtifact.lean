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
def artifactSha256 : String := "62cd8b38ed65e890ddac462cb6c66a1de22a30093865fa9169e420d77be9f605"
def witnessSha256 : String := "4a7c6f6e7d43fae90beded4d4646d6982494dfcf4e5a15161cc0b59ced98bda4"

def rowCount : Nat := 31
def colCount : Nat := 36

def pinnedColumns : List Nat :=
  [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 28, 29, 30, 31, 24, 25, 26, 27, 9, 10, 19]

def pinnedValues : List Nat :=
  [13105892220216807217, 9061102668333545749, 1787228973076538554, 10620686771465448400,
   17168707872888128320, 11050799198242575901, 16730522141919911230, 5655123306428251295,
   12016668175201939073, 18153209110320184117, 13406471054362354849, 7608310618811630534,
   12016668175201939073, 18153209110320184117, 13406471054362354849, 7608310618811630534,
   0, 0, 0, 0,
   4571933635639886311, 10921333946711030885, 15152476362960729356, 1784988741163108546,
   9315490177404697914, 7329743875688419918, 14709518665197956757, 2566779649216726902,
   0, 0, 1]

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
  [1, 13105892220216807217, 9061102668333545749, 1787228973076538554,
   10620686771465448400, 17168707872888128320, 11050799198242575901,
   16730522141919911230, 5655123306428251295, 0, 0,
   12016668175201939073, 18153209110320184117, 13406471054362354849,
   7608310618811630534, 12016668175201939073, 18153209110320184117,
   13406471054362354849, 7608310618811630534, 1, 0, 0, 0, 0,
   9315490177404697914, 7329743875688419918, 14709518665197956757,
   2566779649216726902, 4571933635639886311, 10921333946711030885,
   15152476362960729356, 1784988741163108546, 0, 0, 0, 0]

theorem rows_length : rows.length = rowCount := by decide
theorem pins_canonical : ∀ pin ∈ pins, pin.2 < goldilocksP := by decide

end Nightstream.Implementation.R1CS.FPrimeBaseState
