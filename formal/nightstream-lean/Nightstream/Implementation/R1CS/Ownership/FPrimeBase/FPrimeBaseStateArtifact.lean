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
def artifactSha256 : String := "2555085716adc94b2bd8c5e9858e2affdc60670f372445161cf12fa1089b665b"
def witnessSha256 : String := "ea5aea1636073c96bb3c1c44097f23897124787c5ded087ffee22d28a3b550c2"

def rowCount : Nat := 31
def colCount : Nat := 36

def pinnedColumns : List Nat :=
  [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 28, 29, 30, 31, 24, 25, 26, 27, 9, 10, 19]

def pinnedValues : List Nat :=
  [15098281187220978216, 8172334620013804168, 5307991145549437136, 1226446791757975599,
   2467705724746103983, 13983624219613256042, 15663551642325268602, 10092319468191714374,
   15669055425327964029, 17664972994601734879, 5409715868785689033, 5614119764957744445,
   15669055425327964029, 17664972994601734879, 5409715868785689033, 5614119764957744445,
   0, 0, 0, 0,
   17541660989439515505, 13024523900316705357, 16201612681584537812, 7125213107700718395,
   6050346961767540117, 6831654115071457408, 13584604561938226767, 10634950855421314340,
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
  [1, 15098281187220978216, 8172334620013804168, 5307991145549437136,
   1226446791757975599, 2467705724746103983, 13983624219613256042,
   15663551642325268602, 10092319468191714374, 0, 0,
   15669055425327964029, 17664972994601734879, 5409715868785689033,
   5614119764957744445, 15669055425327964029, 17664972994601734879,
   5409715868785689033, 5614119764957744445, 1, 0, 0, 0, 0,
   6050346961767540117, 6831654115071457408, 13584604561938226767,
   10634950855421314340, 17541660989439515505, 13024523900316705357,
   16201612681584537812, 7125213107700718395, 0, 0, 0, 0]

theorem rows_length : rows.length = rowCount := by decide
theorem pins_canonical : ∀ pin ∈ pins, pin.2 < goldilocksP := by decide

end Nightstream.Implementation.R1CS.FPrimeBaseState
