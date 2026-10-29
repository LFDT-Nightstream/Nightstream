import NightstreamFPrime.Spec.Nebula.SetupCheck
import NightstreamFPrime.Lifecycle.Nebula.MachineApp

/-! Owns the geometry of the first memory-application package (spec §4) and
its default program: ROM and RAM of four words each, two ports, segments of two
steps with four scan slots each, at most four segments, and 5-bit timestamps.
The default ROM holds the machine program `loadi 5; store 0; load 0; halt`,
and the RAM starts at zero. This is a test geometry for the first end-to-end
run; a production plan selects its own. -/

namespace NightstreamFPrime.Lifecycle.Nebula.FirstPlan

open NightstreamFPrime.Spec.Nebula

/-- The ROM words `op + 4 · arg` of `Machine`: `loadi 5`, `store 0`, `load 0`,
`halt`. -/
def romImage : List ℕ := [23, 2, 1, 0]

/-- The geometry of the first package. -/
def geometry : Geometry where
  r := 2
  μ := 2
  wTs := 5
  bOps := 2
  bScan := 4
  n := 2
  sMax := 4

/-- The default program of the first package. -/
def plan : Plan := { geometry with rom := fun a => romImage.getD a 0, ram := fun _ => 0 }

/-- A second program of the geometry with distinct nonzero words in every
cell, for the Rust `D_init` check (`--application-start nebula-memory-v1-other`). -/
def otherPlan : Plan :=
  { geometry with
    rom := fun a => [4294967295, 2147483649, 305419896, 1].getD a 0
    ram := fun a => [2, 3735928559, 2147483647, 3405691582].getD a 0 }

theorem two : geometry.bOps = 2 := rfl

theorem valid : geometry.Valid where
  exactCover := by decide
  timestampRange := by decide
  fieldEncoding := by decide
  laneBits := by decide
  addressWidth := le_rfl
  belowModulus := by decide
  positive := by decide

theorem secure : geometry.Secure := by
  unfold Geometry.Secure
  decide

/-- The machine as the geometry's application. -/
def app : PortApp geometry := Machine.portApp geometry two

end NightstreamFPrime.Lifecycle.Nebula.FirstPlan
