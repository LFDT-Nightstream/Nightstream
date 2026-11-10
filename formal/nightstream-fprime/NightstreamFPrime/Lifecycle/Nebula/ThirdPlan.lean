import NightstreamFPrime.Spec.Nebula.SetupCheck
import NightstreamFPrime.Lifecycle.Nebula.MachineApp

/-! Owns the geometry of the third memory-application test geometry, whose RAM is smaller
than its ROM: ROM has eight words and RAM four, so `μ < r`, the address field has `r` bits, and
the RAM range rows of row O6 exist. A segment has two steps, there are at most four segments,
and timestamps have 5 bits. This is a test geometry, not a production plan. -/

namespace NightstreamFPrime.Lifecycle.Nebula.ThirdPlan

open NightstreamFPrime.Spec.Nebula

/-- The geometry of the third test application. -/
def geometry : Geometry where
  r := 3
  μ := 2
  wTs := 5
  bOps := 2
  bScan := 6
  n := 2
  sMax := 4

theorem two : geometry.bOps = 2 := rfl

theorem valid : geometry.Valid where
  exactCover := by decide
  timestampRange := by decide
  fieldEncoding := by decide
  laneBits := by decide
  belowModulus := by decide
  positive := by decide

theorem secure : geometry.Secure := by
  unfold Geometry.Secure
  decide

/-- The machine as the geometry's application. -/
def app : PortApp geometry := Machine.portApp geometry two

end NightstreamFPrime.Lifecycle.Nebula.ThirdPlan
