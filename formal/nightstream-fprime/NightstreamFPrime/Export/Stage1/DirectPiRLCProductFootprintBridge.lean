import NightstreamFPrime.Export.Stage1.DirectPiRLCProductFootprint
import NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocations

/-!
Connects the fast direct PiRLC footprint constants to the canonical Lean
invocation schedule and the superseded generic R1CS template cost.
-/

namespace NightstreamFPrime.Export.Stage1.DirectPiRLCProductFootprintBridge

open DirectPiRLCProductFootprint

/-- The fast count is exactly the canonical invocation-list length. -/
theorem invocationCount_eq_canonical :
    invocationCount = PiRLCCombinationInvocations.invocations.length := by
  rw [PiRLCCombinationInvocations.invocations_length, invocationCount_eq]

/-- The old generic R1CS templates used this many rows for the same 53 ring
cells across all 17 sources. -/
def genericRowCount : Nat :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.sourceCount * 53 *
    PiRLCCombinationInvocations.laneRowCosts.sum

@[simp] theorem genericRowCount_eq : genericRowCount = 7346754 := by
  unfold genericRowCount
  rw [NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.sourceCount_eq,
    PiRLCCombinationInvocations.laneRowCosts_sum]

@[simp] theorem removedRowCount_eq : genericRowCount - rowCount = 5692518 := by
  rw [genericRowCount_eq, rowCount_eq]

end NightstreamFPrime.Export.Stage1.DirectPiRLCProductFootprintBridge
