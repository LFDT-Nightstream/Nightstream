import NightstreamFPrime.Export.Stage1.DirectPiRLCProductFootprint
import NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocations

/-!
Connects the fast direct PiRLC footprint constants to the canonical Lean
invocation schedule.
-/

namespace NightstreamFPrime.Export.Stage1.DirectPiRLCProductFootprintBridge

open DirectPiRLCProductFootprint

theorem invocationCount_eq_canonical :
    invocationCount = PiRLCCombinationInvocations.invocations.length := by
  rw [PiRLCCombinationInvocations.invocations_length, invocationCount_eq]

end NightstreamFPrime.Export.Stage1.DirectPiRLCProductFootprintBridge
