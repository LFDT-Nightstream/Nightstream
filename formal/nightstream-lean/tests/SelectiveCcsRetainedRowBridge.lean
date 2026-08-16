import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.RetainedRowBridge

/-! Focused API regression for the generic retained-row bridge. -/

namespace Tests.SelectiveCcsRetainedRowBridge

open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage

example : slotRadix 41 = 3 := by native_decide
example : slotRadix 23 = 7 := by native_decide
example : slotRadix 1 = 2 := by native_decide

#check Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.RetainedRowBridge.actions_eq_source_values
#check Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.RetainedRowBridge.residual_zero_iff_rowHolds
#check Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.RetainedRowBridge.physical_residual_zero_iff_rowHolds

end Tests.SelectiveCcsRetainedRowBridge
