import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.RetainedRowBridge
import tests.Axioms.Support

/-! Fail-closed dependency gate for the generic retained-row bridge. -/

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.RetainedRowBridge.actions_eq_source_values' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.RetainedRowBridge.actions_eq_source_values

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.RetainedRowBridge.residual_zero_iff_rowHolds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.RetainedRowBridge.residual_zero_iff_rowHolds

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.RetainedRowBridge.physical_residual_zero_iff_rowHolds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.RetainedRowBridge.physical_residual_zero_iff_rowHolds
