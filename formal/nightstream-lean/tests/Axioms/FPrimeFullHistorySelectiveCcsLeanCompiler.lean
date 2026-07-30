import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.CurrentNifsPhysicalRefinement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.CurrentTerminalPhysicalRefinement
import tests.Axioms.Support

/-! Fail-closed dependency gate for the Lean-owned current-program compiler. -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.DirectRows.constraintSatisfied_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.DirectRows.constraintSatisfied_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.EncodingRows.encoding_row_column_allocated' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.EncodingRows.encoding_row_column_allocated

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.EncodingRows.indexedAccepts_iff_physicalSatisfies' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.EncodingRows.indexedAccepts_iff_physicalSatisfies

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.EncodingRows.indexedAssignment_accepts_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.EncodingRows.indexedAssignment_accepts_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.Ownership.compiledRow_has_exactly_one_source_owner' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.Ownership.compiledRow_has_exactly_one_source_owner

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.Profile.exactRowDomain' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.Profile.exactRowDomain

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentCompiler.obligationTree' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentCompiler.obligationTree

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentCompiler.manifestCanonical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentCompiler.manifestCanonical

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentCompiler.evidence' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentCompiler.evidence

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentDeployment.deployment_step_cir_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentDeployment.deployment_step_cir_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentDeployment.deployment_step_cir_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentDeployment.deployment_step_cir_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentDeployment.deployment_terminal_cir_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentDeployment.deployment_terminal_cir_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentDeployment.deployment_terminal_cir_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentDeployment.deployment_terminal_cir_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentDeployment.deployment_recursive_nifs_refines_or_bound_event' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentDeployment.deployment_recursive_nifs_refines_or_bound_event

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentDeployment.deployment_structural_evidence' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentDeployment.deployment_structural_evidence

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentNifsPhysicalRefinement.deployment_recursive_nifs_refines_from_physical_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentNifsPhysicalRefinement.deployment_recursive_nifs_refines_from_physical_rows

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.SchemaBundles.decode_exists' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Goldilocks.SchemaBundles.decode_exists

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.Columns.decode_exists' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Goldilocks.Columns.decode_exists

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalApplicationRecovery.terminalInputSchema_exactWidthRecoverable' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalApplicationRecovery.terminalInputSchema_exactWidthRecoverable

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalApplicationRecovery.terminalInput_decode_exists' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalApplicationRecovery.terminalInput_decode_exists

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentTerminalPhysicalRefinement.deployment_terminal_refines_from_physical_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentTerminalPhysicalRefinement.deployment_terminal_refines_from_physical_rows
