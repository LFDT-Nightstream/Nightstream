import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol
import tests.Axioms.Support

/-! Kernel dependency report for the bounded PiRLC shared + `y_zcol` source bridge. -/

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Checked.exactRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Checked.exactRows

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Checked.structureValid' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Checked.structureValid

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Checked.sourceStagePathsUnique' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Checked.sourceStagePathsUnique

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Census.definitionOutputs_eq_allocatedColumns' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Census.definitionOutputs_eq_allocatedColumns

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ArtifactRows.certificate_covers' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ArtifactRows.certificate_covers

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ArtifactRows.rowsSatisfied_of_sourceRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ArtifactRows.rowsSatisfied_of_sourceRows

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ProducerBinding.serializerIndicesMatch' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProducerBinding.serializerIndicesMatch

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ActiveBridge.rows_decodedOutput_eq_messageAggregate_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBridge.rows_decodedOutput_eq_messageAggregate_or_badRoot

/-! ## Selective-row semantic refinement -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.Coefficients.rewriteCoefficientsMatch_of_shape_check_true' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Selective.RewriteBridge.Coefficients.rewriteCoefficientsMatch_of_shape_check_true

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Checked.uniqueOwnerAndFamily' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Selective.Materialized.Checked.uniqueOwnerAndFamily

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Artifact.allRowsDecode_true' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Selective.Materialized.Artifact.allRowsDecode_true

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode.completeSourcePartition' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Selective.SourceDecode.completeSourcePartition

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.retainedSlotFast_exists' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Selective.HonestAssignment.retainedSlotFast_exists

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceProgram.compilerSourceOutputDefinitions_exact' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Selective.SourceProgram.compilerSourceOutputDefinitions_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.decodedDerivedRecurrenceRegistryExact' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Selective.RewriteBridge.decodedDerivedRecurrenceRegistryExact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.rewriteCoefficientsExact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Selective.RewriteBridge.rewriteCoefficientsExact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.retainedCoefficientsExact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Selective.RewriteBridge.retainedCoefficientsExact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.evaluationGroupsExact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Selective.QuadraticRefinement.evaluationGroupsExact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.productGroupsExact' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Selective.QuadraticRefinement.productGroupsExact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Soundness.selectedRows_imply_rowsSatisfied' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Selective.Soundness.selectedRows_imply_rowsSatisfied

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Soundness.selectedRows_decodedOutput_eq_messageAggregate_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Selective.Soundness.selectedRows_decodedOutput_eq_messageAggregate_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.materializedWitnessRewriteProgramHolds' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Selective.HonestAssignment.materializedWitnessRewriteProgramHolds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.materializedDerivedWitnessRecurrencesHold' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Selective.HonestAssignment.materializedDerivedWitnessRecurrencesHold

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.TerminalSemantics.rewriteTerminalsHold_of_honestSource' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Selective.HonestAssignment.TerminalSemantics.rewriteTerminalsHold_of_honestSource

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.TerminalSemantics.exists_selectedRows_of_honestSource' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Selective.HonestAssignment.TerminalSemantics.exists_selectedRows_of_honestSource
