import tests.FPrimeFullHistoryCurrentTerminalLinkPlacement
import tests.Axioms.Support

/-!
Fail-closed guards for the bounded current full-history
`terminal.latest_link` placement certificate.
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkPlacementSound.mapped_rows_eq_generated' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkPlacementSound.mapped_rows_eq_generated

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkPlacementSound.generatedRows_iff_logicalPaperLink' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkPlacementSound.generatedRows_iff_logicalPaperLink

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkPlacementSound.generatedRows_iff_sourceProgram' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkPlacementSound.generatedRows_iff_sourceProgram

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkPlacementSound.generatedRows_iff_freshPublic_eq_encodeInstance' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkPlacementSound.generatedRows_iff_freshPublic_eq_encodeInstance

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkPlacementSound.generatedRows_iff_loweringPriorLinkAccepted' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkPlacementSound.generatedRows_iff_loweringPriorLinkAccepted

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkPlacementSound.output_and_generated_rows_construct_currentPlainOwner' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkPlacementSound.output_and_generated_rows_construct_currentPlainOwner
