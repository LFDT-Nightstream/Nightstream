import Nightstream.Implementation.NebulaV2.TerminalManifestNifsCall
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalManifestNifsCall.Call.CarryBlocks.priorAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestNifsCall.Call.CarryBlocks.priorAccepted

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalManifestNifsCall.Call.CarryBlocks.intermediateAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestNifsCall.Call.CarryBlocks.intermediateAccepted

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalClosedCarryRows.parsed_phase_closed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalClosedCarryRows.parsed_phase_closed

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalManifestNifsCall.Call.consumesExactAcceptedTrailingClaim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestNifsCall.Call.consumesExactAcceptedTrailingClaim

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalManifestNifsCall.Call.selectedTransitionToClosed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestNifsCall.Call.selectedTransitionToClosed

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalManifestNifsCall.Call.priorStateLinkedAndClosesExactAcceptedTrailingClaim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestNifsCall.Call.priorStateLinkedAndClosesExactAcceptedTrailingClaim

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalManifestNifsCall.Call.exactPaperInputMatchesRunningRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestNifsCall.Call.exactPaperInputMatchesRunningRows

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalManifestSchema.Artifact.foldedBundlesCommonOpenings' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestSchema.Artifact.foldedBundlesCommonOpenings

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalManifestSchema.Artifact.knownNumericRows_lower_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestSchema.Artifact.knownNumericRows_lower_bound

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalManifestSchema.Artifact.combinedKnownRows_lower_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestSchema.Artifact.combinedKnownRows_lower_bound

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalManifestSchema.Artifact.typedOpeningRows_exceed_25_variable_cube' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestSchema.Artifact.typedOpeningRows_exceed_25_variable_cube

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalManifestSchema.Artifact.cannot_fit_generated_domain_at_25' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestSchema.Artifact.cannot_fit_generated_domain_at_25

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalManifestSchema.Artifact.opensSelectedRelationExponent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestSchema.Artifact.opensSelectedRelationExponent
