import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTraceRawProjectionRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.FinalRoundFactorization
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.ProductionHonestAssignment
import tests.Axioms.Support

/-! Fail-closed dependency gate for terminal raw-old-block composition. -/

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary.claimsAcceptedTerminalRawProjection_implies_packed_or_parentOpeningBadEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBoundary.claimsAcceptedTerminalRawProjection_implies_packed_or_parentOpeningBadEvent

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTrace.Trace.terminalRawProjection_implies_baseAndAllPacked_or_parentOpeningFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveTrace.Trace.terminalRawProjection_implies_baseAndAllPacked_or_parentOpeningFailure

/- The fixed generated-row execution proof and its delayed trace composition
have the same kernel dependency boundary as the surrounding protocol proofs. -/
/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.productionRows_projectionOpeningAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalRawOldBlockProjectionArtifact.productionRows_projectionOpeningAccepted

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTrace.Trace.terminalRawProjectionRows_imply_baseAndAllPacked_or_parentOpeningFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveTrace.Trace.terminalRawProjectionRows_imply_baseAndAllPacked_or_parentOpeningFailure

/- The executable-terminal and strong active compositions preserve the same
kernel dependency boundary. -/
/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTrace.Trace.terminalRawProjectionRowsChecked_implies_terminalChecked' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveTrace.Trace.terminalRawProjectionRowsChecked_implies_terminalChecked

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTrace.Trace.terminalRawProjectionRows_imply_baseAllPackedAndAllPaper_or_parentOpeningFailure_or_paperFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveTrace.Trace.terminalRawProjectionRows_imply_baseAllPackedAndAllPaper_or_parentOpeningFailure_or_paperFailure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.productionWeightedPrefixProjection_factorFinalRound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalRawOldBlockProjectionArtifact.productionWeightedPrefixProjection_factorFinalRound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.honestArtifactRowsSatisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalRawOldBlockProjectionArtifact.honestArtifactRowsSatisfied
