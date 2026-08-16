import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedOuterNorm
import tests.Axioms.Support

/-! Dependency audit for the normalized PiRLC outer-norm transfer. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOuterNorm.Normalized

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOuterNorm.Normalized.borrowCoordinatesNormFour_of_outerNorm' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms borrowCoordinatesNormFour_of_outerNorm

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOuterNorm.Normalized.borrowCoordinatesNormFour_of_freshCcsHolds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms borrowCoordinatesNormFour_of_freshCcsHolds

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOuterNorm.Normalized.radixFourCandidate_borrowCoordinatesNormFour' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms radixFourCandidate_borrowCoordinatesNormFour
