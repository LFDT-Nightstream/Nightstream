import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedFamilyRows
import tests.Axioms.Support

/-! Dependency audit for the joint normalized production PiRLC family rows. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.challengeAssignment_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms challengeAssignment_eq

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.carryRange_implies_algebraRange' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms carryRange_implies_algebraRange

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.decodedChallenges_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decodedChallenges_eq

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.carryAccepted_implies_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms carryAccepted_implies_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraAccepted_implies_output' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms algebraAccepted_implies_output

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.jointAccepted_implies_concrete_phase' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms jointAccepted_implies_concrete_phase
