import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCArtifact
import tests.Axioms.Support

/-! Dependency audit for the generated production PiRLC phase relation. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.rawArtifact_valid' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.rawArtifact_valid

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.sourceRows_below' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.sourceRows_below

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.generated_selective_ccs_implies_concrete_phase' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.generated_selective_ccs_implies_concrete_phase
