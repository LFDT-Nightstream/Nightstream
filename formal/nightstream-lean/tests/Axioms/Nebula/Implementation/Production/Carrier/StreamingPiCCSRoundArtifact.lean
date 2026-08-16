import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCCSRoundArtifact
import tests.Axioms.Support

/-! Dependency audit for the generated production PiCCS round relation. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundArtifact.Generated.rawArtifact_valid' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundArtifact.Generated.rawArtifact_valid

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundArtifact.Generated.sourceRows_below' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundArtifact.Generated.sourceRows_below

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundArtifact.Generated.generated_selective_ccs_implies_roundPhaseRelation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundArtifact.Generated.generated_selective_ccs_implies_roundPhaseRelation
