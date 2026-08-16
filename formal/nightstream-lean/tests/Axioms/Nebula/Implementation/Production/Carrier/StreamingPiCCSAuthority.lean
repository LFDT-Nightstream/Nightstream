import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCCSAuthority
import tests.Axioms.Support

/-! Dependency audit for authoritative phased production PiCCS. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsAuthority.runRounds_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiCcsAuthority.runRounds_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsAuthority.productionCheck_eq_piCcsCheck' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiCcsAuthority.productionCheck_eq_piCcsCheck

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsAuthority.accepted_different_round_implies_collision' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiCcsAuthority.accepted_different_round_implies_collision
