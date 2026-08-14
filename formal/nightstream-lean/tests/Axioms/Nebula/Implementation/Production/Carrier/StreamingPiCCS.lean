import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCCS
import tests.Axioms.Support

/-! Dependency audit for bounded-round PiCCS refinement. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcs.checkRoundsFrom_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiCcs.checkRoundsFrom_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcs.derive_transcript_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiCcs.derive_transcript_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcs.check_eq_protocolVerifier_check' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiCcs.check_eq_protocolVerifier_check

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcs.check_implies_tableTruth_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiCcs.check_implies_tableTruth_or_badEvent
