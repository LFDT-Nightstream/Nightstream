import Nightstream.Implementation.Lowering.Nebula.SourceSemantics
import tests.Axioms.Support

/-! Fail-closed dependency guards for the Nebula source refinement. -/

/-- info: 'Nightstream.Implementation.Lowering.Nebula.SourceSemantics.entryOfFields_packed' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.SourceSemantics.entryOfFields_packed

/-- info: 'Nightstream.Implementation.Lowering.Nebula.SourceSemantics.operationPrefixPair' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.SourceSemantics.operationPrefixPair

/-- info: 'Nightstream.Implementation.Lowering.Nebula.SourceSemantics.operationFactor_eq_fingerprint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.SourceSemantics.operationFactor_eq_fingerprint

/-- info: 'Nightstream.Implementation.Lowering.Nebula.SourceSemantics.scanFactor_eq_fingerprint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.SourceSemantics.scanFactor_eq_fingerprint
