import Nightstream.Implementation.Nebula.Production.FPrime.Base.AcceptedRowsFor
import tests.Axioms.Support

/-! Dependency audit for the row-accepted base F-prime package. -/

open Nightstream.Implementation.Nebula.ProductionPaperBaseAcceptedRowsFor

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperBaseAcceptedRowsFor.Accepted.openedExists' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Accepted.openedExists

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperBaseAcceptedRowsFor.Accepted.opened' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Accepted.opened

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperBaseAcceptedRowsFor.Accepted.activeOfWireExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Accepted.activeOfWireExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperBaseAcceptedRowsFor.Accepted.outgoingSemanticExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Accepted.outgoingSemanticExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperBaseAcceptedRowsFor.Supplement.memoryResult' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Supplement.memoryResult

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperBaseAcceptedRowsFor.Supplement.outgoingValue_eq_firstBoundary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Supplement.outgoingValue_eq_firstBoundary

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperBaseAcceptedRowsFor.Supplement.memoryStartsAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Supplement.memoryStartsAt

#print axioms Supplement.challengeAuthorityExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperBaseAcceptedRowsFor.Supplement.evidence' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Supplement.evidence
