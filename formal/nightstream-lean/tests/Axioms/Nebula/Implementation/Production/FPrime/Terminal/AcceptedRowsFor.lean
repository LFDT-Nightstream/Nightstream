import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.AcceptedRowsFor
import tests.Axioms.Support

/-! Dependency audit for the row-accepted terminal F-prime package. -/

open Nightstream.Implementation.Nebula.ProductionPaperTerminalAcceptedRowsFor

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalAcceptedRowsFor.Rows.existsResult' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Rows.existsResult

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalAcceptedRowsFor.Rows.result' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Rows.result

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalAcceptedRowsFor.Rows.recursiveCompactManifestExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Rows.recursiveCompactManifestExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalAcceptedRowsFor.Accepted.exactInvocation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Accepted.exactInvocation
