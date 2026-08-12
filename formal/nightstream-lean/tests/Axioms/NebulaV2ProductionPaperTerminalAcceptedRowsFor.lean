import Nightstream.Implementation.NebulaV2.ProductionPaperTerminalAcceptedRowsFor
import tests.Axioms.Support

/-! Dependency audit for the row-accepted terminal F-prime package. -/

open Nightstream.Implementation.NebulaV2.ProductionPaperTerminalAcceptedRowsFor

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperTerminalAcceptedRowsFor.Rows.existsResult' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Rows.existsResult

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperTerminalAcceptedRowsFor.Rows.result' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Rows.result

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperTerminalAcceptedRowsFor.Rows.recursiveCompactManifestExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Rows.recursiveCompactManifestExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperTerminalAcceptedRowsFor.Accepted.exactInvocation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Accepted.exactInvocation
