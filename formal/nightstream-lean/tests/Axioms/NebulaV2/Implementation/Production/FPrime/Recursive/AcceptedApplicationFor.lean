import Nightstream.Implementation.NebulaV2.Production.FPrime.Recursive.AcceptedApplicationFor
import tests.Axioms.Support

/-! Dependency audit for the row-derived recursive F-prime continuation. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Rows.statementIdExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Rows.statementIdExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Rows.nifsOutputAlias' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Rows.nifsOutputAlias

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Rows.currentMemory' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Rows.currentMemory

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Rows.currentMemoryStartParsed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Rows.currentMemoryStartParsed

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Application.outgoingParsed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Application.outgoingParsed

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Application.successorRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Application.successorRows

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Application.successorPlaced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Application.successorPlaced

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Application.authorityPlaced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Application.authorityPlaced

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Application.exactInvocation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor.Application.exactInvocation
