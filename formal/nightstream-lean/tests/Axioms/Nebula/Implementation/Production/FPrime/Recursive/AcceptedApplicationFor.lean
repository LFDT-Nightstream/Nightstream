import Nightstream.Implementation.Nebula.Production.FPrime.Recursive.AcceptedApplicationFor
import tests.Axioms.Support

/-! Dependency audit for the row-derived recursive F-prime continuation. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Rows.statementIdExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Rows.statementIdExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Rows.nifsOutputAlias' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Rows.nifsOutputAlias

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Rows.currentMemory' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Rows.currentMemory

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Rows.currentMemoryStartParsed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Rows.currentMemoryStartParsed

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Application.outgoingParsed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Application.outgoingParsed

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Application.successorRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Application.successorRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Application.successorPlaced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Application.successorPlaced

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Application.authorityPlaced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Application.authorityPlaced

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Application.exactInvocation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor.Application.exactInvocation
