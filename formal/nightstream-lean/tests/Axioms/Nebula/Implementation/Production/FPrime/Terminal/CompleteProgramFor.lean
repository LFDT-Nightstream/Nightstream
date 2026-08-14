import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.CompleteProgramFor
import tests.Axioms.Support

/-! Dependency audit for the verifier-owned complete terminal program. -/

open Nightstream.Implementation.Nebula.ProductionPaperTerminalCompleteProgramFor

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalCompleteProgramFor.ChildProgram.childRows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ChildProgram.childRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalCompleteProgramFor.ChildProgram.coreRows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ChildProgram.coreRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalCompleteProgramFor.ChildProgram.rows_satisfied_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ChildProgram.rows_satisfied_iff

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalCompleteProgramFor.splitRowsSatisfied_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms splitRowsSatisfied_iff

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalCompleteProgramFor.Program.RowsSatisfied.child' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.RowsSatisfied.child

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalCompleteProgramFor.Program.family' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.family
