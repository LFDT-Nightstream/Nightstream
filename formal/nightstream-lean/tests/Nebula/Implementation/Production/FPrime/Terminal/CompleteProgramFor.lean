import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.CompleteProgramFor

/-! Regression surface for the verifier-owned complete terminal program. -/

namespace tests.NebulaProductionPaperTerminalCompleteProgramFor

open Nightstream.Implementation.Nebula.ProductionPaperTerminalCompleteProgramFor

#check ChildProgram.childRows
#check ChildProgram.coreRows
#check ChildProgram.rows
#check ChildProgram.rows_satisfied_iff
#check splitRowsSatisfied_iff
#check Program.foldFrame
#check Program.childrenRows
#check Program.rows
#check Program.RowsSatisfied
#check Program.RowsSatisfied.child
#check Program.rowCount
#check Program.family

end tests.NebulaProductionPaperTerminalCompleteProgramFor
