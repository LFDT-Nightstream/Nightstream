import Nightstream.Assurance.TerminalStatementBoundary

namespace tests.TerminalStatementBoundary

open Nightstream.Assurance.TerminalStatementBoundary

example : guardNames.length = 11 := by decide

example (guard : Guard) : verify (removalWitness guard) = false := by
  cases guard <;> decide

example :
    Nightstream.SuperNeo.CheckPlan.InclusionMinimalSound
      semantics Target guards := inclusionMinimalSound

end tests.TerminalStatementBoundary
