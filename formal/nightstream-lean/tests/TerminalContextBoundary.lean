import Nightstream.Assurance.TerminalContextBoundary

namespace tests.TerminalContextBoundary

open Nightstream.Assurance.TerminalContextBoundary

example : guardNames = [
    "terminal.context.induction",
    "terminal.context.plain_chain",
    "terminal.context.public_width",
    "terminal.context.relation_structure"
  ] := guardNames_exact

example (guard : Guard) : verify (removalWitness guard) = false := by
  cases guard <;> decide

example :
    Nightstream.SuperNeo.CheckPlan.InclusionMinimalSound
      semantics Target guards := inclusionMinimalSound

end tests.TerminalContextBoundary
