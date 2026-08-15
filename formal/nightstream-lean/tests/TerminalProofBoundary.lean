import Nightstream.Assurance.TerminalProofBoundary

namespace tests.TerminalProofBoundary

open Nightstream.Assurance.TerminalProofBoundary

example : guardNames = [
    "terminal.proof.expected_public_image",
    "terminal.proof.spartan_verification",
    "terminal.proof.public_statement"
  ] := guardNames_exact

example : verify expectedPublicImageWitness = false := by decide
example : verify backendVerificationWitness = false := by decide
example : verify publicStatementWitness = false := by decide

example :
    Nightstream.SuperNeo.CheckPlan.InclusionMinimalSound
      semantics Target guards := inclusionMinimalSound

end tests.TerminalProofBoundary
