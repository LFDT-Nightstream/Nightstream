import Nightstream.Assurance.TerminalNativeGuardPlan

namespace tests.TerminalNativeGuardPlan

open Nightstream.Assurance.TerminalNativeGuardPlan

theorem exactRustLedger :
    Nightstream.Implementation.R1CS.Artifacts.TerminalVerifierNativeGuards.names =
      guardNames :=
  artifact_guard_names_exact

theorem allEighteenAreInclusionNecessary :
    Nightstream.SuperNeo.CheckPlan.InclusionMinimalSound
      semantics Target guards :=
  inclusionMinimalSound

theorem rustShapedExecutionIsExact (candidate : Candidate) :
    verify candidate = .ok () ↔ Target candidate :=
  verify_eq_ok_iff_target candidate

end tests.TerminalNativeGuardPlan
