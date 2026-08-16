import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFPrimeProgram

/-! Regression surface for the verifier-owned streaming F-prime program. -/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace tests.NebulaProductionStreamingFPrimeProgram

open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram

#check Phase.code_injective
#check Step.uses_exact_work_item
#check no_step_from_complete
#check Runs.cursor_exact
#check complete_run_steps_exact
#check production_complete_run_steps_exact

example : (program productionConfig).length = 400 := by decide

example :
    (program productionConfig)[0]? =
      some { phase := Phase.prelude, index := 0 } := by
  decide

example :
    (program productionConfig)[1]? =
      some { phase := .priorStateReplay, index := 0 } := by
  decide

example :
    (program productionConfig)[82]? =
      some { phase := .priorStateReplay, index := 81 } := by
  decide

example :
    (program productionConfig)[83]? =
      some { phase := .claimReplay, index := 0 } := by
  decide

example :
    (program productionConfig)[168]? =
      some { phase := .claimReplay, index := 85 } := by
  decide

example :
    (program productionConfig)[169]? =
      some { phase := Phase.piCcsStart, index := 0 } := by
  decide

example :
    (program productionConfig)[195]? =
      some { phase := .piCcsRound, index := 25 } := by
  decide

example :
    (program productionConfig)[199]? =
      some { phase := .piRlcFamily, index := 0 } := by
  decide

example :
    (program productionConfig)[308]? =
      some { phase := .piRlcFamily, index := 109 } := by
  decide

example :
    (program productionConfig)[314]? =
      some { phase := .successorPrefixReplay, index := 0 } := by
  decide

example :
    (program productionConfig)[395]? =
      some { phase := .successorPrefixReplay, index := 81 } := by
  decide

example :
    (program productionConfig)[399]? =
      some { phase := Phase.semanticLinks, index := 0 } := by
  decide

example : (program productionConfig)[400]? = none := by decide

example :
    circuitKind productionConfig { phase := .piRlcFamily, index := 0 } =
      .piRlcFamilyEven := by
  decide

example :
    circuitKind productionConfig { phase := .piRlcFamily, index := 1 } =
      .piRlcFamilyOdd := by
  decide

example :
    circuitKind productionConfig { phase := .piRlcFamily, index := 109 } =
      .piRlcFamilyOdd := by
  decide

end tests.NebulaProductionStreamingFPrimeProgram
