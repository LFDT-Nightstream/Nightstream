import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.ChallengeWiring

/-!
Public artifact regressions for the active PiRLC challenge wiring.

| Fact | Result |
|---|---|
| selected-output formula | `5_045_274 + 7_984*rho + 45*coefficient` |
| producer/consumer edge | all 810 projection rho inputs are those selected outputs |
-/

namespace NightstreamTests.FPrimeRecursivePiRlcChallengeWiring

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.ChallengeWiring

example
    (rho : Fin rhoCount) (coefficient : Fin coefficientCount) :
    samplerOutputColumn rho coefficient =
      5045274 + 7984 * rho.val + 45 * coefficient.val :=
  sampler_output_formula rho coefficient

example : projectionConsumerColumns = samplerOutputColumns :=
  projection_consumers_are_sampler_outputs

end NightstreamTests.FPrimeRecursivePiRlcChallengeWiring
