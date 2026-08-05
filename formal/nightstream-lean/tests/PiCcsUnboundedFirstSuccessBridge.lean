import Nightstream.Protocol.FPrime.Frozen.PiCcsFirstSuccessBridge

/-!
Focused operational regression for the unbounded PiCCS success-gated bridge.

The two-seed fixture exercises the geometric retry interface without
constructing any protocol-specific witness. The remaining checks pin the
asymptotic PiCCS and exact frozen-facade theorem signatures.
-/

set_option autoImplicit false

namespace tests.PiCcsUnboundedFirstSuccessBridge

open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace

#check Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime.extractorExpectedPolynomialTime
#check Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime.oneRunExpectedWork_le_bound
#check Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime.gatedRetryExpectedWork_le_oneRun
#check Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime.expectedWork_le_twoRunWorkBound
#check Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.AsymptoticPaperStrong.paperStrong
#check Nightstream.Protocol.FPrime.Frozen.PiCcsFirstSuccessBridge.piCcsStrong_of_successGatedRetry

def twoSeeds : Support Nat where
  values := [0, 1]
  nodup := by decide
  nonempty := by decide

def baseExperiment : Experiment Nat :=
  twoSeeds.uniform

def succeeds (value : Nat) : Bool :=
  value == 0

def successfulSeedsNonempty :
    baseExperiment.support.values.filter
      (fun seed => succeeds (baseExperiment.outcome seed)) ≠ [] := by
  decide

example :
    FailureTailVanishes baseExperiment succeeds :=
  failureTail_vanishes
    baseExperiment succeeds successfulSeedsNonempty

example :
    AlmostSurelyTerminates baseExperiment succeeds :=
  firstSuccess_terminates_almostSurely
    baseExperiment succeeds successfulSeedsNonempty

example (event : Nat × Nat -> Bool) :
    jointProbability baseExperiment succeeds event =
      (conditionedFreshProduct
        baseExperiment succeeds successfulSeedsNonempty).probabilityBool
        event :=
  jointProbability_eq_conditionedFreshProduct
    baseExperiment succeeds successfulSeedsNonempty event

example :
    expectedRetryExecutions baseExperiment succeeds =
      1 + failureProbability baseExperiment succeeds *
        expectedRetryExecutions baseExperiment succeeds :=
  expectedRetryExecutions_firstStep
    baseExperiment succeeds successfulSeedsNonempty

end tests.PiCcsUnboundedFirstSuccessBridge
