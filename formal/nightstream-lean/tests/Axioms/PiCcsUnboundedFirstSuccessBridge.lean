import Nightstream.Protocol.FPrime.Frozen.PiCcsFirstSuccessBridge
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency guards for the retry trace, success-gated
runtime, PiCCS strong reduction, and exact frozen bridge.
-/

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.partialTerminationMass_add_failureTail' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.partialTerminationMass_add_failureTail

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.failureTail_le_inverseNatural' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.failureTail_le_inverseNatural

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.failureTail_vanishes' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.failureTail_vanishes

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.trace_totalMass_eq_one' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.trace_totalMass_eq_one

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.firstSuccess_terminates_almostSurely' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.firstSuccess_terminates_almostSurely

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.jointProbability_firstStep' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.jointProbability_firstStep

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.jointProbability_unique' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.jointProbability_unique

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.jointProbability_eq_firstConditionedFreshSecond' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.jointProbability_eq_firstConditionedFreshSecond

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.jointProbability_eq_conditionedFreshProduct' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.jointProbability_eq_conditionedFreshProduct

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.freshSecond_marginal' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.freshSecond_marginal

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.expectedRetryExecutions_firstStep' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace.expectedRetryExecutions_firstStep

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime.oneRunExpectedWork_le_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime.oneRunExpectedWork_le_bound

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime.gatedRetryExpectedWork_le_oneRun' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime.gatedRetryExpectedWork_le_oneRun

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime.expectedWork_le_twoRunWorkBound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime.expectedWork_le_twoRunWorkBound

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime.extractorExpectedPolynomialTime' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime.extractorExpectedPolynomialTime

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.AsymptoticPaperStrong.paperStrong' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.AsymptoticPaperStrong.paperStrong

/-- info: 'Nightstream.Protocol.FPrime.Frozen.PiCcsFirstSuccessBridge.piCcsStrong_of_successGatedRetry' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.PiCcsFirstSuccessBridge.piCcsStrong_of_successGatedRetry
