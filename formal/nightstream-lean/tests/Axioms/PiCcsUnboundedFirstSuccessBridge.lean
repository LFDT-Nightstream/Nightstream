import Nightstream.Protocol.FPrime.Frozen.PiCcsFirstSuccessBridge
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency guards for the operational unbounded
first-success trace, runtime, PiCCS strong reduction, and exact frozen bridge.
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

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime.traceWork_le_executionCount_mul_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime.traceWork_le_executionCount_mul_bound

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime.oneRunExpectedWork_le_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime.oneRunExpectedWork_le_bound

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime.expectedRetryWork_firstStep' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime.expectedRetryWork_firstStep

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime.expectedWork_eq_oneRun_mul_totalExecutions' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime.expectedWork_eq_oneRun_mul_totalExecutions

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime.expectedWork_le_floorWorkBound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime.expectedWork_le_floorWorkBound

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime.extractorExpectedPolynomialTime' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime.extractorExpectedPolynomialTime

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.AsymptoticPaperStrong.paperStrong' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.AsymptoticPaperStrong.paperStrong

/-- info: 'Nightstream.Protocol.FPrime.Frozen.PiCcsFirstSuccessBridge.piCcsStrong_of_unboundedFirstSuccess' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.PiCcsFirstSuccessBridge.piCcsStrong_of_unboundedFirstSuccess
