import NightstreamFPrime.Export.Stage1.PiCCSActionPayloadBlock

/-!
Owns the assertion facts for the four semantic C compiler traces.
The expected-sample expressions are the existing owned sample readers.
Statement and output absorption have no sample assertions. These facts do
not assert that an arbitrary environment satisfies permutation recipes.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSCompilerAssertions

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open PiCCSInvocations

private theorem assertions_of_samples (env : Env) (start : Nat)
    (state : Layer.EState) (actions : List Formal.Action)
    (same : Formal.expectedSamples actions = (Formal.compile start state actions).samples) :
    ConstraintsHold env (Formal.compile start state actions).assertions := by
  apply (Formal.compile_assertions_hold_iff env start state actions).mpr
  exact congrArg (List.map (KExpr.eval env)) same.symm

/-- The statement program contains only absorption actions, so its semantic
compiler has no sample assertion to check. -/
theorem statement_assertions (env : Env) :
    ConstraintsHold env (Formal.compile statementWitnessStart Hash.zeroE
      PiCCSActionPayloadBlock.statementActions).assertions := by
  apply assertions_of_samples env
  apply Invocations.expectedSamples_eq_samples_of_assertionCount_zero
  simpa only [PiCCSActionPayloadBlock.statementActions, PiCCSInvocations.statementActions] using
    StatementAbsorption.assertionCount_eq
      (statementInterface Data.logicalWidth Data.publicFits) statementWitnessStart

/-- Challenge expectations are exactly the owned challenge program's sample
expressions. Thus its assertion expressions hold in every source environment. -/
theorem challenge_assertions (env : Env) :
    ConstraintsHold env (Formal.compile challengeWitnessStart
      ((challengeInterface Data.logicalWidth Data.publicFits).initialState challengeWitnessStart)
      PiCCSActionPayloadBlock.challengeActions).assertions := by
  apply assertions_of_samples env
  simpa only [PiCCSActionPayloadBlock.challengeActions, ChallengeDerivation.program] using
    ChallengeDerivation.expectedSamples_eq_samples
      (challengeInterface Data.logicalWidth Data.publicFits) challengeWitnessStart

/-- Round expectations use the same sample expressions as the owned round
program, independently of the separate permutation-recipe predicates. -/
theorem round_assertions (env : Env) :
    ConstraintsHold env (Formal.compile roundWitnessStart
      ((roundInterface Data.logicalWidth Data.publicFits).initialState roundWitnessStart)
      PiCCSActionPayloadBlock.roundActions).assertions := by
  apply assertions_of_samples env
  simpa only [PiCCSActionPayloadBlock.roundActions, RoundTranscript.program] using
    RoundTranscript.expectedSamples_eq_samples
      (roundInterface Data.logicalWidth Data.publicFits) roundWitnessStart

/-- Output absorption has no assertions in its owned compiler program. -/
theorem output_assertions (env : Env) :
    ConstraintsHold env (Formal.compile outputWitnessStart
      ((outputInterface Data.logicalWidth Data.publicFits).initialState outputWitnessStart)
      PiCCSActionPayloadBlock.outputActions).assertions := by
  have empty : (Formal.compile outputWitnessStart
      ((outputInterface Data.logicalWidth Data.publicFits).initialState outputWitnessStart)
      PiCCSActionPayloadBlock.outputActions).assertions = [] := by
    simpa only [Formal.Owned.allAssertions, Formal.Owned.program, OutputBinding.duplexInterface,
      PiCCSActionPayloadBlock.outputActions, PiCCSInvocations.outputActions] using
      OutputBinding.noAssertions (outputInterface Data.logicalWidth Data.publicFits) outputWitnessStart
  rw [empty]
  intro expression member
  cases member

end NightstreamFPrime.Export.Stage1.PiCCSCompilerAssertions
