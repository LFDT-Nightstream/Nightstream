import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context

/-!
Focused compile-time regression for the complete model-level fixed-active
honest-baseline context.

| Stage path | Regression |
|---|---|
| `fprime.active.honest_baseline.context.profile` | the 270-coordinate source shape is covered by the exact typed domain and supported FE profile |
| `fprime.active.honest_baseline.context.sampler.chunk` | candidate value two, not literal candidate zero, decodes to the centered-zero coefficient |
| `fprime.active.honest_baseline.context.sampler.bound` | generic first-accepted executions form an exact batch for the all-zero RingF challenge vector |
| `fprime.active.honest_baseline.context.radix_zero` | the canonical concrete PiDEC split of zero remains zero for every child |
| `fprime.active.honest_baseline.context.sources` | the physical context binds to the independent source family |
| `fprime.active.honest_baseline.context.parent_opening` | the context-owned zero parent has the context-owned zero opening |
| `fprime.active.honest_baseline.context.running` | the installed parent and children satisfy checked running authority |
| `fprime.active.honest_baseline.context.premises` | both semantic-only and explicit successful-sampler honest NIFS premises are inhabited |
| `fprime.active.honest_baseline.context.transition` | the model fixture has one accepted certificate with an independent fixed-active result transition |
-/

namespace Nightstream.Tests.FPrimeConcretePhi81ActiveHonestBaselineContext

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline

#check Context.covers
#check Context.profile
#check Context.centeredZeroChunk_symbol
#check Context.centeredZeroScalar_embed_eq_zero
#check Context.piRlcMachine_digestChunk_accepted
#check Context.exists_centeredZeroExecution
#check Context.centeredZeroBatch_challenge
#check Context.samplerBound
#check Context.splitZero
#check Context.sourceBound
#check Context.semanticInput
#check Context.parentHolds
#check Context.runningAccepted
#check Context.semanticPremises
#check Context.honestPremises
#check Context.exists_resultTransition

example :
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticInput
      Context.context Sources.data :=
  Context.semanticInput

example :
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority.Accepted
      Context.context :=
  Context.runningAccepted

example :
    Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs.SemanticPremises
      Context.setup Context.input Context.selected :=
  Context.semanticPremises

noncomputable example :
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Bound
      Context.piRlcMachine () Context.zeroChallenges :=
  Context.samplerBound ()

example :
    exists certificate :
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Certificate
          Context.context,
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Accepted
          Context.context certificate /\
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.ResultTransition
          Context.context
          (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.resultOf
            Context.context certificate) :=
  Context.exists_resultTransition

end Nightstream.Tests.FPrimeConcretePhi81ActiveHonestBaselineContext
