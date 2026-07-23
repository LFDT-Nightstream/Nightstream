import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive

/-!
Transcript timing and domain authority for the delayed projection challenges.

Owns: the exact statement fields fixed before challenge derivation, the
kernel-distinct producer/residual domains, and the definitional sampling order
`statement -> core -> producerBeta -> batchWeight`.

Does not own: Poseidon2 internals, random-oracle security, raw-assignment
opening validity, or Rust transcript conformance.

Emits constraints: no.

Authority boundary: every equality in `Holds` is recomputed from the installed
`FixedActive.Context`; no caller-supplied challenge or statement binding is
accepted.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.delayed.transcript.statement` | bind sources, running parent, pending value, and polynomial to the installed statement | checked | `Holds`, `holds` |
| `fprime.delayed.transcript.domains` | producer and residual challenges use distinct domain labels | checked | `Holds.domainsDistinct`, `holds` |
| `fprime.delayed.transcript.order` | derive core, then `producerBeta`, then `batchWeight` | computed | `Holds.producerAfterStatement`, `Holds.batchAfterProducer`, `holds` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.ChallengeAuthority

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Complete typed transcript contract for one fixed-active context. Every
field is computed from the context; callers supply no challenge-equality
premise. -/
structure Holds
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows) : Prop where
  sourcesBound : context.piCcsStatement.input.sources = context.input
  runningParentBound :
    context.piCcsStatement.input.runningParent = context.runningParent
  pendingBound : context.piCcsStatement.input.pending = context.pending
  polynomialBound :
    context.piCcsStatement.input.polynomial = context.piCcsInput
  domainsDistinct :
    DelayedChallengeDomain.producerBeta ≠ DelayedChallengeDomain.batchWeight
  producerAfterStatement :
    context.producerBeta =
      (context.piCcsSchedule.squeezeDelayedChallenge
        (context.piCcsSchedule.enterDelayedDomain .producerBeta
          (context.piCcsSchedule.deriveCore
            (context.piCcsSchedule.bindStatement context.priorState
              context.piCcsStatement)).state)).1
  batchAfterProducer :
    context.batchWeight =
      (context.piCcsSchedule.squeezeDelayedChallenge
        (context.piCcsSchedule.enterDelayedDomain .batchWeight
          (context.piCcsSchedule.squeezeDelayedChallenge
            (context.piCcsSchedule.enterDelayedDomain .producerBeta
              (context.piCcsSchedule.deriveCore
                (context.piCcsSchedule.bindStatement context.priorState
                  context.piCcsStatement)).state)).2)).1

/-- Every fixed-active context satisfies the transcript contract by
construction. -/
theorem holds
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows) : Holds context := by
  exact {
    sourcesBound := context.piCcsStatement_sources
    runningParentBound := context.piCcsStatement_runningParent
    pendingBound := context.piCcsStatement_pending
    polynomialBound := context.piCcsStatement_polynomial
    domainsDistinct := DelayedChallengeDomain.producerBeta_ne_batchWeight
    producerAfterStatement := by
      exact derivePreSumcheck_producerBeta context.piCcsSchedule
        context.priorState context.piCcsStatement
    batchAfterProducer := by
      exact derivePreSumcheck_batchWeight context.piCcsSchedule
        context.priorState context.piCcsStatement
  }

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.ChallengeAuthority
