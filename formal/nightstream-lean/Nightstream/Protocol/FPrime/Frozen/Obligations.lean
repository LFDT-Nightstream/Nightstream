import Nightstream.SuperNeo.InteractiveReduction.Paper
import Nightstream.HyperNova.NonInteractiveMultiFold

/-!
Frozen target propositions for the paper-authoritative F-prime proof program.

Owns: only the signatures of obligations 1--5 and their exact quantitative or
named-event boundaries.

Does not own: a proof of any target, a concrete SuperNeo game, a concrete NIFS
verifier, Construction 2, an implementation, Rust, R1CS, or costs.

Emits constraints: no.

The final headline theorems must instantiate these targets from the permitted
primitive contracts.  Receiving one of these targets, an extractor callback,
an arithmetization, or semantic acceptance as a premise does not discharge an
obligation.
-/

namespace Nightstream.Protocol.FPrime.Frozen.Obligations

open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.HyperNova.NonInteractiveMultiFold

universe uWeight uPiCcsAdversary uPiCcsExtractor
  uPiRlcAdversary uPiRlcPairedAdversary uPiRlcExtractor
  uPiDecAdversary uPiDecExtractor uComposedAdversary uComposedExtractor
  uIntermediate uProjection

/-- Independently defined quantitative games and shared projection for the
three SuperNeo reductions.  This is game data, not evidence that a target
holds. -/
structure SuperNeoGames where
  Weight : Type uWeight
  scale : ProbabilityScale Weight

  PiCcsAdversary : Type uPiCcsAdversary
  PiCcsExtractor : Type uPiCcsExtractor
  piCcs : StrongGame Weight PiCcsAdversary PiCcsExtractor

  PiRlcAdversary : Type uPiRlcAdversary
  PiRlcPairedAdversary : Type uPiRlcPairedAdversary
  PiRlcExtractor : Type uPiRlcExtractor
  piRlc : WeakGame Weight PiRlcAdversary PiRlcPairedAdversary PiRlcExtractor

  PiDecAdversary : Type uPiDecAdversary
  PiDecExtractor : Type uPiDecExtractor
  piDec : KnowledgeGame Weight PiDecAdversary PiDecExtractor

  ComposedAdversary : Type uComposedAdversary
  ComposedExtractor : Type uComposedExtractor
  composed : KnowledgeGame Weight ComposedAdversary ComposedExtractor

  IntermediateInstance : Type uIntermediate
  Projection : Type uProjection
  piCcsProjection : IntermediateInstance -> Projection
  piRlcProjection : IntermediateInstance -> Projection

  errorBudget : InteractiveErrorBudget Weight

/-- Obligation 1: Pi_CCS is rejection-adjusted strong with intrinsic SumCheck
and Schwartz--Zippel losses. Pi_RLC supplies the raw repeated-output witness
disagreement bound; Pi_CCS charges its success-conditioned adjustment. -/
def PiCcsStrong (games : SuperNeoGames) : Prop :=
  RejectionAdjustedStrong games.scale
    games.errorBudget.adjustUniqueness games.piCcs
    games.errorBudget.piCcsSuccessFloor
    (games.scale.add games.errorBudget.piCcsSumCheck
      games.errorBudget.piCcsSchwartzZippel)
    games.errorBudget.relaxedBindingRaw

/-- Obligation 2: Pi_RLC is weak with the fork-sampling loss and relaxed
binding as its exact witness-uniqueness boundary. -/
def PiRlcWeak (games : SuperNeoGames) : Prop :=
  Weak games.scale games.piRlc
    games.errorBudget.piRlcForkSampling
    games.errorBudget.relaxedBindingRaw

/-- Theorem 6 requires literal equality of the two commitment projections. -/
def SharedCommitmentProjection (games : SuperNeoGames) : Prop :=
  games.piCcsProjection = games.piRlcProjection

/-- Obligation 3: Pi_DEC is a zero-loss reduction of knowledge. -/
def PiDecReductionOfKnowledge (games : SuperNeoGames) : Prop :=
  ReductionOfKnowledge games.scale games.piDec games.scale.zero

/-- Obligation 4: the complete composition is a reduction of knowledge with
SumCheck, Schwartz--Zippel, fork-sampling, and the one adjusted binding loss.
The raw binding error is not charged a second time. -/
def SuperNeoCompositionReductionOfKnowledge (games : SuperNeoGames) : Prop :=
  ReductionOfKnowledge games.scale games.composed
    (games.errorBudget.total games.scale)

/-- Complete frozen SuperNeo target; no generic bad event or final `negl` term
is available. -/
def SuperNeoPaperObligations (games : SuperNeoGames) : Prop :=
  PiCcsStrong games /\
  PiRlcWeak games /\
  SharedCommitmentProjection games /\
  PiDecReductionOfKnowledge games /\
  SuperNeoCompositionReductionOfKnowledge games

universe uKey uRunning uFresh uProof

/-- Obligation-5 soundness target for a concrete one-message deterministic
NIFS verifier and independently defined paper transition. -/
def NifsSoundModulo
    {Key : Type uKey}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (verifier : Verifier Key Running Fresh Proof)
    (transition : Key -> Running -> Fresh -> Running -> Prop)
    (badEvent : Key -> Running -> Fresh -> Proof -> Running -> Prop) : Prop :=
  forall key running fresh proof output,
    Accepts verifier key running fresh proof output ->
      transition key running fresh output \/
      badEvent key running fresh proof output

/-- Obligation-5 honest completeness target. -/
def NifsComplete
    {Key : Type uKey}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (verifier : Verifier Key Running Fresh Proof)
    (transition : Key -> Running -> Fresh -> Running -> Prop) : Prop :=
  forall key running fresh output,
    transition key running fresh output ->
      exists proof, Accepts verifier key running fresh proof output

/-- Full obligation-5 target. The bad-event predicate must be instantiated by
a closed finite type of protocol/security events, never `refinementFailure` or
`outputUnbound`. -/
def NifsSoundAndCompleteModulo
    {Key : Type uKey}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (verifier : Verifier Key Running Fresh Proof)
    (transition : Key -> Running -> Fresh -> Running -> Prop)
    (badEvent : Key -> Running -> Fresh -> Proof -> Running -> Prop) : Prop :=
  NifsSoundModulo verifier transition badEvent /\
  NifsComplete verifier transition

end Nightstream.Protocol.FPrime.Frozen.Obligations
